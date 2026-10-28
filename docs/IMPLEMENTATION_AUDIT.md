# Implementation audit (paper vs repository)

**Paper:** Zhao et al., *MQAdapter: Multi-Modal Quantum Adapter for Coarse-to-Fine VLM Fine-tuning*, arXiv:2607.12418v1 (14 Jul 2026).  
**Local PDF:** `MQAdapter Multi-Modal Quantum Adapter for.pdf` (same content as the 12-page arXiv preprint).  
**Official code:** none located (see `docs/SOURCE_VERIFICATION_REPORT.md`). This repository is a paper-based reimplementation plus an MRI-dataset adaptation.

This audit was written from the paper and the files that existed **before** the remaining modules were added. Later sections of the repository document what was implemented afterwards.

## 1. Components already implemented (pre-completion snapshot)

| Component | Location | Status |
|---|---|---|
| Independent NumPy statevector oracle of Eqs. 5–8 | `mqa/quantum/reference_paper_vqc.py` | Complete; used as correctness oracle |
| Oracle property tests (param count, CRZ convention, φ¹ redundancy, Z-invariance when φ²=0) | `tests/test_reference_paper_vqc.py` | 6 tests, executed: **6 passed** |
| TorchQuantum circuit sketch (not a library module) | `scripts/phase3_env_check.py` (`build_paper_vqc`) | Written; **not executed** in this environment |
| CLIP structure / preprocessing / injection probe | `scripts/phase3_clip_verify.py` | Written; **not executed** (no `open_clip`, no weights) |
| GPU memory budget probe | `scripts/phase3_budget.py` | Written; **not executed** (this process sees CPU-only PyTorch) |
| Hardware declaration | `configs/hardware_12gb_64gb.json` | Present (declared, not measured) |
| Decision register D1–D18 | `docs/SOURCE_VERIFICATION_REPORT.md` | Present |

## 2. Missing components (pre-completion snapshot)

- Shared VQC as an importable `nn.Module` used in training
- MQAdapter block (Eqs. 11–15): projections \(W^v_\ell\), \(W^t\), \(W_\ell\), cross-modal attention, residual injection
- CLIP ViT-B/16 wrapper with insertion from the 6th transformer layer to the last
- Coarse Top-K semantic-anchor retrieval (Section IV-A)
- Temperature-scaled cosine cross-entropy (Eq. 16)
- Dataset discovery for `Training/<class>` and `Testing/<class>`
- Stratified train/validation split; frozen official test split
- Training loop (seed, AMP isolation for VQC, checkpoint, early stopping, resume, OOM)
- Evaluation (macro/per-class metrics, confusion matrix, latency, parameter counts)
- Baselines and ablations (frozen CLIP, classical adapter, K variants, VQC replacement)
- Configs, end-to-end scripts, README for a complete workflow
- Tests beyond the NumPy oracle

## 3. Incorrect or questionable implementations

None of the adapter/training code existed yet, so there was no silent MLP-for-quantum swap in the library path.

Questionable items in the **written-but-unrun** Phase 3 scripts (preserved, then hardened):

1. `scripts/phase3_env_check.py` reads `/proc/meminfo` (Linux-only). This process is Windows; that call would crash before the circuit check.
2. `tqf.ry(..., params=x[:, n])` may need shape `[B, 1]` depending on TorchQuantum version. Unverified here because TorchQuantum is not installed.
3. `scripts/phase3_budget.py` and CLIP verify require CUDA + `open_clip` weights.

The NumPy CRZ convention (`control_first` vs `target_first`) is an acknowledged paper ambiguity (Eq. 6 lists unordered pairs in \(E\)). The oracle exposes both; the trainable circuit defaults to `control_first` (TorchQuantum `wires=[control, target]` reading).

## 4. Confirmed bugs and reproduction

**Bug A — existing oracle tests are not a bug.** Reproduction:

```
python -m pytest -q tests/test_reference_paper_vqc.py
```

Result in this environment: 6 passed, exit code 0.

**Bug B — Linux-only memory probe.** Reproduction on this Windows machine: call `meminfo_gb()` from `scripts/phase3_env_check.py` → `FileNotFoundError: /proc/meminfo`. Fix: use a portable memory query and keep `/proc` as a fallback.

**Environment blockers (not code bugs):**

| Check | Result in this session |
|---|---|
| Python | 3.12.7 (Anaconda) |
| PyTorch | 2.6.0+cpu, `cuda.is_available() == False` |
| `nvidia-smi` | not on PATH |
| `open_clip` | not installed |
| `torchquantum` | not installed |
| Kaggle MRI dataset | not present in the workspace |
| CLIP ViT-B/16 weights | not present |

A second interpreter (`silent` conda env) has `torch 2.7.1+cu118` but still `cuda.is_available() == False` in this session. GPU training was therefore **not** run here. The user-reported 12 GB NVIDIA GPU is treated as the intended training machine, not as a verified property of this process.

## 5. Paper requirements that cannot be verified from available sources

The following are **not specified** in the paper. They are implementation choices (see `docs/ARCHITECTURE.md` and the decision register). They are **not** invented paper claims.

| Topic | What the paper states | Gap |
|---|---|---|
| CLIP checkpoint | “CLIP … ViT-B/16” | Not OpenAI vs LAION vs OpenCLIP tag |
| CRZ control vs target on ring edges | \(E=\{(n,n+1)\}\cup\{(N,1)\}\) | Ordered control/target unspecified |
| Insertion indexing | “from the 6-th transformer layer to the final layer” | 1-indexed 6–12 (7 blocks) vs 0-indexed 6–11 (6 blocks) |
| Injection site inside a block | Eq. 10/15: modify \(c_\ell\) after layer \(\ell\) | Pre-LN vs post-attn vs post-block unspecified |
| VQC input scaling | Angle encoding \(R_y(x_n)\) | No \(\pi\)-rescale / tanh |
| Init of \(W^v_\ell, W^t, W_\ell, \phi\) | — | Unspecified |
| Bias on linear maps | Parameter count 0.078M fits **bias-free** 768↔8 (6 blocks) + 512→8 + 24 VQC params | Biases not mentioned |
| One VQC vs per-layer VQC | “shared VQC”; VQC “0.024K” parameters | Sharing across **layers** implied by 24 params total |
| \(\tau\) | Eq. 16 | CLIP `logit_scale` is an implementation choice |
| Optimizer, LR, epochs, batch, seed | Follow host PEFT protocols (MaPLe / PromptSRC / MMRL++) | Those hosts are not this MRI setup |
| Text encoder training | “jointly optimizes the image and text encoders” (IV-D) while the adapter injects only into visual class tokens | Prompt hosts train text prompts; stand-alone frozen-CLIP+adapter does not |
| MRI preprocessing | None (natural-image CLIP) | Pad-to-square instead of center-crop is an MRI choice |
| Patient-level splits | Not applicable to the MRI Kaggle dump | Patient IDs are typically absent |

## 6. Original method vs MRI adaptation

### Original method (paper)

- Few-shot classification on **11 natural-image datasets** with **tens to thousands of classes** (ImageNet, StanfordCars, …).
- Coarse stage: frozen CLIP retrieves **Top-K=5** textual categories as semantic anchors \(P\).
- Fine stage: shared 8-qubit VQC + attention + residual on visual class tokens of later ViT blocks.
- Evaluated as a **plug-in** to MaPLe, PromptSRC, and MMRL++ (prompt-learning hosts), not as a stand-alone classifier.
- Loss (Eq. 16): temperature-scaled cosine softmax over **all C** classes.
- Additional protocols: base-to-new, cross-dataset, domain generalization.

### Required adaptations (this repository)

| Paper | MRI experiment | Interpretation |
|---|---|---|
| C ≫ K (e.g. ImageNet C=1000, K=5) | C = 4 (`glioma`, `meningioma`, `pituitary`, `notumor`) | Top-K retrieval is a **much weaker** filter |
| K=5 default | K=5 is **invalid** (only 4 classes) | Run K=2, K=3, and K=4 |
| K=4 | Retrieves **every** class; attention is permutation-invariant in the selected set | K=4 is a **control** (coarse stage is a no-op up to class order), not evidence of subset selection |
| 16-shot typical | Full Training split is available | Primary protocol: stratified train/val from `Training/`, official `Testing/` held out. Optional 1/2/4/8/16-shot configs are provided and must not be mixed with full-data numbers |
| Host PEFT (MaPLe / PromptSRC / MMRL++) | Not reimplemented | **Stand-alone** MQAdapter on frozen CLIP ViT-B/16. This is a documented deviation: we implement the adapter and Eq. 16, not the three host prompt methods |
| Natural RGB photos | Grayscale/RGB brain MRI | Repeat grayscale to 3 channels; pad-to-square; CLIP mean/std/224 |
| Base/new class split | Four diagnostic classes, all seen | No base-to-new protocol unless explicitly configured |

**Do not change the number of MRI classes** to imitate ImageNet-scale retrieval. K=4 results must be reported as full-set anchors, not as successful coarse filtering.

## 7. Equations implemented (verbatim targets)

- (5) Angle encoding: \(\lvert\psi(x)\rangle=\bigotimes_n R_y(x_n)\lvert 0\rangle_n\)
- (6)–(7) \(U(\phi)=\bigotimes_n R_y(\phi_n^{(2)})\cdot\prod_{(n,m)\in E}\mathrm{CRZ}(\phi_{n,m})\cdot\bigotimes_n R_y(\phi_n^{(1)})\), ring \(E\)
- (8) \(f_\phi(x)=[\langle Z_1\rangle,\ldots,\langle Z_N\rangle]\)
- (11) \(\bar c_\ell=W^v_\ell c_\ell\), \(\bar P=W^t P\)
- (12)–(14) shared VQC; \(\alpha_\ell=\mathrm{softmax}(\tilde P^\top\tilde c_\ell/\sqrt{N})\); \(h_\ell=\tilde P\alpha_\ell\)
- (15) \(\hat c_\ell=c_\ell+\gamma W_\ell f_\phi(h_\ell)\)
- (16) \(\mathcal L_{\mathrm{cls}}=-\log\frac{\exp(\cos(z^v,z^t_y)/\tau)}{\sum_i\exp(\cos(z^v,z^t_i)/\tau)}\)

N = 8 qubits; trainable VQC parameters = 3N = 24.

## 8. What this environment did and did not verify (at audit time)

| Claim | Evidence |
|---|---|
| NumPy oracle matches Eqs. 5–8 properties | `pytest` 6 passed |
| TorchQuantum parity | **Not run** (package missing) |
| CLIP weights / 224 preprocess / 12 blocks | **Not run** (package/weights missing) |
| GPU batch budget | **Not run** (CPU-only torch in this process) |
| Dataset counts | **Unknown until inspect**; dataset not in workspace |
| Training / test-set accuracy | **Not run**; must not be fabricated |
