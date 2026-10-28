# MQAdapter source-verification report (Phase 2)

**Provenance.** Paper: arXiv 2607.12418 (v1, 14 Jul 2026). **No official implementation was located** (arXiv page, CatalyzeX,
Pith, web search; GitHub search itself was not possible) and the user has no repository link. Everything below is therefore a
*paper-based reimplementation*. The two items the task wanted confirmed "from the original implementation" (layer placement /
forward behaviour; the actual TorchQuantum circuit) are **unverifiable from code** and are resolved by the decisions D1-D14.
Paper sections read in full: abstract, I-IV, V-A, Table I, start of V-B/Table II. Read only via search excerpts: V-C1-3, V-F.
Not read: V-D, V-E, VI, any supplement, Tables III-VI.

## A. Verified from the paper
| Item | Paper | Source |
|---|---|---|
| Backbone, simulator | CLIP ViT-B/16 (checkpoint unspecified), 8-qubit shared VQC in TorchQuantum | V-A2 |
| Coarse stage | frozen CLIP image/text similarities over all C classes -> Top-K as anchors P; K=5 | IV-A, V-A2 |
| Projections (Eq. 11) | c̄_ℓ = W^v_ℓ c_ℓ (per layer), P̄ = W^t P (no layer index), both to N=8 dims | IV-C |
| Circuit (Eqs. 5-8) | Ry(x_n) encoding; Ry(φ¹); ring CRZ (E = {(n,n+1)} ∪ {(N,1)}); Ry(φ²); Pauli-Z on each qubit | IV-B |
| Interaction (Eqs. 12-15) | same VQC on c̄ and each p̄_k; α = softmax(P̃ᵀc̃/√N) over K anchors; h = P̃α; h through VQC; ĉ_ℓ = c_ℓ + γ W_ℓ f(h) | IV-C |
| Loss (Eq. 16) | cos(z^v, z^t_y)/τ cross-entropy over **all C** classes | IV-D |
| Insertion | from the 6th transformer layer to the final layer; only the class token is modified | V-A2, Eq. 15 |
| γ | 0.001 (base-to-new; few-shot MaPLe); 0.0001 PromptSRC; 0.002 MMRL++ | V-A2 |
| Host frameworks | added to MaPLe, PromptSRC, MMRL++; their training protocols are followed | V-A2 |
| Shots | 1/2/4/8/16 per class (few-shot); 16-shot elsewhere | V-A1 |
| Efficiency | +0.078M parameters over MMRL++; VQC 0.024K; 300 vs 386 FPS (batch 100, RTX 3090) | V-F |
| Ablations | cross-attention replacement +0.23 H vs MQAdapter +0.82; K=2: 81.89 vs K=5: 82.36; shared > separate VQC; shallow insertion hurts, too-deep adapts too few blocks | V-C |

## B. Verified by computation on the paper's equations (not on any official code)
`tests/test_reference_paper_vqc.py` (numpy oracle, 6 tests, executed): parameters = 3N = 24 for N=8 (matches 0.024K);
Ry(φ¹) after Ry(x) equals Ry(x+φ¹) so φ¹ is only an input-angle bias; with φ²=0 the diagonal CRZ gates cannot change the Z outputs
(output = cos(x+φ¹)); the two possible CRZ control/target readings give different outputs, so TorchQuantum parity can resolve the ambiguity.
**Parameter count 0.078M** is reproduced only under assumptions: 6 adapted blocks, bias-free 768→8 and 8→768 per block, one shared
512→8 text projection, one shared VQC = 77,848. 7 blocks ≈ 0.090M; with biases ≈ 0.083M. This is *consistent with*, not *confirmation of*, the paper.
These numbers are **never hard-coded**; the implementation counts parameters at runtime and reports any difference.

## C. Discrepancies / ambiguities (unresolved in the paper)
1. The paper evaluates MQAdapter only as a plug-in to prompt-learning hosts; stand-alone frozen-CLIP use is not evaluated.
2. K=C makes retrieval input-independent and the attention permutation-invariant (coarse stage = no-op).
3. "6th to final layer": 1-indexed 6-12 (7 blocks) vs 0-indexed 6-11 (6 blocks).
4. Injection point relative to attention/LN/residual is not stated (Eq. 10 implies after block ℓ's output).
5. VQC input scaling, parameter/W_ℓ initialisation, τ, bias terms, text-feature normalisation, optimizer/LR/epochs, seeds: unstated.
6. Eq. 16 says image *and text* encoders are optimised; the adapter touches only visual tokens (text changes only via host prompts).
7. CRZ control/target direction on ring edges is unstated.
8. Whether one VQC is shared across layers is unstated (0.024K total implies yes).

## D. Decision register (all changeable; each is recorded in run metadata)
| ID | Decision | Rationale / consequence | Verified by |
|---|---|---|---|
| D1 | OpenAI CLIP ViT-B/16 (open_clip `openai` weights or `clip`), fp32; weights fingerprint recorded | Matches the CoOp/MaPLe lineage; paper does not name a checkpoint | phase3_clip_verify |
| D2 | Adapt blocks 6-11 (0-indexed, 6 blocks); 7-block (5-11) as ablation | Reproduces 0.078M; if wrong, counts differ and are reported | param-count test |
| D3 | c_ℓ = class token output of block ℓ (768-d, before ln_post); injected after block ℓ, before block ℓ+1, CLS only | Eq. 10/15 literal reading | hook test in phase3_clip_verify |
| D4 | Anchors = frozen final CLIP text embeddings of the 4 fixed prompts (512-d, L2-normalised); same prompts for A/B/C | Text side frozen; prompts not tuned on test | tests |
| D5 | Single shared VQC across all adapted blocks and both modalities (24 params) | Matches 0.024K; per-block VQC as ablation | param-count test |
| D6 | Faithful circuit Eqs. 5-8; CRZ direction fixed by TorchQuantum parity (phase3a reports which reading matches) | Resolves C7 empirically | phase3_env_check |
| D7 | Angles = raw linear projections (no tanh/rescale); log angle statistics; default Linear init, bias-free | Faithful; Ry periodic so no overflow | unit tests + logging |
| D8 | Logits = CLIP logit_scale (τ=1/scale, frozen) × cos(z^v, z^t_i), z^v = adapted image embedding through remaining blocks, ln_post, proj; classification over all 4 classes | Eq. 16 with CLIP's own scale | tests |
| D9 | K=4 primary (retrieval = no-op, reported honestly); K=2,3 exploratory with recall and stratified accuracy | Dataset has 4 classes; no duplicated anchors | retrieval tests |
| D10 | Every test image is evaluated and counted; never filtered by retrieval success; report accuracy for retrieved vs missed cases | User requirement; architecture classifies over all C | len(preds)==len(test manifest) test |
| D11 | Keep CLIP resolution/bicubic/mean/std; replace center-crop by pad-to-square (MRI geometry); augmentation train-only | Center-crop could remove anatomy | preprocessing test |
| D12 | Optimiser/LR not specified by the paper: AdamW, LR/γ tuned on development data only, identical search budget for B and C | Host-framework optimisers not verified | logged configs |
| D13 | Shots 1/2/4/8/16 (16 primary); 5 seeds; support samples disjoint from development pool; development pool from Training only; test untouched | Matches paper shots; dev pool is extra labelled data used only for selection, disclosed | manifest tests |
| D14 | Model B keeps projections, attention, residual, γ; replaces f_φ by classical 8→8 maps: identity, cos(x+b), MLP 8→h→8; extra control: VQC with frozen random parameters | VQC is ~0.03% of trainable parameters, so only nonlinearity/entanglement can differ; controls isolate it | param-count report |

## E. Test plan mapping (task section 9)
Label mapping, manifest disjointness, duplicate checks -> data tests. CLIP preprocessing / text dim / insertion layers -> phase3_clip_verify.
VQC qubits/params/measurement shape/gradients -> phase3_env_check + oracle parity. Attention sums to 1, residual shape, frozen backbone,
nonzero gradients, optimizer step, 4 logits, one-batch and tiny-overfit tests -> Phase 4/8 tests (not yet written; gated on Phase 3 results).

## F. Phase 3 gate (run, then send the three JSON reports)
```
pip install -r requirements-phase3.txt
python scripts/phase3_env_check.py  --out phase3_env_report.json     # prints PHASE3a PASSED
python scripts/phase3_clip_verify.py --backend open_clip --out phase3_clip_report.json   # prints PHASE3b PASSED
python scripts/phase3_budget.py --out phase3_budget_report.json                           # prints PHASE3c PASSED (GPU memory/throughput)
pytest -q tests
```
Expected: 24 parameters; output shape (5, 8); parity diff < 1e-5 with one CRZ reading named; non-zero gradients for ry1/crz/ry2;
12 blocks, width 768, 197 tokens, 512-d text embeddings, image shape (1,3,224,224), recon diff < 1e-4, injection changes only blocks ≥ 6.
Likely failures: TorchQuantum API/version mismatch (try `torchquantum-ng` on Python ≥ 3.12); `tqf.ry` expecting params of shape [B,1];
parity matching neither reading (report the diffs); open_clip downloading weights without network.

## G. Hardware profile: 12 GB NVIDIA GPU, 64 GB RAM (user-reported; the actual GPU model is not known)
Config: `configs/hardware_12gb_64gb.json` (declared targets). Measured limits come from `phase3_budget_report.json`.
| ID | Decision | Rationale |
|---|---|---|
| D15 | Quantum block always runs in fp32, outside autocast (`torch.autocast(enabled=False)` + `.float()`); verified in phase3_env_check | complex64 simulation under bf16/fp16 would corrupt angles and gradients |
| D16 | Default precision fp32 for A/B/C; fp16/bf16 autocast only for the frozen CLIP trunk if phase3_budget shows it is needed (bf16 only if the GPU supports it) | ViT-B/16 (150M params) with gradients through 6 blocks should fit 12 GB at fp32; the profiler decides, no guess |
| D17 | Batch sizes are taken from `max_batch_within_budget` (budget 10.5 GB = 12 GB minus 1.5 GB headroom), then rounded down to a power of two; gradient accumulation if the paper-faithful batch is larger | Section 10 of the task: profile before choosing |
| D18 | Optional exact optimisation (Phase 8, tested before use): cache the frozen prefix (outputs of blocks 0-5, 197x768) and frozen retrieval embeddings in RAM for support/dev/test images; ~0.3 MB/image in fp16, so all 7k images ~2 GB, well inside 64 GB | Identical mathematics (engineering, not scientific change); makes 5 seeds x 5 shot counts x ablations cheap. Not applicable to augmented images unless a fixed set of augmented copies is cached |
Not yet measured (estimates only): per-epoch time, total sweep time. The VQC state is 256 amplitudes per sample, so memory is
negligible; its cost is kernel-launch count (one gate op per layer), which phase3_env_check profiles per batch size on your GPU.
CUDA wheel: match `nvidia-smi` driver, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
(or cu124). TorchQuantum on CUDA is untested here; phase3_env_check will tell.
