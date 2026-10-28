# How much of the MQAdapter paper this repository reproduces

There is no official MQAdapter code. Everything here is re-implemented from the paper text (arXiv:2607.12418v1)
and, for the three host methods, from their own papers. "Implemented" means the code exists and follows the
written description; it does **not** mean the numbers match the paper (they cannot: the data is different).

## Coverage by paper element

| Paper element | Status | Where |
|---|---|---|
| Coarse stage: frozen-CLIP Top-K semantic anchors (IV-A) | implemented | `mqa/hosts/base.py::retrieve_topk` |
| Shared 8-qubit VQC, Eqs. 5-8 (angle enc., Ry, ring CRZ, Ry, Pauli-Z) | implemented + NumPy oracle | `mqa/quantum/` |
| TorchQuantum backend (library named in V-A2) | implemented, unverified | `mqa/quantum/tq_vqc.py` |
| Cross-modal quantum learning + residual injection, Eqs. 11-15 | implemented | `mqa/models/adapter.py`, `plugin.py` |
| Loss Eq. 16 | implemented (each host keeps its own extra terms, as the paper says it follows host protocols) | hosts |
| 0.078M extra params / 0.024K VQC params | matches (77,848 / 24) | `tests/test_hosts_engine.py` |
| Host 1: MaPLe | implemented | `mqa/hosts/maple.py` |
| Host 2: PromptSRC (prompts, 3 self-regulating losses, GPA, template ensemble) | implemented | `mqa/hosts/promptsrc.py` |
| Host 3: MMRL++ (representation space, SRRA, PRC, decoupled inference, Eq. 22 loss) | implemented (+ plain MMRL) | `mqa/hosts/mmrl.py` |
| Baselines: CLIP zero-shot, linear probe, CoOp, CoCoOp, CLIP-Adapter | implemented | `mqa/hosts/simple.py`, `engine/linear_probe.py` |
| Baselines MMA, TCP, COMMA, GraphAdapter, DePT, DPC, ATPrompt | **not implemented** | paper copies their numbers from their papers |
| Task: base-to-new (Table I) | implemented | `engine/data.py` |
| Task: few-shot 1/2/4/8/16 (Table II) | implemented | |
| Task: cross-dataset (Table III) | implemented (MRI targets: Figshare, Br35H; any folder dataset) | |
| Task: domain generalisation (Table IV) | implemented with synthetic MRI shifts (+ real target folders if you have them) | `mqa/shifts.py` |
| Table V ablation (cross-attention, separate VQC) | implemented (+ MLP / identity / frozen-VQC controls) | `main.py --tasks ablation` |
| Fig. 3 K and gamma sweeps, Fig. 4 insertion depth | implemented | `main.py --tasks sweeps` |
| Table VI stability over 3 seeds | implemented | `main.py --tasks stability` |
| Table VII params / FPS at batch 100 | implemented | `main.py --tasks efficiency` |
| Fig. 1 Top-K analysis, Fig. 5 t-SNE | implemented | `main.py --tasks topk tsne` |
| The 15 natural-image datasets | specs + templates only; you must export them to `Training/`/`Testing/` folders; CoOp's official split files are not used | `mqa/datasets.py` |

**Estimate:** about **85-90 %** of what the paper describes is now implemented as code
(all of the method, all three hosts, all four protocols, every ablation/analysis). What is missing is
mostly the paper's *comparison rows* (7 other methods whose numbers the paper copies) and ready-made loaders
for its 15 natural-image datasets. On brain MRI the reproduction is necessarily an *adaptation*: 4 classes
instead of 10-1000, no ImageNet source, synthetic domain shifts.

## Things that are inferred, not stated in a paper (check these first if results look odd)

| Item | Choice made | Why it matters |
|---|---|---|
| Adapted blocks "from the 6-th layer to the final" | 0-indexed blocks 6-11 (6 blocks) | matches 0.078M exactly; a 1-indexed reading gives 7 blocks / 0.090M |
| CRZ control/target on the ring | `control_first` | unspecified in the paper |
| K on a 4-class dataset | default K=2 (`--k`), clamped to the number of classes | paper K=5 needs >5 classes; with 2 base classes K is at most 2 |
| MMRL++ lambda / beta | 0.5 / 0.9 | paper tunes lambda per dataset (Table A1) |
| MMRL++ representation-token pooling | ln_post on each token, then mean | the paper says "mean pooling", not where ln_post sits |
| MMRL inference for domain-shifted sets | "base" (same classes) | the paper only says base tasks vs. novel tasks; `--set dg_mode=novel` to change |
| Few-shot epochs for MaPLe / CoCoOp / CLIP-Adapter / stand-alone adapter | 50 / 10 / 200 / 50 | not stated for these hosts in the MQAdapter paper |
| PromptSRC GPA mean when epochs differ from 20 | config `gpa_mean` (15 by default; 0.75 x epochs if unset) | official configs use 15 for 20 epochs |
| CoCoOp batch size | 4 (official 1) | speed |
| Precision | fp32 (official hosts use fp16) | `--mixed-precision` enables AMP; the VQC always runs in fp32 |

## What has and has not been executed

* The NumPy VQC oracle tests and protocol tests from the earlier version were executed.
* The new engine, hosts and `main.py` were written and syntax-checked; see the final message of the
  conversation that produced them (and `tests/test_hosts_engine.py`, `tests/test_engine_smoke.py`) for what was run.
* No real CLIP weights, no MRI data and no GPU were available while writing. **No accuracy number in this
  repository was produced by the author.** Every number you report must come from `runs/.../result.json`.
