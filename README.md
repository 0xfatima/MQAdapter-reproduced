# MQAdapter for 4-class brain-tumor MRI (end-to-end pipeline)

Paper-based reimplementation of **MQAdapter** (Zhao et al., arXiv:2607.12418v1) on the
[Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(`glioma`, `meningioma`, `notumor`, `pituitary`). There is no official code; the NumPy circuit in
`mqa/quantum/reference_paper_vqc.py` is the equation-level oracle (Eqs. 5-8).

* Is there a separate Q-VLM architecture in the paper? **No** - MQAdapter is the hybrid part. See `docs/Q_VLM_ANALYSIS.md`.
* What differs from the paper and what is unverified: `docs/COMPLIANCE.md`, `docs/LIMITATIONS.md`, `docs/IMPLEMENTATION_AUDIT.md`.

## Honest status

| Item | Status |
|---|---|
| NumPy VQC oracle (6 tests) | **executed, passed** |
| Protocol logic: all / base-to-new / few-shot (3 tests) | **executed, passed** (torch stubbed) |
| Cross-file import consistency, `py_compile` on every file | **executed, clean** |
| Everything that needs `torch` (VQC module, adapter, MaPLe, training, eval) | **written, not executed** - the authoring sandbox had no torch, no GPU, no network. Run `pytest -q tests` first on your machine |
| TorchQuantum backend + parity test | **written, not executed**; test auto-skips without `torchquantum` |
| Real CLIP weights / Kaggle data / accuracy numbers | **none produced** - nothing here reports a result you did not generate |

If `pytest` shows a failure, that is a bug in my code, not in your setup - please report the traceback.

## What is implemented

| Paper component | Where |
|---|---|
| Frozen-CLIP Top-K semantic anchors (IV-A) | `retrieve_topk` in `MQAdapterCLIP`, `MaPLeCLIP` |
| Shared 8-qubit VQC (Eqs. 5-8): PyTorch statevector | `mqa/quantum/torch_vqc.py` |
| Same VQC in **TorchQuantum** (paper's library) | `mqa/quantum/tq_vqc.py` (`quantum_map: "tq_vqc"`) |
| Cross-modal quantum learning + residual injection (Eqs. 11-15) | `mqa/models/plugin.py`, `adapter.py` |
| Eq. 16 cosine cross-entropy | `forward` of every model |
| **MaPLe host** (deep coupled text/vision prompts) + MQAdapter on top | `mqa/models/maple.py` |
| Stand-alone MQAdapter on frozen CLIP | `mqa/models/mqadapter_clip.py` |
| Table V ablations: cross-attention, separate VQC; plus MLP / identity / frozen-random-VQC controls | `interaction`, `vqc_sharing`, `quantum_map` config keys |
| Few-shot (1/2/4/8/16), base-to-new (+ harmonic mean) | `mqa/protocols.py` |
| Fig. 1 Top-K analysis, Fig. 5 t-SNE, Table VI seeds, Table VII params/FPS, K / gamma / depth sweeps | `scripts/topk_analysis.py`, `tsne_features.py`, `run_seeds.py`, `efficiency.py`, `run_sweep.py` |
| One-command pipeline | `scripts/run_pipeline.py` |

**Not implemented:** PromptSRC and MMRL++ hosts (their exact training details could not be checked offline, and
a look-alike would be misleading), cross-dataset transfer and domain generalisation (need a second dataset).

## Bugs fixed in the earlier code

1. `visual_prefix` assumed sequence-first tokens. Recent `open_clip` is batch-first, which would have silently run attention across the batch and rewritten the wrong "class token". Now layout-aware (`run_block`, tested in `tests/test_layout.py`).
2. The VQC was only *documented* as fp32-safe; it now disables autocast internally.
3. `load_clip(download=False)` still downloaded; it now sets `HF_HUB_OFFLINE`. Configs default to `download_clip: true` (first run fetches ~350 MB).
4. The frozen CLIP could drift into train mode; models now keep it in `eval()`.
5. `best_val` was saved one epoch stale in checkpoints.

## Setup (12 GB GPU, 64 GB RAM)

```text
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# optional, the paper's quantum library (Python <= 3.11 is safest):  pip install torchquantum
python -m pytest -q tests
python scripts/download_dataset.py            # needs Kaggle credentials; or unzip into data/brain_tumor_mri
```

## Run

```text
python scripts/run_pipeline.py --stages env data inspect topk zeroshot --quick     # cheap plumbing check
python scripts/run_pipeline.py --stages all                                        # full run (hours)
```

Individual runs (Testing is read only with `--eval-after` or `scripts/evaluate.py`):

```text
python scripts/train.py --config configs/maple.json            --eval-after   # host only
python scripts/train.py --config configs/maple_mqadapter.json  --eval-after   # paper setting, K=2
python scripts/train.py --config configs/maple_mqadapter.json  --eval-after --set k=3 gamma=0.002
python scripts/train.py --config configs/maple_mqadapter_base2new.json --eval-after
python scripts/run_seeds.py --config configs/maple_mqadapter.json --seeds 0 1 2
python scripts/run_sweep.py --sweep k gamma depth table5 --base configs/maple_mqadapter.json
python scripts/efficiency.py --configs configs/maple.json configs/maple_mqadapter.json
python scripts/tsne_features.py --checkpoint runs/maple_mqadapter_k2/best.pt
python scripts/summarize.py                                                    # runs/summary.md
```

**VRAM:** defaults are batch 16 (MaPLe) and 8 (stand-alone). These are my estimates, not measurements; on `CUDA out of memory` lower `batch_size` (`--set batch_size=8`).
`scripts/phase3_budget.py` measures real usage.

## Things to know before interpreting results

* **C = 4.** The paper uses K = 5 on datasets with up to 1000 classes. Here K in {2, 3, 4}; K = 4 retrieves every class, so it is a control, not evidence of coarse filtering.
* **gamma = 0.001 is tuned for hosts.** In the *stand-alone* adapter on frozen CLIP the residual `gamma * W f(h)` is tiny next to the class-token magnitude, so it may barely change predictions (my reasoning, not measured). Sweep gamma (`--sweep gamma`) and pick by validation loss.
* Pick sweep winners by `best_val_loss` in `train_summary.json`, never by Testing accuracy.
* The Kaggle split is not patient-independent; do not claim patient-level generalisation.
* Quantum part is a **classical simulation** of 8 qubits; no quantum speed-up is implied.
