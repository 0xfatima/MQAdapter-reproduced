# MQAdapter for 4-class brain tumor MRI

Paper-based reimplementation of **MQAdapter** (Zhao et al., arXiv:2607.12418v1) adapted to the [Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (`glioma`, `meningioma`, `pituitary`, `notumor`).

There is **no official code**. The NumPy circuit in `mqa/quantum/reference_paper_vqc.py` is the equation-level oracle (Eqs. 5–8). Training uses a matching PyTorch **statevector** VQC, not a classical MLP labelled as quantum.

Read `docs/IMPLEMENTATION_AUDIT.md`, `docs/ARCHITECTURE.md`, `docs/COMPLIANCE.md`, and `docs/LIMITATIONS.md` before claiming paper equivalence.

## What is implemented vs verified

| Stage | Status in the author’s environment (Windows, `torch 2.6.0+cpu`) |
|---|---|
| Paper VQC oracle | **Unit-tested** (6 tests, passed) |
| PyTorch statevector VQC + adapter + metrics + data + checkpoint | **Unit/integration-tested** (see pytest below) |
| Real CLIP ViT-B/16 | **Code present, not run** (`open_clip` / weights missing) |
| TorchQuantum | **Not installed**; optional |
| GPU training | **Not run** (CUDA not visible in this process) |
| Kaggle MRI training / held-out test | **Not run** (dataset not in the workspace) |

Do not invent accuracy numbers. If you train, report the JSON files under `runs/`.

## Environment

- Python 3.10–3.12
- Intended training machine: **12 GB NVIDIA GPU, 64 GB RAM** (user-declared)
- This checkout’s default `python` was CPU-only. Install a CUDA wheel yourself if `nvidia-smi` works on your box:

```text
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CLIP ViT-B/16 weights are ~350 MB. This repo **does not download them unless you pass `--download-clip`**.

TorchQuantum is **not required**. `scripts/phase3_env_check.py` still tries it if present.

## Dataset

```text
data/brain_tumor_mri/Training/<class>/*.jpg|png
data/brain_tumor_mri/Testing/<class>/*.jpg|png
```

```text
python scripts/inspect_dataset.py --data-root data/brain_tumor_mri --out docs/dataset_inspect.json
```

Training/validation are a **stratified** split of `Training/` only. `Testing/` is untouched until `scripts/evaluate.py`. The Kaggle split is **not** claimed to be patient-independent.

## Tests

```text
python -m pytest -q tests
python scripts/env_check.py
python scripts/smoke_test.py
```

CLIP / GPU / TorchQuantum checks (need extra deps + hardware):

```text
python scripts/phase3_clip_verify.py --out phase3_clip_report.json
python scripts/phase3_env_check.py --out phase3_env_report.json
python scripts/phase3_budget.py --out phase3_budget_report.json
```

## Training (ask before a long GPU run)

Smoke (FakeCLIP, synthetic images, no download):

```text
python scripts/smoke_test.py
```

Real CLIP + MRI (after data + weights are local):

```text
python scripts/train.py --config configs/mqadapter_k2.json --data-root data/brain_tumor_mri --download-clip
python scripts/train.py --config configs/mqadapter_k3.json --data-root data/brain_tumor_mri
python scripts/train.py --config configs/mqadapter_k4.json --data-root data/brain_tumor_mri
python scripts/train.py --config configs/classical_adapter_mlp.json --data-root data/brain_tumor_mri
python scripts/train.py --config configs/clip_adapter.json --data-root data/brain_tumor_mri
python scripts/train.py --config configs/gpu_12gb.json --data-root data/brain_tumor_mri --download-clip
```

K=4 is a **control** (all classes retrieved). Frozen CLIP:

```text
python scripts/eval_zeroshot.py --data-root data/brain_tumor_mri --download-clip
```

Evaluate a trained checkpoint on official Testing:

```text
python scripts/evaluate.py --checkpoint runs/mqadapter_k2/best.pt --data-root data/brain_tumor_mri
```

Outputs: `runs/<exp>/best.pt`, `last.pt`, `history.jsonl`, `train_summary.json`, `eval/metrics.json`, `eval/predictions.json`, `eval/confusion_matrix.png`.

## Top-K on 4 classes

The paper uses K=5 on datasets with many classes. Here C=4, so K∈{2,3,4}. K=4 cannot test subset selection.

## Citation

Zhao, Jiang, Lu, Wang, Tang. *MQAdapter: Multi-Modal Quantum Adapter for Coarse-to-Fine VLM Fine-tuning*. arXiv:2607.12418.
