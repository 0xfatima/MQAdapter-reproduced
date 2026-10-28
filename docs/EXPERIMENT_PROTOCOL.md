# Experiment protocol (brain tumor MRI)

## Dataset

Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected layout (do not assume counts):

```
<data_root>/Training/{glioma,meningioma,notumor,pituitary}/...
<data_root>/Testing/{glioma,meningioma,notumor,pituitary}/...
```

```
python scripts/inspect_dataset.py --data-root data/brain_tumor_mri --out docs/dataset_inspect.json
```

The official **Testing** split is held out until final evaluation. **Training** is split stratified into train/val (`val_ratio`, `seed`). Patient IDs are not assumed; **do not claim patient-independent evaluation**.

## Model selection

Select checkpoints by **validation loss** (early stopping). Then run `scripts/evaluate.py` on Testing once.

## Experiments (same split seed unless noted)

| Config | What it tests |
|---|---|
| `configs/frozen_clip_zeroshot.json` + `scripts/eval_zeroshot.py` | Frozen CLIP cosine classifier |
| `configs/clip_adapter.json` | CLIP-Adapter-style MLP on 512-d embeddings |
| `configs/mqadapter_k2.json` | Paper adapter, K=2 |
| `configs/mqadapter_k3.json` | Paper adapter, K=3 |
| `configs/mqadapter_k4.json` | Control: all 4 classes as anchors |
| `configs/classical_adapter_mlp.json` | Same adapter topology, MLP instead of VQC |
| `configs/quantum_ablation_identity.json` | \(f_\phi=\) identity |
| `configs/fewshot_16.json` | Optional 16-shot (paper-like); do not mix with full-data scores |

K=4 **cannot** demonstrate coarse subset selection.

## Hardware

User-declared: 12 GB NVIDIA GPU, 64 GB RAM. This implementation process used **CPU PyTorch** (`torch 2.6.0+cpu`, CUDA not visible). Measure batch size with `scripts/phase3_budget.py` on the GPU machine before a long run. Do not start a full training sweep from this CPU session.

## Metrics

Accuracy; macro P/R/F1; per-class P/R/F1 and support; confusion matrix; inference time; trainable parameter count. Saved under `runs/<exp>/eval/`.
