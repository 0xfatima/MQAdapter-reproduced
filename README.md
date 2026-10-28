# MQAdapter for brain-tumor MRI - full reproduction pipeline

Paper-based reimplementation of **MQAdapter** (Zhao et al., arXiv:2607.12418v1) with all three host methods of the
paper (MaPLe, PromptSRC, MMRL++), the simple baselines (zero-shot CLIP, linear probe, CoOp, CoCoOp, CLIP-Adapter), all
four evaluation protocols and every ablation/analysis, driven by one script:

```text
python main.py --make-dummy-data data/dummy --backend fake --quick --seeds 1 --shots 1   # plumbing check, no downloads (~5-10 min CPU)
python main.py --data-root data/brain_tumor_mri                        # the whole paper suite on MRI (GPU hours)
```

* What is reproduced and what is not (about 85-90 %): `docs/REPRODUCTION_STATUS.md`
* How to run, outputs, cost: `docs/RUNNING.md`; datasets and the paper's 15 datasets: `docs/DATASETS.md`
* Engine code: `mqa/hosts/` (methods), `mqa/engine/` (splits, trainer, evaluation, analyses, report)

## Run it from scratch: everything you need to do

These are all the steps, in order, on the training machine (NVIDIA GPU, ~12 GB is enough; Python 3.10-3.12).
Nothing else has to be downloaded or configured by hand.

### 1. Unpack the code and create an environment

```text
unzip mqadapter_brain_tumor_v2.zip
cd mqadapter_brain_tumor
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch (CUDA build) and the requirements

```text
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```
The last line must print `CUDA available: True`. If it prints `False`, install the torch build that matches your
NVIDIA driver (https://pytorch.org/get-started/locally/); CPU training works but is very slow.
Optional, the paper's quantum library: `pip install torchquantum` (not needed; the default circuit is pure PyTorch).

### 3. Check the code (1-2 minutes, no downloads)

```text
python -m pytest -q tests
```
Expected: all passed, 2 skipped (the TorchQuantum and real-CLIP tests skip when those are absent).

### 4. Download the Brain Tumor MRI dataset

Dataset page: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**4a. Kaggle API token (once).** On kaggle.com: profile picture -> *Settings* -> *API* -> *Create New Token*.
This downloads `kaggle.json`. Put it in place:

```text
# Linux / macOS
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```
(Instead of the file you can set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.)

**4b. Download into `data/brain_tumor_mri/`.** Use one of these:

```text
python scripts/download_dataset.py                                 # kagglehub (installed by requirements.txt); falls back to the Kaggle CLI
python scripts/download_dataset.py --method cli                    # Kaggle CLI only (pip install kaggle)
python scripts/download_dataset.py --zip ~/Downloads/archive.zip   # no token: click "Download" on the Kaggle page first
```
With kagglehub the images stay in kagglehub's cache and `data/brain_tumor_mri` links to them; add `--copy` for a real
copy (on Windows without symlink rights it copies automatically). The script prints the image count per class and
must end with `layout OK`. The result must look like this:

```text
data/brain_tumor_mri/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
data/brain_tumor_mri/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
```

**4c. (Optional) cross-dataset targets for Table III.** Only used by `--tasks cross`; missing targets are skipped
with a message.

```text
data/figshare_brain/Testing/{glioma,meningioma,pituitary}/*.jpg   # Figshare brain tumour set (Cheng et al. 2017), exported to JPG
data/br35h/Testing/{no,yes}/*.jpg                                  # Br35H brain tumor detection 2020
```

### 5. Download the CLIP ViT-B/16 weights (once)

Every experiment uses OpenAI's CLIP ViT-B/16 through `open_clip`. Download and cache the weights once:

```text
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai'); print('CLIP ViT-B/16 weights cached')"
```
They are stored in the Hugging Face cache (`~/.cache/huggingface`; Windows `%USERPROFILE%\.cache\huggingface`).
To keep them on another disk, set the environment variable `HF_HOME` to that folder before this command and before
every later run. Check that they load without the network (this also produces the paper's Fig. 1 in
`runs/check_clip/topk/`):

```text
python main.py --tasks topk --no-download --out runs/check_clip
```
If you skip this step, the first real run downloads the weights automatically (unless you pass `--no-download`).

### 6. Plumbing check of the whole pipeline (fake CLIP, synthetic images, ~5-10 min)

```text
python main.py --make-dummy-data data/dummy --backend fake --quick --seeds 1 --shots 1 --out runs/smoke
```
It must finish with `report -> runs/smoke/REPORT.md` (the numbers in that report are meaningless).
If data loading hangs on Windows, add `--num-workers 0` here and to every later command.

### 7. The real experiments

```text
python main.py --dry-run                                          # list every run (default suite: ~212 runs, 3 seeds)
python main.py --tasks topk base2new report --seeds 1             # first real results (Table I, one seed)
python main.py                                                    # the whole paper suite on the MRI data (many GPU hours)
python main.py --tasks all report --methods baselines hosts qvlm   # full-data 4-class MRI comparison (not in the paper)
```
Runs that already finished are skipped, so you can stop and restart at any time. If GPU memory runs out, add
`--set batch_size=2` (and/or `--set eval_batch_size=16`).

### 8. Where the results are

* `runs/main/REPORT.md`: Tables I-VII, Fig. 1/3/4 numbers and the full-data table
* `runs/main/all_results.csv`: one row per run and evaluation set
* figures: `runs/main/fig5_tsne.png`, `runs/main/fig3_fig4_sweeps.png`, `runs/main/topk/fig1_topk.png`
* every run: `runs/main/<task>/<method>/shots*/seed*/` with `result.json`, metrics, predictions and confusion matrices

More options and cost notes: `docs/RUNNING.md`.

## Methods (`--methods`)

| name | what | paper role |
|---|---|---|
| `zeroshot` | frozen CLIP, hand prompts | "CLIP" row |
| `linear_probe` | logistic regression on frozen features | extra baseline |
| `clip_adapter`, `coop`, `cocoop` | classic fine-tuning | compared methods |
| `maple`, `promptsrc`, `mmrlpp` (`mmrl`) | the three host methods | hosts |
| `<any>+mqa` | same method with the MQAdapter plug-in | "+MQAdapter" rows |
| `zeroshot+mqa` | MQAdapter alone on frozen CLIP | extra |

## Tasks (`--tasks`)

`topk` (Fig. 1), `base2new` (Table I), `fewshot` (Table II), `cross` (Table III), `dg` (Table IV), `ablation`
(Table V), `sweeps` (Fig. 3/4), `stability` (Table VI), `efficiency` (Table VII), `tsne` (Fig. 5), `report`,
and `all` (full-data MRI, not in the paper). Results: `runs/main/REPORT.md`.

---

# Earlier single-run tooling (still works)


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
| Everything that needs `torch` (VQC, adapter, all hosts, training, eval, `main.py` engine) | **executed with a fake CLIP on CPU**: `pytest -q tests` -> 68 passed, 2 skipped (torch 2.4.1). Not run with real CLIP weights, real MRI data or a GPU |
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

PromptSRC, MMRL++, CoOp, CoCoOp, cross-dataset and domain generalisation are now in the `main.py` engine (see the top of this file).

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
