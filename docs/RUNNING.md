# Running the end-to-end pipeline (`main.py`)

## 0. Install (12 GB GPU, 64 GB RAM machine)

```text
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest -q tests          # do this first; report any failure with its traceback
```

## 1. Plumbing check on synthetic data (no downloads, CPU is fine)

```text
python main.py --make-dummy-data data/dummy --backend fake --quick --seeds 1 --shots 1 2
```
This runs every paper task for the default methods with a fake CLIP and writes `runs/main/REPORT.md`.
The numbers are meaningless; the point is that nothing crashes. Add `--out runs/smoke` to keep it apart.

## 2. Data

```text
python scripts/download_dataset.py                 # Kaggle Brain Tumor MRI -> data/brain_tumor_mri/{Training,Testing}
```
Cross-dataset targets (optional; skipped with a message if absent):
* `data/figshare_brain/Testing/{glioma,meningioma,pituitary}/*.jpg` (Cheng et al. 2017, exported to JPG)
* `data/br35h/Testing/{no,yes}/*.jpg` (Br35H)

Other locations: `--target-root figshare_brain=/path/to/it br35h=/path/to/that`.

## 3. Real runs

```text
python main.py --dry-run                                    # list every run that would be executed
python main.py --tasks topk                                 # Fig. 1 only (minutes)
python main.py --tasks base2new report --seeds 1            # Table I, one seed
python main.py                                              # whole paper suite, 3 seeds
python main.py --tasks all --methods baselines hosts qvlm   # full-data MRI comparison (not in the paper)
```

Useful flags

| flag | meaning |
|---|---|
| `--methods maple maple+mqa` | any method, `+mqa` adds MQAdapter; groups `baselines`, `hosts`, `qvlm`, `paper`, `everything` |
| `--k 3` | Top-K anchors (default 2 on 4-class MRI; paper 5) |
| `--set gamma=0.002 mmrl_lambda=1.0` | override any hyper-parameter for all runs |
| `--mixed-precision` | AMP for the CLIP towers (the VQC stays fp32) |
| `--selection best_val` | pick the epoch with the lowest validation loss instead of the last epoch |
| `--epochs-scale 0.5` | shorten every schedule |
| `--continue-on-error` | keep going if one run fails (failures in `runs/main/failures.json`) |
| `--force` | re-run runs that already have `result.json` (otherwise they are skipped, so you can resume) |

## 4. Outputs

```
runs/main/
  REPORT.md                 Tables I-VII, Fig. 1/3/4 numbers, full-data table
  all_results.csv           one row per (run, evaluation set)
  fig3_fig4_sweeps.png, fig5_tsne.png, topk/fig1_topk.png
  efficiency.json
  <task>/<method>[__variant]/shots<S>/seed<N>/
      result.json  config.json  split.json  train_summary.json  weights.pt (non-CLIP weights only)
      eval/<set>/metrics.json  predictions.csv  confusion_matrix.png  [features.npz]
```

## 5. Cost

Rough planning numbers, **not measured**: base-to-new and cross/DG runs train on 32-64 images (16 shots) and
take minutes; few-shot runs at 16 shots with 100-200 epochs take up to ~10-20 min for MMRL++ / CoOp; the full
"all" task trains on ~4.8k images and is the expensive one. `--dry-run` prints the run count; the default paper
suite is roughly 200 runs. Start with `--seeds 1`.

## 6. Interpreting MRI results

* With 4 classes the coarse Top-K stage filters little; with 2 base classes it filters nothing. Expect the
  MQAdapter effect (if any) to come from the fine stage, not from anchor selection. `coarse_topk_recall` in each
  `metrics.json` shows how often the true class was among the anchors.
* Base-to-new on 4 classes = train on {glioma, meningioma}, test new on {notumor, pituitary}; a 2-class "new"
  test is easy to get lucky on. Use 3 seeds and report the spread.
* The Kaggle split is not patient-independent; do not claim patient-level generalisation.
