#!/usr/bin/env python
"""End-to-end pipeline on the Brain Tumor MRI dataset.

  python scripts/run_pipeline.py --stages all            # everything (hours on a 12 GB GPU)
  python scripts/run_pipeline.py --stages env data zeroshot --quick
  python scripts/run_pipeline.py --stages maple maple_mqa summary

Stages: env data inspect topk zeroshot clip_adapter mqadapter classical maple maple_mqa base2new efficiency summary
--quick caps every training run at 2 epochs / 30 steps to check that the plumbing works (numbers are meaningless).
Testing is evaluated once per run (train.py --eval-after). Nothing here fabricates numbers: it only runs scripts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ALL = ["env", "data", "inspect", "topk", "zeroshot", "clip_adapter", "mqadapter", "classical", "maple", "maple_mqa", "base2new", "efficiency", "summary"]


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def train(cfg, quick, *extra):
    cmd = [PY, "scripts/train.py", "--config", f"configs/{cfg}.json", "--eval-after", *extra]
    if quick:
        cmd += ["--epochs", "2", "--max-steps", "30", "--out-dir", f"runs/quick_{cfg}"]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["all"])
    ap.add_argument("--data-root", default="data/brain_tumor_mri")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    stages = ALL if "all" in args.stages else args.stages
    dr = ["--data-root", args.data_root]
    for st in stages:
        if st == "env":
            run([PY, "scripts/env_check.py"])
        elif st == "data":
            run([PY, "scripts/download_dataset.py"])
        elif st == "inspect":
            run([PY, "scripts/inspect_dataset.py", *dr, "--out", "docs/dataset_inspect.json"])
        elif st == "topk":
            run([PY, "scripts/topk_analysis.py", *dr])
        elif st == "zeroshot":
            run([PY, "scripts/eval_zeroshot.py", *dr, "--download-clip"])
        elif st == "clip_adapter":
            train("clip_adapter", args.quick, *dr)
        elif st == "mqadapter":
            for c in ("mqadapter_k2", "mqadapter_k3", "mqadapter_k4"):
                train(c, args.quick, *dr)
        elif st == "classical":
            train("classical_adapter_mlp", args.quick, *dr)
        elif st == "maple":
            train("maple", args.quick, *dr)
        elif st == "maple_mqa":
            for c in ("maple_mqadapter", "maple_mqadapter_k3", "maple_mqadapter_k4", "maple_classical_mlp"):
                train(c, args.quick, *dr)
        elif st == "base2new":
            for c in ("mqadapter_k2_base2new", "maple_base2new", "maple_mqadapter_base2new"):
                train(c, args.quick, *dr)
        elif st == "efficiency":
            run([PY, "scripts/efficiency.py", "--configs", "configs/maple.json", "configs/maple_mqadapter.json", "configs/mqadapter_k2.json", "configs/clip_adapter.json"])
        elif st == "summary":
            run([PY, "scripts/summarize.py"])
        else:
            sys.exit(f"unknown stage {st}")


if __name__ == "__main__":
    main()
