#!/usr/bin/env python
"""Paper Table VI-style stability: train the same config with several seeds, report mean +- std on Testing.

  python scripts/run_seeds.py --config configs/maple_mqadapter.json --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="extra key=value overrides")
    args = ap.parse_args()
    stem = Path(args.config).stem
    accs, f1s = [], []
    for s in args.seeds:
        out = ROOT / "runs" / f"{stem}_seed{s}"
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", args.config, "--seed", str(s), "--out-dir", str(out), "--eval-after"]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.extra:
            cmd += ["--set", *args.extra]
        subprocess.run(cmd, check=True)
        m = json.loads((out / "eval" / "metrics.json").read_text()) if (out / "eval" / "metrics.json").exists() else None
        if m is None:  # base2new writes eval/base and eval/new
            m = json.loads((out / "eval" / "base2new_summary.json").read_text())
            accs.append(m["harmonic_mean"]); f1s.append(float("nan"))
        else:
            accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
    res = {"config": stem, "seeds": args.seeds, "accuracy": accs, "macro_f1": f1s,
           "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs, ddof=0)),
           "macro_f1_mean": float(np.nanmean(f1s)) if not np.all(np.isnan(f1s)) else None}
    (ROOT / "runs" / f"{stem}_seeds_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
