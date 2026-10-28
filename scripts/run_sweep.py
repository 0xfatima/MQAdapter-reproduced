#!/usr/bin/env python
"""Paper ablations on MRI: K (Fig. 3a), gamma (Fig. 3b), insertion depth (Fig. 4), Table V variants.

  python scripts/run_sweep.py --sweep k gamma depth table5 --base configs/maple_mqadapter.json
Each point = one train.py run with --eval-after (Testing is read only at the end of each run).
Selecting a sweep winner by Testing accuracy would leak the test set: choose by best_val_loss in
runs/<exp>/train_summary.json and report Testing only for that choice.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SWEEPS = {
    "k": [(f"k{k}", {"k": k}) for k in (2, 3, 4)],  # paper K in {2,4,5,6,8,10}; limited by C=4
    "gamma": [(f"gamma{g}", {"gamma": g}) for g in (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1)],
    "depth": [(f"from{d}", {"adapt_layers": list(range(d, 12))}) for d in (4, 5, 6, 7, 8, 9, 10, 11)],
    "table5": [
        ("cross_attention", {"interaction": "cross_attention"}),
        ("separate_vqc", {"vqc_sharing": "separate"}),
        ("classical_mlp", {"quantum_map": "mlp"}),
        ("identity", {"quantum_map": "identity"}),
        ("frozen_random_vqc", {"quantum_map": "frozen_vqc"}),
    ],
}


def main():
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", choices=list(SWEEPS) + ["all"], required=True)
    ap.add_argument("--base", default=str(ROOT / "configs" / "maple_mqadapter.json"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = list(SWEEPS) if "all" in args.sweep else args.sweep
    stem = Path(args.base).stem
    for sw in names:
        for tag, ov in SWEEPS[sw]:
            out = ROOT / "runs" / f"sweep_{stem}_{sw}_{tag}"
            sets = [f"{k}={json.dumps(v)}" for k, v in ov.items()]
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", args.base, "--out-dir", str(out), "--eval-after", "--set", *sets]
            if args.data_root:
                cmd += ["--data-root", args.data_root]
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
