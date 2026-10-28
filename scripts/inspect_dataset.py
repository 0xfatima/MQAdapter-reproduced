#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mqa.data.inspect_dataset import inspect_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "brain_tumor_mri"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "dataset_inspect.json"))
    ap.add_argument("--no-near-dup", action="store_true")
    args = ap.parse_args()
    report = inspect_dataset(args.data_root, compute_near_dup=not args.no_near_dup)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "splits"}, indent=2))
    for split, info in report.get("splits", {}).items():
        slim = {k: info[k] for k in info if k not in ("exact_duplicate_groups_sample",)}
        print(split, json.dumps(slim, indent=2)[:2000])


if __name__ == "__main__":
    main()
