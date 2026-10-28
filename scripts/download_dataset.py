#!/usr/bin/env python
"""Download the Kaggle Brain Tumor MRI dataset (masoudnickparvar) and link it at data/brain_tumor_mri.

Needs Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY) and `pip install kagglehub`.
If you already downloaded the zip, just unzip it so that  data/brain_tumor_mri/Training/<class>  exists.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_root(base: Path) -> Path | None:
    for p in [base, *base.rglob("*")]:
        if p.is_dir() and (p / "Training").is_dir() and (p / "Testing").is_dir():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "data" / "brain_tumor_mri"))
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()
    dest = Path(args.dest)
    if dest.exists() and find_root(dest):
        print(f"already present: {dest}")
        return
    try:
        import kagglehub
    except ImportError:
        sys.exit("pip install kagglehub  (or download the zip manually and unzip it into data/brain_tumor_mri)")
    cache = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    src = find_root(cache)
    if src is None:
        sys.exit(f"Training/ and Testing/ not found under {cache}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink() if dest.is_symlink() else shutil.rmtree(dest)
    if args.copy:
        shutil.copytree(src, dest)
    else:
        try:
            dest.symlink_to(src, target_is_directory=True)
        except OSError:  # Windows without developer mode
            shutil.copytree(src, dest)
    print(f"dataset at {dest}")


if __name__ == "__main__":
    main()
