"""Dataset inspection: counts, modes, hashes, unreadable files. Does not assume a published count."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from mqa import CLASS_NAMES
from mqa.data.dataset import IMAGE_EXTS, discover_split


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ahash(img: Image.Image, size: int = 8) -> str:
    g = img.convert("L").resize((size, size))
    pixels = list(g.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return f"{int(bits, 2):0{size * size // 4}x}"


def inspect_dataset(data_root: str | Path, compute_near_dup: bool = True, max_files: int | None = None) -> dict:
    root = Path(data_root)
    report = {
        "data_root": str(root.resolve()) if root.exists() else str(root),
        "exists": root.exists(),
        "layout_note": "Expected Training/<class>/... and Testing/<class>/... (Kaggle masoudnickparvar/brain-tumor-mri-dataset).",
        "splits": {},
        "class_names_expected": list(CLASS_NAMES),
        "warnings": [],
        "patient_split": (
            "The supplied Training/Testing folders are used as-is for the official test split. "
            "Patient identifiers are not assumed to exist; the split is therefore not claimed to be patient-independent."
        ),
    }
    if not root.exists():
        report["warnings"].append("data_root does not exist")
        return report

    for split in ("Training", "Testing"):
        info = discover_split(root / split)
        recs = info.pop("records")
        if max_files is not None:
            recs = recs[:max_files]
        modes = Counter()
        sizes = Counter()
        exts = Counter()
        unreadable = []
        hashes = defaultdict(list)
        ahashes = defaultdict(list)
        for rec in recs:
            p = Path(rec["path"])
            exts[p.suffix.lower()] += 1
            try:
                hashes[ _md5(p) ].append(str(p))
                with Image.open(p) as im:
                    im.load()
                    modes[im.mode] += 1
                    sizes[f"{im.size[0]}x{im.size[1]}"] += 1
                    if compute_near_dup:
                        ahashes[_ahash(im)].append(str(p))
            except Exception as exc:
                unreadable.append({"path": str(p), "error": repr(exc)})
        exact_dups = {h: ps for h, ps in hashes.items() if len(ps) > 1}
        near = {}
        if compute_near_dup:
            near = {h: ps for h, ps in ahashes.items() if len(ps) > 1}
            # average-hash collisions are not true duplicates; report only as candidates
        info.update(
            {
                "n_scanned": len(recs),
                "unreadable": unreadable,
                "modes": dict(modes),
                "extensions": dict(exts),
                "size_histogram_top": dict(sizes.most_common(20)),
                "n_unique_sizes": len(sizes),
                "n_exact_duplicate_groups": len(exact_dups),
                "exact_duplicate_groups_sample": dict(list(exact_dups.items())[:20]),
                "n_ahash_collision_groups": len(near),
                "ahash_note": "8x8 average hash; collisions are near-duplicate *candidates*, not verified duplicates.",
            }
        )
        report["splits"][split] = info

    # cross-split exact hash overlap
    if "Training" in report["splits"] and "Testing" in report["splits"]:
        # recompute lightweight: inspect already stored sample only — full overlap needs hashes.
        # Re-hash both splits for overlap if they are not huge; MRI ~7k files is fine.
        def all_hashes(split_name):
            info = discover_split(root / split_name)
            m = {}
            for rec in info["records"]:
                p = Path(rec["path"])
                try:
                    m.setdefault(_md5(p), []).append(str(p))
                except Exception:
                    pass
            return m

        tr, te = all_hashes("Training"), all_hashes("Testing")
        overlap = sorted(set(tr) & set(te))
        report["cross_split_exact_hash_overlap"] = {
            "n_hashes": len(overlap),
            "sample": [{ "hash": h, "train": tr[h], "test": te[h] } for h in overlap[:20]],
            "note": "Identical files in Training and Testing leak evaluation if present.",
        }
    return report


def write_inspect_report(data_root: str | Path, out_path: str | Path, **kwargs) -> dict:
    report = inspect_dataset(data_root, **kwargs)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
