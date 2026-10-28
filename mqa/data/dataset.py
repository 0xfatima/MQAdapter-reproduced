from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from mqa import CLASS_NAMES, CLASS_TO_IDX

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class CorruptImageError(RuntimeError):
    pass


def discover_split(root: Path, class_names: Sequence[str] = CLASS_NAMES) -> dict:
    """Inspect Training/ or Testing/ with class subfolders. Does not assume counts."""
    root = Path(root)
    found_dirs = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.is_dir() else []
    unexpected = [n for n in found_dirs if n not in class_names]
    missing = [n for n in class_names if n not in found_dirs]
    records = []
    per_class = {n: 0 for n in class_names}
    for cls in class_names:
        d = root / cls
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                records.append({"path": str(p), "class": cls, "label": CLASS_TO_IDX[cls]})
                per_class[cls] += 1
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "found_class_dirs": found_dirs,
        "missing_classes": missing,
        "unexpected_classes": unexpected,
        "per_class": per_class,
        "n_images": len(records),
        "records": records,
    }


class BrainTumorMRI(Dataset):
    def __init__(
        self,
        records: list[dict],
        transform: Callable | None = None,
        skip_corrupt: bool = False,
    ):
        self.records = list(records)
        self.transform = transform
        self.skip_corrupt = skip_corrupt

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = rec["path"]
        try:
            with Image.open(path) as im:
                im = im.copy()
        except Exception as exc:
            if self.skip_corrupt:
                # Return a black RGB image so DataLoader batching still works; caller should inspect.
                im = Image.new("RGB", (224, 224), 0)
            else:
                raise CorruptImageError(f"unreadable image: {path}") from exc
        if self.transform is not None:
            im = self.transform(im)
        return {"image": im, "label": rec["label"], "path": path, "class_name": rec["class"]}


def subset_records(records: list[dict], indices: Sequence[int]) -> list[dict]:
    return [records[i] for i in indices]
