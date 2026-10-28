"""Dataset specifications for the end-to-end engine (main.py).

A *spec* says where a dataset lives, which folder holds which split, what the classes are called in a
prompt and which hand-written template(s) to use. Every dataset is read from an image-folder layout:

    <root>/<train_split>/<class_folder>/*.jpg
    <root>/<test_split>/<class_folder>/*.jpg

Built-in specs:
  brain_tumor_mri   Kaggle "Brain Tumor MRI" (Nickparvar), 4 classes          (primary dataset)
  figshare_brain    Figshare/Cheng 2017 T1-CE brain tumour set, 3 classes     (cross-dataset target)
  br35h             Br35H "Brain Tumor Detection 2020", yes/no                 (cross-dataset target)
  + the 11 paper datasets (imagenet, caltech101, ...) with the templates of the paper / CoOp, if you
    export them to the same folder layout (see docs/DATASETS.md).

Any JSON file with the same keys can be passed instead of a built-in name (configs/datasets/*.json).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

MRI_TEMPLATES = [
    "An MRI scan showing {}.",
    "A brain MRI showing {}.",
    "A T1-weighted MRI of {}.",
    "A contrast-enhanced brain MRI with {}.",
    "An axial MRI slice showing {}.",
    "A medical image of {}.",
    "A radiology image of {}.",
    "A photo of {}.",
]


@dataclass
class DatasetSpec:
    name: str
    root: str = ""
    train_split: str = "Training"
    test_split: str = "Testing"
    val_split: str | None = None
    classes: list[str] | None = None          # folder names, in label order; None = sorted folders
    display: dict[str, str] = field(default_factory=dict)  # folder name -> words used in prompts
    template: str = "a photo of a {}."        # hand-written prompt (zero-shot, anchors, MMRL text input)
    templates: list[str] = field(default_factory=list)  # ensemble (PromptSRC teacher); default [template]
    aliases: dict[str, str] = field(default_factory=dict)  # folder name -> canonical class (cross-dataset)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def display_name(self, cls: str) -> str:
        return self.display.get(cls, cls.replace("_", " "))

    def hand_prompt(self, cls: str) -> str:
        return self.template.format(self.display_name(cls))

    def all_templates(self) -> list[str]:
        return list(self.templates) if self.templates else [self.template]


BUILTIN: dict[str, dict[str, Any]] = {
    "brain_tumor_mri": dict(
        root="data/brain_tumor_mri",
        classes=["glioma", "meningioma", "notumor", "pituitary"],
        display={
            "glioma": "a glioma brain tumor",
            "meningioma": "a meningioma brain tumor",
            "notumor": "a healthy brain with no tumor",
            "pituitary": "a pituitary brain tumor",
        },
        template="An MRI scan showing {}.",
        templates=MRI_TEMPLATES,
        notes="Kaggle masoudnickparvar/brain-tumor-mri-dataset; split is not patient-independent.",
    ),
    "figshare_brain": dict(
        root="data/figshare_brain",
        train_split="Training",
        test_split="Testing",
        classes=["glioma", "meningioma", "pituitary"],
        display={
            "glioma": "a glioma brain tumor",
            "meningioma": "a meningioma brain tumor",
            "pituitary": "a pituitary brain tumor",
        },
        template="An MRI scan showing {}.",
        templates=MRI_TEMPLATES,
        notes="Cheng et al. 2017 (figshare 1512427) exported to JPG folders. Put all images in Testing/ if only used as target.",
    ),
    "br35h": dict(
        root="data/br35h",
        test_split="Testing",
        classes=["no", "yes"],
        display={"no": "a healthy brain with no tumor", "yes": "a brain tumor"},
        template="An MRI scan showing {}.",
        templates=MRI_TEMPLATES,
        aliases={"no": "notumor"},
        notes="Br35H 2020 (yes/no). Layout: Testing/no, Testing/yes.",
    ),
    # --- the paper's datasets (templates from CoOp / MMRL++ Table 1). Export to Training/Testing folders. ---
    "imagenet": dict(root="data/imagenet", template="a photo of a {}."),
    "caltech101": dict(root="data/caltech101", template="a photo of a {}."),
    "oxford_pets": dict(root="data/oxford_pets", template="a photo of a {}, a type of pet."),
    "stanford_cars": dict(root="data/stanford_cars", template="a photo of a {}."),
    "oxford_flowers": dict(root="data/oxford_flowers", template="a photo of a {}, a type of flower."),
    "food101": dict(root="data/food101", template="a photo of {}, a type of food."),
    "fgvc_aircraft": dict(root="data/fgvc_aircraft", template="a photo of a {}, a type of aircraft."),
    "sun397": dict(root="data/sun397", template="a photo of a {}."),
    "dtd": dict(root="data/dtd", template="{} texture."),
    "eurosat": dict(root="data/eurosat", template="a centered satellite photo of {}."),
    "ucf101": dict(root="data/ucf101", template="a photo of a person doing {}."),
    "imagenet_v2": dict(root="data/imagenet_v2", template="a photo of a {}."),
    "imagenet_sketch": dict(root="data/imagenet_sketch", template="a photo of a {}."),
    "imagenet_a": dict(root="data/imagenet_a", template="a photo of a {}."),
    "imagenet_r": dict(root="data/imagenet_r", template="a photo of a {}."),
}


def load_spec(name_or_path: str | dict, root: str | None = None) -> DatasetSpec:
    """Built-in name, path to a JSON spec, or a dict."""
    if isinstance(name_or_path, dict):
        d = dict(name_or_path)
    elif str(name_or_path).endswith(".json"):
        d = json.loads(Path(name_or_path).read_text(encoding="utf-8"))
    else:
        key = str(name_or_path)
        if key not in BUILTIN:
            raise KeyError(f"unknown dataset {key!r}; built-ins: {sorted(BUILTIN)} (or pass a .json spec)")
        d = dict(BUILTIN[key], name=key)
    d.setdefault("name", Path(str(name_or_path)).stem if not isinstance(name_or_path, dict) else "custom")
    if root:
        d["root"] = root
    return DatasetSpec(**d)


def _class_dirs(split_dir: Path) -> list[str]:
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir()) if split_dir.is_dir() else []


def resolve_classes(spec: DatasetSpec) -> list[str]:
    if spec.classes:
        return list(spec.classes)
    root = Path(spec.root)
    for split in (spec.train_split, spec.test_split):
        found = _class_dirs(root / split) if split else []
        if found:
            return found
    raise SystemExit(f"cannot infer classes for {spec.name}: no class folders under {root}")


def scan_split(spec: DatasetSpec, split: str | None, classes: list[str] | None = None) -> list[dict]:
    """Records {path, label, class} for one split; label = index into `classes`."""
    if not split:
        return []
    classes = classes or resolve_classes(spec)
    d = Path(spec.root) / split
    recs: list[dict] = []
    for i, c in enumerate(classes):
        cd = d / c
        if not cd.is_dir():
            continue
        for p in sorted(cd.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                recs.append({"path": str(p), "label": i, "class": c})
    return recs


def describe(spec: DatasetSpec) -> dict:
    classes = resolve_classes(spec)
    out = {"name": spec.name, "root": spec.root, "classes": classes}
    for split in (spec.train_split, spec.val_split, spec.test_split):
        if split:
            recs = scan_split(spec, split, classes)
            out[split] = {c: sum(1 for r in recs if r["class"] == c) for c in classes}
    return out


class ClassSpace:
    """Names + prompts for one evaluation/training label space."""

    def __init__(self, spec: DatasetSpec, classes: list[str]):
        self.spec = spec
        self.classes = list(classes)
        self.display = [spec.display_name(c) for c in classes]
        self.hand_prompts = [spec.hand_prompt(c) for c in classes]
        self.templates = spec.all_templates()

    def __len__(self) -> int:
        return len(self.classes)

    def key(self) -> tuple:
        return (self.spec.name, tuple(self.classes))
