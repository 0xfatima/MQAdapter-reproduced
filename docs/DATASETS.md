# Dataset specs

`main.py --dataset NAME` takes a built-in spec (`mqa/datasets.py::BUILTIN`) or a JSON file:

```json
{
  "name": "my_mri",
  "root": "data/my_mri",
  "train_split": "Training",
  "test_split": "Testing",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"],
  "display": {"notumor": "a healthy brain with no tumor"},
  "template": "An MRI scan showing {}.",
  "templates": ["An MRI scan showing {}.", "A brain MRI showing {}."]
}
```

* `classes` = folder names in label order (omit to use sorted folder names).
* `display` = words put into prompts; `template` = hand-written prompt (zero-shot, Top-K anchors, MMRL text);
  `templates` = ensemble used by PromptSRC's textual-diversity teacher.
* Layout: `<root>/<train_split>/<class>/*.{jpg,png,...}` and `<root>/<test_split>/<class>/...`.
  A `val_split` can be given; otherwise 15 % of the training split is held out (stratified).

## The paper's 15 datasets

Built-in names `imagenet, caltech101, oxford_pets, stanford_cars, oxford_flowers, food101, fgvc_aircraft,
sun397, dtd, eurosat, ucf101, imagenet_v2, imagenet_sketch, imagenet_a, imagenet_r` carry the paper's prompt
templates. No loaders for their original formats are included: export each to the folder layout above and
use `--augment clip`. Class order and base/new halves then follow sorted folder names, which differs from
CoOp's `split_zhou_*.json` files, so numbers are not directly comparable to the paper.

Example (cross-dataset like Table III):
```text
python main.py --dataset imagenet --data-root /data/imagenet_folders --tasks cross \
   --cross-targets caltech101 oxford_pets dtd --target-root caltech101=/data/caltech oxford_pets=/data/pets dtd=/data/dtd
```
