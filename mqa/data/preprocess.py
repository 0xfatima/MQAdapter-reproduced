"""CLIP ViT-B/16 preprocessing. Implementation choice D11: pad-to-square instead of center-crop."""
from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageOps
from torchvision import transforms

from mqa import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD


class PadToSquare:
    def __init__(self, fill: int | Sequence[int] = 0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        return ImageOps.pad(img, (side, side), color=self.fill, centering=(0.5, 0.5))


class RepeatToRGB:
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode == "RGB":
            return img
        if img.mode == "L":
            return img.convert("RGB")
        if img.mode == "RGBA":
            return img.convert("RGB")
        return img.convert("RGB")


def build_clip_transform(train: bool = False, image_size: int = CLIP_IMAGE_SIZE) -> transforms.Compose:
    """CLIP normalization and 224 resolution. Train: optional small rotation only (no hflip).

    Horizontal flip is disabled: laterality can be clinically meaningful in brain MRI
    (implementation choice; paper does not discuss MRI).
    """
    ops = [RepeatToRGB(), PadToSquare(fill=0)]
    if train:
        ops.append(transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC))
        ops.append(transforms.RandomRotation(degrees=8, interpolation=transforms.InterpolationMode.BICUBIC, fill=0))
    else:
        ops.append(transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    return transforms.Compose(ops)
