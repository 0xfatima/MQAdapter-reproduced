"""MQAdapter (paper-based reimplementation) adapted to 4-class brain MRI."""

__version__ = "0.1.0"

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: n for n, i in CLASS_TO_IDX.items()}

# Implementation choice (paper: "textual categories" / CLIP templates, unspecified MRI wording).
CLASS_PROMPTS = {
    "glioma": "An MRI scan showing a glioma brain tumor.",
    "meningioma": "An MRI scan showing a meningioma brain tumor.",
    "notumor": "A brain MRI showing no tumor.",
    "pituitary": "An MRI scan showing a pituitary brain tumor.",
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_IMAGE_SIZE = 224
N_QUBITS = 8
PAPER_VQC_PARAMS = 24  # 3N
DEFAULT_ADAPT_LAYERS = (6, 7, 8, 9, 10, 11)  # 0-indexed; D2
DEFAULT_GAMMA = 0.001
VISUAL_WIDTH = 768
TEXT_DIM = 512
