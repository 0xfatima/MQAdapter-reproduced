"""CLIP ViT-B/16 loading and a layer-wise visual forward with CLS injection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa import CLIP_MEAN, CLIP_STD, CLASS_PROMPTS, CLASS_NAMES


@dataclass
class CLIPInfo:
    backend: str
    n_blocks: int
    width: int
    layout: str
    text_dim: int
    logit_scale: float
    preprocess_repr: str


def load_clip(backend: str = "open_clip", device: str | torch.device = "cpu", download: bool = True):
    """Load OpenAI CLIP ViT-B/16 (fp32) via open_clip or the openai `clip` package.

    download=False sets HF_HUB_OFFLINE so a missing local cache fails instead of fetching ~350 MB.
    """
    if backend == "open_clip":
        import os

        if not download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:  # huggingface_hub reads the env var at import time
                import huggingface_hub.constants as _hc

                _hc.HF_HUB_OFFLINE = True
            except Exception:
                pass
        import open_clip

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai", precision="fp32", device=device
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load OpenAI CLIP ViT-B/16. If this is the first run set "
                '"download_clip": true in the config (or pass --download-clip); weights are ~350 MB and cached afterwards.'
            ) from exc
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        return model.eval(), preprocess, tokenizer
    if backend == "clip":
        import clip

        model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
        return model.float().eval(), preprocess, clip.tokenize
    raise ValueError(f"unknown CLIP backend {backend}")


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def tokenize_prompts(tokenizer, prompts: list[str], device) -> torch.Tensor:
    tokens = tokenizer(prompts)
    if not torch.is_tensor(tokens):
        tokens = torch.tensor(tokens)
    return tokens.to(device)


def encode_class_texts(model, tokenizer, class_names=CLASS_NAMES, prompts=CLASS_PROMPTS, device="cpu"):
    texts = [prompts[n] for n in class_names]
    tokens = tokenize_prompts(tokenizer, texts, device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
    return F.normalize(feat.float(), dim=-1)


def transformer_batch_first(transformer: nn.Module) -> bool:
    """open_clip >= ~2.23 transformers are batch-first; the original OpenAI layout is sequence-first."""
    return bool(getattr(transformer, "batch_first", False))


def run_block(blk: nn.Module, x_nld: torch.Tensor, batch_first: bool, attn_mask: torch.Tensor | None = None):
    """Run one residual block on tokens kept canonically as [N, L, D], whatever the block expects."""
    x = x_nld if batch_first else x_nld.transpose(0, 1)
    out = blk(x) if attn_mask is None else blk(x, attn_mask=attn_mask)
    return out if batch_first else out.transpose(0, 1)


def visual_prefix(
    visual: nn.Module, images: torch.Tensor, extra_tokens: torch.Tensor | None = None
) -> tuple[torch.Tensor, str]:
    """Patch embed + CLS + pos (+ optional extra tokens) + ln_pre. Returns tokens as [N, L, D] ('nld').

    extra_tokens [N, n, D] are appended before ln_pre, which is where MaPLe attaches visual prompts.
    """
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    if hasattr(visual, "patch_dropout"):
        x = visual.patch_dropout(x)
    if extra_tokens is not None:
        x = torch.cat([x, extra_tokens.to(x.dtype)], dim=1)
    x = visual.ln_pre(x)
    return x, "nld"


def visual_pool(visual: nn.Module, tokens_nld: torch.Tensor) -> torch.Tensor:
    pooled = visual.ln_post(tokens_nld[:, 0, :])
    proj = getattr(visual, "proj", None)
    if proj is not None:
        pooled = pooled @ proj
    return pooled


class FakeCLIPVisual(nn.Module):
    """Minimal ViT-like visual encoder for tests that must not download CLIP weights."""

    def __init__(self, width: int = 768, n_blocks: int = 12, n_tokens: int = 197, out_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=16, stride=16, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width) * 0.02)
        self.positional_embedding = nn.Parameter(torch.randn(n_tokens, width) * 0.02)
        self.ln_pre = nn.LayerNorm(width)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, out_dim) * (width**-0.5))
        blocks = []
        for _ in range(n_blocks):
            blocks.append(_FakeBlock(width))
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(blocks)
        self.transformer.width = width
        self.output_dim = out_dim

    def forward(self, x):
        t, _ = visual_prefix(self, x)
        for b in self.transformer.resblocks:
            t = run_block(b, t, False)
        return visual_pool(self, t)


class _FakeBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.lin = nn.Linear(width, width)

    def forward(self, x):
        # x: LND
        return x + self.lin(self.ln(x))


class FakeCLIP(nn.Module):
    def __init__(self, width: int = 768, n_blocks: int = 12, text_dim: int = 512, vocab_ctx: int = 77):
        super().__init__()
        self.visual = FakeCLIPVisual(width=width, n_blocks=n_blocks, out_dim=text_dim)
        self.text_dim = text_dim
        self.token_embedding = nn.Embedding(49408, 64)
        self.text_proj = nn.Linear(64, text_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(4.60517))  # ~100, CLIP default-ish

    def encode_image(self, x):
        return self.visual(x)

    def encode_text(self, tokens):
        # tokens: [B, L]
        e = self.token_embedding(tokens.clamp(min=0, max=self.token_embedding.num_embeddings - 1))
        pooled = e.mean(dim=1)
        return self.text_proj(pooled)


class FakeTokenizer:
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for t in texts:
            ids = [1] + [min(100 + (ord(c) % 50), 200) for c in t[:75]]
            ids += [0] * (77 - len(ids))
            rows.append(ids[:77])
        return torch.tensor(rows, dtype=torch.long)


class _FakeTextBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.lin = nn.Linear(width, width)

    def forward(self, x, attn_mask=None):  # x: LND (sequence-first), like the original OpenAI CLIP
        return x + self.lin(self.ln(x))


class FakeCLIPWithText(FakeCLIP):
    """FakeCLIP plus the text-tower attributes MaPLe touches (token_embedding, transformer, ln_final, ...)."""

    def __init__(self, width: int = 768, n_blocks: int = 12, text_dim: int = 512, text_blocks: int = 12, ctx_len: int = 77):
        super().__init__(width=width, n_blocks=n_blocks, text_dim=text_dim)
        self.token_embedding = nn.Embedding(49408, text_dim)
        self.positional_embedding = nn.Parameter(torch.randn(ctx_len, text_dim) * 0.01)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([_FakeTextBlock(text_dim) for _ in range(text_blocks)])
        self.ln_final = nn.LayerNorm(text_dim)
        self.text_projection = nn.Parameter(torch.randn(text_dim, text_dim) * (text_dim**-0.5))
        self.register_buffer("attn_mask", torch.zeros(ctx_len, ctx_len), persistent=False)

    def encode_text(self, tokens):
        x = self.token_embedding(tokens.clamp(0, self.token_embedding.num_embeddings - 1)) + self.positional_embedding
        x = x.transpose(0, 1)
        for b in self.transformer.resblocks:
            x = b(x, attn_mask=self.attn_mask)
        x = self.ln_final(x.transpose(0, 1))
        return x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
