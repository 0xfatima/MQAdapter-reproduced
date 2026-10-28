"""run_block must give identical results whether the blocks are batch-first (new open_clip) or sequence-first."""
from __future__ import annotations

import torch
import torch.nn as nn

from mqa.models.clip_backbone import run_block, transformer_batch_first


class SeqFirstBlock(nn.Module):  # mixes along dim 0 == tokens in [L, N, D]
    def forward(self, x, attn_mask=None):
        return x + x.mean(dim=0, keepdim=True)


class BatchFirstBlock(nn.Module):  # mixes along dim 1 == tokens in [N, L, D]
    def forward(self, x, attn_mask=None):
        return x + x.mean(dim=1, keepdim=True)


def test_run_block_layout_invariance():
    x = torch.randn(3, 7, 5)  # canonical [N, L, D]
    a = run_block(SeqFirstBlock(), x, batch_first=False)
    b = run_block(BatchFirstBlock(), x, batch_first=True)
    assert torch.allclose(a, b, atol=1e-6)
    wrong = SeqFirstBlock()(x)  # what the old code did on batch-first CLIP: mixes across the batch axis
    assert not torch.allclose(a, wrong)


def test_batch_first_detection():
    m = nn.Module()
    assert transformer_batch_first(m) is False
    m.batch_first = True
    assert transformer_batch_first(m) is True
