import pytest


def test_open_clip_import_or_document_blocker():
    pytest.importorskip(
        "open_clip",
        reason="open_clip_torch is not installed; real CLIP tests are blocked. "
        "Install open_clip_torch and run scripts/phase3_clip_verify.py",
    )
