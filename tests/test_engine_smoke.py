"""main.py's engine end to end on synthetic folders with FakeCLIP (plumbing only; numbers are meaningless)."""
import json

import pytest

torch = pytest.importorskip("torch")

from mqa.engine import Engine  # noqa: E402
from mqa.engine.report import build_report  # noqa: E402
from scripts.make_dummy_dataset import make_all  # noqa: E402


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    roots = make_all(root)
    out = tmp_path_factory.mktemp("runs")
    cfg = dict(dataset="brain_tumor_mri", data_root=str(roots["brain_tumor_mri"]), backend="fake", device="cpu", out_root=str(out),
               augment="mri", num_workers=0, quick=True, quick_steps=2, overrides={"k": 2, "eval_batch_size": 8},
               cross_targets=["figshare_brain", "br35h"], target_roots={"figshare_brain": str(roots["figshare_brain"]), "br35h": str(roots["br35h"])})
    return Engine(cfg), out


@pytest.mark.parametrize("task,method,shots", [
    ("base2new", "zeroshot", None), ("base2new", "mmrlpp+mqa", None), ("fewshot", "maple+mqa", 2),
    ("cross", "promptsrc+mqa", None), ("dg", "coop", None), ("all", "linear_probe", None),
])
def test_engine_tasks(engine, task, method, shots):
    eng, out = engine
    r = eng.run(task, method, seed=1, shots=shots)
    assert r["evals"], r
    if task == "base2new":
        assert r["summary"]["H"] is not None
    if task == "cross":
        assert set(r["summary"]["targets"]) == {"figshare_brain", "br35h"}


def test_report(engine):
    eng, out = engine
    p = build_report(out)
    assert "Table I" in p.read_text()
    assert (out / "all_results.csv").exists()
    json.loads(next(out.glob("**/result.json")).read_text())
