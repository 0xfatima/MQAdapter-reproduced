#!/usr/bin/env python
"""Collect runs/*/eval/**/metrics.json into one markdown + JSON table. Reports only what files contain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "summary.md"))
    args = ap.parse_args()
    runs = Path(args.runs)
    rows = []
    for mpath in sorted(runs.glob("*/eval/**/metrics.json")) + sorted(runs.glob("*/eval/metrics.json")):
        exp = mpath.relative_to(runs).parts[0]
        m = json.loads(mpath.read_text())
        rows.append({
            "experiment": exp, "set": m.get("eval_set", "all"), "n": m.get("n_images", m.get("n_samples")),
            "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
            "topk_recall": m.get("coarse_topk_recall"), "trainable_params": m.get("trainable_params"),
            "img_per_s": m.get("images_per_second"),
        })
    seen, uniq = set(), []
    for r in rows:
        key = (r["experiment"], r["set"])
        if key not in seen:
            seen.add(key); uniq.append(r)
    for sp in sorted(runs.glob("*/eval/base2new_summary.json")):
        s = json.loads(sp.read_text())
        uniq.append({"experiment": sp.parts[-3], "set": "H(base,new)", "n": None, "accuracy": s["harmonic_mean"], "macro_f1": None, "topk_recall": None, "trainable_params": None, "img_per_s": None})
    f = lambda v, p=4: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| experiment | eval set | n | accuracy | macro-F1 | coarse Top-K recall | trainable params | img/s |", "|---|---|---|---|---|---|---|---|"]
    for r in uniq:
        lines.append(f"| {r['experiment']} | {r['set']} | {f(r['n'])} | {f(r['accuracy'])} | {f(r['macro_f1'])} | {f(r['topk_recall'])} | {f(r['trainable_params'])} | {f(r['img_per_s'], 1)} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(uniq, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
