"""Collect <out_root>/**/result.json into the paper's tables (markdown + CSV + JSON).

  Table I    base-to-new (Base / New / H) with the Δ of each "+mqa" row over its host
  Table II   few-shot accuracy for 1/2/4/8/16 shots
  Table III  cross-dataset transfer
  Table IV   domain generalisation
  Table V    ablation on MMRL++ (cross-attention / separate VQC / MQAdapter; plus our extra controls)
  Table VI   training stability (mean ± std over seeds)
  Table VII  efficiency (params, FPS)
  Fig. 3/4   K, gamma and insertion-depth sweeps
  Full data  4-class MRI with all training images (not in the paper; the practical MRI result)
Only numbers present in result files are reported; missing cells are "-".
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["zeroshot", "linear_probe", "clip_adapter", "coop", "cocoop", "zeroshot+mqa", "coop+mqa", "maple", "maple+mqa",
         "promptsrc", "promptsrc+mqa", "mmrl", "mmrl+mqa", "mmrlpp", "mmrlpp+mqa"]
LABEL = {"zeroshot": "CLIP (zero-shot)", "zeroshot+mqa": "MQAdapter (stand-alone)", "linear_probe": "Linear probe",
         "clip_adapter": "CLIP-Adapter", "coop": "CoOp", "cocoop": "CoCoOp", "maple": "MaPLe", "promptsrc": "PromptSRC",
         "mmrl": "MMRL", "mmrlpp": "MMRL++"}


def label(m: str) -> str:
    if m in LABEL:
        return LABEL[m]
    base = m.split("+")[0]
    return f"{LABEL.get(base, base)} + MQAdapter" if m.endswith("+mqa") else m


def load_results(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.glob("**/result.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def _ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v)) if len(v) > 1 else 0.0), len(v)


def f(x, p=2):
    return "-" if x is None else f"{x:.{p}f}"


def _sorted_methods(ms):
    return sorted(ms, key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))


def _md(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def group(results, task, variant=""):
    g = defaultdict(list)
    for r in results:
        if r["task"] == task and (r.get("variant") or "") == variant:
            g[(r["method"], r.get("shots"))].append(r)
    return g


def table_base2new(results):
    g = group(results, "base2new")
    rows, data = [], {}
    for m in _sorted_methods({k[0] for k in g}):
        rs = [r for (mm, _), v in g.items() if mm == m for r in v]
        b, sb, n = _ms([r["summary"]["base"] for r in rs])
        nn_, sn, _ = _ms([r["summary"]["new"] for r in rs])
        h, sh, _ = _ms([r["summary"]["H"] for r in rs])
        data[m] = (b, nn_, h)
        rows.append([label(m), f(b), f(nn_), f(h), n])
        host = m.split("+")[0]
        if m.endswith("+mqa") and host in data and host != "zeroshot":
            hb, hn, hh = data[host]
            d = lambda a, c: "-" if a is None or c is None else f"({a - c:+.2f})"  # noqa: E731
            rows.append(["Δ", d(b, hb), d(nn_, hn), d(h, hh), ""])
    return _md(["Method", "Base", "New", "H", "seeds"], rows), data


def table_fewshot(results):
    g = group(results, "fewshot")
    shots = sorted({k[1] for k in g if k[1]})
    rows, data = [], {}
    for m in _sorted_methods({k[0] for k in g}):
        accs = []
        for s in shots:
            mean, _, _ = _ms([r["summary"]["accuracy"] for r in g.get((m, s), [])])
            accs.append(mean)
        data[m] = accs
        rows.append([label(m)] + [f(a) for a in accs])
        host = m.split("+")[0]
        if m.endswith("+mqa") and host in data and host != "zeroshot":
            rows.append(["Δ"] + ["-" if a is None or b is None else f"({a - b:+.2f})" for a, b in zip(accs, data[host])])
    return _md(["Method"] + [f"{s}-shot" for s in shots], rows), data


def table_transfer(results, task):
    g = group(results, task)
    targets = []
    for v in g.values():
        for r in v:
            for t in r["summary"].get("targets", {}):
                if t not in targets:
                    targets.append(t)
    rows = []
    for m in _sorted_methods({k[0] for k in g}):
        rs = [r for (mm, _), v in g.items() if mm == m for r in v]
        src, _, n = _ms([r["summary"]["source"] for r in rs])
        avg, _, _ = _ms([r["summary"]["target_avg"] for r in rs])
        cells = [f(_ms([r["summary"]["targets"].get(t) for r in rs])[0]) for t in targets]
        rows.append([label(m), f(src), f(avg)] + cells)
    return _md(["Method", "Source", "Target avg"] + targets, rows)


def table_variants(results, task, method_prefix, title_map):
    """Baseline host, host+MQAdapter and the named ablation variants only (sweep variants are excluded)."""
    rows = []
    g = defaultdict(list)
    for r in results:
        v = r.get("variant") or ""
        if r["task"] == task and r["method"].split("+")[0] == method_prefix and (v == "" or v in title_map):
            g[(r["method"], v)].append(r)
    for (m, v), rs in sorted(g.items(), key=lambda kv: (kv[0][1] != "", kv[0][0].endswith("+mqa"), kv[0][1])):
        b, _, n = _ms([r["summary"]["base"] for r in rs])
        nn_, _, _ = _ms([r["summary"]["new"] for r in rs])
        h, _, _ = _ms([r["summary"]["H"] for r in rs])
        name = title_map[v] if v else ("Baseline+MQAdapter" if m.endswith("+mqa") else "Baseline")
        rows.append([name, f(b), f(nn_), f(h), n])
    return _md(["Method", "Base", "New", "H", "seeds"], rows)


def table_stability(results, methods=("mmrlpp", "mmrlpp+mqa")):
    rows = []
    for m in methods:
        rs = [r for r in results if r["task"] == "base2new" and r["method"] == m and not r.get("variant")]
        b, sb, n = _ms([r["summary"]["base"] for r in rs])
        nn_, sn, _ = _ms([r["summary"]["new"] for r in rs])
        rows.append([label(m) if not m.endswith("+mqa") else "+MQAdapter", f"{f(b)} ± {f(sb)}", f"{f(nn_)} ± {f(sn)}", n])
    return _md(["Method", "Base", "New", "seeds"], rows)


def table_sweep(results, key):
    g = defaultdict(list)
    for r in results:
        v = r.get("variant") or ""
        if r["task"] == "base2new" and v.startswith(key + "="):
            g[v].append(r)
    pts = []
    for v, rs in g.items():
        x = float(v.split("=", 1)[1])
        pts.append((x, _ms([r["summary"]["base"] for r in rs])[0], _ms([r["summary"]["new"] for r in rs])[0], _ms([r["summary"]["H"] for r in rs])[0]))
    pts.sort()
    return _md([key, "Base", "New", "H"], [[f"{p[0]:g}", f(p[1]), f(p[2]), f(p[3])] for p in pts]), pts


def plot_sweeps(sweeps: dict, out: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    items = [(k, v) for k, v in sweeps.items() if v]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.3), squeeze=False)
    for ax, (k, pts) in zip(axes[0], items):
        xs = list(range(len(pts)))
        for j, (lab, mk) in enumerate((("Base", "+"), ("New", "o"), ("H", "o"))):
            ax.plot(xs, [p[j + 1] for p in pts], marker=mk, label=lab)
        ax.set_xticks(xs, [f"{p[0]:g}" for p in pts])
        ax.set_title(k)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def build_report(root: Path) -> Path:
    results = load_results(root)
    parts = [f"# Results ({len(results)} runs under `{root}`)\n",
             "All accuracies in %. Means over the seeds that exist on disk. '-' = not run.\n"]
    t1, _ = table_base2new(results)
    parts += ["## Table I - base-to-new generalisation\n", t1, ""]
    t2, _ = table_fewshot(results)
    parts += ["## Table II - few-shot classification\n", t2, ""]
    parts += ["## Table III - cross-dataset transfer\n", table_transfer(results, "cross"), ""]
    parts += ["## Table IV - domain generalisation\n", table_transfer(results, "dg"), ""]
    parts += ["## Table V - ablation (MMRL++ host)\n",
              table_variants(results, "base2new", "mmrlpp", {"cross_attention": "Baseline+cross-attention", "separate_vqc": "Baseline+separate VQC",
                                                             "mlp": "Baseline+MLP instead of VQC (extra)", "identity": "Baseline+no VQC (extra)",
                                                             "frozen_vqc": "Baseline+frozen random VQC (extra)"}), ""]
    parts += ["## Table VI - training stability\n", table_stability(results), ""]
    sweeps = {}
    for key, title in (("k", "Fig. 3(a) Top-K"), ("gamma", "Fig. 3(b) gamma"), ("start", "Fig. 4 insertion start block (0-indexed)")):
        t, pts = table_sweep(results, key)
        sweeps[title] = pts
        parts += [f"## {title}\n", t, ""]
    plot_sweeps(sweeps, root / "fig3_fig4_sweeps.png")
    eff = root / "efficiency.json"
    if eff.exists():
        rows = [[label(r["method"]), r["trainable_params"], r["mqadapter_params"], r["vqc_params"], f(r["fps"], 1)] for r in json.loads(eff.read_text())]
        parts += ["## Table VII - efficiency\n", _md(["Method", "Trainable params", "MQAdapter params", "VQC params", "FPS (bs 100)"], rows), ""]
    ga = group(results, "all")
    if ga:
        rows = []
        for m in _sorted_methods({k[0] for k in ga}):
            rs = [r for (mm, _), v in ga.items() if mm == m for r in v]
            a, sa, n = _ms([r["summary"]["accuracy"] for r in rs])
            mf, _, _ = _ms([(r["summary"].get("macro_f1") or 0) * 100 for r in rs])
            rows.append([label(m), f"{f(a)} ± {f(sa)}", f(mf), n])
        parts += ["## Full-data 4-class MRI (not in the paper)\n", _md(["Method", "Accuracy", "Macro-F1", "seeds"], rows), ""]
    topk = root / "topk" / "topk_analysis.json"
    if topk.exists():
        t = json.loads(topk.read_text())
        parts += ["## Fig. 1 - frozen CLIP Top-K accuracy\n", _md(list(t["top_k_accuracy"].keys()) + ["mean sim"],
                  [[f(v) for v in t["top_k_accuracy"].values()] + [f(t["mean_similarity_topk"], 4)]]), ""]
    out = root / "REPORT.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    with open(root / "all_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "method", "variant", "shots", "seed", "eval_set", "accuracy", "macro_f1", "topk_recall", "trainable_params"])
        for r in results:
            for k, v in r["evals"].items():
                w.writerow([r["task"], r["method"], r.get("variant"), r.get("shots"), r["seed"], k,
                            None if v.get("accuracy") is None else round(v["accuracy"] * 100, 3),
                            None if v.get("macro_f1") is None else round(v["macro_f1"] * 100, 3),
                            v.get("coarse_topk_recall"), r.get("trainable_params")])
    return out
