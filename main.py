#!/usr/bin/env python
"""End-to-end reproduction driver: zero-shot VLM -> fine-tuned VLMs -> quantum (MQAdapter) fine-tuned VLMs,
over every protocol of the MQAdapter paper, then one report with the paper's tables.

Quick plumbing check (no downloads, synthetic images, fake CLIP, ~1-3 min on CPU):
    python main.py --make-dummy-data data/dummy --backend fake --quick

Everything on the brain-tumour MRI data (real CLIP ViT-B/16; many GPU hours - see docs/RUNNING.md):
    python main.py --data-root data/brain_tumor_mri

Pick tasks / methods / seeds:
    python main.py --tasks base2new fewshot report --methods maple maple+mqa mmrlpp mmrlpp+mqa --seeds 1
    python main.py --tasks all --methods zeroshot coop maple+mqa --seeds 1 2 3       # full-data MRI
    python main.py --dry-run                                                          # list the runs only

Methods  : zeroshot, linear_probe, clip_adapter, coop, cocoop, maple, promptsrc, mmrl, mmrlpp; any of them
           with "+mqa" adds the MQAdapter plug-in ("zeroshot+mqa" = stand-alone MQAdapter on frozen CLIP).
           Groups: baselines, hosts, qvlm, paper (default), everything.
Tasks    : topk (Fig.1)  all (full data, not in paper)  base2new (Tab.I)  fewshot (Tab.II)  cross (Tab.III)
           dg (Tab.IV)  ablation (Tab.V)  sweeps (Fig.3/4)  stability (Tab.VI)  efficiency (Tab.VII)
           tsne (Fig.5)  report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

GROUPS = {
    "baselines": ["zeroshot", "linear_probe", "clip_adapter", "coop", "cocoop"],
    "hosts": ["maple", "promptsrc", "mmrlpp"],
    "qvlm": ["zeroshot+mqa", "maple+mqa", "promptsrc+mqa", "mmrlpp+mqa"],
}
GROUPS["paper"] = ["zeroshot"] + GROUPS["hosts"] + GROUPS["qvlm"][1:]
GROUPS["everything"] = GROUPS["baselines"] + ["mmrl", "mmrl+mqa", "coop+mqa"] + GROUPS["hosts"] + GROUPS["qvlm"]
PAPER_TASKS = ["topk", "base2new", "fewshot", "cross", "dg", "ablation", "sweeps", "stability", "efficiency", "tsne", "report"]
ALL_TASKS = ["topk", "all"] + PAPER_TASKS[1:]
ABLATION_VARIANTS = {  # Table V (+ extra controls that isolate the circuit)
    "cross_attention": {"interaction": "cross_attention"},
    "separate_vqc": {"vqc_sharing": "separate"},
    "mlp": {"quantum_map": "mlp"},
    "identity": {"quantum_map": "identity"},
    "frozen_vqc": {"quantum_map": "frozen_vqc"},
}
SWEEP_K = [1, 2, 3, 4, 5, 6, 8, 10]
SWEEP_GAMMA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
SWEEP_START = [3, 4, 5, 6, 7, 8, 9, 10, 11]  # first adapted block, 0-indexed (default 6 -> blocks 6..11)


def parse_set(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def expand_methods(names):
    out = []
    for n in names:
        for m in GROUPS.get(n, [n]):
            if m not in out:
                out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["paper"], help="task names, 'paper' (default) or 'everything'")
    ap.add_argument("--methods", nargs="+", default=["paper"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--dataset", default="brain_tumor_mri", help="built-in spec name or a .json spec")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--cross-targets", nargs="*", default=["figshare_brain", "br35h"])
    ap.add_argument("--target-root", nargs="*", default=[], metavar="NAME=PATH", help="root for a cross-dataset target")
    ap.add_argument("--dg-targets", nargs="*", default=[], help="real target-domain specs with the same classes")
    ap.add_argument("--dg-shifts", nargs="*", default=["noise", "bias", "lowres", "sketch"])
    ap.add_argument("--k", type=int, default=None, help="Top-K anchors (paper 5; default 2 when the dataset has <=5 classes)")
    ap.add_argument("--backend", default="open_clip", choices=["open_clip", "clip", "fake"])
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading CLIP weights")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--augment", default=None, choices=[None, "mri", "clip"], help="default: mri for MRI specs, clip otherwise")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--full-epochs", type=int, default=5, help="epochs for the full-data 'all' task")
    ap.add_argument("--epochs-scale", type=float, default=None, help="multiply every method's epochs")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--selection", default="last", choices=["last", "best_val"])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any hyper-parameter for every run")
    ap.add_argument("--quick", action="store_true", help="1 epoch, 3 steps, 32 eval images: plumbing check only")
    ap.add_argument("--force", action="store_true", help="re-run experiments that already have result.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--make-dummy-data", default=None, metavar="DIR", help="create synthetic datasets in DIR and use them")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    tasks = ALL_TASKS if "everything" in args.tasks else (PAPER_TASKS if "paper" in args.tasks else args.tasks)
    methods = expand_methods(args.methods)
    target_roots = dict(t.split("=", 1) for t in args.target_root)
    data_root = args.data_root
    if args.make_dummy_data:
        from scripts.make_dummy_dataset import make_all

        roots = make_all(Path(args.make_dummy_data))
        data_root = data_root or str(roots["brain_tumor_mri"])
        target_roots = {**{k: str(v) for k, v in roots.items() if k != "brain_tumor_mri"}, **target_roots}

    from mqa.datasets import load_spec, resolve_classes

    spec = load_spec(args.dataset, data_root)
    is_mri = spec.name in ("brain_tumor_mri", "figshare_brain", "br35h") or "mri" in spec.template.lower()
    overrides = parse_set(args.set)
    try:
        n_cls = len(resolve_classes(spec))
    except SystemExit:
        n_cls = None
    if args.k is not None:
        overrides["k"] = args.k
    elif n_cls is not None and n_cls <= 5:
        overrides.setdefault("k", 2)
    cfg = dict(
        dataset=args.dataset, data_root=data_root, backend=args.backend, download_clip=not args.no_download, device=args.device,
        out_root=args.out, augment=args.augment or ("mri" if is_mri else "clip"), num_workers=args.num_workers,
        full_epochs=args.full_epochs, epochs_scale=args.epochs_scale, mixed_precision=args.mixed_precision or None,
        selection=args.selection, overrides=overrides, quick=args.quick, force=args.force,
        cross_targets=args.cross_targets, target_roots=target_roots, dg_targets=args.dg_targets, dg_shifts=args.dg_shifts,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "main_config.json").write_text(json.dumps({**cfg, "tasks": tasks, "methods": methods, "seeds": args.seeds,
                                                       "shots": args.shots, "argv": sys.argv}, indent=2))

    plan = build_plan(tasks, methods, args, n_cls)
    print(f"dataset {spec.name} ({n_cls} classes) at {spec.root}; {len(plan)} training/eval runs planned; out -> {out}")
    if args.dry_run:
        for p in plan:
            print("  ", p)
        return

    from mqa.engine import Engine

    eng = Engine(cfg)
    failures = []
    t0 = time.perf_counter()
    for task in tasks:
        if task == "topk":
            run_topk(eng, out)
        elif task == "efficiency":
            run_efficiency(eng, methods, out)
        elif task == "tsne":
            run_tsne(out, args.seeds[0])
        elif task == "report":
            from mqa.engine.report import build_report

            print(f"\nreport -> {build_report(out)}")
        else:
            for p in [p for p in plan if p["task"] == task]:
                try:
                    eng.run(p["run_task"], p["method"], p["seed"], p.get("shots"), p.get("overrides"), p.get("variant", ""))
                except Exception as exc:  # noqa: BLE001
                    failures.append({**p, "error": repr(exc)})
                    print(f"!! failed {p}: {exc!r}", flush=True)
                    if not args.continue_on_error:
                        raise
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2))
        print(f"{len(failures)} runs failed; see {out / 'failures.json'}")
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


def build_plan(tasks, methods, args, n_cls):
    plan = []
    for task in tasks:
        if task in ("all", "base2new", "cross", "dg"):
            for m in methods:
                for s in args.seeds:
                    plan.append({"task": task, "run_task": task, "method": m, "seed": s})
        elif task == "fewshot":
            for m in methods:
                for sh in args.shots:
                    for s in args.seeds:
                        plan.append({"task": task, "run_task": "fewshot", "method": m, "seed": s, "shots": sh})
        elif task == "ablation":
            for s in args.seeds:
                plan.append({"task": task, "run_task": "base2new", "method": "mmrlpp", "seed": s})
                plan.append({"task": task, "run_task": "base2new", "method": "mmrlpp+mqa", "seed": s})
                for v, ov in ABLATION_VARIANTS.items():
                    plan.append({"task": task, "run_task": "base2new", "method": "mmrlpp+mqa", "seed": s, "variant": v, "overrides": ov})
        elif task == "sweeps":
            s = args.seeds[0]
            base_classes = (n_cls + 1) // 2 if n_cls else 0
            for kk in SWEEP_K:
                if base_classes and kk > base_classes:
                    continue  # with few classes, K > #base classes is the same run as K = #base classes
                plan.append({"task": task, "run_task": "base2new", "method": "mmrlpp+mqa", "seed": s, "variant": f"k={kk}", "overrides": {"k": kk}})
            for g in SWEEP_GAMMA:
                plan.append({"task": task, "run_task": "base2new", "method": "mmrlpp+mqa", "seed": s, "variant": f"gamma={g}", "overrides": {"gamma": g}})
            for st in SWEEP_START:
                plan.append({"task": task, "run_task": "base2new", "method": "mmrlpp+mqa", "seed": s, "variant": f"start={st}",
                             "overrides": {"adapt_layers": list(range(st, 12))}})
        elif task == "stability":
            for m in ("mmrlpp", "mmrlpp+mqa"):
                for s in sorted(set(args.seeds) | {1, 2, 3}):
                    plan.append({"task": task, "run_task": "base2new", "method": m, "seed": s})
    return plan


def run_topk(eng, out):
    from mqa.datasets import ClassSpace, resolve_classes, scan_split
    from mqa.engine.analysis import topk_analysis
    from mqa.hosts.factory import build_host, method_config

    clip, tok = eng.backbone()
    classes = resolve_classes(eng.spec)
    recs = scan_split(eng.spec, eng.spec.test_split, classes)
    if eng.cfg.get("quick"):
        recs = recs[:64]
    host = build_host("zeroshot", clip, tok, method_config("zeroshot"))
    res = topk_analysis(host, recs, ClassSpace(eng.spec, classes), eng.device, out / "topk", eng.cfg)
    print("Fig.1 Top-K accuracy:", {k: round(v, 2) for k, v in res["top_k_accuracy"].items()})


def run_efficiency(eng, methods, out):
    from mqa.datasets import ClassSpace, resolve_classes
    from mqa.engine.analysis import efficiency
    from mqa.hosts.factory import build_host

    clip, tok = eng.backbone()
    space = ClassSpace(eng.spec, resolve_classes(eng.spec))
    rows = []
    for m in methods:
        if m.startswith("linear_probe"):
            continue
        cfg = eng.run_config("base2new", m, None, None)
        host = build_host(m, clip, tok, cfg)
        r = efficiency(host, space, eng.device, batch_size=8 if eng.cfg.get("quick") else 100, iters=1 if eng.cfg.get("quick") else 5)
        rows.append({"method": m, **r})
        print(f"[efficiency] {m}: {r}")
        del host
    (out / "efficiency.json").write_text(json.dumps(rows, indent=2))


def run_tsne(out, seed):
    from mqa.engine.analysis import tsne_figure

    files = {}
    for m in ("zeroshot", "mmrlpp", "mmrlpp+mqa"):
        d = out / "base2new" / m / "shotsall" / f"seed{seed}" / "eval"
        files[m] = {p: d / p / "features.npz" for p in ("base", "new")}
    done = tsne_figure(files, out / "fig5_tsne.png", seed)
    print(f"Fig.5 t-SNE panels: {len(done)} -> {out / 'fig5_tsne.png'}")


if __name__ == "__main__":
    main()
