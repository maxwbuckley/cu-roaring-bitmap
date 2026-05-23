#!/usr/bin/env python3
"""
Multi-config comparison plot for cu_roaring vs cuVS bitset, overlaying
several (dtype, D) configs on a single set of axes.

Usage:
  python3 scripts/plot_e2e_multi.py \
      "fp32 D=512:results/raw/2026-05-23/bench_e2e_sweep_v3_*.json" \
      "fp16 D=512:results/raw/2026-05-23/bench_e2e_sweep_fp16_d512_*.json" \
      "fp16 D=1024:results/raw/2026-05-23/bench_e2e_sweep_fp16_d1024_*.json"

Each argument is "label:glob"; the glob expands to one or more JSONs that get
merged (later wins on conflict, keyed by selectivity).
"""
import json
import sys
from pathlib import Path
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_group(label_and_glob: str):
    if ":" not in label_and_glob:
        raise SystemExit(f"expected 'label:glob[,glob,...]', got {label_and_glob!r}")
    label, pats = label_and_glob.split(":", 1)
    paths = []
    for pat in pats.split(","):
        pat = pat.strip()
        if not pat:
            continue
        matched = sorted(glob(pat))
        if not matched:
            raise SystemExit(f"no files match {pat!r}")
        paths.extend(matched)
    rows = {}
    header = None
    for p in paths:
        with open(p) as f:
            j = json.load(f)
        if header is None:
            header = {k: j[k] for k in ("gpu", "n", "dim", "nq", "k") if k in j}
        for r in j["results"]:
            rows[float(r["sel"])] = r
    return label.strip(), header, [rows[s] for s in sorted(rows)]


def plot_speedup(groups, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors  = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    markers = ["o", "s", "^", "D", "v"]
    for i, (label, header, rows) in enumerate(groups):
        sels = [r["sel"] * 100 for r in rows]
        fair = [r.get("speedup_fair", r["speedup"]) for r in rows]
        c, m = colors[i % len(colors)], markers[i % len(markers)]
        ax.plot(sels, fair, marker=m, color=c, linewidth=2.2, markersize=8,
                label=label)
        for s, sp in zip(sels, fair):
            ax.annotate(f"{sp:.1f}x", (s, sp),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=c)
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate("parity (1.0x)", (groups[0][2][0]["sel"] * 100, 1.0),
                textcoords="offset points", xytext=(5, -15),
                color="grey", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Filter selectivity (% of vectors passing)")
    ax.set_ylabel("Speedup vs cuVS bitset (fair: cuVS minus raft popc)")
    g0_header = groups[0][1]
    ax.set_title(
        f"cu_roaring vs cuVS bitset — end-to-end speedup, multi-config\n"
        f"{g0_header['gpu']}, N={g0_header['n']:,}, Q={g0_header['nq']}, "
        f"k={g0_header['k']}   recall@k = 1.000 everywhere"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    all_sels = sorted({s for _, _, rs in groups for s in (r["sel"] * 100 for r in rs)})
    ax.set_xticks(all_sels)
    ax.set_xticklabels([f"{s:g}" for s in all_sels])
    ax.set_xlim(min(all_sels) * 0.7, max(all_sels) * 1.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_ms(groups, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (label, _, rows) in enumerate(groups):
        sels = [r["sel"] * 100 for r in rows]
        cuvs = [r["cuvs_ms"]    for r in rows]
        roar = [r["roaring_ms"] for r in rows]
        c = colors[i % len(colors)]
        ax.plot(sels, cuvs, marker="s", linestyle="--", color=c, linewidth=1.4,
                alpha=0.7, label=f"cuVS ({label})")
        ax.plot(sels, roar, marker="o", linestyle="-", color=c, linewidth=2.2,
                label=f"roaring ({label})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Filter selectivity (% of vectors passing)")
    ax.set_ylabel("End-to-end latency per query batch (ms)")
    g0_header = groups[0][1]
    ax.set_title(
        f"End-to-end latency, multi-config\n"
        f"{g0_header['gpu']}, N={g0_header['n']:,}, Q={g0_header['nq']}, "
        f"k={g0_header['k']}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    all_sels = sorted({s for _, _, rs in groups for s in (r["sel"] * 100 for r in rs)})
    ax.set_xticks(all_sels)
    ax.set_xticklabels([f"{s:g}" for s in all_sels])
    ax.set_xlim(min(all_sels) * 0.7, max(all_sels) * 1.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    groups = [load_group(arg) for arg in sys.argv[1:]]
    for label, header, rows in groups:
        print(f"\n=== {label}  ({header['gpu']}, N={header['n']}, D={header['dim']}) ===")
        print(f"{'sel %':>7}  {'cuvs ms':>9}  {'roar ms':>9}  {'fair':>7}")
        for r in rows:
            sp = r.get("speedup_fair", r["speedup"])
            print(f"{r['sel']*100:>7.4g}  {r['cuvs_ms']:>9.3f}  "
                  f"{r['roaring_ms']:>9.3f}  {sp:>6.2f}x")
    plot_speedup(groups, FIG_DIR / "e2e_sweep_speedup_multi.png")
    plot_ms(groups,      FIG_DIR / "e2e_sweep_ms_multi.png")


if __name__ == "__main__":
    main()
