#!/usr/bin/env python3
"""
Plot the cu_roaring vs cuVS bitset end-to-end selectivity sweep.

Reads JSON files produced by cuvs/cpp/bench/prims/core/bench_e2e_sweep, merges
them by selectivity (later files override earlier ones), and produces:
  - figures/e2e_sweep_speedup.png  : selectivity vs speedup (the headline)
  - figures/e2e_sweep_ms.png       : selectivity vs ms for both paths

Usage:
  python3 scripts/plot_e2e_sweep.py results/raw/2026-05-23/bench_e2e_sweep_low.json \
                                    results/raw/2026-05-23/bench_e2e_sweep_high.json \
                                    results/raw/2026-05-23/bench_e2e_sweep_50.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def merge(paths):
    """Load and merge results across files, keyed by selectivity. Header info
    (gpu, n, dim, nq, k) must agree across files; later files win on conflict."""
    rows = {}
    header = None
    for p in paths:
        with open(p) as f:
            j = json.load(f)
        h = {k: j[k] for k in ("gpu", "n", "dim", "nq", "k")}
        if header is None:
            header = h
        elif header != h:
            print(f"WARN: header mismatch between files: {header} vs {h}",
                  file=sys.stderr)
        for r in j["results"]:
            rows[float(r["sel"])] = r
    return header, [rows[s] for s in sorted(rows)]


def plot_speedup(header, rows, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    sels   = [r["sel"] * 100 for r in rows]  # %, log scale
    fair   = [r.get("speedup_fair", r["speedup"]) for r in rows]
    raw    = [r["speedup"] for r in rows]
    have_fair = any("speedup_fair" in r for r in rows)

    # Highlight the peak of the fair curve
    peak_i = max(range(len(fair)), key=lambda i: fair[i])

    if have_fair:
        ax.plot(sels, raw, marker="s", markersize=7, linewidth=1.4,
                color="#a0a0a0", linestyle=":",
                label="raw (cuVS includes count kernel)")
    line, = ax.plot(sels, fair, marker="o", linewidth=2.3, markersize=9,
                    color="#1f77b4",
                    label=("fair (cuVS minus raft::popc count)"
                           if have_fair else "cu_roaring / cuVS bitset"))
    for s, sp in zip(sels, fair):
        ax.annotate(f"{sp:.2f}x", (s, sp),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color="#1f77b4")

    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate("parity (1.0x)", (sels[0], 1.0),
                textcoords="offset points", xytext=(5, -15),
                color="grey", fontsize=9)

    ax.scatter([sels[peak_i]], [fair[peak_i]], s=160, marker="o",
               facecolors="none", edgecolors="#d62728", linewidths=2.5,
               zorder=4, label=f"peak {fair[peak_i]:.2f}x at {sels[peak_i]:.1f}%")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Filter selectivity (% of vectors passing)")
    ax.set_ylabel("Speedup vs cuVS bitset (higher is better)")
    ax.set_title(
        f"cu_roaring schedule-driven vs cuVS bitset filter — end-to-end\n"
        f"{header['gpu']}, N={header['n']:,}, D={header['dim']}, "
        f"Q={header['nq']}, k={header['k']}   recall@k = 1.000 everywhere"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xticks(sels)
    ax.set_xticklabels([f"{s:g}" for s in sels])
    ax.set_xlim(min(sels) * 0.7, max(sels) * 1.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def plot_ms(header, rows, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sels   = [r["sel"] * 100 for r in rows]
    cuvs   = [r["cuvs_ms"]    for r in rows]
    roar   = [r["roaring_ms"] for r in rows]
    cuvs_p10  = [r["cuvs_p10"]    for r in rows]
    cuvs_p90  = [r["cuvs_p90"]    for r in rows]
    roar_p10  = [r["roaring_p10"] for r in rows]
    roar_p90  = [r["roaring_p90"] for r in rows]

    ax.plot(sels, cuvs, marker="s", color="#ff7f0e", linewidth=2,
            label="cuVS brute_force + bitset_filter")
    ax.fill_between(sels, cuvs_p10, cuvs_p90, color="#ff7f0e", alpha=0.15)

    ax.plot(sels, roar, marker="o", color="#1f77b4", linewidth=2,
            label="cu_roaring schedule-driven")
    ax.fill_between(sels, roar_p10, roar_p90, color="#1f77b4", alpha=0.15)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Filter selectivity (% of vectors passing)")
    ax.set_ylabel("End-to-end latency per query batch (ms)")
    ax.set_title(
        f"End-to-end latency: H2D filter + setup + search + D2H top-k\n"
        f"{header['gpu']}, N={header['n']:,}, D={header['dim']}, "
        f"Q={header['nq']}, k={header['k']}   (shaded = p10..p90)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xticks(sels)
    ax.set_xticklabels([f"{s:g}" for s in sels])
    ax.set_xlim(min(sels) * 0.7, max(sels) * 1.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    header, rows = merge(paths)
    print(f"merged {sum(1 for _ in rows)} cells from {len(paths)} files")
    print(f"  gpu={header['gpu']}  N={header['n']}  D={header['dim']}  "
          f"Q={header['nq']}  k={header['k']}")
    print()
    have_fair = any("speedup_fair" in r for r in rows)
    if have_fair:
        print(f"{'sel %':>8}  {'card':>10}  {'cuvs ms':>10}  {'count ms':>9}  "
              f"{'cuvs-c ms':>10}  {'roar ms':>10}  {'raw':>7}  {'fair':>7}  "
              f"{'recall':>7}  schedule")
    else:
        print(f"{'sel %':>8}  {'card':>10}  {'cuvs ms':>10}  {'roar ms':>10}  "
              f"{'speedup':>9}  {'recall':>7}  schedule")
    for r in rows:
        sched = r["schedule"]
        sd = (f"D={sched['direct_tasks']}({sched['direct_cols']/1e6:.1f}M) "
              f"M={sched['masked_tasks']}({sched['masked_cols']/1e6:.1f}M) "
              f"G={sched['gather_tasks']}({sched['gather_cols']/1e6:.1f}M)")
        if have_fair:
            print(f"{r['sel']*100:>8.4g}  {r['card']:>10}  "
                  f"{r['cuvs_ms']:>10.3f}  {r.get('count_ms',0):>9.3f}  "
                  f"{r.get('cuvs_ms_no_count', r['cuvs_ms']):>10.3f}  "
                  f"{r['roaring_ms']:>10.3f}  {r['speedup']:>6.2f}x  "
                  f"{r.get('speedup_fair', r['speedup']):>6.2f}x  "
                  f"{r['recall']:>7.4f}  {sd}")
        else:
            print(f"{r['sel']*100:>8.4g}  {r['card']:>10}  {r['cuvs_ms']:>10.3f}  "
                  f"{r['roaring_ms']:>10.3f}  {r['speedup']:>8.2f}x  "
                  f"{r['recall']:>7.4f}  {sd}")
    plot_speedup(header, rows, FIG_DIR / "e2e_sweep_speedup.png")
    plot_ms(header, rows, FIG_DIR / "e2e_sweep_ms.png")


if __name__ == "__main__":
    main()
