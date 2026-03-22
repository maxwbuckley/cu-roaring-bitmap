#!/usr/bin/env python3
"""
Analyze B6 point query benchmark results.

Produces:
  fig6a: Heatmap — Roaring vs bitset slowdown across universe x density
  fig6b: Access pattern comparison — when does warp_contains help?
  fig6c: Memory-performance tradeoff scatter
  fig6d: Scaling — throughput vs universe size for each method
  summary table printed to stdout
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

NVIDIA_GREEN = '#76B900'
GPU_LIGHT = '#A3D65C'
CPU_GREY = '#888888'
ACCENT_BLUE = '#1F77B4'
ACCENT_ORANGE = '#FF7F0E'
ACCENT_RED = '#D62728'
ACCENT_PURPLE = '#9467BD'

RAW = os.path.join(os.path.dirname(__file__), 'raw')
FIG = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG, exist_ok=True)


def load():
    path = os.path.join(RAW, 'bench6_point_query.json')
    with open(path) as f:
        return json.load(f)


def fmt_universe(u):
    if u >= 1_000_000_000:
        return f'{u // 1_000_000_000}B'
    return f'{u // 1_000_000}M'


# ============================================================================
# Figure 6a: Heatmap — best Roaring method vs bitset (random access)
# ============================================================================
def fig6a(data):
    results = data['results']

    # Filter to random pattern only
    random_results = [r for r in results if r['pattern'] == 'random']

    universes = sorted(set(r['universe'] for r in random_results))
    densities = sorted(set(r['density'] for r in random_results))

    # Build matrix: ratio = roaring_best / bitset (< 1 means roaring wins)
    matrix = np.zeros((len(densities), len(universes)))
    method_matrix = [['' for _ in universes] for _ in densities]

    for r in random_results:
        ui = universes.index(r['universe'])
        di = densities.index(r['density'])

        # Best roaring method (lowest latency)
        best_ratio = min(r['contains_vs_bitset'], r['warp_vs_bitset'],
                         r['warp_bloom_vs_bitset'])
        matrix[di][ui] = best_ratio

        if best_ratio == r['contains_vs_bitset']:
            method_matrix[di][ui] = 'C'
        elif best_ratio == r['warp_vs_bitset']:
            method_matrix[di][ui] = 'W'
        else:
            method_matrix[di][ui] = 'WB'

    fig, ax = plt.subplots(figsize=(7, 4))

    # Diverging colormap: green = roaring faster, red = roaring slower
    cmap = matplotlib.colormaps.get_cmap('RdYlGn_r')
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=3.0)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    # Annotate cells
    for di in range(len(densities)):
        for ui in range(len(universes)):
            val = matrix[di][ui]
            method = method_matrix[di][ui]
            color = 'white' if val > 1.5 else 'black'
            if val < 1.0:
                label = f'{1/val:.2f}x\nfaster'
            else:
                label = f'{val:.2f}x\nslower'
            ax.text(ui, di, f'{label}\n({method})', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(range(len(universes)))
    ax.set_xticklabels([fmt_universe(u) for u in universes])
    ax.set_yticks(range(len(densities)))
    ax.set_yticklabels([f'{d*100:.1f}%' for d in densities])
    ax.set_xlabel('Universe size')
    ax.set_ylabel('Set density')
    ax.set_title('Best Roaring Method vs Flat Bitset (Random Queries, 10M queries)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Latency ratio (Roaring / Bitset)\n< 1 = Roaring wins')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig6a_roaring_vs_bitset_heatmap.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig6a_roaring_vs_bitset_heatmap.svg'), bbox_inches='tight')
    print('  fig6a saved')


# ============================================================================
# Figure 6b: Access pattern impact — warp_contains benefit by pattern
# ============================================================================
def fig6b(data):
    results = data['results']

    # Focus on 100M universe to show pattern effects clearly
    target_u = 100000000
    sub = [r for r in results if r['universe'] == target_u]

    densities = sorted(set(r['density'] for r in sub))
    patterns = ['random', 'clustered', 'strided']
    pattern_labels = ['Random', 'Clustered\n(32 same-container)', 'Strided\n(graph-like)']

    fig, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 4), sharey=True)
    if len(densities) == 1:
        axes = [axes]

    bar_width = 0.18
    methods = ['bitset_ms', 'contains_ms', 'warp_contains_ms', 'warp_bloom_ms']
    method_labels = ['Flat bitset', 'contains()', 'warp_contains()', 'warp+bloom']
    method_colors = [CPU_GREY, ACCENT_BLUE, NVIDIA_GREEN, ACCENT_ORANGE]

    for ax, d in zip(axes, densities):
        x = np.arange(len(patterns))
        for mi, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
            vals = []
            for p in patterns:
                match = [r for r in sub if r['density'] == d and r['pattern'] == p]
                vals.append(match[0][method]['median'] if match else 0)
            ax.bar(x + mi * bar_width, vals, bar_width, label=label, color=color,
                   edgecolor='white')

        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(pattern_labels, fontsize=7)
        ax.set_title(f'{d*100:.1f}% density')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Latency (ms) — 10M queries')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'Access Pattern Impact — {fmt_universe(target_u)} Universe', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig6b_access_pattern_impact.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig6b_access_pattern_impact.svg'), bbox_inches='tight')
    print('  fig6b saved')


# ============================================================================
# Figure 6c: Memory–performance tradeoff
# ============================================================================
def fig6c(data):
    results = data['results']

    # Random pattern, all universe/density combos
    random_results = [r for r in results if r['pattern'] == 'random']

    fig, ax = plt.subplots(figsize=(7, 5))

    for r in random_results:
        compression = r['compression_ratio']
        # Slowdown = best roaring / bitset latency ratio
        best_ratio = min(r['contains_vs_bitset'], r['warp_vs_bitset'])

        u = r['universe']
        marker = {1000000: 'o', 10000000: 's', 100000000: 'D', 1000000000: '^'}[u]
        color = {0.001: ACCENT_BLUE, 0.01: ACCENT_ORANGE,
                 0.10: NVIDIA_GREEN, 0.50: ACCENT_RED}[r['density']]

        ax.scatter(compression, best_ratio, s=80, marker=marker, color=color,
                   edgecolors='black', linewidths=0.5, zorder=3)

    # Reference line at ratio = 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(60, 0.95, 'Roaring faster', fontsize=9, color=NVIDIA_GREEN, ha='right')
    ax.text(60, 1.05, 'Bitset faster', fontsize=9, color=ACCENT_RED, ha='right')

    # Legend for universe (markers)
    from matplotlib.lines import Line2D
    u_handles = [
        Line2D([0], [0], marker='o', color='grey', linestyle='', markersize=7, label='1M'),
        Line2D([0], [0], marker='s', color='grey', linestyle='', markersize=7, label='10M'),
        Line2D([0], [0], marker='D', color='grey', linestyle='', markersize=7, label='100M'),
        Line2D([0], [0], marker='^', color='grey', linestyle='', markersize=7, label='1B'),
    ]
    d_handles = [
        Line2D([0], [0], marker='o', color=ACCENT_BLUE, linestyle='', markersize=7, label='0.1%'),
        Line2D([0], [0], marker='o', color=ACCENT_ORANGE, linestyle='', markersize=7, label='1%'),
        Line2D([0], [0], marker='o', color=NVIDIA_GREEN, linestyle='', markersize=7, label='10%'),
        Line2D([0], [0], marker='o', color=ACCENT_RED, linestyle='', markersize=7, label='50%'),
    ]
    l1 = ax.legend(handles=u_handles, title='Universe', loc='upper left', fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=d_handles, title='Density', loc='upper right', fontsize=8)

    ax.set_xlabel('Memory compression ratio (bitset / roaring)')
    ax.set_ylabel('Latency ratio (roaring / bitset)\n< 1.0 = roaring wins')
    ax.set_title('Memory Savings vs Query Performance Tradeoff (Random Queries)')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig6c_memory_performance_tradeoff.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig6c_memory_performance_tradeoff.svg'), bbox_inches='tight')
    print('  fig6c saved')


# ============================================================================
# Figure 6d: Throughput scaling by universe size (random pattern)
# ============================================================================
def fig6d(data):
    results = data['results']

    densities = sorted(set(r['density'] for r in results))
    universes = sorted(set(r['universe'] for r in results))

    fig, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 4), sharey=True)
    if len(densities) == 1:
        axes = [axes]

    methods = ['bitset_gqps', 'contains_gqps', 'warp_gqps', 'warp_bloom_gqps']
    method_labels = ['Flat bitset', 'contains()', 'warp_contains()', 'warp+bloom']
    method_colors = [CPU_GREY, ACCENT_BLUE, NVIDIA_GREEN, ACCENT_ORANGE]
    method_markers = ['o', 's', 'D', '^']

    for ax, d in zip(axes, densities):
        for method, label, color, marker in zip(methods, method_labels, method_colors, method_markers):
            vals = []
            for u in universes:
                match = [r for r in results
                         if r['universe'] == u and r['density'] == d and r['pattern'] == 'random']
                vals.append(match[0][method] if match else 0)
            ax.plot([fmt_universe(u) for u in universes], vals,
                    marker=marker, color=color, label=label, linewidth=1.5, markersize=5)

        ax.set_xlabel('Universe size')
        ax.set_title(f'{d*100:.1f}% density')
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')

    axes[0].set_ylabel('Throughput (Gq/s) — 10M random queries')
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Point Query Throughput Scaling', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig6d_throughput_scaling.png'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig6d_throughput_scaling.svg'), bbox_inches='tight')
    print('  fig6d saved')


# ============================================================================
# Summary table (printed to stdout)
# ============================================================================
def print_summary(data):
    results = data['results']

    print('\n' + '=' * 100)
    print('B6 POINT QUERY BENCHMARK SUMMARY')
    print(f'GPU: {data["gpu"]} ({data["n_sms"]} SMs)')
    print('=' * 100)

    # Table 1: Random access overview
    print('\nTable 1: Random Access — Latency Ratio (Roaring best / Bitset)')
    print(f'{"Universe":>10} {"Density":>10} {"Containers":>12} {"Container Mix":>20} '
          f'{"Compression":>12} {"Ratio":>8} {"Verdict":>10}')
    print('-' * 90)

    random_results = sorted(
        [r for r in results if r['pattern'] == 'random'],
        key=lambda r: (r['universe'], r['density']))

    for r in random_results:
        best = min(r['contains_vs_bitset'], r['warp_vs_bitset'])
        mix = f'bmp={r["n_bitmap"]} arr={r["n_array"]} run={r["n_run"]}'
        compression = f'{r["compression_ratio"]:.1f}x'
        verdict = 'ROARING' if best < 1.0 else ('TIE' if best < 1.1 else 'BITSET')
        print(f'{fmt_universe(r["universe"]):>10} {r["density_label"]:>10} '
              f'{r["n_containers"]:>12} {mix:>20} {compression:>12} '
              f'{best:>8.2f} {verdict:>10}')

    # Table 2: Warp benefit (warp / contains ratio)
    print('\nTable 2: Warp Cooperative Benefit (contains / warp_contains)')
    print(f'{"Universe":>10} {"Density":>10} {"Random":>10} {"Clustered":>10} {"Strided":>10}')
    print('-' * 55)

    for u in sorted(set(r['universe'] for r in results)):
        for d in sorted(set(r['density'] for r in results)):
            vals = {}
            for p in ['random', 'clustered', 'strided']:
                match = [r for r in results
                         if r['universe'] == u and r['density'] == d and r['pattern'] == p]
                if match:
                    vals[p] = match[0]['warp_vs_contains']
            if vals:
                dl = [r for r in results
                      if r['universe'] == u and r['density'] == d][0]['density_label']
                print(f'{fmt_universe(u):>10} {dl:>10} '
                      f'{vals.get("random", 0):>10.2f}x '
                      f'{vals.get("clustered", 0):>10.2f}x '
                      f'{vals.get("strided", 0):>10.2f}x')

    # Table 3: Bloom filter impact
    print('\nTable 3: Bloom Filter Impact (warp+bloom / warp — random access)')
    print(f'{"Universe":>10} {"Density":>10} {"warp ms":>10} {"bloom ms":>10} {"Impact":>10}')
    print('-' * 55)

    for r in random_results:
        warp_ms = r['warp_contains_ms']['median']
        bloom_ms = r['warp_bloom_ms']['median']
        impact = bloom_ms / warp_ms
        label = 'HELPS' if impact < 0.95 else ('HURTS' if impact > 1.05 else 'NEUTRAL')
        print(f'{fmt_universe(r["universe"]):>10} {r["density_label"]:>10} '
              f'{warp_ms:>10.3f} {bloom_ms:>10.3f} {impact:>7.2f}x {label:>7}')

    # Key findings
    print('\n' + '=' * 100)
    print('KEY FINDINGS')
    print('=' * 100)

    print("""
1. QUERY OVERHEAD: Roaring's per-query cost is dominated by the binary search
   over the container key array. At 1M universe (16 containers, 4 binary search
   steps), overhead is negligible. At 100M+ (1526+ containers, 11+ steps),
   the search becomes measurable.

2. CONTAINER TYPE MATTERS: At low density (0.1-1%), all containers are ARRAY
   type, and each array_contains() does a second binary search over up to 4096
   elements. This compounds with the key search — 100M/1% random shows 8-10x
   slowdown vs bitset. At 10%+ density, containers are BITMAP type and the
   membership test is a single 64-bit word read — nearly matching bitset speed.

3. WARP COOPERATIVE BENEFIT: warp_contains() rarely helps and sometimes hurts
   slightly. The __match_any_sync + __shfl_sync overhead (~2 instructions) isn't
   amortized well because:
   - Random queries: lanes rarely share the same high-16 key
   - Clustered queries: bitset's cache locality advantage is even larger
   - Strided queries: ~neutral
   The warp variant would shine in CAGRA's actual graph traversal where
   neighbors share locality, but the standalone benchmark can't replicate this.

4. BLOOM FILTER: Consistently adds 5-10% overhead with no benefit for point
   queries. The bloom check (2 hashes + 2 global reads) is redundant when the
   binary search will find or reject the key anyway. Bloom is useful for
   universe-level rejection (query ID in a range with no containers at all),
   which happens rarely with uniform random queries.

5. WHERE ROARING WINS: The sweet spot is large universe (100M+) with moderate
   density (10-50%) where containers are BITMAP type. Here roaring matches
   bitset query speed while using the same memory for the containers that exist,
   but paying zero for the ~90% of containers that don't exist in a 10% density
   bitmap.

6. MEMORY ADVANTAGE: At 0.1% density, roaring uses 59x less memory than a flat
   bitset. At 1B scale, that's 125MB vs 2.1MB. This is the real win for cuVS:
   when you need to store many concurrent filter bitmaps, roaring lets you fit
   far more of them in GPU memory.
""")


if __name__ == '__main__':
    print('Loading B6 point query results...')
    data = load()
    print(f'  {len(data["results"])} data points loaded')

    print('\nGenerating figures...')
    try:
        fig6a(data)
    except Exception as e:
        print(f'  fig6a FAILED: {e}')
    try:
        fig6b(data)
    except Exception as e:
        print(f'  fig6b FAILED: {e}')
    try:
        fig6c(data)
    except Exception as e:
        print(f'  fig6c FAILED: {e}')
    try:
        fig6d(data)
    except Exception as e:
        print(f'  fig6d FAILED: {e}')

    print_summary(data)
