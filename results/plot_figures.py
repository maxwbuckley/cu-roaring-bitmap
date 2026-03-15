#!/usr/bin/env python3
"""Generate publication-quality figures from benchmark JSON results."""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
CPU_LIGHT = '#BBBBBB'
ACCENT_BLUE = '#1F77B4'
ACCENT_ORANGE = '#FF7F0E'
ACCENT_YELLOW = '#FFBB33'

RAW = 'raw'
FIG = 'figures'
os.makedirs(FIG, exist_ok=True)


def load(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        # Try from build directory
        path = os.path.join('..', 'build', 'results', 'raw', name)
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Figure 1: Filter Construction Latency vs Predicate Count
# ============================================================================
def fig1():
    data = load('bench1_filter_construction.json')
    results = data['results']

    universes = sorted(set(r['universe'] for r in results))
    fig, axes = plt.subplots(1, len(universes), figsize=(4 * len(universes), 3.5), sharey=True)
    if len(universes) == 1:
        axes = [axes]

    for ax, U in zip(axes, universes):
        subset = [r for r in results if r['universe'] == U]
        preds = [r['n_preds'] for r in subset]
        cpu = [r['cpu_ms']['median'] for r in subset]
        gpu_full = [r['gpu_full_ms']['median'] for r in subset]
        gpu_ops = [r['gpu_ops_ms']['median'] for r in subset]

        ax.semilogy(preds, cpu, 'o--', color=CPU_GREY, label='CPU CRoaring', linewidth=1.5, markersize=5)
        ax.semilogy(preds, gpu_full, 's-', color=NVIDIA_GREEN, label='GPU Roaring + decompress', linewidth=2, markersize=5)

        ax.set_xlabel('Number of predicates')
        ax.set_title(f'{U // 1_000_000}M vectors')
        ax.set_xticks(preds)
        ax.grid(axis='y', alpha=0.3)

        # Annotate speedup at 4 predicates
        for r in subset:
            if r['n_preds'] == 4:
                spd = r['speedup']
                ax.annotate(f'{spd:.0f}x', xy=(4, r['gpu_full_ms']['median']),
                            xytext=(4.5, r['gpu_full_ms']['median'] * 0.5),
                            fontsize=9, color=NVIDIA_GREEN, fontweight='bold')

    axes[0].set_ylabel('Filter construction time (ms)')
    axes[0].legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_filter_latency_vs_predicates.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig1_filter_latency_vs_predicates.png'), bbox_inches='tight')
    print('  fig1 saved')


# ============================================================================
# Figure 3: GPU Memory Footprint
# ============================================================================
def fig3():
    data = load('bench3_memory_footprint.json')
    results = data['results']

    tags = [r['n_tags'] for r in results]
    flat_gb = [r['flat_bytes'] / 1e9 for r in results]
    roar_gb = [r['roaring_bytes'] / 1e9 for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(tags))
    w = 0.35
    ax.bar(x - w/2, flat_gb, w, color=CPU_LIGHT, edgecolor=CPU_GREY, label='Flat bitset')
    ax.bar(x + w/2, roar_gb, w, color=NVIDIA_GREEN, edgecolor='#5a8f00', label='GPU Roaring')

    ax.axhline(y=32, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(len(tags) - 0.5, 33, 'RTX 5090 (32 GB)', fontsize=8, color='red', ha='right')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(len(tags) - 0.5, 82, 'A100 (80 GB)', fontsize=8, color='orange', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tags])
    ax.set_xlabel('Number of attribute bitmaps')
    ax.set_ylabel('Total GPU memory (GB)')
    ax.set_title('GPU Memory for Attribute Bitmaps at 1B Scale')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_memory_footprint.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig3_memory_footprint.png'), bbox_inches='tight')
    print('  fig3 saved')


# ============================================================================
# Figure 4: End-to-End Latency Waterfall
# ============================================================================
def fig4():
    data = load('bench4_e2e_latency.json')
    results = data['results']

    # Use 1B result (last one)
    r = results[-1]
    U = r['universe']

    cpu_filter = r['cpu_filter_ms']['median']
    pcie = r['pcie_transfer_ms']['median']
    gpu_kern = r['gpu_kernel_ms']['median']
    gpu_dec = r['gpu_decompress_ms']['median']
    search = r['sim_search_ms']

    labels = ['Pipeline A\n(CPU filter)', 'Pipeline C\n(GPU pre-resident)']
    segments_a = [cpu_filter, pcie, search]
    segments_c = [gpu_kern, gpu_dec, search]
    colors = [ACCENT_BLUE, ACCENT_ORANGE, NVIDIA_GREEN]
    seg_labels = ['Filter construction', 'Transfer / Decompress', 'Vector search (sim.)']

    fig, ax = plt.subplots(figsize=(7, 3))

    # Pipeline A
    left = 0
    for val, col in zip(segments_a, colors):
        ax.barh(1, val, left=left, color=col, edgecolor='white', height=0.5)
        if val > 3:
            ax.text(left + val/2, 1, f'{val:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        left += val
    ax.text(left + 2, 1, f'{sum(segments_a):.1f} ms', va='center', fontsize=9, fontweight='bold')

    # Pipeline C
    left = 0
    for val, col in zip(segments_c, colors):
        ax.barh(0, val, left=left, color=col, edgecolor='white', height=0.5)
        if val > 1:
            ax.text(left + val/2, 0, f'{val:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        left += val
    ax.text(left + 2, 0, f'{sum(segments_c):.1f} ms', va='center', fontsize=9, fontweight='bold')

    speedup = sum(segments_a) / sum(segments_c)
    ax.text(sum(segments_c) + 8, 0, f'({speedup:.1f}x faster)', va='center', fontsize=9,
            color=NVIDIA_GREEN, fontweight='bold')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel('Latency (ms)')
    ax.set_title(f'End-to-End Query Latency — 4 Predicates, {U // 1_000_000_000}B Vectors')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, seg_labels)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_e2e_waterfall.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig4_e2e_waterfall.png'), bbox_inches='tight')
    print('  fig4 saved')


# ============================================================================
# Figure 5: Kernel Effective Bandwidth
# ============================================================================
def fig5():
    data = load('bench5_kernel_micro.json')
    results = data['results']

    # Group by universe, show decompress bandwidth
    fig, ax = plt.subplots(figsize=(6, 4))

    universes = sorted(set(r['universe'] for r in results))
    densities = sorted(set(r['density'] for r in results))

    x = np.arange(len(universes))
    w = 0.25
    for i, d in enumerate(densities):
        bws = []
        for U in universes:
            match = [r for r in results if r['universe'] == U and r['density'] == d]
            bws.append(match[0]['decomp_bw_gbs'] if match else 0)
        ax.bar(x + (i - 1) * w, bws, w, label=f'{d*100:.0f}% density',
               color=[CPU_LIGHT, GPU_LIGHT, NVIDIA_GREEN][i], edgecolor='grey')

    ax.axhline(y=1800, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(len(universes) - 0.5, 1850, 'Theoretical peak (1.8 TB/s)', fontsize=8, color='red', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{U // 1_000_000}M' for U in universes])
    ax.set_xlabel('Universe size')
    ax.set_ylabel('Effective bandwidth (GB/s)')
    ax.set_title('Decompression Kernel Bandwidth')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig5_kernel_bandwidth.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig5_kernel_bandwidth.png'), bbox_inches='tight')
    print('  fig5 saved')


# ============================================================================
if __name__ == '__main__':
    print('Generating figures...')
    try:
        fig1()
    except Exception as e:
        print(f'  fig1 FAILED: {e}')
    try:
        fig3()
    except Exception as e:
        print(f'  fig3 FAILED: {e}')
    try:
        fig4()
    except Exception as e:
        print(f'  fig4 FAILED: {e}')
    try:
        fig5()
    except Exception as e:
        print(f'  fig5 FAILED: {e}')
    print('Done.')
