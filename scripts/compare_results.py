#!/usr/bin/env python3
"""
compare_results.py -- Compare cu-roaring-filter against ACORN and VecFlow baselines.

Loads benchmark results from our system and the baselines, then produces:
  1. QPS vs Recall comparison tables (markdown, for the paper)
  2. Per-selectivity comparison (how each system handles different filter ratios)
  3. Memory usage comparison
  4. Build time comparison
  5. A combined JSON file for downstream plotting

Usage:
    python3 scripts/compare_results.py \
        --results-dir results/baselines \
        --our-results-dir results/raw \
        --output-dir results/comparison

Input files expected:
    results/baselines/acorn_results.json      (from run_baselines.py)
    results/baselines/vecflow_results.json    (from run_baselines.py)
    results/raw/bench6_point_query.json       (our benchmark results)
    results/raw/bench9_multi_and.json         (our benchmark results)
    results/system_info.json                  (GPU/system info)

Output:
    results/comparison/comparison_tables.md   (markdown tables for paper)
    results/comparison/comparison_data.json   (all data for plotting)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional


def load_json(path: str) -> Optional[dict]:
    """Load a JSON file, returning None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def format_qps(qps: float) -> str:
    """Format QPS for display: e.g. 1.2M, 450K, 3.5K."""
    if qps >= 1e6:
        return f"{qps/1e6:.1f}M"
    elif qps >= 1e3:
        return f"{qps/1e3:.0f}K"
    else:
        return f"{qps:.0f}"


def format_time(seconds: float) -> str:
    """Format time for display."""
    if seconds < 0.001:
        return f"{seconds*1e6:.0f}us"
    elif seconds < 1:
        return f"{seconds*1e3:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}min"


def format_memory(bytes_val: float) -> str:
    """Format memory for display."""
    if bytes_val >= 1e9:
        return f"{bytes_val/1e9:.1f}GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val/1e6:.1f}MB"
    elif bytes_val >= 1e3:
        return f"{bytes_val/1e3:.1f}KB"
    else:
        return f"{bytes_val:.0f}B"


def extract_pareto_front(curves: list, qps_key: str = "qps",
                          recall_key: str = "recall_at_k") -> list:
    """
    Extract Pareto-optimal points (max recall for a given QPS, or max QPS
    for a given recall). Returns sorted by recall ascending.
    """
    if not curves:
        return []
    # Sort by recall descending, then QPS descending
    sorted_curves = sorted(curves, key=lambda c: (-c[recall_key], -c[qps_key]))
    pareto = []
    max_qps = -1
    for c in sorted_curves:
        if c[qps_key] > max_qps:
            pareto.append(c)
            max_qps = c[qps_key]
    return sorted(pareto, key=lambda c: c[recall_key])


def find_qps_at_recall(curves: list, target_recall: float,
                        qps_key: str = "qps",
                        recall_key: str = "recall_at_k") -> Optional[float]:
    """
    Interpolate QPS at a target recall level from the Pareto curve.
    Returns None if target recall is outside the measured range.
    """
    pareto = extract_pareto_front(curves, qps_key, recall_key)
    if not pareto:
        return None
    # Find bracketing points
    for i in range(len(pareto) - 1):
        r_lo = pareto[i][recall_key]
        r_hi = pareto[i + 1][recall_key]
        if r_lo <= target_recall <= r_hi:
            # Linear interpolation
            frac = (target_recall - r_lo) / (r_hi - r_lo) if r_hi > r_lo else 0
            q_lo = pareto[i][qps_key]
            q_hi = pareto[i + 1][qps_key]
            # QPS typically decreases as recall increases, so interpolate in log space
            import math
            if q_lo > 0 and q_hi > 0:
                log_qps = math.log(q_lo) + frac * (math.log(q_hi) - math.log(q_lo))
                return math.exp(log_qps)
            return q_lo + frac * (q_hi - q_lo)
    # Exact match at boundary
    if pareto and abs(pareto[-1][recall_key] - target_recall) < 0.01:
        return pareto[-1][qps_key]
    if pareto and abs(pareto[0][recall_key] - target_recall) < 0.01:
        return pareto[0][qps_key]
    return None


def find_recall_at_qps(curves: list, target_qps: float,
                        qps_key: str = "qps",
                        recall_key: str = "recall_at_k") -> Optional[float]:
    """Interpolate recall at a target QPS level."""
    pareto = extract_pareto_front(curves, qps_key, recall_key)
    if not pareto:
        return None
    # Sort by QPS descending
    sorted_by_qps = sorted(pareto, key=lambda c: -c[qps_key])
    for i in range(len(sorted_by_qps) - 1):
        q_hi = sorted_by_qps[i][qps_key]
        q_lo = sorted_by_qps[i + 1][qps_key]
        if q_lo <= target_qps <= q_hi:
            frac = (target_qps - q_lo) / (q_hi - q_lo) if q_hi > q_lo else 0
            r_lo = sorted_by_qps[i + 1][recall_key]
            r_hi = sorted_by_qps[i][recall_key]
            return r_lo + frac * (r_hi - r_lo)
    return None


# =============================================================================
# Table Generators
# =============================================================================

def generate_qps_at_recall_table(
    systems: dict,
    target_recalls: list = [0.80, 0.90, 0.95, 0.99],
) -> str:
    """
    Generate a markdown table: QPS achieved at various recall levels.

    | System | Type | recall@0.80 | recall@0.90 | recall@0.95 | recall@0.99 |
    |--------|------|-------------|-------------|-------------|-------------|
    | Ours   | GPU  | 5.2M QPS    | 3.1M QPS    | 1.4M QPS    | 200K QPS    |
    | VecFlow| GPU  | ...         | ...         | ...         | ...         |
    | ACORN  | CPU  | ...         | ...         | ...         | ...         |
    """
    lines = []
    lines.append("### QPS at Target Recall (All Selectivities Combined)")
    lines.append("")

    # Header
    header = "| System | Type |"
    sep = "|--------|------|"
    for r in target_recalls:
        header += f" QPS@{r:.0%} |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for name, info in systems.items():
        row = f"| {name} | {info['type']} |"
        curves = info.get("curves", [])
        for r in target_recalls:
            qps = find_qps_at_recall(curves, r)
            if qps is not None:
                row += f" {format_qps(qps)} |"
            else:
                row += " -- |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_per_selectivity_table(
    systems: dict,
    target_recall: float = 0.90,
) -> str:
    """
    Generate a markdown table: QPS at target recall, broken down by selectivity.

    | Selectivity | ACORN QPS | VecFlow QPS | Ours QPS | Speedup vs ACORN | Speedup vs VecFlow |
    """
    lines = []
    lines.append(f"### QPS at {target_recall:.0%} Recall by Selectivity")
    lines.append("")

    # Collect all selectivity bins
    all_bins = set()
    for info in systems.values():
        for c in info.get("curves", []):
            if "selectivity_bin" in c:
                all_bins.add(c["selectivity_bin"])
    all_bins = sorted(all_bins)

    if not all_bins:
        lines.append("*No per-selectivity data available.*")
        lines.append("")
        return "\n".join(lines)

    # Header
    sys_names = list(systems.keys())
    header = "| Selectivity |"
    sep = "|-------------|"
    for name in sys_names:
        header += f" {name} QPS |"
        sep += "------------|"
    lines.append(header)
    lines.append(sep)

    for bin_label in all_bins:
        row = f"| {bin_label} |"
        for name in sys_names:
            curves = [c for c in systems[name].get("curves", [])
                      if c.get("selectivity_bin") == bin_label]
            qps = find_qps_at_recall(curves, target_recall)
            if qps is not None:
                row += f" {format_qps(qps)} |"
            else:
                row += " -- |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_speedup_table(
    systems: dict,
    our_name: str,
    target_recalls: list = [0.90, 0.95],
) -> str:
    """Generate speedup table: our system vs each baseline."""
    lines = []
    lines.append("### Speedup vs Baselines")
    lines.append("")

    our_curves = systems.get(our_name, {}).get("curves", [])
    if not our_curves:
        lines.append("*Our results not loaded. Cannot compute speedup.*")
        lines.append("")
        return "\n".join(lines)

    header = "| Baseline | Type |"
    sep = "|----------|------|"
    for r in target_recalls:
        header += f" Speedup@{r:.0%} |"
        sep += "-------------|"
    lines.append(header)
    lines.append(sep)

    for name, info in systems.items():
        if name == our_name:
            continue
        row = f"| {name} | {info['type']} |"
        for r in target_recalls:
            our_qps = find_qps_at_recall(our_curves, r)
            their_qps = find_qps_at_recall(info.get("curves", []), r)
            if our_qps and their_qps and their_qps > 0:
                speedup = our_qps / their_qps
                row += f" {speedup:.1f}x |"
            else:
                row += " -- |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_build_time_table(systems: dict) -> str:
    """Compare index build times."""
    lines = []
    lines.append("### Index Build Time (10M Vectors)")
    lines.append("")
    lines.append("| System | Type | Build Time | Notes |")
    lines.append("|--------|------|------------|-------|")

    for name, info in systems.items():
        bt = info.get("build_time_s")
        notes = info.get("build_notes", "")
        if bt is not None:
            lines.append(f"| {name} | {info['type']} | {format_time(bt)} | {notes} |")
        else:
            lines.append(f"| {name} | {info['type']} | -- | {notes} |")

    lines.append("")
    return "\n".join(lines)


def generate_memory_table(systems: dict) -> str:
    """Compare memory usage."""
    lines = []
    lines.append("### Memory Usage")
    lines.append("")
    lines.append("| System | Index Memory | Filter Memory | Total | Notes |")
    lines.append("|--------|-------------|---------------|-------|-------|")

    for name, info in systems.items():
        idx_mem = info.get("index_memory_bytes")
        flt_mem = info.get("filter_memory_bytes")
        total = info.get("total_memory_bytes")
        notes = info.get("memory_notes", "")
        idx_s = format_memory(idx_mem) if idx_mem else "--"
        flt_s = format_memory(flt_mem) if flt_mem else "--"
        tot_s = format_memory(total) if total else "--"
        lines.append(f"| {name} | {idx_s} | {flt_s} | {tot_s} | {notes} |")

    lines.append("")
    return "\n".join(lines)


def generate_architecture_comparison(systems: dict) -> str:
    """Qualitative comparison of architectural approaches."""
    lines = []
    lines.append("### Architectural Comparison")
    lines.append("")
    lines.append("| Property | cu-roaring (Ours) | ACORN | VecFlow |")
    lines.append("|----------|-------------------|-------|---------|")
    lines.append("| Compute | GPU | CPU | GPU |")
    lines.append("| Filter strategy | Predicate-agnostic post-filter | Predicate-aware graph traversal | Label-centric pre-indexed IVF |")
    lines.append("| Filter data structure | GPU Roaring bitmaps | Dense boolean map (O(nq*N)) | Per-label inverted lists |")
    lines.append("| Index build | Once (predicate-agnostic) | Once (predicate-agnostic) | Per-label (must rebuild on label change) |")
    lines.append("| New predicate support | Immediate (upload bitmap) | Immediate (new filter map) | Requires rebuild |")
    lines.append("| Multi-predicate AND | Fused GPU kernel | Dense map intersection | Streaming merge |")
    lines.append("| Memory scaling | O(cardinality) compressed | O(N) per query | O(sum of label postings) |")
    lines.append("| ANN algorithm | CAGRA (GPU graph) | HNSW (CPU graph) | CAGRA + BFS (GPU) |")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare cu-roaring-filter against ACORN and VecFlow baselines",
    )
    parser.add_argument(
        "--results-dir", default="results/baselines",
        help="Directory containing baseline result JSONs",
    )
    parser.add_argument(
        "--our-results-dir", default="results/raw",
        help="Directory containing our benchmark result JSONs",
    )
    parser.add_argument(
        "--output-dir", default="results/comparison",
        help="Output directory for comparison tables and data",
    )
    # Optional: manually provide our QPS-recall data if not in standard format
    parser.add_argument(
        "--our-qps-recall-json", default=None,
        help="Path to a JSON with our QPS-recall curves (override auto-loading)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load all results ----
    print("Loading results...")

    systems = {}

    # Load ACORN results
    acorn_data = load_json(os.path.join(args.results_dir, "acorn_results.json"))
    if acorn_data:
        systems["ACORN"] = {
            "type": "CPU",
            "curves": acorn_data.get("curves", []),
            "build_time_s": acorn_data.get("build_time_s"),
            "build_notes": f"HNSW M={acorn_data.get('index_params', {}).get('M', '?')}, "
                          f"gamma={acorn_data.get('index_params', {}).get('gamma', '?')}",
            "raw": acorn_data,
        }
        print(f"  Loaded ACORN: {len(acorn_data.get('curves', []))} data points")
    else:
        print("  ACORN results not found (run_baselines.py --acorn-only first)")

    # Load VecFlow results
    vecflow_data = load_json(os.path.join(args.results_dir, "vecflow_results.json"))
    if vecflow_data:
        systems["VecFlow"] = {
            "type": "GPU",
            "curves": vecflow_data.get("curves", []),
            "build_time_s": vecflow_data.get("build_time_s"),
            "build_notes": f"graph_degree={vecflow_data.get('index_params', {}).get('graph_degree', '?')}, "
                          f"threshold={vecflow_data.get('index_params', {}).get('specificity_threshold', '?')}",
            "raw": vecflow_data,
        }
        print(f"  Loaded VecFlow: {len(vecflow_data.get('curves', []))} data points")
    else:
        print("  VecFlow results not found (run_baselines.py --vecflow-only first)")

    # Load FAISS HNSW fallback results
    hnsw_data = load_json(os.path.join(args.results_dir, "faiss_hnsw_results.json"))
    if hnsw_data:
        systems["FAISS HNSW+PostFilter"] = {
            "type": "CPU",
            "curves": hnsw_data.get("curves", []),
            "build_time_s": hnsw_data.get("build_time_s"),
            "build_notes": "Stock FAISS HNSW with post-filtering",
            "raw": hnsw_data,
        }
        print(f"  Loaded FAISS HNSW: {len(hnsw_data.get('curves', []))} data points")

    # Load our results
    # Our benchmarks measure filter operations (not full ANN search), so the
    # comparison is nuanced. We load what we have and note the difference.
    our_data = load_json(
        args.our_qps_recall_json
        or os.path.join(args.results_dir, "cu_roaring_results.json")
    )
    if our_data:
        systems["cu-roaring (Ours)"] = {
            "type": "GPU",
            "curves": our_data.get("curves", []),
            "build_time_s": our_data.get("build_time_s"),
            "build_notes": "GPU Roaring + CAGRA",
            "raw": our_data,
        }
        print(f"  Loaded cu-roaring: {len(our_data.get('curves', []))} data points")
    else:
        print("  cu-roaring QPS-recall results not found.")
        print("  NOTE: Our system's full ANN recall-QPS curves require the CAGRA integration")
        print("  benchmark (bench_cagra_roaring.cu). The filter-only benchmarks in results/raw/")
        print("  measure filter operation throughput, not end-to-end ANN recall.")
        print("  Once CAGRA integration is benchmarked, save results as:")
        print(f"    {args.results_dir}/cu_roaring_results.json")
        print("  with the same format: {{curves: [{{recall_at_k, qps, selectivity_bin, ...}}]}}")

    # Also load our filter-operation benchmarks for the filter-only comparison table
    our_filter_data = load_json(os.path.join(args.our_results_dir, "bench6_point_query.json"))
    our_multi_and = load_json(os.path.join(args.our_results_dir, "bench9_multi_and.json"))
    system_info = load_json(os.path.join(os.path.dirname(args.our_results_dir), "system_info.json"))

    if not systems:
        print("\nNo baseline results found. Run evaluation first:")
        print("  python3 scripts/run_baselines.py --data-dir /path/to/yfcc100M")
        sys.exit(1)

    # ---- Generate comparison tables ----
    print("\nGenerating comparison tables...")

    md_sections = []

    # Title
    md_sections.append("# cu-roaring-filter vs Baselines: YFCC-10M Evaluation")
    md_sections.append("")
    md_sections.append("Dataset: YFCC-10M (10M vectors, 192-dim uint8, 200K tags)")
    if system_info:
        md_sections.append(f"GPU: {system_info.get('gpu', 'unknown')}")
    md_sections.append("")

    # Architecture comparison (always include)
    md_sections.append(generate_architecture_comparison(systems))

    # QPS at target recall
    if any(info.get("curves") for info in systems.values()):
        md_sections.append(generate_qps_at_recall_table(systems))
        md_sections.append(generate_per_selectivity_table(systems, target_recall=0.90))

        if "cu-roaring (Ours)" in systems:
            md_sections.append(generate_speedup_table(
                systems, our_name="cu-roaring (Ours)"
            ))

    # Build time
    md_sections.append(generate_build_time_table(systems))

    # Memory comparison
    md_sections.append(generate_memory_table(systems))

    # Filter-only comparison (our unique data)
    if our_filter_data:
        md_sections.append("### Filter Operation Throughput (cu-roaring only)")
        md_sections.append("")
        md_sections.append("These are filter-only operations (not full ANN search).")
        md_sections.append("Neither ACORN nor VecFlow expose equivalent filter-only APIs,")
        md_sections.append("so this data is unique to our system.")
        md_sections.append("")
        md_sections.append("| N (universe) | Density | Roaring Gq/s | Bitset Gq/s | Compression |")
        md_sections.append("|-------------|---------|-------------|------------|-------------|")

        results_list = our_filter_data.get("results", [])
        # Show a representative subset
        seen = set()
        for r in results_list:
            key = (r.get("universe"), r.get("density_label"), r.get("pattern"))
            if r.get("pattern") != "random":
                continue
            ukey = (r.get("universe"), r.get("density_label"))
            if ukey in seen:
                continue
            seen.add(ukey)
            md_sections.append(
                f"| {r.get('universe', '?'):,} | {r.get('density_label', '?')} "
                f"| {r.get('contains_gqps', 0):.1f} "
                f"| {r.get('bitset_gqps', 0):.1f} "
                f"| {r.get('compression_ratio', 0):.1f}x |"
            )
        md_sections.append("")

    if our_multi_and:
        md_sections.append("### Fused Multi-AND Throughput (cu-roaring only)")
        md_sections.append("")
        md_sections.append("| N (universe) | # Inputs | Density | Fused AND (ms) | Sequential (ms) | Speedup |")
        md_sections.append("|-------------|----------|---------|---------------|----------------|---------|")

        for r in our_multi_and.get("results", [])[:10]:
            fused_ms = r.get("fused_ms", {}).get("median", 0)
            seq_ms = r.get("sequential_ms", {}).get("median", 0)
            speedup = seq_ms / fused_ms if fused_ms > 0 else 0
            md_sections.append(
                f"| {r.get('universe', '?'):,} | {r.get('n_inputs', '?')} "
                f"| {r.get('density_label', '?')} | {fused_ms:.3f} "
                f"| {seq_ms:.3f} | {speedup:.1f}x |"
            )
        md_sections.append("")

    # Key takeaways placeholder
    md_sections.append("### Key Takeaways")
    md_sections.append("")
    md_sections.append("1. **vs ACORN (CPU)**: [Fill in after running ACORN evaluation]")
    md_sections.append("   - Expected: significant QPS advantage from GPU parallelism")
    md_sections.append("   - ACORN's filter_ids_map is O(nq*N) memory, limiting batch size")
    md_sections.append("   - Our Roaring bitmaps compress filters, enabling larger batches")
    md_sections.append("")
    md_sections.append("2. **vs VecFlow (GPU)**: [Fill in after running VecFlow evaluation]")
    md_sections.append("   - Key architectural difference: VecFlow pre-indexes by label, we post-filter")
    md_sections.append("   - VecFlow must rebuild index when labels change; we upload a new bitmap")
    md_sections.append("   - VecFlow may have higher raw QPS on common labels (pre-indexed)")
    md_sections.append("   - Our advantage: flexibility, memory efficiency, predicate composability")
    md_sections.append("")
    md_sections.append("3. **Unique strengths of cu-roaring-filter**:")
    md_sections.append("   - Predicate-agnostic: any boolean filter, no index rebuild")
    md_sections.append("   - Compressed: Roaring bitmaps use 2-60x less memory than bitsets")
    md_sections.append("   - Composable: fused multi-AND in a single kernel")
    md_sections.append("   - Drop-in: one-line integration with cuVS/CAGRA")
    md_sections.append("")

    # Write markdown
    md_output = os.path.join(args.output_dir, "comparison_tables.md")
    with open(md_output, "w") as f:
        f.write("\n".join(md_sections))
    print(f"  Markdown tables written to {md_output}")

    # ---- Write combined JSON ----
    combined = {
        "generated": "compare_results.py",
        "dataset": "YFCC-10M",
        "systems": {},
    }
    for name, info in systems.items():
        combined["systems"][name] = {
            "type": info["type"],
            "build_time_s": info.get("build_time_s"),
            "build_notes": info.get("build_notes", ""),
            "n_curve_points": len(info.get("curves", [])),
            "curves": info.get("curves", []),
        }

    json_output = os.path.join(args.output_dir, "comparison_data.json")
    with open(json_output, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Combined JSON written to {json_output}")

    # ---- Print summary to stdout ----
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for name, info in systems.items():
        curves = info.get("curves", [])
        if curves:
            best_recall = max(c.get("recall_at_k", 0) for c in curves)
            best_qps = max(c.get("qps", 0) for c in curves)
            print(f"\n  {name} ({info['type']}):")
            print(f"    Data points: {len(curves)}")
            print(f"    Best recall: {best_recall:.4f}")
            print(f"    Best QPS:    {format_qps(best_qps)}")
            qps_90 = find_qps_at_recall(curves, 0.90)
            if qps_90:
                print(f"    QPS@90%:     {format_qps(qps_90)}")
        else:
            print(f"\n  {name}: no curve data")

    print(f"\n  Full tables: {md_output}")
    print(f"  Raw data:    {json_output}")


if __name__ == "__main__":
    main()
