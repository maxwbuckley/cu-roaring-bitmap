#!/usr/bin/env bash
# =============================================================================
# setup_baselines.sh -- Clone, build, and configure ACORN and VecFlow baselines
#
# This script sets up two external baseline systems for comparison against
# cu-roaring-filter in the YFCC-10M filtered ANN evaluation:
#
#   ACORN  (CPU, SIGMOD 2024) -- Predicate-agnostic HNSW with subgraph traversal
#     Repo: https://github.com/stanford-futuredata/ACORN
#     Built on FAISS. CPU-only. Default in Weaviate and Vespa.
#
#   VecFlow (GPU, SIGMOD 2026) -- Label-centric IVF on GPU
#     Repo: https://github.com/Supercomputing-System-AI-Lab/VecFlow
#     Built on cuVS/CAGRA. Requires CUDA 11+ GPU.
#
# Usage:
#   bash scripts/setup_baselines.sh [--acorn-only] [--vecflow-only] [--dry-run]
#
# Prerequisites:
#   - CMake >= 3.24
#   - C++17 compiler (g++ >= 9 or clang++ >= 10)
#   - BLAS library (OpenBLAS or Intel MKL recommended)
#   - Python 3.10+ with pip
#   - For VecFlow: CUDA 11+ toolkit, conda, NVIDIA GPU with compute >= 7.0
#   - For ACORN: SWIG (for Python bindings, optional)
#
# The script creates:
#   baselines/
#   ├── ACORN/              -- ACORN source + build
#   │   └── build/          -- CMake build directory
#   └── VecFlow/            -- VecFlow source (pip-installed into conda env)
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINES_DIR="$PROJECT_ROOT/baselines"

# Parse arguments
DO_ACORN=true
DO_VECFLOW=true
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --acorn-only)  DO_VECFLOW=false ;;
        --vecflow-only) DO_ACORN=false ;;
        --dry-run)     DRY_RUN=true ;;
        --help|-h)
            echo "Usage: $0 [--acorn-only] [--vecflow-only] [--dry-run]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARNING: $*" >&2; }
die() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; exit 1; }

run() {
    if $DRY_RUN; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

# =============================================================================
# Dependency checks
# =============================================================================
log "Checking dependencies..."

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        warn "$1 not found. $2"
        return 1
    fi
    return 0
}

DEPS_OK=true

check_cmd cmake "Install: apt install cmake (>= 3.24 required)" || DEPS_OK=false
check_cmd git   "Install: apt install git" || DEPS_OK=false
check_cmd python3 "Install: apt install python3" || DEPS_OK=false

if $DO_ACORN; then
    check_cmd g++ "Install: apt install g++ (>= 9 required)" || DEPS_OK=false
    # Check for BLAS
    if ! ldconfig -p 2>/dev/null | grep -qE "lib(openblas|mkl|blas)"; then
        if [ ! -f /usr/lib/x86_64-linux-gnu/libblas.so ] && \
           [ ! -f /usr/lib/x86_64-linux-gnu/libopenblas.so ]; then
            warn "No BLAS library found. Install: apt install libopenblas-dev"
            warn "  (Intel MKL strongly recommended for best ACORN performance)"
        fi
    fi
fi

if $DO_VECFLOW; then
    check_cmd nvcc "Install CUDA toolkit >= 11.0" || DEPS_OK=false
    check_cmd conda "Install miniconda: https://docs.conda.io/en/latest/miniconda.html" || DEPS_OK=false
fi

if ! $DEPS_OK; then
    warn "Some dependencies missing. Proceeding anyway -- build steps may fail."
fi

mkdir -p "$BASELINES_DIR"

# =============================================================================
# ACORN Setup
# =============================================================================
if $DO_ACORN; then
    log "============================================"
    log "Setting up ACORN (CPU filtered ANN baseline)"
    log "============================================"

    ACORN_DIR="$BASELINES_DIR/ACORN"

    # ---- Clone ----
    if [ -d "$ACORN_DIR/.git" ]; then
        log "ACORN already cloned at $ACORN_DIR"
        log "  Pulling latest changes..."
        run git -C "$ACORN_DIR" pull --ff-only || warn "git pull failed; using existing checkout"
    else
        log "Cloning ACORN from https://github.com/stanford-futuredata/ACORN.git"
        run git clone https://github.com/stanford-futuredata/ACORN.git "$ACORN_DIR"
    fi

    # ---- Build (C++ library + Python bindings) ----
    log "Configuring ACORN build..."
    log "  ACORN is a fork of FAISS with the IndexACORNFlat index type."
    log "  Building with: GPU=OFF, Python=ON, Testing=ON, Release mode"

    ACORN_BUILD="$ACORN_DIR/build"
    run cmake \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_OPT_LEVEL=avx2 \
        -B "$ACORN_BUILD" \
        -S "$ACORN_DIR"

    NPROC=$(nproc 2>/dev/null || echo 8)
    log "Building ACORN C++ library (using $NPROC cores)..."
    run make -C "$ACORN_BUILD" -j"$NPROC" faiss

    # ---- Python bindings (optional but needed for our evaluation) ----
    log "Building ACORN Python bindings (requires SWIG)..."
    if check_cmd swig "Install: apt install swig (needed for ACORN Python bindings)"; then
        run make -C "$ACORN_BUILD" -j"$NPROC" swigfaiss || {
            warn "SWIG build failed. You can still use the C++ API directly."
            warn "To fix: apt install swig, then re-run this script."
        }
        if [ -d "$ACORN_BUILD/faiss/python" ] && ! $DRY_RUN; then
            log "Installing ACORN Python package..."
            (cd "$ACORN_BUILD/faiss/python" && python3 setup.py install --user) || {
                warn "Python install failed. Try: cd $ACORN_BUILD/faiss/python && pip install -e ."
            }
        fi
    else
        warn "SWIG not found. Skipping Python bindings."
        warn "C++ library is still built. Our evaluation script will fall back to subprocess mode."
    fi

    log "ACORN setup complete."
    log "  C++ library: $ACORN_BUILD/faiss/libfaiss.so"
    log "  Python: import faiss (if SWIG build succeeded)"
    log ""
    log "  Quick test:"
    log "    cd $ACORN_BUILD && make demo_ivfpq_indexing && ./demos/demo_ivfpq_indexing"
    log ""

    # ---- Document key ACORN API for reference ----
    cat > "$ACORN_DIR/EVALUATION_NOTES.md" << 'ACORN_NOTES'
# ACORN Evaluation Notes for cu-roaring-filter Paper

## Key API (C++)

```cpp
#include <faiss/IndexACORNFlat.h>

// Parameters:
//   d       = vector dimensionality (192 for YFCC)
//   M       = HNSW graph degree (32 or 64 typical)
//   gamma   = ACORN expansion factor (controls connectivity, 1-16)
//   M_beta  = max neighbors in hybrid graph (M*2 typical)
faiss::IndexACORNFlat index(d, M, gamma, M_beta);

// Build index
index.add(n_vectors, float_vectors);  // vectors must be float32

// Filtered search
// filter_ids_map: char array of size (n_queries * n_vectors)
//   filter_ids_map[q * n_vectors + v] = 1 if vector v passes query q's filter
// This is a BRUTE-FORCE boolean map -- O(nq * N) memory!
// For 10M vectors and 10K queries, that is 100 GB. Must batch queries.
index.search(n_queries, query_vectors, k, distances, labels, filter_ids_map);

// Search parameter (controls recall/speed tradeoff):
index.acorn.efSearch = efs;  // vary from 10 to 800 for recall-QPS curve
```

## Important Notes for Fair Comparison

1. ACORN requires float32 vectors. YFCC-10M is uint8. Must convert:
   `float_vec[i] = (float)uint8_vec[i]` (no normalization needed for L2)

2. filter_ids_map is O(nq * N) bytes. For YFCC-10M with batch queries:
   - 100 queries * 10M vectors = 1 GB per batch
   - Must process queries in small batches (100-1000 at a time)

3. Single-threaded QPS is the fair comparison for CPU baseline.
   Also report multi-threaded (OMP_NUM_THREADS=N) for completeness.

4. Build time is significant: HNSW construction on 10M vectors takes minutes.
   Report index build time separately from search throughput.

5. ACORN efSearch parameter controls the recall-QPS tradeoff:
   - Low efSearch (10-50): high QPS, lower recall
   - High efSearch (200-800): high recall, lower QPS
   Sweep this to generate the recall-QPS Pareto curve.
ACORN_NOTES
fi

# =============================================================================
# VecFlow Setup
# =============================================================================
if $DO_VECFLOW; then
    log "============================================"
    log "Setting up VecFlow (GPU filtered ANN baseline)"
    log "============================================"

    VECFLOW_DIR="$BASELINES_DIR/VecFlow"

    # ---- Clone ----
    if [ -d "$VECFLOW_DIR/.git" ]; then
        log "VecFlow already cloned at $VECFLOW_DIR"
        run git -C "$VECFLOW_DIR" pull --ff-only || warn "git pull failed; using existing checkout"
    else
        log "Cloning VecFlow from https://github.com/Supercomputing-System-AI-Lab/VecFlow.git"
        run git clone https://github.com/Supercomputing-System-AI-Lab/VecFlow.git "$VECFLOW_DIR"
    fi

    # ---- Conda environment ----
    VECFLOW_ENV="vecflow"
    log "Setting up conda environment '$VECFLOW_ENV'..."

    if conda env list 2>/dev/null | grep -q "^${VECFLOW_ENV} "; then
        log "  Conda env '$VECFLOW_ENV' already exists."
    else
        log "  Creating conda env '$VECFLOW_ENV' with Python 3.12..."
        run conda create -n "$VECFLOW_ENV" python=3.12 -y
    fi

    # ---- Install VecFlow ----
    log "Installing VecFlow into conda env..."
    log "  VecFlow is built on NVIDIA cuVS. It ships a pre-compiled wheel for CUDA 11."

    # The pip install must happen inside the conda env.
    # We write a helper script since 'conda activate' does not work in subshells.
    INSTALL_SCRIPT="$VECFLOW_DIR/_install_vecflow.sh"
    cat > "$INSTALL_SCRIPT" << 'VECFLOW_INSTALL'
#!/usr/bin/env bash
set -euo pipefail

# This script must be sourced or run inside the vecflow conda env:
#   conda activate vecflow && bash baselines/VecFlow/_install_vecflow.sh

pip install numpy
pip install cupy-cuda11x
pip install scipy  # for spmat loading

# Install VecFlow from the official release
pip install https://github.com/Supercomputing-System-AI-Lab/VecFlow/releases/download/0.0.1/vecflow-0.0.1.tar.gz

echo "VecFlow installed. Test with: python -c 'from vecflow import VecFlow; print(\"OK\")'"
VECFLOW_INSTALL
    chmod +x "$INSTALL_SCRIPT"

    if ! $DRY_RUN; then
        log "  Running install script in conda env..."
        log "  NOTE: If this fails, run manually:"
        log "    conda activate $VECFLOW_ENV"
        log "    bash $INSTALL_SCRIPT"
        # Try to run inside conda env. This may fail if conda init hasn't been
        # run in this shell. That's OK -- the user can do it manually.
        (
            eval "$(conda shell.bash hook 2>/dev/null)" || true
            conda activate "$VECFLOW_ENV" 2>/dev/null && bash "$INSTALL_SCRIPT"
        ) || {
            warn "Automated VecFlow install failed (conda activate issue in subshell)."
            warn "Please run manually:"
            warn "  conda activate $VECFLOW_ENV"
            warn "  bash $INSTALL_SCRIPT"
        }
    fi

    log "VecFlow setup complete."
    log ""
    log "  Quick test:"
    log "    conda activate $VECFLOW_ENV"
    log "    python -c 'from vecflow import VecFlow; print(\"VecFlow OK\")'"
    log ""

    # ---- Document key VecFlow API for reference ----
    cat > "$VECFLOW_DIR/EVALUATION_NOTES.md" << 'VECFLOW_NOTES'
# VecFlow Evaluation Notes for cu-roaring-filter Paper

## Key API (Python)

```python
from vecflow import VecFlow
import numpy as np

# Build index
# dataset: np.ndarray of shape (n_vectors, dim), dtype float32
# data_labels: list of lists, e.g. [[0, 3, 7], [1, 5], ...] per vector
vf = VecFlow()
vf.build(
    dataset=dataset,           # (N, d) float32
    data_labels=data_labels,   # list[list[int]] per vector
    graph_degree=16,           # CAGRA graph degree
    specificity_threshold=2000,# label count threshold for IVF-Graph vs IVF-BFS
    ivf_graph_fname="graph.bin",
    ivf_bfs_fname="bfs.bin"
)

# Search
# queries: (nq, d) float32
# query_labels: list[list[int]] per query (AND semantics)
neighbors, distances = vf.search(
    queries=query_vectors,
    query_labels=query_labels,
    itopk_size=32    # internal top-k, controls recall/speed tradeoff
)
```

## Important Notes for Fair Comparison

1. VecFlow uses label-centric IVF: it pre-indexes by label at build time.
   Our system (cu-roaring-filter) is predicate-agnostic (post-filtering).
   This is a fundamental architectural difference to highlight in the paper.

2. VecFlow requires float32 vectors. YFCC-10M is uint8. Must convert.

3. VecFlow's label format is per-vector list of integer label IDs.
   YFCC-10M tags map directly: each vector has a list of tag IDs.

4. The specificity_threshold parameter is critical:
   - Labels with >= threshold vectors use IVF-CAGRA (fast, graph-based)
   - Labels with < threshold vectors use IVF-BFS (scan-based)
   - YFCC tag distribution is highly skewed (Zipfian), so this matters.

5. VecFlow was evaluated on A100. We run on RTX 5090.
   Report GPU specs clearly. The 5090 has more memory bandwidth
   but different SM count. Both have comparable compute for this workload.

6. itopk_size controls the recall-QPS tradeoff:
   - Low itopk (16-32): high QPS, lower recall
   - High itopk (64-256): high recall, lower QPS
   Sweep this to generate the recall-QPS Pareto curve.

7. VecFlow build time includes per-label subindex construction.
   This is O(n_labels * avg_label_size * log(...)). May be significant
   for YFCC's 200K labels.

8. VecFlow's pre-compiled wheel is for CUDA 11. If running on CUDA 12,
   may need to build from source. Check: nvcc --version
VECFLOW_NOTES
fi

# =============================================================================
# Summary
# =============================================================================
log "============================================"
log "Baseline Setup Summary"
log "============================================"
log ""
log "Directory: $BASELINES_DIR/"
$DO_ACORN && log "  ACORN:   $BASELINES_DIR/ACORN/ (CPU, FAISS-based, MIT license)"
$DO_VECFLOW && log "  VecFlow: $BASELINES_DIR/VecFlow/ (GPU, cuVS-based, Apache 2.0)"
log ""
log "Next steps:"
log "  1. Verify builds:"
$DO_ACORN && log "       python3 -c 'import faiss; idx = faiss.IndexACORNFlat(192, 32, 12, 64); print(\"ACORN OK\")'"
$DO_VECFLOW && log "       conda activate vecflow && python3 -c 'from vecflow import VecFlow; print(\"VecFlow OK\")'"
log ""
log "  2. Run evaluation:"
log "       python3 scripts/run_baselines.py \\"
log "         --data-dir /mnt/c/Users/maxwb/Development/big-ann-benchmarks/data/yfcc100M \\"
log "         --yfcc-export-dir bench/yfcc_data"
log ""
log "  3. Compare results:"
log "       python3 scripts/compare_results.py"
log ""
log "See EVALUATION_NOTES.md in each baseline directory for API details."
