#!/bin/bash
# run_hw7.sh — standard HW7 production run: 64x64, Re=100, CFL=0.5, t=10
# Enables --diagnostics to write exact solution and error norm CSVs,
# then runs the Python plotting pipeline.
#
# Usage:
#   ./scripts/run_hw7.sh              # default run
#   ./scripts/run_hw7.sh --re 400     # override any ns_solver flag
#   ./scripts/run_hw7.sh --no-plots   # skip Python plotting step

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
SOLVER="$ROOT_DIR/build/apps/ns_solver"

if [[ ! -x "$SOLVER" ]]; then
    echo "ERROR: solver not found at $SOLVER — run build_with_spack.sh first" >&2
    exit 1
fi

# Separate --no-plots from flags forwarded to the solver
PLOT=1
SOLVER_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-plots" ]]; then
        PLOT=0
    else
        SOLVER_ARGS+=("$arg")
    fi
done

echo "=== Running ns_solver ==="
"$SOLVER" \
    --nx 64 --ny 64 \
    --re 100 \
    --cfl 0.5 \
    --t_end 10.0 \
    --diagnostics \
    "${SOLVER_ARGS[@]+"${SOLVER_ARGS[@]}"}"

if [[ $PLOT -eq 1 ]]; then
    # Determine run directory from default config (nx=64, ny=64, re=100, dt=0.001)
    RUN_DIR="$ROOT_DIR/data/run_64_64_100_0.001"
    echo ""
    echo "=== Generating plots ==="
    cd "$ROOT_DIR"
    python -m plotting "$RUN_DIR"
    echo "Figures written to $RUN_DIR/figures/"
fi
