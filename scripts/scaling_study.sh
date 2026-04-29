#!/usr/bin/env bash
# scaling_study.sh — Sweep grid sizes, collect run_stats.json, print summary table.
#
# Usage:
#   ./scripts/scaling_study.sh [--nx256] [--re VALUE] [--t_end VALUE] [--cfl VALUE]
#
# Defaults: nx=ny in {32,64,128}, Re=100, CFL=0.5, t_end=2.0
# Add --nx256 to also run 256x256 (slow on CPU).
# Requires the solver to be built first: ./scripts/build_with_spack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
SOLVER="$ROOT_DIR/build/apps/ns_solver"

if [[ ! -x "$SOLVER" ]]; then
    echo "ERROR: solver not found at $SOLVER — run ./scripts/build_with_spack.sh first" >&2
    exit 1
fi

# Defaults
RE=100
CFL=0.5
T_END=2.0
GRIDS=(32 64 128 256 512 1024)
EXTRA_ARGS=()

# Parse arguments
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --nx256) GRIDS=(32 64 128 256) ;;
        --re)    (( i++ )); RE="${ARGS[$i]}"     ;;
        --t_end) (( i++ )); T_END="${ARGS[$i]}"  ;;
        --cfl)   (( i++ )); CFL="${ARGS[$i]}"    ;;
        *)       EXTRA_ARGS+=("${ARGS[$i]}")     ;;
    esac
    (( i++ )) || true
done

echo "=== Scaling Study: Re=${RE}, CFL=${CFL}, t_end=${T_END} ==="
echo ""

RUN_DIRS=()

for N in "${GRIDS[@]}"; do
    echo "--- Running ${N}x${N} ---"
    "$SOLVER" \
        --nx "$N" --ny "$N" \
        --re "$RE" --cfl "$CFL" --t_end "$T_END" \
        --profile \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

    RUN_DIR="$ROOT_DIR/data/run_${N}_${N}_${RE}_cfl${CFL}"
    RUN_DIRS+=("$RUN_DIR")
    echo ""
done

echo "=== Scaling Summary ==="
printf "%-12s  %8s  %10s  %13s  %14s  %10s\n" \
       "Grid" "Steps" "Wall(s)" "Wall/step(ms)" "PressureSolve%" "Predict%"
echo "------------------------------------------------------------------------------"

for RUN_DIR in "${RUN_DIRS[@]}"; do
    JSON="$RUN_DIR/run_stats.json"
    if [[ ! -f "$JSON" ]]; then
        echo "WARNING: missing $JSON — skipping" >&2
        continue
    fi

    python3 - "$JSON" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
cfg = d["config"]
t   = d["timing"]
s   = d["solver_stats"]
grid        = f"{cfg['nx']}x{cfg['ny']}"
steps       = s["n_steps"]
wall        = t["wall_total_s"]
wps         = t["wall_per_step_ms"]
psolve_pct  = 100.0 * t["wall_pressure_solve_s"] / wall if wall > 0 else 0.0
predict_pct = 100.0 * t["wall_predict_s"] / wall if wall > 0 else 0.0
print(f"{grid:<12}  {steps:>8}  {wall:>10.3f}  {wps:>13.3f}  {psolve_pct:>14.1f}  {predict_pct:>10.1f}")
PYEOF
done
echo "------------------------------------------------------------------------------"
