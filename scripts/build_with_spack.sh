#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# build_with_spack.sh — build NavierStokes2D using the installed spack env
#
# Automatically detects which spack environment has been set up on this
# machine by looking for a spack.lock file. Run setup_spack.sh first if
# no environment has been installed yet.
#
# Usage:
#   ./scripts/build_with_spack.sh [build_type]
#
#   build_type : Release | Debug  (default: Release)
#
# Requires spack to be sourced in your shell before running:
#   source /path/to/spack/share/spack/setup-env.sh
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="${1:-Release}"

if ! command -v spack &>/dev/null; then
    echo "Error: spack not found in PATH — source spack's setup-env.sh first"
    exit 1
fi

# --- detect installed environment -------------------------------------------
INSTALLED=()
for yaml in "${PROJECT_ROOT}"/spack/*/spack.lock; do
    [[ -f "$yaml" ]] && INSTALLED+=("$(basename "$(dirname "$yaml")")")
done

if [[ ${#INSTALLED[@]} -eq 0 ]]; then
    echo "Error: no installed spack environment found."
    echo "Run ./scripts/setup_spack.sh <env> first."
    echo "  env : ncam | hmorenom | greatlakes"
    exit 1
fi

if [[ ${#INSTALLED[@]} -gt 1 ]]; then
    echo "Error: multiple installed spack environments found: ${INSTALLED[*]}"
    echo "Only one environment should be installed per machine."
    echo "Remove the extra spack.lock file(s) to resolve the ambiguity."
    exit 1
fi

ENV_NAME="${INSTALLED[0]}"
ENV_PATH="${PROJECT_ROOT}/spack/${ENV_NAME}"

# --- activate and build -----------------------------------------------------
echo "Found installed environment: ${ENV_NAME}"
spack env activate "${ENV_PATH}"

BUILD_DIR="${PROJECT_ROOT}/build"

echo "Configuring (${BUILD_TYPE})..."
cmake -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "Building..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Done. Binary: ${BUILD_DIR}/apps/ns_solver"
