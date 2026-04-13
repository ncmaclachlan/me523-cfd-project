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

# Use 'spack -e' to query the environment without requiring shell support.
# Set CMAKE_PREFIX_PATH so CMake finds spack-installed packages.
CMAKE_PREFIX_PATH="$(spack -e "${ENV_PATH}" find --format '{prefix}' | tr '\n' ':')"
export CMAKE_PREFIX_PATH

# Force FindMPI to use spack's OpenMPI, not conda's.
# MPI_HOME alone is not enough — FindMPI prefers mpicc on PATH (conda).
# Setting MPI_CXX_COMPILER explicitly overrides PATH-based detection.
MPI_HOME="$(spack -e "${ENV_PATH}" find --format '{prefix}' openmpi 2>/dev/null || true)"

# Pick up the C/C++ compiler that spack built the environment with, so that
# CMake uses the same compiler as the spack-installed libraries.
# Step 1: get the compiler spec (e.g. gcc@14.2.0) from any concrete package.
# Step 2: look up the actual binary paths via 'spack compiler info'.
COMPILER_SPEC="$(spack -e "${ENV_PATH}" find --format '{compiler}' cmake 2>/dev/null \
    | head -1 | sed 's|/.*||' || true)"  # strip hash suffix, e.g. gcc@14.2.0/abc -> gcc@14.2.0
COMPILER_CMAKE_ARGS=()
if [[ -n "${COMPILER_SPEC}" ]]; then
    COMPILER_INFO="$(spack compiler info "${COMPILER_SPEC}" 2>/dev/null || true)"
    SPACK_CC="$(echo "${COMPILER_INFO}"  | awk -F': ' '/^[[:space:]]+cc:/{print $2}'  | tr -d ' ')"
    SPACK_CXX="$(echo "${COMPILER_INFO}" | awk -F': ' '/^[[:space:]]+cxx:/{print $2}' | tr -d ' ')"
    if [[ -n "${SPACK_CC}" && -x "${SPACK_CC}" && -n "${SPACK_CXX}" && -x "${SPACK_CXX}" ]]; then
        COMPILER_CMAKE_ARGS+=(
            -DCMAKE_C_COMPILER="${SPACK_CC}"
            -DCMAKE_CXX_COMPILER="${SPACK_CXX}"
        )
    fi
fi

BUILD_DIR="${PROJECT_ROOT}/build"

MPI_CMAKE_ARGS=()
if [[ -n "${MPI_HOME}" ]]; then
    MPI_CMAKE_ARGS+=(
        -DMPI_HOME="${MPI_HOME}"
        -DMPI_CXX_COMPILER="${MPI_HOME}/bin/mpicxx"
        -DMPI_C_COMPILER="${MPI_HOME}/bin/mpicc"
        -DMPIEXEC_EXECUTABLE="${MPI_HOME}/bin/mpiexec"
    )
fi

echo "Configuring (${BUILD_TYPE})..."
cmake -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
      "${COMPILER_CMAKE_ARGS[@]}" \
      "${MPI_CMAKE_ARGS[@]}"

echo "Building..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build "${BUILD_DIR}" -j"${NPROC}"

echo "Done. Binary: ${BUILD_DIR}/apps/ns_solver"
