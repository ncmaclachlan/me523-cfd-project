#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# setup_spack.sh — one-time spack environment setup for a given machine
#
# Concretizes and installs all packages defined in the spack environment.
# Run this once when setting up on a new machine, or after changing spack.yaml.
#
# Usage:
#   ./scripts/setup_spack.sh <env>
#
#   env : ncam | hmorenom | greatlakes
#
# Requires spack to be sourced in your shell before running:
#   source /path/to/spack/share/spack/setup-env.sh
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <env>"
    echo "  env : ncam | hmorenom | greatlakes"
    exit 1
fi

ENV_NAME="$1"
ENV_PATH="${PROJECT_ROOT}/spack/${ENV_NAME}"

VALID_ENVS=("ncam" "hmorenom" "greatlakes")
if [[ ! " ${VALID_ENVS[*]} " =~ " ${ENV_NAME} " ]]; then
    echo "Error: unknown environment '${ENV_NAME}'"
    echo "Valid options: ${VALID_ENVS[*]}"
    exit 1
fi

if [[ ! -f "${ENV_PATH}/spack.yaml" ]]; then
    echo "Error: spack.yaml not found at ${ENV_PATH}"
    exit 1
fi

if ! command -v spack &>/dev/null; then
    echo "Error: spack not found in PATH — source spack's setup-env.sh first"
    exit 1
fi

echo "Using spack environment: ${ENV_NAME}"

echo "Concretizing..."
spack -e "${ENV_PATH}" concretize

echo "Installing (this may take a while)..."
spack -e "${ENV_PATH}" install

echo "Done. Run ./scripts/build_with_spack.sh to build the project."
