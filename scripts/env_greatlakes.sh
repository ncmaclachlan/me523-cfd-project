#!/usr/bin/env bash
# Source this script, do not execute it:
#   source scripts/env_greatlakes.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: this script must be sourced, not executed."
    echo "Usage: source scripts/env_greatlakes.sh"
    exit 1
fi

module load gcc/13.2.0
module load cuda/13.1.0
module load spack/0.23

source "$SPACK_ROOT/share/spack/setup-env.sh"

echo "Great Lakes environment ready."
echo "  gcc:   $(gcc --version | head -1)"
echo "  nvcc:  $(nvcc --version | grep release)"
echo "  spack: $(spack --version)"
