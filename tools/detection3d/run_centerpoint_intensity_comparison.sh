#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

mkdir -p "${UV_CACHE_DIR}" "${MPLCONFIGDIR}"

if [[ "${CENTERPOINT_DEVICE:-cuda}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

exec uv run --offline --no-sync python "${SCRIPT_DIR}/run_centerpoint_intensity_comparison.py" "$@"
