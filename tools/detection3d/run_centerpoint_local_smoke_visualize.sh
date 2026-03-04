#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${1:-/home/taiga/ml_lake/t4-dataset}"
CHECKPOINT_PATH="${2:-${REPO_ROOT}/work_dirs/checkpoints/centerpoint_t4base_v2.1.0_best.pth}"
ANN_FILE_PATH="${3:-info/local_smoke/t4dataset_local_smoke_infos_test.pkl}"
MODEL_CFG_PATH="${MODEL_CFG_PATH:-projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py}"
WORK_DIR="${WORK_DIR:-work_dirs/centerpoint/local_smoke_visualize}"
MAX_WORKERS="${MAX_WORKERS:-2}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:-2}"
CENTERPOINT_DEVICE="${CENTERPOINT_DEVICE:-gpu}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

mkdir -p "${UV_CACHE_DIR}"

if [[ "${CENTERPOINT_DEVICE}" == "cpu" ]]; then
  # Force CPU visualization.
  export CUDA_VISIBLE_DEVICES=""
else
  # deterministic=True in runtime config requires CuBLAS workspace setting on CUDA.
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

ANN_CHECK_PATH="${ANN_FILE_PATH}"
if [[ "${ANN_CHECK_PATH}" != /* ]]; then
  ANN_CHECK_PATH="${DATA_ROOT%/}/${ANN_CHECK_PATH#./}"
fi
if [[ ! -f "${ANN_CHECK_PATH}" ]]; then
  echo "Annotation file not found: ${ANN_CHECK_PATH}" >&2
  exit 1
fi

if ! uv run --offline --no-sync python - <<'PY'
import importlib
import sys

try:
    importlib.import_module("mmcv._ext")
except Exception:
    sys.exit(1)
PY
then
  echo "mmcv CUDA extension is missing. Rebuilding mmcv with CUDA ops..."
  MMCV_WITH_OPS=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
    MAX_JOBS="${MAX_JOBS:-16}" \
    uv pip install --python .venv/bin/python --no-cache \
      --reinstall-package mmcv --no-binary mmcv --no-build-isolation "mmcv==2.1.0"
fi

uv run --offline --no-sync python tools/detection3d/visualize_bboxes.py \
  "${MODEL_CFG_PATH}" \
  "${CHECKPOINT_PATH}" \
  --device "${CENTERPOINT_DEVICE}" \
  --data-root "${DATA_ROOT}" \
  --ann-file-path "${ANN_FILE_PATH}" \
  --frame-range "${FRAME_START}" "${FRAME_END}" \
  --work-dir "${WORK_DIR}" \
  --max_workers "${MAX_WORKERS}"
