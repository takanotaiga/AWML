#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${1:-/home/taiga/ml_lake/t4-dataset}"
CHECKPOINT_PATH="${2:-${REPO_ROOT}/work_dirs/checkpoints/centerpoint_t4base_v2.1.0_best.pth}"
CHECKPOINT_URL="https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/best_NuScenes_metric_T4Metric_mAP_epoch_49.pth"
INFO_OUT_DIR="${DATA_ROOT}/info/local_smoke"
CENTERPOINT_DEVICE="${CENTERPOINT_DEVICE:-cuda}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

mkdir -p "${UV_CACHE_DIR}" "${INFO_OUT_DIR}" "$(dirname "${CHECKPOINT_PATH}")"

if [[ "${CENTERPOINT_DEVICE}" == "cpu" ]]; then
  # Force CPU inference.
  export CUDA_VISIBLE_DEVICES=""
else
  # deterministic=True in runtime config requires CuBLAS workspace setting on CUDA.
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
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

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Downloading checkpoint to ${CHECKPOINT_PATH}"
  curl -fL "${CHECKPOINT_URL}" -o "${CHECKPOINT_PATH}"
fi

echo "[1/2] Creating local smoke infos"
uv run --offline --no-sync python tools/detection3d/create_data_t4dataset.py \
  --root_path "${DATA_ROOT}" \
  --config autoware_ml/configs/detection3d/dataset/t4dataset/local_smoke.py \
  --version local_smoke \
  --max_sweeps 1 \
  --out_dir "${INFO_OUT_DIR}" \
  --dataset_version_config_root autoware_ml/configs/t4dataset/local_smoke

echo "[2/2] Running CenterPoint inference"
uv run --offline --no-sync python tools/detection3d/test.py \
  projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py \
  "${CHECKPOINT_PATH}" \
  --work-dir work_dirs/centerpoint/local_smoke_infer
