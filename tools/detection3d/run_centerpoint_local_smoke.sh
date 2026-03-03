#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${1:-/home/taiga/ml_lake/t4-dataset}"
CHECKPOINT_PATH="${2:-${REPO_ROOT}/work_dirs/checkpoints/centerpoint_t4base_v2.1.0_best.pth}"
CHECKPOINT_URL="https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/t4base/v2.1.0/best_NuScenes_metric_T4Metric_mAP_epoch_49.pth"
INFO_OUT_DIR="${DATA_ROOT}/info/local_smoke"
CENTERPOINT_DEVICE="${CENTERPOINT_DEVICE:-cpu}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${UV_CACHE_DIR}" "${INFO_OUT_DIR}" "$(dirname "${CHECKPOINT_PATH}")"

if [[ "${CENTERPOINT_DEVICE}" == "cpu" ]]; then
  # Torch 2.1 + cu118 cannot run on newer Blackwell GPUs.
  export CUDA_VISIBLE_DEVICES=""
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Downloading checkpoint to ${CHECKPOINT_PATH}"
  curl -fL "${CHECKPOINT_URL}" -o "${CHECKPOINT_PATH}"
fi

echo "[1/2] Creating local smoke infos"
uv run --offline python tools/detection3d/create_data_t4dataset.py \
  --root_path "${DATA_ROOT}" \
  --config autoware_ml/configs/detection3d/dataset/t4dataset/local_smoke.py \
  --version local_smoke \
  --max_sweeps 1 \
  --out_dir "${INFO_OUT_DIR}" \
  --dataset_version_config_root autoware_ml/configs/t4dataset/local_smoke

echo "[2/2] Running CenterPoint inference"
uv run --offline python tools/detection3d/test.py \
  projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py \
  "${CHECKPOINT_PATH}" \
  --work-dir work_dirs/centerpoint/local_smoke_infer
