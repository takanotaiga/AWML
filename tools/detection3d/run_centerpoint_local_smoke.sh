#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

INTENSITY_MODE="original"
RANGE_ATTENUATION_COEFFICIENT="${RANGE_ATTENUATION_COEFFICIENT:-0.02}"
POINTCLOUD_BINS=()
POSITIONAL_ARGS=()

set_intensity_mode() {
  local next_mode="$1"
  if [[ "${INTENSITY_MODE}" != "original" ]]; then
    echo "--zero and --range cannot be used together." >&2
    exit 1
  fi
  INTENSITY_MODE="${next_mode}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zero)
      set_intensity_mode "zero"
      shift
      ;;
    --range)
      set_intensity_mode "range"
      shift
      ;;
    --range-attenuation)
      if [[ $# -lt 2 ]]; then
        echo "--range-attenuation requires a numeric argument." >&2
        exit 1
      fi
      RANGE_ATTENUATION_COEFFICIENT="$2"
      shift 2
      ;;
    --pointcloud-bin)
      if [[ $# -lt 2 ]]; then
        echo "--pointcloud-bin requires a file path argument." >&2
        exit 1
      fi
      POINTCLOUD_BINS+=("$2")
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  run_centerpoint_local_smoke.sh [--zero | --range] [--range-attenuation A] [DATA_ROOT] [CHECKPOINT_PATH]
  run_centerpoint_local_smoke.sh [--zero | --range] [--range-attenuation A] --pointcloud-bin <FILE> [--pointcloud-bin <FILE> ...] [CHECKPOINT_PATH]

Options:
  --zero               Set intensity channel to zero during inference.
  --range              Replace intensity with range-based exponential decay during inference.
  --range-attenuation  Decay coefficient a used by range mode. Default: 0.02.
  --pointcloud-bin     Directly infer from .pcd.bin file(s) without creating local_smoke infos.
EOF
      exit 0
      ;;
    --)
      shift
      POSITIONAL_ARGS+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}"

DEFAULT_DATA_ROOT="/home/taiga/ml_lake/t4-dataset"
DEFAULT_CHECKPOINT_PATH="${REPO_ROOT}/work_dirs/checkpoints/centerpoint_j6gen2_v2.5.1_best.pth"
DIRECT_MODE=0
DATA_ROOT="${DEFAULT_DATA_ROOT}"
CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH}"
if [[ "${#POINTCLOUD_BINS[@]}" -gt 0 ]]; then
  DIRECT_MODE=1
  if [[ $# -gt 1 ]]; then
    echo "In --pointcloud-bin mode, only optional [CHECKPOINT_PATH] positional argument is supported." >&2
    exit 1
  fi
  CHECKPOINT_PATH="${1:-${DEFAULT_CHECKPOINT_PATH}}"
else
  DATA_ROOT="${1:-${DEFAULT_DATA_ROOT}}"
  CHECKPOINT_PATH="${2:-${DEFAULT_CHECKPOINT_PATH}}"
fi

CHECKPOINT_URL="https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/best_NuScenes_metric_T4Metric_mAP_epoch_29.pth"
CHECKPOINT_LOGS_URL="https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/logs.zip"
INFO_OUT_DIR="${DATA_ROOT}/info/local_smoke"
CENTERPOINT_DEVICE="${CENTERPOINT_DEVICE:-cuda}"
MODEL_CFG_PATH="projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py"
RUNTIME_MODEL_CFG_PATH="${MODEL_CFG_PATH}"
TMP_CFG_PATH=""
TMP_DIRECT_CFG_PATH=""
TMP_DIRECT_INFO_DIR=""

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

cleanup() {
  if [[ -n "${TMP_CFG_PATH}" && -f "${TMP_CFG_PATH}" ]]; then
    rm -f "${TMP_CFG_PATH}"
  fi
  if [[ -n "${TMP_DIRECT_CFG_PATH}" && -f "${TMP_DIRECT_CFG_PATH}" ]]; then
    rm -f "${TMP_DIRECT_CFG_PATH}"
  fi
  if [[ -n "${TMP_DIRECT_INFO_DIR}" && -d "${TMP_DIRECT_INFO_DIR}" ]]; then
    rm -rf "${TMP_DIRECT_INFO_DIR}"
  fi
}
trap cleanup EXIT

mkdir -p "${UV_CACHE_DIR}" "${INFO_OUT_DIR}" "$(dirname "${CHECKPOINT_PATH}")"

if [[ "${CENTERPOINT_DEVICE}" == "cpu" ]]; then
  # Force CPU inference.
  export CUDA_VISIBLE_DEVICES=""
else
  # deterministic=True in runtime config requires CuBLAS workspace setting on CUDA.
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

if [[ "${INTENSITY_MODE}" != "original" ]]; then
  if [[ "${INTENSITY_MODE}" == "zero" ]]; then
    echo "Enabling intensity zero-fill mode (--zero)."
  else
    echo "Enabling range-based intensity mode (--range, a=${RANGE_ATTENUATION_COEFFICIENT})."
  fi
  TMP_CFG_PATH="$(uv run --offline --no-sync python - "${MODEL_CFG_PATH}" "${INTENSITY_MODE}" "${RANGE_ATTENUATION_COEFFICIENT}" <<'PY'
import sys
import tempfile
from mmengine.config import Config

src = sys.argv[1]
mode = sys.argv[2]
attenuation = float(sys.argv[3])
cfg = Config.fromfile(src)
pipeline = list(cfg.test_dataloader.dataset.pipeline)

insert_idx = None
for idx, step in enumerate(pipeline):
    if isinstance(step, dict) and step.get("type") == "LoadPointsFromMultiSweeps":
        insert_idx = idx + 1
        break

if insert_idx is None:
    raise RuntimeError("LoadPointsFromMultiSweeps not found in test pipeline.")

if mode == "zero":
    transform = dict(type="SetInferenceIntensityZero")
elif mode == "range":
    transform = dict(
        type="SetInferenceIntensityFromRange",
        attenuation_coefficient=attenuation,
        max_intensity=255.0,
    )
else:
    raise ValueError(f"Unsupported intensity mode: {mode}")

pipeline.insert(insert_idx, transform)
cfg.test_dataloader.dataset.pipeline = pipeline
cfg.test_pipeline = pipeline

with tempfile.NamedTemporaryFile(prefix=f"centerpoint_local_smoke_{mode}_", suffix=".py", delete=False) as fp:
    cfg.dump(fp.name)
    print(fp.name)
PY
)"
  RUNTIME_MODEL_CFG_PATH="${TMP_CFG_PATH}"
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
  if ! curl -fL "${CHECKPOINT_URL}" -o "${CHECKPOINT_PATH}"; then
    echo "Direct checkpoint URL is unavailable. Falling back to logs.zip from model-zoo."
    CHECKPOINT_LOGS_ZIP="${CHECKPOINT_PATH%.pth}_logs.zip"
    curl -fL "${CHECKPOINT_LOGS_URL}" -o "${CHECKPOINT_LOGS_ZIP}"
    uv run --offline --no-sync python - "${CHECKPOINT_LOGS_ZIP}" "${CHECKPOINT_PATH}" <<'PY'
import os
import re
import shutil
import sys
import zipfile

zip_path = sys.argv[1]
out_path = sys.argv[2]

with zipfile.ZipFile(zip_path) as zf:
    pth_names = [n for n in zf.namelist() if n.endswith(".pth")]
    if not pth_names:
        raise RuntimeError(f"No .pth file found in {zip_path}")

    def score(path: str) -> tuple[int, int]:
        name = os.path.basename(path).lower()
        priority = 0
        if "best" in name:
            priority += 1000
        if "t4metric" in name:
            priority += 100
        if "nuscenes" in name:
            priority += 50
        m = re.search(r"epoch[_-]?(\d+)", name)
        epoch = int(m.group(1)) if m else -1
        return (priority, epoch)

    pth_names.sort(key=score, reverse=True)
    selected = pth_names[0]
    print(f"Extracting checkpoint from logs.zip: {selected}")
    with zf.open(selected) as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
PY
    rm -f "${CHECKPOINT_LOGS_ZIP}"
  fi
fi

if [[ "${DIRECT_MODE}" == "1" ]]; then
  mapfile -t DIRECT_MODE_INFO < <(
    uv run --offline --no-sync python - "${RUNTIME_MODEL_CFG_PATH}" "${POINTCLOUD_BINS[@]}" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

import mmengine
import numpy as np
from mmengine.config import Config

from tools.detection3d.t4dataset_converters.update_infos_to_v2 import get_empty_standard_data_info

cfg_path = sys.argv[1]
pointcloud_paths = [Path(p).expanduser().resolve() for p in sys.argv[2:]]
if not pointcloud_paths:
    raise ValueError("No pointcloud files were provided.")

for path in pointcloud_paths:
    if not path.is_file():
        raise FileNotFoundError(f"Pointcloud file not found: {path}")

cfg = Config.fromfile(cfg_path)
camera_types_cfg = cfg.get("camera_types", [])
if isinstance(camera_types_cfg, set):
    camera_types = sorted(camera_types_cfg)
else:
    camera_types = list(camera_types_cfg)
if not camera_types:
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_BACK_LEFT",
    ]

class_names = list(cfg.get("class_names", []))
if not class_names:
    class_names = list(cfg.get("metainfo", {}).get("classes", []))

common_root = Path(os.path.commonpath([str(p.parent) for p in pointcloud_paths])).resolve()
identity = np.eye(4, dtype=np.float32).tolist()
data_list = []
for idx, path in enumerate(pointcloud_paths):
    info = get_empty_standard_data_info(camera_types)
    lidar_rel_path = os.path.relpath(path, common_root).replace(os.sep, "/")
    info["sample_idx"] = idx
    info["token"] = f"pointcloud_bin_{idx}"
    info["scene_token"] = "pointcloud_bin_scene"
    info["timestamp"] = float(idx)
    info["ego2global"] = identity
    info["lidar_points"] = dict(
        num_pts_feats=5,
        lidar_path=lidar_rel_path,
        lidar2ego=identity,
    )
    info["lidar_sweeps"] = []
    info["instances"] = []
    data_list.append(info)

tmp_dir = Path(tempfile.mkdtemp(prefix="centerpoint_local_smoke_pointcloud_bin_"))
ann_path = tmp_dir / "t4dataset_pointcloud_bin_infos_test.pkl"
mmengine.dump(dict(data_list=data_list, metainfo=dict(classes=class_names, version="pointcloud_bin")), ann_path)

print(common_root)
print(ann_path)
PY
  )
  if [[ "${#DIRECT_MODE_INFO[@]}" -ne 2 ]]; then
    echo "Failed to generate temporary info for --pointcloud-bin mode." >&2
    exit 1
  fi

  DIRECT_DATA_ROOT="${DIRECT_MODE_INFO[0]}"
  DIRECT_ANN_FILE_PATH="${DIRECT_MODE_INFO[1]}"
  TMP_DIRECT_INFO_DIR="$(dirname "${DIRECT_ANN_FILE_PATH}")"
  DIRECT_DUMP_PATH="work_dirs/centerpoint/local_smoke_infer/direct_pointcloud_results.pkl"
  mkdir -p "$(dirname "${DIRECT_DUMP_PATH}")"

  TMP_DIRECT_CFG_PATH="$(uv run --offline --no-sync python - "${RUNTIME_MODEL_CFG_PATH}" "${DIRECT_DATA_ROOT}" "${DIRECT_ANN_FILE_PATH}" "${DIRECT_DUMP_PATH}" <<'PY'
import sys
import tempfile
from mmengine.config import Config

src_cfg = sys.argv[1]
data_root = sys.argv[2]
ann_file_path = sys.argv[3]
dump_path = sys.argv[4]

cfg = Config.fromfile(src_cfg)
cfg.data_root = data_root
cfg.info_directory_path = ""
cfg.dataset_test_groups = dict(pointcloud_bin=ann_file_path)
cfg.test_dataloader.dataset.data_root = data_root
cfg.test_dataloader.dataset.ann_file = ann_file_path
cfg.test_evaluator = dict(type="DumpResults", out_file_path=dump_path)

with tempfile.NamedTemporaryFile(prefix="centerpoint_local_smoke_pointcloud_bin_", suffix=".py", delete=False) as fp:
    cfg.dump(fp.name)
    print(fp.name)
PY
)"

  echo "[1/1] Running CenterPoint inference on direct pointcloud files"
  uv run --offline --no-sync python tools/detection3d/test.py \
    "${TMP_DIRECT_CFG_PATH}" \
    "${CHECKPOINT_PATH}" \
    --work-dir work_dirs/centerpoint/local_smoke_infer
  echo "Saved raw predictions to ${DIRECT_DUMP_PATH}"
else
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
    "${RUNTIME_MODEL_CFG_PATH}" \
    "${CHECKPOINT_PATH}" \
    --work-dir work_dirs/centerpoint/local_smoke_infer
fi
