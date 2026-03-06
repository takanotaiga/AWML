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
  run_centerpoint_local_smoke_visualize.sh [--zero | --range] [--range-attenuation A] [DATA_ROOT] [CHECKPOINT_PATH] [ANN_FILE_PATH]
  run_centerpoint_local_smoke_visualize.sh [--zero | --range] [--range-attenuation A] --pointcloud-bin <FILE> [--pointcloud-bin <FILE> ...] [CHECKPOINT_PATH]

Options:
  --zero               Set intensity channel to zero during visualization inference.
  --range              Replace intensity with range-based exponential decay during visualization inference.
  --range-attenuation  Decay coefficient a used by range mode. Default: 0.02.
  --pointcloud-bin     Directly infer/visualize from .pcd.bin file(s) without info pkl preparation.
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
DEFAULT_ANN_FILE_PATH="info/local_smoke/t4dataset_local_smoke_infos_test.pkl"
DIRECT_MODE=0
DATA_ROOT="${DEFAULT_DATA_ROOT}"
CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH}"
ANN_FILE_PATH="${DEFAULT_ANN_FILE_PATH}"
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
  ANN_FILE_PATH="${3:-${DEFAULT_ANN_FILE_PATH}}"
fi

MODEL_CFG_PATH="${MODEL_CFG_PATH:-projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py}"
RUNTIME_MODEL_CFG_PATH="${MODEL_CFG_PATH}"
TMP_CFG_PATH=""
TMP_DIRECT_INFO_DIR=""
WORK_DIR="${WORK_DIR:-work_dirs/centerpoint/local_smoke_visualize}"
MAX_WORKERS="${MAX_WORKERS:-2}"
CENTERPOINT_DEVICE="${CENTERPOINT_DEVICE:-gpu}"
MP4_FPS="${MP4_FPS:-10}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

mkdir -p "${UV_CACHE_DIR}"

cleanup() {
  if [[ -n "${TMP_CFG_PATH}" && -f "${TMP_CFG_PATH}" ]]; then
    rm -f "${TMP_CFG_PATH}"
  fi
  if [[ -n "${TMP_DIRECT_INFO_DIR}" && -d "${TMP_DIRECT_INFO_DIR}" ]]; then
    rm -rf "${TMP_DIRECT_INFO_DIR}"
  fi
}
trap cleanup EXIT

if [[ "${CENTERPOINT_DEVICE}" == "cpu" ]]; then
  # Force CPU visualization.
  export CUDA_VISIBLE_DEVICES=""
else
  # deterministic=True in runtime config requires CuBLAS workspace setting on CUDA.
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

if [[ "${DIRECT_MODE}" == "0" && ! -d "${DATA_ROOT}" ]]; then
  echo "Data root not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
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

tmp_dir = Path(tempfile.mkdtemp(prefix="centerpoint_local_smoke_visualize_pointcloud_bin_"))
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
  DATA_ROOT="${DIRECT_MODE_INFO[0]}"
  ANN_FILE_PATH="${DIRECT_MODE_INFO[1]}"
  TMP_DIRECT_INFO_DIR="$(dirname "${ANN_FILE_PATH}")"
fi

ANN_CHECK_PATH="${ANN_FILE_PATH}"
if [[ "${ANN_CHECK_PATH}" != /* ]]; then
  ANN_CHECK_PATH="${DATA_ROOT%/}/${ANN_CHECK_PATH#./}"
fi
if [[ ! -f "${ANN_CHECK_PATH}" ]]; then
  echo "Annotation file not found: ${ANN_CHECK_PATH}" >&2
  exit 1
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

uv run --offline --no-sync python tools/detection3d/visualize_bboxes.py \
  "${RUNTIME_MODEL_CFG_PATH}" \
  "${CHECKPOINT_PATH}" \
  --device "${CENTERPOINT_DEVICE}" \
  --data-root "${DATA_ROOT}" \
  --ann-file-path "${ANN_FILE_PATH}" \
  --work-dir "${WORK_DIR}" \
  --max_workers "${MAX_WORKERS}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to export mp4 files, but it was not found in PATH." >&2
  exit 1
fi

VIS_DIR="${WORK_DIR}/vis"
if [[ ! -d "${VIS_DIR}" ]]; then
  latest_run_dir="$(find "${WORK_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -n "${latest_run_dir}" && -d "${latest_run_dir}/vis" ]]; then
    VIS_DIR="${latest_run_dir}/vis"
  fi
fi
if [[ ! -d "${VIS_DIR}" ]]; then
  echo "Visualization output directory not found: ${VIS_DIR}" >&2
  exit 1
fi

mapfile -t RENDER_DIRS < <(
  find "${VIS_DIR}" -type d \( -name "BEV" -o -name "CAM" -o -name "bev" -o -name "cam" \) | sort
)

if [[ "${#RENDER_DIRS[@]}" -eq 0 ]]; then
  echo "No visualization render directories found under: ${VIS_DIR}" >&2
  exit 1
fi

for render_dir in "${RENDER_DIRS[@]}"; do
  kind="$(basename "${render_dir}")"
  kind_lower="$(printf "%s" "${kind}" | tr '[:upper:]' '[:lower:]')"
  list_file="$(mktemp)"
  while IFS= read -r png_path; do
    abs_png_path="$(realpath "${png_path}")"
    escaped_path=${abs_png_path//\'/\'\\\'\'}
    printf "file '%s'\n" "${escaped_path}" >> "${list_file}"
  done < <(find "${render_dir}" -maxdepth 1 -type f -name "*.png" | sort)

  if [[ ! -s "${list_file}" ]]; then
    rm -f "${list_file}"
    continue
  fi

  out_mp4="${render_dir}/${kind_lower}.mp4"
  ffmpeg -y -hide_banner -loglevel error \
    -r "${MP4_FPS}" \
    -f concat \
    -safe 0 \
    -i "${list_file}" \
    -vf "fps=${MP4_FPS},format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    "${out_mp4}"
  rm -f "${list_file}"
  echo "Saved mp4: ${out_mp4}"
done
