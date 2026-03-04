#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ZERO_INTENSITY=0
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zero)
      ZERO_INTENSITY=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_centerpoint_local_smoke_visualize.sh [--zero] [DATA_ROOT] [CHECKPOINT_PATH] [ANN_FILE_PATH]

Options:
  --zero    Set intensity channel to zero during visualization inference.
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

DATA_ROOT="${1:-/home/taiga/ml_lake/t4-dataset}"
CHECKPOINT_PATH="${2:-${REPO_ROOT}/work_dirs/checkpoints/centerpoint_j6gen2_v2.5.1_best.pth}"
ANN_FILE_PATH="${3:-info/local_smoke/t4dataset_local_smoke_infos_test.pkl}"
MODEL_CFG_PATH="${MODEL_CFG_PATH:-projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py}"
RUNTIME_MODEL_CFG_PATH="${MODEL_CFG_PATH}"
TMP_CFG_PATH=""
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

if [[ "${ZERO_INTENSITY}" == "1" ]]; then
  echo "Enabling intensity zero-fill mode (--zero)."
  TMP_CFG_PATH="$(uv run --offline --no-sync python - "${MODEL_CFG_PATH}" <<'PY'
import sys
import tempfile
from mmengine.config import Config

src = sys.argv[1]
cfg = Config.fromfile(src)
pipeline = list(cfg.test_dataloader.dataset.pipeline)

insert_idx = None
for idx, step in enumerate(pipeline):
    if isinstance(step, dict) and step.get("type") == "LoadPointsFromMultiSweeps":
        insert_idx = idx + 1
        break

if insert_idx is None:
    raise RuntimeError("LoadPointsFromMultiSweeps not found in test pipeline.")

pipeline.insert(insert_idx, dict(type="SetInferenceIntensityZero"))
cfg.test_dataloader.dataset.pipeline = pipeline
cfg.test_pipeline = pipeline

with tempfile.NamedTemporaryFile(prefix="centerpoint_local_smoke_zero_", suffix=".py", delete=False) as fp:
    cfg.dump(fp.name)
    print(fp.name)
PY
)"
  RUNTIME_MODEL_CFG_PATH="${TMP_CFG_PATH}"
  trap 'if [[ -n "${TMP_CFG_PATH}" && -f "${TMP_CFG_PATH}" ]]; then rm -f "${TMP_CFG_PATH}"; fi' EXIT
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
  --frame-range "${FRAME_START}" "${FRAME_END}" \
  --work-dir "${WORK_DIR}" \
  --max_workers "${MAX_WORKERS}"
