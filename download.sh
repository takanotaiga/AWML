#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADER="${ROOT_DIR}/pipelines/webauto/download_t4dataset/download_t4dataset.py"
CONFIG_ROOT="${ROOT_DIR}/autoware_ml/configs/t4dataset"
OUTPUT_DIR="/home/taiga/ml_lake/t4-dataset"
COMMON_ARGS=(--project-id prd_jt --output "${OUTPUT_DIR}" --num-workers 1)

uv run --preview-features extra-build-dependencies "${DOWNLOADER}" \
    "${CONFIG_ROOT}/db_jpntaxigen2_v1.yaml" \
    "${COMMON_ARGS[@]}"

uv run --preview-features extra-build-dependencies "${DOWNLOADER}" \
    "${CONFIG_ROOT}/db_jpntaxigen2_v2.yaml" \
    "${COMMON_ARGS[@]}"

uv run --preview-features extra-build-dependencies "${DOWNLOADER}" \
    "${CONFIG_ROOT}/db_jpntaxi_v1.yaml" \
    "${COMMON_ARGS[@]}"

uv run --preview-features extra-build-dependencies "${DOWNLOADER}" \
    "${CONFIG_ROOT}/db_jpntaxi_v2.yaml" \
    "${COMMON_ARGS[@]}"

uv run --preview-features extra-build-dependencies "${DOWNLOADER}" \
    "${CONFIG_ROOT}/db_jpntaxi_v1.yaml" \
    "${COMMON_ARGS[@]}"
