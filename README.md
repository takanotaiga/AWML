# 実行手順

## 0. GPU推論用セットアップ（初回のみ）
`/home/taiga/AWML` で実行:

```bash
UV_CACHE_DIR=/tmp/uv_cache uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1+cu128"

UV_CACHE_DIR=/tmp/uv_cache MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 \
  uv pip install --python .venv/bin/python \
  --reinstall-package mmcv --no-binary mmcv --no-build-isolation "mmcv==2.1.0"
```

## 1. CenterPoint 推論（local smoke）
`/home/taiga/AWML/tools/detection3d` で実行:

```bash
./run_centerpoint_local_smoke.sh
```

`mmcv._ext` が見つからない場合、スクリプト内で `mmcv` を CUDA ops 付きで自動再ビルドします。

CPUで動かす場合:

```bash
CENTERPOINT_DEVICE=cpu ./run_centerpoint_local_smoke.sh
```

## 2. 可視化（先頭3フレーム）
`/home/taiga/AWML` で実行:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=/home/taiga/AWML uv run --offline --no-sync \
  python tools/detection3d/visualize_bboxes.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py \
    /home/taiga/AWML/work_dirs/checkpoints/centerpoint_t4base_v2.1.0_best.pth \
    --device gpu \
    --data-root /home/taiga/ml_lake/t4-dataset \
    --ann-file-path info/local_smoke/t4dataset_local_smoke_infos_test.pkl \
    --frame-range 0 2 \
    --work-dir work_dirs/centerpoint/local_smoke_visualize \
    --max_workers 2
```
