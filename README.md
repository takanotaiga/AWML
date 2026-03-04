# 実行手順

## 0. GPU推論用セットアップ（初回のみ）
```bash
UV_CACHE_DIR=/tmp/uv_cache MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 \
  uv sync --frozen \
  --no-binary-package mmcv \
  --no-build-isolation-package mmcv \
  --reinstall-package mmcv
```

## 1. CenterPoint 推論（local smoke）
```bash
./run_centerpoint_local_smoke.sh
```

## 2. 可視化（先頭3フレーム）
```bash
./run_centerpoint_local_smoke_visualize.sh
```

