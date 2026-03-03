# 実行手順

## 1. CenterPoint 推論（local smoke）
`/home/taiga/AWML/tools/detection3d` で実行:

```bash
./run_centerpoint_local_smoke.sh
```

## 2. 可視化（先頭3フレーム）
`/home/taiga/AWML` で実行:

```bash
UV_CACHE_DIR=/tmp/uv_cache PYTHONPATH=/home/taiga/AWML uv run --offline \
  python tools/detection3d/visualize_bboxes.py \
    projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py \
    /home/taiga/AWML/work_dirs/checkpoints/centerpoint_t4base_v2.1.0_best.pth \
    --device cpu \
    --data-root /home/taiga/ml_lake/t4-dataset \
    --ann-file-path info/local_smoke/t4dataset_local_smoke_infos_test.pkl \
    --frame-range 0 2 \
    --work-dir work_dirs/centerpoint/local_smoke_visualize \
    --max_workers 2
```
