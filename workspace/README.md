## What is this?

Minimal CenterPoint inference wrapper that:
- reads a LiDAR point cloud from CSV (`x,y,z,intensity[,ring_id]`),
- runs CenterPoint (config + checkpoint from this repo),
- emits detections as JSON.

All code stays inside ` and uses `uv` for dependency management.

## CSV format

- Header is optional. If present, the parser looks for: `x`, `y`, `z`, `intensity` (or `i`), and `ring_id` (or `ring`).
- Without a header, the assumed order is: `x, y, z, intensity, ring_id`.
- Missing `intensity` / `ring_id` will be filled with `0.0`.
- Values are interpreted as meters and raw intensity.

Example (`points.csv`):
```csv
x,y,z,intensity,ring_id
1.2,0.5,0.0,12.0,3
2.0,-1.0,0.1,20.0,7
```

## Install with uv

```bash
uv sync
```

This installs `torch`, `mmdet3d==1.4.0`, `mmcv==2.1.0`, `mmengine==0.10.7`, `numpy>=1.24`, and `onnxruntime`.

## Run inference (ONNX two-file deployment)

```bash
uv run python -m main onnx \
  --onnx-voxel /path/to/pts_voxel_encoder_centerpoint.onnx \
  --onnx-head /path/to/pts_backbone_neck_head_centerpoint.onnx \
  --input-csv /path/to/points.csv \
  --output /path/to/predictions.json \
  --score-threshold 0.2
```

Defaults match CenterPoint T4 base v2.x deployment:
- `voxel_size`: `[0.24, 0.24, 8.0]`
- `point_cloud_range`: `[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]`
- `max_points_per_voxel`: `32`, `max_voxels`: `96000`
- `out_size_factor`: `2`, `grid_size`: `1020 1020`
- `class_names`: `car truck bus bicycle pedestrian`
- `--y-axis-reference`: set this if ONNX was exported with y-axis clockwise rotation (Autoware deploy option). It swaps L/W and flips yaw like the bbox coder.

Batch all CSVs in a directory (model is loaded once and reused):

```bash
uv run python -m main onnx \
  --onnx-voxel /path/to/pts_voxel_encoder_centerpoint.onnx \
  --onnx-head /path/to/pts_backbone_neck_head_centerpoint.onnx \
  --input-dir ./output_csv \
  --output-dir ./predictions \
  --score-threshold 0.2
```

`--input-csv` can also point to a directory; outputs default to `<input-dir>/predictions` if `--output-dir` is omitted.

## Run inference (PyTorch checkpoint, optional)

```bash
uv run python -m main torch \
  --config ../projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_base_amp.py \
  --checkpoint /path/to/centerpoint_checkpoint.pth \
  --input-csv /path/to/points.csv \
  --output /path/to/predictions.json \
  --device cpu \
  --score-threshold 0.2
```

Notes:
- `--device` can be `cpu` if a GPU is unavailable (expect slower runtime).
- The config/ckpt must match (e.g., CenterPoint v2.1.0 T4 base).
- `--score-threshold` filters low-confidence boxes after model inference.

## Convert LiDAR .pcd.bin -> CSV

`lidar_t4dataset` の fused LiDAR（`dataset/data/LIDAR_CONCAT/*.pcd.bin`）をそのまま CSV に変換します。

```bash
uv run python -m convert_lidar_to_csv \
  --input-dir ./data \
  --output-dir output_csv
```

- 入力は float32 5列を想定（x, y, z, intensity, ring_id の順）。フィルタ処理なしでそのまま書き出します。
- 出力CSVヘッダ: `x,y,z,intensity,ring_id`

## Visualize predictions (matplotlib)

```bash
uv run python -m visualize_predictions \
  --input-csv ./output_csv/00000.pcd.csv \
  --pred-json ./predictions.json \
  --output vis_bev.png \
  --score-threshold 0.2
```

- BEVに点群 (x,y) を散布し、検出した3D bboxを赤枠＋ラベル/スコア付きで描画します。

Directory + video export (frames are `frame_00000.png` and stitched to mp4 via ffmpeg):

```bash
uv run python -m visualize_predictions \
  --input-dir ./output_csv \
  --pred-dir ./predictions \
  --output-dir ./vis_frames \
  --video ./vis_frames/visualization.mp4 \
  --score-threshold 0.2 \
  --xlim -130 130 \
  --ylim -130 130
```

If `--video` is omitted while processing multiple CSVs, an mp4 is still produced at `<output-dir>/visualization.mp4`. Single-file mode remains available via `--input-csv` + `--pred-json` (optionally `--output`).

`--xlim/--ylim` lock the plot window to a fixed BEV range (defaults to +-130 m) so frames stay aligned and readable.

## Notes on ONNX input features

- スクリプトは ONNX voxel encoder の入力チャネル数と CSV の列数（通常5: x,y,z,intensity,ring_id）から pillar 特徴量を構築します。
- 入力11chを期待する場合は 5生特徴 + cluster/center(6) と判断し、ring_idを保持します。10chを期待する場合は 4生特徴として ring_id を無視し、距離チャンネルなしで 4+6=10 と合わせます。

## JSON output shape

```json
{
  "detections": [
    {
      "translation": {"x": 1.0, "y": 2.0, "z": 0.3},
      "size": {"x": 4.1, "y": 1.8, "z": 1.6},
      "yaw": -0.12,
      "velocity": {"x": 0.0, "y": 0.0},  // only if the model outputs velocity
      "score": 0.71,
      "label": 0,
      "label_name": "car"
    }
  ],
  "metadata": {
    "num_points": 12345,
    "has_velocity": true,
    "point_columns": ["x", "y", "z", "intensity", "ring_id"]
  }
}
```

`label_name` is included if the loaded config exposes class names; otherwise only `label` (integer) is set.
