from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

# Ensure repository root is importable (for optional imports)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class PointCloud:
    points: np.ndarray  # shape: (N, 5) -> x, y, z, intensity, ring_id
    columns: Sequence[str]


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_header(row: Sequence[str]) -> Dict[str, int]:
    header = [col.strip().lower() for col in row]
    wanted = {"x", "y", "z", "intensity", "i", "ring", "ring_id"}
    return {name: idx for idx, name in enumerate(header) if name in wanted}


def _coerce_floats(values: Iterable[str]) -> List[float]:
    return [float(v) for v in values]


def load_points_from_csv(csv_path: Path) -> PointCloud:
    """
    Load point cloud from CSV.
    Expected columns: x, y, z, intensity (optional ring_id).
    Missing intensity/ring_id will be filled with zeros.
    """
    with csv_path.open() as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"No data found in {csv_path}")

    header_map = _parse_header(rows[0]) if not all(_is_number(v) for v in rows[0]) else {}
    data_rows = rows[1:] if header_map else rows

    col_order = ["x", "y", "z", "intensity", "ring_id"]
    points: List[List[float]] = []

    for row in data_rows:
        if not row:
            continue

        if header_map:
            x = float(row[header_map["x"]])
            y = float(row[header_map["y"]])
            z = float(row[header_map["z"]])

            intensity_idx = header_map["intensity"] if "intensity" in header_map else header_map["i"] if "i" in header_map else None
            ring_idx = header_map["ring_id"] if "ring_id" in header_map else header_map["ring"] if "ring" in header_map else None

            intensity = float(row[intensity_idx]) if intensity_idx is not None else 0.0
            ring_id = float(row[ring_idx]) if ring_idx is not None else 0.0
        else:
            # Assume order: x, y, z, intensity, (ring_id)
            numeric_row = _coerce_floats(row)
            if len(numeric_row) < 3:
                raise ValueError(f"Row has fewer than 3 values: {row}")
            x, y, z = numeric_row[:3]
            intensity = numeric_row[3] if len(numeric_row) >= 4 else 0.0
            ring_id = numeric_row[4] if len(numeric_row) >= 5 else 0.0

        points.append([x, y, z, intensity, ring_id])

    if not points:
        raise ValueError(f"No valid rows parsed from {csv_path}")

    return PointCloud(points=np.asarray(points, dtype=np.float32), columns=col_order)


def _run_single_inference(
    model,
    points: np.ndarray,
    score_threshold: float,
    class_names: Optional[Sequence[str]],
    point_columns: Sequence[str],
) -> Dict:
    # inference_detector expects a file path. Write a temporary .bin file.
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        points.astype(np.float32).tofile(tmp.name)
        result = inference_detector(model, tmp.name)

    if isinstance(result, tuple):
        data_sample = result[0]
    elif isinstance(result, list):
        data_sample = result[0]
    else:
        data_sample = result
    pred_instances = data_sample.pred_instances_3d
    bboxes = pred_instances.bboxes_3d.tensor.cpu().numpy()
    scores = pred_instances.scores_3d.cpu().numpy()
    labels = pred_instances.labels_3d.cpu().numpy()

    detections = []
    for idx in range(bboxes.shape[0]):
        score = float(scores[idx])
        if score_threshold and score < score_threshold:
            continue

        box = bboxes[idx]
        has_vel = box.shape[0] >= 9
        detection = {
            "translation": {"x": float(box[0]), "y": float(box[1]), "z": float(box[2])},
            "size": {"x": float(box[3]), "y": float(box[4]), "z": float(box[5])},
            "yaw": float(box[6]) if box.shape[0] >= 7 else None,
            "velocity": {"x": float(box[7]), "y": float(box[8])} if has_vel else None,
            "score": score,
            "label": int(labels[idx]),
        }
        if class_names and 0 <= labels[idx] < len(class_names):
            detection["label_name"] = class_names[int(labels[idx])]

        detections.append(detection)

    return {
        "detections": detections,
        "metadata": {
            "num_points": int(points.shape[0]),
            "has_velocity": bboxes.shape[1] >= 9,
            "point_columns": list(point_columns),
        },
    }


def run_inference(
    config_path: Path,
    checkpoint_path: Path,
    csv_path: Path,
    device: str = "cuda:0",
    score_threshold: float = 0.0,
) -> Dict:
    # Lazy import to avoid heavy deps when using ONNX-only mode.
    from mmdet3d.apis import inference_detector, init_model

    point_cloud = load_points_from_csv(csv_path)
    model = init_model(str(config_path), str(checkpoint_path), device=device)
    class_names = model.dataset_meta.get("classes") if hasattr(model, "dataset_meta") else None

    with torch.no_grad():
        return _run_single_inference(
            model=model,
            points=point_cloud.points,
            score_threshold=score_threshold,
            class_names=class_names,
            point_columns=point_cloud.columns,
        )


# ---------------- ONNX path (voxel encoder + backbone_head) ----------------


def voxelize_points(
    points: np.ndarray,
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    max_points_per_voxel: int,
    max_voxels: int,
    base_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimal voxelizer compatible with PointPillarsScatter."""
    pc_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)

    # Filter points inside range
    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] < pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] < pc_range[5])
    )
    points = points[mask]

    if points.shape[0] == 0:
        return (
            np.zeros((0, max_points_per_voxel, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 4), dtype=np.int32),
        )

    # Compute voxel coordinates
    voxel_coords = np.floor((points[:, :3] - pc_range[:3]) / voxel_size).astype(np.int32)
    # coors order: z, y, x
    voxel_coords = voxel_coords[:, [2, 1, 0]]

    voxel_dict: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
    for point, coord in zip(points, voxel_coords):
        key = (coord[0], coord[1], coord[2])
        if key not in voxel_dict:
            if len(voxel_dict) >= max_voxels:
                continue
            voxel_dict[key] = []
        if len(voxel_dict[key]) < max_points_per_voxel:
            voxel_dict[key].append(point[:base_dim])

    num_voxels = len(voxel_dict)
    voxels = np.zeros((num_voxels, max_points_per_voxel, base_dim), dtype=np.float32)
    num_points_per_voxel = np.zeros((num_voxels,), dtype=np.int32)
    coors = np.zeros((num_voxels, 4), dtype=np.int32)  # [batch, z, y, x]

    for idx, (key, pts) in enumerate(voxel_dict.items()):
        pts_arr = np.asarray(pts, dtype=np.float32)
        voxels[idx, : pts_arr.shape[0], :] = pts_arr
        num_points_per_voxel[idx] = pts_arr.shape[0]
        coors[idx] = np.array([0, key[0], key[1], key[2]], dtype=np.int32)

    return voxels, num_points_per_voxel, coors


def build_pillar_features(
    voxels: np.ndarray,
    num_points: np.ndarray,
    coors: np.ndarray,
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    include_distance: bool = False,
    base_dim: int = 4,
) -> np.ndarray:
    """
    Replicate PillarFeatureNet.get_input_features:
    concat raw xyz/i (4) + f_cluster(3) + f_center(3) -> 10 dims.
    """
    vx, vy, vz = voxel_size
    x_offset = vx / 2.0 + point_cloud_range[0]
    y_offset = vy / 2.0 + point_cloud_range[1]
    z_offset = vz / 2.0 + point_cloud_range[2]

    max_points = voxels.shape[1]
    mask = (np.arange(max_points)[None, :] < num_points[:, None]).astype(np.bool_)

    pts = voxels[:, :, :base_dim].copy()
    # cluster center
    pts_sum = pts[:, :, :3].sum(axis=1, keepdims=True)
    pts_mean = pts_sum / np.maximum(num_points[:, None, None], 1e-6)
    f_cluster = pts[:, :, :3] - pts_mean

    # voxel center
    f_center = np.zeros_like(pts[:, :, :3])
    f_center[:, :, 0] = pts[:, :, 0] - (coors[:, 3][:, None] * vx + x_offset)
    f_center[:, :, 1] = pts[:, :, 1] - (coors[:, 2][:, None] * vy + y_offset)
    f_center[:, :, 2] = pts[:, :, 2] - (coors[:, 1][:, None] * vz + z_offset)

    features_ls = [pts, f_cluster, f_center]

    if include_distance:
        points_dist = np.linalg.norm(pts[:, :, :3], axis=2, keepdims=True)
        features_ls.append(points_dist)

    features = np.concatenate(features_ls, axis=-1)
    features = features * mask[..., None]
    return features.astype(np.float32)


def scatter_pillars(
    voxel_features: np.ndarray,
    coors: np.ndarray,
    output_shape: Tuple[int, int],
    num_channels: int,
) -> np.ndarray:
    """Scatter pillar features to a BEV pseudo-image."""
    batch_size = 1
    ny, nx = output_shape  # y, x order
    canvas = np.zeros((batch_size, num_channels, ny, nx), dtype=np.float32)

    for i in range(coors.shape[0]):
        b, z, y, x = coors[i]
        if b >= batch_size:
            continue
        canvas[b, :, y, x] = voxel_features[i]
    return canvas


def _nms_heatmap(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(heat: torch.Tensor, K: int = 100):
    B, C, H, W = heat.shape
    heat = heat.view(B, C, -1)
    topk_scores_per_class, topk_inds_per_class = torch.topk(heat, K)
    topk_inds = topk_inds_per_class % (H * W)
    topk_ys = (topk_inds // W).int().float()
    topk_xs = (topk_inds % W).int().float()

    topk_scores, topk_inds2 = torch.topk(topk_scores_per_class.view(B, -1), K)
    topk_clses = (topk_inds2 // K).int()
    topk_inds = topk_inds.view(B, -1, 1).gather(1, topk_inds2.view(B, -1, 1)).squeeze(2)
    topk_ys = topk_ys.view(B, -1, 1).gather(1, topk_inds2.view(B, -1, 1)).squeeze(2)
    topk_xs = topk_xs.view(B, -1, 1).gather(1, topk_inds2.view(B, -1, 1)).squeeze(2)

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def decode_centerpoint_outputs(
    outputs: List[np.ndarray],
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    out_size_factor: int,
    score_threshold: float,
    class_names: Optional[Sequence[str]],
    y_axis_reference: bool = False,
    topk: int = 100,
) -> List[Dict]:
    heatmap, reg, height, dim, rot, vel = [torch.from_numpy(o) for o in outputs]
    heatmap = torch.sigmoid(heatmap)
    heatmap = _nms_heatmap(heatmap, kernel=3)

    scores, inds, clses, ys, xs = _topk(heatmap, K=topk)

    reg = reg.permute(0, 2, 3, 1).contiguous().view(reg.size(0), -1, 2)
    height = height.permute(0, 2, 3, 1).contiguous().view(height.size(0), -1, 1)
    dim = dim.permute(0, 2, 3, 1).contiguous().view(dim.size(0), -1, 3)
    rot = rot.permute(0, 2, 3, 1).contiguous().view(rot.size(0), -1, 2)
    vel = vel.permute(0, 2, 3, 1).contiguous().view(vel.size(0), -1, 2)

    batch_dets = []
    for b in range(heatmap.size(0)):
        batch_inds = inds[b]
        xs_b = xs[b].unsqueeze(-1) + reg[b, batch_inds, 0:1]
        ys_b = ys[b].unsqueeze(-1) + reg[b, batch_inds, 1:2]

        rot_b = torch.atan2(rot[b, batch_inds, 0:1], rot[b, batch_inds, 1:2])
        dim_b = torch.exp(dim[b, batch_inds])
        height_b = height[b, batch_inds]
        vel_b = vel[b, batch_inds]

        xs_b = xs_b * out_size_factor * voxel_size[0] + point_cloud_range[0]
        ys_b = ys_b * out_size_factor * voxel_size[1] + point_cloud_range[1]

        scores_b = scores[b]
        clses_b = clses[b]

        # Post-center range filter (match config)
        pc_min = torch.tensor([-200.0, -200.0, -10.0], device=xs_b.device)
        pc_max = torch.tensor([200.0, 200.0, 10.0], device=xs_b.device)

        dets = []
        for i in range(scores_b.shape[0]):
            score = float(scores_b[i])
            if score < score_threshold:
                continue
            cx = float(xs_b[i])
            cy = float(ys_b[i])
            cz = float(height_b[i])
            if not (pc_min[0] <= cx <= pc_max[0] and pc_min[1] <= cy <= pc_max[1] and pc_min[2] <= cz <= pc_max[2]):
                continue

            cls_id = int(clses_b[i])
            yaw_val = float(rot_b[i])
            size_x = float(dim_b[i, 1])  # length
            size_y = float(dim_b[i, 0])  # width
            size_z = float(dim_b[i, 2])

            if y_axis_reference:
                # Match CenterPointBBoxCoder.decode conversion when y-axis clockwise is used.
                yaw_val = -yaw_val - math.pi / 2.0
                size_x, size_y = size_y, size_x

            det = {
                "translation": {"x": cx, "y": cy, "z": cz},
                "size": {"x": size_x, "y": size_y, "z": size_z},
                "yaw": yaw_val,
                "velocity": {"x": float(vel_b[i, 0]), "y": float(vel_b[i, 1])},
                "score": score,
                "label": cls_id,
            }
            if class_names and 0 <= cls_id < len(class_names):
                det["label_name"] = class_names[cls_id]
            dets.append(det)
        batch_dets.extend(dets)

    return batch_dets


def run_inference_onnx(
    onnx_voxel: Path,
    onnx_head: Path,
    csv_path: Path,
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    max_points_per_voxel: int,
    max_voxels: int,
    out_size_factor: int,
    score_threshold: float,
    class_names: Optional[Sequence[str]],
    output_shape: Tuple[int, int],
    y_axis_reference: bool,
) -> Dict:
    point_cloud = load_points_from_csv(csv_path)
    sess_voxel = ort.InferenceSession(str(onnx_voxel))
    expected_dim = sess_voxel.get_inputs()[0].shape[-1]

    # Determine base_dim (raw point feature dim) and whether to include distance
    points_dim = point_cloud.points.shape[1]
    candidate_base_dims = [points_dim, max(1, points_dim - 1)]
    base_dim = None
    include_distance = False

    for base in candidate_base_dims:
        # Try without distance
        if base + 6 == expected_dim:
            base_dim = base
            include_distance = False
            break
        # Try with distance
        if base + 7 == expected_dim:
            base_dim = base
            include_distance = True
            break

    if base_dim is None:
        raise ValueError(
            f"Cannot reconcile ONNX input dim {expected_dim} with point dims {points_dim}. "
            "Try adjusting --grid-size/voxel params or ensure CSV columns match model expectations."
        )

    voxels, num_points, coors = voxelize_points(
        points=point_cloud.points,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
        base_dim=base_dim,
    )

    input_features = build_pillar_features(
        voxels=voxels,
        num_points=num_points,
        coors=coors,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        include_distance=include_distance,
        base_dim=base_dim,
    )

    voxel_inputs = {sess_voxel.get_inputs()[0].name: input_features}
    voxel_features = sess_voxel.run(None, voxel_inputs)[0]
    if voxel_features.ndim == 3:  # (num_voxels, 1, C) -> squeeze
        voxel_features = voxel_features.squeeze(1)

    spatial_features = scatter_pillars(
        voxel_features=voxel_features,
        coors=coors,
        output_shape=output_shape,
        num_channels=voxel_features.shape[1],
    )

    sess_head = ort.InferenceSession(str(onnx_head))
    head_inputs = {sess_head.get_inputs()[0].name: spatial_features}
    head_outputs = sess_head.run(None, head_inputs)

    detections = decode_centerpoint_outputs(
        outputs=head_outputs,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=out_size_factor,
        score_threshold=score_threshold,
        class_names=class_names,
        topk=100,
        y_axis_reference=y_axis_reference,
    )

    return {
        "detections": detections,
        "metadata": {
            "num_points": int(point_cloud.points.shape[0]),
            "has_velocity": True,
            "point_columns": list(point_cloud.columns),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CenterPoint inference from CSV point cloud.")
    parser.add_argument("--input-csv", required=True, type=Path, help="Path to input point cloud CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Path to save JSON output. Prints to stdout if omitted.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Filter detections below this score.")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    torch_parser = subparsers.add_parser("torch", help="Use PyTorch checkpoint.")
    torch_parser.add_argument("--config", required=True, type=Path, help="Path to CenterPoint config file.")
    torch_parser.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint (.pth).")
    torch_parser.add_argument("--device", default="cuda:0", help="Device string, e.g., cuda:0 or cpu.")

    onnx_parser = subparsers.add_parser("onnx", help="Use ONNX pair (voxel encoder + backbone/head).")
    onnx_parser.add_argument("--onnx-voxel", required=True, type=Path, help="Path to pts_voxel_encoder_centerpoint.onnx")
    onnx_parser.add_argument("--onnx-head", required=True, type=Path, help="Path to pts_backbone_neck_head_centerpoint.onnx")
    onnx_parser.add_argument(
        "--voxel-size", nargs=3, type=float, default=[0.24, 0.24, 8.0], help="Voxel size [x, y, z]."
    )
    onnx_parser.add_argument(
        "--point-cloud-range",
        nargs=6,
        type=float,
        default=[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0],
        help="Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].",
    )
    onnx_parser.add_argument("--max-points-per-voxel", type=int, default=32, help="Max points per voxel.")
    onnx_parser.add_argument("--max-voxels", type=int, default=96000, help="Max voxels.")
    onnx_parser.add_argument("--out-size-factor", type=int, default=2, help="Downsample factor (stride).")
    onnx_parser.add_argument(
        "--grid-size",
        nargs=2,
        type=int,
        default=[1020, 1020],
        help="Output BEV grid size (ny, nx) used by PointPillarsScatter.",
    )
    onnx_parser.add_argument(
        "--class-names",
        nargs="*",
        default=["car", "truck", "bus", "bicycle", "pedestrian"],
        help="Class names for decoding.",
    )
    onnx_parser.add_argument(
        "--y-axis-reference",
        action="store_true",
        help="Set if ONNX was exported with y-axis clockwise rotation reference (swap l/w and flip yaw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "torch":
        result = run_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            csv_path=args.input_csv,
            device=args.device,
            score_threshold=args.score_threshold,
        )
    else:
        result = run_inference_onnx(
            onnx_voxel=args.onnx_voxel,
            onnx_head=args.onnx_head,
            csv_path=args.input_csv,
            voxel_size=args.voxel_size,
            point_cloud_range=args.point_cloud_range,
            max_points_per_voxel=args.max_points_per_voxel,
            max_voxels=args.max_voxels,
            out_size_factor=args.out_size_factor,
            score_threshold=args.score_threshold,
            class_names=args.class_names,
            output_shape=(args.grid_size[0], args.grid_size[1]),
            y_axis_reference=args.y_axis_reference,
        )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
