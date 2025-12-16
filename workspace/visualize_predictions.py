from __future__ import annotations

import argparse
import json
from math import cos, sin
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def load_points(csv_path: Path) -> np.ndarray:
    """Load points from CSV produced by convert_lidar_to_csv."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, :3]  # x, y, z


def load_predictions(json_path: Path) -> List[Dict]:
    with json_path.open() as fp:
        obj = json.load(fp)
    return obj.get("detections", [])


def _box_corners(cx: float, cy: float, w: float, l: float, yaw: float) -> np.ndarray:
    """Return 4 corners (x,y) of a BEV box given center, width (y), length (x), yaw."""
    # Our decode uses size.x -> length (forward/back), size.y -> width (left/right)
    # corners in box frame
    x_half, y_half = l / 2.0, w / 2.0
    corners = np.array(
        [
            [x_half, y_half],
            [x_half, -y_half],
            [-x_half, -y_half],
            [-x_half, y_half],
        ]
    )
    c, s = cos(yaw), sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([cx, cy])


def plot_bev(points: np.ndarray, detections: List[Dict], score_threshold: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(25, 25))
    ax.scatter(points[:, 0], points[:, 1], s=0.01, c="gray", alpha=1.0, label="points")

    for det in detections:
        if det.get("score", 0.0) < score_threshold:
            continue
        cx = det["translation"]["x"]
        cy = det["translation"]["y"]
        l = det["size"]["x"]
        w = det["size"]["y"]
        yaw = det.get("yaw", 0.0)
        label = det.get("label_name", str(det.get("label", "?")))
        score = det.get("score", 0.0)
        corners = _box_corners(cx, cy, w=w, l=l, yaw=yaw)
        poly = patches.Polygon(corners, closed=True, fill=False, edgecolor="red", linewidth=1.0)
        ax.add_patch(poly)
        ax.text(cx, cy, f"{label}:{score:.2f}", fontsize=6, color="red")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title("CenterPoint BEV predictions")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved BEV visualization to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CenterPoint predictions (BEV) with matplotlib.")
    parser.add_argument("--input-csv", required=True, type=Path, help="CSV with point cloud (x,y,z,intensity,ring_id).")
    parser.add_argument("--pred-json", required=True, type=Path, help="Predictions JSON from main.py.")
    parser.add_argument("--output", type=Path, default=Path("vis_bev.png"), help="Output PNG path.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Filter boxes below this score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.input_csv)
    detections = load_predictions(args.pred_json)
    plot_bev(points, detections, args.score_threshold, args.output)


if __name__ == "__main__":
    main()
