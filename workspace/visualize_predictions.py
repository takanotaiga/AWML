from __future__ import annotations

import argparse
import json
import subprocess
from math import cos, sin
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def plot_bev(
    points: np.ndarray,
    detections: List[Dict],
    score_threshold: float,
    out_path: Path,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
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
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(title or "CenterPoint BEV predictions")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved BEV visualization to {out_path}")


def _collect_pairs(
    input_csv: Optional[Path],
    pred_json: Optional[Path],
    input_dir: Optional[Path],
    pred_dir: Optional[Path],
) -> List[Tuple[Path, Path]]:
    input_dir = input_dir or (input_csv if input_csv and input_csv.is_dir() else None)
    pred_dir = pred_dir or (pred_json if pred_json and pred_json.is_dir() else None)

    if input_dir or pred_dir:
        if not input_dir or not pred_dir:
            raise ValueError("When using directories, specify both --input-dir and --pred-dir.")
        if not input_dir.is_dir():
            raise FileNotFoundError(f"{input_dir} is not a directory.")
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"{pred_dir} is not a directory.")

        csv_paths = sorted(p for p in input_dir.glob("*.csv") if p.is_file())
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in {input_dir}.")

        pred_map = {p.stem: p for p in pred_dir.glob("*.json")}
        missing = [p.stem for p in csv_paths if p.stem not in pred_map]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise FileNotFoundError(f"Missing prediction JSON for: {missing_str}")

        return [(csv_path, pred_map[csv_path.stem]) for csv_path in csv_paths]

    if not input_csv or not pred_json:
        raise ValueError("Provide --input-csv and --pred-json, or use --input-dir with --pred-dir.")
    if not input_csv.exists():
        raise FileNotFoundError(f"{input_csv} does not exist.")
    if not pred_json.exists():
        raise FileNotFoundError(f"{pred_json} does not exist.")
    return [(input_csv, pred_json)]


def _render_video(frame_dir: Path, fps: int, video_output: Path) -> None:
    pattern = frame_dir / "frame_%05d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_output),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to create a video but was not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to create the video: {stderr}") from exc
    print(f"Saved video to {video_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CenterPoint predictions (BEV) with matplotlib.")
    parser.add_argument("--input-csv", type=Path, help="CSV with point cloud (x,y,z,intensity,ring_id).")
    parser.add_argument("--pred-json", type=Path, help="Predictions JSON from main.py.")
    parser.add_argument("--input-dir", type=Path, help="Directory of CSVs to visualize (processes all *.csv).")
    parser.add_argument("--pred-dir", type=Path, help="Directory containing prediction JSONs (one per CSV stem).")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path for single input.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write visualization frames when processing multiple CSVs.",
    )
    parser.add_argument("--video", type=Path, default=None, help="Optional mp4 path to stitch frames into a video.")
    parser.add_argument("--fps", type=int, default=10, help="Video framerate when exporting mp4.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Filter boxes below this score.")
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=[-130.0, 130.0],
        help="Fix x-axis range [min max] to keep frames aligned (defaults to +-130m).",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=[-130.0, 130.0],
        help="Fix y-axis range [min max] to keep frames aligned (defaults to +-130m).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = _collect_pairs(args.input_csv, args.pred_json, args.input_dir, args.pred_dir)
    multiple = len(pairs) > 1

    if multiple and args.output:
        raise ValueError("Use --output-dir (not --output) when visualizing multiple CSVs.")

    if multiple or args.video:
        pred_base_dir = args.pred_dir or pairs[0][1].parent
        frame_dir = args.output_dir or (pred_base_dir / "frames")
        frame_dir.mkdir(parents=True, exist_ok=True)
        video_output = args.video or (frame_dir / "visualization.mp4" if multiple else None)
        for idx, (csv_path, pred_path) in enumerate(pairs):
            frame_path = frame_dir / f"frame_{idx:05d}.png"
            points = load_points(csv_path)
            detections = load_predictions(pred_path)
            plot_bev(
                points,
                detections,
                args.score_threshold,
                frame_path,
                title=csv_path.name,
                xlim=tuple(args.xlim) if args.xlim else None,
                ylim=tuple(args.ylim) if args.ylim else None,
            )
        if video_output:
            _render_video(frame_dir=frame_dir, fps=args.fps, video_output=video_output)
    else:
        csv_path, pred_path = pairs[0]
        output_path = args.output or Path("vis_bev.png")
        points = load_points(csv_path)
        detections = load_predictions(pred_path)
        plot_bev(
            points,
            detections,
            args.score_threshold,
            output_path,
            title=csv_path.name,
            xlim=tuple(args.xlim) if args.xlim else None,
            ylim=tuple(args.ylim) if args.ylim else None,
        )


if __name__ == "__main__":
    main()
