from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def list_bin_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix == ".bin")


def load_pointcloud(bin_path: Path) -> np.ndarray:
    """
    Load fused lidar binary (.pcd.bin) assumed to be float32 and shaped (-1, 5).
    Columns: x, y, z, intensity, ring_id (fifth column kept as-is).
    """
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % 5 != 0:
        raise ValueError(f"{bin_path} size {arr.size} is not divisible by 5; unexpected format.")
    return arr.reshape(-1, 5)


def save_csv(points: np.ndarray, csv_path: Path) -> None:
    header = "x,y,z,intensity,ring_id"
    np.savetxt(csv_path, points, fmt="%.6f", delimiter=",", header=header, comments="")


def convert_dir(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_files = list_bin_files(input_dir)
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found under {input_dir}")

    for bin_path in bin_files:
        points = load_pointcloud(bin_path)
        csv_name = bin_path.with_suffix(".csv").name
        csv_path = output_dir / csv_name
        save_csv(points, csv_path)
        print(f"Wrote {csv_path} (rows={points.shape[0]})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LiDAR .pcd.bin files to CSV.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/Users/taiga/Desktop/lidar_t4dataset/dataset/data/LIDAR_CONCAT"),
        help="Directory containing fused LiDAR .pcd.bin files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_csv"),
        help="Directory to write CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dir(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
