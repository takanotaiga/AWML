#!/usr/bin/env python3
"""Evaluate CenterPoint mAP/NDS for real vs generated intensity conditions.

This script compares four inference conditions over the same frames:
  - real:    real intensity pointclouds (gt_*.pcd.bin)
  - ml:      ML-generated intensity pointclouds (pred_*.pcd.bin)
  - zero:    real/ml source + zero-filled intensity transform
  - range:   real/ml source + range-based intensity transform

Dataset GT annotations are not used. This script performs direct pointcloud
inference and computes custom metrics against pseudo GT made from real-intensity
inference results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mmengine
import numpy as np
import yaml
from mmengine.config import Config
from tools.detection3d.t4dataset_converters.update_infos_to_v2 import get_empty_standard_data_info


MODEL_PRESETS: Dict[str, Dict[str, str]] = {
    "j6gen2_v2.5.1": {
        "model_config": "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py",
        "checkpoint_relpath": "work_dirs/checkpoints/centerpoint_j6gen2_v2.5.1_best.pth",
        "checkpoint_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
        "checkpoint_logs_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.5.1/logs.zip",
        "version_tag": "v2.5.1",
        "checkpoint_candidates": [
            "best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
            "best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
            "checkpoint_best.pth",
        ],
    },
    "j6gen2_v2.4.1": {
        "model_config": "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py",
        "checkpoint_relpath": "work_dirs/checkpoints/centerpoint_j6gen2_v2.4.1_best.pth",
        "checkpoint_logs_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.4.1/logs.zip",
        "version_tag": "v2.4.1",
        "checkpoint_candidates": [
            "best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
            "best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
            "checkpoint_best.pth",
        ],
        # The doc currently links some 2.4.1 artifacts to v2.3.1 paths.
        # Keep alias as fallback for download attempts.
        "version_alias_tags": ["v2.3.1"],
    },
    "j6gen2_v2.3.1": {
        "model_config": "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py",
        "checkpoint_relpath": "work_dirs/checkpoints/centerpoint_j6gen2_v2.3.1_best.pth",
        "checkpoint_logs_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.3.1/logs.zip",
        "version_tag": "v2.3.1",
        "checkpoint_candidates": [
            "best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
            "best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
            "checkpoint_best.pth",
        ],
    },
    "j6gen2_v2.2.1": {
        "model_config": "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py",
        "checkpoint_relpath": "work_dirs/checkpoints/centerpoint_j6gen2_v2.2.1_best.pth",
        "checkpoint_logs_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.2.1/logs.zip",
        "version_tag": "v2.2.1",
        "checkpoint_candidates": [
            "best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
            "best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
            "checkpoint_best.pth",
        ],
    },
    "j6gen2_v2.1.1": {
        "model_config": "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py",
        "checkpoint_relpath": "work_dirs/checkpoints/centerpoint_j6gen2_v2.1.1_best.pth",
        "checkpoint_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
        "checkpoint_logs_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.1.1/logs.zip",
        "version_tag": "v2.1.1",
        "checkpoint_candidates": [
            "best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
            "best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
            "checkpoint_best.pth",
        ],
    },
    "j6gen2_v2.0.1": {
        "model_config": "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py",
        "checkpoint_relpath": "work_dirs/checkpoints/centerpoint_j6gen2_v2.0.1_best.pth",
        "checkpoint_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
        "checkpoint_logs_url": "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/j6gen2/v2.0.1/logs.zip",
        "version_tag": "v2.0.1",
        "checkpoint_candidates": [
            "best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
            "best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
            "checkpoint_best.pth",
        ],
    }
}

DEFAULT_CENTERPOINT_INFER_CFG = (
    "projects/CenterPoint/configs/t4dataset/Centerpoint/second_secfpn_4xb16_121m_local_smoke_infer.py"
)


@dataclass(frozen=True)
class ManifestSample:
    export_index: int
    source_path: Path
    source_rel_path: str
    dataset_version: str
    scene_id: str
    scene_version: str


@dataclass(frozen=True)
class GroupInfo:
    name: str
    sample_indices: Tuple[int, ...]


@dataclass
class DetectionFrame:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CenterPoint metrics (mAP/NDS) across real/ml/zero/range intensity conditions."
    )
    parser.add_argument(
        "--inference-dir",
        action="append",
        default=[],
        help="Inference directory containing manifest.json, gt/, pred/. Repeatable. "
        "If omitted, scans work_dirs/inference_*.",
    )
    parser.add_argument(
        "--work-dirs-root",
        default="work_dirs",
        help="Root used for auto-detecting inference_* dirs when --inference-dir is omitted.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/taiga/ml_lake/t4-dataset",
        help="T4 dataset root path used to build GT info files.",
    )
    parser.add_argument(
        "--output-root",
        default="work_dirs/centerpoint/intensity_comparison",
        help="Output root for temporary files, runtime configs, and summaries.",
    )
    parser.add_argument(
        "--dataset-config",
        default="autoware_ml/configs/detection3d/dataset/t4dataset/j6gen2_base.py",
        help="Base dataset config used by create_data_t4dataset.py.",
    )
    parser.add_argument(
        "--centerpoint-model-version",
        default="j6gen2_v2.5.1",
        help="CenterPoint model preset key (e.g. j6gen2_v2.5.1).",
    )
    parser.add_argument(
        "--list-model-versions",
        action="store_true",
        help="List known --centerpoint-model-version presets and exit.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint path. If omitted, resolved from --centerpoint-model-version preset.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="CenterPoint config path. If omitted, resolved from --centerpoint-model-version preset.",
    )
    parser.add_argument(
        "--centerpoint-device",
        choices=["cuda", "cpu"],
        default=os.environ.get("CENTERPOINT_DEVICE", "cuda"),
        help="Inference device mode.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs for distributed inference. 0 means auto-detect all visible GPUs.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Base master port for torch.distributed.run.",
    )
    parser.add_argument(
        "--range-attenuation",
        type=float,
        default=0.02,
        help="Attenuation coefficient for range-based intensity mode.",
    )
    parser.add_argument(
        "--zero-range-source",
        choices=["real", "ml"],
        default="real",
        help="Which source pointcloud to use before applying zero/range transforms.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=1,
        help="max_sweeps for create_data_t4dataset.py while preparing GT info.",
    )
    parser.add_argument(
        "--disable-per-dataset-version",
        action="store_true",
        help="If set, evaluates only the 'all' group (no per-source-dataset-version splits).",
    )
    parser.add_argument(
        "--reuse-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached filtered ann files when available. Use --no-reuse-cache to disable.",
    )
    parser.add_argument(
        "--require-ann-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require cached ann metadata and skip prepare generation. Fails if cache is missing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="test_dataloader.num_workers per rank. 0 means auto from CPU cores.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without running create_data/test commands.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full subprocess logs instead of compact progress bar.",
    )
    parser.add_argument(
        "--parallel-scenarios",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run scenario inference (real/ml/zero/range) concurrently across GPUs when possible.",
    )
    parser.add_argument(
        "--evaluation-backend",
        choices=["custom", "t4metric"],
        default="custom",
        help="Metric backend. 'custom' performs direct pseudo-GT comparison without T4Metric.",
    )
    return parser.parse_args()


class ProgressTracker:
    def __init__(self, total_steps: int, log_dir: Path) -> None:
        self.total_steps = max(int(total_steps), 1)
        self.done_steps = 0
        self.bar_width = 28
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.command_seq = 0
        self._last_line_len = 0
        self._lock = threading.Lock()

    def _build_bar(self, done_steps: int) -> str:
        filled = int(self.bar_width * done_steps / self.total_steps)
        return "#" * filled + "-" * (self.bar_width - filled)

    def _print_line(self, text: str) -> None:
        pad = max(0, self._last_line_len - len(text))
        sys.stdout.write("\r" + text + (" " * pad))
        sys.stdout.flush()
        self._last_line_len = len(text)

    def _clear_line(self) -> None:
        if self._last_line_len:
            sys.stdout.write("\r" + (" " * self._last_line_len) + "\r")
            sys.stdout.flush()
            self._last_line_len = 0

    def next_log_path(self, label: str) -> Path:
        with self._lock:
            self.command_seq += 1
            file_name = f"{self.command_seq:04d}_{sanitize_name(label)[:120]}.log"
            return self.log_dir / file_name

    def show_running(self, label: str, elapsed_sec: float, spinner: str) -> None:
        with self._lock:
            bar = self._build_bar(self.done_steps)
            text = (
                f"[{bar}] {self.done_steps}/{self.total_steps} {spinner} "
                f"{label[:60]} ({elapsed_sec:6.1f}s)"
            )
            self._print_line(text)

    def complete_step(self, label: str, elapsed_sec: float, log_path: Optional[Path]) -> None:
        with self._lock:
            self.done_steps += 1
            bar = self._build_bar(self.done_steps)
            self._clear_line()
            msg = f"[{bar}] {self.done_steps}/{self.total_steps} OK  {label} ({elapsed_sec:.1f}s)"
            if log_path is not None:
                msg += f"  log={log_path}"
            print(msg)

    def fail_step(self, label: str, elapsed_sec: float, log_path: Optional[Path]) -> None:
        with self._lock:
            self._clear_line()
            msg = (
                f"[{self._build_bar(self.done_steps)}] {self.done_steps}/{self.total_steps} "
                f"NG  {label} ({elapsed_sec:.1f}s)"
            )
            if log_path is not None:
                msg += f"  log={log_path}"
            print(msg)

    def dry_run_step(self, label: str) -> None:
        with self._lock:
            self.done_steps += 1
            bar = self._build_bar(self.done_steps)
            print(f"[{bar}] {self.done_steps}/{self.total_steps} DRY {label}")

    def instant_step(self, label: str) -> None:
        with self._lock:
            self.done_steps += 1
            bar = self._build_bar(self.done_steps)
            print(f"[{bar}] {self.done_steps}/{self.total_steps} SKIP {label}")


def run_cmd(
    cmd: Sequence[str],
    env: Dict[str, str],
    dry_run: bool = False,
    verbose: bool = False,
    tracker: Optional[ProgressTracker] = None,
    label: Optional[str] = None,
    count_step: bool = True,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    cmd_str = " ".join(cmd)
    step_label = label or cmd[0]
    run_env = env.copy()
    if extra_env:
        run_env.update(extra_env)
    if dry_run:
        if tracker is not None and count_step:
            tracker.dry_run_step(step_label)
        print(f"+ {cmd_str}")
        return

    if verbose:
        start = time.time()
        if tracker is not None and count_step:
            print(f"--> {step_label}")
        print(f"+ {cmd_str}")
        subprocess.run(cmd, check=True, env=run_env)
        if tracker is not None and count_step:
            tracker.complete_step(step_label, time.time() - start, log_path=None)
        return

    log_path = tracker.next_log_path(step_label) if tracker is not None else None
    out_file = open(log_path, "w") if log_path is not None else subprocess.DEVNULL
    process = None
    start = time.time()
    spinner_chars = "|/-\\"
    spinner_idx = 0
    try:
        process = subprocess.Popen(
            cmd,
            env=run_env,
            stdout=out_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            ret = process.poll()
            elapsed = time.time() - start
            if tracker is not None and count_step:
                tracker.show_running(step_label, elapsed_sec=elapsed, spinner=spinner_chars[spinner_idx % 4])
            spinner_idx += 1
            if ret is not None:
                break
            time.sleep(0.5)

        elapsed = time.time() - start
        if ret != 0:
            if tracker is not None and count_step:
                tracker.fail_step(step_label, elapsed_sec=elapsed, log_path=log_path)
            raise subprocess.CalledProcessError(ret, cmd)
        if tracker is not None and count_step:
            tracker.complete_step(step_label, elapsed_sec=elapsed, log_path=log_path)
    finally:
        if hasattr(out_file, "close"):
            out_file.close()


def build_uv_env(repo_root: Path, centerpoint_device: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", "/tmp/uv_cache")
    os.makedirs(env["UV_CACHE_DIR"], exist_ok=True)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{current_pythonpath}" if current_pythonpath else str(repo_root)
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if centerpoint_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env


def make_j6gen2_dynamic_preset(model_version: str) -> Dict[str, str]:
    # Expected format: j6gen2_vX.Y.Z
    if not model_version.startswith("j6gen2_"):
        return {}
    version_tag = model_version.split("j6gen2_", 1)[1]
    return {
        "model_config": DEFAULT_CENTERPOINT_INFER_CFG,
        "checkpoint_relpath": f"work_dirs/checkpoints/centerpoint_{model_version}_best.pth",
        "checkpoint_logs_url": (
            "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/"
            f"j6gen2/{version_tag}/logs.zip"
        ),
        "version_tag": version_tag,
        "checkpoint_candidates": [
            "best_NuScenes_metric_T4Metric_mAP_epoch_29.pth",
            "best_NuScenes_metric_T4Metric_mAP_epoch_28.pth",
            "checkpoint_best.pth",
        ],
    }


def build_checkpoint_candidate_urls(preset: Dict[str, str]) -> List[str]:
    urls: List[str] = []
    checkpoint_url = preset.get("checkpoint_url")
    if checkpoint_url:
        urls.append(checkpoint_url)

    version_tag = preset.get("version_tag")
    if version_tag:
        base_dir = (
            "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/"
            f"j6gen2/{version_tag}"
        )
        for filename in preset.get("checkpoint_candidates", []):
            urls.append(f"{base_dir}/{filename}")

    for alias_tag in preset.get("version_alias_tags", []):
        base_dir = (
            "https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint/"
            f"j6gen2/{alias_tag}"
        )
        for filename in preset.get("checkpoint_candidates", []):
            urls.append(f"{base_dir}/{filename}")
        urls.append(f"{base_dir}/best_NuScenes_metric_T4Metric_mAP_epoch_29.pth")
        urls.append(f"{base_dir}/best_NuScenes_metric_T4Metric_mAP_epoch_28.pth")

    # Keep order while removing duplicates
    dedup_urls: List[str] = []
    seen = set()
    for u in urls:
        if u not in seen:
            dedup_urls.append(u)
            seen.add(u)
    return dedup_urls


def resolve_model_assets(
    repo_root: Path,
    model_version: str,
    checkpoint_path_arg: str | None,
    model_config_arg: str | None,
) -> Tuple[Path, Path, Dict[str, str]]:
    preset = dict(MODEL_PRESETS.get(model_version, {}))
    if not preset:
        # Dynamic fallback for j6gen2_vX.Y.Z style versions.
        preset = make_j6gen2_dynamic_preset(model_version=model_version)

    if checkpoint_path_arg:
        checkpoint_path = Path(checkpoint_path_arg).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (repo_root / checkpoint_path).resolve()
    else:
        if not preset:
            raise ValueError(
                f"Unknown --centerpoint-model-version '{model_version}'. "
                "--checkpoint-path must be specified for custom versions."
            )
        checkpoint_path = (repo_root / preset["checkpoint_relpath"]).resolve()

    if model_config_arg:
        model_config_path = Path(model_config_arg).expanduser()
        if not model_config_path.is_absolute():
            model_config_path = (repo_root / model_config_path).resolve()
    else:
        if not preset:
            raise ValueError(
                f"Unknown --centerpoint-model-version '{model_version}'. "
                "--model-config must be specified for custom versions."
            )
        model_config_path = (repo_root / preset["model_config"]).resolve()

    if not model_config_path.is_file():
        raise FileNotFoundError(f"CenterPoint model config not found: {model_config_path}")

    return checkpoint_path, model_config_path, preset or {}


def _score_checkpoint_name(path: str) -> Tuple[int, int]:
    name = os.path.basename(path).lower()
    priority = 0
    if "best" in name:
        priority += 1000
    if "t4metric" in name:
        priority += 100
    if "nuscenes" in name:
        priority += 50
    m = re.search(r"epoch[_-]?(\d+)", name)
    epoch = int(m.group(1)) if m else -1
    return (priority, epoch)


def ensure_checkpoint(
    checkpoint_path: Path,
    preset: Dict[str, str],
    env: Dict[str, str],
    dry_run: bool = False,
    verbose: bool = False,
    tracker: Optional[ProgressTracker] = None,
) -> None:
    if checkpoint_path.is_file():
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_urls = build_checkpoint_candidate_urls(preset)
    checkpoint_logs_url = preset.get("checkpoint_logs_url")
    if not checkpoint_urls and not checkpoint_logs_url:
        raise FileNotFoundError(
            f"Checkpoint does not exist and no preset download URL is available: {checkpoint_path}. "
            "Set --checkpoint-path to an existing file."
        )

    print(f"Checkpoint not found. Downloading: {checkpoint_path}")
    for checkpoint_url in checkpoint_urls:
        try:
            run_cmd(
                ["curl", "-fL", checkpoint_url, "-o", str(checkpoint_path)],
                env=env,
                dry_run=dry_run,
                verbose=verbose,
                tracker=tracker,
                label=f"download_checkpoint:{checkpoint_path.name}",
                count_step=False,
            )
            return
        except subprocess.CalledProcessError:
            continue

    if not checkpoint_logs_url:
        raise FileNotFoundError(
            "All checkpoint URL candidates failed and no logs.zip fallback is available. "
            f"checkpoint_path={checkpoint_path}"
        )

    logs_zip = checkpoint_path.with_suffix(".logs.zip")
    run_cmd(
        ["curl", "-fL", checkpoint_logs_url, "-o", str(logs_zip)],
        env=env,
        dry_run=dry_run,
        verbose=verbose,
        tracker=tracker,
        label=f"download_logs_zip:{logs_zip.name}",
        count_step=False,
    )
    if dry_run:
        return

    with zipfile.ZipFile(logs_zip) as zf:
        pth_names = [n for n in zf.namelist() if n.endswith(".pth")]
        if not pth_names:
            raise RuntimeError(f"No .pth file found in logs zip: {logs_zip}")
        pth_names.sort(key=_score_checkpoint_name, reverse=True)
        selected = pth_names[0]
        print(f"Extracting checkpoint from logs.zip: {selected}")
        with zf.open(selected) as src, open(checkpoint_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    logs_zip.unlink(missing_ok=True)


def discover_inference_dirs(repo_root: Path, explicit_dirs: Sequence[str], work_dirs_root: str) -> List[Path]:
    if explicit_dirs:
        dirs = []
        for d in explicit_dirs:
            p = Path(d).expanduser()
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            dirs.append(p)
    else:
        root = Path(work_dirs_root)
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        dirs = sorted([p for p in root.glob("inference_*") if p.is_dir()])

    if not dirs:
        raise FileNotFoundError("No inference directories found.")
    return dirs


def to_dataset_relative_path(source_path: Path, dataset_root: Path) -> str:
    try:
        rel = source_path.resolve().relative_to(dataset_root.resolve())
        return rel.as_posix()
    except ValueError:
        rel = os.path.relpath(str(source_path.resolve()), str(dataset_root.resolve()))
        return Path(rel).as_posix()


def parse_scene_key_from_lidar_rel_path(lidar_rel_path: str) -> str:
    parts = Path(lidar_rel_path).as_posix().split("/")
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return "/".join(parts)


def parse_frame_index_from_lidar_rel_path(lidar_rel_path: str) -> Optional[int]:
    file_name = Path(lidar_rel_path).name
    frame_head = file_name.split(".", 1)[0]
    if frame_head.isdigit():
        return int(frame_head)
    return None


def load_manifest_samples(inference_dir: Path, dataset_root: Path) -> List[ManifestSample]:
    manifest_path = inference_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) == 0:
        raise ValueError(f"Invalid samples in manifest: {manifest_path}")

    samples: List[ManifestSample] = []
    for idx, raw in enumerate(raw_samples):
        source_path = Path(raw["source_path"]).expanduser().resolve()
        rel_path = to_dataset_relative_path(source_path=source_path, dataset_root=dataset_root)
        parts = rel_path.split("/")
        if len(parts) < 6:
            raise ValueError(f"Unexpected source path structure: {source_path}")
        dataset_version, scene_id, scene_version = parts[0], parts[1], parts[2]
        export_index = int(raw.get("export_index", idx))
        samples.append(
            ManifestSample(
                export_index=export_index,
                source_path=source_path,
                source_rel_path=rel_path,
                dataset_version=dataset_version,
                scene_id=scene_id,
                scene_version=scene_version,
            )
        )
    return samples


def build_scene_map(samples: Sequence[ManifestSample]) -> Dict[str, List[str]]:
    by_dataset_version: Dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        by_dataset_version[sample.dataset_version].add(f"{sample.scene_id}/{sample.scene_version}/unknown/unknown")
    return {k: sorted(v) for k, v in by_dataset_version.items()}


def write_split_yaml_files(scene_map: Dict[str, List[str]], split_root: Path) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    for dataset_version, test_scenes in scene_map.items():
        payload = {
            "version": 1,
            "dataset_version": dataset_version.replace("_", "-"),
            "train": [],
            "val": [],
            "test": test_scenes,
        }
        yaml_path = split_root / f"{dataset_version}.yaml"
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def write_dataset_cfg(base_dataset_cfg: Path, dataset_versions: Sequence[str], out_path: Path) -> None:
    lines = [
        f"_base_ = [{base_dataset_cfg.as_posix()!r}]",
        f"dataset_version_list = {list(dataset_versions)!r}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def generate_filtered_ann_files(
    repo_root: Path,
    env: Dict[str, str],
    dataset_root: Path,
    base_dataset_cfg: Path,
    output_root: Path,
    samples: Sequence[ManifestSample],
    max_sweeps: int,
    reuse_cache: bool,
    dry_run: bool,
    verbose: bool,
    tracker: Optional[ProgressTracker],
    step_label: str,
    require_ann_cache: bool,
) -> Tuple[Path, Dict[str, Path], Dict[str, Tuple[int, ...]]]:
    sample_key_material = "pseudo_gt_from_real_v2\n" + "\n".join(s.source_rel_path for s in samples)
    cache_key = hashlib.sha1(sample_key_material.encode("utf-8")).hexdigest()[:16]
    cache_dir = output_root / "cache" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    ann_all_path = cache_dir / "ann_all.pkl"
    metadata_json = cache_dir / "ann_groups.json"
    if reuse_cache and ann_all_path.is_file() and metadata_json.is_file():
        if tracker is not None:
            tracker.instant_step(f"{step_label}(cache)")
        meta = json.loads(metadata_json.read_text())
        ann_by_dataset_version = {k: Path(v) for k, v in meta["ann_by_dataset_version"].items()}
        group_indices = {k: tuple(v) for k, v in meta["group_indices"].items()}
        return ann_all_path, ann_by_dataset_version, group_indices
    if require_ann_cache:
        raise FileNotFoundError(
            f"Required ann cache is missing: {ann_all_path}. "
            "Run once without --require-ann-cache to build metadata cache."
        )

    scene_map = build_scene_map(samples)
    split_root = cache_dir / "split_config"
    write_split_yaml_files(scene_map=scene_map, split_root=split_root)

    dataset_cfg_path = cache_dir / "dataset_cfg.py"
    write_dataset_cfg(base_dataset_cfg=base_dataset_cfg, dataset_versions=sorted(scene_map.keys()), out_path=dataset_cfg_path)

    create_data_out_dir = cache_dir / "create_data_output"
    create_data_out_dir.mkdir(parents=True, exist_ok=True)
    version_name = f"intensity_eval_{cache_key}"
    info_test_path = create_data_out_dir / f"t4dataset_{version_name}_infos_test.pkl"

    create_data_cmd = [
        "uv",
        "run",
        "--offline",
        "--no-sync",
        "python",
        str((repo_root / "tools/detection3d/create_data_t4dataset.py").resolve()),
        "--root_path",
        str(dataset_root),
        "--config",
        str(dataset_cfg_path),
        "--version",
        version_name,
        "--max_sweeps",
        str(max_sweeps),
        "--out_dir",
        str(create_data_out_dir),
        "--dataset_version_config_root",
        str(split_root),
        "--use_available_dataset_version",
    ]
    run_cmd(
        create_data_cmd,
        env=env,
        dry_run=dry_run,
        verbose=verbose,
        tracker=tracker,
        label=step_label,
        count_step=True,
    )
    if dry_run:
        return ann_all_path, {}, {}
    if not info_test_path.is_file():
        raise FileNotFoundError(f"Generated info file not found: {info_test_path}")

    generated = mmengine.load(info_test_path)
    generated_infos = generated["data_list"]
    metainfo = generated.get("metainfo", {})

    by_lidar_rel_path: Dict[str, List[int]] = defaultdict(list)
    by_scene_key: Dict[str, List[int]] = defaultdict(list)
    generated_rel_paths: List[str] = []
    generated_frame_indices: List[Optional[int]] = []
    for info_idx, info in enumerate(generated_infos):
        rel_path = str(info["lidar_points"]["lidar_path"])
        generated_rel_paths.append(rel_path)
        generated_frame_indices.append(parse_frame_index_from_lidar_rel_path(rel_path))
        by_lidar_rel_path[rel_path].append(info_idx)
        by_scene_key[parse_scene_key_from_lidar_rel_path(rel_path)].append(info_idx)

    for scene_indices in by_scene_key.values():
        scene_indices.sort(
            key=lambda idx: (
                generated_frame_indices[idx] is None,
                generated_frame_indices[idx] if generated_frame_indices[idx] is not None else 10**12,
                generated_rel_paths[idx],
            )
        )

    def pop_first_unused(indices: List[int], used_indices: set[int]) -> Optional[int]:
        while indices and indices[0] in used_indices:
            indices.pop(0)
        if not indices:
            return None
        return indices.pop(0)

    ordered_infos: List[dict] = []
    missing_rel_paths: List[str] = []
    fallback_matches: List[Tuple[str, str]] = []
    used_info_indices: set[int] = set()
    for idx, sample in enumerate(samples):
        selected_info_idx = pop_first_unused(by_lidar_rel_path.get(sample.source_rel_path, []), used_info_indices)
        if selected_info_idx is None:
            scene_key = parse_scene_key_from_lidar_rel_path(sample.source_rel_path)
            scene_candidates = [x for x in by_scene_key.get(scene_key, []) if x not in used_info_indices]
            if scene_candidates:
                sample_frame_index = parse_frame_index_from_lidar_rel_path(sample.source_rel_path)
                if sample_frame_index is None:
                    selected_info_idx = scene_candidates[0]
                else:
                    selected_info_idx = min(
                        scene_candidates,
                        key=lambda x: (
                            abs(
                                (
                                    generated_frame_indices[x]
                                    if generated_frame_indices[x] is not None
                                    else sample_frame_index
                                )
                                - sample_frame_index
                            ),
                            generated_frame_indices[x] is None,
                            generated_frame_indices[x] if generated_frame_indices[x] is not None else 10**12,
                        ),
                    )
                fallback_matches.append((sample.source_rel_path, generated_rel_paths[selected_info_idx]))

        if selected_info_idx is None:
            missing_rel_paths.append(sample.source_rel_path)
            continue

        used_info_indices.add(selected_info_idx)
        selected = deepcopy(generated_infos[selected_info_idx])
        selected["sample_idx"] = idx
        if isinstance(selected.get("lidar_points"), dict):
            selected["lidar_points"]["lidar_path"] = sample.source_rel_path
        selected["lidar_path"] = sample.source_rel_path
        # Do not use dataset annotations. Pseudo GT will be injected later.
        selected["instances"] = []
        selected["instances_ignore"] = []
        ordered_infos.append(selected)

    if missing_rel_paths:
        missing_path = cache_dir / "missing_source_rel_paths.txt"
        missing_path.write_text("\n".join(missing_rel_paths) + "\n")
        raise RuntimeError(
            "Failed to map all manifest source frames to generated infos. "
            f"Missing {len(missing_rel_paths)} frames. See: {missing_path}"
        )

    if fallback_matches:
        fallback_path = cache_dir / "fallback_lidar_path_mapping.tsv"
        with fallback_path.open("w") as f:
            f.write("sample_source_rel_path\tselected_metadata_lidar_path\n")
            for sample_rel_path, selected_rel_path in fallback_matches:
                f.write(f"{sample_rel_path}\t{selected_rel_path}\n")
        print(
            f"Applied nearest metadata fallback for {len(fallback_matches)} samples. "
            f"See: {fallback_path}"
        )

    mmengine.dump(dict(data_list=ordered_infos, metainfo=metainfo), ann_all_path)

    group_indices: Dict[str, Tuple[int, ...]] = defaultdict(tuple)
    tmp_group: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        tmp_group[sample.dataset_version].append(idx)
    group_indices = {k: tuple(v) for k, v in tmp_group.items()}

    ann_by_dataset_version: Dict[str, Path] = {}
    for dataset_version, indices in group_indices.items():
        subset_infos = [deepcopy(ordered_infos[i]) for i in indices]
        for new_idx, info in enumerate(subset_infos):
            info["sample_idx"] = new_idx
        ann_path = cache_dir / f"ann_{dataset_version}.pkl"
        mmengine.dump(dict(data_list=subset_infos, metainfo=metainfo), ann_path)
        ann_by_dataset_version[dataset_version] = ann_path

    metadata_json.write_text(
        json.dumps(
            {
                "ann_by_dataset_version": {k: str(v) for k, v in ann_by_dataset_version.items()},
                "group_indices": {k: list(v) for k, v in group_indices.items()},
            },
            indent=2,
        )
    )

    return ann_all_path, ann_by_dataset_version, group_indices


def ensure_condition_file(inference_dir: Path, sample: ManifestSample, source_kind: str) -> Path:
    if source_kind == "real":
        path = inference_dir / "gt" / f"gt_{sample.export_index:05d}.pcd.bin"
    elif source_kind == "ml":
        path = inference_dir / "pred" / f"pred_{sample.export_index:05d}.pcd.bin"
    else:
        raise ValueError(f"Unsupported source_kind: {source_kind}")

    if not path.is_file():
        raise FileNotFoundError(f"Condition file not found: {path}")
    return path.resolve()


def build_path_mapping(
    samples: Sequence[ManifestSample],
    sample_indices: Iterable[int],
    dataset_root: Path,
    inference_dir: Path,
    source_kind: str,
) -> Dict[str, str]:
    path_mapping: Dict[str, str] = {}
    for idx in sample_indices:
        sample = samples[idx]
        condition_file = ensure_condition_file(inference_dir=inference_dir, sample=sample, source_kind=source_kind)
        path_mapping[sample.source_rel_path] = str(condition_file)
        abs_source_path = (dataset_root / sample.source_rel_path).resolve()
        path_mapping[str(abs_source_path)] = str(condition_file)
    return path_mapping


def insert_intensity_transform(pipeline: List[dict], mode: str, range_attenuation: float) -> None:
    if mode == "original":
        return

    insert_idx = None
    for idx, step in enumerate(pipeline):
        if isinstance(step, dict) and step.get("type") == "LoadPointsFromMultiSweeps":
            insert_idx = idx + 1
            break
    if insert_idx is None:
        raise RuntimeError("LoadPointsFromMultiSweeps not found in test pipeline.")

    if mode == "zero":
        transform = dict(type="SetInferenceIntensityZero")
    elif mode == "range":
        transform = dict(
            type="SetInferenceIntensityFromRange",
            attenuation_coefficient=float(range_attenuation),
            max_intensity=255.0,
        )
    else:
        raise ValueError(f"Unsupported intensity mode: {mode}")

    pipeline.insert(insert_idx, transform)


def build_runtime_config(
    base_model_cfg_path: Path,
    dataset_root: Path,
    ann_file: Path,
    dataset_name: str,
    path_mapping: Optional[Dict[str, str]],
    intensity_mode: str,
    range_attenuation: float,
    dataloader_num_workers: int,
    evaluator_mode: str,
    dump_results_path: Optional[Path],
    output_cfg_path: Path,
    apply_path_override: bool = True,
) -> None:
    cfg = Config.fromfile(str(base_model_cfg_path))

    pipeline = list(cfg.test_dataloader.dataset.pipeline)
    if apply_path_override:
        if path_mapping is None:
            raise ValueError("path_mapping must be provided when apply_path_override=True.")
        pipeline.insert(
            0,
            dict(
                type="OverrideLidarPathByMapping",
                path_mapping=path_mapping,
                strict=True,
            ),
        )
    insert_intensity_transform(pipeline=pipeline, mode=intensity_mode, range_attenuation=range_attenuation)
    cfg.test_dataloader.dataset.pipeline = pipeline
    cfg.test_pipeline = pipeline

    cfg.data_root = str(dataset_root) + "/"
    cfg.info_directory_path = ""
    cfg.dataset_test_groups = {dataset_name: str(ann_file)}

    cfg.test_dataloader.dataset.data_root = str(dataset_root) + "/"
    cfg.test_dataloader.dataset.ann_file = str(ann_file)
    cfg.test_dataloader.num_workers = int(dataloader_num_workers)
    cfg.test_dataloader.persistent_workers = bool(dataloader_num_workers > 0)

    if evaluator_mode == "metric" and isinstance(cfg.test_evaluator, dict):
        cfg.test_evaluator.data_root = str(dataset_root) + "/"
        cfg.test_evaluator.ann_file = str(ann_file)
        cfg.test_evaluator.dataset_name = dataset_name
        cfg.test_evaluator.save_csv = True
    elif evaluator_mode == "dump":
        if dump_results_path is None:
            raise ValueError("dump_results_path must be set when evaluator_mode='dump'.")
        dump_results_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.test_evaluator = dict(type="DumpResults", out_file_path=str(dump_results_path))
    else:
        raise ValueError(f"Unsupported evaluator_mode: {evaluator_mode}")

    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output_cfg_path))


def _to_numpy_array(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_pred_instances(pred_item: dict) -> Tuple[np.ndarray, np.ndarray]:
    pred_instances = pred_item.get("pred_instances_3d", {})
    if not isinstance(pred_instances, dict):
        pred_instances = {}

    bboxes_3d = pred_instances.get("bboxes_3d")
    labels_3d = pred_instances.get("labels_3d")

    if hasattr(bboxes_3d, "tensor"):
        boxes_np = _to_numpy_array(bboxes_3d.tensor)
    else:
        boxes_np = _to_numpy_array(bboxes_3d)
    labels_np = _to_numpy_array(labels_3d).astype(np.int64)

    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(-1, boxes_np.shape[0])
    if boxes_np.size == 0:
        boxes_np = np.zeros((0, 9), dtype=np.float32)
    return boxes_np.astype(np.float32), labels_np


def create_pseudo_gt_ann_from_real_dump(
    metadata_ann_path: Path,
    real_dump_path: Path,
    pseudo_ann_out_path: Path,
) -> None:
    ann = mmengine.load(metadata_ann_path)
    infos = ann["data_list"]
    classes = list(ann.get("metainfo", {}).get("classes", []))

    dumped_results = mmengine.load(real_dump_path)
    pred_by_sample_idx: Dict[int, dict] = {}
    for item in dumped_results:
        if not isinstance(item, dict):
            continue
        sample_idx = int(item.get("sample_idx", -1))
        if sample_idx >= 0:
            pred_by_sample_idx[sample_idx] = item

    for sample_idx, info in enumerate(infos):
        pred_item = pred_by_sample_idx.get(sample_idx)
        if pred_item is None:
            raise RuntimeError(f"Missing real dump prediction for sample_idx={sample_idx}")

        boxes_np, labels_np = _extract_pred_instances(pred_item)
        instances = []
        for box, label in zip(boxes_np, labels_np):
            if box.shape[0] < 7:
                continue
            vel = [0.0, 0.0]
            if box.shape[0] >= 9:
                vel = [float(box[7]), float(box[8])]
            label_int = int(label)
            class_name = classes[label_int] if 0 <= label_int < len(classes) else "unknown"
            instances.append(
                dict(
                    bbox_label=label_int,
                    bbox_3d=[float(x) for x in box[:7]],
                    bbox_3d_isvalid=True,
                    bbox_label_3d=label_int,
                    num_lidar_pts=1,
                    num_radar_pts=0,
                    velocity=vel,
                    gt_nusc_name=class_name,
                    gt_attrs=[],
                )
            )

        info["instances"] = instances
        info["instances_ignore"] = []

    pseudo_ann_out_path.parent.mkdir(parents=True, exist_ok=True)
    mmengine.dump(dict(data_list=infos, metainfo=ann.get("metainfo", {})), pseudo_ann_out_path)


def resolve_class_names_and_camera_types(model_cfg_path: Path) -> Tuple[List[str], List[str]]:
    cfg = Config.fromfile(str(model_cfg_path))

    class_names = list(cfg.get("class_names", []))
    if not class_names:
        class_names = list(cfg.get("metainfo", {}).get("classes", []))
    if not class_names and hasattr(cfg, "test_dataloader"):
        class_names = list(getattr(cfg.test_dataloader.dataset, "metainfo", {}).get("classes", []))
    if not class_names:
        raise RuntimeError(f"Failed to resolve class names from model config: {model_cfg_path}")

    camera_types_cfg = cfg.get("camera_types", [])
    if isinstance(camera_types_cfg, set):
        camera_types = sorted(camera_types_cfg)
    else:
        camera_types = list(camera_types_cfg)

    if not camera_types:
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
            "CAM_BACK_LEFT",
        ]
    return class_names, camera_types


def write_direct_ann_file(
    samples: Sequence[ManifestSample],
    sample_indices: Sequence[int],
    inference_dir: Path,
    source_kind: str,
    class_names: Sequence[str],
    camera_types: Sequence[str],
    out_path: Path,
) -> Path:
    identity = np.eye(4, dtype=np.float32).tolist()
    data_list: List[dict] = []
    for local_idx, global_idx in enumerate(sample_indices):
        sample = samples[global_idx]
        pointcloud_path = ensure_condition_file(
            inference_dir=inference_dir,
            sample=sample,
            source_kind=source_kind,
        )
        info = get_empty_standard_data_info(list(camera_types))
        info["sample_idx"] = local_idx
        info["token"] = f"{source_kind}_{sample.export_index:05d}_{local_idx}"
        info["scene_token"] = f"{sample.dataset_version}/{sample.scene_id}/{sample.scene_version}"
        info["timestamp"] = float(local_idx)
        info["ego2global"] = identity
        info["city"] = "unknown"
        info["vehicle_type"] = "unknown"
        info["lidar_points"] = dict(
            num_pts_feats=5,
            lidar_path=str(pointcloud_path),
            lidar2ego=identity,
        )
        info["lidar_sweeps"] = []
        info["instances"] = []
        info["instances_ignore"] = []
        data_list.append(info)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mmengine.dump(
        dict(
            data_list=data_list,
            metainfo=dict(classes=list(class_names), version="direct_pointcloud_bin"),
        ),
        out_path,
    )
    return out_path


def _extract_pred_scores(pred_item: dict) -> np.ndarray:
    pred_instances = pred_item.get("pred_instances_3d", {})
    if not isinstance(pred_instances, dict):
        return np.empty((0,), dtype=np.float32)
    scores_np = _to_numpy_array(pred_instances.get("scores_3d")).astype(np.float32).reshape(-1)
    return scores_np


def load_dump_predictions(dump_path: Path, num_samples: int) -> Dict[int, DetectionFrame]:
    dumped_results = mmengine.load(dump_path)
    pred_by_sample_idx: Dict[int, DetectionFrame] = {}
    for fallback_idx, item in enumerate(dumped_results):
        if not isinstance(item, dict):
            continue
        sample_idx = int(item.get("sample_idx", fallback_idx))
        boxes_np, labels_np = _extract_pred_instances(item)
        scores_np = _extract_pred_scores(item)
        if scores_np.shape[0] != boxes_np.shape[0]:
            scores_np = np.ones((boxes_np.shape[0],), dtype=np.float32)
        pred_by_sample_idx[sample_idx] = DetectionFrame(
            boxes=boxes_np.astype(np.float32),
            labels=labels_np.astype(np.int64).reshape(-1),
            scores=scores_np.astype(np.float32).reshape(-1),
        )

    empty_frame = DetectionFrame(
        boxes=np.zeros((0, 9), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        scores=np.zeros((0,), dtype=np.float32),
    )
    for sample_idx in range(num_samples):
        if sample_idx not in pred_by_sample_idx:
            pred_by_sample_idx[sample_idx] = DetectionFrame(
                boxes=empty_frame.boxes.copy(),
                labels=empty_frame.labels.copy(),
                scores=empty_frame.scores.copy(),
            )
    return pred_by_sample_idx


def _compute_ap(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt <= 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(float(num_gt), 1.0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _angle_diff_abs(pred_yaw: float, gt_yaw: float) -> float:
    diff = (pred_yaw - gt_yaw + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(diff))


def _scale_iou_aligned_3d(pred_dims: np.ndarray, gt_dims: np.ndarray) -> float:
    pred = np.clip(np.abs(pred_dims.astype(np.float32)), 1e-6, None)
    gt = np.clip(np.abs(gt_dims.astype(np.float32)), 1e-6, None)
    inter = float(np.prod(np.minimum(pred, gt)))
    pred_vol = float(np.prod(pred))
    gt_vol = float(np.prod(gt))
    union = pred_vol + gt_vol - inter
    return inter / max(union, 1e-9)


def _match_by_center_distance(
    gt_frames: Dict[int, DetectionFrame],
    pred_frames: Dict[int, DetectionFrame],
    class_idx: int,
    dist_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]], int]:
    gt_boxes_by_sample: Dict[int, np.ndarray] = {}
    gt_matched_flags: Dict[int, np.ndarray] = {}
    num_gt = 0
    for sample_idx, frame in gt_frames.items():
        mask = frame.labels == class_idx
        gt_boxes = frame.boxes[mask]
        gt_boxes_by_sample[sample_idx] = gt_boxes
        gt_matched_flags[sample_idx] = np.zeros((gt_boxes.shape[0],), dtype=bool)
        num_gt += int(gt_boxes.shape[0])

    pred_items: List[Tuple[float, int, np.ndarray]] = []
    for sample_idx, frame in pred_frames.items():
        if frame.boxes.shape[0] == 0:
            continue
        mask = frame.labels == class_idx
        pred_boxes = frame.boxes[mask]
        pred_scores = frame.scores[mask]
        for box, score in zip(pred_boxes, pred_scores):
            pred_items.append((float(score), int(sample_idx), box))
    pred_items.sort(key=lambda x: x[0], reverse=True)

    tp: List[float] = []
    fp: List[float] = []
    matched_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for _, sample_idx, pred_box in pred_items:
        gt_boxes = gt_boxes_by_sample.get(sample_idx)
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        matched = gt_matched_flags[sample_idx]
        available_idx = np.where(~matched)[0]
        if available_idx.size == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        dists = np.linalg.norm(gt_boxes[available_idx, :2] - pred_box[:2], axis=1)
        best_local = int(np.argmin(dists))
        best_dist = float(dists[best_local])
        if best_dist <= float(dist_threshold):
            gt_idx = int(available_idx[best_local])
            matched[gt_idx] = True
            tp.append(1.0)
            fp.append(0.0)
            matched_pairs.append((pred_box, gt_boxes[gt_idx]))
        else:
            tp.append(0.0)
            fp.append(1.0)

    return np.asarray(tp, dtype=np.float32), np.asarray(fp, dtype=np.float32), matched_pairs, int(num_gt)


def compute_custom_metrics(
    gt_frames: Dict[int, DetectionFrame],
    pred_frames: Dict[int, DetectionFrame],
    class_names: Sequence[str],
    dist_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    tp_distance_threshold: float = 2.0,
) -> Dict[str, object]:
    class_results: Dict[str, dict] = {}
    class_map_values: List[float] = []
    class_ate_values: List[float] = []
    class_ase_values: List[float] = []
    class_aoe_values: List[float] = []
    class_ave_values: List[float] = []

    for class_idx, class_name in enumerate(class_names):
        ap_by_distance: Dict[str, float] = {}
        class_ap_values: List[float] = []
        class_gt_count = int(
            sum(np.count_nonzero(frame.labels == class_idx) for frame in gt_frames.values())
        )
        for dist_th in dist_thresholds:
            tp, fp, _, num_gt = _match_by_center_distance(
                gt_frames=gt_frames,
                pred_frames=pred_frames,
                class_idx=class_idx,
                dist_threshold=float(dist_th),
            )
            ap = _compute_ap(tp=tp, fp=fp, num_gt=num_gt)
            ap_by_distance[f"{dist_th:.2f}"] = float(ap)
            if np.isfinite(ap):
                class_ap_values.append(float(ap))

        class_map = float(np.mean(class_ap_values)) if class_ap_values else float("nan")
        if np.isfinite(class_map):
            class_map_values.append(class_map)

        _, _, matched_pairs, _ = _match_by_center_distance(
            gt_frames=gt_frames,
            pred_frames=pred_frames,
            class_idx=class_idx,
            dist_threshold=float(tp_distance_threshold),
        )
        if matched_pairs:
            ate_arr = np.asarray(
                [np.linalg.norm(pred_box[:2] - gt_box[:2]) for pred_box, gt_box in matched_pairs],
                dtype=np.float32,
            )
            ase_arr = np.asarray(
                [1.0 - _scale_iou_aligned_3d(pred_box[3:6], gt_box[3:6]) for pred_box, gt_box in matched_pairs],
                dtype=np.float32,
            )
            aoe_arr = np.asarray(
                [_angle_diff_abs(float(pred_box[6]), float(gt_box[6])) for pred_box, gt_box in matched_pairs],
                dtype=np.float32,
            )
            ave_arr = np.asarray(
                [
                    np.linalg.norm(pred_box[7:9] - gt_box[7:9]) if pred_box.shape[0] >= 9 and gt_box.shape[0] >= 9 else 0.0
                    for pred_box, gt_box in matched_pairs
                ],
                dtype=np.float32,
            )
            ate = float(np.mean(ate_arr))
            ase = float(np.mean(ase_arr))
            aoe = float(np.mean(aoe_arr))
            ave = float(np.mean(ave_arr))
            class_ate_values.append(ate)
            class_ase_values.append(ase)
            class_aoe_values.append(aoe)
            class_ave_values.append(ave)
        else:
            ate = float("nan")
            ase = float("nan")
            aoe = float("nan")
            ave = float("nan")

        class_results[class_name] = dict(
            num_gt=class_gt_count,
            mAP=class_map,
            AP_dist=ap_by_distance,
            ATE=ate,
            ASE=ase,
            AOE=aoe,
            AVE=ave,
            num_tp_matches=len(matched_pairs),
        )

    m_ap = float(np.mean(class_map_values)) if class_map_values else 0.0
    mean_ate = float(np.mean(class_ate_values)) if class_ate_values else float("nan")
    mean_ase = float(np.mean(class_ase_values)) if class_ase_values else float("nan")
    mean_aoe = float(np.mean(class_aoe_values)) if class_aoe_values else float("nan")
    mean_ave = float(np.mean(class_ave_values)) if class_ave_values else float("nan")

    tp_ate_score = max(0.0, 1.0 - (mean_ate / 2.0)) if np.isfinite(mean_ate) else 0.0
    tp_ase_score = max(0.0, 1.0 - mean_ase) if np.isfinite(mean_ase) else 0.0
    tp_aoe_score = max(0.0, 1.0 - (mean_aoe / np.pi)) if np.isfinite(mean_aoe) else 0.0
    tp_ave_score = max(0.0, 1.0 - (mean_ave / 2.0)) if np.isfinite(mean_ave) else 0.0
    nds = float((5.0 * m_ap + tp_ate_score + tp_ase_score + tp_aoe_score + tp_ave_score) / 9.0)

    return dict(
        metric_backend="custom_pseudo_gt_v1",
        mAP=float(m_ap),
        NDS=float(nds),
        mean_tp_errors=dict(ATE=mean_ate, ASE=mean_ase, AOE=mean_aoe, AVE=mean_ave),
        normalized_tp_scores=dict(ATE=tp_ate_score, ASE=tp_ase_score, AOE=tp_aoe_score, AVE=tp_ave_score),
        class_metrics=class_results,
    )


def write_custom_metric_json(metric_path: Path, metric_dict: Dict[str, object]) -> Path:
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metric_dict, indent=2))
    return metric_path


def extract_metrics_from_csv(work_dir: Path, dataset_name: str) -> Tuple[float, float, Path]:
    csv_candidates = sorted(
        work_dir.glob(f"**/scores_{dataset_name}.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_candidates:
        # fallback if dataset_name was sanitized differently
        csv_candidates = sorted(
            work_dir.glob("**/scores_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not csv_candidates:
        raise FileNotFoundError(f"scores_*.csv not found under: {work_dir}")

    csv_path = csv_candidates[0]
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        raise RuntimeError(f"No rows found in score csv: {csv_path}")

    nds = float(row["NDS"])
    m_ap = float(row["mAP"])
    return m_ap, nds, csv_path


def evaluate_single_scenario(
    repo_root: Path,
    env: Dict[str, str],
    test_cfg_path: Path,
    checkpoint_path: Path,
    work_dir: Path,
    dataset_name: str,
    gpus: int,
    master_port: int,
    dry_run: bool,
    verbose: bool,
    tracker: Optional[ProgressTracker],
    step_label: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[float, float, Path]:
    if gpus > 1:
        cmd = [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpus),
            "--master_port",
            str(master_port),
            str((repo_root / "tools/detection3d/test.py").resolve()),
            str(test_cfg_path),
            str(checkpoint_path),
            "--work-dir",
            str(work_dir),
            "--launcher",
            "pytorch",
        ]
    else:
        cmd = [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            "python",
            str((repo_root / "tools/detection3d/test.py").resolve()),
            str(test_cfg_path),
            str(checkpoint_path),
            "--work-dir",
            str(work_dir),
        ]
    run_cmd(
        cmd=cmd,
        env=env,
        dry_run=dry_run,
        verbose=verbose,
        tracker=tracker,
        label=step_label,
        count_step=True,
        extra_env=extra_env,
    )
    if dry_run:
        return float("nan"), float("nan"), Path("")
    return extract_metrics_from_csv(work_dir=work_dir, dataset_name=dataset_name)


def run_dump_scenario(
    repo_root: Path,
    env: Dict[str, str],
    test_cfg_path: Path,
    checkpoint_path: Path,
    work_dir: Path,
    gpus: int,
    master_port: int,
    dry_run: bool,
    verbose: bool,
    tracker: Optional[ProgressTracker],
    step_label: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    if gpus > 1:
        cmd = [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(gpus),
            "--master_port",
            str(master_port),
            str((repo_root / "tools/detection3d/test.py").resolve()),
            str(test_cfg_path),
            str(checkpoint_path),
            "--work-dir",
            str(work_dir),
            "--launcher",
            "pytorch",
        ]
    else:
        cmd = [
            "uv",
            "run",
            "--offline",
            "--no-sync",
            "python",
            str((repo_root / "tools/detection3d/test.py").resolve()),
            str(test_cfg_path),
            str(checkpoint_path),
            "--work-dir",
            str(work_dir),
        ]
    run_cmd(
        cmd=cmd,
        env=env,
        dry_run=dry_run,
        verbose=verbose,
        tracker=tracker,
        label=step_label,
        count_step=True,
        extra_env=extra_env,
    )


def sanitize_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text)


def detect_visible_gpu_ids(num_gpus: int) -> List[str]:
    if num_gpus <= 0:
        return []
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        parts = [p.strip() for p in cuda_visible.split(",") if p.strip()]
        if len(parts) >= num_gpus:
            return parts[:num_gpus]
    return [str(i) for i in range(num_gpus)]


def write_summary(
    inference_dir_name: str,
    output_root: Path,
    rows: List[dict],
) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"summary_{sanitize_name(inference_dir_name)}.csv"
    json_path = output_root / f"summary_{sanitize_name(inference_dir_name)}.json"

    headers = [
        "inference_dir",
        "group",
        "samples",
        "scenario",
        "source_kind",
        "intensity_mode",
        "mAP",
        "NDS",
        "delta_mAP_vs_real",
        "delta_NDS_vs_real",
        "score_csv",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def print_quick_table(rows: Sequence[dict]) -> None:
    if not rows:
        return

    print("")
    print("group | scenario | samples | mAP | NDS | delta_mAP_vs_real | delta_NDS_vs_real")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['group']} | {row['scenario']} | {row['samples']} | "
            f"{row['mAP']:.4f} | {row['NDS']:.4f} | "
            f"{row['delta_mAP_vs_real']:+.4f} | {row['delta_NDS_vs_real']:+.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.list_model_versions:
        print("Known model presets:")
        for key in sorted(MODEL_PRESETS.keys()):
            print(f"- {key}")
        print("")
        print("Dynamic fallback:")
        print("- j6gen2_vX.Y.Z (tries common checkpoint filenames and logs.zip)")
        return

    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_dataset_cfg = Path(args.dataset_config).expanduser()
    if not base_dataset_cfg.is_absolute():
        base_dataset_cfg = (repo_root / base_dataset_cfg).resolve()
    if args.evaluation_backend == "t4metric" and not base_dataset_cfg.is_file():
        raise FileNotFoundError(f"Dataset config not found: {base_dataset_cfg}")

    checkpoint_path, model_cfg_path, preset = resolve_model_assets(
        repo_root=repo_root,
        model_version=args.centerpoint_model_version,
        checkpoint_path_arg=args.checkpoint_path,
        model_config_arg=args.model_config,
    )
    env = build_uv_env(repo_root=repo_root, centerpoint_device=args.centerpoint_device)

    inference_dirs = discover_inference_dirs(
        repo_root=repo_root,
        explicit_dirs=args.inference_dir,
        work_dirs_root=args.work_dirs_root,
    )

    samples_by_inference_dir: Dict[Path, List[ManifestSample]] = {}
    total_steps = 0
    for inference_dir in inference_dirs:
        samples = load_manifest_samples(inference_dir=inference_dir, dataset_root=dataset_root)
        samples_by_inference_dir[inference_dir] = samples
        group_count = 1
        if not args.disable_per_dataset_version:
            group_count += len({s.dataset_version for s in samples})
        if args.evaluation_backend == "custom":
            total_steps += group_count * 4
        else:
            total_steps += 1 + (group_count * 5)

    tracker = None if args.verbose else ProgressTracker(total_steps=total_steps, log_dir=output_root / "logs")
    if not args.verbose:
        print(f"Progress: {total_steps} steps. command logs: {output_root / 'logs'}")

    ensure_checkpoint(
        checkpoint_path=checkpoint_path,
        preset=preset,
        env=env,
        dry_run=args.dry_run,
        verbose=args.verbose,
        tracker=tracker,
    )

    if args.centerpoint_device == "cpu":
        num_gpus = 0
    else:
        import torch

        detected_gpus = torch.cuda.device_count()
        if args.gpus > 0:
            num_gpus = args.gpus
            if detected_gpus > 0 and num_gpus > detected_gpus:
                raise RuntimeError(f"Requested --gpus {num_gpus}, but only {detected_gpus} CUDA GPU(s) are visible.")
        else:
            num_gpus = detected_gpus
        if num_gpus <= 0:
            raise RuntimeError("No CUDA GPU detected. Use --centerpoint-device cpu or provide visible GPUs.")
    if args.centerpoint_device == "cuda":
        print(f"Using CUDA inference with {num_gpus} GPU(s).")
    scenario_parallelism = 1
    if args.centerpoint_device == "cuda" and args.parallel_scenarios:
        scenario_parallelism = min(4, num_gpus)
    visible_gpu_ids = detect_visible_gpu_ids(num_gpus)
    if scenario_parallelism > 1:
        print(f"Scenario parallelism: {scenario_parallelism} (real/ml/zero/range concurrent)")

    cpu_cores = os.cpu_count() or 1
    if args.num_workers > 0:
        dataloader_num_workers = args.num_workers
    else:
        if scenario_parallelism > 1:
            dataloader_num_workers = max(1, cpu_cores // scenario_parallelism)
        else:
            ranks = max(1, num_gpus)
            dataloader_num_workers = max(1, cpu_cores // ranks)
    print(
        f"DataLoader workers per rank: {dataloader_num_workers} "
        f"(cpu_cores={cpu_cores}, active_processes={scenario_parallelism if scenario_parallelism > 1 else max(1, num_gpus)})"
    )

    if args.evaluation_backend != "custom":
        raise NotImplementedError(
            "t4metric backend is not available in this comparison script path. "
            "Use --evaluation-backend custom."
        )

    class_names, camera_types = resolve_class_names_and_camera_types(model_cfg_path=model_cfg_path)
    print(f"Custom evaluator classes: {len(class_names)}")

    global_rows: List[dict] = []
    scenario_counter = 0
    for inference_dir in inference_dirs:
        samples = samples_by_inference_dir[inference_dir]
        print(f"Processing {inference_dir.name} ({len(samples)} samples)")

        group_indices: Dict[str, Tuple[int, ...]] = {}
        if not args.disable_per_dataset_version:
            tmp_group: Dict[str, List[int]] = defaultdict(list)
            for idx, sample in enumerate(samples):
                tmp_group[sample.dataset_version].append(idx)
            group_indices = {k: tuple(v) for k, v in tmp_group.items()}

        groups: List[GroupInfo] = [
            GroupInfo(
                name="all",
                sample_indices=tuple(range(len(samples))),
            )
        ]
        if not args.disable_per_dataset_version:
            for dataset_version in sorted(group_indices.keys()):
                groups.append(
                    GroupInfo(
                        name=dataset_version,
                        sample_indices=group_indices[dataset_version],
                    )
                )

        rows_for_inference: List[dict] = []
        for group in groups:
            if args.verbose:
                print(f"\n-- Group: {group.name} ({len(group.sample_indices)} samples)")
            scenario_defs = [
                ("real", "real", "original"),
                ("ml", "ml", "original"),
                ("zero", args.zero_range_source, "zero"),
                ("range", args.zero_range_source, "range"),
            ]

            group_dir = output_root / sanitize_name(inference_dir.name) / sanitize_name(group.name)
            direct_ann_dir = (
                output_root
                / "direct_anns"
                / sanitize_name(inference_dir.name)
                / sanitize_name(group.name)
            )
            source_ann_files: Dict[str, Path] = {}
            for source_kind in sorted({source for _, source, _ in scenario_defs}):
                ann_path = direct_ann_dir / f"{source_kind}.pkl"
                source_ann_files[source_kind] = ann_path
                if not args.dry_run:
                    write_direct_ann_file(
                        samples=samples,
                        sample_indices=group.sample_indices,
                        inference_dir=inference_dir,
                        source_kind=source_kind,
                        class_names=class_names,
                        camera_types=camera_types,
                        out_path=ann_path,
                    )

            scenario_jobs = []
            for idx, (scenario_name, source_kind, intensity_mode) in enumerate(scenario_defs):
                dataset_name = sanitize_name(f"{inference_dir.name}_{group.name}_{scenario_name}")
                runtime_cfg_path = (
                    output_root
                    / "runtime_cfgs"
                    / sanitize_name(inference_dir.name)
                    / sanitize_name(group.name)
                    / f"{scenario_name}.py"
                )
                dump_results_path = (
                    output_root
                    / "dump_results"
                    / sanitize_name(inference_dir.name)
                    / sanitize_name(group.name)
                    / f"{scenario_name}.pkl"
                )
                run_work_dir = (
                    output_root
                    / "runs"
                    / sanitize_name(inference_dir.name)
                    / sanitize_name(group.name)
                    / scenario_name
                )

                build_runtime_config(
                    base_model_cfg_path=model_cfg_path,
                    dataset_root=Path("/"),
                    ann_file=source_ann_files[source_kind],
                    dataset_name=dataset_name,
                    path_mapping=None,
                    intensity_mode=intensity_mode,
                    range_attenuation=args.range_attenuation,
                    dataloader_num_workers=dataloader_num_workers,
                    evaluator_mode="dump",
                    dump_results_path=dump_results_path,
                    output_cfg_path=runtime_cfg_path,
                    apply_path_override=False,
                )

                assigned_gpu = None
                run_gpus = num_gpus
                extra_env = None
                if scenario_parallelism > 1:
                    assigned_gpu = visible_gpu_ids[idx % scenario_parallelism]
                    run_gpus = 1
                    extra_env = {"CUDA_VISIBLE_DEVICES": assigned_gpu}
                step_label = f"infer:{inference_dir.name}:{group.name}:{scenario_name}"
                if assigned_gpu is not None:
                    step_label += f":gpu{assigned_gpu}"

                scenario_jobs.append(
                    dict(
                        scenario_name=scenario_name,
                        source_kind=source_kind,
                        intensity_mode=intensity_mode,
                        dataset_name=dataset_name,
                        runtime_cfg_path=runtime_cfg_path,
                        run_work_dir=run_work_dir,
                        dump_results_path=dump_results_path,
                        run_gpus=run_gpus,
                        master_port=args.master_port + scenario_counter,
                        step_label=step_label,
                        extra_env=extra_env,
                    )
                )
                scenario_counter += 1

            scenario_results: Dict[str, Path] = {}
            if scenario_parallelism > 1:
                with ThreadPoolExecutor(max_workers=scenario_parallelism) as executor:
                    future_to_job = {}
                    for job in scenario_jobs:
                        future = executor.submit(
                            run_dump_scenario,
                            repo_root=repo_root,
                            env=env,
                            test_cfg_path=job["runtime_cfg_path"],
                            checkpoint_path=checkpoint_path,
                            work_dir=job["run_work_dir"],
                            gpus=job["run_gpus"],
                            master_port=job["master_port"],
                            dry_run=args.dry_run,
                            verbose=args.verbose,
                            tracker=tracker,
                            step_label=job["step_label"],
                            extra_env=job["extra_env"],
                        )
                        future_to_job[future] = job
                    for future in as_completed(future_to_job):
                        job = future_to_job[future]
                        future.result()
                        scenario_results[job["scenario_name"]] = job["dump_results_path"]
            else:
                for job in scenario_jobs:
                    run_dump_scenario(
                        repo_root=repo_root,
                        env=env,
                        test_cfg_path=job["runtime_cfg_path"],
                        checkpoint_path=checkpoint_path,
                        work_dir=job["run_work_dir"],
                        gpus=job["run_gpus"],
                        master_port=job["master_port"],
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                        tracker=tracker,
                        step_label=job["step_label"],
                        extra_env=job["extra_env"],
                    )
                    scenario_results[job["scenario_name"]] = job["dump_results_path"]

            if args.dry_run:
                scenario_metric_results: Dict[str, Tuple[float, float, Path]] = {
                    scenario_name: (float("nan"), float("nan"), Path("."))
                    for scenario_name, _, _ in scenario_defs
                }
            else:
                if "real" not in scenario_results:
                    raise RuntimeError("Missing 'real' scenario result for pseudo-GT baseline.")
                real_pred_frames = load_dump_predictions(
                    dump_path=scenario_results["real"],
                    num_samples=len(group.sample_indices),
                )
                scenario_metric_results = {}
                for scenario_name, _, _ in scenario_defs:
                    scenario_pred_frames = load_dump_predictions(
                        dump_path=scenario_results[scenario_name],
                        num_samples=len(group.sample_indices),
                    )
                    metric_dict = compute_custom_metrics(
                        gt_frames=real_pred_frames,
                        pred_frames=scenario_pred_frames,
                        class_names=class_names,
                    )
                    metric_json_path = (
                        group_dir
                        / "custom_metrics"
                        / f"{scenario_name}.json"
                    )
                    write_custom_metric_json(metric_path=metric_json_path, metric_dict=metric_dict)
                    scenario_metric_results[scenario_name] = (
                        float(metric_dict["mAP"]),
                        float(metric_dict["NDS"]),
                        metric_json_path,
                    )

            for scenario_name, source_kind, intensity_mode in scenario_defs:
                m_ap, nds, metric_path = scenario_metric_results[scenario_name]
                rows_for_inference.append(
                    dict(
                        inference_dir=inference_dir.name,
                        group=group.name,
                        samples=len(group.sample_indices),
                        scenario=scenario_name,
                        source_kind=source_kind,
                        intensity_mode=intensity_mode,
                        mAP=m_ap,
                        NDS=nds,
                        delta_mAP_vs_real=0.0,  # filled later
                        delta_NDS_vs_real=0.0,  # filled later
                        score_csv=str(metric_path),
                    )
                )

            if "real" in scenario_metric_results:
                baseline_map, baseline_nds, _ = scenario_metric_results["real"]
                for row in rows_for_inference:
                    if row["inference_dir"] == inference_dir.name and row["group"] == group.name:
                        row["delta_mAP_vs_real"] = row["mAP"] - baseline_map
                        row["delta_NDS_vs_real"] = row["NDS"] - baseline_nds

        if rows_for_inference:
            print_quick_table(rows_for_inference)
            summary_csv, summary_json = write_summary(
                inference_dir_name=inference_dir.name,
                output_root=output_root,
                rows=rows_for_inference,
            )
            print(f"\nSaved summary CSV : {summary_csv}")
            print(f"Saved summary JSON: {summary_json}")
            global_rows.extend(rows_for_inference)

    if global_rows:
        all_csv, all_json = write_summary(
            inference_dir_name="all_inference_dirs",
            output_root=output_root,
            rows=global_rows,
        )
        print(f"\nSaved global summary CSV : {all_csv}")
        print(f"Saved global summary JSON: {all_json}")


if __name__ == "__main__":
    main()
