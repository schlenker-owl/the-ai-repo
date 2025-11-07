from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import cv2
import yaml


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def save_json(path: Path | str, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2))


def load_yaml(path: Path | str) -> Any:
    return yaml.safe_load(Path(path).read_text())


def save_yaml(path: Path | str, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(yaml.safe_dump(payload, sort_keys=False))


def safe_glob(root: Path | str, patterns: Iterable[str]) -> List[Path]:
    r = Path(root)
    out: List[Path] = []
    for pat in patterns:
        out.extend(r.glob(pat))
    # dedupe + keep files only
    return sorted({p.resolve() for p in out if p.is_file()})


def video_meta(video: Path | str) -> Tuple[int, int, float, int]:
    """
    Return (width, height, fps, frame_count). If FPS or count is unknown, we best-effort.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return w, h, fps, n
