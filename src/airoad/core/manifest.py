from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .io import ensure_dir, save_json


@dataclass
class Manifest:
    name: str  # e.g., "cv/video_analysis"
    version: str  # script or pipeline version
    timestamp: str  # ISO8601
    config: Optional[str]  # path to YAML used
    inputs: List[str]
    outputs: Dict[str, str]
    seed: Optional[int]
    env: Dict[str, str]
    git: Dict[str, str]


def _git_info() -> Dict[str, str]:
    def _run(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = "true" if _run(["git", "status", "--porcelain"]) else "false"
    return {"commit": commit, "dirty": dirty}


def _env_info(device: str) -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "platform": platform.platform(),
    }


def write_manifest(
    out_dir: Path | str,
    name: str,
    version: str,
    config: Optional[str],
    inputs: List[str],
    outputs: Dict[str, str],
    seed: Optional[int],
    device: str,
) -> Manifest:
    ts = datetime.now(timezone.utc).isoformat()
    man = Manifest(
        name=name,
        version=version,
        timestamp=ts,
        config=config,
        inputs=inputs,
        outputs=outputs,
        seed=seed,
        env=_env_info(device),
        git=_git_info(),
    )
    out_dir = ensure_dir(Path(out_dir))
    save_json(out_dir / "manifest.json", asdict(man))
    return man
