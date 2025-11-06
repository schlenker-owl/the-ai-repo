#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
from ultralytics import YOLO


def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/cv/train.yaml")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    model_path = cfg.get("train", {}).get("model", "yolo11s.pt")
    data_yaml = cfg.get("data", {}).get("yaml")  # optional explicit YAML path
    if not data_yaml:
        # construct from dataset root if provided
        root = cfg.get("data", {}).get("root", None)
        if not root:
            raise ValueError(
                "Provide data.yaml path (data.yaml) or dataset root (data.root) in configs/cv/train.yaml"
            )
        data_yaml = str((Path(root) / "data.yaml").resolve())

    yolo = YOLO(model_path)
    print(f"[train] Training {model_path} with data={data_yaml}")

    # Ultralytics TrainingArguments-compatible keys
    t = cfg.get("train", {})
    yolo.train(
        data=data_yaml,
        imgsz=int(t.get("imgsz", 640)),
        epochs=int(t.get("epochs", 50)),
        batch=int(t.get("batch", 16)),
        device=str(t.get("device", "0")),  # "mps" or "0"
        seed=int(t.get("seed", 42)),
        lr0=float(t.get("lr0", 0.01)),
        optimizer=str(t.get("optimizer", "auto")),
        patience=int(t.get("patience", 50)),
        project=str(t.get("project", "runs/train")),
        name=str(t.get("name", "exp")),
        exist_ok=bool(t.get("exist_ok", True)),
        # You can add more params here (augment, mosaic, etc.)
    )
    print("[train] Done")


if __name__ == "__main__":
    main()
