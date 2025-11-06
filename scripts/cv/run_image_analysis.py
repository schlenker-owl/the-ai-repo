#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

from airoad.cv.image_analysis import ImageAnalyzer, ImageAnalyzerConfig

_DEFAULT_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]


def _load_yaml_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _iter_images(source_dir: Path, patterns: Iterable[str], recursive: bool) -> List[Path]:
    imgs: List[Path] = []
    for pat in patterns:
        imgs.extend(source_dir.rglob(pat) if recursive else source_dir.glob(pat))
    uniq = sorted(set([p for p in imgs if p.is_file()]))
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = _load_yaml_cfg(args.config)

    # Base config (defaults)
    base = ImageAnalyzerConfig(
        source=str(cfg.get("source", "")),
        model=str(cfg.get("model", "yolo11n.pt")),
        conf=float(cfg.get("conf", 0.3)),
        iou=float(cfg.get("iou", 0.5)),
        imgsz=int(cfg.get("imgsz", 640)),
        classes=cfg.get("classes"),
        device=cfg.get("device"),
        out_dir=str(cfg.get("out_dir", "outputs/cv_images/sample")),
        save_annotated=bool(cfg.get("save_annotated", True)),
        draw_labels=bool(cfg.get("draw_labels", True)),
        draw_conf=bool(cfg.get("draw_conf", True)),
        save_json=bool(cfg.get("save_json", True)),
        save_crops=bool(cfg.get("save_crops", False)),
        save_masks=bool(cfg.get("save_masks", False)),
        crop_max_size=cfg.get("crop_max_size", None),
    )

    # Single-image mode?
    source_dir = cfg.get("source_dir", None)
    output_root = Path(cfg.get("output_root", "outputs/cv_images")).resolve()
    patterns = cfg.get("source_glob", _DEFAULT_PATTERNS)
    if isinstance(patterns, str):
        patterns = [patterns]
    recursive = bool(cfg.get("recursive", False))

    analyzer = ImageAnalyzer(base)

    det_rows: List[Dict[str, Any]] = []
    batch_summary: List[Dict[str, Any]] = []

    if source_dir:
        src_dir = Path(source_dir).expanduser().resolve()
        if not src_dir.exists():
            raise FileNotFoundError(f"source_dir not found: {src_dir}")

        images = _iter_images(src_dir, patterns, recursive)
        if not images:
            print(
                f"[image_analysis] No images in {src_dir} with {list(patterns)} (recursive={recursive})"
            )
            return

        print(f"[image_analysis] Found {len(images)} images. Output root: {output_root}")
        for i, ip in enumerate(images, 1):
            stem = ip.stem
            per_dir = output_root / stem
            per_dir.mkdir(parents=True, exist_ok=True)
            # run
            res = analyzer.analyze_image(str(ip), str(per_dir))
            det_rows.extend(res["detections"])
            batch_summary.append(
                {
                    "image": str(ip),
                    "outputs": {
                        "annotated": res["annotated"],
                        "json": res["json"],
                        "dir": str(per_dir),
                    },
                    "counts": res["counts"],
                }
            )

        # write batch CSV / summary
        if det_rows:
            df = pd.DataFrame(det_rows)
            df.to_csv(output_root / "_detections.csv", index=False)
        with open(output_root / "_batch_summary.json", "w") as f:
            json.dump(batch_summary, f, indent=2)
        print(f"[image_analysis] Done. Wrote outputs under {output_root}")

    else:
        # single image mode
        if not base.source:
            raise ValueError(
                "YAML must include either 'source' (image file) or 'source_dir' (folder)."
            )
        out_dir = Path(base.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        res = analyzer.analyze_image(base.source, str(out_dir))
        if res["detections"]:
            pd.DataFrame(res["detections"]).to_csv(out_dir / "_detections.csv", index=False)
        with open(out_dir / "_summary.json", "w") as f:
            json.dump(
                {
                    "image": res["image"],
                    "counts": res["counts"],
                    "outputs": {"annotated": res["annotated"], "json": res["json"]},
                },
                f,
                indent=2,
            )
        print(f"[image_analysis] Done. Wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
