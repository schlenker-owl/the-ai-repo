#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import yaml
from PIL import Image

from airoad.search.backbone import ImageTextBackbone, SearchBackboneConfig
from airoad.search.indexer import IndexConfig, VectorIndex

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _resolve_inputs(cfg: dict, cli_roots: List[str] | None) -> tuple[list[Path], list[str]]:
    # Roots
    if cli_roots:
        roots = [Path(r).resolve() for r in cli_roots]
    else:
        roots = [
            Path(r).resolve() for r in (cfg.get("inputs", {})).get("roots", ["outputs/cv_images"])
        ]
    # Patterns
    patterns = (cfg.get("inputs", {})).get("patterns", [])
    if not patterns:
        # sensible defaults if config has none
        patterns = [
            "**/crops/*/*.jpg",
            "**/crops/*/*.jpeg",
            "**/crops/*/*.png",
            "**/crops/*/*.webp",
            "**/masks/*.png",
            "**/*_det.png",
            "**/*_seg.png",
            "**/*_pose.png",
            "**/*_cls.png",
        ]
    return roots, patterns


def _iter_images(roots: list[Path], patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        for pat in patterns:
            out.extend(root.glob(pat))
    out = [p for p in out if p.is_file() and p.suffix.lower() in IMG_EXTS + (".png",)]
    # de-dup and sort
    return sorted(set(out))


def torch_stack(tensors):
    import torch

    return torch.stack(tensors, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/cv/search.yaml")
    ap.add_argument(
        "--roots",
        type=str,
        nargs="*",
        default=None,
        help="override roots; e.g. --roots outputs/cv_images outputs/cv",
    )
    ap.add_argument(
        "--out", type=str, default="outputs/cv_images/.index", help="index output directory"
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}

    # Backbone
    bcfg = cfg.get("backbone", {})
    sb = ImageTextBackbone(
        SearchBackboneConfig(
            name=bcfg.get("name", "auto"),
            device=bcfg.get("device", None),
            dtype=bcfg.get("dtype", "float32"),
            image_size=int(bcfg.get("image_size", 224)),
        )
    )

    # Inputs
    roots, patterns = _resolve_inputs(cfg, args.roots)
    items = _iter_images(roots, patterns)
    if not items:
        print("[search/index_crops] No images found for indexing.")
        print("  Hints:")
        print("   • Enable crops in image_analysis (save_crops: true) or")
        print(
            "   • Use annotated files (*_det.png, *_seg.png, masks/*.png) and keep the default patterns in configs/cv/search.yaml"
        )
        return

    print(f"[search/index_crops] Found {len(items)} images to index.")

    # Embed in batches
    batch_size = int(cfg.get("embed", {}).get("batch_size", 128))
    vecs = []
    total = len(items)
    for i in range(0, total, batch_size):
        chunk = items[i : i + batch_size]
        imgs = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            imgs.append(sb.preprocess(im))
        feats = sb.encode_images(torch_stack(imgs)).cpu().numpy()
        vecs.append(feats)
        if (i // batch_size) % 10 == 0 or i + batch_size >= total:
            print(f"[search/index_crops] embedded {min(i + batch_size, total)}/{total}")

    emb = np.vstack(vecs).astype(np.float32)
    paths = [str(p) for p in items]

    # Build index
    icfg = cfg.get("index", {})
    vindex = VectorIndex(
        sb.out_dim,
        IndexConfig(
            index_type=icfg.get("type", "auto"),
            nlist=int(icfg.get("nlist", 1024)),
            m=int(icfg.get("m", 64)),
            nprobe=int(icfg.get("nprobe", 16)),
        ),
    )
    vindex.build(emb, paths)
    out_dir = Path(args.out).resolve()
    vindex.save(out_dir)
    print(f"[search/index_crops] Saved index to {out_dir}")


if __name__ == "__main__":
    main()
