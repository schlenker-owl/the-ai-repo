#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from airoad.search.backbone import ImageTextBackbone, SearchBackboneConfig
from airoad.search.indexer import VectorIndex


def _load_backbone(cfg_path: Path) -> ImageTextBackbone:
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    bcfg = cfg.get("backbone", {})
    return ImageTextBackbone(
        SearchBackboneConfig(
            name=bcfg.get("name", "auto"),
            device=bcfg.get("device", None),
            dtype=bcfg.get("dtype", "float32"),
            image_size=int(bcfg.get("image_size", 224)),
        )
    )


def _embed_image(sb: ImageTextBackbone, path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    t = sb.preprocess(im)
    import torch

    v = sb.encode_images(torch.stack([t], dim=0)).cpu().numpy()
    return v


def _embed_text(sb: ImageTextBackbone, text: str) -> np.ndarray:
    feats = (
        sb.encode_texts([text]).cpu().numpy()
        if hasattr(sb, "encode_texts")
        else sb.encode_texts([text])
    )
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True, help="index directory to load")
    ap.add_argument("--config", type=str, default="configs/cv/search.yaml")
    ap.add_argument("--text", type=str, default=None, help="text prompt (requires OpenCLIP)")
    ap.add_argument("--image", type=str, default=None, help="reference image path")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument(
        "--copy-to",
        type=str,
        default=None,
        help="copy matched files into this folder for inspection",
    )
    ap.add_argument("--json-out", type=str, default=None, help="write matches to JSON")
    args = ap.parse_args()

    if (args.text is None) == (args.image is None):
        raise ValueError("Provide exactly one of --text or --image")

    sb = _load_backbone(Path(args.config))
    vindex = VectorIndex.load(Path(args.index))

    if args.text:
        try:
            q = _embed_text(sb, args.text)
        except RuntimeError as e:
            raise RuntimeError(
                "Text queries require OpenCLIP backbone. Set backbone.name='open-clip:ViT-B-32' and install open-clip-torch."
            ) from e
    else:
        q = _embed_image(sb, Path(args.image))

    D, Index = vindex.search(q.astype(np.float32), topk=args.topk)
    idxs = Index[0].tolist()
    sims = D[0].tolist()
    paths = [vindex.paths[i] for i in idxs]

    # Print results
    print(
        json.dumps(
            [
                {"rank": r + 1, "path": p, "score": float(s)}
                for r, (p, s) in enumerate(zip(paths, sims))
            ],
            indent=2,
        )
    )

    # Optional copy for quick inspection
    if args.copy_to:
        out_dir = Path(args.copy_to).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for r, p in enumerate(paths):
            src = Path(p)
            dst = out_dir / f"{r+1:03d}_{src.name.replace('/', '_')}"
            if src.exists():
                shutil.copy2(src, dst)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                [
                    {"rank": r + 1, "path": p, "score": float(s)}
                    for r, (p, s) in enumerate(zip(paths, sims))
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
