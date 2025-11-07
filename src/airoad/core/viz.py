from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """
    Accepts RGB or BGR HxWxC [0..255] uint8, returns BGR.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HxWx3 image")
    # Heuristic: if converting RGB->BGR same as flipping channels
    return img[:, :, ::-1] if np.mean(img[:, :, 0]) < np.mean(img[:, :, 2]) else img


def save_image_grid(
    images: Iterable[np.ndarray], out_path: Path | str, max_cols: int = 4, pad: int = 2
) -> None:
    imgs = [np.ascontiguousarray(_to_bgr(im).astype(np.uint8)) for im in images]
    if not imgs:
        raise ValueError("No images to grid")

    H = max(im.shape[0] for im in imgs)
    W = max(im.shape[1] for im in imgs)

    # resize to same size
    resized = [cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in imgs]

    cols = min(max_cols, len(resized))
    rows = ceil(len(resized) / cols)
    canvas = np.full((rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad, 3), 20, np.uint8)

    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= len(resized):
                break
            y0 = r * H + (r + 1) * pad
            x0 = c * W + (c + 1) * pad
            canvas[y0 : y0 + H, x0 : x0 + W] = resized[k]
            k += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
