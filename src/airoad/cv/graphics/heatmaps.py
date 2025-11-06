from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def _first_frame(path: str | Path) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def save_activity_heatmap(
    tracks: pd.DataFrame,
    out_png: str | Path,
    bg_image_path: str | Path | None = None,
    frame_size_hint: Tuple[int, int] | None = None,  # (w,h) if no bg available
    bins: int = 64,
    alpha: float = 0.45,
) -> str:
    """
    Build an activity heatmap from cx,cy and overlay on a frame background.

    - If bg_image_path (e.g., annotated video) is provided, we grab its first frame for background.
    - Else we use a blank canvas sized by frame_size_hint or by the max in tracks.
    """
    out_png = str(out_png)

    # infer width/height if needed
    if bg_image_path:
        bg = _first_frame(bg_image_path)
    else:
        bg = None

    if bg is not None:
        H, W = bg.shape[:2]
    else:
        if frame_size_hint:
            W, H = frame_size_hint
        else:
            # fall back to max coords
            W = int(np.nanmax(tracks["cx"].values) + 1) if "cx" in tracks else 1280
            H = int(np.nanmax(tracks["cy"].values) + 1) if "cy" in tracks else 720
        bg = np.full((H, W, 3), 230, dtype=np.uint8)

    # collect coordinates
    if "cx" not in tracks or "cy" not in tracks or tracks.empty:
        cv2.imwrite(out_png, bg)
        return out_png

    xs = tracks["cx"].astype(float).values
    ys = tracks["cy"].astype(float).values

    # clamp to frame
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    # 2D histogram
    heat, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=[[0, W], [0, H]])
    heat = heat.T  # to match image coordinates

    # normalize to 0..255
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_img = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    heat_resized = cv2.resize(heat_color, (W, H), interpolation=cv2.INTER_CUBIC)

    overlay = cv2.addWeighted(bg, 1.0 - alpha, heat_resized, alpha, 0)
    cv2.imwrite(out_png, overlay)
    return out_png
