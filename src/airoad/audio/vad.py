from __future__ import annotations

import numpy as np


def simple_vad(
    wav: np.ndarray, sr: int, frame_ms: int = 30, thr: float = 0.01
) -> list[tuple[int, int]]:
    """
    Naive energy-based VAD: returns list of (start_sample, end_sample) speech segments.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    energy = np.convolve(wav**2, np.ones(frame_len), mode="same") / frame_len
    mask = energy > thr
    segs: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_seg:
            in_seg = True
            start = i
        elif not m and in_seg:
            in_seg = False
            segs.append((start, i))
    if in_seg:
        segs.append((start, len(wav)))
    # merge tiny gaps
    merged: list[tuple[int, int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] < frame_len:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged
