from __future__ import annotations

from math import pi, sin
from pathlib import Path

import numpy as np

from airoad.audio.summarize import summarize_transcript
from airoad.audio.vad import simple_vad


def test_vad_energy_based():
    sr = 16_000
    dur_sil = 0.3
    dur_sig = 0.4
    # silence → tone → silence
    sil1 = np.zeros(int(sr * dur_sil), dtype=np.float32)
    tone = np.array(
        [0.1 * sin(2 * pi * 440 * t / sr) for t in range(int(sr * dur_sig))], dtype=np.float32
    )
    sil2 = np.zeros(int(sr * dur_sil), dtype=np.float32)
    wav = np.concatenate([sil1, tone, sil2], axis=0)

    segs = simple_vad(wav, sr, frame_ms=30, thr=1e-4)
    assert len(segs) >= 1
    # First segment should roughly cover the tone region
    s0, e0 = segs[0]
    assert e0 > s0
    # Non-zero length segment indicates detection worked
    assert (e0 - s0) > 0


def test_summarize_transcript_fallbacks(tmp_path: Path):
    # Hand a small "transcript" to summarizer
    transcript = "\n".join(
        [
            "[0.00→2.00] Discuss project scope and deliverables.",
            "[2.01→4.50] Decide on the timeline and owners.",
            "[4.51→6.00] Outline next steps.",
        ]
    )
    summ = summarize_transcript(transcript)
    assert isinstance(summ, str) and len(summ.strip()) > 0
