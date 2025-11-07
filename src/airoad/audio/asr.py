from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf  # type: ignore


@dataclass
class Segment:
    start: float
    end: float
    text: str


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, int(sr)


def whisper_transcribe(
    path: Path, model_size: str = "base", device: str | None = None
) -> List[Segment]:
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "whisper not installed. Install: uv add --group audio openai-whisper"
        ) from e
    model = whisper.load_model(model_size, device=device or "cpu")
    result = model.transcribe(str(path))
    out: List[Segment] = []
    for seg in result.get("segments", []):
        out.append(
            Segment(start=float(seg["start"]), end=float(seg["end"]), text=str(seg["text"]).strip())
        )
    return out
