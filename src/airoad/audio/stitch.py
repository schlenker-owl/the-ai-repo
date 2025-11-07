from __future__ import annotations

from typing import Iterable, List

from .asr import Segment


def to_transcript(segments: Iterable[Segment]) -> str:
    lines = []
    for s in segments:
        lines.append(f"[{s.start:7.2f} → {s.end:7.2f}] {s.text}")
    return "\n".join(lines)


def to_jsonl(segments: Iterable[Segment]) -> List[dict]:
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
