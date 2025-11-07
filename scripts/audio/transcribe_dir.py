#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from airoad.audio.asr import whisper_transcribe
from airoad.core.io import ensure_dir
from airoad.core.manifest import write_manifest

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _iter_audio(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/audio/transcripts")
    ap.add_argument("--model", type=str, default="base")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    inp = Path(args.inp).resolve()
    out_dir = ensure_dir(Path(args.out).resolve())
    files = _iter_audio(inp)
    if not files:
        print("[audio/transcribe_dir] no audio files found.")
        return

    all_out = []
    for f in files:
        segs = whisper_transcribe(f, model_size=args.model, device=args.device)
        rec = {
            "path": str(f),
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segs],
        }
        all_out.append(rec)
    (out_dir / "transcripts.jsonl").write_text("\n".join(json.dumps(x) for x in all_out))

    write_manifest(
        out_dir,
        "audio/transcribe_dir",
        "0.1.0",
        None,
        [str(inp)],
        {"jsonl": str(out_dir / "transcripts.jsonl")},
        None,
        args.device or "cpu",
    )
    print(f"[audio/transcribe_dir] wrote {out_dir / 'transcripts.jsonl'}")


if __name__ == "__main__":
    main()
