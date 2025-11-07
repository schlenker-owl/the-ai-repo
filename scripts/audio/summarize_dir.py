#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from airoad.audio.summarize import summarize_transcript
from airoad.core.io import ensure_dir
from airoad.core.manifest import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl", type=str, required=True, help="transcripts.jsonl from transcribe_dir"
    )
    ap.add_argument("--out", type=str, default="outputs/audio/summaries")
    args = ap.parse_args()

    records = []
    for line in Path(args.jsonl).read_text().splitlines():
        records.append(json.loads(line))

    out_dir = ensure_dir(Path(args.out).resolve())
    md_lines = ["# Audio Summaries", ""]
    for rec in records:
        text = "\n".join(
            f"[{s['start']:.2f}→{s['end']:.2f}] {s['text']}" for s in rec.get("segments", [])
        )
        summ = summarize_transcript(text)
        md_lines.append(f"## {rec['path']}\n\n{summ}\n")

    md_path = out_dir / "summaries.md"
    md_path.write_text("\n".join(md_lines))

    write_manifest(
        out_dir,
        "audio/summarize_dir",
        "0.1.0",
        None,
        [args.jsonl],
        {"md": str(md_path)},
        None,
        "cpu",
    )
    print(f"[audio/summarize_dir] wrote {md_path}")


if __name__ == "__main__":
    main()
