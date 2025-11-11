#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

HDR = "You are a compassionate, practical spiritual coach. Be concise, kind, and useful."


def as_md_heading(s: str) -> str:
    s = s.strip().splitlines()[0]
    s = re.sub(r"[^\S\r\n]+", " ", s).strip()
    # guard long headings
    return "## " + (s[:80] + ("…" if len(s) > 80 else ""))


def to_bullets(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return text.strip()
    # if lines already look like bullet/steps, normalize
    bulletish = any(ln[:2] in ("- ", "* ") or re.match(r"^\d+[\).\s]\s", ln) for ln in lines)
    if bulletish:
        out = []
        for ln in lines:
            ln = re.sub(r"^\s*[-*]\s*", "", ln)
            ln = re.sub(r"^\s*\d+[\).\s]\s*", "", ln)
            out.append(f"- {ln}")
        return "\n".join(out)
    # else chunk by sentences into bullets
    parts = re.split(r"(?<=[.!?])\s+", " ".join(lines))
    parts = [p.strip() for p in parts if p.strip()]
    return "\n".join(f"- {p}" for p in parts)


def convert_record(rec: Dict) -> Dict:
    assert "messages" in rec and isinstance(rec["messages"], list)
    system = None
    user = None
    assistant = None
    for m in rec["messages"]:
        role = m.get("role")
        if role == "system":
            system = m.get("content", "")
        if role == "user":
            user = m.get("content", "")
        if role == "assistant":
            assistant = m.get("content", "")
    system = system or HDR
    user_h = as_md_heading(user or "Alignment")
    md = f"{user_h}\n\n{to_bullets(assistant or '')}".strip()
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user or ""},
            {"role": "assistant", "content": md},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="ChatML jsonl (messages[..])")
    ap.add_argument("--out", required=True, help="Output markdown-styled ChatML jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap of examples")
    args = ap.parse_args()

    src = Path(args.infile).read_text().splitlines()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in src if x.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    with outp.open("w") as f:
        for r in rows:
            f.write(json.dumps(convert_record(r), ensure_ascii=False) + "\n")

    print(f"✅ wrote markdown slice -> {outp} (n={len(rows)})")


if __name__ == "__main__":
    main()
