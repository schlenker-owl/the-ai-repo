#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_chatml(path: Path) -> List[Dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def key_of(rec: Dict) -> str:
    msgs = rec.get("messages", [])
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    h = hashlib.sha256((user + "\n\n" + asst).encode("utf-8")).hexdigest()
    return h


def parse_input_specs(specs: List[str]) -> List[Tuple[Path, int]]:
    out = []
    for s in specs:
        if ":" in s:
            p, w = s.split(":", 1)
            out.append((Path(p), max(1, int(w))))
        else:
            out.append((Path(s), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="FILE[:weight] ... e.g. spirit_chatml.jsonl:3 spirit_chatml_md.jsonl:1 spirit_chatml_neg.jsonl:1",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max", type=int, default=0, help="optional cap")
    ap.add_argument("--dedupe", action="store_true", default=True)
    args = ap.parse_args()

    random.seed(args.seed)
    pairs = parse_input_specs(args.inputs)
    bag: List[Dict] = []

    for path, w in pairs:
        rows = load_chatml(path)
        if w > 1:
            # repeat to approximate weights
            rows = rows * w
        bag.extend(rows)

    if args.dedupe:
        uniq = {}
        for r in bag:
            uniq[key_of(r)] = r
        bag = list(uniq.values())

    if args.shuffle:
        random.shuffle(bag)

    if args.max > 0:
        bag = bag[: args.max]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        for r in bag:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ mixed -> {outp}  (n={len(bag)})")


if __name__ == "__main__":
    main()
