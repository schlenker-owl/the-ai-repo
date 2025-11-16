#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_prompt(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(model, tok, prompt: str, dev, max_new=128, temp=0.7, top_p=0.9) -> str:
    enc = tok(prompt, return_tensors="pt").to(dev)
    out = model.generate(
        **enc,
        do_sample=(temp > 0),
        temperature=temp,
        top_p=top_p,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    # keep only assistant answer (after prompt)
    ans = txt[len(tok.decode(enc["input_ids"][0], skip_special_tokens=True)) :].strip()
    return ans


def make_rejected(chosen: str, mode: str) -> str:
    if mode == "truncate":
        return chosen.split("\n")[0].split(".")[0][:80].strip()
    if mode == "prefix":
        first = re.split(r"(?<=[.!?])\s+", chosen.strip())[0]
        return first.strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile", required=True, help="ChatML JSONL with messages[..] and assistant gold"
    )
    ap.add_argument(
        "--out", required=True, help="Output DPO pairs JSONL with keys: prompt, chosen, rejected"
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap examples")
    ap.add_argument("--seed", type=int, default=42)

    # negative generation mode
    ap.add_argument(
        "--neg", choices=["truncate", "prefix", "gen_base", "gen_adapters"], default="truncate"
    )

    # model args for gen_* modes
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapters", default="", help="PEFT adapters dir for gen_adapters")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(Path(args.infile))
    if args.limit > 0:
        rows = rows[: args.limit]

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = None
    if args.neg in ("gen_base", "gen_adapters"):
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(dev).eval()
        if args.neg == "gen_adapters":
            assert _HAS_PEFT, "peft not installed for gen_adapters"
            model = PeftModel.from_pretrained(model, args.adapters).eval()

    pairs = []
    for r in rows:
        if "messages" not in r:
            continue
        msgs = r["messages"]
        sys = next(
            (m["content"] for m in msgs if m.get("role") == "system"),
            "You are a helpful assistant.",
        )
        usr = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        chosen = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if not usr or not chosen:
            continue

        prompt = build_prompt(tok, sys, usr)

        if args.neg in ("truncate", "prefix"):
            rejected = make_rejected(chosen, args.neg)
        else:
            # generate a competing (usually worse) answer
            rejected = generate(model, tok, prompt, dev, args.max_new, args.temp, args.top_p)  # type: ignore[arg-type]

        # skip degenerate cases
        if not rejected or rejected.strip() == chosen.strip():
            continue

        pairs.append({"prompt": prompt, "chosen": chosen.strip(), "rejected": rejected.strip()})

    write_jsonl(Path(args.out), pairs)
    print(f"✅ wrote DPO pairs -> {args.out} (n={len(pairs)})")


if __name__ == "__main__":
    main()
