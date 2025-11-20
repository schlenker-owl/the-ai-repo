#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel

from airoad.core import pick_device
from transformers import AutoModelForCausalLM, AutoTokenizer

TPL = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"


def load_jsonl(p: str):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def ppl(model, tok, rows, dev, max_len=1024):
    model.eval()
    nll_sum, tok_sum = 0.0, 0
    for r in rows:
        text = TPL.format(
            **{
                "instruction": r.get("instruction", ""),
                "input": r.get("input", ""),
                "output": r.get("output", ""),
            }
        )
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(dev)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            nll_sum += float(out.loss) * enc["input_ids"].size(1)
            tok_sum += enc["input_ids"].size(1)
    return math.exp(nll_sum / max(1, tok_sum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapters", default="outputs/qwen05b_lora_fast")
    ap.add_argument("--dev", default="data/sft/dev_toy.jsonl")
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    dev = pick_device(None)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(dev)
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(dev),
        args.adapters,
    )

    rows = load_jsonl(args.dev)
    print("Base  PPL:", round(ppl(base_m, tok, rows, dev, args.max_len), 3))
    print("Tuned PPL:", round(ppl(tuned, tok, rows, dev, args.max_len), 3))


if __name__ == "__main__":
    main()
