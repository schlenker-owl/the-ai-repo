#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from airoad.core import pick_device
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="HF id or local path (merged ok)")
    ap.add_argument("--adapters", type=str, default="", help="LoRA adapters dir (optional)")
    ap.add_argument("--prompt", type=str, default="Give me 3 creative macOS AI app names.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    dev = pick_device(None)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(dev).eval()

    if args.adapters:
        assert _HAS_PEFT, "peft not installed"
        model = PeftModel.from_pretrained(model, args.adapters).to(dev).eval()

    text = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{args.prompt}\n<|assistant|>\n"
    enc = tok(text, return_tensors="pt").to(dev)
    out = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip())


if __name__ == "__main__":
    main()
