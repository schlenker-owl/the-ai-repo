#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="e.g., Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapters", type=str, required=True, help="outputs/qwen05b_lora")
    ap.add_argument("--out", type=str, default="checkpoints/qwen05b_merged")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(args.base)
    peft_model = PeftModel.from_pretrained(base, args.adapters)
    merged = peft_model.merge_and_unload()  # bake LoRA into base weights
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base, use_fast=True).save_pretrained(args.out)
    print(f"✅ merged checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
