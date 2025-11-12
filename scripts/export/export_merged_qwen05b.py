#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from airoad.sft.infer import merge_and_save_lora  # uses your existing helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF model id (e.g., Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument(
        "--adapters", required=True, help="LoRA adapters dir (e.g., outputs/qwen05b_fast_plain)"
    )
    ap.add_argument(
        "--out", default="checkpoints/qwen05b_merged", help="Output checkpoint directory"
    )
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_dir = merge_and_save_lora(args.base, args.adapters, args.out)
    print(f"✅ merged checkpoint written to: {out_dir}")


if __name__ == "__main__":
    main()
