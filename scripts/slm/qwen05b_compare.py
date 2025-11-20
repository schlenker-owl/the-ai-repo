#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch

from airoad.core import pick_device
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# -----------------------------
# Built-in prompts (ChatML "user" content) that mirror training style
# -----------------------------
DEFAULT_PROMPTS: List[str] = [
    "Offer three morning affirmations to start the day aligned.",
    "Reframe this thought into something more allowing.\n\nI’m behind in life.",
    "Give a 60-second reset during conflict.",
    "Explain ‘allowing mode’ with a metaphor.",
    "Create a short ‘rampage of appreciation’ about my body.",
    "Give a 7-day alignment plan (short bullets).",
    "Provide a mantra for creative flow.",
    "Design a 3-minute alignment check before meetings.",
    "Give a scripting exercise for a goal 30 days out.\n\nI want steady clients.",
    "Provide 10 ‘better-feeling’ replacements.\n\nList: overwhelmed, stuck, behind, lonely, broke, confused, guilty, rushed, doubtful, resentful",
    "Create a gratitude micro-practice for meals.",
    "Coach me to act without overthinking.",
    "Write three relationship intentions.",
    "Turn fear of the unknown into curiosity.",
    "Offer an abundance visualization (2 minutes).",
    "Give compassionate boundaries script.\n\nA colleague keeps DM’ing late at night.",
    "Provide a daily ‘vibration tune-up’ checklist.",
    "Write a gentle evening release ritual.",
    "Offer three grounding practices without equipment.",
    "Coach me through jealousy with compassion.",
    "Rewrite this belief to be less resistant.\n\nIf I rest, I’ll fall behind.",
]

SYSTEM_DEFAULT = "You are a compassionate, practical spiritual coach. Be concise, kind, and useful."


# ANSI colors (fallback if not a TTY)
def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


BOLD = "1"
CYAN = "36"
GREEN = "32"
GREY = "90"


def build_chat_prompt(tok: AutoTokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_assistant(decoded: str) -> str:
    # Keep only the first assistant turn, strip any accidental next tags
    out = decoded
    if "<|assistant|>" in out:
        out = out.split("<|assistant|>", 1)[-1]
    # stop at any subsequent tag if present
    for tag in (
        "<|user|>",
        "<|system|>",
        "### Instruction:",
        "### Response:",
        "Human:",
        "Machine:",
    ):
        if tag in out:
            out = out.split(tag, 1)[0]
    return out.strip()


@torch.no_grad()
def gen_once(model, tok, system: str, user: str, dev, max_new: int, temp: float, top_p: float):
    prompt = build_chat_prompt(tok, system, user)
    enc = tok(prompt, return_tensors="pt").to(dev)
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        do_sample=(temp > 0.0),
        temperature=temp,
        top_p=top_p,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    dt = time.perf_counter() - t0
    decoded = tok.decode(out[0], skip_special_tokens=True)
    ans = decoded.strip()
    gen_tokens = int(out.shape[1] - enc["input_ids"].shape[1])
    return ans, dt, gen_tokens


def load_prompts_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".jsonl":
        # Expect ChatML records with {"messages":[...]} or plain {"user": "..."} style
        prompts: List[str] = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if "messages" in obj:
                # pull the last user content or the first user content
                msgs = obj["messages"]
                user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
                prompts.append(user_msgs[-1] if user_msgs else "")
            else:
                prompts.append(obj.get("user", ""))
        return [q for q in prompts if q.strip()]
    else:
        # plaintext: one prompt per line; blank lines ignored
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF id or local path for the BASE model",
    )
    ap.add_argument("--adapters", default="", help="LoRA adapters dir for TUNED model (optional)")
    ap.add_argument(
        "--tuned",
        default="",
        help="Path to a merged TUNED checkpoint (optional; ignored if --adapters is set)",
    )
    ap.add_argument("--system", default=SYSTEM_DEFAULT, help="system prompt")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--prompts-file",
        default="",
        help="Optional file with prompts (txt lines or jsonl w/ ChatML). If omitted, use built-in set.",
    )
    args = ap.parse_args()

    dev = pick_device(None)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Load BASE model
    base_m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(dev).eval()

    # Load TUNED model: adapters > merged > base (fallback)
    if args.adapters:
        assert _HAS_PEFT, "peft is not installed; cannot load adapters"
        tuned_base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(dev)
        tuned_m = PeftModel.from_pretrained(tuned_base, args.adapters).eval()
        tuned_label = f"{args.base} + adapters:{Path(args.adapters).name}"
    elif args.tuned:
        tuned_m = (
            AutoModelForCausalLM.from_pretrained(args.tuned, dtype=torch.float32).to(dev).eval()
        )
        tuned_label = f"merged:{Path(args.tuned).name}"
    else:
        tuned_m = base_m  # fallback for convenience
        tuned_label = "(same as base)"

    # Prompts
    if args.prompts_file:
        prompts = load_prompts_file(args.prompts_file)
    else:
        prompts = DEFAULT_PROMPTS

    # Pretty header
    line = "=" * 96
    print(line)
    print(_color("Qwen-0.5B Compare (ChatML) — Base vs Tuned", BOLD))
    print(f"System: {args.system}")
    print(f"Base : {args.base}")
    print(f"Tuned: {tuned_label}")
    print(f"Gen  : temp={args.temp} top_p={args.top_p} max_new={args.max_new}")
    print(line)

    # Compare
    for i, p in enumerate(prompts, 1):
        print(_color(f"\n[{i}/{len(prompts)}] PROMPT", BOLD), "—", p)
        # Base
        base_txt, base_dt, base_ntok = gen_once(
            base_m, tok, args.system, p, dev, args.max_new, args.temp, args.top_p
        )
        # Tuned
        tuned_txt, tuned_dt, tuned_ntok = gen_once(
            tuned_m, tok, args.system, p, dev, args.max_new, args.temp, args.top_p
        )

        # Print clearly
        print(_color("\n-- BASE -------------------------------", CYAN))
        print(base_txt if base_txt else _color("[empty output]", GREY))
        print(
            _color(
                f"[time {base_dt:.2f}s | new_tokens {base_ntok} | {base_ntok/max(base_dt,1e-6):.1f} tok/s]",
                GREY,
            )
        )

        print(_color("\n-- TUNED ------------------------------", GREEN))
        print(tuned_txt if tuned_txt else _color("[empty output]", GREY))
        print(
            _color(
                f"[time {tuned_dt:.2f}s | new_tokens {tuned_ntok} | {tuned_ntok/max(tuned_dt,1e-6):.1f} tok/s]",
                GREY,
            )
        )
        print(line)


if __name__ == "__main__":
    main()
