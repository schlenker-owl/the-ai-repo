#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

BANNED = [
    "Human:",
    "Machine:",
    "<|user|>",
    "<|system|>",
    "<|assistant|>",
    "As an AI language model",
    "### Instruction:",
    "### Response:",
]

SYS_DEFAULT = "You are a compassionate, practical spiritual coach. Be concise, kind, and useful."


def load_prompts_from_chatml(path: str, limit: int) -> List[str]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    prompts: List[str] = []
    for r in rows:
        msgs = r.get("messages", [])
        us = [m["content"] for m in msgs if m.get("role") == "user"]
        if us:
            prompts.append(us[-1])
        if limit and len(prompts) >= limit:
            break
    return prompts


def build_prompt(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(
    model, tok, system: str, user: str, dev, max_new: int, temp: float, top_p: float
) -> Tuple[str, float, int]:
    prompt = build_prompt(tok, system, user)
    enc = tok(prompt, return_tensors="pt").to(dev)
    t0 = time.perf_counter()
    out = model.generate(
        **enc,
        do_sample=(temp > 0),
        temperature=temp,
        top_p=top_p,
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    dt = time.perf_counter() - t0
    decoded = tok.decode(out[0], skip_special_tokens=True)
    new_tokens = int(out.shape[1] - enc["input_ids"].shape[1])
    return decoded.strip(), dt, new_tokens


def score_markdown(ans: str) -> Dict[str, float]:
    lines = ans.splitlines()
    has_heading = any(re.match(r"^\s*#{1,3}\s+\S", ln) for ln in lines)
    has_list = any(
        re.match(r"^\s*-\s+\S", ln) or re.match(r"^\s*\d+[\).\s]\s+\S", ln) for ln in lines
    )
    has_bold = bool(re.search(r"\*\*[^*]{2,}\*\*", ans))
    banned_hits = [b for b in BANNED if b in ans]
    length_tokens_approx = len(ans.split())
    return {
        "has_heading": float(has_heading),
        "has_list": float(has_list),
        "has_bold": float(has_bold),
        "banned_hits": float(len(banned_hits)),
        "len_tokens": float(length_tokens_approx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapters", default="")
    ap.add_argument("--tuned", default="")
    ap.add_argument("--prompts", default="data/sft/spirit_chatml.jsonl", help="ChatML jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--system", default=SYS_DEFAULT)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", default="outputs/style_eval.json")
    args = ap.parse_args()

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base_m = (
        AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32).to(dev).eval()
    )
    if args.adapters:
        assert _HAS_PEFT, "peft not installed"
        tuned = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32).to(dev),
            args.adapters,
        ).eval()
        tuned_label = f"{args.base}+adapters:{Path(args.adapters).name}"
    elif args.tuned:
        tuned = (
            AutoModelForCausalLM.from_pretrained(args.tuned, torch_dtype=torch.float32)
            .to(dev)
            .eval()
        )
        tuned_label = f"merged:{Path(args.tuned).name}"
    else:
        tuned = base_m
        tuned_label = "(same as base)"

    prompts = load_prompts_from_chatml(args.prompts, args.n)
    results: Dict[str, List[Dict]] = {"base": [], "tuned": []}

    for user in prompts:
        for name, model in (("base", base_m), ("tuned", tuned)):
            ans, dt, ntok = generate(
                model, tok, args.system, user, dev, args.max_new, args.temp, args.top_p
            )
            s = score_markdown(ans)
            row = {"user": user, "answer": ans, "time": dt, "new_tokens": ntok}
            row.update(s)
            results[name].append(row)

    def agg(rows: List[Dict]) -> Dict[str, float]:
        if not rows:
            return {}
        n = len(rows)

        def avg(key: str) -> float:
            return sum(float(r[key]) for r in rows) / n

        return {
            "n": n,
            "pct_heading": 100 * avg("has_heading"),
            "pct_list": 100 * avg("has_list"),
            "pct_bold": 100 * avg("has_bold"),
            "avg_len_tokens": avg("len_tokens"),
            "avg_banned_hits": avg("banned_hits"),
            "avg_tokens_per_s": sum(r["new_tokens"] / max(r["time"], 1e-6) for r in rows) / n,
        }

    summary = {
        "base": agg(results["base"]),
        "tuned": agg(results["tuned"]),
        "labels": {"base": args.base, "tuned": tuned_label},
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"summary": summary, "samples": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"✅ wrote style eval -> {outp}")


if __name__ == "__main__":
    main()
