#!/usr/bin/env python
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from transformers import PreTrainedTokenizerBase

from datasets import Dataset as HFDataset

"""
src/airoad/sft/pref_data.py

Preference data utilities (DPO):
- Load JSONL with (prompt, chosen, rejected) or ChatML + chosen/rejected
- Build DPO string tuples using Qwen chat template for the prompt
- Optional ablation subset
- HF Dataset builder for TRL DPOTrainer
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def maybe_subset(items: Sequence[Any], n: int, seed: int) -> List[Any]:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [items[i] for i in idx]


def make_chatml_prompt(
    tok: PreTrainedTokenizerBase,
    system: str,
    user: str,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # produce prefill prompt up to assistant turn
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def records_to_pairs(
    rows: List[Dict[str, Any]],
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
) -> List[Dict[str, str]]:
    """
    Convert records to a list of dicts with keys: prompt, chosen, rejected.

    Supported input formats per row:
    - {"prompt": "...", "chosen":"...", "rejected":"..."}  (already prepared)
    - {"messages":[...], "chosen":"...", "rejected":"..."} (ChatML + labels)
    - {"messages":[... system,user,assistant=chosen], "rejected":"..."}  (infer chosen)
    """
    pairs: List[Dict[str, str]] = []

    for r in rows:
        if "prompt" in r and "chosen" in r and "rejected" in r:
            p = r["prompt"]
            c = r["chosen"]
            j = r["rejected"]
        elif "messages" in r:
            msgs = r["messages"]
            sys = next((m["content"] for m in msgs if m.get("role") == "system"), system_prompt)
            usr = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            # chosen from top-level or from assistant in messages
            chosen = r.get("chosen")
            if chosen is None:
                chosen = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            rejected = r.get("rejected", "")
            p = make_chatml_prompt(tok, sys, usr)
            c, j = chosen, rejected
        else:
            # unsupported row: skip
            continue

        if not p or not c or not j:
            # skip incomplete pairs
            continue

        pairs.append({"prompt": p, "chosen": c.strip(), "rejected": j.strip()})

    return pairs


def build_dpo_dataset(
    pairs_path: str,
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
    ablate_examples: int = 0,
    ablate_seed: int = 42,
) -> HFDataset:
    rows = read_jsonl(Path(pairs_path))
    pairs = records_to_pairs(rows, tok, system_prompt)
    pairs = maybe_subset(pairs, ablate_examples, ablate_seed)
    ds = HFDataset.from_list(pairs)
    return ds.with_format("python")
