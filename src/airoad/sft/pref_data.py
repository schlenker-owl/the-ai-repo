#!/usr/bin/env python
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from transformers import PreTrainedTokenizerBase

from datasets import Dataset as HFDataset

"""
src/airoad/sft/pref_data.py

Preference data utilities (DPO/ORPO/KTO):
- build_dpo_dataset (pairs: prompt, chosen, rejected)
- build_kto_dataset (chosen + synthetic negatives; fields: prompt, completion, label)
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


def make_chatml_prompt(tok: PreTrainedTokenizerBase, system: str, user: str) -> str:
    """Build a Qwen ChatML prefill up to the assistant turn."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -------- DPO/ORPO (pairs) ---------------------------------------------


def records_to_pairs(
    rows: List[Dict[str, Any]], tok: PreTrainedTokenizerBase, system_prompt: str
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for r in rows:
        if "prompt" in r and "chosen" in r and "rejected" in r:
            p, c, j = r["prompt"], r["chosen"], r["rejected"]
        elif "messages" in r:
            msgs = r["messages"]
            sys = next((m["content"] for m in msgs if m.get("role") == "system"), system_prompt)
            usr = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            chosen = r.get("chosen")
            if chosen is None:
                chosen = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            rejected = r.get("rejected", "")
            p = make_chatml_prompt(tok, sys, usr)
            c, j = chosen, rejected
        else:
            continue
        if not p or not c or not j:
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


# -------- KTO (chosen + synthetic negatives) ----------------------------


def _neg_from_completion(text: str, mode: str) -> str:
    """
    Create a quick 'worse' answer from a good completion.
    - prefix: keep only the first sentence
    - truncate: keep only the first ~60 chars or first line
    """
    s = text.strip()
    if not s:
        return ""
    if mode == "prefix":
        parts = re.split(r"(?<=[.!?])\s+", s)
        return parts[0].strip()
    # truncate
    first_line = s.splitlines()[0].strip()
    if len(first_line) > 60:
        first_line = first_line[:60].rstrip()
    return first_line


def records_to_kto_items(
    rows: List[Dict[str, Any]],
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
    neg_mode: str,
    neg_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Return items with keys required by KTOTrainer:
      {"prompt": str, "completion": str, "label": int}
    Generates negatives (label=0) alongside positives (label=1) to avoid degenerate gradients.
    """
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []

    # collect positives (label=1)
    positives: List[Dict[str, str]] = []
    for r in rows:
        if "prompt" in r and "chosen" in r and "rejected" not in r:
            p, c = r["prompt"], r["chosen"]
        elif "messages" in r:
            msgs = r["messages"]
            sys = next((m["content"] for m in msgs if m.get("role") == "system"), system_prompt)
            usr = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            c = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            p = make_chatml_prompt(tok, sys, usr)
        else:
            continue
        if not p or not c:
            continue
        positives.append({"prompt": p, "completion": c.strip()})

    # decide how many negatives to add
    # simple scheme: at most int(neg_ratio * len(positives)) negatives
    n_pos = len(positives)
    n_neg_target = int(max(0, round(neg_ratio * n_pos)))

    # shuffle positives to pick a subset for negatives
    idx = list(range(n_pos))
    rng.shuffle(idx)
    idx = idx[:n_neg_target]

    negatives: List[Dict[str, str]] = []
    for i in idx:
        p = positives[i]["prompt"]
        c = positives[i]["completion"]
        neg_c = _neg_from_completion(c, neg_mode)
        if not neg_c:
            continue
        # skip degenerate case where neg == pos (single-sentence answers)
        if neg_c.strip() == c.strip():
            continue
        negatives.append({"prompt": p, "completion": neg_c.strip()})

    # Build output with labels
    for ex in positives:
        items.append({"prompt": ex["prompt"], "completion": ex["completion"], "label": 1})
    for ex in negatives:
        items.append({"prompt": ex["prompt"], "completion": ex["completion"], "label": 0})

    return items


def build_kto_dataset(
    dataset_path: str,
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
    ablate_examples: int = 0,
    ablate_seed: int = 42,
    kto_neg_mode: str = "prefix",  # "prefix" | "truncate"
    kto_neg_ratio: float = 1.0,  # negatives per positive (1.0 = 1:1)
) -> HFDataset:
    rows = read_jsonl(Path(dataset_path))
    items = records_to_kto_items(
        rows,
        tok,
        system_prompt,
        neg_mode=kto_neg_mode,
        neg_ratio=kto_neg_ratio,
        seed=ablate_seed,
    )
    items = maybe_subset(items, ablate_examples, ablate_seed)
    ds = HFDataset.from_list(items)
    return ds.with_format("python")
