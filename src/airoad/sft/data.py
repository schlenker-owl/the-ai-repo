#!/usr/bin/env python
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import PreTrainedTokenizerBase

from datasets import Dataset as HFDataset

"""
src/airoad/sft/data.py

Data utilities for SFT/LoRA runs:
- Load ChatML or Alpaca JSONL
- Convert to Qwen chat-template strings
- Optional ablation subset
- Build HF datasets
- Approximate token counts for throughput metrics
"""

# -------------------------------
# General helpers
# -------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Returns [] if file missing."""
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def ensure_tok(tok: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Pad-left decoder-only tokenizer and ensure pad_token is set."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def maybe_subset(items: Sequence[Any], n: int, seed: int) -> List[Any]:
    """Optionally keep a shuffled subset of size n. If n<=0, returns full list."""
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [items[i] for i in idx]


# -------------------------------
# Conversions → Qwen ChatML strings
# -------------------------------


def _messages_from_alpaca_row(row: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    """Convert an Alpaca triple into a ChatML messages list."""
    instr = (row.get("instruction") or "").strip()
    ctx = (row.get("input") or "").strip()
    ans = (row.get("output") or "").strip()
    user = instr if not ctx else f"{instr}\n\n{ctx}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ans},
    ]


def records_to_texts(
    rows: List[Dict[str, Any]],
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
) -> List[str]:
    """
    Convert records (ChatML or Alpaca) into **Qwen chat-template strings**.

    If a ChatML record lacks a system message, we inject the provided system_prompt.
    """
    texts: List[str] = []
    for r in rows:
        if "messages" in r:
            messages = r["messages"]
            if not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages = _messages_from_alpaca_row(r, system_prompt)

        s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(s)
    return texts


# -------------------------------
# Dataset builders + token counts
# -------------------------------


def build_dataset(texts: List[str]) -> HFDataset:
    """
    Build a HuggingFace Dataset with python formatting (stable for TRL/HF trainers).
    """
    ds = HFDataset.from_dict({"text": texts})
    return ds.with_format("python")


def approx_token_count(
    texts: List[str],
    tok: PreTrainedTokenizerBase,
    max_len: int,
    add_special_tokens: bool = True,
) -> int:
    """
    Approximate the number of tokens processed for reporting throughput.

    NOTE: this is a single-pass approximation (does not multiply by steps/accum).
    """
    enc = tok(
        texts,
        truncation=True,
        max_length=max_len,
        add_special_tokens=add_special_tokens,
    )
    return int(sum(len(ids) for ids in enc["input_ids"]))


# -------------------------------
# High-level loader used by CLI
# -------------------------------


def load_train_texts(
    dataset_path: str,
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
    ablate_examples: int = 0,
    ablate_seed: int = 42,
) -> List[str]:
    """
    Load JSONL at dataset_path (ChatML or Alpaca) and convert to chat-template strings.
    Optionally keep a shuffled subset for fast ablations.
    """
    rows = read_jsonl(Path(dataset_path))
    if not rows:
        return []

    texts = records_to_texts(rows, tok, system_prompt)
    texts = maybe_subset(texts, ablate_examples, ablate_seed)
    return texts


def load_eval_texts(
    dev_path: Optional[str],
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
) -> Optional[List[str]]:
    """
    Load a dev set (if provided) and convert to chat-template strings.
    Returns None if dev_path is None or file missing.
    """
    if not dev_path:
        return None
    rows = read_jsonl(Path(dev_path))
    if not rows:
        return None
    return records_to_texts(rows, tok, system_prompt)


# -------------------------------
# Convenience orchestration
# -------------------------------


def prepare_datasets(
    dataset_path: str,
    tok: PreTrainedTokenizerBase,
    system_prompt: str,
    max_len: int,
    ablate_examples: int = 0,
    ablate_seed: int = 42,
    dev_path: Optional[str] = None,
) -> Tuple[HFDataset, Optional[HFDataset], int]:
    """
    One-shot helper:
      - load & format train texts (with optional ablation)
      - build train HF dataset
      - load & format eval texts (optional)
      - build eval HF dataset (optional)
      - return approx train token count for reporting

    Returns:
      (train_ds, eval_ds, approx_tokens)
    """
    texts = load_train_texts(
        dataset_path=dataset_path,
        tok=tok,
        system_prompt=system_prompt,
        ablate_examples=ablate_examples,
        ablate_seed=ablate_seed,
    )
    train_ds = build_dataset(texts)
    approx_tokens = approx_token_count(texts, tok, max_len)

    eval_ds = None
    eval_texts = load_eval_texts(dev_path, tok, system_prompt)
    if eval_texts:
        eval_ds = build_dataset(eval_texts)

    return train_ds, eval_ds, approx_tokens
