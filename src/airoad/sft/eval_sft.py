from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel  # type: ignore

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

try:
    import wandb  # type: ignore

    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# Keep the lightweight batch metrics for pipeline evals
try:
    from .metrics import compute_metrics  # type: ignore
except Exception:
    compute_metrics = None  # type: ignore

# --------------------------------------------------------------------------------------
# Pairwise metrics required by tests: exact_match, rouge_l_f, evaluate_pairs
# --------------------------------------------------------------------------------------

_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().translate(_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, ref: str) -> float:
    """Pairwise exact match after normalization."""
    return 1.0 if _normalize_text(pred) == _normalize_text(ref) else 0.0


def _lcs_len(a_tokens: List[str], b_tokens: List[str]) -> int:
    """Classic LCS DP used for ROUGE-L."""
    n, m = len(a_tokens), len(b_tokens)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f(pred: str, ref: str) -> float:
    """Pairwise ROUGE-L F1 on whitespace tokens, normalized."""
    a = _normalize_text(pred).split()
    b = _normalize_text(ref).split()
    if not a or not b:
        return 0.0
    lcs = _lcs_len(a, b)
    prec = lcs / max(1, len(a))
    rec = lcs / max(1, len(b))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def evaluate_pairs(pairs: Iterable[Tuple[str, str]]) -> dict:
    """Average pairwise EM and ROUGE-L F1 across (pred, ref) pairs."""
    pairs_list = list(pairs)
    if not pairs_list:
        return {"exact_match": 0.0, "rougeL_f": 0.0}
    em = sum(exact_match(p, r) for p, r in pairs_list) / len(pairs_list)
    rl = sum(rouge_l_f(p, r) for p, r in pairs_list) / len(pairs_list)
    return {"exact_match": em, "rougeL_f": rl}


# --------------------------------------------------------------------------------------
# Generation-based evaluator used by training scripts (kept as-is, improved padding)
# --------------------------------------------------------------------------------------

PROMPT_TPL = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        # small default toy set
        return [
            {"instruction": "Answer briefly: capital of France?", "input": "", "output": "paris"},
            {"instruction": "Reverse the word", "input": "stressed", "output": "desserts"},
            {"instruction": "Compute 2+5", "input": "", "output": "7"},
            {"instruction": "Synonym for quick", "input": "", "output": "fast"},
            {"instruction": "Lowercase this", "input": "Hello WORLD", "output": "hello world"},
        ]
    out: List[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _format_prompts(rows: Iterable[dict]) -> Tuple[List[str], List[str]]:
    prompts, refs = [], []
    for r in rows:
        inst = r.get("instruction", "")
        inp = r.get("input", "")
        out = r.get("output", "")
        prompts.append(PROMPT_TPL.format(instruction=inst, input=inp))
        refs.append(out)
    return prompts, refs


def _load_model(model_path: str, base_model: str | None, device: torch.device):
    """Load model + tokenizer. Left-pad for decoder-only models to avoid generation issues."""
    is_adapter = (Path(model_path) / "adapter_config.json").exists()

    if is_adapter and _HAS_PEFT and base_model:
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
        model = PeftModel.from_pretrained(base, model_path)
        try:
            model = model.merge_and_unload()  # type: ignore[attr-defined]
        except Exception:
            pass
        is_encdec = bool(getattr(model.config, "is_encoder_decoder", False))
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        is_encdec = bool(getattr(model.config, "is_encoder_decoder", False))
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not is_encdec:
        tok.padding_side = "left"

    model.to(device)
    model.eval()
    return tok, model


@torch.inference_mode()
def _generate(
    model,
    tok,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    batch_size: int = 4,
) -> List[str]:
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for EM
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen_text = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        outs.extend([g.strip() for g in gen_text])
    return outs


def evaluate_and_print(
    model_path: str,
    base_model: str | None = None,
    dev_path: str = "data/sft/dev_toy.jsonl",
    limit: int | None = 20,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    log_wandb: bool = False,
    wandb_project: str = "airepo-sft",
) -> dict:
    """
    Batch evaluator used by training scripts; prints EM/ROUGE-L via the lightweight metrics module if available.
    """
    device = _device()
    rows = _load_jsonl(Path(dev_path))
    if limit is not None:
        rows = rows[:limit]
    prompts, refs = _format_prompts(rows)

    tok, model = _load_model(model_path, base_model, device)
    preds = _generate(
        model, tok, prompts, device, max_new_tokens=max_new_tokens, batch_size=batch_size
    )

    if compute_metrics is not None:
        metrics = compute_metrics(preds, refs)  # {"em": ..., "rougeL": ...}
        printable = {"exact_match": metrics.get("em", 0.0), "rougeL_f": metrics.get("rougeL", 0.0)}
    else:
        # Fallback using pairwise fns
        pairs = list(zip(preds, refs))
        printable = evaluate_pairs(pairs)

    print("\n[eval_sft] Results")
    for k, v in printable.items():
        print(f"  {k}: {v:.4f}")

    if log_wandb and _HAS_WANDB:
        if not wandb.run:
            wandb.init(
                project=wandb_project, config={"model_path": model_path, "base_model": base_model}
            )
        wandb.log({f"eval/{k}": v for k, v in printable.items()})

    return printable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--dev-path", type=str, default="data/sft/dev_toy.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--log-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="airepo-sft")
    args = ap.parse_args()

    evaluate_and_print(
        model_path=args.model_path,
        base_model=args.base_model,
        dev_path=args.dev_path,
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        log_wandb=args.log_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
