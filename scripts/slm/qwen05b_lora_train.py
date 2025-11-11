#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from airoad.core import pick_device, seed_everything
from airoad.sft.eval_sft import evaluate_and_print
from datasets import Dataset as HFDataset

try:
    from trl import SFTConfig, SFTTrainer

    _HAS_TRL = True
except Exception:
    _HAS_TRL = False

try:
    from peft import LoraConfig, get_peft_model

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

"""
Qwen-0.5B LoRA trainer (Apple Silicon friendly, TRL-first with safe fallback)
ChatML-ready + ablation-friendly metrics and rich timing/logging.

- Detects dataset format:
    * ChatML lines: {"messages":[{"role":"system|user|assistant","content":...}, ...]}
    * Alpaca triples: {"instruction","input","output"} (auto-wrapped into ChatML on-the-fly)
- Uses tokenizer.apply_chat_template(...) for Qwen-style formatting.
- Adds elapsed seconds (setup/train/total), steps/sec, approx tokens/sec.
- Logs trainable vs total params (LoRA size & %).
- Saves per-step logs to CSV/JSON (loss trend).
- Optional controls via YAML:
    ablate_examples: 0      # 0 = off, else keep first N shuffled examples
    ablate_seed: 42
    print_every_steps: 10
    system_prompt: "..."    # used if a ChatML example lacks system
    max_minutes: 0          # hard time limit, 0 = off

    # NEW (Train-loss early stop)
    early_stop_patience: 0        # 0 disables (counts logging events)
    early_stop_min_delta: 0.003
    early_stop_window: 5
    early_stop_min_steps: 100

    # NEW (Eval-loss early stop)
    eval_early_stop: false
    eval_steps: 100               # evaluate every N steps
    eval_patience: 5              # ES patience across eval events

    # QoL
    max_grad_norm: 1.0            # grad clipping; 0 disables
"""

# ---------- constants ----------
PROMPT_TPL_ALPACA = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
SYS_DEFAULT = "You are a compassionate, practical spiritual coach. Be concise, kind, and useful."


# ---------- helpers ----------
def _load_yaml(p: Path) -> Dict:
    import yaml

    with p.open("r") as f:
        return yaml.safe_load(f)


def _ensure_tok(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only stability
    return tok


def _sft_maxlen_kw() -> str:
    """Return the correct kw name for sequence length in SFTConfig (api varies by TRL version)."""
    try:
        params = inspect.signature(SFTConfig).parameters  # type: ignore[name-defined]
        if "max_seq_length" in params:
            return "max_seq_length"
        if "max_length" in params:
            return "max_length"
    except Exception:
        pass
    return "max_seq_length"  # sane default


def _count_params(model) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(1, total)
    return {"total": int(total), "trainable": int(trainable), "trainable_pct": round(pct, 4)}


def _detect_lora_targets(model) -> List[str]:
    """Auto-pick LoRA targets for Qwen/LLaMA-style blocks."""
    names = [n for n, _ in model.named_modules()]

    def has(s: str) -> bool:
        return any(s in n for n in names)

    if has("q_proj") and has("v_proj"):
        t = ["q_proj", "k_proj", "v_proj"]
        t += [x for x in ("o_proj", "out_proj") if has(x)]
        t += [x for x in ("gate_proj", "up_proj", "down_proj") if has(x)]
        return t
    if has("c_attn"):
        return ["c_attn", "c_proj", "c_fc"]
    if has("query_key_value"):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    return []


def _maybe_subset(texts: List[str], n: int, seed: int) -> List[str]:
    if n <= 0 or n >= len(texts):
        return texts
    import random

    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [texts[i] for i in idx]


# ---------- callbacks ----------
class LossLoggerCallback(TrainerCallback):
    """Capture on_log events into memory (for CSV/JSON export & loss trend)."""

    def __init__(self, t0: float):
        self.rows: List[Dict[str, Any]] = []
        self._t0 = t0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": int(state.global_step), "epoch": getattr(state, "epoch", None)}
        for k, v in (logs or {}).items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
        row["wall_seconds"] = time.perf_counter() - self._t0
        self.rows.append(row)


class LivePrinterCallback(TrainerCallback):
    """Pretty progress prints (loss, lr, ETA, tokens/s)."""

    def __init__(self, total_steps_est: int, t0: float):
        self.total = total_steps_est
        self.t0 = t0

    @staticmethod
    def _fmt(x: Optional[float], fmt=".4f"):
        if x is None:
            return "-"
        try:
            return format(x, fmt)
        except Exception:
            return "-"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = int(state.global_step)
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        gn = logs.get("grad_norm")
        ent = logs.get("entropy")
        mta = logs.get("mean_token_accuracy")
        tokens = logs.get("num_tokens")
        elapsed = time.perf_counter() - self.t0
        eta = "-"
        if step > 0 and elapsed > 0 and self.total:
            rate = step / elapsed
            rem = max(0, self.total - step)
            eta = f"{(rem / rate) / 60:.1f}m"
        tps = "-" if (tokens is None or elapsed <= 0) else f"{float(tokens)/elapsed:,.1f}"
        print(
            f"[{step}/{self.total}] "
            f"loss={self._fmt(loss)} lr={self._fmt(lr,'.2e')} "
            f"gn={self._fmt(gn,'.2f')} ent={self._fmt(ent,'.3f')} acc={self._fmt(mta,'.3f')} "
            f"tokens/s={tps} elapsed={elapsed:.1f}s eta={eta}",
            flush=True,
        )


class TimeLimitCallback(TrainerCallback):
    """Hard time limit in minutes (0 = disabled)."""

    def __init__(self, minutes: int, t0: float):
        self.deadline = t0 + max(0, minutes) * 60
        self.enabled = minutes > 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.enabled and time.perf_counter() >= self.deadline:
            print("[TimeLimit] Time budget reached; stopping now.")
            control.should_training_stop = True
            return control


class PlateauStopCallback(TrainerCallback):
    """
    Early stop on training-loss plateau (uses on_log).
    Patience & window measured in logging events (not raw steps).
    """

    def __init__(
        self,
        patience_logs: int,
        min_delta: float,
        window: int,
        min_steps: int = 0,
        verbose: bool = True,
    ):
        self.patience = max(1, patience_logs)
        self.min_delta = float(min_delta)
        self.window = max(1, window)
        self.min_steps = int(min_steps)
        self.verbose = verbose
        self.history = deque(maxlen=self.window)
        self.best = float("inf")
        self.stalled = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        self.history.append(loss)
        if len(self.history) < self.window:
            return  # need enough points for smoothing
        smoothed = sum(self.history) / len(self.history)
        improved = (self.best - smoothed) >= self.min_delta
        if improved:
            self.best = smoothed
            self.stalled = 0
        else:
            self.stalled += 1
            if state.global_step >= self.min_steps and self.stalled >= self.patience:
                if self.verbose:
                    print(
                        f"[EarlyStop] Plateau: best={self.best:.4f}, now={smoothed:.4f}, "
                        f"Δ<{self.min_delta} for {self.patience} logs @ step {state.global_step}"
                    )
                control.should_training_stop = True
                return control


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/slm/qwen05b_lora.yaml")
    args = ap.parse_args()

    t0 = time.perf_counter()  # timing start
    cfg = _load_yaml(Path(args.config))
    seed_everything(42)

    # Apple Silicon stability
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dev = pick_device(None)  # cuda → mps → cpu

    base = cfg["base_model"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    max_len = int(cfg["max_seq_len"])
    print_every = int(cfg.get("print_every_steps", 10))
    sys_prompt = cfg.get("system_prompt", SYS_DEFAULT)
    max_minutes = int(cfg.get("max_minutes", 0))

    # Early stop knobs (train-loss plateau)
    es_patience = int(cfg.get("early_stop_patience", 0))  # 0 disables
    es_min_delta = float(cfg.get("early_stop_min_delta", 0.0))
    es_window = int(cfg.get("early_stop_window", 1))
    es_min_steps = int(cfg.get("early_stop_min_steps", 0))

    # Eval-based early stop knobs (optional)
    eval_early_stop = bool(cfg.get("eval_early_stop", False))
    eval_steps = int(cfg.get("eval_steps", max(1, print_every * 5)))
    eval_patience = int(cfg.get("eval_patience", 5))

    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    # tokenizer early (needed for chat template)
    tok = _ensure_tok(AutoTokenizer.from_pretrained(base, use_fast=True))

    # ----- Load dataset & format with ChatML (or Alpaca→ChatML) -----
    path = Path(cfg["dataset_path"])
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def as_chatml_texts(items: List[Dict[str, Any]]) -> List[str]:
        texts_local: List[str] = []
        if items and "messages" in items[0]:
            for r in items:
                messages = r["messages"]
                if not any(m.get("role") == "system" for m in messages):
                    messages = [{"role": "system", "content": sys_prompt}] + messages
                s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts_local.append(s)
        else:
            for r in items:
                instr = (r.get("instruction") or "").strip()
                ctx = (r.get("input") or "").strip()
                ans = (r.get("output") or "").strip()
                user = instr if not ctx else f"{instr}\n\n{ctx}"
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": ans},
                ]
                s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts_local.append(s)
        return texts_local

    # Train texts
    texts = as_chatml_texts(rows)

    # Optional ablation subset
    ablate_n = int(cfg.get("ablate_examples", 0))
    ablate_seed = int(cfg.get("ablate_seed", 42))
    texts = _maybe_subset(texts, ablate_n, ablate_seed)

    train_ds = HFDataset.from_dict({"text": texts}).with_format("python")

    # Optional eval dataset (only if eval_early_stop enabled and dev_path present)
    eval_ds = None
    if eval_early_stop and cfg.get("dev_path"):
        dev_path = Path(cfg["dev_path"])
        if dev_path.exists():
            dev_rows = [json.loads(x) for x in dev_path.read_text().splitlines() if x.strip()]
            eval_texts = as_chatml_texts(dev_rows)
            eval_ds = HFDataset.from_dict({"text": eval_texts}).with_format("python")

    # ----- Approx token count (for throughput report) -----
    # enc = tok(texts, truncation=True, max_length=max_len, add_special_tokens=True)
    # approx_tokens = int(sum(len(ids) for ids in enc["input_ids"]))

    # ----- Model -----
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model.to(dev)

    # ----- LoRA -----
    is_lora = True and _HAS_PEFT
    targets_cfg = cfg.get("target_modules", "auto")
    targets = (
        _detect_lora_targets(model)
        if targets_cfg == "auto"
        else [t.strip() for t in str(targets_cfg).split(",") if t.strip()]
    )
    if not targets:
        print("[qwen05b_lora_train] Could not detect LoRA targets; training full-finetune instead.")
        is_lora = False
    if is_lora:
        lconf = LoraConfig(
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)

    # Param report
    pcounts = _count_params(model)
    print(
        f"[params] total={pcounts['total']:,}  trainable={pcounts['trainable']:,}  "
        f"({pcounts['trainable_pct']}%)  targets={targets if is_lora else 'FULL'}"
    )

    # ----- Trainer (TRL first) -----
    max_steps = int(cfg["max_steps"])
    lr = float(cfg["learning_rate"])
    bs = int(cfg["batch_size"])
    gacc = int(cfg["grad_accum"])

    loss_cb = LossLoggerCallback(t0)
    live_cb = LivePrinterCallback(total_steps_est=max_steps, t0=t0)
    time_cb = TimeLimitCallback(max_minutes, t0)

    plateau_cb = None
    if int(cfg.get("early_stop_patience", 0)) > 0:
        plateau_cb = PlateauStopCallback(
            patience_logs=es_patience,
            min_delta=es_min_delta,
            window=es_window,
            min_steps=es_min_steps,
            verbose=True,
        )

    if _HAS_TRL:
        print("[qwen05b_lora_train] Using TRL SFTTrainer (ChatML)")
        maxlen_kw = _sft_maxlen_kw()
        sft_kwargs = {
            "output_dir": str(out_dir),
            "max_steps": max_steps,
            "learning_rate": lr,
            "per_device_train_batch_size": bs,
            "gradient_accumulation_steps": gacc,
            "packing": False,
            "logging_steps": print_every,
            "save_steps": max_steps,
            "report_to": ["wandb"] if (cfg.get("log_wandb", False)) else [],
            "max_grad_norm": max_grad_norm,
        }
        sft_kwargs[maxlen_kw] = max_len

        # eval strategy (optional)
        if eval_ds is not None:
            sft_kwargs.update(
                dict(
                    evaluation_strategy="steps",
                    eval_steps=eval_steps,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    save_total_limit=2,
                )
            )

        # Try to quiet MPS pin-memory warning if supported
        if "dataloader_pin_memory" in inspect.signature(SFTConfig).parameters:
            sft_kwargs["dataloader_pin_memory"] = False

        sft_args = SFTConfig(**sft_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )
        trainer.add_callback(loss_cb)
        trainer.add_callback(live_cb)
        trainer.add_callback(time_cb)
        if plateau_cb is not None:
            trainer.add_callback(plateau_cb)
        if eval_ds is not None and bool(cfg.get("eval_early_stop", False)):
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=eval_patience))
    else:
        print("[qwen05b_lora_train] Using HF Trainer fallback (ChatML)")
        enc2 = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors=None)
        train_dataset = [
            {"input_ids": ids, "attention_mask": am}
            for ids, am in zip(enc2["input_ids"], enc2["attention_mask"])
        ]

        def collate(batch):
            maxlen_b = max(len(b["input_ids"]) for b in batch)
            ids, attn = [], []
            for b in batch:
                pad = maxlen_b - len(b["input_ids"])
                ids.append(b["input_ids"] + [tok.eos_token_id] * pad)
                attn.append(b["attention_mask"] + [1] * pad)
            import torch as T

            return {
                "input_ids": T.tensor(ids, dtype=T.long),
                "attention_mask": T.tensor(attn, dtype=T.long),
                "labels": T.tensor(ids, dtype=T.long),
            }

        targs = dict(
            output_dir=str(out_dir),
            max_steps=max_steps,
            learning_rate=lr,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=gacc,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            logging_steps=print_every,
            save_steps=max_steps,
            report_to=["wandb"] if (cfg.get("log_wandb", False)) else [],
            max_grad_norm=max_grad_norm,
        )

        if eval_ds is not None:
            targs.update(
                dict(
                    evaluation_strategy="steps",
                    eval_steps=eval_steps,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    save_total_limit=2,
                )
            )

        trainer = Trainer(
            model=model,
            args=TrainingArguments(**targs),
            train_dataset=train_dataset,
            eval_dataset=eval_ds,
            data_collator=collate,
            tokenizer=tok,
        )
        trainer.add_callback(loss_cb)
        trainer.add_callback(live_cb)
        trainer.add_callback(time_cb)
        if plateau_cb is not None:
            trainer.add_callback(plateau_cb)
        if eval_ds is not None and bool(cfg.get("eval_early_stop", False)):
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=eval_patience))

    # ----- Train -----
    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    train_result = trainer.train()
    t_train = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # ----- Metrics summary -----
    metrics = getattr(train_result, "metrics", {}) or {}
    steps = int(metrics.get("train_steps", metrics.get("global_step", cfg.get("max_steps", 0))))
    steps_per_sec = (steps / t_train) if t_train > 0 else 0.0

    # Prefer TRL cumulative token metric if present; else approx dataset tokens
    # (We didn’t store cumulative tokens explicitly; keep approx throughput)
    tokens_per_sec = (float(len(texts) * max_len) / t_train) if t_train > 0 else 0.0

    first_loss = next((r["loss"] for r in loss_cb.rows if "loss" in r), None)
    last_loss = next((r["loss"] for r in reversed(loss_cb.rows) if "loss" in r), None)
    loss_delta = (
        (last_loss - first_loss) if (first_loss is not None and last_loss is not None) else None
    )
    loss_pct = (
        (100.0 * loss_delta / first_loss) if (loss_delta is not None and first_loss) else None
    )

    report = {
        "examples": len(texts),
        "approx_dataset_tokens": int(
            sum(len(ids) for ids in tok(texts, truncation=True, max_length=max_len)["input_ids"])
        ),
        "steps": steps,
        "setup_seconds": round(t_setup, 3),
        "train_seconds": round(t_train, 3),
        "total_seconds": round(t_total, 3),
        "steps_per_second": round(steps_per_sec, 3),
        "approx_tokens_per_second": round(tokens_per_sec, 2),
        "trainable_params": pcounts["trainable"],
        "total_params": pcounts["total"],
        "trainable_pct": pcounts["trainable_pct"],
        "first_logged_loss": round(first_loss, 6) if first_loss is not None else None,
        "last_logged_loss": round(last_loss, 6) if last_loss is not None else None,
        "loss_delta": round(loss_delta, 6) if loss_delta is not None else None,
        "loss_delta_pct": round(loss_pct, 3) if loss_pct is not None else None,
        "ablate_examples": int(cfg.get("ablate_examples", 0)),
        "ablate_seed": int(cfg.get("ablate_seed", 42)),
        "early_stop_plateau": bool(es_patience > 0),
        "eval_early_stop": bool(eval_early_stop and eval_ds is not None),
    }
    print("⏱ timing/ablation report:\n", json.dumps(report, indent=2))

    # Save reports
    (out_dir / "time_metrics.json").write_text(json.dumps(report, indent=2))
    if loss_cb.rows:
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(loss_cb.rows, f, indent=2)
        with open(out_dir / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in loss_cb.rows for k in r.keys()}))
            w.writeheader()
            w.writerows(loss_cb.rows)

    # Optional quick eval (post-train qualitative check)
    if cfg.get("eval_after_train", True):
        base_for_merge = base if _HAS_PEFT else None
        evaluate_and_print(
            model_path=str(out_dir),
            base_model=base_for_merge,
            dev_path=str(cfg.get("dev_path", "data/sft/dev_toy.jsonl")),
            limit=int(cfg.get("eval_limit", 20)),
            batch_size=2,
            max_new_tokens=int(cfg.get("max_new_tokens", 128)),
            log_wandb=bool(cfg.get("log_wandb", False)),
            wandb_project=cfg.get("wandb_project", "airepo-sft"),
        )


if __name__ == "__main__":
    main()
