#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from airoad.core import pick_device, seed_everything
from airoad.sft.callbacks import (
    LivePrinterCallback,
    LossLoggerCallback,
    PlateauStopCallback,
    TimeLimitCallback,
)
from airoad.sft.data import ensure_tok, prepare_datasets
from airoad.sft.lora import apply_lora
from airoad.sft.trainers import SFTConfig, build_hf_trainer, build_trl_trainer  # type: ignore

try:
    from trl import SFTConfig as TRL_SFTConfig  # type: ignore

    _HAS_TRL = True
except Exception:
    TRL_SFTConfig = object  # type: ignore
    _HAS_TRL = False


"""
Slim SFT/LoRA training CLI for Qwen-0.5B.

Uses modular helpers:
- airoad.sft.data       (loading, formatting, datasets, token counts)
- airoad.sft.callbacks  (logging, live printer, time limit, plateau early-stop)
- airoad.sft.trainers   (plain or KL-regularized TRL/HF trainers)
- airoad.sft.lora       (targets + LoRA/DoRA application)

Behavior is driven entirely by YAML config (see qwen05b_lora_fast.yaml).
"""


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/slm/qwen05b_lora.yaml")
    args = ap.parse_args()

    # --- Setup
    t0 = time.perf_counter()
    cfg = _load_yaml(args.config)
    seed_everything(42)

    # Apple Silicon env knobs
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dev = pick_device(None)  # cuda → mps → cpu

    base = cfg["base_model"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training knobs
    max_len = int(cfg["max_seq_len"])
    print_every = int(cfg.get("print_every_steps", 10))
    lr = float(cfg["learning_rate"])
    bs = int(cfg["batch_size"])
    gacc = int(cfg["grad_accum"])
    max_steps = int(cfg["max_steps"])
    log_wandb = bool(cfg.get("log_wandb", False))
    max_minutes = int(cfg.get("max_minutes", 0))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    system_prompt = cfg.get(
        "system_prompt",
        "You are a compassionate, practical spiritual coach. Be concise, kind, and useful.",
    )

    # Early stop (train-loss plateau)
    es_patience = int(cfg.get("early_stop_patience", 0))
    es_min_delta = float(cfg.get("early_stop_min_delta", 0.0))
    es_window = int(cfg.get("early_stop_window", 1))
    es_min_steps = int(cfg.get("early_stop_min_steps", 0))

    # Eval-loss early stop (optional)
    eval_early_stop = bool(cfg.get("eval_early_stop", False))
    eval_steps = int(cfg.get("eval_steps", max(1, print_every * 5)))
    eval_patience = int(cfg.get("eval_patience", 5))
    dev_path = cfg.get("dev_path")

    # LoRA knobs
    lora_r = int(cfg["lora_r"])
    lora_alpha = int(cfg["lora_alpha"])
    lora_dropout = float(cfg["lora_dropout"])
    lora_target_mode = str(cfg.get("lora_target_mode", "auto"))
    use_dora = bool(cfg.get("use_dora", False))

    # KL knobs (0 disables)
    kl_lambda = float(cfg.get("kl_lambda", 0.0))
    kl_tau = float(cfg.get("kl_tau", 1.0))

    # --- Tokenizer
    tok = ensure_tok(AutoTokenizer.from_pretrained(base, use_fast=True))

    # --- Datasets
    train_ds, eval_ds, approx_tokens = prepare_datasets(
        dataset_path=cfg["dataset_path"],
        tok=tok,
        system_prompt=system_prompt,
        max_len=max_len,
        ablate_examples=int(cfg.get("ablate_examples", 0)),
        ablate_seed=int(cfg.get("ablate_seed", 42)),
        dev_path=dev_path if eval_early_stop else None,
    )

    # --- Models
    # student
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.to(dev)

    # reference (teacher) for KL
    ref_model = None
    if kl_lambda > 0.0:
        try:
            ref_model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(dev)
        except TypeError:
            ref_model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(dev)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # --- Apply LoRA/DoRA
    model, targets = apply_lora(
        model,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_mode=lora_target_mode,
        use_dora=use_dora,
    )
    is_lora = bool(targets)
    if not is_lora:
        print("[warn] LoRA targets not detected; continuing without PEFT.")

    # --- Callbacks
    loss_cb = LossLoggerCallback(t0)
    live_cb = LivePrinterCallback(total_steps_est=max_steps, t0=t0)
    time_cb = TimeLimitCallback(max_minutes, t0)
    plateau_cb = (
        PlateauStopCallback(
            patience_logs=es_patience,
            min_delta=es_min_delta,
            window=es_window,
            min_steps=es_min_steps,
            verbose=True,
        )
        if es_patience > 0
        else None
    )

    # --- Build trainer
    if _HAS_TRL:
        # TRL SFTConfig
        sft_kwargs: Dict[str, Any] = {
            "output_dir": str(out_dir),
            "max_steps": max_steps,
            "learning_rate": lr,
            "per_device_train_batch_size": bs,
            "gradient_accumulation_steps": gacc,
            "packing": False,
            "logging_steps": print_every,
            "save_steps": max_steps,
            "report_to": ["wandb"] if log_wandb else [],
            "max_grad_norm": max_grad_norm,
        }
        # optional eval in TRL
        if eval_ds is not None and eval_early_stop:
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
        # quiet pin_memory if arg supported
        try:
            if "dataloader_pin_memory" in inspect.signature(TRL_SFTConfig).parameters:  # type: ignore[attr-defined]
                sft_kwargs["dataloader_pin_memory"] = False
        except Exception:
            pass

        sft_args: SFTConfig = TRL_SFTConfig(**sft_kwargs)  # type: ignore[call-arg]
        trainer = build_trl_trainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds if eval_early_stop else None,
            sft_args=sft_args,
            ref_model=ref_model,
            kl_lambda=kl_lambda,
            kl_tau=kl_tau,
        )
        trainer.add_callback(loss_cb)
        trainer.add_callback(live_cb)
        trainer.add_callback(time_cb)
        if plateau_cb is not None:
            trainer.add_callback(plateau_cb)
        if eval_ds is not None and eval_early_stop:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=eval_patience))
    else:
        # HF fallback
        # build tokenized batch (collator) for HF path
        enc2 = tok([e["text"] for e in train_ds], truncation=True, max_length=max_len, padding=True, return_tensors=None)  # type: ignore[index]
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

        targs = TrainingArguments(
            output_dir=str(out_dir),
            max_steps=max_steps,
            learning_rate=lr,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=gacc,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            logging_steps=print_every,
            save_steps=max_steps,
            report_to=["wandb"] if log_wandb else [],
            max_grad_norm=max_grad_norm,
            evaluation_strategy=("steps" if (eval_ds is not None and eval_early_stop) else "no"),
            eval_steps=(eval_steps if (eval_ds is not None and eval_early_stop) else None),
            load_best_model_at_end=bool(eval_ds is not None and eval_early_stop),
            metric_for_best_model=(
                "eval_loss" if (eval_ds is not None and eval_early_stop) else None
            ),
            greater_is_better=False,
            save_total_limit=(2 if (eval_ds is not None and eval_early_stop) else None),
        )

        trainer = build_hf_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_ds if eval_early_stop else None,
            training_args=targs,
            data_collator=collate,
            tokenizer=tok,
            ref_model=ref_model,
            kl_lambda=kl_lambda,
            kl_tau=kl_tau,
        )
        trainer.add_callback(loss_cb)
        trainer.add_callback(live_cb)
        trainer.add_callback(time_cb)
        if plateau_cb is not None:
            trainer.add_callback(plateau_cb)
        if eval_ds is not None and eval_early_stop:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=eval_patience))

    # --- Train
    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    train_result = trainer.train()
    t_train = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # --- Report
    metrics = getattr(train_result, "metrics", {}) or {}
    steps = int(metrics.get("train_steps", metrics.get("global_step", max_steps)))
    steps_per_sec = (steps / t_train) if t_train > 0 else 0.0
    tokens_per_sec = (approx_tokens / t_train) if t_train > 0 else 0.0

    first_loss = next((r["loss"] for r in loss_cb.rows if "loss" in r), None)
    last_loss = next((r["loss"] for r in reversed(loss_cb.rows) if "loss" in r), None)
    loss_delta = (
        (last_loss - first_loss) if (first_loss is not None and last_loss is not None) else None
    )
    loss_pct = (
        (100.0 * loss_delta / first_loss) if (loss_delta is not None and first_loss) else None
    )

    report = {
        "examples": len(train_ds),  # type: ignore[arg-type]
        "approx_dataset_tokens": approx_tokens,
        "steps": steps,
        "setup_seconds": round(t_setup, 3),
        "train_seconds": round(t_train, 3),
        "total_seconds": round(t_total, 3),
        "steps_per_second": round(steps_per_sec, 3),
        "approx_tokens_per_second": round(tokens_per_sec, 2),
        "first_logged_loss": round(first_loss, 6) if first_loss is not None else None,
        "last_logged_loss": round(last_loss, 6) if last_loss is not None else None,
        "loss_delta": round(loss_delta, 6) if loss_delta is not None else None,
        "loss_delta_pct": round(loss_pct, 3) if loss_pct is not None else None,
        "ablate_examples": int(cfg.get("ablate_examples", 0)),
        "ablate_seed": int(cfg.get("ablate_seed", 42)),
        "early_stop_plateau": bool(es_patience > 0),
        "eval_early_stop": bool(eval_early_stop and eval_ds is not None),
        "use_dora": use_dora,
        "kl_lambda": kl_lambda,
        "kl_tau": kl_tau,
        "lora_target_mode": lora_target_mode,
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

    # Optional post-train eval (your existing evaluator)
    if cfg.get("eval_after_train", False):
        from airoad.sft.eval_sft import evaluate_and_print  # type: ignore

        evaluate_and_print(
            model_path=str(out_dir),
            base_model=base if is_lora else None,
            dev_path=str(cfg.get("dev_path", "data/sft/dev_toy.jsonl")),
            limit=int(cfg.get("eval_limit", 20)),
            batch_size=2,
            max_new_tokens=int(cfg.get("max_new_tokens", 128)),
            log_wandb=bool(cfg.get("log_wandb", False)),
            wandb_project=cfg.get("wandb_project", "airepo-sft"),
        )


if __name__ == "__main__":
    main()
