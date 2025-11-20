#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch

from airoad.core import pick_device, seed_everything
from airoad.sft.callbacks import (
    LivePrinterCallback,
    LossLoggerCallback,
    PlateauStopCallback,
    TimeLimitCallback,
)
from airoad.sft.data import ensure_tok
from airoad.sft.lora import apply_lora
from airoad.sft.pref_data import build_dpo_dataset
from airoad.sft.pref_trainers import build_orpo_trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

"""
Slim ORPO training CLI for Qwen-0.5B (LoRA/DoRA compatible).
- Uses pairs dataset (prompt, chosen, rejected), like DPO.
"""


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/slm/qwen05b_orpo_fast_plain.yaml")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = _load_yaml(args.config)
    seed_everything(42)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dev = pick_device(None)

    base = cfg["base_model"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    pairs_path = cfg["pairs_path"]
    max_steps = int(cfg.get("max_steps", 200))
    lr = float(cfg.get("learning_rate", 5e-5))
    bs = int(cfg.get("batch_size", 1))
    gacc = int(cfg.get("grad_accum", 8))
    print_every = int(cfg.get("print_every_steps", 10))
    log_wandb = bool(cfg.get("log_wandb", False))
    max_minutes = int(cfg.get("max_minutes", 0))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    orpo_alpha = float(cfg.get("orpo_alpha", 0.5))  # default OK for small runs

    eval_early_stop = bool(cfg.get("eval_early_stop", False))
    eval_steps = int(cfg.get("eval_steps", max(1, print_every * 5)))
    eval_patience = int(cfg.get("eval_patience", 5))
    dev_pairs_path = cfg.get("dev_pairs_path")

    lora_r = int(cfg.get("lora_r", 16))
    lora_alpha = int(cfg.get("lora_alpha", 32))
    lora_dropout = float(cfg.get("lora_dropout", 0.05))
    lora_target_mode = str(cfg.get("lora_target_mode", "attn_mlp"))
    use_dora = bool(cfg.get("use_dora", False))

    tok = ensure_tok(AutoTokenizer.from_pretrained(base, use_fast=True))
    train_ds = build_dpo_dataset(
        pairs_path=pairs_path,
        tok=tok,
        system_prompt=system_prompt,
        ablate_examples=int(cfg.get("ablate_examples", 0)),
        ablate_seed=int(cfg.get("ablate_seed", 42)),
    )
    eval_ds = None
    if eval_early_stop and dev_pairs_path:
        eval_ds = build_dpo_dataset(
            pairs_path=dev_pairs_path,
            tok=tok,
            system_prompt=system_prompt,
            ablate_examples=0,
            ablate_seed=42,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.to(dev)

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

    loss_cb = LossLoggerCallback(t0)
    live_cb = LivePrinterCallback(total_steps_est=max_steps, t0=t0)
    time_cb = TimeLimitCallback(max_minutes, t0)
    plateau_cb = (
        PlateauStopCallback(
            patience_logs=int(cfg.get("early_stop_patience", 0)),
            min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
            window=int(cfg.get("early_stop_window", 1)),
            min_steps=int(cfg.get("early_stop_min_steps", 0)),
            verbose=True,
        )
        if int(cfg.get("early_stop_patience", 0)) > 0
        else None
    )

    trainer = build_orpo_trainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        alpha=orpo_alpha,
        max_steps=max_steps,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=gacc,
        logging_steps=print_every,
        output_dir=str(out_dir),
        save_steps=max_steps,
        report_to=(["wandb"] if log_wandb else []),
        max_grad_norm=max_grad_norm,
        dataloader_pin_memory=False,
        evaluation_strategy=("steps" if (eval_ds is not None and eval_early_stop) else "no"),
        eval_dataset=(eval_ds if (eval_ds is not None and eval_early_stop) else None),
        eval_steps=(eval_steps if (eval_ds is not None and eval_early_stop) else None),
        load_best_model_at_end=bool(eval_ds is not None and eval_early_stop),
        metric_for_best_model=("eval_loss" if (eval_ds is not None and eval_early_stop) else None),
        greater_is_better=False,
        save_total_limit=(2 if (eval_ds is not None and eval_early_stop) else None),
    )

    trainer.add_callback(loss_cb)
    trainer.add_callback(live_cb)
    trainer.add_callback(time_cb)
    if plateau_cb is not None:
        trainer.add_callback(plateau_cb)
    if eval_ds is not None and eval_early_stop:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=eval_patience))

    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    train_result = trainer.train()
    t_train = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    metrics = getattr(train_result, "metrics", {}) or {}
    steps = int(metrics.get("train_steps", metrics.get("global_step", max_steps)))
    steps_per_sec = (steps / t_train) if t_train > 0 else 0.0

    report = {
        "pairs": len(train_ds),
        "steps": steps,
        "setup_seconds": round(t_setup, 3),
        "train_seconds": round(t_train, 3),
        "total_seconds": round(t_total, 3),
        "steps_per_second": round(steps_per_sec, 3),
        "use_dora": use_dora,
        "orpo_alpha": orpo_alpha,
        "lora_target_mode": lora_target_mode,
    }
    print("⏱ ORPO timing report:\n", json.dumps(report, indent=2))
    (out_dir / "time_metrics.json").write_text(json.dumps(report, indent=2))
    if loss_cb.rows:
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(loss_cb.rows, f, indent=2)


if __name__ == "__main__":
    main()
