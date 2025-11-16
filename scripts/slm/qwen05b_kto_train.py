#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

from airoad.core import pick_device, seed_everything
from airoad.sft.callbacks import (
    LivePrinterCallback,
    LossLoggerCallback,
    PlateauStopCallback,
    TimeLimitCallback,
)
from airoad.sft.data import ensure_tok
from airoad.sft.lora import apply_lora
from airoad.sft.pref_data import build_kto_dataset
from airoad.sft.pref_trainers import build_kto_trainer

"""
Slim KTO training CLI for Qwen-0.5B (LoRA/DoRA compatible).
- Uses chosen-only dataset + synthetic negatives: (prompt, completion, label)
- Extra safeguards to avoid no-grad crashes:
  * Force GC off on the model
  * Force embedding outputs to require grad via a forward hook
"""


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _force_no_gradient_checkpointing(model) -> None:
    """
    Disable gradient checkpointing at the model level as hard as possible.
    KTO can re-enable it internally; this avoids most sources of no-grad graphs.
    """
    # 1) model API (if present)
    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    except Exception:
        pass
    # 2) common config flags transformers/trl check
    try:
        if hasattr(model, "config"):
            setattr(model.config, "gradient_checkpointing", False)
            setattr(model.config, "use_cache", False)
    except Exception:
        pass


def _ensure_inputs_require_grad(model) -> None:
    """
    Make sure the embedding outputs require grad.
    This creates a valid grad path even if GC sneaks in.
    """
    try:
        # Newer models have a convenience method
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            return
    except Exception:
        pass

    # Generic: forward hook on the input embedding to set requires_grad on its outputs
    try:
        emb = model.get_input_embeddings()
        if emb is None:
            return

        def _mark_out_requires_grad(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
            elif isinstance(output, (list, tuple)):
                for t in output:
                    if isinstance(t, torch.Tensor):
                        t.requires_grad_(True)
            return output

        emb.register_forward_hook(_mark_out_requires_grad)

        # Also ensure embedding weights are trainable = not necessary for LoRA, but harmless
        try:
            emb.weight.requires_grad_(True)
        except Exception:
            pass
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/slm/qwen05b_kto_fast_plain.yaml")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = _load_yaml(args.config)
    seed_everything(42)
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dev = pick_device(None)  # "cuda" | "mps" | "cpu"

    base = cfg["base_model"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    dataset_path = cfg["dataset_path"]

    # ---- train knobs
    max_steps = int(cfg.get("max_steps", 200))
    lr = float(cfg.get("learning_rate", 5e-5))
    bs = int(cfg.get("batch_size", 2))  # KTO requires per-device batch > 1
    if bs < 2:
        print(
            f"[kto] batch_size={bs} < 2 → setting per_device_train_batch_size=2 (KTO requires >1)"
        )
        bs = 2
    gacc = int(cfg.get("grad_accum", 8))
    print_every = int(cfg.get("print_every_steps", 10))
    log_wandb = bool(cfg.get("log_wandb", False))
    max_minutes = int(cfg.get("max_minutes", 0))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    # ---- KTO hyperparams
    kto_beta = float(cfg.get("kto_beta", 0.1))
    # Balanced weights (since we synthesize negatives 1:1)
    desirable_weight = cfg.get("desirable_weight", 1.0)
    undesirable_weight = cfg.get("undesirable_weight", 1.0)

    # ---- negatives (to avoid degenerate gradients)
    kto_neg_mode = str(cfg.get("kto_neg_mode", "prefix"))  # "prefix" | "truncate"
    kto_neg_ratio = float(cfg.get("kto_neg_ratio", 1.0))  # 1 negative per positive

    # ---- eval (optional)
    eval_early_stop = bool(cfg.get("eval_early_stop", False))
    eval_steps = int(cfg.get("eval_steps", max(1, print_every * 5)))
    eval_patience = int(cfg.get("eval_patience", 5))
    dev_path = cfg.get("dev_path")

    # ---- LoRA/DoRA
    lora_r = int(cfg.get("lora_r", 16))
    lora_alpha = int(cfg.get("lora_alpha", 32))
    lora_dropout = float(cfg.get("lora_dropout", 0.05))
    lora_target_mode = str(cfg.get("lora_target_mode", "attn_mlp"))
    use_dora = bool(cfg.get("use_dora", False))

    # ---- tokenizer & datasets
    tok = ensure_tok(AutoTokenizer.from_pretrained(base, use_fast=True))

    train_ds = build_kto_dataset(
        dataset_path=dataset_path,
        tok=tok,
        system_prompt=system_prompt,
        ablate_examples=int(cfg.get("ablate_examples", 0)),
        ablate_seed=int(cfg.get("ablate_seed", 42)),
        kto_neg_mode=kto_neg_mode,
        kto_neg_ratio=kto_neg_ratio,
    )
    eval_ds = None
    if eval_early_stop and dev_path:
        eval_ds = build_kto_dataset(
            dataset_path=dev_path,
            tok=tok,
            system_prompt=system_prompt,
            ablate_examples=0,
            ablate_seed=42,
            kto_neg_mode=kto_neg_mode,
            kto_neg_ratio=kto_neg_ratio,
        )

    # ---- model
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model.to(dev)

    # Apply LoRA/DoRA
    model, targets = apply_lora(
        model,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_mode=lora_target_mode,
        use_dora=use_dora,
    )

    # Put the model in training mode now (some wrappers default to eval)
    model.train()

    # Hard-disable GC at the model level and ensure a grad path from inputs
    _force_no_gradient_checkpointing(model)
    _ensure_inputs_require_grad(model)

    # Optional: show trainable param %
    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"[kto] trainable={trainable:,} / total={total:,} ({100.0*trainable/max(total,1):.2f}%)"
        )
    except Exception:
        pass

    # ---- callbacks
    loss_cb = LossLoggerCallback(t0)
    live_cb = LivePrinterCallback(total_steps_est=max_steps, t0=t0)
    time_cb = TimeLimitCallback(max_minutes, t0)
    plateau_cb = None
    if int(cfg.get("early_stop_patience", 0)) > 0:
        plateau_cb = PlateauStopCallback(
            patience_logs=int(cfg.get("early_stop_patience", 0)),
            min_delta=float(cfg.get("early_stop_min_delta", 0.0)),
            window=int(cfg.get("early_stop_window", 1)),
            min_steps=int(cfg.get("early_stop_min_steps", 0)),
            verbose=True,
        )

    # ---- trainer (signature-safe via builder)
    trainer = build_kto_trainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        beta=kto_beta,
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
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
    )

    trainer.add_callback(loss_cb)
    trainer.add_callback(live_cb)
    trainer.add_callback(time_cb)
    if plateau_cb is not None:
        trainer.add_callback(plateau_cb)
    if eval_ds is not None and eval_early_stop:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=eval_patience))

    # ---- train
    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    train_result = trainer.train()
    t_train = time.perf_counter() - t1
    t_total = time.perf_counter() - t0

    # ---- save + report
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    metrics = getattr(train_result, "metrics", {}) or {}
    steps = int(metrics.get("train_steps", metrics.get("global_step", max_steps)))
    steps_per_sec = (steps / t_train) if t_train > 0 else 0.0

    report = {
        "examples": len(train_ds),
        "steps": steps,
        "setup_seconds": round(t_setup, 3),
        "train_seconds": round(t_train, 3),
        "total_seconds": round(t_total, 3),
        "steps_per_second": round(steps_per_sec, 3),
        "use_dora": use_dora,
        "kto_beta": kto_beta,
        "lora_target_mode": lora_target_mode,
        "neg_mode": kto_neg_mode,
        "neg_ratio": kto_neg_ratio,
        "per_device_train_batch_size": bs,
        "grad_accum": gacc,
    }
    print("⏱ KTO timing report:\n", json.dumps(report, indent=2))
    (out_dir / "time_metrics.json").write_text(json.dumps(report, indent=2))
    if loss_cb.rows:
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(loss_cb.rows, f, indent=2)


if __name__ == "__main__":
    main()
