#!/usr/bin/env python
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from transformers import TrainingArguments

# TRL availability
try:
    from trl import DPOTrainer  # type: ignore

    _HAS_TRL_DPO = True
except Exception:
    _HAS_TRL_DPO = False

# Optional DPOConfig (newer TRL)
try:
    from trl import DPOConfig  # type: ignore

    _HAS_TRL_DPOCONFIG = True
except Exception:
    _HAS_TRL_DPOCONFIG = False

"""
src/airoad/sft/pref_trainers.py

Preference optimization trainers (DPO):
- build_dpo_trainer: TRL DPOTrainer with LoRA/DoRA-ready student and a frozen reference model.

Compatibility layer:
- Different TRL/Transformers versions expose slightly different signatures.
  We introspect `TrainingArguments.__init__`, `DPOConfig.__init__` (if present),
  and `DPOTrainer.__init__` and only pass kwargs that are supported.
"""

# ---------- small helpers ----------


def _sig_has_param(callable_obj, name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _filter_kwargs_for_signature(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = set(inspect.signature(fn).parameters.keys())
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in kwargs.items() if k in params}


# ---------- main builder ----------


def build_dpo_trainer(
    model,
    ref_model,
    tokenizer,
    train_dataset,
    *,
    beta: float,
    max_steps: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    output_dir: str,
    save_steps: int,
    report_to: Optional[list] = None,
    max_grad_norm: float = 1.0,
    dataloader_pin_memory: bool = False,
    evaluation_strategy: str = "no",
    eval_dataset=None,
    eval_steps: Optional[int] = None,
    load_best_model_at_end: bool = False,
    metric_for_best_model: Optional[str] = None,
    greater_is_better: bool = False,
    save_total_limit: Optional[int] = None,
):
    if not _HAS_TRL_DPO:
        raise RuntimeError(
            "TRL DPOTrainer is not available. Please install/update `trl` to a version that provides DPOTrainer."
        )

    # ---- Build arguments object (prefer TRL DPOConfig if available) ----
    if _HAS_TRL_DPOCONFIG:
        # Build kwargs for DPOConfig; start from a superset and filter
        cfg_kwargs: Dict[str, Any] = {
            "output_dir": output_dir,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "report_to": report_to or [],
            "max_grad_norm": max_grad_norm,
            "dataloader_pin_memory": dataloader_pin_memory,
            "evaluation_strategy": evaluation_strategy,
            "eval_steps": eval_steps,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
            "save_total_limit": save_total_limit,
            # common TRL fields that some versions require:
            "remove_unused_columns": False,
            "model_init_kwargs": {},  # ensure attribute exists
            "ref_model_init_kwargs": {},  # ensure attribute exists
            # Some versions have beta in config; we’ll filter below.
            "beta": beta,
            "dpo_beta": beta,
        }
        cfg_kwargs = _filter_kwargs_for_signature(DPOConfig.__init__, cfg_kwargs)
        args_obj = DPOConfig(**cfg_kwargs)  # type: ignore

    else:
        # Fallback: vanilla TrainingArguments (filter by signature)
        ta_kwargs: Dict[str, Any] = {
            "output_dir": output_dir,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "report_to": report_to or [],
            "max_grad_norm": max_grad_norm,
            "dataloader_pin_memory": dataloader_pin_memory,
            "evaluation_strategy": evaluation_strategy,
            "eval_steps": eval_steps,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
            "save_total_limit": save_total_limit,
            "remove_unused_columns": False,
        }
        ta_kwargs = _filter_kwargs_for_signature(TrainingArguments.__init__, ta_kwargs)
        args_obj = TrainingArguments(**ta_kwargs)
        # Monkey-patch TRL-specific fields expected by some builds
        # (TrainingArguments is a dataclass; adding attrs at runtime is fine)
        if not hasattr(args_obj, "model_init_kwargs"):
            setattr(args_obj, "model_init_kwargs", {})
        if not hasattr(args_obj, "ref_model_init_kwargs"):
            setattr(args_obj, "ref_model_init_kwargs", {})
        # Some builds expect args.beta/dpo_beta; we won't set them here.
        # We will pass beta/dpo_beta directly to DPOTrainer if supported.

    # ---- Build DPOTrainer kwargs (filtered by signature) ----
    dpo_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": args_obj,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
    }

    # beta param name varies (beta vs dpo_beta); pass only if supported
    if _sig_has_param(DPOTrainer.__init__, "beta"):
        dpo_kwargs["beta"] = beta
    elif _sig_has_param(DPOTrainer.__init__, "dpo_beta"):
        dpo_kwargs["dpo_beta"] = beta
    # else: rely on library default

    # Some TRL versions accept greater_is_better on DPOTrainer; pass if supported
    if _sig_has_param(DPOTrainer.__init__, "greater_is_better"):
        dpo_kwargs["greater_is_better"] = greater_is_better

    # Instantiate trainer with signature-filtered kwargs
    trainer = DPOTrainer(**_filter_kwargs_for_signature(DPOTrainer.__init__, dpo_kwargs))
    return trainer
