#!/usr/bin/env python
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from transformers import TrainingArguments

# ---- TRL availability ---------------------------------------------------
try:
    from trl import DPOTrainer  # type: ignore

    _HAS_TRL_DPO = True
except Exception:
    _HAS_TRL_DPO = False

try:
    from trl import ORPOTrainer  # type: ignore

    _HAS_TRL_ORPO = True
except Exception:
    _HAS_TRL_ORPO = False

try:
    from trl import KTOTrainer  # type: ignore

    _HAS_TRL_KTO = True
except Exception:
    _HAS_TRL_KTO = False

# Optional TRL Configs (newer TRL releases)
try:
    from trl import DPOConfig  # type: ignore

    _HAS_TRL_DPOCONFIG = True
except Exception:
    _HAS_TRL_DPOCONFIG = False

try:
    from trl import ORPOConfig  # type: ignore

    _HAS_TRL_ORPOCONFIG = True
except Exception:
    _HAS_TRL_ORPOCONFIG = False

try:
    from trl import KTOConfig  # type: ignore

    _HAS_TRL_KTOCONFIG = True
except Exception:
    _HAS_TRL_KTOCONFIG = False

"""
src/airoad/sft/pref_trainers.py

Preference optimization trainers (DPO / ORPO / KTO):

- Robust builders that handle TRL/Transformers signature differences:
  * Prefer TRL Config objects (DPOConfig/ORPOConfig/KTOConfig) if available.
  * Otherwise fall back to transformers.TrainingArguments and monkey-patch
    only the attributes that some TRL builds expect.
  * Only pass kwargs that are actually supported (introspection).
"""

# ---- helpers ------------------------------------------------------------


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


def _build_args_obj(
    *,
    prefer_config_cls,
    fallback_ta_cls,
    output_dir: str,
    max_steps: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_steps: int,
    report_to: Optional[list],
    max_grad_norm: float,
    dataloader_pin_memory: bool,
    evaluation_strategy: str,
    eval_steps: Optional[int],
    load_best_model_at_end: bool,
    metric_for_best_model: Optional[str],
    greater_is_better: bool,
    save_total_limit: Optional[int],
    include_model_kwargs: bool,
):
    if prefer_config_cls is not None:
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
            "remove_unused_columns": False,
        }
        if include_model_kwargs:
            if _sig_has_param(prefer_config_cls.__init__, "model_init_kwargs"):
                cfg_kwargs["model_init_kwargs"] = None
            if _sig_has_param(prefer_config_cls.__init__, "ref_model_init_kwargs"):
                cfg_kwargs["ref_model_init_kwargs"] = None
        cfg_kwargs = _filter_kwargs_for_signature(prefer_config_cls.__init__, cfg_kwargs)
        return prefer_config_cls(**cfg_kwargs)

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
    ta_kwargs = _filter_kwargs_for_signature(fallback_ta_cls.__init__, ta_kwargs)
    args_obj = fallback_ta_cls(**ta_kwargs)
    if include_model_kwargs:
        if not hasattr(args_obj, "model_init_kwargs"):
            setattr(args_obj, "model_init_kwargs", None)
        if not hasattr(args_obj, "ref_model_init_kwargs"):
            setattr(args_obj, "ref_model_init_kwargs", None)
    return args_obj


# ---- DPO ----------------------------------------------------------------


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
        raise RuntimeError("TRL DPOTrainer is not available. Please install/update `trl`.")

    args_obj = _build_args_obj(
        prefer_config_cls=DPOConfig if _HAS_TRL_DPOCONFIG else None,  # type: ignore
        fallback_ta_cls=TrainingArguments,
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to=report_to,
        max_grad_norm=max_grad_norm,
        dataloader_pin_memory=dataloader_pin_memory,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        include_model_kwargs=True,
    )

    dpo_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": args_obj,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
    }
    if _sig_has_param(DPOTrainer.__init__, "beta"):
        dpo_kwargs["beta"] = beta
    elif _sig_has_param(DPOTrainer.__init__, "dpo_beta"):
        dpo_kwargs["dpo_beta"] = beta
    if _sig_has_param(DPOTrainer.__init__, "greater_is_better"):
        dpo_kwargs["greater_is_better"] = greater_is_better
    if _sig_has_param(DPOTrainer.__init__, "processing_class"):
        dpo_kwargs["processing_class"] = tokenizer

    trainer = DPOTrainer(**_filter_kwargs_for_signature(DPOTrainer.__init__, dpo_kwargs))
    return trainer


# ---- ORPO ---------------------------------------------------------------


def build_orpo_trainer(
    model,
    tokenizer,
    train_dataset,
    *,
    alpha: float,
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
    if not _HAS_TRL_ORPO:
        raise RuntimeError("TRL ORPOTrainer is not available. Please install/update `trl`.")

    args_obj = _build_args_obj(
        prefer_config_cls=ORPOConfig if _HAS_TRL_ORPOCONFIG else None,  # type: ignore
        fallback_ta_cls=TrainingArguments,
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to=report_to,
        max_grad_norm=max_grad_norm,
        dataloader_pin_memory=dataloader_pin_memory,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        include_model_kwargs=False,
    )

    orpo_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args_obj,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
    }
    for cand in ("alpha", "orpo_alpha", "beta"):
        if _sig_has_param(ORPOTrainer.__init__, cand):
            orpo_kwargs[cand] = alpha
            break
    if _sig_has_param(ORPOTrainer.__init__, "greater_is_better"):
        orpo_kwargs["greater_is_better"] = greater_is_better
    if _sig_has_param(ORPOTrainer.__init__, "processing_class"):
        orpo_kwargs["processing_class"] = tokenizer

    trainer = ORPOTrainer(**_filter_kwargs_for_signature(ORPOTrainer.__init__, orpo_kwargs))
    return trainer


# ---- KTO ----------------------------------------------------------------


def build_kto_trainer(
    model,
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
    desirable_weight: Optional[float] = None,
    undesirable_weight: Optional[float] = None,
):
    if not _HAS_TRL_KTO:
        raise RuntimeError("TRL KTOTrainer is not available. Please install/update `trl`.")

    # Build args obj
    args_obj = _build_args_obj(
        prefer_config_cls=KTOConfig if _HAS_TRL_KTOCONFIG else None,  # type: ignore
        fallback_ta_cls=TrainingArguments,
        output_dir=output_dir,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to=report_to,
        max_grad_norm=max_grad_norm,
        dataloader_pin_memory=dataloader_pin_memory,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        include_model_kwargs=False,
    )
    # 🔒 Force gradient checkpointing off at the Trainer level (some KTO builds re-enable it)
    try:
        setattr(args_obj, "gradient_checkpointing", False)
    except Exception:
        pass

    kto_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args_obj,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
    }
    if _sig_has_param(KTOTrainer.__init__, "beta"):
        kto_kwargs["beta"] = beta
    elif _sig_has_param(KTOTrainer.__init__, "kto_beta"):
        kto_kwargs["kto_beta"] = beta
    if _sig_has_param(KTOTrainer.__init__, "greater_is_better"):
        kto_kwargs["greater_is_better"] = greater_is_better
    if _sig_has_param(KTOTrainer.__init__, "processing_class"):
        kto_kwargs["processing_class"] = tokenizer
    # class weights
    if desirable_weight is not None:
        for cand in ("desirable_weight", "positive_weight"):
            if _sig_has_param(KTOTrainer.__init__, cand):
                kto_kwargs[cand] = desirable_weight
                break
    if undesirable_weight is not None:
        for cand in ("undesirable_weight", "negative_weight"):
            if _sig_has_param(KTOTrainer.__init__, cand):
                kto_kwargs[cand] = undesirable_weight
                break

    trainer = KTOTrainer(**_filter_kwargs_for_signature(KTOTrainer.__init__, kto_kwargs))
    return trainer
