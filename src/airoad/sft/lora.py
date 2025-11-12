#!/usr/bin/env python
from __future__ import annotations

from typing import List, Tuple

try:
    from peft import LoraConfig, get_peft_model  # type: ignore

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False
"""
src/airoad/sft/lora.py

LoRA / DoRA utilities:
- target detection (auto, attention-only, attention+MLP)
- robust PEFT LoraConfig construction with optional DoRA
- apply_lora(...) to wrap a base model
"""


def _has_name(names: List[str], s: str) -> bool:
    return any(s in n for n in names)


def detect_targets_auto(model) -> List[str]:
    names = [n for n, _ in model.named_modules()]
    if _has_name(names, "q_proj") and _has_name(names, "v_proj"):
        t = ["q_proj", "k_proj", "v_proj"]
        t += [x for x in ("o_proj", "out_proj") if _has_name(names, x)]
        for mlp in ("gate_proj", "up_proj", "down_proj"):
            if _has_name(names, mlp):
                t.append(mlp)
        return t
    if _has_name(names, "c_attn"):
        return ["c_attn", "c_proj", "c_fc"]
    if _has_name(names, "query_key_value"):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    return []


def detect_targets_by_mode(model, mode: str) -> List[str]:
    mode = (mode or "auto").lower()
    names = [n for n, _ in model.named_modules()]

    if mode == "attn":
        t = ["q_proj", "k_proj", "v_proj"]
        t += [x for x in ("o_proj", "out_proj") if _has_name(names, x)]
        return t

    if mode in ("attn_mlp", "attn+mlp"):
        t = ["q_proj", "k_proj", "v_proj"]
        t += [x for x in ("o_proj", "out_proj") if _has_name(names, x)]
        for mlp in ("gate_proj", "up_proj", "down_proj"):
            if _has_name(names, mlp):
                t.append(mlp)
        return t

    return detect_targets_auto(model)


def build_lora_config(
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: List[str],
    use_dora: bool = False,
):
    """
    Construct a PEFT LoraConfig, tolerating older versions (where use_dora is unsupported).
    """
    if not _HAS_PEFT:
        raise RuntimeError("PEFT is not installed.")

    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # attach use_dora if supported by this PEFT
    try:
        cfg = LoraConfig(use_dora=use_dora, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Fall back if the installed PEFT doesn't support use_dora
        cfg = LoraConfig(**kwargs)

    return cfg


def apply_lora(
    model,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_mode: str = "auto",
    use_dora: bool = False,
) -> Tuple[object, List[str]]:
    """
    Detect targets according to target_mode, wrap model with PEFT LoRA/DoRA, and return (model, targets).
    """
    if not _HAS_PEFT:
        return model, []

    targets = detect_targets_by_mode(model, target_mode)
    if not targets:
        return model, []

    cfg = build_lora_config(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=targets,
        use_dora=use_dora,
    )
    wrapped = get_peft_model(model, cfg)
    return wrapped, targets
