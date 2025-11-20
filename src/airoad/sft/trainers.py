#!/usr/bin/env python
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from transformers import Trainer, TrainingArguments

# TRL is optional; if missing we still export the builders (HF path will work)
try:
    from trl import SFTConfig, SFTTrainer  # type: ignore

    _HAS_TRL = True
except Exception:
    SFTConfig = object  # type: ignore
    SFTTrainer = object  # type: ignore
    _HAS_TRL = False

"""
src/airoad/sft/trainers.py

Trainer builders:
- KLSFTTrainer: TRL SFTTrainer with KL regularization to a frozen reference model
- KLTrainer: HF Trainer fallback with the same KL term
- build_trl_trainer / build_hf_trainer helpers

Compatibility:
- Recent TRL/Transformers pass `num_items_in_batch` to `compute_loss`. We accept it
  (and **kwargs) to avoid TypeError across versions.
"""

# -------------------------------
# KL-regularized trainers
# -------------------------------


class KLSFTTrainer(SFTTrainer):  # type: ignore[misc]
    """
    SFTTrainer with a KL penalty to a frozen reference (base) model.
    loss = CE + kl_lambda * KL( teacher || student ), with temperature kl_tau
    """

    def __init__(
        self, *args, ref_model=None, kl_lambda: float = 0.0, kl_tau: float = 1.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.kl_lambda = float(kl_lambda)
        self.kl_tau = float(kl_tau)
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()

    def compute_loss(  # type: ignore[override]
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        # Shift for next-token CE
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss = ce

        # KL penalty to the frozen teacher (reference/base) if enabled
        if self.ref_model is not None and self.kl_lambda > 0.0:
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_out.logits[:, :-1, :].contiguous()

            tau = self.kl_tau
            log_p_s = F.log_softmax(shift_logits / tau, dim=-1)
            log_p_t = F.log_softmax(ref_logits / tau, dim=-1)
            p_t = torch.exp(log_p_t)

            # Token-level KL(teacher || student)
            kl_pos = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # [B, T-1]

            mask = (shift_labels != -100).float()
            kl = (kl_pos * mask).sum() / mask.sum().clamp_min(1.0)

            # tau^2 scaling maintains gradient scale roughly consistent
            loss = loss + self.kl_lambda * (tau * tau) * kl

        return (loss, outputs) if return_outputs else loss


class KLTrainer(Trainer):
    """HF Trainer fallback with the same KL penalty."""

    def __init__(
        self, *args, ref_model=None, kl_lambda: float = 0.0, kl_tau: float = 1.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.kl_lambda = float(kl_lambda)
        self.kl_tau = float(kl_tau)
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()

    def compute_loss(  # type: ignore[override]
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss = ce

        if self.ref_model is not None and self.kl_lambda > 0.0:
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_out.logits[:, :-1, :].contiguous()

            tau = self.kl_tau
            log_p_s = F.log_softmax(shift_logits / tau, dim=-1)
            log_p_t = F.log_softmax(ref_logits / tau, dim=-1)
            p_t = torch.exp(log_p_t)
            kl_pos = (p_t * (log_p_t - log_p_s)).sum(dim=-1)

            mask = (shift_labels != -100).float()
            kl = (kl_pos * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + self.kl_lambda * (tau * tau) * kl

        return (loss, outputs) if return_outputs else loss


# -------------------------------
# Builder helpers
# -------------------------------


def build_trl_trainer(
    model,
    train_dataset,
    eval_dataset,
    sft_args: SFTConfig,  # type: ignore[name-defined]
    *,
    ref_model=None,
    kl_lambda: float = 0.0,
    kl_tau: float = 1.0,
):
    """
    Build a TRL trainer (SFTTrainer or KLSFTTrainer depending on kl_lambda).
    """
    if not _HAS_TRL:
        raise RuntimeError("TRL is not available; cannot build SFTTrainer.")

    if kl_lambda > 0.0 and ref_model is not None:
        return KLSFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ref_model=ref_model,
            kl_lambda=kl_lambda,
            kl_tau=kl_tau,
        )
    return SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def build_hf_trainer(
    model,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    *,
    data_collator,
    tokenizer,
    ref_model=None,
    kl_lambda: float = 0.0,
    kl_tau: float = 1.0,
):
    """
    Build an HF Trainer, optionally KL-regularized if kl_lambda>0 and ref_model provided.
    """
    if kl_lambda > 0.0 and ref_model is not None:
        return KLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            ref_model=ref_model,
            kl_lambda=kl_lambda,
            kl_tau=kl_tau,
        )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
