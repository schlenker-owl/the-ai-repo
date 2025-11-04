#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Optional deps
try:
    from peft import LoraConfig, get_peft_model  # type: ignore

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

try:
    from trl import SFTTrainer  # type: ignore

    _HAS_TRL = True
except Exception:
    _HAS_TRL = False

try:
    from trl import SFTConfig  # type: ignore

    _HAS_SFTCONFIG = True
except Exception:
    _HAS_SFTCONFIG = False

try:
    from datasets import Dataset  # type: ignore

    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

try:
    import wandb  # type: ignore

    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# Local evaluator
from airoad.sft.eval_sft import evaluate_and_print

PROMPT_TPL = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _load_jsonl_or_toy(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return [
            {"instruction": "Say hello", "input": "", "output": "hello"},
            {"instruction": "Add", "input": "2+3", "output": "5"},
            {"instruction": "Uppercase", "input": "hello world", "output": "HELLO WORLD"},
            {"instruction": "Opposite of hot", "input": "", "output": "cold"},
        ]
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _format_text(row: Dict[str, str]) -> str:
    inst = row.get("instruction", "")
    inp = row.get("input", "")
    out = row.get("output", "")
    return PROMPT_TPL.format(instruction=inst, input=inp) + out


def _build_text_list(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"text": _format_text(r)} for r in rows]


def _ensure_tok_pad(tok):
    # For decoder-only models (GPT-2/LLaMA/etc.) set left padding for stable generation.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _maybe_init_wandb(enable: bool, project: str, config: dict):
    if enable and _HAS_WANDB:
        if not wandb.run:
            wandb.init(project=project, config=config)


def _detect_lora_targets(model: nn.Module) -> List[str]:
    names = [n for n, _ in model.named_modules()]

    def has(substr: str) -> bool:
        return any(substr in n for n in names)

    # GPT-2 style (tiny-gpt2)
    if has("c_attn"):
        return ["c_attn", "c_proj", "c_fc"]

    # LLaMA/Mistral/Gemma/Qwen style
    if has("q_proj") and has("v_proj"):
        targets = ["q_proj", "k_proj", "v_proj"]
        if has("o_proj"):
            targets.append("o_proj")
        elif has("out_proj"):
            targets.append("out_proj")
        for mlp_name in ("gate_proj", "up_proj", "down_proj"):
            if has(mlp_name):
                targets.append(mlp_name)
        return targets

    # GPT-NeoX/Falcon-ish
    if has("query_key_value"):
        targets = ["query_key_value"]
        for mlp_name in ("dense", "dense_h_to_4h", "dense_4h_to_h"):
            if has(mlp_name):
                targets.append(mlp_name)
        return targets

    # Conservative fallback: pick common Linear leaf suffixes
    leaf_suffixes = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            leaf_suffixes.add(n.split(".")[-1])
    preferred = [
        x for x in leaf_suffixes if x in ("c_proj", "c_fc", "out_proj", "proj", "fc", "mlp")
    ]
    if preferred:
        return sorted(set(preferred))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="sshleifer/tiny-gpt2")
    ap.add_argument("--dataset", type=str, default="data/processed/alpaca_format.jsonl")
    ap.add_argument("--output-dir", type=str, default="outputs/local-lora")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--method", type=str, choices=["none", "lora"], default="lora")
    ap.add_argument(
        "--target-modules",
        type=str,
        default="auto",
        help="Comma-separated substrings for LoRA target modules, or 'auto' to detect per model.",
    )
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-after-train", action="store_true")
    ap.add_argument("--eval-dev-path", type=str, default="data/sft/dev_toy.jsonl")
    ap.add_argument("--eval-limit", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--log-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="airepo-sft")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = _load_jsonl_or_toy(Path(args.dataset))
    text_rows = _build_text_list(rows)  # [{"text": "..."}]

    tok = _ensure_tok_pad(AutoTokenizer.from_pretrained(args.model_name, use_fast=True))
    device = _device()

    # Prepare model (dtype= to avoid deprecation warnings)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    model.to(device)

    # Decide LoRA targets
    is_lora = args.method == "lora"
    target_modules: List[str] = []
    if is_lora and not _HAS_PEFT:
        print("[train_sft_lora] PEFT not installed; falling back to full-finetune.")
        is_lora = False
    elif is_lora:
        if args.target_modules.strip().lower() == "auto":
            target_modules = _detect_lora_targets(model)
        else:
            target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
        if not target_modules:
            print(
                "[train_sft_lora] No compatible target modules found; falling back to full-finetune."
            )
            is_lora = False
        else:
            print(f"[train_sft_lora] Using LoRA targets: {target_modules}")

    # ---------- Build dataset objects ----------
    # TRL expects a HuggingFace datasets.Dataset with .map/.column_names
    if _HAS_TRL and _HAS_DATASETS:
        hf_ds = Dataset.from_list(text_rows)  # column 'text'
        trl_train_dataset = hf_ds
        hf_fallback_dataset = None
    else:
        # Minimal PyTorch dataset for HF Trainer fallback
        class MapDataset(torch.utils.data.Dataset):
            column_names = ["text"]

            def __init__(self, data: List[Dict[str, str]]):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {"text": self.data[idx]["text"]}

            def __iter__(self):
                for i in range(len(self.data)):
                    yield {"text": self.data[i]["text"]}

        trl_train_dataset = None
        hf_fallback_dataset = MapDataset(text_rows)

    # Collator for HF Trainer fallback
    def collate(examples):
        texts_ = [e["text"] for e in examples]
        enc = tok(texts_, return_tensors="pt", padding=True, truncation=True)
        enc["labels"] = enc["input_ids"].clone()
        return enc

    # W&B (optional)
    _maybe_init_wandb(
        args.log_wandb,
        args.wandb_project,
        {
            "model": args.model_name,
            "method": ("lora" if is_lora else "full"),
            "max_steps": args.max_steps,
            "lr": args.lr,
            "targets": ",".join(target_modules) if target_modules else "",
        },
    )

    # ---------- Trainer setup ----------
    if _HAS_TRL and trl_train_dataset is not None:
        print("[train_sft_lora] Using TRL SFTTrainer")
        # Disable packing to avoid flash-attention requirement on CPU/MPS.
        if _HAS_SFTCONFIG:
            sft_args = SFTConfig(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                max_steps=args.max_steps,
                logging_steps=10,
                save_steps=args.max_steps,
                report_to=["wandb"] if (args.log_wandb and _HAS_WANDB) else [],
                max_length=512,
                packing=False,
            )
            lconf = (
                LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                if (is_lora and _HAS_PEFT)
                else None
            )
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=trl_train_dataset,
                processing_class=tok,
                formatting_func=lambda sample: sample["text"],
                peft_config=lconf,
            )
        else:
            trainer = SFTTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=args.output_dir,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.lr,
                    max_steps=args.max_steps,
                    logging_steps=10,
                    save_steps=args.max_steps,
                    report_to=["wandb"] if (args.log_wandb and _HAS_WANDB) else [],
                ),
                train_dataset=trl_train_dataset,
                processing_class=tok,
                formatting_func=lambda sample: sample["text"],
            )
            # Manual LoRA wrap for older TRL
            if is_lora and _HAS_PEFT:
                lconf = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lconf)
                trainer.model = model
    else:
        print("[train_sft_lora] Using HF Trainer fallback")
        # Only wrap with LoRA in the HF fallback
        if is_lora and _HAS_PEFT:
            lconf = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lconf)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.lr,
                max_steps=args.max_steps,
                logging_steps=10,
                save_steps=args.max_steps,
                report_to=["wandb"] if (args.log_wandb and _HAS_WANDB) else [],
            ),
            data_collator=collate,
            train_dataset=hf_fallback_dataset,
            tokenizer=tok,
        )

    # Train & save
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # Optional: quick eval on toy dev
    if args.eval_after_train:
        print("[train_sft_lora] Running post-train eval...")
        base_for_merge = args.model_name if (is_lora and _HAS_PEFT) else None
        evaluate_and_print(
            model_path=args.output_dir,
            base_model=base_for_merge,
            dev_path=args.eval_dev_path,
            limit=args.eval_limit,
            batch_size=2,
            max_new_tokens=args.max_new_tokens,
            log_wandb=args.log_wandb,
            wandb_project=args.wandb_project,
        )


if __name__ == "__main__":
    main()
