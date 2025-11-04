# src/airoad/sft/lora_sft.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------
# Formatting / data prep
# ---------------------------

INSTR_PROMPT = "### Instruction:\n{instruction}\n\n" "### Input:\n{input}\n\n" "### Response:\n"


def format_example(ex: Dict[str, str]) -> Tuple[str, str]:
    """Return (prompt, target) using an Alpaca-like format."""
    instruction = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    out = ex.get("output", "").strip()
    prompt = INSTR_PROMPT.format(instruction=instruction, input=inp)
    return prompt, out


def build_alpaca_like_examples() -> List[Dict[str, str]]:
    """Tiny, in-repo instruct set for CPU/MPS demos (no network)."""
    return [
        {
            "instruction": "Summarize: The cat sat on the mat.",
            "input": "",
            "output": "The cat sat on a mat.",
        },
        {
            "instruction": "Rewrite in a friendly tone",
            "input": "I cannot attend the meeting.",
            "output": "Sorry, I won’t be able to make the meeting.",
        },
        {"instruction": "Answer: What is 2+2?", "input": "", "output": "4"},
        {
            "instruction": "Provide a short encouragement",
            "input": "",
            "output": "You’ve got this! Keep going.",
        },
        {"instruction": "Translate to French", "input": "Good morning", "output": "Bonjour"},
        {
            "instruction": "Give a haiku about rain",
            "input": "",
            "output": "Soft rain taps the earth\nWindows hum with silver threads\nMorning wears a smile",
        },
        {
            "instruction": "Turn into a bullet list",
            "input": "apples bananas carrots",
            "output": "- apples\n- bananas\n- carrots",
        },
        {
            "instruction": "Polish the sentence",
            "input": "this code is bad",
            "output": "This code could use some improvements.",
        },
        {
            "instruction": "Define a variable in Python named x with value 3",
            "input": "",
            "output": "x = 3",
        },
        {
            "instruction": "Respond kindly",
            "input": "I failed the test",
            "output": "I’m sorry you’re disappointed. One setback doesn’t define you; you can learn and improve.",
        },
    ]


# ---------------------------
# Training (version-robust)
# ---------------------------


@dataclass
class SFTConfig:
    base_model: str = "sshleifer/tiny-gpt2"
    output_dir: str = "outputs/sft-lora"
    max_steps: int = 50
    learning_rate: float = 5e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_len: int = 256
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    use_trl: bool = True  # if TRL available; otherwise plain Trainer


def run_lora_sft(cfg: SFTConfig) -> None:
    """
    Tiny LoRA SFT run on a tiny model; CPU/MPS-friendly.
    Requires: transformers, peft; optionally trl.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as e:
        raise RuntimeError("Transformers required: pip install transformers") from e

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError("peft required: pip install peft") from e

    # Optional TRL (SFTTrainer)
    has_trl = False
    SFTTrainer = None
    if cfg.use_trl:
        try:
            from trl import SFTTrainer as _SFTTrainer

            SFTTrainer = _SFTTrainer
            has_trl = True
        except Exception:
            has_trl = False

    # Build tiny dataset (no downloads)
    pairs = [format_example(ex) for ex in build_alpaca_like_examples()]
    texts = [(p + t) for (p, t) in pairs]

    # Tokenizer / model
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)

    # Apply LoRA (GPT-2 tiny: target c_attn & c_proj; fan_in_fan_out=True for Conv1D)
    lconf = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=True,
    )
    model = get_peft_model(model, lconf)

    # TrainingArguments (for both TRL and plain Trainer)
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=10,
        save_steps=0,
        report_to=[],
    )

    # ---------- Try TRL SFTTrainer with multiple signatures ----------
    if has_trl and SFTTrainer is not None:
        dataset = [{"text": t} for t in texts]
        # 1) Newer TRL signature
        try:
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                tokenizer=tok,
                max_seq_length=cfg.max_seq_len,
                packing=False,
                dataset_text_field="text",
            )
            trainer.train()
            trainer.save_model(cfg.output_dir)
            return
        except TypeError:
            pass
        # 2) Older TRL: no dataset_text_field, maybe no tokenizer kwarg
        try:
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=dataset,
                max_seq_length=cfg.max_seq_len,
                packing=False,
            )
            trainer.train()
            trainer.save_model(cfg.output_dir)
            return
        except Exception:
            # fallthrough to plain Trainer
            has_trl = False

    # ---------- Plain HF Trainer fallback (robust across environments) ----------
    enc = tok(texts, truncation=True, max_length=cfg.max_seq_len, padding=True, return_tensors=None)
    train_dataset = [
        {"input_ids": ids, "attention_mask": am}
        for ids, am in zip(enc["input_ids"], enc["attention_mask"])
    ]

    def collate(batch):
        import torch

        maxlen = max(len(b["input_ids"]) for b in batch)
        ids, attn = [], []
        for b in batch:
            pad_len = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.eos_token_id] * pad_len)
            attn.append(b["attention_mask"] + [1] * pad_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }

    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, data_collator=collate, tokenizer=tok
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
