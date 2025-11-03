from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GenConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0          # greedy by default
    top_p: float = 1.0
    do_sample: bool = False


def _require_tf():
    try:
        import transformers  # noqa: F401
    except Exception as e:
        raise RuntimeError("Transformers required: pip install transformers") from e


def load_base(model_name: str):
    """
    Load a base Causal LM + tokenizer. Align pad/eos.
    """
    _require_tf()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok


def load_with_lora(model_name: str, lora_dir: str):
    """
    Load base + LoRA adapters (PEFT).
    """
    _require_tf()
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("peft required: pip install peft") from e

    model, tok = load_base(model_name)
    model = PeftModel.from_pretrained(model, lora_dir)
    return model, tok


def merge_and_save_lora(model_name: str, lora_dir: str, out_dir: str):
    """
    Load base + adapters, merge them into the base weights, and save to out_dir.
    """
    _require_tf()
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("peft required: pip install peft") from e

    base, tok = load_base(model_name)
    peft_model = PeftModel.from_pretrained(base, lora_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def generate(
    model, tok, prompts: List[str], cfg: Optional[GenConfig] = None
) -> List[str]:
    """
    Generate continuations for a list of prompts.
    """
    _require_tf()
    import torch

    if cfg is None:
        cfg = GenConfig()

    device = "cpu"
    model.to(device)
    model.eval()

    outs: List[str] = []
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        text = tok.decode(gen_ids[0], skip_special_tokens=True)
        outs.append(text)
    return outs
