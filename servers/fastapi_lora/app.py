#!/usr/bin/env python
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# -------- settings via env ----------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_ADAPTERS_DIR = os.getenv("LORA_ADAPTERS_DIR", "")  # if empty -> use merged or pure base
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a compassionate, practical spiritual coach. Be concise, kind, and useful.",
)
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "160"))
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE", "0.2"))

# CPU container default; running natively on mac can use device="mps"
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(title="Qwen-LoRA FastAPI", version="0.1.0")
_model = None
_tok = None
_lock = asyncio.Lock()  # simple concurrency guard


def _load_base(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    return model, tok


def _load_base_plus_lora(model_id: str, lora_dir: str):
    if not _HAS_PEFT:
        raise RuntimeError("peft not installed; cannot load LoRA adapters")
    model, tok = _load_base(model_id)
    model = PeftModel.from_pretrained(model, lora_dir)
    return model, tok


def build_chatml_prompt(tok, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    # Qwen chat template → prefill prompt for assistant
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_once(model, tok, prompt: str, max_new_tokens: int, temperature: float) -> str:
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # extract assistant turn (template already includes assistant prefill)
    return text.split("</s>")[-1].strip() if "</s>" in text else text.strip()


@app.on_event("startup")
def startup_event():
    global _model, _tok
    # Try base+adapters first if provided; else assume merged or pure base in MODEL_ID
    if LORA_ADAPTERS_DIR:
        _model, _tok = _load_base_plus_lora(MODEL_ID, LORA_ADAPTERS_DIR)
    else:
        _model, _tok = _load_base(MODEL_ID)
    _model.to(DEVICE).eval()
    # tiny warmup
    _ = _tok("hi", return_tensors="pt")


# -------- simple schema ----------
class ChatRequest(BaseModel):
    user: str = Field(..., description="User message content")
    system: Optional[str] = Field(None, description="Optional system override")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.5)


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 160


@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    global _model, _tok
    if _model is None or _tok is None:
        raise HTTPException(503, "model not loaded")
    system = req.system or SYSTEM_PROMPT
    prompt = build_chatml_prompt(_tok, system, req.user)
    async with _lock:
        text = await asyncio.get_event_loop().run_in_executor(
            None,
            generate_once,
            _model,
            _tok,
            prompt,
            int(req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT),
            float(req.temperature or TEMPERATURE_DEFAULT),
        )
    return {"answer": text}


@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest) -> Dict[str, Any]:
    global _model, _tok
    if _model is None or _tok is None:
        raise HTTPException(503, "model not loaded")

    # Build ChatML from OpenAI-style messages
    sys = SYSTEM_PROMPT
    usr_parts = []
    for m in req.messages:
        if m.role == "system":
            sys = m.content
        elif m.role == "user":
            usr_parts.append(m.content)
    user_text = "\n\n".join(usr_parts) if usr_parts else ""

    prompt = build_chatml_prompt(_tok, sys, user_text)
    async with _lock:
        text = await asyncio.get_event_loop().run_in_executor(
            None,
            generate_once,
            _model,
            _tok,
            prompt,
            int(req.max_tokens or MAX_NEW_TOKENS_DEFAULT),
            float(req.temperature or TEMPERATURE_DEFAULT),
        )
    # minimal OpenAI-compatible response
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {},
    }
