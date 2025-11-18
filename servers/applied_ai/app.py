# path: servers/applied_ai/app.py
# ---
#!/usr/bin/env python
from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
DEVICE = os.getenv("DEVICE", "cpu")  # cpu | mps | cuda

app = FastAPI(title="Qwen-LoRA FastAPI", version="0.1.0")

# ✨ CORS (allow local dev & prod SPA by default; override via CORS_ORIGINS)
_default_cors = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://host.docker.internal:5173",
    "http://host.docker.internal:8080",
]
_cors_env = os.getenv("CORS_ORIGINS")
_cors_list = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else _default_cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tok = None
_lock = asyncio.Lock()  # simple concurrency guard


# -------- filesystem helpers (outputs/) ----------


def get_outputs_root() -> Path:
    """
    Resolve the root for all artifacts.

    Priority:
      1) OUTPUTS_ROOT env (absolute or relative)
      2) /app/outputs (container default)
      3) ./outputs (local dev from repo root)
    """
    env = os.getenv("OUTPUTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    candidates = [Path("/app/outputs"), Path("outputs")]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # fallback even if not present yet
    return candidates[0].resolve()


def _safe_join(root: Path, rel: str) -> Path:
    """
    Join a user-provided relative path to root, preventing path traversal.
    """
    if rel.startswith(("/", "\\")):
        raise HTTPException(400, detail="absolute paths are not allowed")
    candidate = (root / rel).resolve()
    root = root.resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    raise HTTPException(400, detail="invalid path")


# -------- model loading & generation ----------


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
    return text.split("</s>")[-1].strip() if "</s>" in text else text.strip()


def _is_ready() -> bool:
    return _model is not None and _tok is not None


@app.on_event("startup")
def startup_event():
    global _model, _tok
    if LORA_ADAPTERS_DIR:
        _model, _tok = _load_base_plus_lora(MODEL_ID, LORA_ADAPTERS_DIR)
    else:
        _model, _tok = _load_base(MODEL_ID)
    _model.to(DEVICE).eval()
    _ = _tok("hi", return_tensors="pt")


# -------- schemas ----------


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


class MetaResponse(BaseModel):
    model_id: str
    device: str
    has_peft: bool
    lora_adapters_dir: Optional[str]
    using_lora: bool
    max_new_tokens_default: int
    temperature_default: float
    ready: bool
    server_version: str


class ArtifactItem(BaseModel):
    name: str
    path: str  # relative to outputs root
    type: Literal["dir", "file"]


class ArtifactListResponse(BaseModel):
    items: List[ArtifactItem]


class ArtifactFileResponse(BaseModel):
    content: str


class CVArtifactPreview(BaseModel):
    kind: Literal["text", "image", "bin"]
    content: str


# -------- health & meta ----------


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """
    Liveness probe: if this returns 200, the process is up.
    Does not imply the model is loaded.
    """
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> Dict[str, str]:
    """
    Readiness probe: only 200 once the model & tokenizer are loaded.
    """
    if not _is_ready():
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/meta", response_model=MetaResponse)
async def meta() -> MetaResponse:
    """
    Small metadata surface for the System Status UI.
    """
    return MetaResponse(
        model_id=MODEL_ID,
        device=DEVICE,
        has_peft=_HAS_PEFT,
        lora_adapters_dir=LORA_ADAPTERS_DIR or None,
        using_lora=bool(LORA_ADAPTERS_DIR),
        max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT,
        temperature_default=TEMPERATURE_DEFAULT,
        ready=_is_ready(),
        server_version=app.version or "0.1.0",
    )


# -------- chat endpoints ----------


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

    sys = SYSTEM_PROMPT
    usr_parts: List[str] = []
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
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {},
    }


# -------- generic artifacts API (/artifacts/*) ----------


@app.get("/artifacts/list", response_model=ArtifactListResponse)
async def list_artifacts(
    path: Optional[str] = Query(
        default=None,
        description="Optional relative subdirectory under outputs/ to list. If omitted, list top-level.",
    )
) -> ArtifactListResponse:
    root = get_outputs_root()
    base = root if not path else _safe_join(root, path)

    if not base.exists() or not base.is_dir():
        return ArtifactListResponse(items=[])

    items: List[ArtifactItem] = []
    for entry in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        typ: Literal["dir", "file"] = "dir" if entry.is_dir() else "file"
        rel_path = entry.relative_to(root).as_posix()
        items.append(ArtifactItem(name=entry.name, path=rel_path, type=typ))

    return ArtifactListResponse(items=items)


@app.get("/artifacts/file", response_model=ArtifactFileResponse)
async def read_artifact(
    path: str = Query(..., description="Relative file path under outputs/"),
) -> ArtifactFileResponse:
    root = get_outputs_root()
    full = _safe_join(root, path)

    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    try:
        text = full.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback: binary or non-UTF8; just show a small hint.
        stat = full.stat()
        text = f"[binary file] {full.name} ({stat.st_size} bytes)"

    return ArtifactFileResponse(content=text)


# -------- CV artifacts API (/cv/*) ----------


def _collect_cv_files(root: Path, max_items: int = 512) -> List[Path]:
    """
    Scan outputs/cv and outputs/cv_images for interesting artifacts.

    We include:
      - annotated images (*.png/jpg/jpeg/webp)
      - JSON/CSV summaries and detections
      - logs
    """
    img_exts = {".png", ".jpg", ".jpeg", ".webp"}
    text_exts = {".json", ".csv", ".log", ".txt"}
    interesting_exts = img_exts | text_exts

    out: List[Path] = []
    for base in (root / "cv", root / "cv_images"):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() in interesting_exts:
                out.append(p)
                if len(out) >= max_items:
                    return out
    return out


@app.get("/cv/artifacts", response_model=ArtifactListResponse)
async def list_cv_artifacts() -> ArtifactListResponse:
    """
    List a flattened set of "interesting" CV artifacts under outputs/cv and outputs/cv_images.
    """
    root = get_outputs_root()
    files = _collect_cv_files(root)
    items: List[ArtifactItem] = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        items.append(ArtifactItem(name=f.name, path=rel, type="file"))
    return ArtifactListResponse(items=items)


@app.get("/cv/artifact", response_model=CVArtifactPreview)
async def cv_artifact(
    path: str = Query(..., description="Relative file path under outputs/ for CV artifact"),
) -> CVArtifactPreview:
    root = get_outputs_root()
    full = _safe_join(root, path)

    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    suffix = full.suffix.lower()
    img_exts = {".png", ".jpg", ".jpeg", ".webp"}
    text_exts = {".json", ".csv", ".log", ".txt"}

    if suffix in img_exts:
        mime, _ = mimetypes.guess_type(full.name)
        if mime is None:
            mime = "image/png"
        data = full.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        return CVArtifactPreview(kind="image", content=data_url)

    if suffix in text_exts:
        text = full.read_text(encoding="utf-8", errors="replace")
        return CVArtifactPreview(kind="text", content=text)

    # Fallback for other binary blobs (e.g., mp4)
    stat = full.stat()
    msg = f"[binary file] {full.name} ({stat.st_size} bytes)"
    return CVArtifactPreview(kind="bin", content=msg)
