# path: servers/applied_ai/app.py
# ---
#!/usr/bin/env python
from __future__ import annotations

import asyncio
import base64
import math
import mimetypes
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torchvision import utils as tvutils
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForCausalLM, AutoTokenizer

from airoad.generative.ddpm_mini import (  # DDPM-Mini diffusion :contentReference[oaicite:0]{index=0}
    DiffusionSchedule,
    TinyUNet,
    sample_loop,
)

try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# =====================================================================
# CONFIGURATION
# =====================================================================

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_ADAPTERS_DIR = os.getenv("LORA_ADAPTERS_DIR", "")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a compassionate, practical spiritual coach. Be concise, kind, and useful.",
)
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "160"))
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE", "0.2"))
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" | "mps" | "cuda"
DDPM_T = int(os.getenv("DDPM_T", "6"))
DDPM_CKPT_NAME = os.getenv("DDPM_MNIST_CKPT", "ddpm_mini.pth")

app = FastAPI(title="Applied AI Server", version="0.2.0")

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

# =====================================================================
# OUTPUTS ROOT & PATH HELPERS
# =====================================================================


def get_outputs_root() -> Path:
    """
    Resolve the root for all artifacts.

    Priority:
      1) OUTPUTS_ROOT env
      2) /app/outputs (container default)
      3) ./outputs (local dev)
    """
    env = os.getenv("OUTPUTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    candidates = [Path("/app/outputs"), Path("outputs")]
    for c in candidates:
        if c.exists():
            return c.resolve()
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


# =====================================================================
# LLM MODEL LOADING
# =====================================================================

_model = None
_tok = None
_lock = asyncio.Lock()  # LLM concurrency guard


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


# =====================================================================
# DDPM-MINI (MNIST DIFFUSION) STATE
# =====================================================================

_ddpm_model: Optional[TinyUNet] = None
_ddpm_sched: Optional[DiffusionSchedule] = None
_ddpm_device: Optional[str] = None
_ddpm_lock = asyncio.Lock()


def _ensure_ddpm_loaded() -> None:
    """Lazy-load DDPM-Mini model and schedule on first use."""
    global _ddpm_model, _ddpm_sched, _ddpm_device
    if _ddpm_model is not None and _ddpm_sched is not None:
        return

    dev = DEVICE  # reuse same device policy as LLM
    _ddpm_device = dev
    _ddpm_sched = DiffusionSchedule(T=DDPM_T, device=dev)
    model = TinyUNet(base_c=32, with_t_embed=True, T_max=DDPM_T)
    model.to(dev).eval()

    # Optional checkpoint: outputs/ddpm_mini.pth relative to outputs root
    ckpt_path = get_outputs_root() / DDPM_CKPT_NAME
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state)
    _ddpm_model = model


# =====================================================================
# SCHEMAS
# =====================================================================


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


class StartLoRARequest(BaseModel):
    config_path: str


class DDPMSampleRequest(BaseModel):
    n: int = Field(16, ge=1, le=64, description="Number of samples to draw")
    nrow: Optional[int] = Field(
        None, ge=1, le=64, description="Grid columns (defaults to sqrt(n) rounded)"
    )
    seed: Optional[int] = Field(None, description="Optional torch.manual_seed for reproducibility")


class DDPMSampleResponse(BaseModel):
    image: str  # data URL: data:image/png;base64,...
    n: int
    T: int
    out_path: Optional[str] = None  # relative to outputs root (for Runs browser)


# =====================================================================
# HEALTH / META
# =====================================================================


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> Dict[str, str]:
    if not _is_ready():
        raise HTTPException(503, "model not loaded")
    return {"status": "ready"}


@app.get("/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
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


# =====================================================================
# CHAT
# =====================================================================


@app.post("/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
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


# =====================================================================
# LORA JOB MANAGEMENT
# =====================================================================

JOBS: Dict[str, Dict[str, Any]] = {}


@app.post("/jobs/train/lora")
async def start_lora_job(req: StartLoRARequest) -> Dict[str, str]:
    """
    Launch a LoRA training job as a background subprocess.

    Uses scripts/slm/qwen05b_lora_train.py with the given config path.
    """
    job_id = str(uuid.uuid4())
    root = get_outputs_root()
    output_dir = root / f"lora_job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    script = [
        "python",
        "scripts/slm/qwen05b_lora_train.py",
        "--config",
        req.config_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "output_dir": str(output_dir),
        "lines": [],
        "process": proc,
    }

    async def pump_logs():
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                JOBS[job_id]["lines"].append(text)
        finally:
            rc = await proc.wait()
            JOBS[job_id]["status"] = "completed" if rc == 0 else "error"

    asyncio.create_task(pump_logs())

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "id": job_id,
        "status": job["status"],
        "lines": job["lines"][-200:],  # tail
        "output_dir": job["output_dir"],
    }


# =====================================================================
# GENERIC ARTIFACTS API (/artifacts/*)
# =====================================================================


@app.get("/artifacts/list", response_model=ArtifactListResponse)
async def list_artifacts(
    path: Optional[str] = Query(
        default=None,
        description="Optional relative subdirectory under outputs/ to list.",
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
        stat = full.stat()
        text = f"[binary file] {full.name} ({stat.st_size} bytes)"

    return ArtifactFileResponse(content=text)


# =====================================================================
# CV ARTIFACTS API (/cv/*)
# =====================================================================


def _collect_cv_files(root: Path, max_items: int = 512) -> List[Path]:
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

    stat = full.stat()
    msg = f"[binary file] {full.name} ({stat.st_size} bytes)"
    return CVArtifactPreview(kind="bin", content=msg)


# =====================================================================
# DIFFUSION API (DDPM-MINI → MNIST GRID)
# =====================================================================


@app.post("/diffusion/mnist/sample", response_model=DDPMSampleResponse)
async def diffusion_mnist_sample(req: DDPMSampleRequest) -> DDPMSampleResponse:
    """
    Sample a tiny MNIST grid from the DDPM-Mini model.

    Returns:
      - image: data:image/png;base64,... grid
      - out_path: relative PNG path under outputs/ for the Runs browser
    """
    async with _ddpm_lock:
        _ensure_ddpm_loaded()
        if _ddpm_model is None or _ddpm_sched is None:
            raise HTTPException(500, "DDPM model failed to load")

        if req.seed is not None:
            torch.manual_seed(req.seed)

        n = req.n
        nrow = req.nrow or max(1, int(math.sqrt(n)))
        dev = _ddpm_device or DEVICE

        samples = sample_loop(_ddpm_model, _ddpm_sched, n=n, device=dev).cpu()
        grid = tvutils.make_grid(samples, nrow=nrow, padding=2)
        pil = to_pil_image(grid)

        # Encode as data URL
        buf = BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        # Save under outputs/diffusion/ddpm_mini for later browsing
        out_root = get_outputs_root()
        out_dir = out_root / "diffusion" / "ddpm_mini"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f"mnist_grid_{ts}.png"
        pil.save(out_path)
        rel = out_path.relative_to(out_root).as_posix()

        return DDPMSampleResponse(
            image=data_url,
            n=n,
            T=_ddpm_sched.T,
            out_path=rel,
        )
