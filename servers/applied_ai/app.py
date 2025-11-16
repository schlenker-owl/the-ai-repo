#!/usr/bin/env python
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import pathlib
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional LoRA
try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# Optional CV libs (Ultralytics; otherwise we stub)
try:
    from ultralytics import YOLO  # pip install ultralytics

    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

try:
    from PIL import Image, ImageDraw

    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

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

# Job running & artifacts
OUTPUTS_DIR = pathlib.Path("outputs").resolve()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Applied CV settings
CV_MODELS = [
    m.strip() for m in os.getenv("CV_MODELS", "yolov8n.pt,yolov8n-seg.pt").split(",") if m.strip()
]
CV_OUTPUTS = (OUTPUTS_DIR / "cv").resolve()
CV_OUTPUTS.mkdir(parents=True, exist_ok=True)

# -------- app & CORS ----------
app = FastAPI(title="Qwen-LoRA FastAPI", version="0.1.0")

_default_cors = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://host.docker.internal:5173",
    "http://host.docker.internal:8080",
]
_cors_env = os.getenv("CORS_ORIGINS")  # comma-separated list
_cors_list = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else _default_cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list,
    allow_credentials=False,  # set True only if you use cookies/auth headers
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- global model state ----------
_model = None
_tok = None
_lock = asyncio.Lock()  # simple concurrency guard


# -------- helpers: text model ----------
def _load_base(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
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


# -------- text endpoints ----------
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


# =====================================================================================
# LoRA jobs (spawn training script) + artifact browsing for outputs/
# =====================================================================================

# Simple in-process job store for logs
_JOBS: Dict[str, Dict[str, Any]] = {}


def _spawn_reader(proc, q: queue.Queue):
    for line in iter(proc.stdout.readline, b""):
        q.put(line.decode(errors="ignore"))
    try:
        proc.stdout.close()
    except Exception:
        pass


@app.post("/jobs/train/lora")
def start_lora_job(payload: Dict[str, Any]):
    cfg = payload.get("config_path")
    if not cfg:
        raise HTTPException(400, "config_path required")
    jid = str(uuid.uuid4())[:8]
    # Use uv to match your repo’s env
    cmd = ["uv", "run", "python", "scripts/slm/qwen05b_lora_train.py", "--config", cfg]
    try:
        proc = torch.subprocess.Popen(cmd, stdout=torch.subprocess.PIPE, stderr=torch.subprocess.STDOUT, bufsize=1)  # type: ignore[attr-defined]
    except AttributeError:
        import subprocess

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_spawn_reader, args=(proc, q), daemon=True)
    t.start()
    _JOBS[jid] = {"proc": proc, "q": q, "status": "running", "lines": []}
    return {"job_id": jid}


@app.get("/jobs/{jid}")
def job_status(jid: str):
    job = _JOBS.get(jid)
    if not job:
        raise HTTPException(404, "job not found")
    proc = job["proc"]
    q = job["q"]
    while True:
        try:
            line = q.get_nowait()
            job["lines"].append(line.rstrip())
            if len(job["lines"]) > 400:
                job["lines"] = job["lines"][-400:]
        except queue.Empty:
            break
    if proc.poll() is not None and job["status"] == "running":
        job["status"] = "done" if proc.returncode == 0 else "error"
    return {"id": jid, "status": job["status"], "lines": job["lines"]}


@app.get("/artifacts/list")
def artifacts_list():
    items = []
    if OUTPUTS_DIR.exists():
        for p in sorted(OUTPUTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            items.append({"name": p.name, "path": str(p), "type": "dir" if p.is_dir() else "file"})
            # show time_metrics.json if present
            if p.is_dir():
                tm = p / "time_metrics.json"
                if tm.exists():
                    items.append(
                        {"name": f"{p.name}/time_metrics.json", "path": str(tm), "type": "file"}
                    )
    return {"items": items}


@app.get("/artifacts/file")
def artifacts_file(path: str):
    p = pathlib.Path(path).resolve()
    if not str(p).startswith(str(OUTPUTS_DIR)):
        raise HTTPException(400, "path outside outputs/")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    # text-ish only
    if p.suffix.lower() in (".json", ".txt", ".log", ".csv"):
        try:
            return {"content": p.read_text(errors="ignore")[:200_000]}
        except Exception as e:
            raise HTTPException(500, str(e))
    # small image view (rare in non-CV runs)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        try:
            b64 = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            return {"content": b64}
        except Exception as e:
            raise HTTPException(500, str(e))
    return {"content": f"[binary or large file: {p.name}]"}


# =====================================================================================
# Applied CV endpoints
# =====================================================================================


def _load_yolo(model_id: str):
    if not _HAS_YOLO:
        raise RuntimeError("Ultralytics not available")
    return YOLO(model_id)


def _pil_to_b64(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _draw_boxes(im, boxes: List[List[float]], labels: List[str], scores: List[float]):
    if not _HAS_PIL:
        return im
    draw = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), lbl, score in zip(boxes, labels, scores):
        draw.rectangle([x1, y1, x2, y2], outline=(94, 234, 212), width=3)  # teal
        text = f"{lbl}:{score:.2f}"
        tw = int(draw.textlength(text))
        draw.rectangle([x1, y1 - 18, x1 + tw + 6, y1], fill=(17, 20, 23))
        draw.text((x1 + 3, y1 - 15), text, fill=(235, 245, 255))
    return im


@app.get("/cv/models")
def cv_models():
    return {"models": CV_MODELS or ["yolov8n.pt"]}


@app.post("/cv/infer/image")
async def cv_infer_image(
    file: UploadFile = File(...),
    model_id: str = Form("yolov8n.pt"),
    task: str = Form("detect"),  # detect | segment (segment not used in stub)
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
):
    if not _HAS_PIL:
        raise HTTPException(500, "Pillow not available in this image")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    subdir = (CV_OUTPUTS / f"{stamp}_{pathlib.Path(file.filename).stem}").resolve()
    subdir.mkdir(parents=True, exist_ok=True)

    raw = await file.read()
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")

    if not _HAS_YOLO:
        # stub result
        w, h = im.size
        boxes = [[w * 0.1, h * 0.1, w * 0.4, h * 0.4]]
        labels = ["object"]
        scores = [0.80]
        overlay = _draw_boxes(im.copy(), boxes, labels, scores)
        preview = _pil_to_b64(overlay)
        (subdir / "input.png").write_bytes(raw)
        with open(subdir / "pred.json", "w") as f:
            json.dump({"boxes": boxes, "labels": labels, "scores": scores}, f, indent=2)
        return {
            "preview": preview,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "artifact_dir": str(subdir),
        }

    # Real YOLO inference
    try:
        yolo = _load_yolo(model_id)
        result = yolo.predict(im, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        boxes, labels, scores = [], [], []
        if result.boxes is not None:
            for b in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append([x1, y1, x2, y2])
                cid = int(b.cls[0].item())
                labels.append(result.names.get(cid, str(cid)))
                scores.append(float(b.conf[0].item()))
        overlay = _draw_boxes(im.copy(), boxes, labels, scores)
        preview = _pil_to_b64(overlay)
        # save artifacts
        (subdir / "input.png").write_bytes(raw)
        (subdir / "preview.png").write_bytes(base64.b64decode(preview.split(",")[1]))
        with open(subdir / "pred.json", "w") as f:
            json.dump({"boxes": boxes, "labels": labels, "scores": scores}, f, indent=2)
        return {
            "preview": preview,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "artifact_dir": str(subdir),
        }
    except Exception as e:
        raise HTTPException(500, f"infer failed: {e}")


# Background job store for CV video
_CV_JOBS: Dict[str, Dict[str, Any]] = {}


def _video_worker(
    jid: str, model_id: str, src_path: pathlib.Path, conf: float, iou: float, imgsz: int
):
    job = _CV_JOBS[jid]
    try:
        out_dir = (CV_OUTPUTS / f"video_{jid}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not _HAS_YOLO:
            # stub: copy the uploaded video to out_dir
            dst = out_dir / src_path.name
            dst.write_bytes(src_path.read_bytes())
            job.update(status="done", out_dir=str(out_dir))
            return

        yolo = _load_yolo(model_id)
        # ultralytics will save renders into project/name
        yolo.predict(
            source=str(src_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=True,
            project=str(out_dir),
            name="pred",
            verbose=False,
        )
        job.update(status="done", out_dir=str(out_dir))
    except Exception as e:
        job.update(status="error", error=str(e))


@app.post("/cv/jobs/infer_video")
async def cv_infer_video(
    file: UploadFile = File(...),
    model_id: str = Form("yolov8n.pt"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tmp_dir = (CV_OUTPUTS / f"upload_{stamp}").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src = tmp_dir / file.filename
    src.write_bytes(await file.read())

    jid = str(uuid.uuid4())[:8]
    _CV_JOBS[jid] = {"status": "running", "out_dir": None}
    t = threading.Thread(
        target=_video_worker, args=(jid, model_id, src, conf, iou, imgsz), daemon=True
    )
    t.start()
    return {"job_id": jid}


@app.get("/cv/jobs/{jid}")
def cv_job_status(jid: str):
    job = _CV_JOBS.get(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/cv/artifacts")
def cv_artifacts():
    items = []
    if CV_OUTPUTS.exists():
        for p in sorted(CV_OUTPUTS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            items.append({"name": p.name, "path": str(p), "type": "dir" if p.is_dir() else "file"})
            if p.is_dir():
                prev = p / "preview.png"
                if prev.exists():
                    items.append(
                        {"name": f"{p.name}/preview.png", "path": str(prev), "type": "file"}
                    )
    return {"items": items}


@app.get("/cv/artifact")
def cv_artifact(path: str):
    p = pathlib.Path(path).resolve()
    # guard inside CV_OUTPUTS
    if not str(p).startswith(str(CV_OUTPUTS)):
        raise HTTPException(400, "outside cv outputs")
    if not p.exists():
        raise HTTPException(404, "not found")
    if p.is_dir():
        return {"items": [str(x) for x in p.iterdir()]}
    # small text or image
    if p.suffix.lower() in (".json", ".txt", ".log", ".csv"):
        return {"kind": "text", "content": p.read_text(errors="ignore")[:200_000]}
    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        b64 = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
        return {"kind": "image", "content": b64}
    return {"kind": "bin", "content": f"[binary: {p.name}]"}
