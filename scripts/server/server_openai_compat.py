"""
Minimal OpenAI-compatible /v1/chat/completions server.

Run:
  uv run python scripts/serve_openai_compat.py --host 0.0.0.0 --port 8000

Dependencies:
  pip install fastapi uvicorn

Notes:
  - Uses GPTTiny from your repo if importable; else echoes last user content.
  - Not covered by tests; meant as a dev-time smoke server.
"""
import argparse
from typing import List, Dict, Any

def build_app():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except Exception as e:
        raise RuntimeError("FastAPI required: pip install fastapi uvicorn") from e

    # lazy import to avoid test-time failures
    try:
        import torch
        from airoad.transformers.gpt_tiny import GPTTiny
        from airoad.dl.char_data import build_vocab_from_text
        _HAS_MODEL = True
    except Exception:
        _HAS_MODEL = False

    app = FastAPI(title="OpenAI-Compatible Chat API (Tiny)")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: float | None = 0.7
        max_tokens: int | None = 64

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        user_msgs = [m.content for m in req.messages if m.role == "user"]
        last = user_msgs[-1] if user_msgs else ""

        if not _HAS_MODEL:
            # echo fallback
            return {
                "id": "chatcmpl-fallback",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"(echo) {last}"}, "finish_reason": "stop"}],
                "model": req.model,
            }

        # Tiny char model generation (toy)
        dev = "cpu"
        vocab = build_vocab_from_text(last if last else "hello world")
        # map characters to IDs
        ids = [vocab.stoi.get(ch, 0) for ch in (last[:32] or "h")]
        import torch
        x = torch.tensor([ids], dtype=torch.long, device=dev)

        model = GPTTiny(vocab_size=vocab.size, block_size=64, n_layer=1, n_head=2, n_embd=32).to(dev)
        out_ids = model.generate(x, max_new_tokens=min(50, req.max_tokens or 50)).squeeze(0).tolist()
        inv = {i: ch for i, ch in enumerate(vocab.itos)}
        text = "".join(inv.get(i, "?") for i in out_ids)

        return {
            "id": "chatcmpl-tiny",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "model": req.model,
        }

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = build_app()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
