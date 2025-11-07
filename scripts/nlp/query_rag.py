#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from airoad.core.index_store import IndexStore
from airoad.nlp.embed import TextEmbedder
from airoad.nlp.rag import assemble_context, generate_answer, retrieve


def _load_chunks_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        out[obj["chunk_id"]] = obj["text"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--embed-model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    idx = IndexStore.load(Path(args.index_dir))
    chunks = _load_chunks_jsonl(Path(args.index_dir) / "chunks.jsonl")

    emb = TextEmbedder(model_name=args.embed_model, device=args.device)
    emb.fit([args.question])  # no-op for sbert; creates vocab for tfidf fallback
    q = emb.encode([args.question])

    paths, scores = retrieve(idx, q, topk=args.topk)
    texts = [chunks.get(p, "") for p in paths]
    context = assemble_context(texts)
    answer = generate_answer(args.question, context)

    print(
        json.dumps(
            {
                "question": args.question,
                "topk": args.topk,
                "hits": [{"chunk_id": p, "score": float(s)} for p, s in zip(paths, scores)],
                "answer": answer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
