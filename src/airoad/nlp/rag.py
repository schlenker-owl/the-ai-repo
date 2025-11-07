from __future__ import annotations

from typing import List, Tuple

import numpy as np

from airoad.core.index_store import IndexCfg, IndexStore


def build_index(embeddings: np.ndarray, ids: List[str]) -> IndexStore:
    idx = IndexStore(dim=embeddings.shape[1], cfg=IndexCfg(index_type="auto"))
    idx.build(embeddings, ids)
    return idx


def retrieve(index: IndexStore, q: np.ndarray, topk: int = 5) -> Tuple[List[str], List[float]]:
    scores, indices = index.search(q, topk)
    idxs = indices[0].tolist()
    scrs = scores[0].tolist()
    paths = [index.paths[i] for i in idxs]
    return paths, scrs


def assemble_context(hit_texts: List[str], max_chars: int = 4000) -> str:
    buf: List[str] = []
    total = 0
    for t in hit_texts:
        if total + len(t) > max_chars:
            break
        buf.append(t)
        total += len(t)
    return "\n\n".join(buf)


def generate_answer(question: str, context: str) -> str:
    """
    Lightweight generator: try transformers pipeline; else context-aware template.
    """
    try:
        from transformers import pipeline  # type: ignore

        summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
        prompt = f"Use the provided context to answer the question.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        out = summarizer(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        return out.strip()
    except Exception:
        return f"Answer (heuristic): Based on context, {question}\n\nContext excerpt:\n{context[:600]}..."
