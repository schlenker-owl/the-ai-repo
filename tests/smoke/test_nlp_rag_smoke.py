from __future__ import annotations

import json
from pathlib import Path

from airoad.core.index_store import IndexCfg, IndexStore
from airoad.nlp.chunk import chunk_text
from airoad.nlp.embed import TextEmbedder
from airoad.nlp.ingest import ingest_dir
from airoad.nlp.rag import assemble_context, generate_answer, retrieve


def test_nlp_rag_end_to_end(tmp_path: Path):
    # Prepare tiny corpus
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "sky.txt").write_text(
        "The sky is blue on clear days. " "Sometimes the sky looks orange at sunset."
    )
    (docs_dir / "ocean.txt").write_text(
        "The ocean can appear blue or green, depending on depth and light."
    )

    # Ingest + chunk
    docs = ingest_dir(docs_dir)
    all_chunks = []
    chunk_ids = []
    texts = []
    for d in docs:
        chunks = chunk_text(d.doc_id, d.text, max_tokens=30, overlap_tokens=5)
        for ch in chunks:
            all_chunks.append(ch)
            chunk_ids.append(ch.chunk_id)
            texts.append(ch.text)

    assert len(texts) > 0, "no chunks created"

    # Embed (SBERT if present; otherwise TF-IDF fallback)
    embedder = TextEmbedder(model_name=None, device=None)
    embedder.fit(texts)
    X = embedder.encode(texts)
    assert X.shape[0] == len(texts)
    assert X.shape[1] == embedder.dim

    # Build in-memory index
    idx = IndexStore(dim=X.shape[1], cfg=IndexCfg(index_type="auto"))
    idx.build(X, chunk_ids)

    # Query
    question = "What color is the sky?"
    q = embedder.encode([question])
    paths, scores = retrieve(idx, q, topk=3)
    assert len(paths) > 0

    # Expect a chunk from 'sky' among top hits
    assert any("sky" in p for p in paths), f"expected 'sky' chunk in {paths}"

    # Assemble + generate (template or T5 if available)
    mapping = {cid: t for cid, t in zip(chunk_ids, texts)}
    ctx = assemble_context([mapping[c] for c in paths])
    answer = generate_answer(question, ctx)
    assert isinstance(answer, str) and len(answer.strip()) > 0

    # Basic JSON shape sanity
    payload = {
        "question": question,
        "hits": [{"chunk_id": p, "score": float(s)} for p, s in zip(paths, scores)],
        "answer": answer,
    }
    json.dumps(payload)  # ensure serializable
