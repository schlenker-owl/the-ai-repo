#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from airoad.core.io import ensure_dir
from airoad.core.manifest import write_manifest
from airoad.nlp.chunk import chunk_text
from airoad.nlp.embed import TextEmbedder
from airoad.nlp.ingest import ingest_dir
from airoad.nlp.rag import build_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/nlp/rag.yaml")
    ap.add_argument("--out", type=str, default="outputs/nlp/.index_docs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    roots = cfg.get("inputs", {}).get("roots", ["data/text"])
    max_tokens = int(cfg.get("chunk", {}).get("max_tokens", 350))
    overlap = int(cfg.get("chunk", {}).get("overlap_tokens", 50))
    model_name = cfg.get("embed", {}).get("model", None)
    device = cfg.get("embed", {}).get("device", None)

    # Ingest + chunk
    docs = []
    for r in roots:
        docs.extend(ingest_dir(r))
    chunks = []
    chunk_ids = []
    texts = []
    for d in docs:
        for ch in chunk_text(d.doc_id, d.text, max_tokens=max_tokens, overlap_tokens=overlap):
            chunks.append(ch)
            chunk_ids.append(ch.chunk_id)
            texts.append(ch.text)

    # Embed
    emb = TextEmbedder(model_name=model_name, device=device)
    emb.fit(texts)
    vecs = emb.encode(texts)

    # Build index
    idx = build_index(vecs, ids=chunk_ids)
    out_dir = ensure_dir(args.out)
    # Save index files (core index store writes meta + vectors)
    idx.save(out_dir)

    # Save mapping chunk_id -> text/jsonl
    mapping = [{"chunk_id": cid, "text": t} for cid, t in zip(chunk_ids, texts)]
    (out_dir / "chunks.jsonl").write_text("\n".join(json.dumps(x) for x in mapping))

    write_manifest(
        out_dir=out_dir,
        name="nlp/index_docs",
        version="0.1.0",
        config=str(Path(args.config).resolve()),
        inputs=[str(Path(r).resolve()) for r in roots],
        outputs={"index_dir": str(out_dir)},
        seed=None,
        device=device or "auto",
    )
    print(f"[nlp/index_docs] Indexed {len(texts)} chunks to {out_dir}")


if __name__ == "__main__":
    main()
