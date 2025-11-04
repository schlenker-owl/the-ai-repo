from __future__ import annotations

from typing import Dict, List, Tuple


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip().lower() == ref.strip().lower())


def _cosine_sim(a, b) -> float:
    import numpy as np

    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(a.dot(b) / (na * nb))


def embed_sentences(texts: List[str]) -> Tuple[object, list]:
    """
    Try SentenceTransformers, else TF-IDF fallback. Returns (backend, embeddings/model)
    - backend == 'st' -> list[np.ndarray]
    - backend == 'tfidf' -> (vectorizer, sparse_matrix)
    """
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = m.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return ("st", (m, vecs))
    except Exception:
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer

        V = TfidfVectorizer(stop_words="english")
        X = V.fit_transform(texts)
        return ("tfidf", (V, X))


def retrieve_topk(query: str, corpus: List[str], k: int = 3) -> List[int]:
    backend, obj = embed_sentences(corpus + [query])
    if backend == "st":
        import numpy as np

        _, vecs = obj
        cor = np.array(vecs[:-1])
        qv = vecs[-1]
        sims = cor @ qv  # normalized embeddings
        return list(sims.argsort()[::-1][:k])
    else:
        # tfidf
        V, X = obj
        qX = V.transform([query])
        sims = (X[: len(corpus)] @ qX.T).toarray().ravel()
        return list(sims.argsort()[::-1][:k])


def cosine_answer_sim(pred: str, ref: str) -> float:
    backend, obj = embed_sentences([pred, ref])
    if backend == "st":
        _, vecs = obj
        return _cosine_sim(vecs[0], vecs[1])
    else:
        V, X = obj
        # project both through same vectorizer; pack into tiny 2-row matrix
        # (already done by embed_sentences)

        v0 = X[0].toarray().ravel()
        v1 = X[1].toarray().ravel()
        return _cosine_sim(v0, v1)


def evaluate_qa(items: List[Dict[str, str]]) -> Dict[str, float]:
    """
    items: [{question, predicted, reference}]
    returns metrics: exact_match, cosine_sim (mean)
    """
    ems, coss = [], []
    for it in items:
        ems.append(exact_match(it["predicted"], it["reference"]))
        coss.append(cosine_answer_sim(it["predicted"], it["reference"]))
    import numpy as np

    return {"exact_match": float(np.mean(ems)), "cosine_sim": float(np.mean(coss))}
