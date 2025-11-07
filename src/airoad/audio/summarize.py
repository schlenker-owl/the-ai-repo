from __future__ import annotations


def summarize_transcript(text: str) -> str:
    """
    Try a local summarizer; fallback to heuristic bullets.
    """
    try:
        from transformers import pipeline  # type: ignore

        summ = pipeline("summarization", model="facebook/bart-large-cnn")
        out = summ(text[:4000], max_length=180, min_length=60, do_sample=False)
        return out[0]["summary_text"].strip()
    except Exception:
        # heuristic: pick top-N long sentences
        sents = [s.strip() for s in text.split("\n") if len(s.strip()) > 8]
        bullets = sents[:8] if sents else ["(no content)"]
        return "- " + "\n- ".join(bullets)
