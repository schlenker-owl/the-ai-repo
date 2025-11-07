from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class Doc:
    doc_id: str
    text: str
    meta: Dict[str, str]


def _read_text_file(p: Path) -> str:
    return p.read_text(errors="ignore")


def _read_pdf_file(p: Path) -> str:
    try:
        import pypdf  # type: ignore
    except Exception as e:
        raise RuntimeError("pypdf not installed. Install with: uv add --group nlp pypdf") from e
    reader = pypdf.PdfReader(str(p))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_html_file(p: Path) -> str:
    raw = p.read_text(errors="ignore")
    # very light tag strip
    txt = re.sub(r"<script.*?>.*?</script>", " ", raw, flags=re.S | re.I)
    txt = re.sub(r"<style.*?>.*?</style>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    return html.unescape(txt)


def _read_any(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in {".txt", ".md"}:
        return _read_text_file(p)
    if ext in {".html", ".htm"}:
        return _read_html_file(p)
    if ext == ".pdf":
        return _read_pdf_file(p)
    return _read_text_file(p)


def ingest_dir(root: str | Path, exts: Optional[Iterable[str]] = None) -> List[Doc]:
    root = Path(root)
    if exts is None:
        exts = [".txt", ".md", ".html", ".htm", ".pdf"]
    out: List[Doc] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in set(exts):
            continue
        txt = _read_any(p)
        if not txt.strip():
            continue
        out.append(Doc(doc_id=p.stem, text=txt, meta={"path": str(p.resolve())}))
    return out
