# Applied CV — **Semantic Visual Search** (Text↔Image & Image↔Image)

**Repo:** [`the-ai-repo`](https://github.com/schlenker-owl/the-ai-repo)

This guide documents the tools to **embed** your analysis artifacts (crops, annotated images, masks, or sampled frames), **build a vector index**, and **query** it by **text** (“red jersey”, “referee”, “goalpost”) or **image** (find similar).

The system is:

* **Local-first** (MPS/CUDA/CPU auto-pick)
* **Zero-dependency friendly** (OpenCLIP/FAISS optional; fallbacks provided)
* **Flexible inputs** (prefers `crops/`, falls back to annotated images `*_det.png`, `*_seg.png`, `*_pose.png`, `*_cls.png`, or `masks/`)
* **Portable** (index saved with image paths & config)

---

## Contents

* [1. Requirements](#1-requirements)
* [2. Install](#2-install)
* [3. What gets indexed](#3-what-gets-indexed)
* [4. Quick Start](#4-quick-start)
* [5. Queries](#5-queries)
* [6. Configuration](#6-configuration)
* [7. Outputs](#7-outputs)
* [8. Performance Tips](#8-performance-tips)
* [9. Troubleshooting](#9-troubleshooting)
* [10. Acceptance Checks](#10-acceptance-checks)
* [11. Code Map](#11-code-map)
* [12. Roadmap](#12-roadmap)

---

## 1. Requirements

* Python **3.11**
* [`uv`](https://docs.astral.sh/uv/)
* macOS (MPS), Linux, or Windows

Optional (recommended):

* **OpenCLIP** for **Text↔Image** search
* **FAISS** for fast indexing at scale

---

## 2. Install

```bash
# core CV deps (already used across the repo)
uv sync -g cv

# optional (text search + faster large-scale)
uv add --group cv "open-clip-torch>=2.24.0"
uv add --group cv faiss-cpu     # or faiss-gpu on CUDA hosts
uv sync -g cv
```

Without OpenCLIP you still get **Image↔Image** search. Without FAISS the indexer falls back to a fast NumPy cosine search (good for up to ~100k vectors on a laptop, depending on RAM).

---

## 3. What gets indexed

By default we look under `outputs/cv_images/` and index any of:

* **Crops** (if you saved them): `outputs/cv_images/*/crops/*/*.jpg`
* **Masks**: `outputs/cv_images/*/masks/*.png`
* **Annotated images**:

  * detection: `*_det.png`
  * segmentation: `*_seg.png`
  * pose: `*_pose.png`
  * classification: `*_cls.png`

> Tip — to save **crops** going forward, set in your image-analysis YAML:
> `save_crops: true` (and optionally `crop_max_size: 1024`)

You can also index **video frames** sampled from `outputs/cv/*/analysis.mp4` (optional script included).

---

## 4. Quick Start

### A) Index images (crops / masks / annotated)

```bash
uv run python scripts/cv/search/index_crops.py \
  --config configs/cv/search.yaml \
  --out outputs/cv_images/.index
```

This embeds images and writes a reusable index into `outputs/cv_images/.index`.

### B) (Optional) Index sampled **video frames**

```bash
uv run python scripts/cv/search/index_frames.py \
  --root outputs/cv \
  --stride 30 \
  --max-per-video 200 \
  --config configs/cv/search.yaml \
  --out outputs/cv/.index_frames
```

---

## 5. Queries

### Text → Image  *(requires OpenCLIP)*

```bash
uv run python scripts/cv/search/query_search.py \
  --index outputs/cv_images/.index \
  --config configs/cv/search.yaml \
  --text "football player wearing red jersey" \
  --topk 20 \
  --copy-to tmp/search_red_jersey   # optional: copy top-K to inspect
```

### Image → Image

```bash
uv run python scripts/cv/search/query_search.py \
  --index outputs/cv_images/.index \
  --config configs/cv/search.yaml \
  --image outputs/cv_images/football/football_det.png \
  --topk 20
```

The script prints ranked JSON results and (optionally) copies top-K to a folder for quick eyeballing.

---

## 6. Configuration

**File:** `configs/cv/search.yaml`

```yaml
backbone:
  name: "open-clip:ViT-B-32"   # "auto" | "open-clip:ViT-B-32" | "resnet50"
  device: null                 # "mps" | "0" | "cpu"
  dtype: "float32"
  image_size: 224

embed:
  batch_size: 128

index:
  type: "auto"                 # "auto" | "faiss_flat" | "ivfpq" | "numpy"
  nlist: 1024                  # IVF clusters (ivfpq)
  m: 64                        # PQ sub-vectors
  nprobe: 16                   # IVF probes

inputs:
  roots:
    - "outputs/cv_images"
  patterns:
    - "**/crops/*/*.jpg"
    - "**/crops/*/*.jpeg"
    - "**/crops/*/*.png"
    - "**/crops/*/*.webp"
    - "**/masks/*.png"
    - "**/*_det.png"
    - "**/*_seg.png"
    - "**/*_pose.png"
    - "**/*_cls.png"
```

* Set `backbone.name: "resnet50"` for image-only search (no text).
* For large corpora, switch `index.type: "ivfpq"` (requires FAISS).
* Override roots/patterns as needed:

  ```bash
  uv run python scripts/cv/search/index_crops.py \
    --config configs/cv/search.yaml \
    --roots outputs/cv_images outputs/cv
  ```

---

## 7. Outputs

An index directory contains:

```
.index/
  meta.json        # paths, dim, backend, config
  index.faiss      # if FAISS used
  embeddings.npy   # if NumPy fallback used
```

Example query output (printed to console):

```json
[
  {"rank": 1, "path": "outputs/cv_images/football/football_seg.png", "score": 0.76},
  {"rank": 2, "path": "outputs/cv_images/tree3/masks/mask_014.png", "score": 0.73},
  ...
]
```

If you provided `--copy-to`, top-K files are copied there for quick review.

---

## 8. Performance Tips

* **OpenCLIP** + **FAISS/IVFPQ** → best combo for large & fast search (millions of items).
* On Apple Silicon, `device: "mps"` is great for embedding throughput.
* Use **crops** for object-level semantics, **annotated images** or **frames** for scene-level semantics—index both if helpful.
* Batch size (`embed.batch_size`) controls embed throughput vs. memory.

---

## 9. Troubleshooting

**“No images found for indexing.”**

* You have no `crops/`—that’s fine. Ensure your `inputs.patterns` include `*_det.png`, `*_seg.png`, `masks/*.png`.
* Verify your root (`outputs/cv_images`) actually contains those files.

**Text queries error:**

* Requires OpenCLIP:

  ```bash
  uv add --group cv "open-clip-torch>=2.24.0"
  uv sync -g cv
  ```

  And set `backbone.name: "open-clip:ViT-B-32"`.

**Index too slow / large:**

* Install FAISS and set `index.type: "ivfpq"`. Tune `nlist/m/nprobe`.

**Results look “off”:**

* Try more descriptive prompts (“a football player wearing red jersey and shorts”, “green grass field”).
* For image queries, use a clean exemplar (a crop rather than a busy frame).

---

## 10. Acceptance Checks

* [ ] Index builds without errors; `.index/meta.json` exists.
* [ ] Text queries return sensible matches (with OpenCLIP).
* [ ] Image queries return visually similar items.
* [ ] `--copy-to` folder contains the top-K files and appears qualitatively correct.
* [ ] FAISS runs if installed; NumPy fallback works otherwise.

---

## 11. Code Map

* **Backbone (OpenCLIP/ResNet50):**
  `src/airoad/search/backbone.py`

* **Indexer (FAISS or NumPy fallback):**
  `src/airoad/search/indexer.py`

* **Index builders:**
  `scripts/cv/search/index_crops.py` (crops / masks / annotated)
  `scripts/cv/search/index_frames.py` (sampled frames; optional)

* **Query CLI:**
  `scripts/cv/search/query_search.py`

* **Config:**
  `configs/cv/search.yaml`

---

## 12. Roadmap

* **HTML gallery** for query results (grid preview with paths/scores).
* **Gradio UI** to type prompts or drag-drop images and browse results.
* **Filters** (by class, source folder, time ranges) using your existing metadata.
* **Relevance feedback** (click to re-rank or refine queries).
* **RAG over vision+text** (attach captions/attributes to vectors and rerank).
