# Applied CV — **Active Learning + Fine-Tune** Loop

**Repo:** [`the-ai-repo`](https://github.com/schlenker-owl/the-ai-repo)

This guide documents the end-to-end loop to **auto-label → review (optional) → fine-tune → evaluate → reuse model to relabel better** for both **images** and **videos**. It’s designed to be:

* **Local-first & fast** (CPU/MPS/GPU; one command per step)
* **Reproducible** (YAML configs, deterministic splits)
* **Practical** (handles tiny datasets, empty splits, class-ID mismatches)
* **Task-flexible** (DET or SEG; SEG can fall back to **rectangle polygons** if masks are missing)

---

## Contents

* [1. Requirements](#1-requirements)
* [2. Install](#2-install)
* [3. Dataset layout](#3-dataset-layout)
* [4. Quick Start — Images](#4-quick-start--images)
* [5. Quick Start — Videos → Frames](#5-quick-start--videos--frames)
* [6. Train](#6-train)
* [7. Evaluate](#7-evaluate)
* [8. Iterate (Active Learning v2+)](#8-iterate-active-learning-v2)
* [9. Tuning & Tips](#9-tuning--tips)
* [10. Troubleshooting](#10-troubleshooting)
* [11. Acceptance Checks](#11-acceptance-checks)
* [12. Roadmap / Extensions](#12-roadmap--extensions)
* [13. CLI Reference](#13-cli-reference)

---

## 1. Requirements

* Python **3.11**
* [`uv`](https://docs.astral.sh/uv/)
* macOS (MPS) / Linux / Windows
* Optional: NVIDIA GPU (CUDA) or Apple Silicon M-series (MPS)

---

## 2. Install

CV dependencies live in the `cv` group.

```bash
# from repo root
uv sync -g cv
```

Installs: `ultralytics`, `opencv-python`, `pandas`, `pyyaml`, `matplotlib` (headless), etc.

---

## 3. Dataset layout

All auto-labelers write a YOLO-style dataset:

```
datasets/autolabel/<dataset_name>/
  data.yaml
  images/
    train/  *.jpg|*.png
    val/
    test/
  labels/
    train/  *.txt          # YOLO DET or SEG lines
    val/
    test/
  exports/
    coco.json              # COCO-style export (local class IDs)
    label_studio.jsonl     # lightweight task index for review tools
```

**DET label line:**
`<class> <cx> <cy> <w> <h>` (all normalized 0..1)

**SEG label line:**
`<class> <x1> <y1> <x2> <y2> ...` (polygon normalized 0..1)

**Class IDs are local** (0..K-1). We compact model’s native IDs to a contiguous local set and write the same order into `data.yaml["names"]` so training never errors with “label class X exceeds nc”.

**Guardrails:**

* We **pre-create** all six split dirs (even if empty) so Ultralytics can resolve paths.
* If **train** or **val** ends empty, we **duplicate** 1 image (+ label) from another split to unblock training.

---

## 4. Quick Start — Images

### A) Detection (recommended to bootstrap)

```bash
uv run python scripts/cv/al/auto_label_images.py \
  --source-dir data/images \
  --output-root datasets/autolabel \
  --dataset-name my_det_ds \
  --model yolo11s.pt \
  --task det \
  --conf 0.15 \
  --copy-mode link
```

* Use `--classes 0,2,3,5,7` to restrict to person/vehicles (optional).
* `--conf` can be lowered to 0.10–0.20 for more pseudo-labels on small sets.

### B) Segmentation

If the seg model doesn’t output masks on your data yet, enable **rectangle polygons from boxes** to produce non-empty seg labels:

```bash
uv run python scripts/cv/al/auto_label_images.py \
  --source-dir data/images \
  --output-root datasets/autolabel \
  --dataset-name my_seg_ds \
  --model yolo11s-seg.pt \
  --task seg \
  --conf 0.15 \
  --seg-from-boxes \
  --copy-mode link
```

The script prints the split counts and class names at the end.

---

## 5. Quick Start — Videos → Frames

Sample frames (scene-aware) then auto-label **DET** or **SEG**:

```bash
uv run python scripts/cv/al/auto_label_videos.py \
  --source-dir data/videos \
  --output-root datasets/autolabel \
  --dataset-name my_video_ds \
  --model yolo11s.pt \
  --task det \
  --frame-stride 15 \
  --scene-threshold 40.0 \
  --max-frames-per-video 300
```

* Lower `--scene-threshold` or increase `--frame-stride` to change sampling density.
* Swap model/task for segmentation: `--model yolo11s-seg.pt --task seg`.

---

## 6. Train

Set your dataset in `configs/cv/train.yaml`:

```yaml
# configs/cv/train.yaml
data:
  root: "datasets/autolabel/my_seg_ds"   # or explicit: yaml: ".../data.yaml"

train:
  model: "yolo11s-seg.pt"   # or "yolo11s.pt" for DET datasets
  imgsz: 640
  epochs: 20
  batch: 8
  device: "mps"             # "0" for first CUDA GPU, "mps" for Apple Silicon
  seed: 42
  project: "runs/train"
  name: "exp"
  exist_ok: true
```

Run:

```bash
uv run python scripts/cv/train/train_yolo.py --config configs/cv/train.yaml
```

The trainer preflights your dataset, ensures split dirs exist, and (if needed) duplicates one sample to avoid empty train/val splits.

---

## 7. Evaluate

```bash
uv run python scripts/cv/train/eval_yolo.py \
  --model runs/train/exp/weights/best.pt \
  --data  datasets/autolabel/my_seg_ds/data.yaml \
  --split val
```

You’ll get `eval_val.json` next to the weights with mAP and related metrics. Keep in mind tiny splits (e.g., 1–2 images) will yield unstable metrics—use at least ~10 images in `val` for sane numbers.

---

## 8. Iterate (Active Learning v2)

1. Use the **fine-tuned model** to auto-label **new** images (or re-label old ones) at a stricter `--conf`:

```bash
uv run python scripts/cv/al/auto_label_images.py \
  --source-dir data/images_new \
  --output-root datasets/autolabel \
  --dataset-name my_seg_ds_v2 \
  --model runs/train/exp/weights/best.pt \
  --task seg \
  --conf 0.20 \
  --seg-from-boxes
```

2. Train again on v2 and compare metrics.
3. (Optional) Review COCO/Label-Studio exports to curate a “gold” core set for the next round.

---

## 9. Tuning & Tips

* **Model choice**

  * Start with **DET** (`yolo11s.pt`) to bootstrap labels quickly; switch to **SEG** when masks start appearing or use `--seg-from-boxes`.
* **Thresholds**

  * `--conf`: lower (0.10–0.20) for more pseudo-labels; raise (0.30–0.50) for precision.
  * `--classes`: filter to relevant classes to reduce junk labels.
* **Splits**

  * For small N, use `split: "0.9,0.1,0.0"` initially, or rely on our min-split guard.
* **Performance**

  * MPS is great on Apple Silicon. On CUDA, set `train.device: "0"` and consider `workers: 2`.
* **Quality**

  * Rectangle seg fallback is a bootstrap—replace later with true polygons/masks from a better model or human review.

---

## 10. Troubleshooting

* **“No images found in …/images/val”**
  We pre-create split directories; if val ends empty, we **duplicate** one sample so training starts. Confirm with:

  ```bash
  tree -L 2 datasets/autolabel/my_seg_ds/images
  ```

* **“Label class X exceeds dataset class count Y”**
  We compact to **local IDs** automatically. If you edited labels manually, ensure they still use 0..nc-1 and `data.yaml["names"]` matches.

* **“Labels missing or empty”**
  Means your model didn’t predict at that threshold. Lower `--conf`, or switch to DET for a round.

* **SEG trainer can’t compute metrics**
  Happens if no seg labels in `val`. Use `--seg-from-boxes` or add a few clearer images.

* **Slow or stalls on Mac**
  Ultralytics often defaults to `workers=0` on MPS—fine for tiny sets. If needed, set `ultralytics settings` → `dataset` or add workers via `ultralytics.yaml`.

---

## 11. Acceptance Checks

* [ ] Auto-label produced **non-empty** `.txt` labels in train/val.
* [ ] `data.yaml` references the dataset root and lists **local** class `names` (0..K-1).
* [ ] Training completes and metrics JSON is produced.
* [ ] Re-running auto-label with the **fine-tuned model** improves label quality or coverage.
* [ ] (Optional) Human review loop established for a small “gold” set.

---

## 12. Roadmap / Extensions

* **Human-in-the-loop review**

  * Wire `exports/label_studio.jsonl` into Label Studio for point-and-click fixes; write a small importer back to YOLO txt.
* **Active data selection**

  * Rank frames by prediction uncertainty or diversity; only pull the top-K for review.
* **Model cards & registry**

  * Auto-generate `model_card.md` per run with training args, data snapshot, and metrics.
* **Data versioning**

  * Add DVC or git-LFS pointers for `datasets/autolabel/…`.
* **Export**

  * ONNX/TensorRT for serving.

---

## 13. CLI Reference

### `scripts/cv/al/auto_label_images.py`

* **required:** `--source-dir`, `--dataset-name`
* **common:** `--model`, `--task [auto|det|seg]`, `--conf`, `--classes 0,2,3`, `--seg-from-boxes`, `--copy-mode [link|copy|hardlink]`

Example:

```bash
uv run python scripts/cv/al/auto_label_images.py \
  --source-dir data/images \
  --output-root datasets/autolabel \
  --dataset-name my_seg_ds \
  --model yolo11s-seg.pt \
  --task seg \
  --conf 0.15 \
  --seg-from-boxes
```

### `scripts/cv/al/auto_label_videos.py`

* **required:** `--source-dir`, `--dataset-name`
* **common:** `--model`, `--task`, `--frame-stride`, `--scene-threshold`, `--max-frames-per-video`

### `scripts/cv/train/train_yolo.py`

* **required:** `--config configs/cv/train.yaml`
* Looks for `data.yaml` via `data.yaml` or `data.root`. Preflights dataset; ensures min splits.

### `scripts/cv/train/eval_yolo.py`

```bash
uv run python scripts/cv/train/eval_yolo.py \
  --model runs/train/exp/weights/best.pt \
  --data  datasets/autolabel/my_seg_ds/data.yaml \
  --split val
```
