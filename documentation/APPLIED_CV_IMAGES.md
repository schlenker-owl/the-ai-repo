# Applied Computer Vision — **Image** Analytics with Ultralytics YOLO

**Repo:** [`the-ai-repo`](https://github.com/schlenker-owl/the-ai-repo)

This guide documents the **image-centric** CV tools in this repo for running **object detection**, **instance segmentation**, **pose estimation**, and **image classification** on single images *or* entire folders. The pipeline produces **annotated images**, **JSON exports** (boxes / polygons / keypoints / top-k), optional **instance mask PNGs** and **crops**, plus **batch rollups**.

The system is:

* **Local-first & fast** (CPU/MPS/GPU; single command)
* **Reproducible** (YAML config, consistent outputs)
* **Modular** (analysis → per-image artifacts → batch rollups)
* **Auto-adaptive** to the **model type** you load (seg/pose/det/cls)

---

## Contents

* [1. Requirements](#1-requirements)
* [2. Install](#2-install)
* [3. Quick Start (single image)](#3-quick-start-single-image)
* [4. Batch Mode (folder of images)](#4-batch-mode-folder-of-images)
* [5. Pick a Mode & Config (det / seg / pose / cls)](#5-pick-a-mode--config-det--seg--pose--cls)
* [6. What You Get (outputs)](#6-what-you-get-outputs)
* [7. Configuration (YAML reference)](#7-configuration-yaml-reference)
* [8. JSON & CSV Schemas](#8-json--csv-schemas)
* [9. Performance Tips](#9-performance-tips)
* [10. Troubleshooting](#10-troubleshooting)
* [11. Roadmap / Extensions](#11-roadmap--extensions)

---

## 1. Requirements

* Python **3.11**
* [`uv`](https://docs.astral.sh/uv/) (fast package manager)
* macOS (Intel/Apple Silicon), Linux, or Windows
* Optional: NVIDIA GPU (CUDA) or Apple Silicon **MPS** for acceleration

---

## 2. Install

We keep CV deps in a dedicated extra:

```bash
# from repo root
uv sync -g cv
```

This installs: `ultralytics` (YOLOv8/YOLO11), `opencv-python`, `pandas`, `pyyaml`, `matplotlib` (headless).

---

## 3. Quick Start (single image)

1. Put an image under `data/images/`, e.g. `data/images/example.jpg`.

2. Choose a config (see [Section 5](#5-pick-a-mode--config-det--seg--pose--cls)). For a quick detection run, use:

```yaml
# configs/cv/image_analysis_det.yaml
source: "data/images/example.jpg"
out_dir: "outputs/cv_images/example"

model: "yolo11n.pt"     # detection
conf: 0.35
save_annotated: true
save_json: true
save_crops: true
```

3. Run:

```bash
uv run python scripts/cv/run_image_analysis.py --config configs/cv/image_analysis_det.yaml
```

**Outputs** appear under `outputs/cv_images/example/` (see [Section 6](#6-what-you-get-outputs)).

---

## 4. Batch Mode (folder of images)

Use `source_dir` instead of `source`:

```yaml
# configs/cv/image_analysis_seg.yaml (segmentation)
source_dir: "data/images"
source_glob: ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"]
recursive: false
output_root: "outputs/cv_images"

model: "yolo11n-seg.pt"
save_annotated: true
render_style: "masks_only"  # clean mask overlay (no boxes)
save_masks: true
```

Run:

```bash
uv run python scripts/cv/run_image_analysis.py --config configs/cv/image_analysis_seg.yaml
```

A subfolder is created **per image** under `outputs/cv_images/<image-stem>/`, plus batch rollups:
`outputs/cv_images/_detections.csv`, `outputs/cv_images/_batch_summary.json`.

---

## 5. Pick a Mode & Config (det / seg / pose / cls)

We ship four ready-to-use YAMLs in `configs/cv/`:

### A) Detection — `image_analysis_det.yaml`

* **Model:** `yolo11n.pt` (try `yolo11s.pt` for better accuracy)
* Outputs boxes, labels, optional **crops** per detection
* Good for quick dataset triage

```yaml
model: "yolo11n.pt"
conf: 0.3
save_annotated: true
render_style: "ultra_default"   # Ultralytics default boxes+labels
save_crops: true
save_masks: false
```

### B) Segmentation — `image_analysis_seg.yaml`

* **Model:** `yolo11*-seg.pt`
* Writes **instance masks** (PNG, optional), **mask-only overlay** (clean), and **COCO-style polygons** in JSON

```yaml
model: "yolo11n-seg.pt"
render_style: "masks_only"       # clean colored masks (no boxes)
save_masks: true                  # write per-instance mask PNGs
mask_alpha: 0.45
```

### C) Pose — `image_analysis_pose.yaml`

* **Model:** `yolo11*-pose.pt`
* Writes keypoints for each detected person/object; annotated image includes skeletons

```yaml
model: "yolo11n-pose.pt"
save_annotated: true
save_json: true
```

### D) Classification — `image_analysis_cls.yaml`

* **Model:** `yolo11*-cls.pt` or another Ultralytics-style classifier
* Exports **top-k** class predictions to JSON + preview image

```yaml
model: "yolo11n-cls.pt"
imgsz: 224
topk: 5
```

> The pipeline auto-detects which task to run by inspecting the model’s result:
> `segmentation` if `r.masks`, `pose` if `r.keypoints`, `classification` if `r.probs`, otherwise `detection`.
> Some builds may expose **OBB** (`r.obb`) — we export if present.

---

## 6. What You Get (outputs)

For single image: `outputs/cv_images/<image-stem>/` contains:

```
<stem>_det.png | _seg.png | _pose.png | _cls.png     # annotated image
<stem>_detections.json | _seg.json | _pose.json | _cls.json
masks/<stem>_###.png                   # seg only (if save_masks=true)
crops/<class>/<stem>_###.jpg           # when save_crops=true
_detections.csv                        # per-image CSV (single-image mode)
_summary.json                          # per-image summary (single-image mode)
```

For batch runs (with `source_dir`):

```
outputs/cv_images/
  <stemA>/
  <stemB>/
  ...
  _detections.csv          # all detections/instances across folder
  _batch_summary.json      # per-image outputs and counts
```

---

## 7. Configuration (YAML reference)

Key fields (all modes):

```yaml
# Single-file OR folder
# source: "data/images/example.jpg"
# out_dir: "outputs/cv_images/example"
# OR
# source_dir: "data/images"
# source_glob: ["*.jpg","*.png"]  # string or list
recursive: false
output_root: "outputs/cv_images"

# Model / inference
model: "yolo11n.pt"     # yolo11n-seg.pt for seg, yolo11n-pose.pt for pose, yolo11n-cls.pt for cls
conf: 0.3               # raise for stricter results (det/seg/pose)
iou: 0.5
imgsz: 640
classes: null           # e.g., [0,2,3,5,7] to restrict to person/vehicles
device: null            # "cpu", "mps", "0", ...

# Rendering / saving
save_annotated: true
render_style: "auto"    # "auto" chooses masks overlay when masks exist; or "masks_only" / "ultra_default"
mask_alpha: 0.45
draw_labels: true
draw_conf: true
save_json: true
save_crops: false
save_masks: true        # only meaningful for seg models
crop_max_size: 1024

# Classification (cls models)
topk: 5
```

---

## 8. JSON & CSV Schemas

### Segmentation JSON — `<stem>_seg.json`

```json
{
  "image": "data/images/example.jpg",
  "width": 1920,
  "height": 1080,
  "instances": [
    {
      "class_id": 0,
      "class_name": "person",
      "conf": 0.92,
      "bbox": [x1, y1, x2, y2],
      "segmentation": [[x,y, x,y, ...]],   // COCO-style polygons
      "mask_path": "masks/example_000.png"
    }
  ]
}
```

### Pose JSON — `<stem>_pose.json`

```json
{
  "image": "data/images/example.jpg",
  "width": 1920,
  "height": 1080,
  "instances": [
    {
      "class_id": 0, "class_name": "person", "conf": 0.91,
      "bbox": [x1, y1, x2, y2],
      "keypoints": [[x,y], [x,y], ...]
    }
  ]
}
```

### Detection JSON — `<stem>_detections.json>`

Same as seg but without `segmentation`/`mask_path`.

### Classification JSON — `<stem>_cls.json>`

```json
{
  "image": "data/images/example.jpg",
  "width": 224,
  "height": 224,
  "topk": [
    {"class_id": 834, "class_name": "goldfinch", "score": 0.73},
    {"class_id": 12,  "class_name": "bee_eater", "score": 0.12}
  ]
}
```

### Batch CSV — `_detections.csv`

One row per detected instance or class prediction (for cls). Columns:

| column        | type    | notes                                       |
| ------------- | ------- | ------------------------------------------- |
| image         | str     | image path                                  |
| width,height  | int     | image size                                  |
| class_id/name | int,str | model class outputs                         |
| conf          | float   | confidence (det/seg/pose) or prob (cls)     |
| x1,y1,x2,y2   | float?  | box coords (det/seg/pose/obb); null for cls |
| area          | float?  | box area (det/seg/pose)                     |
| crop_path     | str?    | saved crop when `save_crops=true`           |
| mask_path     | str?    | seg only (if `save_masks=true`)             |
| segmentation  | json?   | seg polygons (list of lists)                |
| keypoints     | json?   | pose keypoints (list of [x,y])              |

---

## 9. Performance Tips

* Use `yolo11s` (or larger) for better accuracy; `n` is fastest.
* Restrict `classes:` to reduce false positives and speed up postprocessing.
* `imgsz: 512` or `640` works well; lower for speed, higher for detail.
* Set `device: "mps"` (Apple Silicon) or `"0"` (CUDA) for big speedups.
* For massive folders: run with `source_glob: "*.jpg"` first, then refine by subfolders or class filters.

---

## 10. Troubleshooting

**No masks appear**

* Ensure you’re using a `*-seg.pt` model and `save_annotated: true`.
* If you want only mask overlays (no boxes), set `render_style: "masks_only"`.

**Classification error about `Probs.topk`**

* Fixed in our pipeline. We use `top5/top5conf` if present; otherwise we sort `probs.data`.

**Empty or missing outputs**

* Check file permissions and that `output_root` / `out_dir` exists (the runner creates them).
* Empty/missing JSON/CSV should still let the pipeline complete; verify your model and `conf` threshold.

---

## 11. Roadmap / Extensions

* **Image gallery** (HTML) with thumbnails and links to masks/crops/JSON
* **Per-class spatial heatmaps** across a folder
* **Zero-shot classification** (e.g., CLIP-style prompts)
* **ONNX/TensorRT** export for deployment

---

### Code map

* **Runner & configs**

  * `scripts/cv/run_image_analysis.py`
  * `configs/cv/image_analysis_det.yaml`
  * `configs/cv/image_analysis_seg.yaml`
  * `configs/cv/image_analysis_pose.yaml`
  * `configs/cv/image_analysis_cls.yaml`
* **Core image logic**

  * `src/airoad/cv/image_analysis.py`

---

Happy segmenting / detecting / posing / classifying!
