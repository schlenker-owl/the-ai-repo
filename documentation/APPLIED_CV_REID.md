# Applied CV — Multi-Video Re-Identification (ReID) & Global Analytics
**Repo:** [the-ai-repo](https://github.com/schlenker-owl/the-ai-repo)

This guide explains how to turn per-video tracks into **global identities** across many videos/cameras using appearance embeddings + simple constraints. You’ll get:

- **global_id** per `(video, track_id)`
- per-video CSVs augmented with global IDs
- **galleries** (thumbnails) per global ID for QA and demos
- **cross-video flows** (Sankey-ready)
- **merged analytics** (global uniques by class, track counts)

---

## 1) Prereqs

- You’ve run the standard video analysis so each video has:
```

outputs/cv/<video-stem>/
analysis.mp4
tracks.csv
summary.json

````
- Python 3.11 + `uv sync -g cv`
- (Optional) OpenCLIP for stronger embeddings:
```bash
uv add --group cv "open-clip-torch>=2.24.0"
uv sync -g cv
````

---

## 2) Configure

`configs/cv/reid.yaml` (defaults shown):

```yaml
backbone:
  name: "auto"        # "auto" | "open-clip:ViT-B-32" | "resnet50"
  device: null        # "cpu" | "mps" | "0" | "cuda"
  dtype: "float32"
  image_size: 224

embed:
  batch_size: 64
  frame_stride: 5
  per_track_max_samples: 12
  min_box_size: 16
  class_filter: null   # e.g., [0] for person only

associate:
  min_sim: 0.45
  same_class_only: true
  max_same_video_time_overlap: 0
  allow_merge_same_video: false
```

**Tips**

* Start with `class_filter: [0]` (persons) to validate the flow.
* If merges are too aggressive, raise `min_sim`; if too conservative, lower it slightly.

---

## 3) Build embeddings (per video)

```bash
uv run python scripts/cv/reid/build_track_embeddings.py --root outputs/cv --config configs/cv/reid.yaml
```

This writes:

```
outputs/cv/reid/embeddings/
  <video>_track_meta.csv
  <video>_track_embeds.npz
```

---

## 4) Associate tracks across videos → `global_id`

```bash
uv run python scripts/cv/reid/associate_tracks.py --root outputs/cv --config configs/cv/reid.yaml
```

You’ll get:

```
outputs/cv/reid/global_id_map.csv
outputs/cv/reid/merged_analytics.json
```

`merged_analytics.json` includes `num_globals`, `unique_by_class`, etc.

---

## 5) Attach global IDs back to each video

```bash
uv run python scripts/cv/reid/attach_global_ids.py --root outputs/cv
```

Per video folder:

```
outputs/cv/<video>/tracks_global.csv     # tracks.csv + global_id
outputs/cv/<video>/summary_global.json   # summary + global uniques
```

---

## 6) Visual QA — Galleries

```bash
uv run python scripts/cv/reid/build_galleries.py --root outputs/cv --per-id 8 --thumb 224
```

Thumbnails per global:

```
outputs/cv/reid/galleries/global_<id>/*.jpg
```

Open a few folders and eyeball consistency. 👍

---

## 7) Cross-Video Flows (Sankey-ready)

```bash
uv run python scripts/cv/reid/build_cross_video_flows.py --root outputs/cv
```

Outputs:

```
outputs/cv/reid/flows.csv    # columns: from, to, count
outputs/cv/reid/flows.json
```

Load into any Sankey tool (Plotly, Flourish, Grafana) for cross-camera movement.

---

## 8) Acceptance checks

* No cross-class merges when `same_class_only: true`.
* Sanity of merges via `galleries/global_<id>/…`.
* Global unique by class > per-video uniques (no duplication).
* Adjust `associate.min_sim` until results match your domain.

---

## 9) Notes & scaling

* For 100k+ tracks, switch to FAISS (IVF+PQ) for fast similarity; the core code is pluggable.
* Maintain embeddings per class (block matrix compare) to reduce false merges.
* Add scene/time gating if you know camera topology.

---

## 10) Code map

```
src/airoad/reid/
  backbone.py     # OpenCLIP or ResNet50 embeddings
  embedder.py     # sample crops from analysis.mp4 and average features per track
  matcher.py      # cosine + constraints → global_id

scripts/cv/reid/
  build_track_embeddings.py
  associate_tracks.py
  attach_global_ids.py
  build_galleries.py
  build_cross_video_flows.py
```
