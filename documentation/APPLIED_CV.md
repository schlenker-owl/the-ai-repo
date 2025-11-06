# Applied Computer Vision — Video Analytics with Ultralytics YOLO

**Repo:** [`the-ai-repo`](https://github.com/schlenker-owl/the-ai-repo)

This guide documents the **Applied CV** tools in this repo for running **object detection + tracking** on videos, generating **analytics (zones, lines, dwell, speed)**, and producing **figures and recap videos** for quick reviews.

The pipeline is designed to be:

* **Local-first & fast** (runs on CPU/MPS/GPU; single command)
* **Reproducible** (config via YAML; deterministic outputs)
* **Modular** (analysis → data → graphics/video are cleanly separated)
* **Batch-friendly** (analyze whole folders at once)

---

## Contents

* [1. Requirements](#1-requirements)
* [2. Install](#2-install)
* [3. Quick Start (single video)](#3-quick-start-single-video)
* [4. Batch Mode (folder of videos)](#4-batch-mode-folder-of-videos)
* [5. Configuration (YAML)](#5-configuration-yaml)
* [6. What You Get (outputs)](#6-what-you-get-outputs)
* [7. Graphics & Heatmaps](#7-graphics--heatmaps)
* [8. Recap Video](#8-recap-video)
* [9. Overlays (lines/zones/HUD)](#9-overlays-lineszoneshud)
* [10. Analytics Explained](#10-analytics-explained)
* [11. Performance Tips](#11-performance-tips)
* [12. Troubleshooting](#12-troubleshooting)
* [13. CSV Schemas](#13-csv-schemas)
* [14. Licensing Notes](#14-licensing-notes)
* [15. Roadmap / Extending](#15-roadmap--extending)

---

## 1. Requirements

* Python **3.11**
* [`uv`](https://docs.astral.sh/uv/) (fast package manager)
* macOS (Intel/Apple Silicon), Linux, or Windows
* Optional: NVIDIA GPU (CUDA) or Apple Silicon **MPS** for faster inference

---

## 2. Install

We keep CV dependencies in a dedicated extra group.

```bash
# from repo root
uv sync -g cv
```

This installs:

* `ultralytics` (YOLO11)
* `opencv-python`
* `pandas`
* `pyyaml`
* `matplotlib` (for charts; headless backend)

---

## 3. Quick Start (single video)

1. Put your input video somewhere in `data/videos/`, e.g.:

```
data/videos/aivideo_12.mp4
```

2. Edit `configs/cv/video_analysis.yaml`:

```yaml
source: "data/videos/aivideo_12.mp4"   # single-file mode

# clean video by default (only tracking boxes/IDs from YOLO)
draw_hud: false
draw_lines: false
label_lines: false
draw_zones: false
label_zones: false
```

3. Run analysis:

```bash
uv run python scripts/cv/run_video_analysis.py --config configs/cv/video_analysis.yaml
```

You’ll get per-video outputs under:

```
outputs/cv/aivideo_12/
  analysis.mp4       # annotated tracking video (clean, no HUD/labels by default)
  events.csv
  tracks.csv
  summary.json
  analysis.log
```

4. Build figures & heatmaps:

```bash
uv run python scripts/cv/build_graphics.py --video-dir outputs/cv/aivideo_12
```

Figures land in `outputs/cv/aivideo_12/figs/`.

5. (Optional) Recap video:

```bash
uv run python scripts/cv/render_recap.py --video-dir outputs/cv/aivideo_12
```

By default this **just copies** `analysis.mp4` to `recap.mp4` (no overlay panel).

---

## 4. Batch Mode (folder of videos)

Instead of `source`, use `source_dir` and optional patterns:

```yaml
# configs/cv/video_analysis.yaml
source_dir: "data/videos"
source_glob: ["*.mp4","*.mov","*.mkv","*.avi","*.m4v","*.webm"]
recursive: false
output_root: "outputs/cv"

# clean video defaults
draw_hud: false
draw_lines: false
label_lines: false
draw_zones: false
label_zones: false
```

Run:

```bash
uv run python scripts/cv/run_video_analysis.py --config configs/cv/video_analysis.yaml
```

You’ll get a per-video subfolder under `outputs/cv/<video-stem>/…` and a batch index:

```
outputs/cv/_batch_summary.json
```

Then generate figures for all runs:

```bash
uv run python scripts/cv/build_graphics.py --batch-root outputs/cv
```

---

## 5. Configuration (YAML)

Key fields in `configs/cv/video_analysis.yaml`:

```yaml
# Single-file OR folder
# source: "data/videos/example.mp4"
# OR
# source_dir: "data/videos"
# source_glob: ["*.mp4","*.mov"]  # optional; string or list
recursive: false
output_root: "outputs/cv"

# Model & tracking
model: "yolo11n.pt"       # change to yolo11s.pt for better accuracy (slower)
tracker: "botsort.yaml"   # or "bytetrack.yaml"

# Inference
conf: 0.3
iou: 0.5
imgsz: 640
classes: null             # e.g., [0,2,3,5,7] for person/car/motorcycle/bus/truck
device: null              # "cpu", "mps", "0", ...

frame_stride: 1           # process every Nth frame (increase for speed)

# Display/drawing (clean defaults)
draw_labels: true
draw_masks: true
draw_tracks: true
show: false
save_video: true

draw_hud: false           # big black stats panel (off)
draw_lines: false         # draw line geometry
label_lines: false        # print "entry_line: ..." text
draw_zones: false         # draw zone polygons
label_zones: false        # print zone labels

# Analytics
lines:
  - name: entry_line
    p1: [100, 200]
    p2: [540, 200]
    classes: [0]          # filter to person=0
zones:
  - name: roi_zone
    points: [[50,50],[600,50],[600,400],[50,400]]
    classes: [0]

meter_per_pixel: null     # enable speed when set, e.g., 0.05 (meters per pixel)
max_speed_kmh: 200.0
speed_ema_alpha: 0.3
track_lost_patience: 30

# Logging
log_level: "INFO"
log_every_n_frames: 100
log_file: "outputs/cv/analysis.log"
```

> **Tip:** Analytics (events & counts) still run even if `draw_lines/draw_zones` are false — that only controls visualization.

---

## 6. What You Get (outputs)

Per video (e.g., `outputs/cv/aivideo_12/`):

* `analysis.mp4` — Annotated video with YOLO detections/tracks.
  *By default, only boxes/IDs — no HUD/labels overlay.*
* `events.csv` — Discrete events:

  * `zone_enter`, `zone_exit`, `zone_exit_lost` (with `zone_dwell_s` on exits)
  * `line_cross` (with `line_dir`: `A->B` or `B->A`)
* `tracks.csv` — One row per detection per frame; includes:

  * `frame`, `track_id`, `class_id`, `class_name`, `x1 y1 x2 y2`, center `cx cy`, `conf`, `speed_kmh`, `speed_kmh_ema`
* `summary.json` — Run summary:

  * `frames`, `fps`, `unique_ids_by_class`, `zone_counts`, `zone_dwell_stats`, `line_counts`, `line_counts_dir`
* `analysis.log` — structured log output
* `figs/` — (after running `build_graphics.py`) charts & heatmaps, plus `_figs_index.json`

Batch index:

* `outputs/cv/_batch_summary.json` — list of videos + per-video outputs & summaries

---

## 7. Graphics & Heatmaps

Generate charts/heatmaps for a single run:

```bash
uv run python scripts/cv/build_graphics.py --video-dir outputs/cv/aivideo_12
```

Or for all `outputs/cv/*/`:

```bash
uv run python scripts/cv/build_graphics.py --batch-root outputs/cv
```

Produced (in `figs/`):

* `classes_bar.png` — Unique tracks per class
* `active_tracks_timeseries.png` — Active tracks vs time
* `zone_dwell_hist.png` — Dwell time histogram per zone
* `line_direction_bars.png` — A→B vs B→A counts per line
* `speed_distribution.png` — Histogram of `speed_kmh_ema` if speed enabled
* `activity_heatmap.png` — Spatial heatmap (overlays on first frame of `analysis.mp4` if present)

Robust to **empty/missing** `events.csv`.

---

## 8. Recap Video

By default, `render_recap.py` simply copies `analysis.mp4` to `recap.mp4` (no overlay panel):

```bash
uv run python scripts/cv/render_recap.py --video-dir outputs/cv/aivideo_12
```

> Want a tiny live scoreboard later? We can add an `--enable-panel` flag and draw a minimal corner overlay without obscuring action.

---

## 9. Overlays (lines/zones/HUD)

* **draw_lines / draw_zones** control whether geometry is drawn.
* **label_lines / label_zones** control whether counts and labels are printed on top.
* **draw_hud** toggles the large “stats panel.” Defaults are false for a **clean** video.

These **do not** affect analytics; events and counts are still computed.

---

## 10. Analytics Explained

* **Tracking IDs**: Ultralytics `track(...)` maintains IDs via BoT-SORT / ByteTrack.
* **Zones**: Polygon inclusion checks on **box center**. We record `zone_enter` on first entry, `zone_exit` on leaving, plus dwell (`zone_dwell_s`).
* **Lines**: Directional crossing between `p1→p2` using the signed area test on successive centers:

  * negative→positive → `A->B`
  * positive→negative → `B->A`
* **Speed** (optional):
  `speed_kmh = distance(px) × meter_per_pixel × fps × 3.6`

  * Enable by setting `meter_per_pixel` (e.g., 0.05 m/px if 2.0 m spans 40 px).
  * `speed_kmh_ema` is an exponentially smoothed value (alpha: `speed_ema_alpha`).

---

## 11. Performance Tips

* Increase `frame_stride` to 2–5 to speed up long videos (trades temporal resolution).
* Use a slightly larger model (`yolo11s.pt`) for fewer mislabels (slower than `yolo11n.pt`).
* Filter classes (e.g., `classes: [0,2,3,5,7]`) for person/vehicles only.
* Set `device: "mps"` (Apple Silicon) or `"0"` (first CUDA GPU) for big speedups.

---

## 12. Troubleshooting

**No events (empty `events.csv`)**

* Lines/zones might not be in the path of the tracked centers. Reposition geometry, verify class filters, and check `line_counts`/`zone_counts` in `summary.json`.
* `build_graphics.py` is robust to empty/missing events and will still produce other charts/heatmaps.

**Weird class names (e.g., “teddy bear”)**

* That’s model noise on small/tiny models in non-COCO scenes. Use `classes:` to restrict or upgrade to a larger model (`yolo11s.pt`).

**Panel covers video**

* By default it’s off; if you enabled it, set `draw_hud: false`.

**Tracking resets**

* Ensure you are using `track(...)` (we do), and avoid cutting the stream mid-video.

---

## 13. CSV Schemas

**`tracks.csv`** (one row per frame detection)

| column        | type  | notes                    |
| ------------- | ----- | ------------------------ |
| frame         | int   | 0-indexed                |
| time_s        | float | frame / fps              |
| track_id      | int   | -1 if no ID              |
| class_id      | int   | COCO class ID            |
| class_name    | str   | derived from model names |
| x1,y1,x2,y2   | float | box corners              |
| cx,cy         | float | box center               |
| conf          | float | detection confidence     |
| speed_kmh     | float | optional                 |
| speed_kmh_ema | float | optional                 |

**`events.csv`** (sparse; only events)

| column        | notes                                                     |
| ------------- | --------------------------------------------------------- |
| frame,time_s  | event timestamp                                           |
| event         | `zone_enter`, `zone_exit`, `zone_exit_lost`, `line_cross` |
| zone          | zone name if zone event                                   |
| line,line_dir | line name and direction (`A->B` or `B->A`)                |
| track_id      | track involved                                            |
| class_id/name | classification                                            |
| zone_dwell_s  | provided on exits / lost                                  |

---

## 14. Licensing Notes

* The `ultralytics` package (YOLO11) is **AGPL-3.0**. It’s fine for open repos and internal use; redistribution/closed embedding may require a separate license.
* Check model/data licenses before publishing derivative outputs.

---

## 15. Roadmap / Extending

* **Highlights video**: auto-cut the top-K busiest windows into a single reel.
* **Per-class heatmaps**: split `activity_heatmap.png` by class groups (people vs vehicles).
* **Small recap overlay**: optional, corner HUD with current counts (opt-in flag).
* **ONNX/TensorRT**: lower latency serving path for productionization.

---

### Maintainers

* Repo: [`the-ai-repo`](https://github.com/schlenker-owl/the-ai-repo)
* Applied CV code:

  * `src/airoad/cv/video_analysis.py`
  * `scripts/cv/run_video_analysis.py`
  * `scripts/cv/build_graphics.py`
  * `src/airoad/cv/graphics/*` (charts & heatmap)
  * `scripts/cv/render_recap.py`

---

Happy analyzing!
