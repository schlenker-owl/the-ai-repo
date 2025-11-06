#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

VID_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


@dataclass
class VCfg:
    source_dir: Path
    output_root: Path
    dataset_name: str
    model: str = "yolo11n-seg.pt"
    task: str = "auto"  # "auto" | "seg" | "det"
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 640
    classes: Optional[List[int]] = None
    device: Optional[str] = None
    frame_stride: int = 15  # sample every N frames
    scene_threshold: float = 40.0  # simple scene cut via HSV hist delta (0-100); 0 disables
    max_frames_per_video: int = 300
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    copy_mode: str = "link"  # copied frames are new files, so "link" is mostly irrelevant


def _iter_videos(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in VID_EXTS])


def _scene_score(prev_h: np.ndarray, cur_h: np.ndarray) -> float:
    # L1 distance (%) between normalized histograms
    return float(100.0 * np.abs(prev_h - cur_h).sum())


def _hist_hsv(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    h = h / max(1e-6, h.sum())
    return h.reshape(-1)


def _sample_frames(
    video: Path, out_dir: Path, stride: int, scene_thr: float, max_frames: int, start_idx: int
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    # fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last_hist = None
    idx = start_idx
    saved = []
    f = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1
        if f % stride != 0:
            continue
        if scene_thr > 0:
            hist = _hist_hsv(frame)
            if last_hist is not None:
                if _scene_score(last_hist, hist) < scene_thr:
                    continue
            last_hist = hist
        # save frame
        fn = out_dir / f"{video.stem}_f{f:06d}.jpg"
        cv2.imwrite(str(fn), frame)
        saved.append(fn)
        idx += 1
        if len(saved) >= max_frames:
            break
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=str, required=True)
    ap.add_argument("--output-root", type=str, default="datasets/autolabel")
    ap.add_argument("--dataset-name", type=str, required=True)
    ap.add_argument("--model", type=str, default="yolo11n-seg.pt")
    ap.add_argument("--task", type=str, default="auto", choices=["auto", "seg", "det"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", type=str, default=None)  # comma list
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=15)
    ap.add_argument("--scene-threshold", type=float, default=40.0)
    ap.add_argument("--max-frames-per-video", type=int, default=300)
    ap.add_argument("--split", type=str, default="0.8,0.1,0.1")
    args = ap.parse_args()

    classes = None
    if args.classes:
        classes = [int(x) for x in args.classes.split(",") if x.strip()]
    s = [float(x) for x in args.split.split(",")]
    assert len(s) == 3 and abs(sum(s) - 1.0) < 1e-6, "split must sum to 1.0"

    cfg = VCfg(
        source_dir=Path(args.source_dir),
        output_root=Path(args.output_root),
        dataset_name=args.dataset_name,
        model=args.model,
        task=args.task,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=classes,
        device=args.device,
        frame_stride=args.frame_stride,
        scene_threshold=args.scene_threshold,
        max_frames_per_video=args.max_frames_per_video,
        split=(s[0], s[1], s[2]),
    )

    videos = _iter_videos(cfg.source_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {cfg.source_dir}")

    # 1) Extract frames from all videos into tmp images folder
    out_ds = cfg.output_root / cfg.dataset_name
    frames_dir = out_ds / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    all_frames: List[Path] = []
    for v in videos:
        saved = _sample_frames(
            v,
            frames_dir,
            cfg.frame_stride,
            cfg.scene_threshold,
            cfg.max_frames_per_video,
            len(all_frames),
        )
        all_frames.extend(saved)

    if not all_frames:
        raise RuntimeError("No frames sampled; adjust stride/threshold.")

    # 2) Reuse the image auto-label logic: run model -> save YOLO labels + data.yaml
    model = YOLO(cfg.model)
    names_map = None

    # split frames
    N = len(all_frames)
    idx = list(range(N))
    import random

    random.seed(42)
    random.shuffle(idx)
    n_train = int(cfg.split[0] * N)
    n_val = int(cfg.split[1] * N)
    train_idx = set(idx[:n_train])
    val_idx = set(idx[n_train : n_train + n_val])
    # test_idx = set(idx[n_train + n_val :])

    img_train = out_ds / "images" / "train"
    img_val = out_ds / "images" / "val"
    img_test = out_ds / "images" / "test"
    lab_train = out_ds / "labels" / "train"
    lab_val = out_ds / "labels" / "val"
    lab_test = out_ds / "labels" / "test"
    for d in [img_train, img_val, img_test, lab_train, lab_val, lab_test]:
        d.mkdir(parents=True, exist_ok=True)

    def _yolo_det_line(w, h, c, x1, y1, x2, y2):
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def _largest_poly(mask):
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, max(1.0, 0.01 * cv2.arcLength(c, True)), True)
        return c.reshape(-1, 2)

    def _yolo_seg_line(w, h, c, poly):
        poly = poly.astype(float)
        poly[:, 0] = np.clip(poly[:, 0] / w, 0, 1)
        poly[:, 1] = np.clip(poly[:, 1] / h, 0, 1)
        return f"{c} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1))

    B = 32
    batch = []
    bpaths = []
    for i, p in enumerate(all_frames):
        batch.append(str(p))
        bpaths.append(p)
        if len(batch) < B and i < N - 1:
            continue
        rlist = model.predict(
            source=batch,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            classes=cfg.classes,
            device=cfg.device,
            verbose=False,
            save=False,
            stream=False,
        )
        for k, (r, pp) in enumerate(zip(rlist, bpaths)):
            img = cv2.imread(str(pp))
            h, w = img.shape[:2]
            if names_map is None:
                names_map = r.names if hasattr(r, "names") else {}
            task = (
                cfg.task
                if cfg.task != "auto"
                else ("seg" if getattr(r, "masks", None) is not None else "det")
            )
            # choose split
            lbl_dir = lab_train if i + k in train_idx else lab_val if i + k in val_idx else lab_test
            img_dir = img_train if i + k in train_idx else img_val if i + k in val_idx else img_test
            img_dst = img_dir / pp.name
            (
                os.replace(str(pp), str(img_dst))
                if pp.parent == img_dir
                else (pp.rename(img_dst) if (pp.parent == frames_dir) else None)
            )  # move if from frames
            if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                (lbl_dir / f"{pp.stem}.txt").write_text("")
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(float)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            lines = []
            if task == "seg" and getattr(r, "masks", None) is not None:
                polys = getattr(r.masks, "xy", None)
                for j in range(len(xyxy)):
                    c = int(cls[j])
                    if polys is not None and j < len(polys):
                        poly = np.array(polys[j], dtype=float)
                    else:
                        m = r.masks.data[j].cpu().numpy()
                        poly = _largest_poly((m * 255).astype(np.uint8))
                        if poly is None:
                            x1, y1, x2, y2 = xyxy[j]
                            lines.append(_yolo_det_line(w, h, c, x1, y1, x2, y2))
                            continue
                    lines.append(_yolo_seg_line(w, h, c, poly))
            else:
                for j in range(len(xyxy)):
                    c = int(cls[j])
                    x1, y1, x2, y2 = xyxy[j]
                    lines.append(_yolo_det_line(w, h, c, x1, y1, x2, y2))
            (lbl_dir / f"{pp.stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        batch, bpaths = [], []

    names = [v for k, v in sorted(names_map.items(), key=lambda kv: kv[0])] if names_map else []
    data_yaml = {
        "path": str(out_ds),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "nc": len(names),
    }
    (out_ds / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"[auto_label_videos] Wrote dataset at {out_ds} with classes: {names}")


if __name__ == "__main__":
    main()
