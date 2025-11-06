#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class ALConfig:
    source_dir: Path
    output_root: Path
    dataset_name: str
    model: str = "yolo11n-seg.pt"  # supports det too
    task: str = "auto"  # "auto" | "seg" | "det"
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 640
    classes: Optional[List[int]] = None
    device: Optional[str] = None
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train/val/test
    copy_mode: str = "link"  # "link" | "copy" | "hardlink"
    seed: int = 42
    save_coco: bool = True
    save_labelstudio: bool = True
    verbose: bool = True  # print progress


def _iter_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def _infer_task_from_results(r) -> str:
    if getattr(r, "masks", None) is not None:
        return "seg"
    if getattr(r, "boxes", None) is not None:
        return "det"
    return "det"


def _symlink_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            import shutil

            shutil.copy2(src, dst)
    else:
        # symlink, overwrite if exists (common in re-runs)
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except Exception:
                pass
        dst.symlink_to(src.resolve())


def _yolo_det_line_local(
    w: int, h: int, lid: int, x1: float, y1: float, x2: float, y2: float
) -> str:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{lid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _yolo_seg_line_local(w: int, h: int, lid: int, poly_xy: np.ndarray) -> str:
    if poly_xy.ndim == 1:
        poly_xy = poly_xy.reshape(-1, 2)
    pts = poly_xy.astype(float).copy()
    pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
    flat = " ".join([f"{v:.6f}" for v in pts.reshape(-1)])
    return f"{lid} {flat}"


def _largest_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(1.0, 0.01 * cv2.arcLength(c, True)), True)
    return c.reshape(-1, 2)


def _write_txt(txt_path: Path, lines: List[str]):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def _save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _names_from_results(r) -> Dict[int, str]:
    return r.names if hasattr(r, "names") else {}


def _count_images(d: Path) -> int:
    return sum(1 for ext in IMG_EXTS for _ in d.glob(f"*{ext}"))


def _ensure_txt_exists(lbl_path: Path):
    if not lbl_path.exists():
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text("")  # empty label OK


def _duplicate_pair(src_img: Path, src_lbl: Path, dst_img_dir: Path, dst_lbl_dir: Path, mode: str):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    img_dst = dst_img_dir / src_img.name
    _symlink_or_copy(src_img, img_dst, mode)
    lbl_dst = dst_lbl_dir / (src_img.stem + ".txt")
    if src_lbl.exists():
        _symlink_or_copy(src_lbl, lbl_dst, "copy" if mode == "copy" else "link")
    else:
        _ensure_txt_exists(lbl_dst)


def _fix_min_splits(
    img_train: Path,
    lab_train: Path,
    img_val: Path,
    lab_val: Path,
    img_test: Path,
    lab_test: Path,
    copy_mode: str,
    verbose: bool = True,
):
    """
    Ensure at least 1 image in train and val.
    If val empty and train has >=1, duplicate (not move) 1 sample from train->val.
    If train empty and val has >=1, duplicate 1 sample val->train (else from test).
    """
    n_tr = _count_images(img_train)
    n_va = _count_images(img_val)
    n_te = _count_images(img_test)

    if verbose:
        print(
            f"[auto_label_images][post] current split counts: train={n_tr}, val={n_va}, test={n_te}"
        )

    # ensure train has at least 1
    if n_tr == 0:
        # try val first, else test
        src_img_dir, src_lbl_dir = (img_val, lab_val) if n_va > 0 else (img_test, lab_test)
        imgs = sorted([p for ext in IMG_EXTS for p in src_img_dir.glob(f"*{ext}")])
        if imgs:
            src_img = imgs[0]
            src_lbl = src_lbl_dir / (src_img.stem + ".txt")
            if verbose:
                print(f"[auto_label_images][post] duplicating {src_img.name} -> train/")
            _duplicate_pair(src_img, src_lbl, img_train, lab_train, copy_mode)
            n_tr += 1

    # ensure val has at least 1
    if n_va == 0:
        src_img_dir, src_lbl_dir = (img_train, lab_train) if n_tr > 0 else (img_test, lab_test)
        imgs = sorted([p for ext in IMG_EXTS for p in src_img_dir.glob(f"*{ext}")])
        if imgs:
            src_img = imgs[0]
            src_lbl = src_lbl_dir / (src_img.stem + ".txt")
            if verbose:
                print(f"[auto_label_images][post] duplicating {src_img.name} -> val/")
            _duplicate_pair(src_img, src_lbl, img_val, lab_val, copy_mode)

    if verbose:
        print(
            f"[auto_label_images][post] ensured min splits; "
            f"train={_count_images(img_train)}, val={_count_images(img_val)}"
        )


def auto_label_images(cfg: ALConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    images = _iter_images(cfg.source_dir)
    if cfg.verbose:
        print(f"[auto_label_images] scanning {cfg.source_dir} ... found {len(images)} images")

    if not images:
        raise FileNotFoundError(
            f"No images under {cfg.source_dir}. " f"Supported extensions: {', '.join(IMG_EXTS)}"
        )

    out_ds = cfg.output_root / cfg.dataset_name

    # Pre-create split dirs so Ultralytics finds them even if empty
    img_train = out_ds / "images" / "train"
    img_val = out_ds / "images" / "val"
    img_test = out_ds / "images" / "test"
    lab_train = out_ds / "labels" / "train"
    lab_val = out_ds / "labels" / "val"
    lab_test = out_ds / "labels" / "test"
    for d in [img_train, img_val, img_test, lab_train, lab_val, lab_test]:
        d.mkdir(parents=True, exist_ok=True)

    # Split indices
    N = len(images)
    idx = list(range(N))
    random.shuffle(idx)
    n_train = int(cfg.split[0] * N)
    n_val = int(cfg.split[1] * N)
    train_idx = set(idx[:n_train])
    val_idx = set(idx[n_train : n_train + n_val])
    test_idx = set(idx[n_train + n_val :])

    if cfg.verbose:
        print(
            f"[auto_label_images] split ratios={cfg.split} -> counts approx: "
            f"train~{len(train_idx)}, val~{len(val_idx)}, test~{len(test_idx)}"
        )

    # Load model once
    model = YOLO(cfg.model)

    # Local class mapping (COCO id → local contiguous 0..K-1)
    id_map: Dict[int, int] = {}
    names_list: List[str] = []

    def _lid(cid: int, cname: str) -> int:
        if cid not in id_map:
            id_map[cid] = len(names_list)
            names_list.append(cname)
        return id_map[cid]

    coco_images, coco_anns = [], []
    ann_id = 1

    # Batch predict by (index, path)
    B = 32
    batch: List[Tuple[int, Path]] = []

    def _dst_dirs_for_idx(i: int) -> tuple[Path, Path]:
        if i in train_idx:
            return img_train, lab_train
        if i in val_idx:
            return img_val, lab_val
        return img_test, lab_test

    total_batches = (N + B - 1) // B
    for i, p in enumerate(images):
        batch.append((i, p))
        if len(batch) < B and i < N - 1:
            continue

        if cfg.verbose:
            cur_batch = (i // B) + 1
            print(
                f"[auto_label_images] processing batch {cur_batch}/{total_batches} "
                f"({len(batch)} images) ..."
            )

        # Run prediction on this batch
        rlist = model.predict(
            source=[str(pp) for _, pp in batch],
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            classes=cfg.classes,
            device=cfg.device,
            verbose=False,
            save=False,
            stream=False,
        )

        for (img_idx, path), r in zip(batch, rlist):
            img = cv2.imread(str(path))
            if img is None:
                if cfg.verbose:
                    print(f"[auto_label_images][warn] failed to read {path}")
                continue
            h, w = img.shape[:2]
            names_map = _names_from_results(r)
            task = cfg.task if cfg.task != "auto" else _infer_task_from_results(r)
            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)

            img_dst_dir, lab_dst_dir = _dst_dirs_for_idx(img_idx)
            img_dst = img_dst_dir / path.name
            _symlink_or_copy(path, img_dst, cfg.copy_mode)
            yolo_txt = lab_dst_dir / f"{path.stem}.txt"
            yolo_lines: List[str] = []

            # COCO image record
            img_id = img_idx + 1
            coco_images.append(
                {
                    "id": img_id,
                    "file_name": f"{img_dst_dir.name}/{path.name}",
                    "width": w,
                    "height": h,
                }
            )

            if boxes is None or len(boxes) == 0:
                _write_txt(yolo_txt, [])
                continue

            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            cls = boxes.cls.cpu().numpy().astype(int)

            if task == "seg" and masks is not None:
                polys_native = getattr(masks, "xy", None)  # list of Nx2 arrays in image coords
                for j in range(len(xyxy)):
                    cid = int(cls[j])
                    cname = names_map.get(cid, str(cid))
                    lid = _lid(cid, cname)
                    if polys_native is not None and j < len(polys_native):
                        poly = np.array(polys_native[j], dtype=float)
                    else:
                        mbin = masks.data[j].cpu().numpy()
                        poly = _largest_polygon((mbin * 255).astype(np.uint8))
                        if poly is None or len(poly) < 3:
                            x1, y1, x2, y2 = xyxy[j]
                            yolo_lines.append(_yolo_det_line_local(w, h, lid, x1, y1, x2, y2))
                            x, y, bw, bh = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                            coco_anns.append(
                                {
                                    "id": ann_id,
                                    "image_id": img_id,
                                    "category_id": lid,
                                    "bbox": [x, y, bw, bh],
                                    "area": float(bw * bh),
                                    "iscrowd": 0,
                                }
                            )
                            ann_id += 1
                            continue
                    yolo_lines.append(_yolo_seg_line_local(w, h, lid, poly))
                    seg = poly.reshape(-1).tolist()
                    x1, y1, x2, y2 = xyxy[j]
                    x, y, bw, bh = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                    coco_anns.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": lid,
                            "segmentation": [seg],
                            "bbox": [x, y, bw, bh],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
            else:
                # detection only
                for j in range(len(xyxy)):
                    cid = int(cls[j])
                    cname = names_map.get(cid, str(cid))
                    lid = _lid(cid, cname)
                    x1, y1, x2, y2 = xyxy[j]
                    yolo_lines.append(_yolo_det_line_local(w, h, lid, x1, y1, x2, y2))
                    x, y, bw, bh = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                    coco_anns.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": lid,
                            "bbox": [x, y, bw, bh],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            _write_txt(yolo_txt, yolo_lines)

        batch = []

    # --- Ensure min split sizes (so YOLO doesn't crash when val/train empty) ---
    _fix_min_splits(
        img_train,
        lab_train,
        img_val,
        lab_val,
        img_test,
        lab_test,
        cfg.copy_mode,
        verbose=cfg.verbose,
    )

    # Write data.yaml with LOCAL names
    names = names_list
    data_yaml = {
        "path": str(out_ds),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "nc": len(names),
    }
    (out_ds / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # COCO export (optional)
    if cfg.save_coco:
        cats = [{"id": i, "name": n} for i, n in enumerate(names)]
        coco = {"images": coco_images, "annotations": coco_anns, "categories": cats}
        (out_ds / "exports").mkdir(parents=True, exist_ok=True)
        with open(out_ds / "exports" / "coco.json", "w") as f:
            json.dump(coco, f, indent=2)

    # Label Studio export (optional)
    if cfg.save_labelstudio:
        tasks = [
            {"data": {"image": str((out_ds / rec["file_name"]).as_posix())}} for rec in coco_images
        ]
        with open(out_ds / "exports" / "label_studio.jsonl", "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

    # Final counts
    ntr = _count_images(img_train)
    nva = _count_images(img_val)
    nte = _count_images(img_test)
    print(f"[auto_label_images] dataset at {out_ds}")
    print(f"[auto_label_images] split: train={ntr}, val={nva}, test={nte}, classes={names}")
    if len(names) == 0:
        print(
            "[auto_label_images][warn] No classes found; verify the model produced predictions at conf/iou thresholds."
        )


# ---------- CLI entrypoint ----------


def parse_args() -> ALConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=str, required=True)
    ap.add_argument("--output-root", type=str, default="datasets/autolabel")
    ap.add_argument("--dataset-name", type=str, required=True)
    ap.add_argument("--model", type=str, default="yolo11n-seg.pt")
    ap.add_argument("--task", type=str, default="auto", choices=["auto", "seg", "det"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", type=str, default=None, help="comma list, e.g. 0,2,3")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--split", type=str, default="0.8,0.1,0.1")
    ap.add_argument("--copy-mode", type=str, default="link", choices=["link", "copy", "hardlink"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-coco", action="store_true")
    ap.add_argument("--no-labelstudio", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="suppress batch progress")
    args = ap.parse_args()

    classes = None
    if args.classes:
        classes = [int(x) for x in args.classes.split(",") if x.strip()]

    s = [float(x) for x in args.split.split(",")]
    assert (
        len(s) == 3 and abs(sum(s) - 1.0) < 1e-6
    ), "split must be like '0.8,0.1,0.1' summing to 1.0"

    return ALConfig(
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
        split=(s[0], s[1], s[2]),
        copy_mode=args.copy_mode,
        seed=args.seed,
        save_coco=not args.no_coco,
        save_labelstudio=not args.no_labelstudio,
        verbose=not args.quiet,
    )


def main():
    cfg = parse_args()
    auto_label_images(cfg)


if __name__ == "__main__":
    main()
