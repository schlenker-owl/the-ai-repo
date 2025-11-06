from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class ImageAnalyzerConfig:
    # Single image mode (optional; batch runner passes per-image path directly)
    source: str = ""

    # Model / inference
    model: str = (
        "yolo11n.pt"  # use "*-seg.pt" for segmentation, "*-pose.pt" for pose, "*-cls.pt" for classification
    )
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 640
    classes: Optional[Sequence[int]] = None
    device: Optional[str] = None

    # Outputs (batch runner will override per-image dirs)
    out_dir: str = "outputs/cv_images/sample"

    # Rendering / saving
    save_annotated: bool = True
    render_style: str = "auto"  # "auto" | "masks_only" | "ultra_default"
    mask_alpha: float = 0.45  # mask overlay alpha if masks are used
    draw_labels: bool = True  # (Ultralytics plot includes labels by default)
    draw_conf: bool = True
    save_json: bool = True
    save_crops: bool = False
    save_masks: bool = True  # save per-instance mask PNGs when masks exist
    crop_max_size: Optional[int] = None

    # Classification
    topk: int = 5  # top-k for classification outputs (if cls model)


class ImageAnalyzer:
    """
    Image analysis on top of Ultralytics YOLO with segmentation/pose/det/cls auto-detect.

    For each image, writes (as applicable):
      - annotated image (mask-only overlay for segmentation by default)
      - per-image JSON (boxes OR polygons OR keypoints OR top-k probs)
      - optional crops per detection and instance masks
    Also returns detection rows for batch CSV aggregation.
    """

    def __init__(self, cfg: ImageAnalyzerConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model)

    # ---------------- Rendering helpers ----------------

    @staticmethod
    def _color_for_class(cid: int) -> Tuple[int, int, int]:
        # deterministic bright colors per class id (BGR)
        rng = np.random.default_rng(cid * 9973)
        c = rng.integers(64, 256, size=3, dtype=np.int32)
        return int(c[2]), int(c[1]), int(c[0])

    @staticmethod
    def _render_masks_only(
        img_bgr: np.ndarray,
        masks_bin: List[np.ndarray],
        cls_ids: List[int],
        names_map: Dict[int, str],
        alpha: float = 0.45,
        put_labels: bool = True,
    ) -> np.ndarray:
        """Blend instance masks with class colors; place labels at mask centroids."""
        out = img_bgr.copy()
        H, W = out.shape[:2]
        overlay = np.zeros_like(out, dtype=np.uint8)

        for i, m in enumerate(masks_bin):
            if m is None:
                continue
            # ensure mask is HxW uint8
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            if m.dtype != np.uint8:
                m = (m > 0).astype(np.uint8) * 255
            cid = int(cls_ids[i]) if i < len(cls_ids) else -1
            color = ImageAnalyzer._color_for_class(cid)
            colored = np.zeros_like(out)
            colored[:, :] = color
            mask_3 = cv2.merge([m, m, m])
            overlay = np.where(mask_3 > 0, colored, overlay)

            if put_labels and cid >= 0:
                ys, xs = np.where(m > 0)
                if xs.size > 0:
                    cx = int(xs.mean())
                    cy = int(ys.mean())
                    cname = names_map.get(cid, str(cid))
                    cv2.putText(
                        out,
                        cname,
                        (max(0, cx - 20), max(15, cy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

        out = cv2.addWeighted(out, 1.0, overlay, float(alpha), 0)
        return out

    @staticmethod
    def _contours_from_mask(
        mask_bin: np.ndarray, approx_eps_ratio: float = 0.01
    ) -> List[List[float]]:
        """Extract polygon(s) from binary mask via contours; return list of flat x,y lists (COCO-style)."""
        if mask_bin.dtype != np.uint8:
            mask_bin = (mask_bin > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segs: List[List[float]] = []
        for c in contours:
            if c.shape[0] < 3:
                continue
            eps = approx_eps_ratio * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            poly = approx.reshape(-1, 2).astype(float)
            if poly.shape[0] >= 3:
                segs.append(poly.flatten().tolist())
        return segs

    # ---------------- Main per-image ----------------

    def analyze_image(self, image_path: str, out_dir: str) -> Dict:
        p = Path(image_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run model
        rlist = self.model.predict(
            source=str(p),
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=list(self.cfg.classes) if self.cfg.classes else None,
            device=self.cfg.device,
            verbose=False,
        )
        if not rlist:
            return {
                "image": str(p),
                "detections": [],
                "counts": {},
                "annotated": None,
                "json": None,
            }

        r = rlist[0]
        names_map = r.names if hasattr(r, "names") else {}
        orig = r.orig_img  # BGR
        H, W = orig.shape[:2]

        # Decide task by result attributes
        task = "detect"
        if getattr(r, "masks", None) is not None:
            task = "segment"
        elif getattr(r, "keypoints", None) is not None:
            task = "pose"
        elif getattr(r, "probs", None) is not None:
            task = "classify"
        elif getattr(r, "obb", None) is not None:
            task = "obb"

        det_rows: List[Dict] = []
        counts: Dict[str, int] = {}
        annotated_path = None
        json_path = None

        # -------- Classification --------
        if task == "classify":
            probs = r.probs
            k = max(1, int(self.cfg.topk))

            # Use provided top5/top1 if available; else fall back to sorting raw scores
            top_indices: List[int] = []
            top_scores: List[float] = []

            try:
                top5_idxs = list(getattr(probs, "top5", []))
                top5_confs = getattr(probs, "top5conf", None)
                if top5_idxs and top5_confs is not None:
                    confs_list = (
                        top5_confs.tolist() if hasattr(top5_confs, "tolist") else list(top5_confs)
                    )
                    n = min(k, len(top5_idxs))
                    top_indices = top5_idxs[:n]
                    top_scores = [float(c) for c in confs_list[:n]]
                else:
                    raise AttributeError("top5/top5conf not available")
            except Exception:
                data = getattr(probs, "data", None)
                if data is None and hasattr(probs, "numpy"):
                    data = probs.numpy()
                if hasattr(data, "cpu"):
                    data = data.cpu().numpy()
                arr = np.array(data).reshape(-1)
                order = arr.argsort()[::-1]
                n = min(k, len(order))
                top_indices = order[:n].tolist()
                top_scores = arr[top_indices].astype(float).tolist()

            preds = [
                {
                    "class_id": int(ci),
                    "class_name": names_map.get(int(ci), str(ci)),
                    "score": float(sc),
                }
                for ci, sc in zip(top_indices, top_scores)
            ]

            if self.cfg.save_annotated:
                ann = r.plot()  # Ultralytics will stamp top-1 label on the image
                annotated_path = str(out_dir / f"{p.stem}_cls.png")
                cv2.imwrite(annotated_path, ann)

            if self.cfg.save_json:
                json_path = str(out_dir / f"{p.stem}_cls.json")
                with open(json_path, "w") as f:
                    # ruff: E101 fix tabs to spaces
                    json.dump(
                        {"image": str(p), "width": W, "height": H, "topk": preds}, f, indent=2
                    )

            for pr in preds:
                det_rows.append(
                    {
                        "image": str(p),
                        "width": W,
                        "height": H,
                        "class_id": pr["class_id"],
                        "class_name": pr["class_name"],
                        "conf": pr["score"],
                        "x1": None,
                        "y1": None,
                        "x2": None,
                        "y2": None,
                        "area": None,
                        "crop_path": None,
                        "mask_path": None,
                        "segmentation": None,
                        "keypoints": None,
                    }
                )
                counts[pr["class_name"]] = 1

            return {
                "image": str(p),
                "annotated": annotated_path,
                "json": json_path,
                "counts": counts,
                "detections": det_rows,
            }

        # -------- Pose --------
        if task == "pose":
            boxes = r.boxes
            kps = r.keypoints
            xyxy = boxes.xyxy.cpu().numpy().astype(float) if boxes is not None else np.zeros((0, 4))
            cls = (
                boxes.cls.cpu().numpy().astype(int)
                if boxes is not None
                else np.zeros((0,), dtype=int)
            )
            conf = (
                boxes.conf.cpu().numpy().astype(float)
                if boxes is not None
                else np.zeros((0,), dtype=float)
            )
            kparr = (
                kps.xy.cpu().numpy()
                if kps is not None and getattr(kps, "xy", None) is not None
                else np.zeros((0, 0, 2))
            )

            if self.cfg.save_annotated:
                ann = r.plot()
                annotated_path = str(out_dir / f"{p.stem}_pose.png")
                cv2.imwrite(annotated_path, ann)

            keypoints_list = []
            for i in range(len(xyxy)):
                cid = int(cls[i])
                cname = names_map.get(cid, str(cid))
                conf_i = float(conf[i])
                x1, y1, x2, y2 = xyxy[i]
                if i < len(kparr) and kparr.shape[1] > 0:
                    kps_xy = kparr[i].astype(float).tolist()
                else:
                    kps_xy = []
                det_rows.append(
                    {
                        "image": str(p),
                        "width": W,
                        "height": H,
                        "class_id": cid,
                        "class_name": cname,
                        "conf": conf_i,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "area": float(max(0, (x2 - x1 + 1) * (y2 - y1 + 1))),
                        "crop_path": None,
                        "mask_path": None,
                        "segmentation": None,
                        "keypoints": kps_xy,
                    }
                )
                counts[cname] = counts.get(cname, 0) + 1
                keypoints_list.append(
                    {
                        "class_id": cid,
                        "class_name": cname,
                        "conf": conf_i,
                        "bbox": [x1, y1, x2, y2],
                        "keypoints": kps_xy,
                    }
                )

            if self.cfg.save_json:
                json_path = str(out_dir / f"{p.stem}_pose.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"image": str(p), "width": W, "height": H, "instances": keypoints_list},
                        f,
                        indent=2,
                    )

            return {
                "image": str(p),
                "annotated": annotated_path,
                "json": json_path,
                "counts": counts,
                "detections": det_rows,
            }

        # -------- OBB (graceful) --------
        if task == "obb":
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(float) if boxes is not None else np.zeros((0, 4))
            cls = (
                boxes.cls.cpu().numpy().astype(int)
                if boxes is not None
                else np.zeros((0,), dtype=int)
            )
            conf = (
                boxes.conf.cpu().numpy().astype(float)
                if boxes is not None
                else np.zeros((0,), dtype=np.float64)
            )

            if self.cfg.save_annotated:
                ann = r.plot()
                annotated_path = str(out_dir / f"{p.stem}_obb.png")
                cv2.imwrite(annotated_path, ann)

            for i in range(len(xyxy)):
                cid = int(cls[i])
                cname = names_map.get(cid, str(cid))
                conf_i = float(conf[i])
                x1, y1, x2, y2 = xyxy[i]
                det_rows.append(
                    {
                        "image": str(p),
                        "width": W,
                        "height": H,
                        "class_id": cid,
                        "class_name": cname,
                        "conf": conf_i,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "area": float(max(0, (x2 - x1 + 1) * (y2 - y1 + 1))),
                        "crop_path": None,
                        "mask_path": None,
                        "segmentation": None,
                        "keypoints": None,
                    }
                )
                counts[cname] = counts.get(cname, 0) + 1

            if self.cfg.save_json:
                json_path = str(out_dir / f"{p.stem}_obb.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"image": str(p), "width": W, "height": H, "detections": det_rows},
                        f,
                        indent=2,
                    )

            return {
                "image": str(p),
                "annotated": annotated_path,
                "json": json_path,
                "counts": counts,
                "detections": det_rows,
            }

        # -------- Segmentation (preferred) OR Detection fallback --------
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(float) if boxes is not None else np.zeros((0, 4))
        cls = (
            boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.zeros((0,), dtype=int)
        )
        conf = (
            boxes.conf.cpu().numpy().astype(float)
            if boxes is not None
            else np.zeros((0,), dtype=float)
        )

        have_masks = (getattr(r, "masks", None) is not None) and (r.masks is not None)
        masks_bin: List[np.ndarray] = []
        if have_masks:
            m_t = r.masks.data.cpu().numpy()  # (N,h,w) float in [0,1]
            for m in m_t:
                m8 = (m * 255).astype(np.uint8)
                m_res = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
                masks_bin.append(m_res)

        # Save annotated
        annotated_path = None
        if self.cfg.save_annotated:
            if have_masks and (self.cfg.render_style in ("auto", "masks_only")):
                ann = self._render_masks_only(
                    orig,
                    masks_bin,
                    cls.tolist(),
                    names_map,
                    alpha=self.cfg.mask_alpha,
                    put_labels=self.cfg.draw_labels,
                )
            else:
                ann = r.plot()  # default Ultralytics rendering
            annotated_path = str(
                out_dir / (f"{p.stem}_seg.png" if have_masks else f"{p.stem}_det.png")
            )
            cv2.imwrite(annotated_path, ann)

        # Save mask PNGs
        if have_masks and self.cfg.save_masks:
            (out_dir / "masks").mkdir(parents=True, exist_ok=True)
            for i, mb in enumerate(masks_bin):
                mp = out_dir / "masks" / f"{p.stem}_{i:03d}.png"
                cv2.imwrite(str(mp), mb)

        # Optional crops from boxes
        if self.cfg.save_crops and len(xyxy) > 0:
            (out_dir / "crops").mkdir(parents=True, exist_ok=True)

        # Prepare JSON and rows
        instances_json: List[Dict] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            cid = int(cls[i])
            cname = names_map.get(cid, str(cid))
            conf_i = float(conf[i])
            x1i = max(0, min(W - 1, int(round(x1))))
            y1i = max(0, min(H - 1, int(round(y1))))
            x2i = max(0, min(W - 1, int(round(x2))))
            y2i = max(0, min(H - 1, int(round(y2))))
            bw = max(0, x2i - x1i + 1)
            bh = max(0, y2i - y1i + 1)
            area = float(bw * bh)

            crop_path = None
            if self.cfg.save_crops and bw > 0 and bh > 0:
                crop = orig[y1i : y2i + 1, x1i : x2i + 1]
                if self.cfg.crop_max_size and max(bw, bh) > self.cfg.crop_max_size:
                    scale = self.cfg.crop_max_size / float(max(bw, bh))
                    crop = cv2.resize(
                        crop,
                        (max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                class_dir = out_dir / "crops" / cname
                class_dir.mkdir(parents=True, exist_ok=True)
                crop_path = str(class_dir / f"{p.stem}_{i:03d}.jpg")
                cv2.imwrite(crop_path, crop)

            seg_polys = None
            mask_path = None
            if have_masks and i < len(masks_bin):
                seg_polys = self._contours_from_mask(masks_bin[i])
                mask_path = (
                    str(out_dir / "masks" / f"{p.stem}_{i:03d}.png")
                    if self.cfg.save_masks
                    else None
                )

            det_rows.append(
                {
                    "image": str(p),
                    "width": W,
                    "height": H,
                    "class_id": cid,
                    "class_name": cname,
                    "conf": conf_i,
                    "x1": float(x1i),
                    "y1": float(y1i),
                    "x2": float(x2i),
                    "y2": float(y2i),
                    "area": area,
                    "crop_path": crop_path,
                    "mask_path": mask_path,
                    "segmentation": seg_polys,
                    "keypoints": None,
                }
            )
            counts[cname] = counts.get(cname, 0) + 1

            instances_json.append(
                {
                    "class_id": cid,
                    "class_name": cname,
                    "conf": conf_i,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "segmentation": seg_polys,
                    "mask_path": mask_path,
                }
            )

        if self.cfg.save_json:
            json_name = f"{p.stem}_seg.json" if have_masks else f"{p.stem}_detections.json"
            json_path = str(out_dir / json_name)
            with open(json_path, "w") as f:
                json.dump(
                    {"image": str(p), "width": W, "height": H, "instances": instances_json},
                    f,
                    indent=2,
                )

        return {
            "image": str(p),
            "annotated": annotated_path,
            "json": json_path,
            "counts": counts,
            "detections": det_rows,
        }
