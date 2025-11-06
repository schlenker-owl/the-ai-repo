from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class ImageAnalyzerConfig:
    # Single image mode (optional; batch runner passes per-image path directly)
    source: str = ""

    # Model / inference
    model: str = "yolo11n.pt"
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 640
    classes: Optional[Sequence[int]] = None
    device: Optional[str] = None

    # Outputs (batch runner will override per-image dirs)
    out_dir: str = "outputs/cv_images/sample"

    # Drawing / saving
    save_annotated: bool = True
    draw_labels: bool = True  # (Ultralytics plot includes labels by default)
    draw_conf: bool = True  # included by Ultralytics plot
    save_json: bool = True
    save_crops: bool = False
    save_masks: bool = False  # requires a *-seg.pt model to produce masks
    crop_max_size: Optional[int] = None  # max edge for crops (resize down if larger)


class ImageAnalyzer:
    """
    Lightweight image analyzer on top of Ultralytics YOLO.

    For each image, writes:
      - annotated image (if enabled)
      - per-image detections JSON
      - optional crops per detection and masks if seg model used
    Also returns a list of detection rows for batch CSV aggregation.
    """

    def __init__(self, cfg: ImageAnalyzerConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model)

    def analyze_image(self, image_path: str, out_dir: str) -> Dict:
        p = Path(image_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = out_dir / "crops"
        masks_dir = out_dir / "masks"

        # one pass predict
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

        det_rows: List[Dict] = []
        counts: Dict[str, int] = {}

        # Extract boxes/cls/conf
        boxes = r.boxes  # may be None
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy().astype(float)

            # Save annotated
            annotated_path = None
            if self.cfg.save_annotated:
                ann = r.plot()  # Ultralytics draws labels/conf by default
                annotated_path = str(out_dir / f"{p.stem}_det.png")
                cv2.imwrite(annotated_path, ann)

            # Save crops/masks optionally
            if self.cfg.save_crops:
                crops_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.save_masks and getattr(r, "masks", None) is not None:
                masks_dir.mkdir(parents=True, exist_ok=True)

            # Prepare mask arrays if present
            mask_arr = None
            if (
                getattr(r, "masks", None) is not None
                and r.masks is not None
                and self.cfg.save_masks
            ):
                # r.masks.data: (N,Hm,Wm) in torch; we will resize to original frame size
                mask_t = r.masks.data.cpu().numpy()  # (N,h,w)
                mask_arr = []
                for m in mask_t:
                    # resize to W,H
                    m8 = (m * 255).astype(np.uint8)
                    m_res = cv2.resize(m8, (W, H), interpolation=cv2.INTER_NEAREST)
                    mask_arr.append(m_res)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cid = int(cls[i])
                cname = names_map.get(cid, str(cid))
                conf_i = float(conf[i])
                # clamp box
                x1i = max(0, min(W - 1, int(round(x1))))
                y1i = max(0, min(H - 1, int(round(y1))))
                x2i = max(0, min(W - 1, int(round(x2))))
                y2i = max(0, min(H - 1, int(round(y2))))
                bw = max(0, x2i - x1i + 1)
                bh = max(0, y2i - y1i + 1)
                area = float(bw * bh)

                crop_path = None
                if self.cfg.save_crops and bw > 0 and bh > 0:
                    crop = orig[y1i : y1i + bh, x1i : x1i + bw]
                    if self.cfg.crop_max_size and max(bw, bh) > self.cfg.crop_max_size:
                        scale = self.cfg.crop_max_size / float(max(bw, bh))
                        new_w = max(1, int(round(bw * scale)))
                        new_h = max(1, int(round(bh * scale)))
                        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    class_dir = crops_dir / cname
                    class_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = str(class_dir / f"{p.stem}_{i:03d}.jpg")
                    cv2.imwrite(crop_path, crop)

                seg_path = None
                if mask_arr is not None and i < len(mask_arr):
                    m_out = mask_arr[i]
                    seg_path = str(masks_dir / f"{p.stem}_{i:03d}.png")
                    cv2.imwrite(seg_path, m_out)

                # Append detection row
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
                        "mask_path": seg_path,
                    }
                )
                counts[cname] = counts.get(cname, 0) + 1
        else:
            annotated_path = None

        # Save per-image JSON
        json_path = None
        if self.cfg.save_json:
            json_path = str(out_dir / f"{p.stem}_detections.json")
            with open(json_path, "w") as f:
                json.dump(
                    {"image": str(p), "width": W, "height": H, "detections": det_rows}, f, indent=2
                )

        return {
            "image": str(p),
            "annotated": annotated_path,
            "json": json_path,
            "counts": counts,
            "detections": det_rows,
        }
