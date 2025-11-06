#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to trained model .pt (e.g., runs/train/exp/weights/best.pt)",
    )
    ap.add_argument("--data", type=str, required=True, help="path to data.yaml")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    model = YOLO(args.model)
    print(f"[eval] model={args.model} data={args.data} split={args.split}")
    metrics = model.val(
        data=args.data, split=args.split, conf=args.conf, iou=args.iou, save_json=True, imgsz=640
    )
    # metrics is a ultralytics.yolo.utils.metrics.Metrics object with .results_dict
    out_dir = Path(args.model).parent  # e.g., .../weights
    out_json = out_dir / f"eval_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump(getattr(metrics, "results_dict", {}), f, indent=2)
    print(f"[eval] wrote {out_json}")


if __name__ == "__main__":
    main()
