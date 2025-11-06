from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO  # pip install ultralytics

# -------------------------------
# Config dataclasses
# -------------------------------

Point = Tuple[int, int]


@dataclass
class LineSpec:
    name: str
    p1: Point
    p2: Point
    classes: Optional[Sequence[int]] = None  # if provided, only count these classes


@dataclass
class ZoneSpec:
    name: str
    points: List[Point]
    classes: Optional[Sequence[int]] = None


@dataclass
class AnalyzerConfig:
    # I/O
    source: str | int
    model: str = "yolo11n.pt"  # detect/seg/pose variants supported
    tracker: str = "botsort.yaml"  # or "bytetrack.yaml"
    out_video: str = "outputs/cv/analysis.mp4"
    events_csv: str = "outputs/cv/events.csv"
    tracks_csv: str = "outputs/cv/tracks.csv"
    summary_json: str = "outputs/cv/summary.json"
    log_file: str = "outputs/cv/analysis.log"

    # Inference
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 640
    classes: Optional[Sequence[int]] = None  # class id filter
    device: Optional[str] = None  # 'cpu', 'mps', '0', etc.
    frame_stride: int = 1  # pass-through to YOLO vid_stride

    # Drawing/behavior
    draw_labels: bool = True
    draw_masks: bool = True
    draw_tracks: bool = True
    show: bool = False
    save_video: bool = True

    # NEW: fine-grained overlay toggles (all optional)
    draw_hud: bool = False  # big black stats panel (disabled for clean video)
    draw_lines: bool = False  # draw line geometry
    label_lines: bool = False  # write "entry_line ..." labels
    draw_zones: bool = False  # draw zone polygons
    label_zones: bool = False  # write zone count labels

    # Analytics
    lines: List[LineSpec] = field(default_factory=list)
    zones: List[ZoneSpec] = field(default_factory=list)
    meter_per_pixel: Optional[float] = None  # if set, compute speed (km/h)
    max_speed_kmh: float = 200.0
    speed_ema_alpha: float = 0.3  # EMA smoothing for speed
    track_lost_patience: int = 30  # frames before considering a track lost (for dwell closeout)

    # Logging
    log_level: str = "INFO"
    log_every_n_frames: int = 100


# -------------------------------
# Geometry helpers
# -------------------------------


def _point_in_poly(pt: Point, poly: List[Point]) -> bool:
    # ray casting algorithm
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        if cond:
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            if x < xint:
                inside = not inside
    return inside


def _side_of_line(p: Point, a: Point, b: Point) -> float:
    # sign of cross((b-a),(p-a))
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


# -------------------------------
# Analyzer
# -------------------------------


class VideoAnalyzer:
    """
    High-level video analytics on Ultralytics YOLO tracking.
    """

    def __init__(self, cfg: AnalyzerConfig):
        self.cfg = cfg
        for p in [cfg.out_video, cfg.events_csv, cfg.tracks_csv, cfg.summary_json, cfg.log_file]:
            d = os.path.dirname(p) or "."
            os.makedirs(d, exist_ok=True)

        self._init_logger()
        self.log.info("Initializing VideoAnalyzer", extra={"cfg": cfg.__dict__})

        self.model = YOLO(cfg.model)

        # dynamic state
        self.fps = 30.0
        self.frame_w: Optional[int] = None
        self.frame_h: Optional[int] = None
        self.total_frames: Optional[int] = None
        self.frame_idx = -1
        self.writer = None
        self.needs_vis = bool(cfg.save_video or cfg.show)

        # analytics state
        self.track_state: Dict[int, Dict[str, float]] = (
            {}
        )  # id -> {cx,cy, speed_kmh, speed_kmh_ema, last_seen_frame, last_side_<line>}
        self.unique_ids_by_class: Dict[int, set] = {}
        self.zone_counts: Dict[str, int] = {}
        self.zone_seen_once: Dict[str, set] = {}  # legacy "count once" behavior
        self.zone_state: Dict[str, Dict[int, Dict[str, int]]] = (
            {}
        )  # zone -> track_id -> {"inside":0/1, "enter_frame":int}

        self.line_counts: Dict[str, int] = {}
        self.line_counts_dir: Dict[str, Dict[str, int]] = {}  # {line: {"A->B": n, "B->A": n}}

        for z in cfg.zones:
            self.zone_counts[z.name] = 0
            self.zone_seen_once[z.name] = set()
            self.zone_state[z.name] = {}
        for ln in cfg.lines:
            self.line_counts[ln.name] = 0
            self.line_counts_dir[ln.name] = {"A->B": 0, "B->A": 0}

        # data sinks
        self.events_rows: List[Dict] = []
        self.tracks_rows: List[Dict] = []

        # class name cache (ultralytics result provides names per frame too)
        self.class_names: Dict[int, str] = {}

    # --------------- Logging ---------------

    def _init_logger(self):
        self.log = logging.getLogger("airoad.cv")
        # Avoid adding multiple handlers on repeated runs
        if not self.log.handlers:
            level = getattr(logging, self.cfg.log_level.upper(), logging.INFO)
            self.log.setLevel(level)
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            # Console
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(fmt)
            self.log.addHandler(ch)
            # File
            fh = logging.FileHandler(self.cfg.log_file)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            self.log.addHandler(fh)

    # --------------- Meta ---------------

    def _open_cap_for_meta(self):
        # try to get FPS/size and total frames
        cap = cv2.VideoCapture(self.cfg.source)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.frame_w, self.frame_h = w, h
            self.fps = float(fps) if fps and not math.isnan(fps) else 30.0
            self.total_frames = count if count > 0 else None
            self.log.info(f"Video meta: {w}x{h}@{self.fps:.2f} fps, frames={self.total_frames}")
        else:
            # fallback defaults
            self.frame_w, self.frame_h, self.fps = 1280, 720, 30.0
            self.total_frames = None
            self.log.warning("Could not open source for metadata; defaulting to 1280x720@30fps")
        cap.release()

    def _init_writer(self):
        if not self.cfg.save_video:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.cfg.out_video, fourcc, self.fps, (self.frame_w, self.frame_h)
        )
        if not self.writer.isOpened():
            self.log.error("Failed to open video writer. Disabling save_video.")
            self.cfg.save_video = False
            self.writer = None

    # --------------- Helpers ---------------

    def _class_name(self, cid: int, names_map: Dict[int, str]) -> str:
        if cid in self.class_names:
            return self.class_names[cid]
        name = names_map.get(cid, str(cid)) if isinstance(names_map, dict) else str(cid)
        self.class_names[cid] = name
        return name

    def _speed_kmh_instant(self, tid: int, cx: float, cy: float) -> Optional[float]:
        if self.cfg.meter_per_pixel is None:
            return None
        prev = self.track_state.get(tid)
        if not prev:
            return None
        dx = cx - prev["cx"]
        dy = cy - prev["cy"]
        meters = math.hypot(dx, dy) * float(self.cfg.meter_per_pixel)
        kmh = meters * self.fps * 3.6
        return min(kmh, self.cfg.max_speed_kmh)

    def _update_speed(self, tid: int, cx: float, cy: float):
        inst = self._speed_kmh_instant(tid, cx, cy)
        if inst is None:
            return None, None
        # EMA smoothing
        alpha = float(self.cfg.speed_ema_alpha)
        prev_ema = self.track_state[tid].get("speed_kmh_ema")
        ema = inst if prev_ema is None else alpha * inst + (1.0 - alpha) * prev_ema
        self.track_state[tid]["speed_kmh"] = inst
        self.track_state[tid]["speed_kmh_ema"] = ema
        return inst, ema

    def _maybe_line_cross(
        self,
        tid: int,
        cls_id: int,
        prev_pt: Optional[Point],
        new_pt: Point,
        names_map: Dict[int, str],
    ):
        if prev_pt is None:
            return
        for ln in self.cfg.lines:
            if ln.classes and cls_id not in ln.classes:
                continue
            s_prev = _side_of_line(prev_pt, ln.p1, ln.p2)
            s_new = _side_of_line(new_pt, ln.p1, ln.p2)
            if s_prev == 0 or s_new == 0:
                continue
            sign_prev = math.copysign(1.0, s_prev)
            sign_new = math.copysign(1.0, s_new)
            key = f"last_side_{ln.name}"

            # Initialize saved side if missing
            last_side = self.track_state.get(tid, {}).get(key)
            if last_side is None:
                self.track_state.setdefault(tid, {})[key] = sign_new
                continue

            # crossing detected when sign flips
            if sign_prev != sign_new and last_side != sign_new:
                direction = "A->B" if sign_prev < 0 and sign_new > 0 else "B->A"
                self.line_counts[ln.name] += 1
                self.line_counts_dir[ln.name][direction] += 1
                self.events_rows.append(
                    {
                        "frame": self.frame_idx,
                        "time_s": self.frame_idx / self.fps,
                        "event": "line_cross",
                        "line": ln.name,
                        "line_dir": direction,
                        "track_id": tid,
                        "class_id": cls_id,
                        "class_name": self._class_name(cls_id, names_map),
                    }
                )
                self.log.debug(f"LineCross {ln.name} {direction} tid={tid} cls={cls_id}")
                self.track_state[tid][key] = sign_new  # update remembered side

    def _update_zone_state(self, tid: int, cid: int, center: Point, names_map: Dict[int, str]):
        # per-zone enter/exit + dwell time
        for z in self.cfg.zones:
            if z.classes and cid not in z.classes:
                continue
            inside = _point_in_poly(center, z.points)
            state = self.zone_state[z.name].get(tid, {"inside": 0, "enter_frame": -1})
            was_inside = bool(state["inside"])

            if inside and not was_inside:
                # ENTER
                self.zone_state[z.name][tid] = {"inside": 1, "enter_frame": self.frame_idx}
                self.zone_counts[z.name] += 1
                self.events_rows.append(
                    {
                        "frame": self.frame_idx,
                        "time_s": self.frame_idx / self.fps,
                        "event": "zone_enter",
                        "zone": z.name,
                        "track_id": tid,
                        "class_id": cid,
                        "class_name": self._class_name(cid, names_map),
                    }
                )
                self.log.debug(f"ZoneEnter {z.name} tid={tid} cls={cid}")

            elif (not inside) and was_inside:
                # EXIT
                enter_f = int(state["enter_frame"])
                dwell_s = max(0.0, (self.frame_idx - enter_f) / self.fps)
                self.zone_state[z.name][tid] = {"inside": 0, "enter_frame": -1}
                self.events_rows.append(
                    {
                        "frame": self.frame_idx,
                        "time_s": self.frame_idx / self.fps,
                        "event": "zone_exit",
                        "zone": z.name,
                        "track_id": tid,
                        "class_id": cid,
                        "class_name": self._class_name(cid, names_map),
                        "zone_dwell_s": dwell_s,
                    }
                )
                self.log.debug(f"ZoneExit {z.name} tid={tid} dwell={dwell_s:.2f}s")

    def _flush_lost_tracks(self, alive_ids: set, names_map: Dict[int, str]):
        # close open zone dwell when a track disappears beyond patience
        patience = int(self.cfg.track_lost_patience)
        cutoff = self.frame_idx - patience
        for z in self.cfg.zones:
            zmap = self.zone_state[z.name]
            to_close: List[int] = []
            for tid, state in zmap.items():
                if state.get("inside") == 1:
                    last_seen = int(self.track_state.get(tid, {}).get("last_seen_frame", -(10**9)))
                    if last_seen <= cutoff and tid not in alive_ids:
                        to_close.append(tid)
            for tid in to_close:
                enter_f = int(zmap[tid]["enter_frame"])
                dwell_s = max(
                    0.0,
                    (
                        (min(self.frame_idx, enter_f) - enter_f) / self.fps
                        if self.frame_idx < enter_f
                        else (self.frame_idx - enter_f) / self.fps
                    ),
                )
                # mark as exited
                zmap[tid] = {"inside": 0, "enter_frame": -1}
                # class may be unknown here; best-effort
                cid = int(self.track_state.get(tid, {}).get("class_id", -1))
                self.events_rows.append(
                    {
                        "frame": self.frame_idx,
                        "time_s": self.frame_idx / self.fps,
                        "event": "zone_exit_lost",
                        "zone": z.name,
                        "track_id": tid,
                        "class_id": cid,
                        "class_name": self._class_name(cid, names_map) if cid >= 0 else str(cid),
                        "zone_dwell_s": dwell_s,
                    }
                )
                self.log.debug(f"ZoneExitLost {z.name} tid={tid} dwell={dwell_s:.2f}s")

    def _draw_overlays(self, frame: np.ndarray):
        # Lines
        if self.cfg.draw_lines:
            for ln in self.cfg.lines:
                cv2.line(frame, ln.p1, ln.p2, (0, 255, 255), 2)
                if self.cfg.label_lines:
                    lbl = f"{ln.name}: {self.line_counts[ln.name]} (A->B {self.line_counts_dir[ln.name]['A->B']}, B->A {self.line_counts_dir[ln.name]['B->A']})"
                    cv2.putText(
                        frame,
                        lbl,
                        (ln.p1[0], max(15, ln.p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

        # Zones
        if self.cfg.draw_zones:
            for z in self.cfg.zones:
                pts = np.array(z.points, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 140, 0), thickness=2)
                if self.cfg.label_zones:
                    cX = int(np.mean(pts[:, 0]))
                    cY = int(np.mean(pts[:, 1]))
                    cv2.putText(
                        frame,
                        f"{z.name}:{self.zone_counts[z.name]}",
                        (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 140, 0),
                        2,
                    )

        # HUD (disabled by default)
        if self.cfg.draw_hud:
            y = 28
            cv2.rectangle(
                frame, (5, 5), (420, 5 + 22 * (len(self.unique_ids_by_class) + 3)), (0, 0, 0), -1
            )
            cv2.putText(
                frame,
                f"FPS:{self.fps:.1f}  frame:{self.frame_idx}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            y += 20
            for cid in sorted(self.unique_ids_by_class):
                cv2.putText(
                    frame,
                    f"class[{cid}] unique:{len(self.unique_ids_by_class[cid])}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                y += 20

    # --------------- Main ---------------

    def run(self) -> Dict:
        start_time = time.time()
        self._open_cap_for_meta()
        self._init_writer()

        # Tracking stream
        self.log.info("Starting tracking stream...")
        results_gen = self.model.track(
            source=self.cfg.source,
            stream=True,
            tracker=self.cfg.tracker,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=list(self.cfg.classes) if self.cfg.classes else None,
            device=self.cfg.device,
            verbose=False,
            show=False,
            vid_stride=int(self.cfg.frame_stride),
            persist=True,  # keep tracker state if the source restarts
        )

        last_log_frame = 0

        for self.frame_idx, r in enumerate(results_gen):
            frame = r.orig_img  # BGR
            names_map = r.names if hasattr(r, "names") else {}
            need_vis = self.needs_vis

            annotated = r.plot() if need_vis else frame

            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                if need_vis:
                    self._draw_overlays(annotated)
                    if self.writer:
                        self.writer.write(annotated)
                    if self.cfg.show:
                        cv2.imshow("analysis", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                continue

            has_ids = hasattr(boxes, "id") and boxes.id is not None
            ids = (
                boxes.id.cpu().numpy().astype(int)
                if has_ids
                else np.full((len(boxes),), -1, dtype=int)
            )
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy().astype(float)

            alive_ids = set()

            # per-detection processing
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                center = (int(cx), int(cy))
                cid = int(cls[i])
                cname = self._class_name(cid, names_map)
                tid = int(ids[i]) if i < len(ids) else -1

                if tid >= 0:
                    alive_ids.add(tid)
                    self.unique_ids_by_class.setdefault(cid, set()).add(tid)
                    self.track_state.setdefault(tid, {})
                    self.track_state[tid]["last_seen_frame"] = self.frame_idx
                    self.track_state[tid]["class_id"] = cid

                # Speed
                prev_pt = None
                if tid in self.track_state and "cx" in self.track_state[tid]:
                    prev_pt = (int(self.track_state[tid]["cx"]), int(self.track_state[tid]["cy"]))
                inst_spd, ema_spd = self._update_speed(tid, cx, cy) if tid >= 0 else (None, None)

                # remember current position
                if tid >= 0:
                    self.track_state[tid]["cx"] = cx
                    self.track_state[tid]["cy"] = cy

                # Line crossing (directional)
                if tid >= 0:
                    self._maybe_line_cross(tid, cid, prev_pt, center, names_map)

                # Zone enter/exit + dwell
                if tid >= 0:
                    self._update_zone_state(tid, cid, center, names_map)

                # annotate ID & speed near box
                if need_vis and self.cfg.draw_tracks and tid >= 0:
                    label = f"ID {tid}"
                    if ema_spd is not None:
                        label += f" | {ema_spd:.1f} km/h"
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), int(max(15, y1 - 8))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (50, 220, 255),
                        2,
                    )

                # per-frame track log
                self.tracks_rows.append(
                    {
                        "frame": self.frame_idx,
                        "time_s": self.frame_idx / self.fps,
                        "track_id": tid,
                        "class_id": cid,
                        "class_name": cname,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                        "conf": float(conf[i]),
                        "speed_kmh": float(inst_spd) if inst_spd is not None else None,
                        "speed_kmh_ema": float(ema_spd) if ema_spd is not None else None,
                    }
                )

            # close open dwell for lost tracks
            self._flush_lost_tracks(alive_ids, names_map)

            # overlays (lines/zones/hud as requested)
            if need_vis:
                self._draw_overlays(annotated)

            # write frame / preview
            if need_vis and self.writer:
                self.writer.write(annotated)
            if need_vis and self.cfg.show:
                cv2.imshow("analysis", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # periodic logs
            if (
                self.cfg.log_every_n_frames > 0
                and (self.frame_idx - last_log_frame) >= self.cfg.log_every_n_frames
            ):
                last_log_frame = self.frame_idx
                self.log.info(
                    f"Progress frame={self.frame_idx} "
                    f"unique_ids={sum(len(s) for s in self.unique_ids_by_class.values())} "
                    f"zones={self.zone_counts} lines={self.line_counts}"
                )

        # finalize
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

        # write CSVs / summary
        pd.DataFrame(self.events_rows).to_csv(self.cfg.events_csv, index=False)
        pd.DataFrame(self.tracks_rows).to_csv(self.cfg.tracks_csv, index=False)

        # dwell aggregates per zone
        dwell_stats: Dict[str, Dict[str, float]] = {}
        for z in self.cfg.zones:
            dwell = [
                float(e.get("zone_dwell_s", 0.0))
                for e in self.events_rows
                if e.get("zone") == z.name and "zone_dwell_s" in e
            ]
            dwell_stats[z.name] = {
                "count_exits": len(dwell),
                "avg_dwell_s": (sum(dwell) / len(dwell)) if dwell else 0.0,
                "max_dwell_s": max(dwell) if dwell else 0.0,
            }

        summary = {
            "frames": self.frame_idx + 1,
            "fps": self.fps,
            "unique_ids_by_class": {int(k): len(v) for k, v in self.unique_ids_by_class.items()},
            "zone_counts": self.zone_counts,
            "zone_dwell_stats": dwell_stats,
            "line_counts": self.line_counts,
            "line_counts_dir": self.line_counts_dir,
        }
        with open(self.cfg.summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        elapsed = time.time() - start_time
        self.log.info(
            f"Done. frames={self.frame_idx+1} elapsed={elapsed:.2f}s fps_app={(self.frame_idx+1)/max(1e-9,elapsed):.2f}"
        )
        return summary
