#!/usr/bin/env python
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

from transformers.trainer_callback import TrainerCallback

"""
src/airoad/sft/callbacks.py

Training callbacks:
- LossLoggerCallback: captures logs for CSV/JSON export
- LivePrinterCallback: pretty per-log prints (loss, lr, ETA, tokens/s)
- TimeLimitCallback: hard stop after N minutes
- PlateauStopCallback: early-stop on train-loss plateau (smoothed)
"""


class LossLoggerCallback(TrainerCallback):
    """Capture on_log events into memory (for CSV/JSON export & loss trend)."""

    def __init__(self, t0: float) -> None:
        self.rows: List[Dict[str, Any]] = []
        self._t0 = t0

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        row: Dict[str, Any] = {
            "step": int(state.global_step),
            "epoch": getattr(state, "epoch", None),
        }
        for k, v in (logs or {}).items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
        row["wall_seconds"] = time.perf_counter() - self._t0
        self.rows.append(row)


class LivePrinterCallback(TrainerCallback):
    """Pretty progress prints (loss, lr, ETA, tokens/s)."""

    def __init__(self, total_steps_est: int, t0: float) -> None:
        self.total = total_steps_est
        self.t0 = t0

    @staticmethod
    def _fmt(x: Optional[float], fmt: str = ".4f") -> str:
        if x is None:
            return "-"
        try:
            return format(x, fmt)
        except Exception:
            return "-"

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None:
            return
        step = int(state.global_step)
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        gn = logs.get("grad_norm")
        ent = logs.get("entropy")
        mta = logs.get("mean_token_accuracy")
        tokens = logs.get("num_tokens")
        elapsed = time.perf_counter() - self.t0

        eta = "-"
        if step > 0 and elapsed > 0 and self.total:
            rate = step / elapsed
            rem = max(0, self.total - step)
            eta = f"{(rem / rate) / 60:.1f}m"

        tps = "-" if (tokens is None or elapsed <= 0) else f"{float(tokens)/elapsed:,.1f}"

        print(
            f"[{step}/{self.total}] "
            f"loss={self._fmt(loss)} lr={self._fmt(lr,'.2e')} "
            f"gn={self._fmt(gn,'.2f')} ent={self._fmt(ent,'.3f')} acc={self._fmt(mta,'.3f')} "
            f"tokens/s={tps} elapsed={elapsed:.1f}s eta={eta}",
            flush=True,
        )


class TimeLimitCallback(TrainerCallback):
    """Hard time limit in minutes (0 = disabled)."""

    def __init__(self, minutes: int, t0: float) -> None:
        self.deadline = t0 + max(0, minutes) * 60
        self.enabled = minutes > 0

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self.enabled and time.perf_counter() >= self.deadline:
            print("[TimeLimit] Time budget reached; stopping now.")
            control.should_training_stop = True
            return control


class PlateauStopCallback(TrainerCallback):
    """
    Early stop on training-loss plateau (uses on_log).
    Patience & window measured in logging events (not raw steps).
    """

    def __init__(
        self,
        patience_logs: int,
        min_delta: float,
        window: int,
        min_steps: int = 0,
        verbose: bool = True,
    ) -> None:
        self.patience = max(1, patience_logs)
        self.min_delta = float(min_delta)
        self.window = max(1, window)
        self.min_steps = int(min_steps)
        self.verbose = verbose
        self.history: deque[float] = deque(maxlen=self.window)
        self.best = float("inf")
        self.stalled = 0

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        self.history.append(loss)
        if len(self.history) < self.window:
            return  # need enough points for smoothing

        smoothed = sum(self.history) / len(self.history)
        improved = (self.best - smoothed) >= self.min_delta

        if improved:
            self.best = smoothed
            self.stalled = 0
        else:
            self.stalled += 1
            if state.global_step >= self.min_steps and self.stalled >= self.patience:
                if self.verbose:
                    print(
                        f"[EarlyStop] Plateau: best={self.best:.4f}, now={smoothed:.4f}, "
                        f"Δ<{self.min_delta} for {self.patience} logs @ step {state.global_step}"
                    )
                control.should_training_stop = True
                return control
