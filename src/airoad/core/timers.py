from __future__ import annotations

import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def __repr__(self) -> str:
        return f"{self.elapsed:.3f}s"


@contextmanager
def timed(label: str):
    t = Timer()
    yield
    print(f"[timer] {label}: {t.elapsed:.3f}s")
