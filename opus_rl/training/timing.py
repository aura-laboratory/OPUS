"""Small wall-clock counters for training phases and simulator subphases."""

import time
from contextlib import contextmanager

import torch


class Timings:
    def __init__(self, device="cpu"):
        self.device = device
        self.rows = {}

    def record(self, phase, seconds):
        row = self.rows.setdefault(phase, {"calls": 0, "seconds": 0.0, "last_seconds": 0.0,
                                           "max_seconds": 0.0})
        row["calls"] += 1
        row["seconds"] += seconds
        row["last_seconds"] = seconds
        row["max_seconds"] = max(row["max_seconds"], seconds)

    @contextmanager
    def measure(self, phase, gpu=False):
        synchronize = gpu and str(self.device).startswith("cuda")
        if synchronize:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize:
                torch.cuda.synchronize(self.device)
            self.record(phase, time.perf_counter() - started)

    def snapshot(self):
        total = sum(row["seconds"] for name, row in self.rows.items() if "." not in name)
        return {name: {**row, "mean_ms": 1000 * row["seconds"] / row["calls"],
                       "percent": 100 * row["seconds"] / max(total, 1e-9)}
                for name, row in self.rows.items()}
