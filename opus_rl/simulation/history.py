"""Bounded visual history; never used to restore or alter a simulator."""

from collections import Counter, OrderedDict
from uuid import uuid4
import zlib

import numpy as np

from .trajectory import Trajectories


class VisualHistory:
    def __init__(self, limit=10001):
        self.limit = limit
        self.clear()

    def clear(self):
        self.identity = uuid4().hex
        self.records = OrderedDict()
        self.frames = OrderedDict()
        self.episodes = Counter()
        self.trails = {}

    def append(self, step, episode, seconds, telemetry, frame=None):
        if step not in self.records:
            self.episodes[episode] += 1
            self.trails.setdefault(episode, Trajectories()).record_samples(telemetry["uavs"], step)
        self.records[step] = {"step": step, "episode": episode, "seconds": seconds, **telemetry}
        if frame is not None:
            heat = np.asarray(frame["heatmap"], float)
            mask = np.asarray(frame["heatmap_mask"], np.uint8)
            self.frames[step] = {
                "base": {key: value for key, value in frame.items() if key not in (
                    "info", "uavs", "heatmap", "heatmap_mask", "trails", "timeline")},
                "size": heat.shape[0],
                "heat": zlib.compress(np.clip(np.rint(heat * 255), 0, 255).astype(np.uint8).tobytes()),
                "mask": zlib.compress(np.packbits(mask).tobytes())}
        while len(self.records) > self.limit:
            _, old = self.records.popitem(last=False)
            self.episodes[old["episode"]] -= 1
            if not self.episodes[old["episode"]]:
                del self.trails[old["episode"]]
                del self.episodes[old["episode"]]
        if self.records:
            first = next(iter(self.records))
            while len(self.frames) > 1:
                iterator = iter(self.frames)
                next(iterator)
                if next(iterator) > first:
                    break
                self.frames.popitem(last=False)

    def metadata(self):
        if not self.records:
            return {"id": self.identity, "first_step": 0, "last_step": 0}
        first, last = self.records[next(iter(self.records))], self.records[next(reversed(self.records))]
        return {"id": self.identity, "first_step": first["step"], "last_step": last["step"],
                "first_seconds": first["seconds"], "last_seconds": last["seconds"]}

    def frame(self, step, length_m=0):
        if step not in self.records:
            raise ValueError("This timestamp is outside the recorded visual history.")
        record = self.records[step]
        frame_step = next((key for key in reversed(self.frames) if key <= step), None)
        if frame_step is None:
            raise ValueError("No map frame was recorded for this timestamp.")
        packed = self.frames[frame_step]
        size = packed["size"]
        heat = np.frombuffer(zlib.decompress(packed["heat"]), np.uint8).reshape(size, size) / 255
        mask = np.unpackbits(np.frombuffer(zlib.decompress(packed["mask"]), np.uint8))[:size**2]
        return {**packed["base"], "info": record["info"], "uavs": record["uavs"],
                "episode": record["episode"], "heatmap": heat.tolist(), "heatmap_mask": mask.reshape(size, size).tolist(),
                "trails": self.trails[record["episode"]].snapshot(step, length_m),
                "history_step": step, "history_seconds": record["seconds"], "map_history_step": frame_step,
                "timeline": self.metadata()}
