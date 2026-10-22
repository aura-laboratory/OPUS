"""Distance-limited map trails, separated at episode resets."""

from collections import deque

import numpy as np


class Trajectories:
    def __init__(self):
        self.points = {}

    def record(self, env):
        self.record_samples([{"inventory_index": uav.inventory_index, "status": uav.status,
            "position": uav.position, "geographic": env.area.geographic(uav.position[None, :2])[0]}
            for uav in env.uavs], env.step_index)

    def record_samples(self, uavs, step):
        for uav in uavs:
            if uav["status"] == "queued":
                continue
            history = self.points.setdefault(uav["inventory_index"], deque(maxlen=10001))
            if history and history[-1][0] == step:
                continue
            xy = np.asarray(uav["position"][:2], float).copy()
            distance = history[-1][3] + float(np.linalg.norm(xy - history[-1][2])) if history else 0.0
            history.append((step, list(uav["geographic"]), xy, distance))

    def snapshot(self, step, length_m):
        if length_m is not None and length_m <= 0:
            return []
        result = []
        for index, history in self.points.items():
            points, target, newer = [], None, None
            for recorded, geographic, xy, distance in reversed(history):
                if recorded > step:
                    continue
                if target is None:
                    target = -np.inf if length_m is None else distance - length_m
                if distance <= target and newer is not None:
                    fraction = (target - distance) / (newer[3] - distance)
                    point = (np.asarray(geographic) + fraction * (np.asarray(newer[1]) - geographic)).tolist()
                    if not points or point != points[-1]:
                        points.append(point)
                    break
                if not points or geographic != points[-1]:
                    points.append(geographic)
                newer = (recorded, geographic, xy, distance)
            if points:
                result.append({"inventory_index": index, "points": list(reversed(points))})
        return result
