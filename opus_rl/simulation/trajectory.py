"""Bounded geographic trails, separated at episode resets."""

from collections import deque


class Trajectories:
    def __init__(self):
        self.points = {}

    def record(self, env):
        for uav in env.uavs:
            if uav.status == "queued":
                continue
            history = self.points.setdefault(uav.inventory_index, deque(maxlen=2001))
            geographic = env.area.geographic(uav.position[None, :2])[0].tolist()
            history.append((env.step_index, geographic))

    def snapshot(self, step, length):
        return [{"inventory_index": index,
                 "points": [point for recorded, point in points if recorded >= step - length]}
                for index, points in self.points.items()] if length else []
