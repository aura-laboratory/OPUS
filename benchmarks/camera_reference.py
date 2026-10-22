"""Frozen pre-footprint camera implementation for correctness and timing comparisons."""

import time

import numpy as np


def intrinsics(spec):
    width, height = spec.camera_width_px, spec.camera_height_px
    extent = np.hypot(width, height) if spec.camera_fov_axis == "diagonal" else width
    focal = extent / (2 * np.tan(np.deg2rad(spec.camera_fov_deg) / 2))
    return float(focal), width, height


def observation_quality(gsd, incidence_cosine, required_gsd):
    valid = (gsd <= required_gsd) & (gsd > 0) & (incidence_cosine > 0) & np.isfinite(gsd)
    return np.where(valid, (1 - gsd / (2 * required_gsd)) * np.sqrt(np.maximum(0, incidence_cosine)), 0)


def scan(area, uav, required_gsd):
    phases = {}
    started = time.perf_counter()

    def mark(name):
        nonlocal started
        now = time.perf_counter()
        phases[name] = now - started
        started = now

    quality = np.zeros(area.mask.shape, np.float32)
    gsd_map = np.full(area.mask.shape, np.nan, np.float32)
    mark("allocation")
    if uav.position[2] <= 0 or uav.status != "coverage":
        return quality, gsd_map, {"valid_cells": 0, "invalid_gsd_cells": 0, "mean_gsd": 0.0,
                                  "mean_incidence_deg": 0.0, "timings": phases}
    focal, width, height = intrinsics(uav.spec)
    yaw, pitch = uav.yaw, uav.gimbal
    forward = np.array([np.cos(yaw), np.sin(yaw), 0])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0])
    optical = forward * np.cos(pitch) + np.array([0, 0, np.sin(pitch)])
    down = forward * np.sin(pitch) - np.array([0, 0, np.cos(pitch)])
    mark("pose")
    # Dot the ground ray with the orthonormal camera basis without allocating an H×W×3 array.
    dx, dy, z = area.x - uav.position[0], area.y - uav.position[1], -uav.position[2]
    depth = dx * optical[0] + dy * optical[1] + z * optical[2]
    camera_x = dx * right[0] + dy * right[1]
    camera_y = dx * down[0] + dy * down[1] + z * down[2]
    mark("projection")
    visible = area.mask & (depth > 1e-8)
    visible &= (np.abs(focal * camera_x) <= width / 2 * depth)
    visible &= (np.abs(focal * camera_y) <= height / 2 * depth)
    indexes = np.nonzero(visible)
    mark("visibility")
    if not len(indexes[0]):
        return quality, gsd_map, {"valid_cells": 0, "invalid_gsd_cells": 0, "mean_gsd": 0.0,
                                  "mean_incidence_deg": 0.0, "timings": phases}
    depth_v, x_v, y_v = depth[indexes], camera_x[indexes], camera_y[indexes]
    u, v = x_v / depth_v, y_v / depth_v
    ray = optical[None, :] + u[:, None] * right + v[:, None] * down
    ray_z = ray[:, 2]
    h = uav.position[2]
    j_u = -h / focal * (right[:2] * ray_z[:, None] - ray[:, :2] * right[2]) / ray_z[:, None]**2
    j_v = -h / focal * (down[:2] * ray_z[:, None] - ray[:, :2] * down[2]) / ray_z[:, None]**2
    aa, bb, ab = np.sum(j_u**2, axis=1), np.sum(j_v**2, axis=1), np.sum(j_u * j_v, axis=1)
    gsd = np.sqrt(np.maximum(0, (aa + bb + np.sqrt((aa - bb)**2 + 4 * ab**2)) / 2))
    mark("gsd")
    cosine = -ray_z / np.linalg.norm(ray, axis=1)
    q = observation_quality(gsd, cosine, required_gsd)
    mark("quality")
    quality[indexes], gsd_map[indexes] = q, gsd
    valid = q > 0
    statistics = {"valid_cells": int(valid.sum()), "invalid_gsd_cells": int((~valid).sum()),
                             "mean_gsd": float(gsd[valid].mean()) if valid.any() else 0.0,
                             "mean_incidence_deg": float(np.rad2deg(np.arccos(cosine[valid])).mean())
                             if valid.any() else 0.0}
    mark("statistics")
    return quality, gsd_map, {**statistics, "timings": phases}
