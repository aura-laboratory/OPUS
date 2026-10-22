"""Pinhole camera; local GSD is the largest image-to-ground Jacobian singular value."""

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


def footprint_window(area, position, optical, right, down, focal, width, height):
    """Clip the ground FOV to the raster bounds, including horizon-crossing views.

    Only the small polygon is transformed here. Expanded clipping and a one-cell
    margin give conservative candidates; the original exact FOV test runs later.
    """
    low = area.low - position[:2]
    high = low + area.side
    polygon = np.array([[low[0], low[1]], [high[0], low[1]],
                        [high[0], high[1]], [low[0], high[1]]])
    planes = [(optical, 1e-8), (width / 2 * optical + focal * right, 0),
              (width / 2 * optical - focal * right, 0),
              (height / 2 * optical + focal * down, 0),
              (height / 2 * optical - focal * down, 0)]
    whole_visible = True
    for normal, threshold in planes:
        values = polygon @ normal[:2] - position[2] * normal[2] - threshold
        tolerance = 64 * np.finfo(float).eps * max(1, float(np.max(np.abs(values))))
        if np.all(values >= 0):
            whole_visible = whole_visible and bool(np.all(values > tolerance))
            continue
        whole_visible = False
        values += tolerance
        if np.all(values < 0):
            return None
        clipped = []
        for index, end in enumerate(polygon):
            start, before, after = polygon[index - 1], values[index - 1], values[index]
            if (before >= 0) != (after >= 0):
                clipped.append(start + before / (before - after) * (end - start))
            if after >= 0:
                clipped.append(end)
        polygon = np.asarray(clipped)
    bounds_low = polygon.min(axis=0) + position[:2]
    bounds_high = polygon.max(axis=0) + position[:2]
    first = np.maximum(0, np.floor((bounds_low - area.low) / area.cell - 0.5).astype(int) - 1)
    last = np.minimum(area.size, np.ceil((bounds_high - area.low) / area.cell - 0.5).astype(int) + 2)
    return int(first[1]), int(last[1]), int(first[0]), int(last[0]), whole_visible


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
    window = footprint_window(area, uav.position, optical, right, down, focal, width, height)
    mark('footprint')
    if window is None:
        return quality, gsd_map, {"valid_cells": 0, "invalid_gsd_cells": 0, "mean_gsd": 0.0,
                                  "mean_incidence_deg": 0.0, "timings": phases}
    row0, row1, col0, col1, whole_visible = window
    # Gathering almost every cell is slower than contiguous arithmetic. Keep a
    # dense window for nearly solid, wide views; sparse/small views use AoI cells.
    dense = area.aoi_fraction > 0.95 and (row1 - row0) * (col1 - col0) > area.size**2 * 0.8
    if dense:
        dx = area.x[row0:row1, col0:col1] - uav.position[0]
        dy = area.y[row0:row1, col0:col1] - uav.position[1]
    else:
        rows, cols = np.nonzero(area.mask[row0:row1, col0:col1])
        dx = area.x_axis[cols + col0] - uav.position[0]
        dy = area.y_axis[rows + row0] - uav.position[1]
    z = -uav.position[2]
    mark('candidates')
    depth = dx * optical[0] + dy * optical[1] + z * optical[2]
    camera_x = dx * right[0] + dy * right[1]
    camera_y = dx * down[0] + dy * down[1] + z * down[2]
    mark("projection")
    if dense:
        visible = area.mask[row0:row1, col0:col1]
        if not whole_visible:
            visible = visible & (depth > 1e-8)
            visible &= np.abs(focal * camera_x) <= width / 2 * depth
            visible &= np.abs(focal * camera_y) <= height / 2 * depth
        local_indexes = np.nonzero(visible)
        indexes = (local_indexes[0] + row0 if row0 else local_indexes[0],
                   local_indexes[1] + col0 if col0 else local_indexes[1])
        depth_v, x_v, y_v = depth[local_indexes], camera_x[local_indexes], camera_y[local_indexes]
    elif whole_visible:
        indexes = rows + row0, cols + col0
        depth_v, x_v, y_v = depth, camera_x, camera_y
    else:
        visible = depth > 1e-8
        visible &= (np.abs(focal * camera_x) <= width / 2 * depth)
        visible &= (np.abs(focal * camera_y) <= height / 2 * depth)
        indexes = rows[visible] + row0, cols[visible] + col0
        depth_v, x_v, y_v = depth[visible], camera_x[visible], camera_y[visible]
    mark("visibility")
    if not len(indexes[0]):
        return quality, gsd_map, {"valid_cells": 0, "invalid_gsd_cells": 0, "mean_gsd": 0.0,
                                  "mean_incidence_deg": 0.0, "timings": phases}
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
