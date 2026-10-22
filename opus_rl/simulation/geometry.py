"""Square cells, metric projection, polygon rasterization and line feasibility."""

import numpy as np
from pyproj import CRS, Transformer
from shapely import contains_xy, intersects_xy
from shapely.geometry import LineString, Point, Polygon

from opus_rl.configs.models import ScenarioConfig


def rasterize_boundary(mask, vertices):
    """Include every grid cell touched by an edge, including both sides of a grid line."""
    size = mask.shape[0]
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        delta = end - start
        breaks = [np.array([0.0, 1.0])]
        for axis in range(2):
            if delta[axis]:
                crossings = np.arange(min(start[axis], end[axis]) + 1, max(start[axis], end[axis]))
                breaks.append((crossings - start[axis]) / delta[axis])
        times = np.unique(np.concatenate(breaks))
        samples = np.concatenate((times, (times[:-1] + times[1:]) / 2))
        points = start + samples[:, None] * delta
        lower = np.floor(points).astype(int)
        adjacent = np.where(np.isclose(points, np.rint(points), atol=1e-9, rtol=0),
                            np.rint(points).astype(int) - 1, lower)
        for x in (lower[:, 0], adjacent[:, 0]):
            for y in (lower[:, 1], adjacent[:, 1]):
                valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
                mask[y[valid], x[valid]] = True


class Area:
    def __init__(self, config: ScenarioConfig):
        lon, lat = config.map_origin_wgs84
        if config.polygon_wgs84:
            coords = np.asarray(config.polygon_wgs84)
            if np.any(np.abs(coords[:, 0]) > 180) or np.any(np.abs(coords[:, 1]) >= 85):
                raise ValueError("Use geographic coordinates between ±180° longitude and ±85° latitude.")
            if np.ptp(coords[:, 0]) > 180:
                raise ValueError("An area crossing the date line must be split before use in v1.")
            lon, lat = coords.mean(axis=0)
        local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
        self.forward = Transformer.from_crs("EPSG:4326", local, always_xy=True)
        self.inverse = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
        coords = np.asarray(config.polygon_xy, dtype=float)
        if config.polygon_wgs84:
            geographic = np.asarray(config.polygon_wgs84)
            coords = np.column_stack(self.forward.transform(geographic[:, 0], geographic[:, 1]))
        polygon = Polygon(coords)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("The area must be a simple polygon with positive area and no self-intersections.")
        low = coords.min(axis=0)
        self.side = float(np.ptp(coords, axis=0).max())
        if self.side < 1 or self.side > 100000:
            raise ValueError("v1 area extent must be between 1 m and 100 km.")
        self.low = low
        self.size = config.grid_size
        self.cell = self.side / self.size
        snapped = low + np.rint((coords - low) / self.cell) * self.cell
        self.polygon = Polygon(snapped)
        if not self.polygon.is_valid or self.polygon.area <= 0:
            raise ValueError("Polygon becomes invalid after grid snapping; increase its size or resolution.")
        self.vertices = snapped
        x = low[0] + (np.arange(self.size) + 0.5) * self.cell
        y = low[1] + (np.arange(self.size) + 0.5) * self.cell
        self.x_axis, self.y_axis = x, y
        self.x, self.y = np.meshgrid(x, y)
        self.mask = intersects_xy(self.polygon, self.x, self.y)
        rasterize_boundary(self.mask, np.rint((snapped - low) / self.cell).astype(int))
        self.aoi_fraction = float(self.mask.mean())
        if not self.mask.any():
            raise ValueError("Area has no cell centers at the selected grid resolution.")
        self.station_positions = np.asarray(config.station_positions_xy, dtype=float)
        if config.station_positions_wgs84 is not None:
            coords = np.asarray(config.station_positions_wgs84, dtype=float).reshape(-1, 2)
            self.station_positions = np.column_stack(self.forward.transform(coords[:, 0], coords[:, 1]))
        if config.station_mode == "fixed":
            if len(self.station_positions) and not intersects_xy(
                self.polygon, self.station_positions[:, 0], self.station_positions[:, 1]
            ).all():
                raise ValueError("Every station must be inside or on the area boundary.")

    def normalized_xy(self, xy):
        return 2 * (np.asarray(xy) - self.low) / self.side - 1

    def geographic(self, xy):
        xy = np.asarray(xy)
        return np.column_stack(self.inverse.transform(xy[..., 0], xy[..., 1]))

    def segment_inside(self, start, end):
        if np.linalg.norm(np.asarray(end) - start) < 1e-10:
            return self.polygon.covers(Point(start))
        return self.polygon.buffer(self.cell * 1e-8).covers(LineString([start, end]))

    def constrain_segment(self, start, end):
        """Stop before the first exit, including concave polygons; preserve straight motion."""
        start, end = np.asarray(start, float), np.asarray(end, float)
        if self.segment_inside(start, end):
            return end.copy()
        delta = end - start
        if np.linalg.norm(delta) < 1e-12:
            return start.copy()
        line = LineString([start, end])
        intersection = line.intersection(self.polygon)
        pieces = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
        distances = []
        for piece in pieces:
            if piece.geom_type == "LineString" and piece.distance(Point(start)) < self.cell * 1e-7:
                distances.append(max(line.project(Point(c)) for c in piece.coords))
        fraction = max(distances, default=0) / line.length
        return start + delta * max(0, fraction - 1e-8)

    def random_positions(self, rng, count, separation):
        positions = []
        for _ in range(20000):
            xy = self.low + rng.random(2) * self.side
            if contains_xy(self.polygon, *xy) and all(np.linalg.norm(xy - p) >= separation for p in positions):
                positions.append(xy)
                if len(positions) == count:
                    return np.asarray(positions)
        raise ValueError("Cannot fit separated stations inside this area; reduce inventory or enlarge it.")
