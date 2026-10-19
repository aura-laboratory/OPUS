"""A Gymnasium team environment with padded UAV slots and a cooperative reward."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from opus_rl.configs.models import ScenarioConfig
from opus_rl.simulation.camera import scan
from opus_rl.simulation.coverage import fuse, metrics, reward, update_coverage
from opus_rl.simulation.geometry import Area
from opus_rl.simulation.motion import (
    AnalyticalBackend, Station, UAV, minimum_separation, profile, stopped_distance, wrap_angle,
)
from opus_rl.simulation.scheduler import RechargeScheduler

STATUSES = ("coverage", "takeoff", "recharge-transit", "landing", "charging", "crashed")
TYPE_FIELDS = (
    "max_horizontal_speed_mps", "max_climb_speed_mps", "max_descent_speed_mps",
    "max_horizontal_accel_mps2", "max_vertical_accel_mps2", "max_yaw_rate_degps",
    "max_yaw_accel_degps2", "max_gimbal_pitch_rate_degps", "battery_capacity_Wh",
    "P_airborne_base_W", "k_vxy", "k_vz_up", "k_vz_down", "k_acc", "collision_radius_m",
    "camera_width_px", "camera_height_px", "camera_fov_deg", "takeoff_energy_Wh", "landing_energy_Wh",
)
TYPE_SCALES = np.array([15, 6, 6, 4, 3, 200, 90, 100, 77, 121.6, 0.05, 8, 3, 1, 1,
                        5280, 3956, 84, 0.3, 0.25])
OWN_DIM = 20 + len(STATUSES) + len(TYPE_FIELDS) + 3
CONTEXT_DIM = 7


class OpusTeamEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(self, config=None, render_mode=None, motion_backend=None):
        self.config = (config or ScenarioConfig()).model_copy(deep=True)
        self.area = Area(self.config)
        self.backend = motion_backend or AnalyticalBackend()
        self.scheduler = RechargeScheduler(self.backend, self.area, self.config.delta_t)
        self.n_max = sum(t.available_count for t in self.config.uav_types)
        self.s_max = sum(t.available_count for t in self.config.station_types)
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1, 1, (self.n_max, 5), np.float32)
        other = max(1, self.n_max - 1)  # Gymnasium MultiBinary cannot contain a zero-length dimension.
        self.observation_space = spaces.Dict({
            "uav_own": spaces.Box(-1, 1, (self.n_max, OWN_DIM), np.float32),
            "other_uavs": spaces.Box(-1, 1, (self.n_max, other, OWN_DIM), np.float32),
            "other_uav_mask": spaces.MultiBinary((self.n_max, other)),
            "map": spaces.Box(0, 1, (self.n_max, 3, self.config.observation_size,
                                     self.config.observation_size), np.float32),
            "active_mask": spaces.MultiBinary(self.n_max),
            "actor_training_mask": spaces.MultiBinary(self.n_max),
            "mission_context": spaces.Box(0, 1, (self.n_max, CONTEXT_DIM), np.float32),
        })
        self.uavs, self.stations = [], []
        self.coverage = np.zeros(self.area.mask.shape, np.float32)
        self.last_gsd = np.full_like(self.coverage, np.nan)
        self.step_index = 0
        self.finished = False
        self.last_info = {}
        self.return_estimates = np.zeros(self.n_max)

    def _inventory(self, types, mode, minimum, maximum, at_least=1):
        if mode == "fixed":
            return [j for j, t in enumerate(types) for _ in range(t.fixed_count)]
        pool = np.array([j for j, t in enumerate(types) for _ in range(t.available_count)])
        count = int(self.np_random.integers(max(minimum, at_least), (maximum or len(pool)) + 1))
        return self.np_random.choice(pool, size=count, replace=False).tolist()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.config
        inventory = [j for j, kind in enumerate(cfg.uav_types) for _ in range(kind.available_count)]
        if cfg.fleet_mode == "variable":
            count = int(self.np_random.integers(cfg.min_uavs, (cfg.max_uavs or self.n_max) + 1))
            slots = sorted(self.np_random.choice(self.n_max, count, replace=False).tolist())
        else:
            slots, offset = [], 0
            for kind in cfg.uav_types:
                slots.extend(range(offset, offset + kind.fixed_count))
                offset += kind.available_count
        fleet = [inventory[j] for j in slots]
        required_docks = max((cfg.uav_launch_docks[j] + 1 for j in slots), default=1) if cfg.uav_launch_docks else 1
        dock_types = self._inventory(cfg.station_types, cfg.station_mode, cfg.min_stations,
                                     cfg.max_stations, required_docks)
        if cfg.station_mode == "fixed" and cfg.station_position_types is not None:
            dock_types = cfg.station_position_types
        separation = 2 * max(t.collision_radius_m for t in cfg.uav_types) + 0.5
        positions = self.area.station_positions
        if cfg.station_mode == "variable":
            positions = self.area.random_positions(self.np_random, len(dock_types), separation)
        if len(positions) > 1:
            for j, xy in enumerate(positions):
                if any(np.linalg.norm(xy - previous) < separation for previous in positions[:j]):
                    raise ValueError(f"Stations must be at least {separation:g} m apart for this inventory.")
        self.stations = [Station(np.r_[xy, 0.0], kind, cfg.station_types[kind].charge_power_W,
                                 max(t.collision_radius_m for t in cfg.uav_types))
                         for xy, kind in zip(positions, dock_types)]
        assignment = [cfg.uav_launch_docks[j] if cfg.uav_launch_docks else j % len(self.stations) for j in slots]
        self.uavs = []
        for j, (kind, station_index) in enumerate(zip(fleet, assignment)):
            spec = cfg.uav_types[kind]
            self.uavs.append(UAV(spec, kind, self.stations[station_index].position.copy(),
                                 spec.battery_capacity_Wh, station=int(station_index), inventory_index=slots[j]))
            if self.stations[station_index].occupied is None:
                self.stations[station_index].occupied = j
            else:
                self.uavs[-1].status = "queued"
        self.coverage.fill(0)
        self.last_gsd.fill(np.nan)
        self.step_index, self.finished = 0, False
        self.return_estimates.fill(0)
        self.last_info = self._info(0, {}, np.zeros(self.n_max, np.int8), np.zeros(self.n_max), [], 0, 0)
        return self._observation(), self.last_info.copy()

    def project_action(self, uav, action):
        cfg, spec = self.config, uav.spec
        dt = cfg.delta_t
        clipped = np.clip(np.asarray(action, float), -1, 1)
        displacement = clipped[:3].copy()
        length = np.linalg.norm(displacement)
        if length > 1:
            displacement /= length
        radii = np.array([stopped_distance(spec.max_horizontal_speed_mps,
                                          spec.max_horizontal_accel_mps2, dt)] * 2 +
                        [stopped_distance(spec.max_climb_speed_mps if displacement[2] >= 0
                                          else spec.max_descent_speed_mps, spec.max_vertical_accel_mps2, dt)])
        target = uav.position + displacement * radii
        target[2] = np.clip(target[2], cfg.min_altitude_m, cfg.max_altitude_m)
        target[:2] = self.area.constrain_segment(uav.position[:2], target[:2])
        target = self.backend.reachable_target(uav.position, target, spec, dt)
        yaw = clipped[3] * np.pi
        maximum_yaw = np.deg2rad(stopped_distance(spec.max_yaw_rate_degps,
                                                spec.max_yaw_accel_degps2, dt))
        yaw = wrap_angle(uav.yaw + np.clip(wrap_angle(yaw - uav.yaw), -maximum_yaw, maximum_yaw))
        lower, upper = np.deg2rad([spec.gimbal_pitch_min_deg, spec.gimbal_pitch_max_deg])
        pitch = lower + (clipped[4] + 1) / 2 * (upper - lower)
        gimbal_step = np.deg2rad(spec.max_gimbal_pitch_rate_degps) * dt
        pitch = np.clip(pitch, uav.gimbal - gimbal_step, uav.gimbal + gimbal_step)
        projected = np.r_[(target - uav.position) / radii, yaw / np.pi,
                           2 * (pitch - lower) / (upper - lower) - 1]
        return target, float(yaw), float(pitch), np.clip(projected, -1, 1)

    def _controlled_motion(self, index, uav):
        cfg, spec = self.config, uav.spec
        overhead = 0.0
        if uav.status == "queued":
            dock = self.stations[uav.station]
            if dock.occupied is not None or dock.reserved is not None:
                return None
            if any(other.status not in ("queued", "crashed") and other is not uav
                   and np.linalg.norm(other.position[:2] - dock.position[:2]) <
                   spec.collision_radius_m + other.spec.collision_radius_m + 0.5
                   and other.position[2] <= cfg.min_altitude_m + spec.collision_radius_m + other.spec.collision_radius_m
                   for other in self.uavs):
                return None
            dock.occupied, uav.status = index, "charging"
        if uav.status == "charging":
            if uav.battery < spec.battery_capacity_Wh - 1e-8:
                dock = self.stations[uav.station]
                uav.battery = min(spec.battery_capacity_Wh, uav.battery + dock.charge_power * cfg.delta_t / 3600)
                return None
            dock = self.stations[uav.station]
            dock.occupied = None
            dock.reserved = index
            uav.status = "takeoff"
            overhead = spec.takeoff_energy_Wh
        if uav.status == "takeoff":
            target = uav.position.copy()
            target[2] = cfg.min_altitude_m
        else:
            dock = self.stations[uav.station]
            target = dock.position.copy()
            if np.linalg.norm(uav.position[:2] - target[:2]) <= spec.collision_radius_m:
                uav.status = "landing"
            if np.linalg.norm(target - uav.position) < 1e-7:
                uav.status = "charging"
                dock.reserved, dock.occupied = None, index
                return None
        target = self.backend.reachable_target(uav.position, target, spec, cfg.delta_t)
        movement = self.backend.propagate(uav.position, target, spec, cfg.delta_t)
        movement.overhead_Wh = overhead
        if uav.status in ("landing", "recharge-transit") and target[2] <= 1e-7:
            movement.overhead_Wh += spec.landing_energy_Wh
        return movement

    def _crash(self, index):
        uav = self.uavs[index]
        uav.status = "crashed"
        uav.velocity.fill(0)
        uav.acceleration.fill(0)
        for station in self.stations:
            if station.occupied == index:
                station.occupied = None
            if station.reserved == index:
                station.reserved = None
        uav.station = None

    def step(self, action):
        if self.finished or not self.uavs:
            raise RuntimeError("Reset the environment before stepping a new episode.")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n_max, 5) or not np.isfinite(action).all():
            raise ValueError(f"Expected a finite action with shape ({self.n_max}, 5).")
        cfg, dt = self.config, self.config.delta_t
        actual_mask = np.zeros(self.n_max, np.int8)
        energies, projections = np.zeros(self.n_max), np.zeros(self.n_max)
        movements, orientations, denied = {}, {}, 0
        for index in sorted(range(len(self.uavs)), key=lambda j: self.uavs[j].battery /
                            self.uavs[j].spec.battery_capacity_Wh):
            uav = self.uavs[index]
            if uav.status == "crashed":
                continue
            if uav.status == "coverage":
                target, yaw, pitch, projected = self.project_action(uav, action[index])
                movement = self.backend.propagate(uav.position, target, uav.spec, dt)
                overridden, estimate, unavailable = self.scheduler.decide(index, uav, movement, self.stations)
                self.return_estimates[index] = estimate if np.isfinite(estimate) else uav.spec.battery_capacity_Wh
                denied += int(unavailable)
                if not overridden:
                    actual_mask[index] = 1
                    projections[index] = np.clip(np.linalg.norm(action[index] - projected) / (2 * np.sqrt(5)), 0, 1)
                    movements[index], orientations[index] = movement, (yaw, pitch)
                    continue
            movement = self._controlled_motion(index, uav)
            if movement is not None:
                movements[index] = movement
        # Include stationary grounded UAVs in synchronized separation checks.
        trajectories = {j: movements.get(j) or self.backend.propagate(u.position, u.position, u.spec, dt)
                        for j, u in enumerate(self.uavs) if u.status not in ("crashed", "queued")}
        collision_times, collisions = {}, []
        for i in trajectories:
            for j in trajectories:
                if j <= i:
                    continue
                distance, time = minimum_separation(trajectories[i], trajectories[j])
                if distance < self.uavs[i].spec.collision_radius_m + self.uavs[j].spec.collision_radius_m:
                    collisions.append([i, j])
                    for k in (i, j):
                        collision_times[k] = min(collision_times.get(k, dt), time)
        for j, movement in movements.items():
            uav = self.uavs[j]
            until = collision_times.get(j, dt)
            depleted = movement.energy(until) >= uav.battery
            if depleted:
                lo, hi = 0.0, until
                for _ in range(40):  # Root of the analytic energy integral, not dynamics integration.
                    mid = (lo + hi) / 2
                    if movement.energy(mid) < uav.battery:
                        lo = mid
                    else:
                        hi = mid
                until = hi
            energies[j] = min(uav.battery, movement.energy(until))
            uav.battery = max(0, uav.battery - energies[j])
            uav.position, uav.velocity, uav.acceleration = movement.state(until)
            if depleted or j in collision_times:
                self._crash(j)
                continue
            if j in orientations:
                yaw, pitch = orientations[j]
                # Both targets were projected to reach within ΔT, so their endpoint rates are zero.
                yaw_distance = abs(wrap_angle(yaw - uav.yaw))
                profile(yaw_distance, np.deg2rad(uav.spec.max_yaw_rate_degps),
                        np.deg2rad(uav.spec.max_yaw_accel_degps2), dt)
                uav.yaw, uav.gimbal = yaw, pitch
                uav.yaw_rate = uav.yaw_acceleration = uav.gimbal_rate = 0
            if uav.status == "takeoff" and uav.position[2] >= cfg.min_altitude_m - 1e-7:
                self.stations[uav.station].reserved = None
                uav.station, uav.status = None, "coverage"
            elif uav.status in ("landing", "recharge-transit") and uav.position[2] <= 1e-7:
                dock = self.stations[uav.station]
                dock.occupied, dock.reserved = j, None
                uav.status = "charging"
        for j in collision_times:
            if j not in movements:
                self._crash(j)
        instantaneous = np.zeros_like(self.coverage)
        camera_stats = {"valid_cells": 0, "invalid_gsd_cells": 0, "mean_gsd": 0.0, "mean_incidence_deg": 0.0}
        gsd_weight = 0
        self.last_gsd.fill(np.nan)
        for uav in self.uavs:
            q, gsd, statistics = scan(self.area, uav, cfg.required_gsd_m)
            best = q > instantaneous
            instantaneous[best], self.last_gsd[best] = q[best], gsd[best]
            for key in ("valid_cells", "invalid_gsd_cells"):
                camera_stats[key] += statistics[key]
            gsd_weight += statistics["valid_cells"]
            for key in ("mean_gsd", "mean_incidence_deg"):
                camera_stats[key] += statistics[key] * statistics["valid_cells"]
        for key in ("mean_gsd", "mean_incidence_deg"):
            camera_stats[key] /= max(1, gsd_weight)
        self.coverage = update_coverage(self.coverage, instantaneous, self.area.mask, cfg.coverage_lambda)
        n0 = len(self.uavs)
        energy_norm = sum(min(1, energies[j] / (2 * u.spec.P_airborne_base_W * dt / 3600))
                          for j, u in enumerate(self.uavs)) / n0
        crashes = sum(u.status == "crashed" for u in self.uavs)
        collided = len({i for pair in collisions for i in pair})
        total, terms = reward(cfg.reward, metrics(self.coverage, self.area.mask), energy_norm,
                              projections.sum() / n0, collided / n0, crashes / n0)
        self.step_index += 1
        terminated, truncated = crashes == n0, self.step_index >= cfg.horizon
        self.finished = terminated or truncated
        self.last_info = self._info(total, terms, actual_mask, energies, collisions, denied,
                                    float(projections.sum()), camera_stats)
        return self._observation(), total, terminated, truncated, self.last_info.copy()

    def _own_features(self, uav):
        spec = uav.spec
        limits = np.array([spec.max_horizontal_speed_mps] * 2 + [max(spec.max_climb_speed_mps,
                                                                            spec.max_descent_speed_mps)])
        accelerations = np.array([spec.max_horizontal_accel_mps2] * 2 + [spec.max_vertical_accel_mps2])
        physical = np.r_[self.area.normalized_xy(uav.position[:2]), uav.position[2] / self.config.max_altitude_m,
                         uav.velocity / limits, uav.acceleration / accelerations,
                         np.sin(uav.yaw), np.cos(uav.yaw), uav.gimbal / np.pi,
                         uav.yaw_rate / np.deg2rad(spec.max_yaw_rate_degps),
                         uav.yaw_acceleration / np.deg2rad(spec.max_yaw_accel_degps2),
                         uav.gimbal_rate / np.deg2rad(spec.max_gimbal_pitch_rate_degps),
                         uav.battery / spec.battery_capacity_Wh, spec.reserve_battery_ratio,
                         float(uav.status == "coverage"), float(uav.status == "crashed"), 1.0]
        # Queued UAVs are staged off the berth; retain checkpoint-compatible observation dimensions.
        status = [float(("charging" if uav.status == "queued" else uav.status) == s) for s in STATUSES]
        properties = np.array([getattr(spec, f) for f in TYPE_FIELDS])
        properties = properties / (properties + TYPE_SCALES)
        camera = [spec.gimbal_pitch_min_deg / 90, spec.gimbal_pitch_max_deg / 90,
                  float(spec.camera_fov_axis == "diagonal")]
        return np.clip(np.r_[physical, status, properties, camera], -1, 1).astype(np.float32)

    def _observation(self):
        own = np.zeros((self.n_max, OWN_DIM), np.float32)
        active = np.zeros(self.n_max, np.int8)
        actor = active.copy()
        for j, uav in enumerate(self.uavs):
            own[j] = self._own_features(uav)
            active[j], actor[j] = 1, int(uav.status == "coverage")
        other = np.zeros(self.observation_space["other_uavs"].shape, np.float32)
        other_mask = np.zeros(self.observation_space["other_uav_mask"].shape, np.int8)
        for j in range(len(self.uavs)):
            indexes = [k for k in range(len(self.uavs)) if k != j]
            if indexes:
                other[j, :len(indexes)] = own[indexes]
                other_mask[j, :len(indexes)] = 1
        cfg = self.config
        context = np.array([cfg.required_gsd_m / (cfg.required_gsd_m + 0.05), cfg.delta_t / (cfg.delta_t + 5),
                            self.area.cell / (self.area.cell + 1), cfg.coverage_lambda,
                            min(1, self.step_index / cfg.horizon), cfg.min_altitude_m / cfg.max_altitude_m,
                            cfg.max_altitude_m / (cfg.max_altitude_m + 100)], np.float32)
        shared = fuse(self.coverage, self.area.mask, cfg.observation_size)
        return {"uav_own": own, "other_uavs": other, "other_uav_mask": other_mask,
                "map": np.broadcast_to(shared, (self.n_max, *shared.shape)).copy(),
                "active_mask": active, "actor_training_mask": actor,
                "mission_context": np.broadcast_to(context, (self.n_max, CONTEXT_DIM)).copy()}

    def central_state(self, observation=None):
        observation = self._observation() if observation is None else observation
        assignment = np.zeros((self.n_max, self.s_max), np.float32)
        for j, u in enumerate(self.uavs):
            if u.station is not None:
                assignment[j, u.station] = 1
        stations = np.zeros((self.s_max, 6), np.float32)
        station_mask = np.zeros(self.s_max, np.float32)
        for j, s in enumerate(self.stations):
            stations[j] = np.r_[self.area.normalized_xy(s.position[:2]), s.charge_power / (s.charge_power + 100),
                                 float(s.occupied is not None), float(s.reserved is not None), s.radius / (s.radius + 1.5)]
            station_mask[j] = 1
        uavs = np.column_stack([observation["uav_own"], assignment,
                                self.return_estimates / np.array([u.spec.battery_capacity_Wh for u in self.uavs] +
                                                                 [1] * (self.n_max - len(self.uavs)))])
        return {"uavs": uavs.astype(np.float32), "active_mask": observation["active_mask"].astype(np.float32),
                "stations": stations, "station_mask": station_mask, "map": observation["map"][0].copy(),
                "context": observation["mission_context"][0].copy()}

    def _info(self, total, terms, actor_mask, energies, collisions, denied, projection, camera=None):
        return {"step": self.step_index, "simulated_seconds": self.step_index * self.config.delta_t,
                "reward": total, "reward_terms": terms, "coverage": metrics(self.coverage, self.area.mask),
                "actor_training_mask": actor_mask.tolist(), "energy_Wh": energies.tolist(),
                "collision_pairs": collisions, "no_station_events": denied, "projection_sum": projection,
                "uav_statuses": [u.status for u in self.uavs], "active_uavs": len(self.uavs),
                "crashed_uavs": sum(u.status == "crashed" for u in self.uavs), "camera": camera or {}}

    def frame(self):
        stride = max(1, self.config.grid_size // 64)
        heatmap = fuse(self.coverage, self.area.mask, self.config.grid_size // stride)
        return {"info": self.last_info, "uavs": [{"id": j, "inventory_index": u.inventory_index,
                  "launch_dock": self.config.uav_launch_docks[u.inventory_index] if self.config.uav_launch_docks
                  else u.inventory_index % len(self.stations), "position": u.position.tolist(),
                  "geographic": self.area.geographic(u.position[None, :2])[0].tolist(),
                  "battery_ratio": u.battery / u.spec.battery_capacity_Wh,
                  "yaw": u.yaw, "gimbal": u.gimbal, "type": u.spec.name, "status": u.status}
                  for j, u in enumerate(self.uavs)],
                "stations": [{"position": s.position.tolist(), "type_index": s.type_index,
                  "geographic": self.area.geographic(s.position[None, :2])[0].tolist(),
                  "occupied": s.occupied, "reserved": s.reserved} for s in self.stations],
                "heatmap": heatmap[1].tolist(), "heatmap_mask": heatmap[0].tolist(),
                "bounds_xy": [self.area.low.tolist(), (self.area.low + self.area.side).tolist()],
                "bounds_wgs84": self.area.geographic(np.array([self.area.low,
                                                               self.area.low + self.area.side])).tolist(),
                "polygon_wgs84": self.area.geographic(self.area.vertices).tolist(),
                "cell_size_m": self.area.cell, "area_m2": self.area.polygon.area}

    def render(self):
        image = np.zeros((*self.coverage.shape, 3), np.uint8)
        image[self.area.mask] = np.array([20, 35, 48], np.uint8)
        image[..., 0] += (self.coverage * 200).astype(np.uint8)
        image[..., 1] += (self.coverage * 120).astype(np.uint8)
        return image[::-1].copy()
