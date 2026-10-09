"""Piecewise analytical straight motion. No physics substeps or real-time waits."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from opus_rl.configs.models import UAVType


def wrap_angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def stopped_distance(speed, acceleration, duration):
    """Maximum rest-to-rest distance within the decision interval."""
    return acceleration * duration**2 / 4 if duration <= 2 * speed / acceleration else speed * (
        duration - speed / acceleration
    )


@dataclass
class Phase:
    duration: float
    velocity: float
    acceleration: float


@dataclass
class Profile:
    phases: list[Phase]
    reach_time: float
    distance: float

    def sample(self, t):
        position = 0.0
        for phase in self.phases:
            elapsed = min(max(t, 0), phase.duration)
            position += phase.velocity * elapsed + 0.5 * phase.acceleration * elapsed**2
            velocity = phase.velocity + phase.acceleration * elapsed
            t -= elapsed
            if t <= 1e-10:
                return position, velocity, phase.acceleration if elapsed < phase.duration - 1e-9 else 0.0
        return position, 0.0, 0.0


def profile(distance, vmax, amax, duration, v0=0.0):
    if distance < v0**2 / (2 * amax) - 1e-8 or v0 < -1e-8 or v0 > vmax + 1e-8:
        raise ValueError("Straight motion needs a feasible waypoint and aligned, nonnegative initial speed.")
    if distance < 1e-10:
        return Profile([Phase(duration, 0, 0)], 0, 0)
    peak = min(vmax, np.sqrt(amax * distance + v0**2 / 2))
    accel_time = max(0, (peak - v0) / amax)
    decel_time = peak / amax
    accel_distance = (v0 + peak) * accel_time / 2
    decel_distance = peak * decel_time / 2
    cruise_time = max(0, (distance - accel_distance - decel_distance) / peak)
    reach_time = accel_time + cruise_time + decel_time
    candidates = [Phase(accel_time, v0, amax), Phase(cruise_time, peak, 0), Phase(decel_time, peak, -amax)]
    phases = []
    left = duration
    for phase in candidates:
        use = min(left, phase.duration)
        if use > 1e-10:
            phases.append(Phase(use, phase.velocity, phase.acceleration))
        left -= use
        if left <= 1e-10:
            break
    if left > 1e-10:
        phases.append(Phase(left, 0, 0))
    return Profile(phases, reach_time, distance)


def direction_limits(direction, spec):
    horizontal, vertical = np.linalg.norm(direction[:2]), abs(direction[2])
    speeds, accelerations = [], []
    if horizontal > 1e-10:
        speeds.append(spec.max_horizontal_speed_mps / horizontal)
        accelerations.append(spec.max_horizontal_accel_mps2 / horizontal)
    if vertical > 1e-10:
        speed = spec.max_climb_speed_mps if direction[2] >= 0 else spec.max_descent_speed_mps
        speeds.append(speed / vertical)
        accelerations.append(spec.max_vertical_accel_mps2 / vertical)
    return min(speeds, default=1), min(accelerations, default=1)


@dataclass
class Trajectory:
    start: np.ndarray
    direction: np.ndarray
    profile: Profile
    spec: UAVType
    duration: float
    overhead_Wh: float = 0

    def state(self, t):
        distance, speed, acceleration = self.profile.sample(min(self.duration, max(0, t)))
        return self.start + self.direction * distance, self.direction * speed, self.direction * acceleration

    def breaks(self):
        return np.cumsum([0] + [p.duration for p in self.profile.phases])

    def polynomial(self, t):
        """Local p + v*u + a*u² coefficients at the start of an interval."""
        position, velocity, _ = self.state(t)
        elapsed = 0.0
        acceleration = 0.0
        for phase in self.profile.phases:
            if elapsed + phase.duration > t + 1e-9:
                acceleration = phase.acceleration
                break
            elapsed += phase.duration
        return position, velocity, self.direction * acceleration / 2

    def energy(self, until=None):
        until = self.duration if until is None else min(self.duration, max(0, until))
        energy = self.spec.P_airborne_base_W * until
        left = until
        horizontal = float(np.dot(self.direction[:2], self.direction[:2]))
        for phase in self.profile.phases:
            t = min(left, phase.duration)
            v, a = phase.velocity, phase.acceleration
            speed_squared_integral = v * v * t + v * a * t * t + a * a * t**3 / 3
            path_length = v * t + a * t * t / 2
            energy += self.spec.k_vxy * horizontal * speed_squared_integral
            coefficient = self.spec.k_vz_up if self.direction[2] >= 0 else self.spec.k_vz_down
            energy += coefficient * abs(self.direction[2]) * path_length
            energy += self.spec.k_acc * a * a * t * float(np.dot(self.direction, self.direction))
            left -= t
            if left <= 1e-10:
                break
        return max(0, energy / 3600 + (self.overhead_Wh if until > 0 else 0))


class MotionBackend(Protocol):
    def propagate(self, start, target, spec, duration, velocity=None) -> Trajectory: ...


class AnalyticalBackend:
    def propagate(self, start, target, spec, duration, velocity=None):
        start, target = np.asarray(start, float), np.asarray(target, float)
        delta = target - start
        distance = float(np.linalg.norm(delta))
        direction = delta / distance if distance > 1e-10 else np.zeros(3)
        vmax, amax = direction_limits(direction, spec)
        velocity = np.zeros(3) if velocity is None else np.asarray(velocity, float)
        v0 = float(np.dot(velocity, direction))
        if np.linalg.norm(velocity - direction * v0) > 1e-7:
            raise ValueError("A noncollinear velocity requires a curved-motion backend; v1 actions end at rest.")
        return Trajectory(start, direction, profile(distance, vmax, amax, duration, v0), spec, duration)

    def reachable_target(self, start, target, spec, duration):
        start, target = np.asarray(start, float), np.asarray(target, float)
        delta = target - start
        distance = np.linalg.norm(delta)
        if distance < 1e-10:
            return start.copy()
        vmax, amax = direction_limits(delta / distance, spec)
        return start + delta * min(1, stopped_distance(vmax, amax, duration) / distance)


def minimum_separation(first, second):
    """Solve the cubic stationary points of squared separation on each common phase."""
    times = sorted(set(np.r_[first.breaks(), second.breaks()]))
    best_distance, best_time = np.inf, 0.0
    for start, end in zip(times[:-1], times[1:]):
        if end - start <= 1e-10:
            continue
        p1, v1, a1 = first.polynomial(start)
        p2, v2, a2 = second.polynomial(start)
        p, v, a = p1 - p2, v1 - v2, a1 - a2
        coefficients = np.array([np.dot(p, v), np.dot(v, v) + 2 * np.dot(p, a),
                                 3 * np.dot(v, a), 2 * np.dot(a, a)])
        candidates = [0, end - start]
        if np.any(np.abs(coefficients) > 1e-14):
            roots = np.polynomial.polynomial.polyroots(np.trim_zeros(coefficients, "b"))
            candidates += [float(r.real) for r in roots if abs(r.imag) < 1e-7 and 0 <= r.real <= end - start]
        for t in candidates:
            distance = np.linalg.norm(p + v * t + a * t * t)
            if distance < best_distance:
                best_distance, best_time = float(distance), start + t
    return best_distance, best_time


@dataclass
class UAV:
    spec: UAVType
    type_index: int
    position: np.ndarray
    battery: float
    status: str = "charging"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    gimbal: float = -np.pi / 2
    yaw_rate: float = 0.0
    yaw_acceleration: float = 0.0
    gimbal_rate: float = 0.0
    station: int | None = None


@dataclass
class Station:
    position: np.ndarray
    type_index: int
    charge_power: float
    radius: float
    occupied: int | None = None
    reserved: int | None = None
