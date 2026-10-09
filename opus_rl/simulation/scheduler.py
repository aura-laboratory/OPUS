"""Deterministic one-step look-ahead; reserves a visible dock before return flight."""

import numpy as np


class RechargeScheduler:
    def __init__(self, backend, area, delta_t):
        self.backend, self.area, self.delta_t = backend, area, delta_t

    def estimate_return(self, uav, position, station):
        if not self.area.segment_inside(position[:2], station.position[:2]):
            return np.inf
        # Use exactly the same rest-to-rest chunks as the actual transit controller.
        position = np.asarray(position).copy()
        energy = uav.spec.landing_energy_Wh
        for _ in range(20000):
            if np.linalg.norm(position - station.position) < 1e-7:
                return energy
            target = self.backend.reachable_target(position, station.position, uav.spec, self.delta_t)
            movement = self.backend.propagate(position, target, uav.spec, self.delta_t)
            energy += movement.energy()
            position = target
            if energy > uav.spec.battery_capacity_Wh * 2:
                return np.inf
        return np.inf

    def nearest(self, index, uav, position, stations):
        candidates = sorted(
            (j for j, s in enumerate(stations)
             if s.occupied in (None, index) and s.reserved in (None, index)),
            key=lambda j: np.linalg.norm(position[:2] - stations[j].position[:2])
        )
        for j in candidates:
            energy = self.estimate_return(uav, position, stations[j])
            if np.isfinite(energy):
                return j, energy
        return None, np.inf

    def decide(self, index, uav, proposed, stations):
        station, return_energy = self.nearest(index, uav, uav.position, stations)
        reserve = uav.spec.reserve_battery_ratio * uav.spec.battery_capacity_Wh
        next_station, next_return = self.nearest(index, uav, proposed.state(self.delta_t)[0], stations)
        should_return = uav.battery <= return_energy + reserve
        should_return |= next_station is None or uav.battery - proposed.energy() <= next_return + reserve
        if should_return and station is not None:
            stations[station].reserved = index
            uav.station = station
            uav.status = "recharge-transit"
            return True, return_energy, False
        # No station can be reached safely: retain coverage control, expose the failure, and let depletion crash.
        return False, return_energy, bool(should_return and station is None)
