"""Validated, serializable defaults. Distances in metres; camera GSD in m/pixel."""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class UAVType(Config):
    name: str = "Mavic 3E reference"
    available_count: int = Field(2, ge=1, le=128)
    fixed_count: int = Field(2, ge=0, le=128)
    max_horizontal_speed_mps: float = Field(15, gt=0)
    max_climb_speed_mps: float = Field(6, gt=0)
    max_descent_speed_mps: float = Field(6, gt=0)
    max_horizontal_accel_mps2: float = Field(4, gt=0)
    max_vertical_accel_mps2: float = Field(3, gt=0)
    max_yaw_rate_degps: float = Field(200, gt=0)
    max_yaw_accel_degps2: float = Field(90, gt=0)
    max_gimbal_pitch_rate_degps: float = Field(100, gt=0)
    gimbal_pitch_min_deg: float = Field(-90, ge=-90, le=90)
    gimbal_pitch_max_deg: float = Field(35, ge=-90, le=90)
    battery_capacity_Wh: float = Field(77, gt=0)
    initial_battery_ratio: float = Field(1, gt=0, le=1)
    reserve_battery_ratio: float = Field(0.15, ge=0, lt=1)
    P_airborne_base_W: float = Field(121.6, gt=0)
    k_vxy: float = Field(0.05, ge=0)
    k_vz_up: float = Field(8, ge=0)
    k_vz_down: float = Field(3, ge=0)
    k_acc: float = Field(1, ge=0)
    takeoff_energy_Wh: float = Field(0.30, ge=0)
    landing_energy_Wh: float = Field(0.25, ge=0)
    collision_radius_m: float = Field(1, gt=0)
    camera_width_px: int = Field(5280, ge=2)
    camera_height_px: int = Field(3956, ge=2)
    camera_fov_deg: float = Field(84, gt=1, lt=175)
    camera_fov_axis: Literal["diagonal", "horizontal"] = "diagonal"

    @model_validator(mode="after")
    def consistent(self):
        if self.fixed_count > self.available_count:
            raise ValueError("Fixed UAV count cannot exceed available inventory.")
        if self.gimbal_pitch_min_deg >= self.gimbal_pitch_max_deg:
            raise ValueError("Gimbal minimum must be below its maximum.")
        return self


class StationType(Config):
    name: str = "100 W dock"
    available_count: int = Field(2, ge=1, le=128)
    fixed_count: int = Field(2, ge=0, le=128)
    charge_power_W: float = Field(100, gt=0)
    landing_radius_m: float = Field(1.5, gt=0)

    @model_validator(mode="after")
    def consistent(self):
        if self.fixed_count > self.available_count:
            raise ValueError("Fixed station count cannot exceed available inventory.")
        return self


class RewardConfig(Config):
    mean_weight: float = Field(0.5, ge=0)
    fairness_weight: float = Field(0.4, ge=0)
    p10_weight: float = Field(0.1, ge=0)
    energy_weight: float = Field(0.2, ge=0)
    projection_weight: float = Field(0.1, ge=0)
    collision_weight: float = Field(0.3, ge=0)
    crash_weight: float = Field(0.4, ge=0)

    @model_validator(mode="after")
    def coverage_sum(self):
        if not np.isclose(self.mean_weight + self.fairness_weight + self.p10_weight, 1):
            raise ValueError("Coverage weights must sum to one.")
        return self


class ScenarioConfig(Config):
    polygon_xy: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0, 0), (160, 0), (160, 160), (0, 160)]
    )
    polygon_wgs84: list[tuple[float, float]] | None = None  # (longitude, latitude)
    map_origin_wgs84: tuple[float, float] = (22.94, 40.63)
    station_positions_xy: list[tuple[float, float]] = Field(default_factory=lambda: [(20, 20), (140, 140)])
    station_positions_wgs84: list[tuple[float, float]] | None = None
    station_position_types: list[int] | None = None
    uav_types: list[UAVType] = Field(default_factory=lambda: [UAVType()], min_length=1)
    station_types: list[StationType] = Field(default_factory=lambda: [StationType()], min_length=1)
    fleet_mode: Literal["fixed", "variable"] = "fixed"
    station_mode: Literal["fixed", "variable"] = "fixed"
    min_uavs: int = Field(1, ge=1)
    max_uavs: int | None = Field(None, ge=1)
    min_stations: int = Field(1, ge=1)
    max_stations: int | None = Field(None, ge=1)
    delta_t: float = Field(5, gt=0, le=120)
    horizon: int = Field(1000, ge=1, le=1000000)
    coverage_lambda: float = Field(0.9, ge=0, lt=1)
    min_altitude_m: float = Field(20, gt=0)
    max_altitude_m: float = Field(100, gt=0)
    required_gsd_m: float = Field(0.05, gt=0)
    grid_size: int = Field(1024, ge=16, le=2048)
    observation_size: int = Field(128, ge=8, le=256)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="after")
    def consistent(self):
        if len(self.polygon_wgs84 or self.polygon_xy) < 3:
            raise ValueError("The area needs at least three polygon vertices.")
        if self.min_altitude_m >= self.max_altitude_m:
            raise ValueError("Minimum altitude must be below maximum altitude.")
        if self.grid_size % self.observation_size:
            raise ValueError("Grid size must be divisible by observation size.")
        owned = sum(t.available_count for t in self.uav_types)
        stations = sum(t.available_count for t in self.station_types)
        if owned > 128 or stations > 128:
            raise ValueError("v1 supports at most 128 UAV and station inventory slots.")
        max_fleet = self.max_uavs or owned
        max_docks = self.max_stations or stations
        if not self.min_uavs <= max_fleet <= owned or not self.min_stations <= max_docks <= stations:
            raise ValueError("Variable count bounds must fit the available inventory.")
        fixed_fleet = sum(t.fixed_count for t in self.uav_types)
        fixed_docks = sum(t.fixed_count for t in self.station_types)
        fleet_max = fixed_fleet if self.fleet_mode == "fixed" else max_fleet
        docks_max = fixed_docks if self.station_mode == "fixed" else max_docks
        if fleet_max < 1 or docks_max < fleet_max:
            raise ValueError("Each active UAV needs a distinct initial station.")
        if self.station_mode == "fixed":
            positions = self.station_positions_wgs84
            if positions is None:
                positions = self.station_positions_xy
            if len(positions) != fixed_docks:
                raise ValueError("Place exactly one station position per fixed active station.")
            if self.station_position_types is not None:
                if len(self.station_position_types) != fixed_docks:
                    raise ValueError("Each station position needs a type index.")
                for j, station_type in enumerate(self.station_types):
                    if self.station_position_types.count(j) != station_type.fixed_count:
                        raise ValueError("Placed station types must match the selected fixed inventory.")
        return self


class TrainConfig(Config):
    total_steps: int = Field(10000, ge=1, le=100000000)
    num_envs: int = Field(2, ge=1, le=32)
    rollout_steps: int = Field(128, ge=1, le=4096)
    learning_rate: float = Field(3e-4, gt=0, le=0.1)
    gamma: float = Field(0.99, gt=0, le=1)
    gae_lambda: float = Field(0.95, ge=0, le=1)
    clip_range: float = Field(0.2, gt=0, le=1)
    entropy_coef: float = Field(0.01, ge=0)
    value_coef: float = Field(0.5, ge=0)
    max_grad_norm: float = Field(0.5, gt=0)
    batch_size: int = Field(64, ge=1)
    epochs: int = Field(4, ge=1, le=100)
    evaluation_frequency: int = Field(2048, ge=1)
    evaluation_episodes: int = Field(2, ge=1, le=100)
    evaluation_horizon: int = Field(1000, ge=1)
    checkpoint_frequency: int = Field(2048, ge=1)
    seed: int = Field(42, ge=0, le=2147483647)
    torch_threads: int = Field(2, ge=1, le=64)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    hidden_size: int = Field(128, ge=32, le=512)
    target_kl: float = Field(0.03, gt=0)


class ExperimentConfig(Config):
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)
    training: TrainConfig = Field(default_factory=TrainConfig)
