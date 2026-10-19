from collections import Counter

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from opus_rl.configs.models import ScenarioConfig, StationType, UAVType
from opus_rl.envs import OpusTeamEnv


def small_config(**changes):
    return ScenarioConfig(grid_size=64, observation_size=16, **changes)


def test_gymnasium_api_seeding_observation_ownership():
    env = OpusTeamEnv(small_config())
    check_env(env, skip_render_check=True)
    old, _ = env.reset(seed=7)
    snapshot = {k: v.copy() for k, v in old.items()}
    for _ in range(5):
        new, _, _, _, info = env.step(np.zeros((2, 5), np.float32))
        assert env.observation_space.contains(new)
    for key in old:
        assert np.array_equal(old[key], snapshot[key])
    assert info["actor_training_mask"] == [1, 1]


def test_single_uav_zero_teammates_and_vector_environments():
    cfg = small_config(uav_types=[UAVType(available_count=1, fixed_count=1)])
    env = OpusTeamEnv(cfg)
    obs, _ = env.reset(seed=3)
    assert not obs["other_uav_mask"].any()
    vec = gym.vector.SyncVectorEnv([lambda: OpusTeamEnv(cfg), lambda: OpusTeamEnv(cfg)])
    observations, _ = vec.reset(seed=2)
    assert observations["uav_own"].shape[0] == 2
    vec.step(np.zeros((2, 1, 5), np.float32))
    vec.close()


def test_variable_typed_inventory_sampling_and_masks():
    cfg = small_config(uav_types=[UAVType(name="A", available_count=2, fixed_count=2),
                                  UAVType(name="B", available_count=1, fixed_count=1)],
                       station_types=[StationType(available_count=3, fixed_count=3)],
                       fleet_mode="variable", station_mode="variable")
    env = OpusTeamEnv(cfg)
    seen = set()
    for seed in range(25):
        obs, _ = env.reset(seed=seed)
        counts = Counter(u.type_index for u in env.uavs)
        assert counts[0] <= 2 and counts[1] <= 1
        assert 1 <= len(env.stations) <= 3
        assert obs["active_mask"].sum() == len(env.uavs)
        assert env.observation_space.contains(obs)
        seen.add(len(env.uavs))
    assert seen == {1, 2, 3}


def test_projection_bounds_and_horizon():
    env = OpusTeamEnv(small_config(horizon=10))
    env.reset(seed=42)
    for step in range(10):
        _, _, terminated, truncated, info = env.step(np.ones((2, 5), np.float32))
        assert not terminated
        assert truncated == (step == 9)
        for uav in env.uavs:
            if uav.status == "coverage":
                assert 20 - 1e-6 <= uav.position[2] <= 100 + 1e-6
                assert env.area.polygon.covers(__import__("shapely").geometry.Point(uav.position[:2]))
    with pytest.raises(RuntimeError):
        env.step(np.zeros((2, 5)))
    assert info["simulated_seconds"] == 50


def test_recharge_reservation_masks_charging_and_resume():
    env = OpusTeamEnv(small_config())
    env.reset(seed=1)
    for _ in range(3):
        env.step(np.zeros((2, 5)))
    for uav in env.uavs:
        uav.battery = 5
    _, _, _, _, info = env.step(np.zeros((2, 5)))
    assert info["actor_training_mask"] == [0, 0]
    assignments = [u.station for u in env.uavs]
    assert len(set(assignments)) == 2
    for _ in range(5):
        env.step(np.zeros((2, 5)))
    assert all(u.status == "charging" for u in env.uavs)
    before = [u.battery for u in env.uavs]
    env.step(np.zeros((2, 5)))
    assert [u.battery for u in env.uavs] == pytest.approx([
        b + env.stations[u.station].charge_power * 5 / 3600 for b, u in zip(before, env.uavs)
    ])
    for u in env.uavs:
        u.battery = u.spec.battery_capacity_Wh
    for _ in range(3):
        env.step(np.zeros((2, 5)))
    assert all(u.status == "coverage" for u in env.uavs)


def test_battery_depletion_and_persistent_crash_penalty():
    env = OpusTeamEnv(small_config())
    env.reset(seed=1)
    for _ in range(3):
        env.step(np.zeros((2, 5)))
    env.uavs[0].battery = 0.00001
    _, _, terminal, _, info = env.step(np.zeros((2, 5)))
    assert not terminal
    assert env.uavs[0].status == "crashed"
    assert info["reward_terms"]["crash"] == 0.5
    _, _, _, _, next_info = env.step(np.zeros((2, 5)))
    assert next_info["reward_terms"]["crash"] == 0.5
    env._crash(1)
    assert env.step(np.zeros((2, 5)))[2]


def test_shared_launch_dock_stages_full_batteries_and_launches_in_sequence():
    cfg = small_config(station_types=[StationType(available_count=1, fixed_count=1)],
                       station_positions_xy=[(20, 20)], uav_launch_docks=[0, 0])
    env = OpusTeamEnv(cfg)
    observation, _ = env.reset(seed=4)
    assert [u.status for u in env.uavs] == ["charging", "queued"]
    assert all(u.battery == u.spec.battery_capacity_Wh for u in env.uavs)
    assert env.observation_space.contains(observation)
    assert [u.station for u in env.uavs] == [0, 0]
    for _ in range(4):
        _, _, _, _, info = env.step(np.zeros((2, 5)))
        assert info["actor_training_mask"][1] == 0
        assert not info["collision_pairs"]
    assert env.uavs[1].status == "queued"  # First UAV is hovering over the launch column.
    action = np.zeros((2, 5))
    action[0, 0] = 1
    env.step(action)  # Move the first UAV clear of the dock.
    env.step(action)
    assert env.uavs[1].status == "takeoff"
    assert env.uavs[1].battery < env.uavs[1].spec.battery_capacity_Wh
    assert env.frame()["uavs"][1]["launch_dock"] == 0


def test_launch_assignments_follow_inventory_slots_when_sampling():
    cfg = small_config(fleet_mode="variable", station_mode="variable", uav_launch_docks=[0, 1])
    env = OpusTeamEnv(cfg)
    for seed in range(15):
        env.reset(seed=seed)
        for uav in env.uavs:
            expected = cfg.uav_launch_docks[uav.inventory_index]
            assert uav.station == expected
            assert np.array_equal(uav.position, env.stations[expected].position)


def test_legacy_battery_and_landing_inputs_are_removed():
    spec = UAVType.model_validate({"initial_battery_ratio": 0.2})
    assert "initial_battery_ratio" not in spec.model_dump()
    station = StationType.model_validate({"landing_radius_m": 3})
    assert "landing_radius_m" not in station.model_dump()
    env = OpusTeamEnv(small_config(uav_types=[spec]))
    env.reset()
    assert all(u.battery == u.spec.battery_capacity_Wh for u in env.uavs)


def test_fixed_mode_ignores_unused_sampling_bounds():
    cfg = small_config(min_uavs=3, max_uavs=4, min_stations=3, max_stations=4)
    env = OpusTeamEnv(cfg)
    env.reset()
    assert len(env.uavs) == 2 and len(env.stations) == 2
    with pytest.raises(ValueError, match="Variable count bounds"):
        ScenarioConfig.model_validate({**cfg.model_dump(), "fleet_mode": "variable"})
