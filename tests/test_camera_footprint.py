from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.camera_reference import scan as reference_scan
from opus_rl.app.defaults import default_experiment
from opus_rl.configs.models import ScenarioConfig
from opus_rl.envs import team_env
from opus_rl.simulation.camera import scan
from opus_rl.simulation.geometry import Area
from test_environment import small_config


def assert_same_capture(area, uav):
    expected = reference_scan(area, uav, 0.05)
    actual = scan(area, uav, 0.05)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    for key in ('valid_cells', 'invalid_gsd_cells', 'mean_gsd', 'mean_incidence_deg'):
        assert actual[2][key] == expected[2][key]


@pytest.mark.parametrize('kind', ['starter', 'square', 'concave'])
def test_footprint_candidates_match_dense_reference_for_random_and_horizon_views(kind):
    if kind == 'starter':
        config = default_experiment().scenario.model_copy(update={'grid_size': 64, 'observation_size': 16})
    elif kind == 'concave':
        config = small_config(polygon_xy=[(0, 0), (200, 0), (200, 40), (40, 40), (40, 200), (0, 200)],
                              station_positions_xy=[(20, 20), (20, 140)])
    else:
        config = small_config(polygon_xy=[(0, 0), (40, 0), (40, 40), (0, 40)],
                              station_positions_xy=[(5, 5), (35, 35)])
    area = Area(config)
    rng = np.random.default_rng(71)
    pitches = [-90, -89.999, -60, -25, -0.000001, 0, 35, 90]
    for index in range(64):
        pitch = pitches[index] if index < len(pitches) else rng.uniform(-90, 90)
        spec = config.uav_types[0].model_copy(update={'camera_fov_deg': [20, 84, 150][index % 3]})
        position = np.r_[area.low + area.side * rng.uniform(-0.25, 1.25, 2), rng.uniform(0.1, 150)]
        uav = SimpleNamespace(spec=spec, position=position, yaw=rng.uniform(-np.pi, np.pi),
                              gimbal=np.deg2rad(pitch), status='coverage')
        assert_same_capture(area, uav)


@pytest.mark.parametrize('height', [1e-9, 3.5, 100])
def test_footprint_preserves_exact_fov_edges_and_full_view(height):
    config = ScenarioConfig(grid_size=16, observation_size=8,
        polygon_xy=[(0, 0), (16, 0), (16, 16), (0, 16)], station_positions_xy=[(1, 1), (15, 15)])
    area = Area(config)
    spec = config.uav_types[0].model_copy(update={
        'camera_width_px': 16, 'camera_height_px': 16, 'camera_fov_deg': 90, 'camera_fov_axis': 'horizontal'})
    uav = SimpleNamespace(spec=spec, position=np.array([8., 8., height]), yaw=0.,
                          gimbal=-np.pi / 2, status='coverage')
    assert_same_capture(area, uav)
    uav.status = 'charging'
    assert_same_capture(area, uav)


def test_complete_environment_transitions_match_the_reference_camera(monkeypatch):
    config = small_config(horizon=16)
    original, optimized = team_env.OpusTeamEnv(config), team_env.OpusTeamEnv(config)
    original.reset(seed=42)
    optimized.reset(seed=42)
    rng = np.random.default_rng(19)
    for _ in range(16):
        action = rng.uniform(-1, 1, (original.n_max, 5))
        action[:, 4] = -1
        monkeypatch.setattr(team_env, 'scan', reference_scan)
        before = original.step(action)
        monkeypatch.setattr(team_env, 'scan', scan)
        after = optimized.step(action)
        for key in before[0]:
            np.testing.assert_array_equal(after[0][key], before[0][key])
        np.testing.assert_array_equal(optimized.coverage, original.coverage)
        assert after[1:4] == before[1:4]
        assert after[4] == before[4]
    original.close()
    optimized.close()


def test_defaults_keep_128_input_with_four_by_four_averaging():
    import json
    from pathlib import Path

    from opus_rl.simulation.coverage import fuse

    config = default_experiment().scenario
    assert config.grid_size == 512 and config.observation_size == 128
    saved = json.loads((Path(__file__).parents[1] / 'configs/default.json').read_text())['scenario']
    assert saved['grid_size'] == 512 and saved['observation_size'] == 128
    coverage = np.zeros((512, 512), np.float32)
    coverage[:4, :4] = np.arange(16).reshape(4, 4) / 16
    fused = fuse(coverage, np.ones_like(coverage, bool), 128)
    assert fused.shape == (2, 128, 128)
    assert fused[1, 0, 0] == pytest.approx(coverage[:4, :4].mean())
