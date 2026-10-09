import numpy as np
import pytest

from opus_rl.configs.models import ScenarioConfig, UAVType
from opus_rl.simulation.camera import intrinsics, observation_quality, scan
from opus_rl.simulation.coverage import fuse, metrics, reward, update_coverage
from opus_rl.simulation.geometry import Area
from opus_rl.simulation.motion import AnalyticalBackend, UAV, minimum_separation, stopped_distance


def test_rest_to_rest_motion_and_energy_closed_form():
    spec = UAVType(P_airborne_base_W=100, k_vxy=0, k_vz_up=0, k_vz_down=0, k_acc=0)
    backend = AnalyticalBackend()
    result = backend.propagate([0, 0, 20], [16, 0, 20], spec, 5)
    assert result.profile.reach_time == pytest.approx(4)
    assert result.state(5)[0] == pytest.approx([16, 0, 20])
    assert result.state(5)[1] == pytest.approx([0, 0, 0])
    assert result.energy() == pytest.approx(100 * 5 / 3600)
    assert stopped_distance(15, 4, 5) == pytest.approx(25)
    incomplete = backend.propagate([0, 0, 20], [100, 0, 20], spec, 1)
    assert incomplete.state(1)[0][0] == pytest.approx(2)
    assert incomplete.state(1)[1][0] == pytest.approx(4)
    assert incomplete.state(1)[2][0] == pytest.approx(4)


def test_aligned_initial_velocity_and_noncollinear_rejection():
    spec = UAVType()
    backend = AnalyticalBackend()
    result = backend.propagate([0, 0, 20], [20, 0, 20], spec, 5, [4, 0, 0])
    assert result.state(5)[0][0] == pytest.approx(20)
    with pytest.raises(ValueError, match="noncollinear"):
        backend.propagate([0, 0, 20], [20, 0, 20], spec, 5, [0, 4, 0])


def test_analytic_collision_detects_crossing_with_safe_endpoints():
    backend, spec = AnalyticalBackend(), UAVType()
    first = backend.propagate([-10, 0, 30], [10, 0, 30], spec, 5)
    second = backend.propagate([0, -10, 30], [0, 10, 30], spec, 5)
    distance, time = minimum_separation(first, second)
    assert distance < 1e-7
    assert time == pytest.approx(np.sqrt(20 / 4), abs=1e-5)


def test_collision_respects_actual_acceleration_timing():
    backend = AnalyticalBackend()
    first = backend.propagate([-10, 0, 20], [10, 0, 20], UAVType(), 5)
    second = backend.propagate([0, -10, 20], [0, 10, 20], UAVType(max_horizontal_accel_mps2=1), 10)
    # Compare only common simulation time, not normalized segment positions.
    second = backend.propagate([0, -10, 20], [0, 10, 20], UAVType(max_horizontal_accel_mps2=1), 5)
    distance, _ = minimum_separation(first, second)
    assert distance > 2


def test_concave_polygon_rejects_segment_excursion():
    cfg = ScenarioConfig(polygon_xy=[(0, 0), (10, 0), (10, 3), (3, 3), (3, 10), (0, 10)],
                         station_positions_xy=[(1, 1), (2, 8)], grid_size=64, observation_size=16)
    area = Area(cfg)
    start, end = np.array([8, 2]), np.array([2, 8])
    assert not area.segment_inside(start, end)
    projected = area.constrain_segment(start, end)
    assert area.segment_inside(start, projected)
    assert projected[1] <= 3.01
    xy = np.array([[1, 2], [2, 3]])
    geographic = area.geographic(xy)
    assert np.column_stack(area.forward.transform(*geographic.T)) == pytest.approx(xy, abs=1e-6)


def test_nadir_gsd_matches_pinhole_and_quality_formula():
    cfg = ScenarioConfig(grid_size=64, observation_size=16)
    area, spec = Area(cfg), UAVType()
    uav = UAV(spec, 0, np.array([80.0, 80.0, 50.0]), 70, status="coverage")
    q, gsd, stats = scan(area, uav, 0.05)
    focal, _, _ = intrinsics(spec)
    assert gsd[np.isfinite(gsd)] == pytest.approx(np.full(np.isfinite(gsd).sum(), 50 / focal), rel=1e-5)
    assert stats["valid_cells"] > 0
    assert q.max() <= 1
    assert observation_quality(np.array([0.01, 0.05, 0.051]), np.ones(3), 0.05) == pytest.approx([0.9, 0.5, 0])
    assert observation_quality(np.array([0.05]), np.array([0.25]), 0.05)[0] == pytest.approx(0.25)
    uav.gimbal = np.deg2rad(-40)
    _, oblique, _ = scan(area, uav, 1)
    assert np.nanmean(oblique) > 50 / focal


def test_raster_boundary_preserves_cells_touched_at_a_vertex():
    area = Area(ScenarioConfig(polygon_xy=[(0, 0), (16, 0), (0, 16)],
        station_positions_xy=[(1, 1), (1, 5)], grid_size=16, observation_size=8))
    assert area.mask[8, 8]  # The diagonal touches this cell's corner; center sampling alone misses it.
    assert not area.mask[15, 15]


def test_ema_fusion_and_zero_map_reward():
    mask = np.ones((16, 16), bool)
    coverage = np.zeros((16, 16), np.float32)
    q = np.ones_like(coverage)
    coverage = update_coverage(coverage, q, mask, 0.9)
    assert coverage == pytest.approx(np.full_like(coverage, 0.1))
    coverage = update_coverage(coverage, np.zeros_like(q), mask, 0.9)
    assert coverage[0, 0] == pytest.approx(0.09)
    assert 5 * np.log(0.5) / np.log(0.9) == pytest.approx(32.894067, rel=1e-5)
    coverage[:2, :2] = np.array([[0, 1], [0, 1]])
    fused = fuse(coverage, mask, 8)
    assert fused[1, 0, 0] == pytest.approx(0.5)
    assert fused[2, 0, 0] == 0
    empty = metrics(np.zeros_like(coverage), mask)
    assert empty["fairness"] == 0
    total, terms = reward(ScenarioConfig().reward, empty, 1, 1, 1, 1)
    assert total == pytest.approx(-1)
    assert all(0 <= v <= 1 for v in terms.values())
