import numpy as np
import pytest
from fastapi.testclient import TestClient

from opus_rl.app.server import create_app
from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import TrainConfig
from opus_rl.simulation.history import VisualHistory
from opus_rl.simulation.trajectory import Trajectories
from opus_rl.training.train import train
from test_environment import small_config


def telemetry(step):
    return {"info": {"step": step, "simulated_seconds": step * 5, "reward": step},
            "uavs": [{"inventory_index": 0, "id": 0, "status": "coverage", "geographic": [step, 0],
                      "position": [step, 0, 20], "battery_ratio": 1 - step / 100, "reward": step}]}


def frame(step):
    return {**telemetry(step), "heatmap": [[step / 10] * 2] * 2, "heatmap_mask": [[1, 1], [0, 1]],
            "stations": [], "polygon_wgs84": [[0, 0], [10, 0], [10, 10]], "bounds_wgs84": [[0, 0], [10, 10]]}


def test_history_returns_exact_readings_and_prior_map_without_modifying_live_state():
    history = VisualHistory(limit=4)
    for step in range(5):
        history.append(step, 1, step * 5, telemetry(step), frame(step) if step % 2 == 0 else None)
    assert history.metadata()['first_step'] == 1 and history.metadata()['last_step'] == 4
    past = history.frame(3, 1.5)
    assert past['uavs'][0]['position'] == [3, 0, 20]
    assert past['info']['reward'] == 3 and past['history_seconds'] == 15
    assert past['map_history_step'] == 2
    assert np.asarray(past['heatmap']) == pytest.approx(np.full((2, 2), 0.2), abs=1 / 255)
    assert past['trails'][0]['points'] == [[1.5, 0], [2, 0], [3, 0]]
    assert history.metadata()['last_step'] == 4
    with pytest.raises(ValueError):
        history.frame(0)


def test_history_keeps_episode_trails_separate_and_evicts_old_episodes():
    history = VisualHistory(limit=2)
    history.append(0, 1, 0, telemetry(0), frame(0))
    history.append(1, 1, 5, telemetry(1), frame(1))
    history.append(2, 2, 10, telemetry(0), frame(0))
    assert history.frame(1, 500)['episode'] == 1
    assert history.frame(2, 500)['trails'][0]['points'] == [[0, 0]]
    history.append(3, 2, 15, telemetry(1), frame(1))
    assert 1 not in history.trails
    old_id = history.identity
    history.clear()
    assert history.identity != old_id and not history.records and not history.frames


def test_trails_clip_to_ground_distance_and_interpolate_cutoff():
    trails = Trajectories()
    for step, xy in enumerate(([0, 0], [3, 0], [3, 4], [3, 4])):
        trails.record_samples([{"inventory_index": 0, "status": "coverage", "position": [*xy, step * 50],
                               "geographic": xy}], step)
    assert trails.snapshot(3, 5)[0]['points'] == [[2, 0], [3, 0], [3, 4]]
    assert trails.snapshot(1, 2)[0]['points'] == [[1, 0], [3, 0]]
    assert trails.snapshot(3, 7)[0]['points'] == [[0, 0], [3, 0], [3, 4]]
    assert trails.snapshot(3, 0) == []


def test_training_rewind_api_does_not_stop_or_restore_the_trainer(tmp_path):
    config = ExperimentConfig(scenario=small_config(horizon=3), training=TrainConfig(
        total_steps=7, rollout_steps=4, hidden_size=32, epochs=1, batch_size=4,
        torch_threads=1, evaluation_frequency=1000, checkpoint_frequency=1000))
    app = create_app(tmp_path)
    state = app.state.opus
    events = []

    def callback(update):
        state.publish(update)
        events.append(update)

    train(config, tmp_path, callback=callback, output_name='rewind')
    with TestClient(app) as client:
        current = client.get('/api/training/status').json()
        past = client.get('/api/visualization/history/training?step=2').json()
        assert past['episode'] == 1 and past['info']['step'] == 2
        reset = client.get('/api/visualization/history/training?step=4').json()
        assert reset['episode'] == 2 and reset['info']['step'] == 1
        assert reset['map_history_step'] == 4  # The first new-episode frame cannot reuse old coverage.
        after = client.get('/api/training/status').json()
        assert current['steps'] == after['steps'] == 7
        assert not state.stop.is_set()
        assert [event['steps'] for event in events if 'sample' in event] == list(range(1, 8))
        assert client.get('/api/visualization/history/training?step=999').status_code == 400


def test_test_rewind_and_unload_do_not_modify_training(tmp_path):
    config = ExperimentConfig(scenario=small_config(horizon=6))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        client.post('/api/simulation/reset', json=config.model_dump())
        for _ in range(4):
            client.post('/api/simulation/step', json={'policy': 'heuristic'})
        timeline = client.get('/api/simulation/frame').json()['timeline']
        past = client.get(f"/api/visualization/history/test?step=1&timeline_id={timeline['id']}").json()
        assert past['info']['step'] == 1
        assert client.get('/api/simulation/frame').json()['info']['step'] == 4
        assert client.post('/api/simulation/unload', json={}).status_code == 200
        assert not app.state.opus.stop.is_set()
        assert client.get('/api/simulation/frame').status_code == 409
        assert client.get(f"/api/visualization/history/test?step=1&timeline_id={timeline['id']}").status_code == 409
