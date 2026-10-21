import csv
import json
import threading
from statistics import mean

import pytest
from fastapi.testclient import TestClient

from opus_rl.app.server import State, create_app
from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import TrainConfig
from opus_rl.training.train import train
from test_environment import small_config


def test_every_training_step_is_published_before_reset_and_independent_of_updates(tmp_path):
    config = ExperimentConfig(scenario=small_config(horizon=3), training=TrainConfig(
        total_steps=9, rollout_steps=5, batch_size=5, epochs=1, hidden_size=32,
        torch_threads=1, evaluation_frequency=5, evaluation_episodes=1, checkpoint_frequency=5))
    events = []
    directory, summary = train(config, tmp_path, callback=events.append, output_name='live')
    samples = [event for event in events if 'sample' in event]
    assert [event['steps'] for event in samples] == list(range(1, 10))
    # Progress is available during collection, before the first optimizer update at step 5.
    assert next(i for i, event in enumerate(events) if 'sample' in event) < next(
        i for i, event in enumerate(events) if 'metrics' in event)
    assert [event['sample']['episode'] for event in samples] == [1] * 3 + [2] * 3 + [3] * 3
    assert [event['sample']['episode_step'] for event in samples] == [1, 2, 3] * 3
    assert [event['telemetry']['info']['simulated_seconds'] for event in samples] == [5, 10, 15] * 3
    for event in samples:
        telemetry = event['telemetry']
        assert telemetry['info']['step'] == event['sample']['episode_step']
        assert event['sample']['mean_step_reward'] == pytest.approx(mean(u['reward'] for u in telemetry['uavs']))
        assert all(0 <= u['battery_ratio'] <= 1 for u in telemetry['uavs'])
        if event['sample']['episode_step'] == 3:
            assert event['frame']['info']['step'] == 3  # Never substitute the reset frame.
    assert len([event for event in events if 'metrics' in event]) == 2
    evaluation_events = [event for event in events if event.get('evaluation_progress')]
    assert evaluation_events and {event['steps'] for event in evaluation_events} == {5}
    assert not any('sample' in event for event in evaluation_events)
    for filename in ('steps.csv', 'uav-rewards.csv', 'timings.csv'):
        with (directory / filename).open(newline='', encoding='utf8') as saved:
            rows = list(csv.DictReader(saved))
        assert rows
        if filename == 'steps.csv':
            assert len(rows) == 9
            assert [float(row['mean_step_reward']) for row in rows] == pytest.approx(
                [event['sample']['mean_step_reward'] for event in samples])
        if filename == 'uav-rewards.csv':
            assert len(rows) == summary['agent_steps']
            assert all('battery_ratio' in row and 'simulated_seconds' in row for row in rows)
    timing = summary['timings']
    assert timing['policy']['calls'] == timing['simulation']['calls'] == timing['bootstrap']['calls'] == 9
    assert timing['update']['calls'] == 2 and timing['evaluation']['calls'] == 1
    assert timing['simulation.camera']['calls'] == 9
    assert sum(row['percent'] for key, row in timing.items() if '.' not in key) == pytest.approx(100)
    assert sum(timing[key]['seconds'] for key in timing if key.startswith('simulation.')
               and key.count('.') == 1) <= timing['simulation']['seconds']
    assert sum(timing[key]['seconds'] for key in timing if key.startswith('simulation.camera.')) == pytest.approx(
        timing['simulation.camera']['seconds'])
    assert all(row['seconds'] >= 0 and row['mean_ms'] >= 0 for row in timing.values())
    assert json.loads((directory / 'summary.json').read_text())['timings'] == timing


def test_status_keeps_update_history_separate_and_returns_only_unseen_steps(tmp_path):
    app = create_app(tmp_path)
    state = app.state.opus
    state.publish({'run_id': 'first', 'steps': 1, 'sample': {'steps': 1, 'episode': 1}})
    state.publish({'steps': 2, 'sample': {'steps': 2, 'episode': 1}})
    state.publish({'steps': 2, 'metrics': {'steps': 2, 'actor_loss': 0.5}})
    state.publish({'phase': 'evaluating', 'evaluation_progress': {'step': 1}})
    with TestClient(app) as client:
        status = client.get('/api/training/status?run_id=first&after_step=1').json()
        assert [row['steps'] for row in status['samples']] == [2]
        assert len(status['history']) == 1
        assert status['live']['steps'] == status['steps'] == 2
        assert not client.get('/api/training/status?run_id=first&after_step=2').json()['samples']
        assert len(client.get('/api/training/status?run_id=previous&after_step=100').json()['samples']) == 2
    # Bounded memory regardless of run length.
    isolated = State(tmp_path)
    for step in range(2100):
        isolated.publish({'steps': step, 'sample': {'steps': step}})
    assert len(isolated.samples) == 2000 and isolated.samples[0]['steps'] == 100


def test_stop_during_collection_preserves_latest_step_and_saves_partial_update(tmp_path):
    config = ExperimentConfig(scenario=small_config(horizon=16), training=TrainConfig(
        total_steps=20, rollout_steps=10, batch_size=5, epochs=1, hidden_size=32,
        torch_threads=1, evaluation_frequency=1000, checkpoint_frequency=1000))
    stop = threading.Event()
    events = []

    def callback(event):
        events.append(event)
        if event.get('sample', {}).get('steps') == 2:
            stop.set()

    directory, summary = train(config, tmp_path, callback, stop, output_name='stopped')
    assert summary['status'] == 'stopped' and summary['steps'] == 2
    assert [event['steps'] for event in events if 'sample' in event] == [1, 2]
    assert summary['timings']['update']['calls'] == 1
    assert events[-1]['phase'] == 'stopped'
    final_frame = next(event['frame'] for event in reversed(events) if 'frame' in event)
    assert final_frame['info']['step'] == 2
    assert (directory / 'final.pt').is_file()
    with (directory / 'steps.csv').open(newline='', encoding='utf8') as saved:
        assert len(list(csv.DictReader(saved))) == 2
