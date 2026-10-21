import queue
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from opus_rl.app.server import create_app
from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import TrainConfig
from opus_rl.training.control import TrainingControl
from opus_rl.training.timing import Timings
from opus_rl.training.train import train
from test_environment import small_config


def configuration():
    return ExperimentConfig(scenario=small_config(horizon=3), training=TrainConfig(
        total_steps=6, rollout_steps=2, batch_size=2, epochs=1, hidden_size=32,
        torch_threads=1, evaluation_frequency=2, evaluation_episodes=1, checkpoint_frequency=2))


def wait_paused(events, expected_step=None):
    while True:
        event = events.get(timeout=15)
        if event.get('control_status') == 'paused':
            if expected_step is not None:
                assert event['steps'] == expected_step
            return event


@pytest.mark.parametrize('finish', ['stop', 'resume', 'steps'])
def test_single_steps_cross_update_and_evaluation_without_losing_rollout(tmp_path, finish):
    control = TrainingControl()
    control.pause()
    events, published, outcome = queue.Queue(), [], {}

    def callback(event):
        published.append(event)
        events.put(event)

    def work():
        try:
            outcome['result'] = train(configuration(), tmp_path, callback=callback,
                                      output_name=finish, control=control)
        except Exception as error:
            outcome['error'] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        wait_paused(events, 0)
        for step in (1, 2, 3):
            control.step()
            wait_paused(events, step)
            assert [event['steps'] for event in published if 'sample' in event] == list(range(1, step + 1))
            assert all(event['frame_step'] == event['steps'] for event in published if 'sample' in event)
        # The third click finishes the pending update/evaluation, then collects one interaction.
        assert len([event for event in published if 'metrics' in event]) == 1
        assert any(event.get('evaluation_progress') for event in published)
        if finish == 'steps':
            for step in (4, 5):
                control.step()
                wait_paused(events, step)
            control.step()  # The final step must finish its scheduled evaluation without another permit.
        else:
            getattr(control, finish)()
        worker.join(15)
        assert not worker.is_alive()
        assert 'error' not in outcome, outcome.get('error')
        directory, summary = outcome['result']
        assert summary['steps'] == (3 if finish == 'stop' else 6)
        assert summary['status'] == ('stopped' if finish == 'stop' else 'completed')
        assert summary['timings']['update']['calls'] == (2 if finish == 'stop' else 3)
        assert summary['paused_seconds'] > 0
        assert (directory / 'final.pt').is_file()
        if finish != 'stop':
            # Pauses and single-step permits must not change exploration or optimization RNG.
            uninterrupted, _ = train(configuration(), tmp_path, output_name='uninterrupted')
            actual = torch.load(directory / 'final.pt', weights_only=False)['model']
            expected = torch.load(uninterrupted / 'final.pt', weights_only=False)['model']
            assert all(torch.equal(actual[key], expected[key]) for key in expected)
    finally:
        control.stop()
        worker.join(15)


@pytest.mark.parametrize('pause_at', [2, 6])
def test_pause_during_scheduled_evaluation_preserves_its_progress(tmp_path, pause_at):
    control, events, outcome = TrainingControl(), queue.Queue(), {}
    evaluation_steps = []

    def callback(event):
        progress = event.get('evaluation_progress')
        if progress:
            evaluation_steps.append((event['steps'], progress['step']))
            if event['steps'] == pause_at and progress['step'] == 1:
                control.pause()
        events.put(event)

    def work():
        try:
            outcome['result'] = train(configuration(), tmp_path, callback=callback,
                                      output_name='evaluation-pause', control=control)
        except Exception as error:
            outcome['error'] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        paused = wait_paused(events, pause_at)
        assert paused['phase'] == 'evaluating'
        assert evaluation_steps[-1] == (pause_at, 1)
        control.resume()
        worker.join(15)
        assert not worker.is_alive()
        assert 'error' not in outcome, outcome.get('error')
        assert outcome['result'][1]['steps'] == 6
        assert evaluation_steps == [(step, position) for step in (2, 4, 6) for position in (1, 2, 3)]
    finally:
        control.stop()
        worker.join(15)


def test_pause_and_step_api_acknowledges_safe_boundaries_and_stop_wakes_worker(tmp_path):
    app = create_app(tmp_path)
    state, events = app.state.opus, queue.Queue()
    original = state.publish

    def publish(event):
        original(event)
        events.put(event)

    state.publish = publish
    with TestClient(app) as client:
        for action in ('pause', 'resume', 'step'):
            assert client.post('/api/training/' + action, json={}).status_code == 409
        assert client.post('/api/training/start', json=configuration().model_dump()).status_code == 200
        assert client.post('/api/training/pause', json={}).json()['status'] in ('pausing', 'paused')
        paused = wait_paused(events)
        before = client.get('/api/training/status').json()
        assert before['status'] == 'paused'
        assert client.post('/api/training/start', json=configuration().model_dump()).status_code == 409
        assert client.post('/api/training/step', json={}).json()['status'] == 'stepping'
        wait_paused(events, paused['steps'] + 1)
        after = client.get('/api/training/status').json()
        assert after['status'] == 'paused' and after['steps'] == before['steps'] + 1
        assert client.post('/api/training/stop', json={}).json()['status'] == 'stopping'
        state.worker.join(15)
        assert not state.worker.is_alive()
        assert client.get('/api/training/status').json()['status'] == 'stopped'


def test_lifespan_shutdown_wakes_paused_training(tmp_path):
    app = create_app(tmp_path)
    state, events = app.state.opus, queue.Queue()
    original = state.publish

    def publish(event):
        original(event)
        events.put(event)

    state.publish = publish
    with TestClient(app) as client:
        client.post('/api/training/start', json=configuration().model_dump())
        client.post('/api/training/pause', json={})
        wait_paused(events)
    assert not state.worker.is_alive()
    assert state.status['status'] == 'stopped'


def test_profiling_excludes_wait_inside_an_evaluation_phase(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('opus_rl.training.control.time.perf_counter', lambda: clock[0])
    control = TrainingControl()
    control.pause()
    timing = Timings(clock=lambda: clock[0] - control.paused_seconds)

    def resume():
        if control.status == 'paused':
            clock[0] += 1000  # A long idle pause must not dominate the evaluation percentage.
            control.resume()

    with timing.measure('evaluation'):
        clock[0] += 2
        control.wait(changed=resume)
        clock[0] += 3
    timing.record('simulation', 10)
    assert control.paused_seconds == 1000
    assert timing.snapshot()['evaluation']['seconds'] == 5
    assert timing.snapshot()['evaluation']['percent'] == pytest.approx(100 / 3)
