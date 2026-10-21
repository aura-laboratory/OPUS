import io
import json
import threading
import time
import zipfile
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from opus_rl.app import server
from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import TrainConfig
from opus_rl.marl.network import ActorCritic
from opus_rl.simulation.trajectory import Trajectories
from test_environment import small_config


@pytest.fixture
def saved_run(tmp_path):
    config = ExperimentConfig(scenario=small_config(horizon=6), training=TrainConfig(hidden_size=32))
    directory = tmp_path / 'saved'
    directory.mkdir()
    (directory / 'config.json').write_text(config.model_dump_json(), encoding='utf8')
    (directory / 'summary.json').write_text(json.dumps({
        'run_id': 'saved', 'status': 'completed', 'steps': 6, 'seed': 42, 'algorithm': 'ippo'}), encoding='utf8')
    torch.save({'model': ActorCritic(32).state_dict(), 'config': config.model_dump(), 'algorithm': 'ippo'},
               directory / 'best.pt')
    return config, directory


def finished_evaluation(client):
    for _ in range(300):
        status = client.get('/api/evaluation/status').json()
        if status['status'] in ('completed', 'failed', 'stopped'):
            assert status['status'] != 'failed', status
            return status
        time.sleep(0.01)
    pytest.fail('Evaluation did not finish.')


def test_run_display_name_export_and_safe_deletion(tmp_path, saved_run):
    _, directory = saved_run
    sentinel = tmp_path / 'keep.txt'
    sentinel.write_text('keep')
    with TestClient(server.create_app(tmp_path)) as client:
        assert client.post('/api/runs/saved/rename', json={'name': '  Πλατεία · v1  '}).json()['name'] == 'Πλατεία · v1'
        assert directory.is_dir() and (directory / 'best.pt').exists()
        assert client.get('/api/runs').json()[0]['name'] == 'Πλατεία · v1'
        for invalid in ('', 'x' * 81, 'one\ntwo'):
            assert client.post('/api/runs/saved/rename', json={'name': invalid}).status_code == 400
        with zipfile.ZipFile(io.BytesIO(client.get('/api/runs/saved/export').content)) as archive:
            assert json.loads(archive.read('metadata.json'))['name'] == 'Πλατεία · v1'
        assert client.post('/api/runs/saved/load', json={}).status_code == 200
        assert client.post('/api/runs/saved/delete', json={}).json() == {'deleted': 'saved'}
        assert not directory.exists() and sentinel.read_text() == 'keep'
        assert not client.get('/api/runs').json()
        assert client.get('/api/simulation/frame').status_code == 409
        assert client.post('/api/runs/saved/delete', json={}).status_code == 404


def test_delete_rejects_training_run_and_escaped_paths(tmp_path, saved_run):
    app = server.create_app(tmp_path)
    state = app.state.opus
    with TestClient(app) as client:
        state.worker = SimpleNamespace(is_alive=lambda: True)
        state.status['run_id'] = 'saved'
        try:
            assert client.post('/api/runs/saved/delete', json={}).status_code == 409
        finally:
            state.worker = None
        for run_id in ('..', '..\\outside', 'C:\\outside'):
            with pytest.raises(server.HTTPException) as error:
                state.run_directory(run_id)
            assert error.value.status_code == 400
        assert saved_run[1].exists()


def test_playback_trails_respect_length_reset_and_invalid_requests(tmp_path, saved_run):
    config, _ = saved_run
    with TestClient(server.create_app(tmp_path)) as client:
        assert client.post('/api/simulation/reset?trail_steps=2', json=config.model_dump()).status_code == 200
        for _ in range(6):
            frame = client.post('/api/simulation/step', json={'policy': 'heuristic', 'trail_steps': 2}).json()
            assert all(len(trail['points']) <= 3 for trail in frame['trails'])
        assert frame['episode'] == 1 and frame['info']['step'] == 6
        assert not client.get('/api/simulation/frame?trail_steps=0').json()['trails']
        assert client.post('/api/simulation/step', json={'trail_steps': 2001}).status_code == 400
        assert client.get('/api/simulation/frame').json()['info']['step'] == 6
        frame = client.post('/api/simulation/step', json={'policy': 'heuristic', 'trail_steps': 2}).json()
        assert frame['episode'] == 2 and frame['info']['step'] == 1
        assert all(len(trail['points']) <= 2 for trail in frame['trails'])
        assert client.post('/api/simulation/reset?trail_steps=-1', json=config.model_dump()).status_code == 422


def test_trajectory_storage_is_bounded():
    import numpy as np

    uav = SimpleNamespace(inventory_index=4, status='coverage', position=np.zeros(3))
    env = SimpleNamespace(uavs=[uav], area=SimpleNamespace(geographic=lambda point: point), step_index=0)
    trails = Trajectories()
    for step in range(2200):
        env.step_index = step
        uav.position[0] = step
        trails.record(env)
    assert len(trails.points[4]) == 2001
    assert len(trails.snapshot(2199, 10)[0]['points']) == 11
    assert trails.snapshot(2199, 10)[0]['points'][0] == [2189, 0]


def test_async_evaluation_has_progress_cancel_guard_and_excludes_partial_episode(tmp_path, saved_run, monkeypatch):
    seen, release = threading.Event(), threading.Event()
    original = server.evaluate

    def held_evaluation(*args, **kwargs):
        callback = kwargs['callback']

        def progress(update):
            callback(update)
            seen.set()
            assert release.wait(5)

        return original(*args, **{**kwargs, 'callback': progress})

    monkeypatch.setattr(server, 'evaluate', held_evaluation)
    with TestClient(server.create_app(tmp_path)) as client:
        assert client.post('/api/runs/saved/evaluation/start', json={'episodes': 2, 'seed': 123}).status_code == 200
        try:
            assert seen.wait(5)
            status = client.get('/api/evaluation/status').json()
            assert status['status'] == 'running' and status['episode'] == status['step'] == 1
            assert status['episodes'] == 2 and status['episode_length'] == 6
            assert client.post('/api/runs/saved/delete', json={}).status_code == 409
            assert client.post('/api/runs/saved/evaluation/start', json={}).status_code == 409
            assert client.post('/api/evaluation/stop', json={}).json()['status'] == 'stopping'
        finally:
            release.set()
        status = finished_evaluation(client)
        assert status['status'] == 'stopped' and status['result']['episodes'] == 0
        assert not (saved_run[1] / 'evaluation-manual.json').exists()


def test_async_saved_and_baseline_evaluation_show_repeatable_results(tmp_path, saved_run):
    config, directory = saved_run
    with TestClient(server.create_app(tmp_path)) as client:
        request = {'episodes': 2, 'seed': 123, 'trail_steps': 2}
        assert client.post('/api/runs/saved/evaluation/start', json=request).status_code == 200
        first = finished_evaluation(client)
        assert first['status'] == 'completed'
        result = first['result']
        assert result['episodes'] == 2 and result['seeds'] == [123, 124]
        assert result['frame']['info']['step'] == 6
        assert all(len(trail['points']) <= 3 for trail in result['frame']['trails'])
        saved = json.loads((directory / 'evaluation-manual.json').read_text())
        assert saved['checkpoint'] == 'best.pt' and saved['return_mean'] == result['return_mean']
        assert client.get('/api/runs').json()[0]['manual_evaluation']['episodes'] == 2
        assert client.post('/api/runs/saved/evaluation/start', json=request).status_code == 200
        second = finished_evaluation(client)
        assert second['job_id'] != first['job_id']
        assert second['result']['uav_returns'] == result['uav_returns']
        assert client.post('/api/evaluation/start', json={**request, 'policy': 'heuristic',
            'scenario': config.scenario.model_dump()}).status_code == 200
        baseline = finished_evaluation(client)
        assert baseline['result']['episodes'] == 2 and baseline['run_id'] is None
        assert client.post('/api/evaluation/start', json={'episodes': 0}).status_code == 422
