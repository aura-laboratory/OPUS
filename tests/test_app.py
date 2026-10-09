import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from opus_rl.app.server import create_app
from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import TrainConfig
from test_environment import small_config


def experiment():
    return ExperimentConfig(scenario=small_config(horizon=16),
                            training=TrainConfig(total_steps=32, num_envs=1, rollout_steps=8,
                              epochs=1, hidden_size=32, batch_size=8,
                              evaluation_frequency=16, evaluation_episodes=1, evaluation_horizon=12,
                              checkpoint_frequency=16, torch_threads=1)).model_dump()


def test_local_app_simulation_and_invalid_configuration(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/health').json()['local_only']
        assert client.get('/').status_code == 200
        defaults = client.get('/api/defaults').json()['config']
        assert defaults['scenario']['coverage_lambda'] == 0.9
        configuration = experiment()
        assert client.post('/api/validate', json=configuration).status_code == 200
        assert client.post('/api/simulation/reset', json=configuration).status_code == 200
        frame = client.post('/api/simulation/step', json={'policy':'heuristic'}).json()
        assert frame['info']['simulated_seconds'] == 5
        configuration['scenario']['min_altitude_m'] = 200
        assert client.post('/api/validate', json=configuration).status_code == 422
        assert client.post('/api/training/stop', json={}, headers={'Origin':'https://example.com'}).status_code == 403


def test_training_checkpoint_reload_evaluation_and_export(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post('/api/training/start', json=experiment()).status_code == 200
        for _ in range(300):
            status = client.get('/api/training/status').json()
            if status['status'] in ('completed','failed'):
                break
            time.sleep(0.05)
        assert status['status'] == 'completed', status
        assert status['steps'] == 32
        assert all(0 <= row['actor_sample_fraction'] <= 1 for row in status['history'])
        run = client.get('/api/runs').json()[0]
        assert {'initial.pt','final.pt','best.pt'}.issubset(run['checkpoints'])
        result = client.post(f"/api/runs/{run['run_id']}/load", json={'checkpoint':'best.pt'})
        assert result.status_code == 200
        frame = client.post('/api/simulation/step', json={'policy':'model'}).json()
        assert frame['info']['step'] == 1
        evaluation = client.post(f"/api/runs/{run['run_id']}/evaluate",json={'episodes':1})
        assert evaluation.status_code == 200
        assert evaluation.json()['episodes'] == 1
        archive = client.get(f"/api/runs/{run['run_id']}/export")
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zipped:
            assert {'config.json','normalization.json','metrics.csv','summary.json','best.pt'}.issubset(zipped.namelist())
            assert json.loads(zipped.read('summary.json'))['seed'] == 42


def test_invalid_inventory_and_station_types():
    data = experiment()
    data['scenario']['uav_types'][0]['fixed_count'] = 3
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)
    data = experiment()
    data['scenario']['station_position_types'] = [0, 1]
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)
