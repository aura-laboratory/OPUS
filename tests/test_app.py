import io
import csv
import json
import time
import zipfile
from statistics import mean

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point, Polygon

from opus_rl.app.server import create_app
from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import TrainConfig
from opus_rl.envs import OpusTeamEnv
from test_environment import small_config


def experiment():
    return ExperimentConfig(scenario=small_config(horizon=16),
                            training=TrainConfig(total_steps=32, num_envs=1, rollout_steps=8,
                              epochs=1, hidden_size=32, batch_size=8,
                              evaluation_frequency=16, evaluation_episodes=1,
                              checkpoint_frequency=16, torch_threads=1)).model_dump()


def test_local_app_simulation_and_invalid_configuration(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/health').json()['local_only']
        assert client.get('/api/health').json()['training_algorithm'] == 'ippo'
        assert client.get('/').status_code == 200
        defaults = client.get('/api/defaults').json()['config']
        assert defaults['scenario']['coverage_lambda'] == 0.9
        configuration = experiment()
        assert client.post('/api/validate', json=configuration).status_code == 200
        assert client.post('/api/simulation/reset', json=configuration).status_code == 200
        frame = client.post('/api/simulation/step', json={'policy':'heuristic'}).json()
        assert frame['info']['simulated_seconds'] == 5
        assert frame['info']['reward_mode'] == 'individual'
        assert len(frame['info']['uav_rewards']) == 2
        assert frame['info']['reward'] == pytest.approx(mean(frame['info']['uav_rewards']))
        configuration['scenario']['min_altitude_m'] = 200
        assert client.post('/api/validate', json=configuration).status_code == 422
        assert client.post('/api/training/stop', json={}, headers={'Origin':'https://example.com'}).status_code == 403


def test_starter_mission_covers_aristotelous_square_with_docks_inside(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        configuration = client.get('/api/defaults').json()['config']
        response = client.post('/api/validate', json=configuration)
        assert response.status_code == 200
        frame = response.json()['frame']
        polygon = Polygon(frame['polygon_wgs84'])
        assert polygon.covers(Point(22.9409, 40.63235))
        assert 5000 < frame['area_m2'] < 10000
        assert all(polygon.covers(Point(dock['geographic'])) for dock in frame['stations'])


@pytest.mark.parametrize('shared_dock', [False, True])
def test_training_checkpoint_reload_evaluation_and_export(tmp_path, shared_dock, monkeypatch):
    def forbidden_privileged_state(*args):
        raise AssertionError('IPPO must not consume centralized critic state.')

    monkeypatch.setattr(OpusTeamEnv, 'central_state', forbidden_privileged_state)
    with TestClient(create_app(tmp_path)) as client:
        configuration = experiment()
        if shared_dock:
            configuration['scenario']['reward_mode'] = 'legacy_team'
            configuration['scenario']['station_types'][0].update(available_count=1, fixed_count=1)
            configuration['scenario']['station_positions_xy'] = [[20,20]]
            configuration['scenario']['uav_launch_docks'] = [0,0]
        assert client.post('/api/training/start', json=configuration).status_code == 200
        for _ in range(300):
            status = client.get('/api/training/status').json()
            if status['status'] in ('completed','failed'):
                break
            time.sleep(0.05)
        assert status['status'] == 'completed', status
        assert status['steps'] == 32
        assert status['summary']['algorithm'] == 'ippo'
        assert status['summary']['agent_steps'] == 64
        assert all(0 <= row['actor_sample_fraction'] <= 1 for row in status['history'])
        run = client.get('/api/runs').json()[0]
        assert {'initial.pt','final.pt','best.pt'}.issubset(run['checkpoints'])
        result = client.post(f"/api/runs/{run['run_id']}/load", json={'checkpoint':'best.pt'})
        assert result.status_code == 200
        assert result.json()['config']['scenario']['reward_mode'] == 'individual'
        frame = client.post('/api/simulation/step', json={'policy':'model'}).json()
        assert frame['info']['step'] == 1
        alternate = result.json()['config']['scenario']
        alternate['reward_mode'] = 'legacy_team'
        evaluation = client.post(f"/api/runs/{run['run_id']}/evaluate", json={'episodes':1, 'scenario':alternate})
        assert evaluation.status_code == 200
        assert evaluation.json()['episodes'] == 1
        assert evaluation.json()['frame']['info']['step'] == configuration['scenario']['horizon']
        assert evaluation.json()['reward_mode'] == 'individual'
        individual_returns = evaluation.json()['uav_returns'][0]['uavs']
        assert len(individual_returns) == 2
        assert evaluation.json()['return_mean'] == pytest.approx(mean([u['return'] for u in individual_returns]))
        archive = client.get(f"/api/runs/{run['run_id']}/export")
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zipped:
            assert {'config.json','normalization.json','metrics.csv','uav-rewards.csv','summary.json','best.pt'}.issubset(zipped.namelist())
            assert json.loads(zipped.read('summary.json'))['seed'] == 42
            rewards = list(csv.DictReader(io.StringIO(zipped.read('uav-rewards.csv').decode())))
            assert len(rewards) == 64
            assert {row['inventory_index'] for row in rewards} == {'1', '2'}
            assert mean([float(row['reward']) for row in rewards[-16:]]) == pytest.approx(
                status['history'][-1]['mean_step_reward'])


def test_invalid_inventory_and_station_types():
    data = experiment()
    data['scenario']['uav_types'][0]['fixed_count'] = 3
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)
    data = experiment()
    data['scenario']['station_position_types'] = [0, 1]
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)


def test_field_validation_and_shared_dock_configuration(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        data = experiment()
        data['scenario']['required_gsd_m'] = -1
        result = client.post('/api/configuration/check', json=data).json()
        assert not result['valid']
        assert result['errors'][0]['loc'] == ['scenario','required_gsd_m']
        data = experiment()
        data['scenario']['station_types'][0].update(available_count=1, fixed_count=1)
        data['scenario']['station_positions_xy'] = [[20,20]]
        data['scenario']['uav_launch_docks'] = [0,0]
        result = client.post('/api/validate', json=data)
        assert result.status_code == 200
        assert [u['status'] for u in result.json()['frame']['uavs']] == ['charging','queued']


def test_interface_refresh_uses_current_assets_without_caching(tmp_path, monkeypatch):
    from opus_rl.app import server

    frontend = tmp_path / 'frontend'
    frontend.mkdir()
    (frontend / 'index.html').write_text(
        '<link href="/static/style.css"><script src="/static/field-help.js"></script><script src="/static/app.js"></script>', encoding='utf8')
    (frontend / 'style.css').write_text('body {color: white}', encoding='utf8')
    (frontend / 'app.js').write_text('const revision = 1;', encoding='utf8')
    (frontend / 'field-help.js').write_text('const fieldHelp = {};', encoding='utf8')
    monkeypatch.setattr(server, 'FRONTEND', frontend)
    with TestClient(create_app(tmp_path / 'runs')) as client:
        first = client.get('/')
        revision = client.get('/api/health').json()['frontend_revision']
        assert first.headers['cache-control'] == 'no-store'
        assert f'/static/app.js?v={revision}' in first.text
        assert f'/static/style.css?v={revision}' in first.text
        assert f'/static/field-help.js?v={revision}' in first.text
        asset = client.get(f'/static/app.js?v={revision}')
        assert asset.headers['cache-control'] == 'no-store'
        assert asset.text == 'const revision = 1;'
        (frontend / 'app.js').write_text('const revision = 2;', encoding='utf8')
        second = client.get('/')
        next_revision = client.get('/api/health').json()['frontend_revision']
        assert next_revision != revision
        assert f'/static/app.js?v={next_revision}' in second.text
        assert client.get(f'/static/app.js?v={next_revision}').text == 'const revision = 2;'
        (frontend / 'field-help.js').write_text('const fieldHelp = {test: "New help"};', encoding='utf8')
        assert client.get('/api/health').json()['frontend_revision'] != next_revision


def test_individual_uav_presets_persist_through_simulation_and_checkpoint(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        defaults = client.get('/api/defaults').json()
        presets = defaults['presets']
        assert len(presets['uavs']) == 4 and len(presets['docks']) == 2
        data = experiment()
        data['scenario']['uav_types'] = [
            {**presets['uavs'][j % 4]['values'], 'name': f'Survey {j + 1}'} for j in range(3)
        ]
        data['scenario']['uav_launch_docks'] = [0, 0, 1]
        data['scenario']['station_types'] = [
            {**presets['docks'][1]['values'], 'available_count': 2, 'fixed_count': 2}
        ]
        frame = client.post('/api/simulation/reset', json=data).json()
        assert [u['type'] for u in frame['uavs']] == ['Survey 1', 'Survey 2', 'Survey 3']
        assert [u['status'] for u in frame['uavs']] == ['charging', 'queued', 'charging']
        assert client.post('/api/training/start', json=data).status_code == 200
        for _ in range(300):
            status = client.get('/api/training/status').json()
            if status['status'] in ('completed', 'failed'):
                break
            time.sleep(.05)
        assert status['status'] == 'completed', status
        loaded = client.post(f"/api/runs/{status['run_id']}/load", json={'checkpoint': 'final.pt'})
        assert loaded.status_code == 200
        saved = loaded.json()['config']['scenario']
        assert [u['preset_id'] for u in saved['uav_types']] == ['mavic-3e', 'mavic-3t', 'matrice-4e']
        assert [u['name'] for u in saved['uav_types']] == ['Survey 1', 'Survey 2', 'Survey 3']
        assert all(u['available_count'] == u['fixed_count'] == 1 for u in saved['uav_types'])
        assert saved['station_types'][0]['charge_power_W'] == 151.2
