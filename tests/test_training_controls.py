import pytest
from fastapi.testclient import TestClient

from opus_rl.app import server
from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import TrainConfig
from opus_rl.training import hardware
from opus_rl.training.evaluate import evaluate
from test_environment import small_config


def test_hardware_discovery_is_cached_and_reports_physical_gpu_without_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(hardware, 'cpu_threads', lambda: 12)
    monkeypatch.setattr(hardware.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(hardware, 'windows_gpu_names', lambda: calls.append(True) or ['Radeon RX 580'])
    hardware.hardware_info.cache_clear()
    try:
        first = hardware.hardware_info()
        assert hardware.hardware_info() is first
        assert first['cpu_threads'] == 12
        assert first['gpu_names'] == ['Radeon RX 580']
        assert not first['cuda_available']
        assert len(calls) == 1
    finally:
        hardware.hardware_info.cache_clear()


@pytest.mark.parametrize('requested,available,fits,expected', [
    ('auto', False, False, 'cpu'), ('cuda', False, False, None),
    ('auto', True, False, 'cpu'), ('cuda', True, False, None),
    ('auto', True, True, 'cuda'), ('cpu', True, True, 'cpu'),
])
def test_device_selection_checks_backend_and_available_memory(monkeypatch, requested, available, fits, expected):
    config = ExperimentConfig(training=TrainConfig(device=requested))
    required = hardware.estimated_gpu_bytes(config)
    monkeypatch.setattr(hardware, 'hardware_info', lambda: {'cuda_available': available})
    monkeypatch.setattr(hardware.torch.cuda.memory, 'mem_get_info',
                        lambda: (required * (2 if fits else 0.5), required * 4))
    if expected is None:
        with pytest.raises(ValueError):
            hardware.select_device(config)
    else:
        assert hardware.select_device(config)[0] == expected


def test_evaluation_uses_episode_length_and_ignores_legacy_evaluation_cap():
    config = TrainConfig.model_validate({'evaluation_horizon': 1, 'num_envs': 8})
    assert 'evaluation_horizon' not in config.model_dump()
    assert 'num_envs' not in config.model_dump()
    result = evaluate(small_config(horizon=5), episodes=1, policy='heuristic', include_frame=True)
    assert result['frame']['info']['step'] == 5
    assert result['frame']['info']['simulated_seconds'] == 25


def test_hardware_defaults_and_excess_cpu_threads_have_inline_error(tmp_path, monkeypatch):
    detected = {'cpu_threads': 1, 'gpu_names': [], 'cuda_available': False, 'torch_cuda_build': None}
    monkeypatch.setattr(server, 'hardware_info', lambda: detected)
    with TestClient(server.create_app(tmp_path)) as client:
        defaults = client.get('/api/defaults').json()
        assert defaults['hardware'] == detected
        assert defaults['config']['training']['torch_threads'] == 1
        assert defaults['config']['training']['device'] == 'auto'
        assert 'evaluation_horizon' not in defaults['config']['training']
        assert 'num_envs' not in defaults['config']['training']
        defaults['config']['training']['torch_threads'] = 2
        result = client.post('/api/configuration/check', json=defaults['config']).json()
        assert not result['valid']
        assert result['errors'][0]['loc'] == ['training', 'torch_threads']
        defaults['config']['training'].update(torch_threads=1, device='cuda')

        def unavailable(config):
            raise ValueError('GPU backend unavailable.')

        monkeypatch.setattr(server, 'select_device', unavailable)
        result = client.post('/api/configuration/check', json=defaults['config']).json()
        assert not result['valid']
        assert result['errors'][0]['loc'] == ['training', 'device']


def test_training_defaults_match_the_selected_mission_settings():
    config = ExperimentConfig()
    assert config.scenario.delta_t == 5
    assert config.scenario.horizon == 500
    cfg = config.training
    assert cfg.total_steps == 10000 and cfg.rollout_steps == 100
    assert cfg.evaluation_frequency == cfg.checkpoint_frequency == 2000
    assert cfg.evaluation_episodes == 5 and cfg.seed == 42
    assert cfg.torch_threads == 6 and cfg.device == 'auto'
    with pytest.raises(ValueError, match='multiple of 2'):
        TrainConfig(hidden_size=129)
