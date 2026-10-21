import csv

import numpy as np
import pytest
import torch

from opus_rl.configs import ExperimentConfig
from opus_rl.configs.models import LegacyRewardConfig, RewardConfig, TrainConfig
from opus_rl.envs import OpusTeamEnv
from opus_rl.marl.network import ActorCritic, pack_observation, stack_dict, tensor_dict
from opus_rl.simulation.coverage import fuse, metrics, reward
from opus_rl.training.train import load_checkpoint, train
from test_environment import small_config


def test_current_map_and_rewards_do_not_calculate_percentiles(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Current coverage must not calculate or sort percentiles.')

    monkeypatch.setattr(np, 'percentile', forbidden)
    monkeypatch.setattr(np, 'sort', forbidden)
    env = OpusTeamEnv(small_config())
    observation, info = env.reset(seed=4)
    assert observation['map'].shape[1] == 2
    assert 'p10' not in info['coverage']
    assert 'p10_weight' not in env.config.reward.model_dump()
    assert ActorCritic(32).map_encoder[0].in_channels == 2
    values = np.linspace(0, 1, 64).reshape(8, 8)
    mask = np.ones((8, 8), bool)
    assert fuse(values, mask, 8).shape == (2, 8, 8)
    coverage = metrics(values, mask)
    score, _ = reward(RewardConfig(), coverage, 0, 0, 0, 0)
    assert score == pytest.approx(0.5 * coverage['mean'] + 0.4 * coverage['fairness'])
    assert RewardConfig(mean_weight=0.5, fairness_weight=0.4).mean_weight == 0.5


def test_old_ippo_checkpoint_preserves_its_channels_reward_and_actions(tmp_path):
    config = ExperimentConfig(scenario=small_config(observation_version=1,
        reward=LegacyRewardConfig()), training=TrainConfig(hidden_size=32))
    original = ActorCritic(32, map_channels=3)
    old_config = config.model_dump()
    old_config['scenario'].pop('observation_version')  # Earlier files did not have a version flag.
    path = tmp_path / 'old-ippo.pt'
    torch.save({'model': original.state_dict(), 'config': old_config, 'algorithm': 'ippo'}, path)
    recovered, restored, _ = load_checkpoint(path)
    env = OpusTeamEnv(restored.scenario)
    observation, info = env.reset(seed=3)
    assert observation['map'].shape[1] == recovered.map_channels == 3
    assert 'p10' in info['coverage'] and restored.scenario.reward.p10_weight == 0.1
    packed = tensor_dict(stack_dict([pack_observation(observation)]), 'cpu')
    assert torch.equal(original.act(packed, deterministic=True)[0], recovered.act(packed, deterministic=True)[0])
    assert restored.scenario.reward.model_dump()['p10_weight'] == 0.1


def test_training_an_old_scenario_creates_a_current_checkpoint_without_p10(tmp_path):
    config = ExperimentConfig(scenario=small_config(observation_version=1,
        reward=LegacyRewardConfig()), training=TrainConfig(total_steps=3, rollout_steps=2,
        hidden_size=32, epochs=1, torch_threads=1, evaluation_frequency=1000))
    directory, _ = train(config, tmp_path, output_name='current')
    model, recovered, data = load_checkpoint(directory / 'final.pt')
    assert data['checkpoint_format'] == 3 and model.map_channels == 2
    assert recovered.scenario.observation_version == 2
    assert 'p10_weight' not in recovered.scenario.reward.model_dump()
    assert recovered.scenario.reward.mean_weight == 0.5 and recovered.scenario.reward.fairness_weight == 0.4
    with (directory / 'steps.csv').open(newline='') as saved:
        assert 'p10' not in csv.DictReader(saved).fieldnames
    assert config.scenario.observation_version == 1  # The source configuration remains reusable.
