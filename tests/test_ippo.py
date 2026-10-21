import numpy as np
import torch

from opus_rl.configs.models import TrainConfig, StationType, UAVType
from opus_rl.marl.network import ActorCritic, pack_observation, stack_dict, tensor_dict
from opus_rl.marl.ppo import generalized_advantage, update
from test_environment import small_config
from opus_rl.envs import OpusTeamEnv


def test_gae_bootstraps_truncation_and_stops_reset_carry():
    r, v, nv = np.array([[1], [10]], np.float32), np.zeros((2, 1)), np.array([[5], [9]])
    adv, _ = generalized_advantage(r, v, nv, np.array([[0], [1]]), np.array([[1], [1]]), 0.9, 0.95)
    assert np.allclose(adv, [[5.5], [10]])


def test_action_logprob_consistency_and_zero_actor_mask_update():
    torch.set_num_threads(1)
    env = OpusTeamEnv(small_config())
    obs, _ = env.reset(seed=2)
    observation = stack_dict([pack_observation(obs)])
    model = ActorCritic(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    actions, latent, logp, values = model.act(tensor_dict(observation, "cpu"))
    assert values.shape == (1, 2)
    assert np.max(np.abs(actions.numpy())) <= 1
    recomputed = model.log_probability(model.distribution(tensor_dict(observation, "cpu")), latent)
    assert torch.allclose(logp, recomputed)
    actor_before = model.actor[-1].weight.detach().clone()
    result = update(model, optimizer, observation, latent.numpy(), logp.numpy(), values.numpy(),
                    np.ones((1, 2), np.float32), np.ones((1, 2), np.float32), np.zeros((1, 2)), np.ones((1, 2)),
                    TrainConfig(epochs=1, batch_size=1), "cpu", np.random.default_rng(2))
    assert np.isfinite(list(result.values())).all()
    assert torch.equal(actor_before, model.actor[-1].weight)


def test_actor_teammate_set_is_permutation_invariant():
    env = OpusTeamEnv(small_config(uav_types=[UAVType(available_count=3, fixed_count=3)],
        station_types=[StationType(available_count=3, fixed_count=3)],
        station_positions_xy=[(20, 20), (140, 20), (140, 140)]))
    obs, _ = env.reset(seed=2)
    observation = tensor_dict(stack_dict([pack_observation(obs)]), "cpu")
    model = ActorCritic(32)
    original = model.distribution(observation).mean.detach()
    observation["other_uavs"] = observation["other_uavs"].flip(-2)
    observation["other_uav_mask"] = observation["other_uav_mask"].flip(-1)
    assert torch.allclose(original, model.distribution(observation).mean.detach())


def test_action_generator_is_independent_of_global_rng():
    env = OpusTeamEnv(small_config())
    observation = tensor_dict(stack_dict([pack_observation(env.reset(seed=2)[0])]), "cpu")
    model = ActorCritic(32)
    first = model.act(observation, generator=torch.Generator().manual_seed(9))[0]
    torch.randn(1000)
    second = model.act(observation, generator=torch.Generator().manual_seed(9))[0]
    assert torch.equal(first, second)


def test_individual_gae_keeps_uav_returns_separate_at_truncation_and_reset():
    rewards = np.array([[[1, 10]], [[2, 20]]], np.float32)
    values = np.zeros_like(rewards)
    next_values = np.array([[[5, 50]], [[9, 90]]], np.float32)
    advantages, returns = generalized_advantage(rewards, values, next_values,
        np.array([[0], [1]]), np.array([[1], [1]]), 0.9, 0.95)
    assert np.allclose(advantages, [[[5.5, 55]], [[2, 20]]])
    assert np.array_equal(returns, advantages)


def test_local_value_and_policy_are_shared_across_slots_and_mask_padding():
    env = OpusTeamEnv(small_config(uav_types=[UAVType(available_count=3, fixed_count=2)]))
    observation = tensor_dict(stack_dict([pack_observation(env.reset(seed=2)[0])]), "cpu")
    for key in ("uav_own", "other_uavs", "other_uav_mask", "mission_context"):
        observation[key][:, 1] = observation[key][:, 0]
    model = ActorCritic(32)
    values = model.value(observation)
    assert values.shape == (1, 3)
    assert values[0, 0].item() == values[0, 1].item()
    assert values[0, 2].item() == 0
    means = model.distribution(observation).mean
    assert torch.equal(means[:, 0], means[:, 1])
    assert not hasattr(model, "global_stations")


def test_ppo_uses_each_uavs_advantage_and_excludes_padded_losses():
    class SlotModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.means = torch.nn.Parameter(torch.zeros(3))
            self.values = torch.nn.Parameter(torch.zeros(3))

        def distribution(self, observation):
            return torch.distributions.Normal(self.means[None, :, None], 1)

        def value(self, observation):
            return self.values[None]

        log_probability = staticmethod(ActorCritic.log_probability)

    model = SlotModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    latent = np.ones((1, 3, 1), np.float32)
    old_logp = model.log_probability(model.distribution({}), torch.from_numpy(latent)).detach().numpy()
    stats = update(model, optimizer, {}, latent, old_logp, np.zeros((1, 3)),
        np.array([[1, -1, 10000]]), np.array([[1, -1, 10000]]), np.array([[1, 1, 0]]),
        np.array([[1, 1, 0]]), TrainConfig(epochs=1, batch_size=1, entropy_coef=0, max_grad_norm=100),
        "cpu", np.random.default_rng(2))
    assert model.means[0] > 0 and model.means[1] < 0
    assert model.values[0] > 0 and model.values[1] < 0
    assert model.means[2] == 0 and model.values[2] == 0
    assert stats["value_loss"] == 0.5


def test_legacy_checkpoint_loading_preserves_policy_reward_and_global_rng(tmp_path):
    from opus_rl.configs import ExperimentConfig
    from opus_rl.marl.network import LegacyActorCritic
    from opus_rl.training.train import load_checkpoint

    config = ExperimentConfig(scenario=small_config(observation_version=1), training=TrainConfig(hidden_size=32))
    env = OpusTeamEnv(config.scenario)
    observation = tensor_dict(stack_dict([pack_observation(env.reset(seed=2)[0])]), "cpu")
    original = LegacyActorCritic(env.s_max, 32)
    expected = original.act(observation, deterministic=True)[0]
    path = tmp_path / 'old.pt'
    torch.save({"model": original.state_dict(), "config": config.model_dump()}, path)
    rng_before = torch.random.get_rng_state()
    loaded, recovered, _ = load_checkpoint(path)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert loaded.algorithm == "mappo" and recovered.scenario.reward_mode == "legacy_team"
    assert torch.equal(expected, loaded.act(observation, deterministic=True)[0])


def test_single_environment_updates_use_individual_rewards_and_ignore_team_scalar(tmp_path, monkeypatch):
    import csv
    from collections import Counter
    from statistics import mean

    from opus_rl.configs import ExperimentConfig
    from opus_rl.training.train import load_checkpoint, train

    original_step = OpusTeamEnv.step

    def unrelated_scalar(self, actions):
        observation, _, terminated, truncated, info = original_step(self, actions)
        return observation, 1e6, terminated, truncated, info

    monkeypatch.setattr(OpusTeamEnv, 'step', unrelated_scalar)
    config = ExperimentConfig(scenario=small_config(horizon=4, fleet_mode='variable',
        uav_types=[UAVType(available_count=3, fixed_count=2)]), training=TrainConfig(
        total_steps=23, rollout_steps=4, batch_size=2, hidden_size=32, epochs=1,
        torch_threads=1, evaluation_frequency=1000, checkpoint_frequency=1000))
    directory, summary = train(config, tmp_path, output_name='single')
    with (directory / 'uav-rewards.csv').open(newline='', encoding='utf8') as saved:
        rows = list(csv.DictReader(saved))
    assert summary['status'] == 'completed'
    assert summary['steps'] == 23  # A partial final update cannot overshoot the budget.
    assert len(rows) == summary['agent_steps']
    assert {row['environment'] for row in rows} == {'0'}
    counts = list(Counter(row['training_step'] for row in rows).values())
    assert len(set(counts)) > 1  # Fleet size varies after resets, with padded slots excluded.
    last_batch = [float(row['reward']) for row in rows if int(row['training_step']) > 20]
    assert np.isclose(summary['metrics']['mean_step_reward'], mean(last_batch))
    assert abs(summary['metrics']['mean_step_reward']) < 1
    model, _, data = load_checkpoint(directory / 'final.pt')
    assert data['algorithm'] == model.algorithm == 'ippo'
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
