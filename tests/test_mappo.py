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
    central = stack_dict([env.central_state(obs)])
    model = ActorCritic(env.s_max, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    actions, latent, logp, values = model.act(tensor_dict(observation, "cpu"), tensor_dict(central, "cpu"))
    assert np.max(np.abs(actions.numpy())) <= 1
    recomputed = model.log_probability(model.distribution(tensor_dict(observation, "cpu")), latent)
    assert torch.allclose(logp, recomputed)
    actor_before = model.actor[-1].weight.detach().clone()
    result = update(model, optimizer, observation, central, latent.numpy(), logp.numpy(), values.numpy(),
                    np.array([1], np.float32), np.array([1], np.float32), np.zeros((1, 2)),
                    TrainConfig(epochs=1, batch_size=1), "cpu", np.random.default_rng(2))
    assert np.isfinite(list(result.values())).all()
    assert torch.equal(actor_before, model.actor[-1].weight)


def test_actor_teammate_set_is_permutation_invariant():
    env = OpusTeamEnv(small_config(uav_types=[UAVType(available_count=3, fixed_count=3)],
        station_types=[StationType(available_count=3, fixed_count=3)],
        station_positions_xy=[(20, 20), (140, 20), (140, 140)]))
    obs, _ = env.reset(seed=2)
    observation = tensor_dict(stack_dict([pack_observation(obs)]), "cpu")
    model = ActorCritic(env.s_max, 32)
    original = model.distribution(observation).mean.detach()
    observation["other_uavs"] = observation["other_uavs"].flip(-2)
    observation["other_uav_mask"] = observation["other_uav_mask"].flip(-1)
    assert torch.allclose(original, model.distribution(observation).mean.detach())


def test_action_generator_is_independent_of_global_rng():
    env = OpusTeamEnv(small_config())
    observation = tensor_dict(stack_dict([pack_observation(env.reset(seed=2)[0])]), "cpu")
    model = ActorCritic(env.s_max, 32)
    first = model.act(observation, generator=torch.Generator().manual_seed(9))[0]
    torch.randn(1000)
    second = model.act(observation, generator=torch.Generator().manual_seed(9))[0]
    assert torch.equal(first, second)
