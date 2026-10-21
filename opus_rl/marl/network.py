import threading

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from opus_rl.envs.team_env import CONTEXT_DIM, OWN_DIM

MODEL_INIT_LOCK = threading.RLock()


def tensor_dict(values, device):
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in values.items()}


def pack_observation(observation):
    return {key: (value[0].astype(np.float16) if key == "map" else value.copy())
            for key, value in observation.items()}


def stack_dict(items):
    return {key: np.stack([item[key] for item in items]) for key in items[0]}


def masked_pool(embeddings, mask):
    mask = mask.unsqueeze(-1)
    mean = (embeddings * mask).sum(-2) / mask.sum(-2).clamp_min(1)
    maximum = embeddings.masked_fill(mask == 0, -1e9).amax(-2)
    maximum = torch.where(mask.sum(-2) > 0, maximum, torch.zeros_like(maximum))
    return torch.cat([mean, maximum], dim=-1)


class ActorCritic(nn.Module):
    """One shared policy and one shared local value function, evaluated per UAV."""

    algorithm = "ippo"

    def __init__(self, hidden=128, map_channels=2):
        super().__init__()
        self.hidden = hidden
        self.map_channels = map_channels
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 16, 5, stride=4, padding=2), nn.Tanh(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.Tanh(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(24 * 4 * 4, hidden), nn.Tanh(),
        )
        self.own_encoder = nn.Sequential(nn.Linear(OWN_DIM, hidden), nn.Tanh())
        self.other_encoder = nn.Sequential(nn.Linear(OWN_DIM, hidden // 2), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(hidden * 3 + CONTEXT_DIM, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 5))
        self.log_std = nn.Parameter(torch.full((5,), -0.5))
        self.critic = nn.Sequential(nn.Linear(hidden * 3 + CONTEXT_DIM, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Initial camera points near nadir; this is an explicit initialization, not a learned result.
        self.actor[-1].bias.data[4] = -1.25
        nn.init.orthogonal_(self.critic[-1].weight, gain=1)

    def features(self, observation):
        own = self.own_encoder(observation["uav_own"])
        map_embedding = self.map_encoder(observation["map"])
        map_embedding = map_embedding.unsqueeze(1).expand(-1, own.shape[1], -1)
        entities = self.other_encoder(observation["other_uavs"])
        team = masked_pool(entities, observation["other_uav_mask"])
        return torch.cat([own, map_embedding, team, observation["mission_context"]], dim=-1)

    def distribution(self, observation):
        mean = self.actor(self.features(observation))
        return Normal(mean, self.log_std.clamp(-5, 1).exp().expand_as(mean))

    def value(self, observation):
        return self.critic(self.features(observation)).squeeze(-1) * observation["active_mask"]

    @staticmethod
    def log_probability(distribution, latent):
        # Numerically stable tanh Jacobian. Store raw latent actions, never projected/overridden actions.
        correction = 2 * (np.log(2) - latent - nn.functional.softplus(-2 * latent))
        return (distribution.log_prob(latent) - correction).sum(-1)

    @torch.no_grad()
    def act(self, observation, deterministic=False, generator=None):
        distribution = self.distribution(observation)
        latent = distribution.mean if deterministic else distribution.mean + distribution.stddev * torch.randn(
            distribution.mean.shape, device=distribution.mean.device, generator=generator
        )
        probability = self.log_probability(distribution, latent)
        value = self.value(observation)
        return torch.tanh(latent), latent, probability, value


class LegacyActorCritic(ActorCritic):
    """Read-only compatibility with the original centralized-critic checkpoints."""

    algorithm = "mappo"

    def __init__(self, station_slots, hidden=128):
        super().__init__(hidden, map_channels=3)
        self.station_slots = station_slots
        self.global_uavs = nn.Sequential(nn.Linear(OWN_DIM + station_slots + 1, hidden // 2), nn.Tanh())
        self.global_stations = nn.Sequential(nn.Linear(6, hidden // 2), nn.Tanh())

    def value(self, central):
        uavs = masked_pool(self.global_uavs(central["uavs"]), central["active_mask"])
        stations = masked_pool(self.global_stations(central["stations"]), central["station_mask"])
        image = self.map_encoder(central["map"])
        return self.critic(torch.cat([uavs, stations, image, central["context"]], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def act(self, observation, central=None, deterministic=False, generator=None):
        distribution = self.distribution(observation)
        latent = distribution.mean if deterministic else distribution.mean + distribution.stddev * torch.randn(
            distribution.mean.shape, device=distribution.mean.device, generator=generator
        )
        value = self.value(central) if central is not None else None
        return torch.tanh(latent), latent, self.log_probability(distribution, latent), value
