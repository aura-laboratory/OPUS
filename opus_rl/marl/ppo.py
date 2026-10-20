import numpy as np
import torch

from .network import tensor_dict


def generalized_advantage(rewards, values, next_values, terminated, ended, gamma, gae_lambda):
    """Bootstrap time-limit truncations, but never propagate GAE across an episode reset."""
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    # Team episode boundaries apply to every UAV, while rewards/values stay distinct.
    terminated, ended = np.asarray(terminated), np.asarray(ended)
    terminated = terminated.reshape(*terminated.shape, *([1] * (rewards.ndim - terminated.ndim)))
    ended = ended.reshape(*ended.shape, *([1] * (rewards.ndim - ended.ndim)))
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        carry = delta + gamma * gae_lambda * (1 - ended[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def update(model, optimizer, observation, latent, old_log_prob, old_values,
           advantages, returns, masks, value_masks, cfg, device, rng):
    observation = tensor_dict(observation, device)
    latent, old_log_prob, old_values, advantages, returns, masks, value_masks = [
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in (latent, old_log_prob, old_values, advantages, returns, masks, value_masks)
    ]
    active_advantages = advantages[masks > 0]
    if active_advantages.numel():
        advantages = (advantages - active_advantages.mean()) / active_advantages.std(unbiased=False).clamp_min(1e-6)
    count, records, stop_early = len(returns), [], False
    for _ in range(cfg.epochs):
        for start in range(0, count, cfg.batch_size):
            # Shuffle once per epoch below; each sample appears exactly once.
            if start == 0:
                permutation = rng.permutation(count)
            indexes = torch.as_tensor(permutation[start:start + cfg.batch_size], device=device)
            batch_obs = {key: value[indexes] for key, value in observation.items()}
            distribution = model.distribution(batch_obs)
            log_prob = model.log_probability(distribution, latent[indexes])
            batch_mask = masks[indexes]
            denominator = batch_mask.sum().clamp_min(1)
            log_ratio = log_prob - old_log_prob[indexes]
            ratio = log_ratio.exp()
            advantage = advantages[indexes]
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantage
            actor_loss = -(torch.minimum(unclipped, clipped) * batch_mask).sum() / denominator
            entropy = (distribution.entropy().sum(-1) * batch_mask).sum() / denominator
            value = model.value(batch_obs)
            clipped_value = old_values[indexes] + (value - old_values[indexes]).clamp(-cfg.clip_range, cfg.clip_range)
            value_error = torch.maximum((value - returns[indexes])**2,
                                        (clipped_value - returns[indexes])**2)
            value_loss = 0.5 * (value_error * value_masks[indexes]).sum() / value_masks[indexes].sum().clamp_min(1)
            loss = actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = (((ratio - 1) - log_ratio) * batch_mask).sum() / denominator
                fraction = ((torch.abs(ratio - 1) > cfg.clip_range).float() * batch_mask).sum() / denominator
            records.append([actor_loss.item(), value_loss.item(), entropy.item(), kl.item(), fraction.item()])
            if kl.item() > cfg.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    stats = np.mean(records, axis=0)
    return dict(zip(["actor_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction"], stats.tolist()))
