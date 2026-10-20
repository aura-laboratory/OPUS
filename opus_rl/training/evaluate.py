import numpy as np
import torch

from opus_rl.envs import OpusTeamEnv
from opus_rl.marl.network import pack_observation, stack_dict, tensor_dict


def heuristic_action(env):
    """Independent cyclic sweep at a GSD-feasible altitude; comparison baseline only."""
    actions = np.zeros((env.n_max, 5), np.float32)
    for j, uav in enumerate(env.uavs):
        phase = (env.step_index * 0.05 + j / max(1, len(env.uavs))) % 1
        angles = 2 * np.pi * phase
        target = np.r_[env.area.low + env.area.side * (0.5 + 0.32 * np.array([np.cos(angles), np.sin(angles)])),
                       min(env.config.max_altitude_m, max(env.config.min_altitude_m, env.area.side * 0.65))]
        delta = target - uav.position
        actions[j, :3] = np.clip(delta / 25, -1, 1)
        actions[j, 3], actions[j, 4] = 0, -1
    return actions


def evaluate(scenario, model=None, episodes=2, seed=10000, policy="model", device="cpu",
             stop_event=None, include_frame=False):
    totals, coverages, energy, collisions, crashes, frames, uav_returns = [], [], [], [], [], [], []
    for episode in range(episodes):
        if stop_event and stop_event.is_set():
            break
        env = OpusTeamEnv(scenario)
        observation, info = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        total = energy_used = collision_count = 0
        individual_returns = np.zeros(env.n_max)
        averages = []
        for _ in range(scenario.horizon):
            if stop_event and stop_event.is_set():
                break
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "heuristic":
                action = heuristic_action(env)
            else:
                if model is None:
                    raise ValueError("A trained model is required for model evaluation.")
                packed = tensor_dict(stack_dict([pack_observation(observation)]), device)
                with torch.no_grad():
                    action = model.act(packed, deterministic=True)[0][0].cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            individual_returns += info["uav_rewards"]
            averages.append(info["coverage"]["mean"])
            energy_used += sum(info["energy_Wh"])
            collision_count += len(info["collision_pairs"])
            if terminated or truncated:
                break
        totals.append(total)
        uav_returns.append({"seed": seed + episode, "uavs": [
            {"id": j, "inventory_index": uav.inventory_index, "name": uav.spec.name,
             "return": float(individual_returns[j])} for j, uav in enumerate(env.uavs)]})
        coverages.append([float(np.mean(averages)) if averages else 0, info["coverage"]["p10"],
                          info["coverage"]["fairness"]])
        energy.append(energy_used)
        collisions.append(collision_count)
        crashes.append(info["crashed_uavs"])
        if include_frame and episode == 0:
            frames.append(env.frame())
        env.close()
    if not totals:
        return {"episodes": 0}
    result = {"episodes": len(totals), "return_mean": float(np.mean(totals)),
              "return_std": float(np.std(totals)), "coverage_mean": float(np.mean(coverages, axis=0)[0]),
              "p10_final": float(np.mean(coverages, axis=0)[1]),
              "fairness_final": float(np.mean(coverages, axis=0)[2]), "energy_Wh": float(np.mean(energy)),
              "collision_count": float(np.mean(collisions)), "crashed_uavs": float(np.mean(crashes)),
              "reward_mode": scenario.reward_mode, "uav_returns": uav_returns,
              "seeds": [seed + i for i in range(len(totals))]}
    if frames:
        result["frame"] = frames[0]
    return result
