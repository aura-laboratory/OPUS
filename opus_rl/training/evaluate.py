import time

import numpy as np
import torch

from opus_rl.envs import OpusTeamEnv
from opus_rl.marl.network import pack_observation, stack_dict, tensor_dict
from opus_rl.simulation.trajectory import Trajectories


def heuristic_action(env):
    """Independent circular targets; comparison baseline only."""
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
             stop_event=None, include_frame=False, callback=None, trail_length_m=0, visual_history=None):
    totals, coverages, energy, collisions, crashes, frames, uav_returns = [], [], [], [], [], [], []
    for episode in range(episodes):
        if stop_event and stop_event.is_set():
            break
        env = OpusTeamEnv(scenario)
        observation, info = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        trails = Trajectories()
        last_frame_at = time.perf_counter()
        if visual_history is not None and episode == 0:
            visual_history.append(0, 1, 0, env.telemetry(), env.frame())
        if include_frame and episode == 0 and trail_length_m:
            trails.record(env)
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
            if include_frame and episode == 0 and trail_length_m:
                trails.record(env)
            if visual_history is not None and episode == 0:
                now = time.perf_counter()
                frame = env.frame() if now - last_frame_at >= 1 or info["step"] == 1 or terminated or truncated else None
                if frame is not None:
                    last_frame_at = now
                visual_history.append(info["step"], 1, info["simulated_seconds"], env.telemetry(), frame)
            if callback:
                callback({"episode": episode + 1, "episodes": episodes, "step": info["step"],
                          "episode_length": scenario.horizon})
            total += reward
            individual_returns += info["uav_rewards"]
            averages.append(info["coverage"]["mean"])
            energy_used += sum(info["energy_Wh"])
            collision_count += len(info["collision_pairs"])
            if terminated or truncated:
                break
        if stop_event and stop_event.is_set() and not env.finished:
            env.close()
            break  # An interrupted episode is not a completed evaluation sample.
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
            frames.append(visual_history.frame(env.step_index, trail_length_m) if visual_history is not None else {
                **env.frame(), "episode": 1, "trails": trails.snapshot(env.step_index, trail_length_m)})
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
