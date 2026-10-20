import csv
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from opus_rl import __version__
from opus_rl.configs import ExperimentConfig
from opus_rl.envs import OpusTeamEnv
from opus_rl.marl.network import (
    MODEL_INIT_LOCK, ActorCritic, LegacyActorCritic, pack_observation, stack_dict, tensor_dict,
)
from opus_rl.marl.ppo import generalized_advantage, update
from .evaluate import evaluate
from .hardware import hardware_info, select_device


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def git_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=False)  # Only load local, trusted OPUS checkpoints.
    config = ExperimentConfig.model_validate(data["config"])
    algorithm = data.get("algorithm", "mappo")
    if algorithm not in ("ippo", "mappo"):
        raise ValueError(f"Unsupported checkpoint algorithm: {algorithm}.")
    config.scenario.reward_mode = "individual" if algorithm == "ippo" else "legacy_team"
    with MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        model = (ActorCritic(config.training.hidden_size) if algorithm == "ippo" else LegacyActorCritic(
            sum(t.available_count for t in config.scenario.station_types), config.training.hidden_size))
    model.load_state_dict(data["model"])
    model.to(device).eval()
    return model, config, data


def train(config: ExperimentConfig, runs_root="runs", callback=None, stop_event=None, output_name=None):
    config = config.model_copy(deep=True)
    config.scenario.reward_mode = "individual"  # New runs always use IPPO, including from legacy run configs.
    cfg = config.training
    hardware = hardware_info()
    cfg.torch_threads = min(cfg.torch_threads, hardware["cpu_threads"])
    device, device_reason = select_device(config)
    stop_event = stop_event or threading.Event()
    run_id = output_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    if Path(run_id).name != run_id:
        raise ValueError("Run name must be a single directory name.")
    directory = Path(runs_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "config.json", config.model_dump())
    save_json(directory / "normalization.json", {"method": "fixed physical feature scales; no running statistics",
               "algorithm": "parameter-sharing IPPO", "value": "one local value estimate per UAV observation",
               "reward": "signed marginal capture contribution minus individual costs",
               "map": "occupancy fraction, mean and p10 quality", "actions": "tanh Gaussian in [-1,1]",
               "map_buffer_dtype": "float16, encoded once per environment per transition"})
    torch.set_num_threads(cfg.torch_threads)
    rng = np.random.default_rng(cfg.seed)
    env = OpusTeamEnv(config.scenario)
    observation, _ = env.reset(seed=cfg.seed)
    with MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model = ActorCritic(cfg.hidden_size).to(device)
    action_rng = torch.Generator(device=device).manual_seed(cfg.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    started = time.perf_counter()
    steps, update_index, next_evaluation, next_checkpoint = 0, 0, cfg.evaluation_frequency, cfg.checkpoint_frequency
    best_return = -np.inf
    episode_returns = np.zeros(env.n_max)
    episode = 0
    agent_steps = 0
    summary = {"run_id": run_id, "status": "running", "steps": 0, "version": __version__,
               "agent_steps": 0,
               "algorithm": "ippo", "reward_mode": "individual",
               "git_commit": git_version(), "seed": cfg.seed, "device": device,
               "device_reason": device_reason, "hardware": hardware,
               "started_utc": datetime.now(timezone.utc).isoformat(), "evaluation": None}
    save_json(directory / "summary.json", summary)

    def checkpoint(name):
        target = directory / name
        temporary = target.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config.model_dump(), "steps": steps, "version": __version__,
                    "agent_steps": agent_steps,
                    "algorithm": "ippo", "checkpoint_format": 2,
                    "git_commit": summary["git_commit"], "torch_rng": action_rng.get_state(),
                    "numpy_rng": rng.bit_generator.state}, temporary)
        temporary.replace(target)

    try:
        checkpoint("initial.pt")
        with (directory / "metrics.csv").open(
            "w", newline="", encoding="utf8"
        ) as csv_file, (directory / "uav-rewards.csv").open("w", newline="", encoding="utf8") as uav_file:
            writer = None
            uav_writer = csv.DictWriter(uav_file, fieldnames=["training_step", "environment", "episode", "episode_step",
                "uav_id", "inventory_index", "name", "reward", "coverage", "energy", "projection", "collision",
                "crash", "energy_Wh", "actor_executed"])
            uav_writer.writeheader()
            while steps < cfg.total_steps and not stop_event.is_set():
                batch = {k: [] for k in ("obs", "latent", "log_prob", "values", "next_values",
                                        "rewards", "terminated", "ended", "masks", "value_masks")}
                coverage_means, energies, completed_returns, reward_components = [], [], [], []
                for _ in range(min(cfg.rollout_steps, cfg.total_steps - steps)):
                    if stop_event.is_set():
                        break
                    obs = stack_dict([pack_observation(observation)])
                    actions, latent, probabilities, values = model.act(
                        tensor_dict(obs, device), generator=action_rng
                    )
                    next_obs, _, term, trunc, info = env.step(actions.cpu().numpy()[0])
                    next_observation = stack_dict([pack_observation(next_obs)])
                    with torch.no_grad():
                        next_values = model.value(tensor_dict(next_observation, device)).cpu().numpy()
                    individual = np.asarray(info["uav_rewards"], np.float32)
                    episode_returns += individual
                    count = info["active_uavs"]
                    agent_steps += count
                    for j, uav in enumerate(env.uavs):
                        uav_writer.writerow({"training_step": steps + 1, "environment": 0,
                            "episode": episode, "episode_step": info["step"], "uav_id": j + 1,
                            "inventory_index": uav.inventory_index + 1, "name": uav.spec.name,
                            "reward": float(individual[j]), **info["uav_reward_terms"][j],
                            "energy_Wh": info["energy_Wh"][j], "actor_executed": info["actor_training_mask"][j]})
                    if term or trunc:
                        completed_returns.extend(episode_returns[:count].tolist())
                        episode_returns.fill(0)
                        episode += 1
                        next_obs, _ = env.reset()
                    coverage_means.append(info["coverage"]["mean"])
                    energies.append(sum(info["energy_Wh"]))
                    reward_components.extend(info["uav_reward_terms"][:count])
                    for key, value in {"obs": obs, "latent": latent.cpu().numpy(),
                                       "log_prob": probabilities.cpu().numpy(), "values": values.cpu().numpy(),
                                       "next_values": next_values, "rewards": individual[None, :], "terminated": [term],
                                       "ended": [term or trunc], "masks": [info["actor_training_mask"]],
                                       "value_masks": obs["active_mask"]}.items():
                        batch[key].append(value)
                    observation = next_obs
                    steps += 1
                if not batch["rewards"]:
                    break
                arrays = {key: np.asarray(batch[key], np.float32) for key in batch if key != "obs"}
                advantage, returns = generalized_advantage(arrays["rewards"], arrays["values"], arrays["next_values"],
                    arrays["terminated"], arrays["ended"], cfg.gamma, cfg.gae_lambda)
                def flatten(a):
                    return a.reshape(-1, *a.shape[2:])
                obs = {key: np.concatenate([item[key] for item in batch["obs"]]) for key in obs}
                stats = update(model, optimizer, obs, flatten(arrays["latent"]),
                    flatten(arrays["log_prob"]), flatten(arrays["values"]), flatten(advantage),
                    flatten(returns), flatten(arrays["masks"]), flatten(arrays["value_masks"]), cfg, device, rng)
                elapsed = time.perf_counter() - started
                update_index += 1
                stats.update({"steps": steps, "update": update_index, "elapsed_seconds": elapsed,
                    "agent_steps": agent_steps, "steps_per_second": steps / max(elapsed, 1e-6),
                    "mean_step_reward": float(arrays["rewards"][arrays["value_masks"] > 0].mean()),
                    "mean_coverage": float(np.mean(coverage_means)), "energy_Wh_per_step": float(np.mean(energies)),
                    "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else 0,
                    "actor_sample_fraction": float(arrays["masks"].sum() / arrays["value_masks"].sum()),
                    "explained_variance": float(1 - np.var((returns - arrays["values"])[arrays["value_masks"] > 0]) /
                                                 (np.var(returns[arrays["value_masks"] > 0]) + 1e-8))})
                for key in ("coverage", "energy", "projection", "collision", "crash"):
                    stats[f"reward_{key}"] = float(np.mean([terms[key] for terms in reward_components]))
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(stats))
                    writer.writeheader()
                writer.writerow(stats)
                csv_file.flush()
                uav_file.flush()
                summary.update(steps=steps, agent_steps=agent_steps, metrics=stats)
                evaluation = None
                if steps >= next_evaluation and not stop_event.is_set():
                    evaluation = evaluate(config.scenario, model, cfg.evaluation_episodes,
                                          seed=cfg.seed + 10000, device=device, stop_event=stop_event, include_frame=True)
                    if evaluation.get("episodes", 0):
                        clean = {key: value for key, value in evaluation.items() if key != "frame"}
                        summary["evaluation"] = clean
                        save_json(directory / f"evaluation-{steps}.json", clean)
                        if clean["return_mean"] > best_return:
                            best_return = clean["return_mean"]
                            checkpoint("best.pt")
                    next_evaluation += cfg.evaluation_frequency
                if steps >= next_checkpoint:
                    checkpoint(f"checkpoint-{steps}.pt")
                    next_checkpoint += cfg.checkpoint_frequency
                save_json(directory / "summary.json", summary)
                if callback:
                    callback({"run_id": run_id, "steps": steps, "total_steps": cfg.total_steps,
                              "metrics": stats, "evaluation": evaluation, "frame": env.frame()})
            checkpoint("final.pt")
            if not (directory / "best.pt").exists():
                checkpoint("best.pt")
        summary["status"] = "stopped" if stop_event.is_set() else "completed"
    except Exception as error:
        summary.update(status="failed", error=str(error))
        if model is not None:
            checkpoint("interrupted.pt")
        raise
    finally:
        summary.update(steps=steps, agent_steps=agent_steps, elapsed_seconds=time.perf_counter() - started,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(directory / "summary.json", summary)
        env.close()
    return directory, summary
