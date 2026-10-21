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
from .timing import Timings


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
    setup_started = time.perf_counter()
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
    started = setup_started
    timings = Timings(device)
    last_frame_at = last_flush_at = started
    last_frame_step = 0
    latest_telemetry = env.telemetry()
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
    timings.record("initialization", time.perf_counter() - started)

    def publish(phase, **extra):
        if callback:
            callback({"run_id": run_id, "steps": steps, "total_steps": cfg.total_steps,
                      "phase": phase, "elapsed_seconds": time.perf_counter() - started,
                      "steps_per_second": steps / max(time.perf_counter() - started, 1e-6),
                      "timings": timings.snapshot(), **extra})

    def checkpoint_file(name):
        target = directory / name
        temporary = target.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config.model_dump(), "steps": steps, "version": __version__,
                    "agent_steps": agent_steps,
                    "algorithm": "ippo", "checkpoint_format": 2,
                    "git_commit": summary["git_commit"], "torch_rng": action_rng.get_state(),
                    "numpy_rng": rng.bit_generator.state}, temporary)
        temporary.replace(target)

    def checkpoint(name):
        publish("checkpointing")
        with timings.measure("checkpoint"):
            checkpoint_file(name)

    try:
        checkpoint("initial.pt")
        with (directory / "metrics.csv").open(
            "w", newline="", encoding="utf8"
        ) as csv_file, (directory / "uav-rewards.csv").open("w", newline="", encoding="utf8") as uav_file, (
            directory / "steps.csv"
        ).open("w", newline="", encoding="utf8") as step_file, (
            directory / "timings.csv"
        ).open("w", newline="", encoding="utf8") as timing_file:
            writer = None
            uav_writer = csv.DictWriter(uav_file, fieldnames=["training_step", "environment", "episode", "episode_step",
                "uav_id", "inventory_index", "name", "reward", "coverage", "energy", "projection", "collision",
                "crash", "energy_Wh", "actor_executed", "battery_ratio", "simulated_seconds"])
            uav_writer.writeheader()
            step_writer = csv.DictWriter(step_file, fieldnames=["steps", "episode", "episode_step", "episode_length",
                "simulated_seconds", "total_simulated_seconds", "mean_coverage", "p10", "fairness",
                "mean_step_reward", "energy_Wh", "crashed_uavs", "collision_pairs"])
            step_writer.writeheader()
            timing_writer = csv.DictWriter(timing_file, fieldnames=["steps", "phase", "calls", "seconds",
                "last_seconds", "max_seconds", "mean_ms", "percent"])
            timing_writer.writeheader()

            def log_timings():
                for phase, row in timings.snapshot().items():
                    timing_writer.writerow({"steps": steps, "phase": phase, **row})
                timing_file.flush()

            if callback:
                with timings.measure("telemetry"):
                    publish("collecting", telemetry=latest_telemetry, frame=env.frame(), frame_step=0,
                            frame_step_delta=0, frame_simulated_delta=0)
            while steps < cfg.total_steps and not stop_event.is_set():
                batch = {k: [] for k in ("obs", "latent", "log_prob", "values", "next_values",
                                        "rewards", "terminated", "ended", "masks", "value_masks")}
                coverage_means, energies, completed_returns, reward_components = [], [], [], []
                for _ in range(min(cfg.rollout_steps, cfg.total_steps - steps)):
                    if stop_event.is_set():
                        break
                    with timings.measure("policy", gpu=True):
                        obs = stack_dict([pack_observation(observation)])
                        actions, latent, probabilities, values = model.act(
                            tensor_dict(obs, device), generator=action_rng)
                        action_array = actions.cpu().numpy()[0]
                    with timings.measure("simulation"):
                        next_obs, _, term, trunc, info = env.step(action_array)
                    for phase, seconds in env.step_timings.items():
                        timings.record("simulation." + phase, seconds)
                    with timings.measure("bootstrap", gpu=True), torch.no_grad():
                        next_observation = stack_dict([pack_observation(next_obs)])
                        next_values = model.value(tensor_dict(next_observation, device)).cpu().numpy()
                    with timings.measure("buffer"):
                        individual = np.asarray(info["uav_rewards"], np.float32)
                        episode_returns += individual
                        count = info["active_uavs"]
                        agent_steps += count
                        coverage_means.append(info["coverage"]["mean"])
                        energies.append(sum(info["energy_Wh"]))
                        reward_components.extend(info["uav_reward_terms"][:count])
                        for key, value in {"obs": obs, "latent": latent.cpu().numpy(),
                                           "log_prob": probabilities.cpu().numpy(), "values": values.cpu().numpy(),
                                           "next_values": next_values, "rewards": individual[None, :],
                                           "terminated": [term], "ended": [term or trunc],
                                           "masks": [info["actor_training_mask"]],
                                           "value_masks": obs["active_mask"]}.items():
                            batch[key].append(value)
                        steps += 1
                        sample = {"steps": steps, "episode": episode + 1, "episode_step": info["step"],
                            "episode_length": config.scenario.horizon, "simulated_seconds": info["simulated_seconds"],
                            "total_simulated_seconds": steps * config.scenario.delta_t,
                            "mean_coverage": info["coverage"]["mean"], "p10": info["coverage"]["p10"],
                            "fairness": info["coverage"]["fairness"], "mean_step_reward": float(individual[:count].mean()),
                            "energy_Wh": energies[-1], "crashed_uavs": info["crashed_uavs"],
                            "collision_pairs": len(info["collision_pairs"])}
                    with timings.measure("logging"):
                        for j, uav in enumerate(env.uavs):
                            uav_writer.writerow({"training_step": steps, "environment": 0,
                                "episode": episode, "episode_step": info["step"], "uav_id": j + 1,
                                "inventory_index": uav.inventory_index + 1, "name": uav.spec.name,
                                "reward": float(individual[j]), **info["uav_reward_terms"][j],
                                "energy_Wh": info["energy_Wh"][j], "actor_executed": info["actor_training_mask"][j],
                                "battery_ratio": uav.battery / uav.spec.battery_capacity_Wh,
                                "simulated_seconds": info["simulated_seconds"]})
                        step_writer.writerow(sample)
                        now = time.perf_counter()
                        if now - last_flush_at >= 1:
                            uav_file.flush()
                            step_file.flush()
                            last_flush_at = now
                    if callback:
                        with timings.measure("telemetry"):
                            latest_telemetry = env.telemetry()
                            frame = {}
                            if now - last_frame_at >= 1 or info["step"] == 1 or term or trunc or steps == cfg.total_steps:
                                interval = steps - last_frame_step
                                frame = {"frame": env.frame(), "frame_step": steps,
                                         "frame_step_delta": interval,
                                         "frame_simulated_delta": interval * config.scenario.delta_t}
                                last_frame_at = now
                                last_frame_step = steps
                            publish("collecting", sample=sample, telemetry=latest_telemetry, **frame)
                    # Publish the final transition before reset, preserving its metrics and UAV readings.
                    if term or trunc:
                        completed_returns.extend(episode_returns[:count].tolist())
                        episode_returns.fill(0)
                        episode += 1
                        with timings.measure("reset"):
                            next_obs, _ = env.reset()
                    observation = next_obs
                if not batch["rewards"]:
                    break
                publish("updating")
                with timings.measure("update", gpu=True):
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
                with timings.measure("logging"):
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=list(stats))
                        writer.writeheader()
                    writer.writerow(stats)
                    csv_file.flush()
                    uav_file.flush()
                    step_file.flush()
                summary.update(steps=steps, agent_steps=agent_steps, metrics=stats)
                publish("updating", metrics=stats)
                evaluation = None
                if steps >= next_evaluation and not stop_event.is_set():
                    publish("evaluating", evaluation_progress=None)
                    with timings.measure("evaluation", gpu=True):
                        evaluation = evaluate(config.scenario, model, cfg.evaluation_episodes,
                            seed=cfg.seed + 10000, device=device, stop_event=stop_event,
                            callback=(lambda progress: publish("evaluating", evaluation_progress=progress))
                            if callback else None)
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
                summary["timings"] = timings.snapshot()
                with timings.measure("logging"):
                    save_json(directory / "summary.json", summary)
                    log_timings()
                publish("collecting", evaluation=evaluation)
            if callback and last_frame_step != steps:
                with timings.measure("telemetry"):
                    interval = steps - last_frame_step
                    publish("checkpointing", frame=env.frame(), frame_step=steps,
                            frame_step_delta=interval, frame_simulated_delta=interval * config.scenario.delta_t)
            checkpoint("final.pt")
            if not (directory / "best.pt").exists():
                checkpoint("best.pt")
            log_timings()
        summary["status"] = "stopped" if stop_event.is_set() else "completed"
        publish(summary["status"])
    except Exception as error:
        summary.update(status="failed", error=str(error))
        if model is not None:
            checkpoint("interrupted.pt")
        raise
    finally:
        summary.update(steps=steps, agent_steps=agent_steps, elapsed_seconds=time.perf_counter() - started,
                       finished_utc=datetime.now(timezone.utc).isoformat(), timings=timings.snapshot())
        save_json(directory / "summary.json", summary)
        env.close()
    return directory, summary
