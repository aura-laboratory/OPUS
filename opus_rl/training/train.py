import csv
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from opus_rl import __version__
from opus_rl.configs import ExperimentConfig
from opus_rl.envs import OpusTeamEnv
from opus_rl.marl.network import MODEL_INIT_LOCK, ActorCritic, pack_observation, stack_dict, tensor_dict
from opus_rl.marl.ppo import generalized_advantage, update
from .evaluate import evaluate


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
    with MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        model = ActorCritic(sum(t.available_count for t in config.scenario.station_types), config.training.hidden_size)
    model.load_state_dict(data["model"])
    model.to(device).eval()
    return model, config, data


def train(config: ExperimentConfig, runs_root="runs", callback=None, stop_event=None, output_name=None):
    cfg = config.training
    stop_event = stop_event or threading.Event()
    run_id = output_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    if Path(run_id).name != run_id:
        raise ValueError("Run name must be a single directory name.")
    directory = Path(runs_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "config.json", config.model_dump())
    save_json(directory / "normalization.json", {"method": "fixed physical feature scales; no running statistics",
               "map": "occupancy fraction, mean and p10 quality", "actions": "tanh Gaussian in [-1,1]",
               "map_buffer_dtype": "float16, encoded once per environment per transition"})
    torch.set_num_threads(cfg.torch_threads)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this environment; use CPU or auto.")
    device = "cuda" if cfg.device in ("cuda", "auto") and torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)
    envs = [OpusTeamEnv(config.scenario) for _ in range(cfg.num_envs)]
    observations = [env.reset(seed=cfg.seed + i)[0] for i, env in enumerate(envs)]
    with MODEL_INIT_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model = ActorCritic(envs[0].s_max, cfg.hidden_size).to(device)
    action_rng = torch.Generator(device=device).manual_seed(cfg.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    started = time.perf_counter()
    steps, update_index, next_evaluation, next_checkpoint = 0, 0, cfg.evaluation_frequency, cfg.checkpoint_frequency
    best_return, episode_returns = -np.inf, np.zeros(cfg.num_envs)
    summary = {"run_id": run_id, "status": "running", "steps": 0, "version": __version__,
               "git_commit": git_version(), "seed": cfg.seed, "device": device,
               "started_utc": datetime.now(timezone.utc).isoformat(), "evaluation": None}
    save_json(directory / "summary.json", summary)

    def checkpoint(name):
        target = directory / name
        temporary = target.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config.model_dump(), "steps": steps, "version": __version__,
                    "git_commit": summary["git_commit"], "torch_rng": action_rng.get_state(),
                    "numpy_rng": rng.bit_generator.state}, temporary)
        temporary.replace(target)

    def collect(pair):
        env, action = pair
        obs, r, terminal, truncation, info = env.step(action)
        return obs, r, terminal, truncation, info, env.central_state(obs)

    try:
        checkpoint("initial.pt")
        with ThreadPoolExecutor(max_workers=cfg.num_envs) as pool, (directory / "metrics.csv").open(
            "w", newline="", encoding="utf8"
        ) as csv_file:
            writer = None
            while steps < cfg.total_steps and not stop_event.is_set():
                batch = {k: [] for k in ("obs", "central", "latent", "log_prob", "values", "next_values",
                                        "rewards", "terminated", "ended", "masks")}
                coverage_means, energies, completed_returns, reward_components = [], [], [], []
                for _ in range(min(cfg.rollout_steps, (cfg.total_steps - steps + cfg.num_envs - 1) // cfg.num_envs)):
                    if stop_event.is_set():
                        break
                    obs = stack_dict([pack_observation(o) for o in observations])
                    central = stack_dict([env.central_state(o) for env, o in zip(envs, observations)])
                    central["map"] = obs["map"]
                    actions, latent, probabilities, values = model.act(
                        tensor_dict(obs, device), tensor_dict(central, device), generator=action_rng
                    )
                    transitions = list(pool.map(collect, zip(envs, actions.cpu().numpy())))
                    new_obs, rewards, terminal, ended, masks = [], [], [], [], []
                    next_central = stack_dict([transition[5] for transition in transitions])
                    with torch.no_grad():
                        next_values = model.value(tensor_dict(next_central, device)).cpu().numpy()
                    for i, (observation, r, term, trunc, info, _) in enumerate(transitions):
                        episode_returns[i] += r
                        if term or trunc:
                            completed_returns.append(float(episode_returns[i]))
                            episode_returns[i] = 0
                            observation = envs[i].reset()[0]
                        new_obs.append(observation)
                        rewards.append(r)
                        terminal.append(term)
                        ended.append(term or trunc)
                        masks.append(info["actor_training_mask"])
                        coverage_means.append(info["coverage"]["mean"])
                        energies.append(sum(info["energy_Wh"]))
                        reward_components.append(info["reward_terms"])
                    for key, value in {"obs": obs, "central": central, "latent": latent.cpu().numpy(),
                                       "log_prob": probabilities.cpu().numpy(), "values": values.cpu().numpy(),
                                       "next_values": next_values, "rewards": rewards, "terminated": terminal,
                                       "ended": ended, "masks": masks}.items():
                        batch[key].append(value)
                    observations = new_obs
                    steps += cfg.num_envs
                if not batch["rewards"]:
                    break
                arrays = {key: np.asarray(batch[key], np.float32) for key in batch if key not in ("obs", "central")}
                advantage, returns = generalized_advantage(arrays["rewards"], arrays["values"], arrays["next_values"],
                    arrays["terminated"], arrays["ended"], cfg.gamma, cfg.gae_lambda)
                def flatten(a):
                    return a.reshape(-1, *a.shape[2:])
                obs = {key: np.concatenate([item[key] for item in batch["obs"]]) for key in obs}
                central = {key: np.concatenate([item[key] for item in batch["central"]]) for key in central}
                stats = update(model, optimizer, obs, central, flatten(arrays["latent"]),
                    flatten(arrays["log_prob"]), flatten(arrays["values"]), flatten(advantage),
                    flatten(returns), flatten(arrays["masks"]), cfg, device, rng)
                elapsed = time.perf_counter() - started
                update_index += 1
                stats.update({"steps": steps, "update": update_index, "elapsed_seconds": elapsed,
                    "steps_per_second": steps / max(elapsed, 1e-6), "mean_step_reward": float(arrays["rewards"].mean()),
                    "mean_coverage": float(np.mean(coverage_means)), "energy_Wh_per_step": float(np.mean(energies)),
                    "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else 0,
                    "actor_sample_fraction": float(arrays["masks"].mean()),
                    "explained_variance": float(1 - np.var(flatten(returns) - flatten(arrays["values"])) /
                                                 (np.var(flatten(returns)) + 1e-8))})
                for key in ("coverage", "energy", "projection", "collision", "crash"):
                    stats[f"reward_{key}"] = float(np.mean([terms[key] for terms in reward_components]))
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(stats))
                    writer.writeheader()
                writer.writerow(stats)
                csv_file.flush()
                summary.update(steps=steps, metrics=stats)
                evaluation = None
                if steps >= next_evaluation and not stop_event.is_set():
                    evaluation = evaluate(config.scenario, model, cfg.evaluation_episodes, cfg.evaluation_horizon,
                                          cfg.seed + 10000, device=device, stop_event=stop_event, include_frame=True)
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
                              "metrics": stats, "evaluation": evaluation, "frame": envs[0].frame()})
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
        summary.update(steps=steps, elapsed_seconds=time.perf_counter() - started,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(directory / "summary.json", summary)
        for env in envs:
            env.close()
    return directory, summary
