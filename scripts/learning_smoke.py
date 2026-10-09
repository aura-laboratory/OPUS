"""Bounded actual-learning comparison on a simple fixed scenario; no operational claim."""

import json
from pathlib import Path

from opus_rl.configs import ExperimentConfig
from opus_rl.training.evaluate import evaluate
from opus_rl.training.train import load_checkpoint, save_json, train


def main():
    configuration = ExperimentConfig.model_validate_json(Path("configs/learning-smoke.json").read_text())
    directory, _ = train(configuration, callback=lambda event: print(json.dumps({
        "steps": event["steps"], "coverage": event["metrics"]["mean_coverage"],
        "reward": event["metrics"]["mean_step_reward"]}), flush=True))
    initial, _, _ = load_checkpoint(directory / "initial.pt")
    trained, _, _ = load_checkpoint(directory / "best.pt")
    result = {"run_id": directory.name, "seed": configuration.training.seed,
              "initial": evaluate(configuration.scenario, initial, 3, seed=20000),
              "trained": evaluate(configuration.scenario, trained, 3, seed=20000),
              "random": evaluate(configuration.scenario, episodes=3, seed=20000, policy="random"),
              "heuristic": evaluate(configuration.scenario, episodes=3, seed=20000, policy="heuristic")}
    result["learned_improvement"] = result["trained"]["return_mean"] > result["initial"]["return_mean"]
    result["beats_random"] = result["trained"]["return_mean"] > result["random"]["return_mean"]
    save_json(directory / "learning-comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
