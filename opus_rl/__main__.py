import argparse
import json
import threading
import webbrowser
from pathlib import Path

from opus_rl.configs import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(prog="opus", description="OPUS local multi-UAV research platform")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local mission and training application")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    serve.add_argument("--runs-root", type=Path)
    configuration = sub.add_parser("config", help="Write the full default experiment configuration")
    configuration.add_argument("--output", type=Path, default=Path("configs/default.json"))
    training = sub.add_parser("train", help="Train MAPPO using a JSON experiment configuration")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--runs-root", type=Path, default=Path("runs"))
    evaluation = sub.add_parser("evaluate", help="Evaluate a local OPUS checkpoint")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--episodes", type=int, default=2)
    evaluation.add_argument("--policy", choices=["model", "random", "heuristic"], default="model")
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        import httpx
        from opus_rl.app import frontend_revision
        url = f"http://127.0.0.1:{args.port}"
        page_url = f"{url}/?v={frontend_revision()}"
        try:
            health = httpx.get(url + "/api/health", timeout=1, trust_env=False)
            if health.is_success and health.json().get("local_only"):
                print(f"OPUS is already running at {url}. Use stop.ps1 to stop this project's service.")
                if not args.no_browser:
                    webbrowser.open(page_url)
                return
        except (httpx.HTTPError, ValueError):
            pass
        import uvicorn
        from opus_rl.app.server import create_app
        if not args.no_browser:
            timer = threading.Timer(1.5, lambda: webbrowser.open(page_url))
            timer.daemon = True
            timer.start()
        uvicorn.run(create_app(args.runs_root), host="127.0.0.1", port=args.port)
    elif args.command == "config":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(ExperimentConfig().model_dump_json(indent=2), encoding="utf8")
        print(args.output.resolve())
    elif args.command == "train":
        from opus_rl.training.train import train
        config = ExperimentConfig.model_validate_json(args.config.read_text(encoding="utf8"))
        stop = threading.Event()
        import signal
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        directory, summary = train(config, args.runs_root,
            lambda event: print(json.dumps({k: event[k] for k in ("steps", "total_steps", "metrics")})), stop)
        print(json.dumps({"directory": str(directory.resolve()), "summary": summary}, indent=2))
    else:
        from opus_rl.training.train import load_checkpoint
        from opus_rl.training.evaluate import evaluate
        import torch
        model, config, _ = load_checkpoint(args.checkpoint)
        torch.set_num_threads(config.training.torch_threads)
        result = evaluate(config.scenario, model, args.episodes, policy=args.policy,
                          seed=config.training.seed + 10000)
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text, encoding="utf8")
        print(text)


if __name__ == "__main__":
    main()
