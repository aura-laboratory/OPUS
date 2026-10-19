import asyncio
import io
import json
import threading
import time
import traceback
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from opus_rl.configs import ExperimentConfig, ScenarioConfig
from opus_rl.envs import OpusTeamEnv
from opus_rl.training.evaluate import evaluate, heuristic_action
from opus_rl.training.train import load_checkpoint, save_json, train

FRONTEND = Path(__file__).parent / "frontend"


class State:
    def __init__(self, runs_root):
        self.runs_root = Path(runs_root).resolve()
        self.lock = threading.RLock()
        self.simulation = None
        self.simulation_model = None
        self.worker = None
        self.stop = threading.Event()
        self.status = {"status": "idle", "history": []}
        self.geocode_lock = asyncio.Lock()
        self.last_geocode = 0.0
        self.geocode_cache = {}

    def publish(self, update):
        with self.lock:
            previous = self.status
            history = previous.get("history", []) + [update["metrics"]]
            self.status = {**previous, **update, "status": "stopping" if self.stop.is_set() else "running",
                           "history": history[-1000:]}

    def run_directory(self, run_id):
        if Path(run_id).name != run_id or run_id in (".", ".."):
            raise HTTPException(400, "Invalid run identifier.")
        directory = (self.runs_root / run_id).resolve()
        if directory.parent != self.runs_root or not directory.is_dir():
            raise HTTPException(404, "Run does not exist.")
        return directory


def create_app(runs_root=None):
    state = State(runs_root or Path(__file__).parents[2] / "runs")

    @asynccontextmanager
    async def lifespan(app):
        yield
        state.stop.set()
        if state.worker and state.worker.is_alive():
            await asyncio.to_thread(state.worker.join, 10)

    app = FastAPI(title="OPUS v1", lifespan=lifespan)
    app.state.opus = state
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Use the local OPUS page to submit requests."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def configuration_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "local_only": True}

    @app.get("/api/defaults")
    def defaults():
        config = ExperimentConfig()
        return {"config": config.model_dump(), "schema": config.model_json_schema()}

    @app.post("/api/configuration/check")
    def check_configuration(body: dict):
        try:
            ExperimentConfig.model_validate(body)
            return {"valid": True, "errors": []}
        except ValidationError as error:
            return {"valid": False, "errors": error.errors(include_url=False, include_context=False, include_input=False)}

    @app.post("/api/validate")
    def validate(config: ExperimentConfig):
        env = OpusTeamEnv(config.scenario)
        env.reset(seed=config.training.seed)
        return {"valid": True, "frame": env.frame(), "config": config.model_dump()}

    @app.post("/api/simulation/reset")
    def reset_simulation(config: ExperimentConfig):
        with state.lock:
            state.simulation = OpusTeamEnv(config.scenario)
            state.simulation.reset(seed=config.training.seed)
            state.simulation.action_space.seed(config.training.seed)
            return state.simulation.frame()

    @app.post("/api/simulation/step")
    def step_simulation(body: dict):
        with state.lock:
            env = state.simulation
            if env is None:
                raise HTTPException(409, "Initialize the simulation first.")
            if env.finished:
                env.reset()
            policy = body.get("policy", "heuristic")
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "heuristic":
                action = heuristic_action(env)
            elif policy == "model":
                from opus_rl.marl.network import pack_observation, stack_dict, tensor_dict
                if state.simulation_model is None:
                    raise HTTPException(409, "Load a saved model from Runs first.")
                obs = tensor_dict(stack_dict([pack_observation(env._observation())]), "cpu")
                action = state.simulation_model.act(obs, deterministic=True)[0][0].cpu().numpy()
            elif policy == "manual":
                action = np.asarray(body.get("actions"))
            else:
                raise HTTPException(400, "Unknown simulation policy.")
            env.step(action)
            return env.frame()

    @app.post("/api/training/start")
    def start(config: ExperimentConfig):
        with state.lock:
            if state.worker and state.worker.is_alive():
                raise HTTPException(409, "An experiment is already running.")
            # Validate all geometry and inventory before creating a background job.
            env = OpusTeamEnv(config.scenario)
            env.reset(seed=config.training.seed)
            approximate_bytes = config.training.rollout_steps * config.training.num_envs * (
                3 * config.scenario.observation_size**2 * 2 + env.n_max**2 * 49 * 4 + 4096
            )
            if approximate_bytes > 3 * 1024**3:
                raise HTTPException(400, "Rollout exceeds the 3 GiB v1 memory budget; reduce rollout size or environments.")
            state.stop = threading.Event()
            state.status = {"status": "starting", "steps": 0, "total_steps": config.training.total_steps,
                            "history": [], "frame": env.frame()}

            def work():
                try:
                    _, summary = train(config, state.runs_root, state.publish, state.stop)
                    with state.lock:
                        state.status.update(status=summary["status"], run_id=summary["run_id"],
                                            summary=summary, steps=summary["steps"])
                except Exception as error:
                    with state.lock:
                        state.status.update(status="failed", error=str(error))
                    traceback.print_exc()

            state.worker = threading.Thread(target=work, daemon=True, name="opus-training")
            state.worker.start()
            return {"status": "starting"}

    @app.post("/api/training/stop")
    def stop():
        with state.lock:
            state.stop.set()
            if state.worker and state.worker.is_alive():
                state.status["status"] = "stopping"
            return {"status": state.status["status"]}

    @app.get("/api/training/status")
    def status():
        with state.lock:
            return state.status.copy()

    @app.get("/api/runs")
    def runs():
        output = []
        if state.runs_root.exists():
            for path in sorted(state.runs_root.glob("*/summary.json"), reverse=True):
                try:
                    summary = json.loads(path.read_text(encoding="utf8"))
                    summary["checkpoints"] = [p.name for p in sorted(path.parent.glob("*.pt"))]
                    output.append(summary)
                except (OSError, json.JSONDecodeError):
                    continue
        return output

    @app.get("/api/runs/{run_id}/config")
    def run_config(run_id):
        return json.loads((state.run_directory(run_id) / "config.json").read_text(encoding="utf8"))

    @app.post("/api/runs/{run_id}/load")
    def load(run_id, body: dict):
        directory = state.run_directory(run_id)
        filename = body.get("checkpoint", "best.pt")
        if filename not in [p.name for p in directory.glob("*.pt")]:
            raise HTTPException(404, "Checkpoint does not exist.")
        model, config, _ = load_checkpoint(directory / filename)
        with state.lock:
            state.simulation_model = model
            state.simulation = OpusTeamEnv(config.scenario)
            state.simulation.reset(seed=config.training.seed + 10000)
            return {"config": config.model_dump(), "frame": state.simulation.frame()}

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate_run(run_id, body: dict):
        directory = state.run_directory(run_id)
        filename = body.get("checkpoint", "best.pt")
        if filename not in [p.name for p in directory.glob("*.pt")]:
            raise HTTPException(404, "Checkpoint does not exist.")
        episodes = int(body.get("episodes", 2))
        if not 1 <= episodes <= 100:
            raise HTTPException(400, "Choose 1–100 evaluation episodes.")
        model, config, _ = load_checkpoint(directory / filename)
        scenario = ScenarioConfig.model_validate(body["scenario"]) if "scenario" in body else config.scenario
        if sum(t.available_count for t in scenario.station_types) != model.station_slots:
            raise HTTPException(400, "Evaluation station inventory slots must match the trained model.")
        result = evaluate(scenario, model, episodes, seed=config.training.seed + 10000,
                          include_frame=True)
        save_json(directory / "evaluation-manual.json", {k: v for k, v in result.items() if k != "frame"})
        return result

    @app.get("/api/runs/{run_id}/export")
    def export_run(run_id):
        directory = state.run_directory(run_id)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in directory.iterdir():
                if path.is_file() and path.suffix in (".pt", ".json", ".csv"):
                    archive.write(path, path.name)
        output.seek(0)
        return StreamingResponse(output, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})

    @app.get("/api/geocode")
    async def geocode(q: str):
        query = q.strip()
        if not 2 <= len(query) <= 200:
            raise HTTPException(400, "Enter a place name between 2 and 200 characters.")
        async with state.geocode_lock:
            if query in state.geocode_cache:
                return state.geocode_cache[query]
            delay = max(0, 1.05 - (time.monotonic() - state.last_geocode))
            await asyncio.sleep(delay)
            state.last_geocode = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get("https://nominatim.openstreetmap.org/search",
                        params={"q": query, "format": "jsonv2", "limit": 5},
                        headers={"User-Agent": "OPUS-v1-local-research/0.1"})
                    response.raise_for_status()
                    data = response.json()
            except (httpx.HTTPError, ValueError) as error:
                raise HTTPException(502, "Place search is unavailable; navigate the map or import a scenario.") from error
            state.geocode_cache[query] = data
            return data

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
    return app
