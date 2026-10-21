import asyncio
import io
import json
import shutil
import threading
import time
import traceback
import zipfile
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from opus_rl.app import FRONTEND, frontend_revision
from opus_rl.app.defaults import default_experiment
from opus_rl.configs import ExperimentConfig, ScenarioConfig
from opus_rl.configs.presets import reference_presets
from opus_rl.envs import OpusTeamEnv
from opus_rl.simulation.history import VisualHistory
from opus_rl.training.evaluate import evaluate, heuristic_action
from opus_rl.training.hardware import hardware_info, select_device
from opus_rl.training.train import load_checkpoint, save_json, train


class PlaybackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: str = "best.pt"
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    trail_length_m: float = Field(500, ge=0, le=1000000)


class EvaluationRequest(PlaybackRequest):
    episodes: int = Field(5, ge=1, le=100)
    policy: Literal["model", "heuristic", "random"] = "model"
    scenario: ScenarioConfig | None = None


class State:
    def __init__(self, runs_root):
        self.runs_root = Path(runs_root).resolve()
        self.lock = threading.RLock()
        self.simulation = None
        self.simulation_model = None
        self.simulation_source = None
        self.simulation_episode = 1
        self.test_history = VisualHistory()
        self.training_history = VisualHistory()
        self.simulation_steps = 0
        self.worker = None
        self.stop = threading.Event()
        self.status = {"status": "idle", "history": []}
        self.samples = deque(maxlen=2000)
        self.evaluation_worker = None
        self.evaluation_stop = threading.Event()
        self.evaluation_status = {"status": "idle"}
        self.evaluation_history = None
        self.geocode_lock = asyncio.Lock()
        self.last_geocode = 0.0
        self.geocode_cache = {}

    def publish(self, update):
        with self.lock:
            previous = self.status
            update = update.copy()
            sample = update.pop("sample", None)
            if sample is not None:
                self.samples.append(sample)
                update["live"] = sample
            telemetry = update.get("telemetry", previous.get("telemetry"))
            if telemetry and (sample is not None or "frame" in update):
                live = sample or previous.get("live", {})
                self.training_history.append(update.get("steps", 0), live.get("episode", 1),
                    live.get("total_simulated_seconds", 0), telemetry, update.get("frame"))
            history = previous.get("history", [])
            if "metrics" in update:
                history = (history + [update["metrics"]])[-1000:]
            self.status = {**previous, **update, "status": "stopping" if self.stop.is_set() else "running",
                           "history": history}

    def run_directory(self, run_id):
        if Path(run_id).name != run_id or run_id in (".", ".."):
            raise HTTPException(400, "Invalid run identifier.")
        directory = (self.runs_root / run_id).resolve()
        if directory.parent != self.runs_root or not directory.is_dir():
            raise HTTPException(404, "Run does not exist.")
        return directory

    def simulation_frame(self, trail_length_m):
        if self.simulation is None:
            raise HTTPException(409, "Initialize the test first.")
        frame = self.simulation.frame()
        self.test_history.append(self.simulation_steps, self.simulation_episode,
            self.simulation_steps * self.simulation.config.delta_t,
            {"info": frame["info"], "uavs": frame["uavs"]}, frame)
        return {**frame, "episode": self.simulation_episode,
                "trails": self.test_history.trails[self.simulation_episode].snapshot(
                    self.simulation_steps, trail_length_m), "timeline": self.test_history.metadata()}

    def reset_trails(self):
        self.test_history.clear()
        self.simulation_steps = 0


def create_app(runs_root=None):
    state = State(runs_root or Path(__file__).parents[2] / "runs")

    @asynccontextmanager
    async def lifespan(app):
        yield
        state.stop.set()
        state.evaluation_stop.set()
        if state.worker and state.worker.is_alive():
            await asyncio.to_thread(state.worker.join, 10)
        if state.evaluation_worker and state.evaluation_worker.is_alive():
            await asyncio.to_thread(state.evaluation_worker.join, 10)

    app = FastAPI(title="OPUS v1", lifespan=lifespan)
    app.state.opus = state
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Use the local OPUS page to submit requests."}, status_code=403)
        response = await call_next(request)
        # A local development app must not keep serving an earlier interface or API state.
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def configuration_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "local_only": True,
                "training_algorithm": "ippo",
                "training_controls_revision": 5,
                "test_controls_revision": 2,
                "frontend_revision": frontend_revision(FRONTEND)}

    @app.get("/api/defaults")
    def defaults():
        config = default_experiment()
        hardware = hardware_info()
        config.training.torch_threads = min(config.training.torch_threads, hardware["cpu_threads"])
        return {"config": config.model_dump(), "schema": config.model_json_schema(),
                "presets": reference_presets(), "hardware": hardware}

    @app.post("/api/configuration/check")
    def check_configuration(body: dict):
        try:
            config = ExperimentConfig.model_validate(body)
            if config.training.torch_threads > hardware_info()["cpu_threads"]:
                return {"valid": False, "errors": [{"loc": ["training", "torch_threads"],
                    "msg": f"Choose at most {hardware_info()['cpu_threads']} available CPU threads."}]}
            if config.training.device == "cuda":
                try:
                    select_device(config)
                except ValueError as error:
                    return {"valid": False, "errors": [{"loc": ["training", "device"], "msg": str(error)}]}
            return {"valid": True, "errors": []}
        except ValidationError as error:
            return {"valid": False, "errors": error.errors(include_url=False, include_context=False, include_input=False)}

    @app.post("/api/validate")
    def validate(config: ExperimentConfig):
        env = OpusTeamEnv(config.scenario)
        env.reset(seed=config.training.seed)
        return {"valid": True, "frame": env.frame(), "config": config.model_dump()}

    @app.post("/api/simulation/reset")
    def reset_simulation(config: ExperimentConfig, trail_length_m: float = Query(500, ge=0, le=1000000)):
        with state.lock:
            state.simulation = OpusTeamEnv(config.scenario)
            state.simulation.reset(seed=config.training.seed)
            state.simulation.action_space.seed(config.training.seed)
            state.simulation_episode = 1
            state.reset_trails()
            return state.simulation_frame(trail_length_m)

    @app.get("/api/simulation/frame")
    def simulation_frame(trail_length_m: float = Query(500, ge=0, le=1000000)):
        with state.lock:
            return state.simulation_frame(trail_length_m)

    @app.post("/api/simulation/step")
    def step_simulation(body: dict):
        with state.lock:
            length = body.get("trail_length_m", 500)
            if type(length) not in (int, float) or not np.isfinite(length) or not 0 <= length <= 1000000:
                raise HTTPException(400, "Path length must be 0–1,000,000 m.")
            env = state.simulation
            if env is None:
                raise HTTPException(409, "Initialize the simulation first.")
            if env.finished:
                env.reset()
                state.simulation_episode += 1
            policy = body.get("policy", "heuristic")
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "heuristic":
                action = heuristic_action(env)
            elif policy == "model":
                from opus_rl.marl.network import pack_observation, stack_dict, tensor_dict
                if state.simulation_model is None:
                    raise HTTPException(409, "Load a saved model from Test first.")
                obs = tensor_dict(stack_dict([pack_observation(env._observation())]), "cpu")
                action = state.simulation_model.act(obs, deterministic=True)[0][0].cpu().numpy()
            elif policy == "manual":
                action = np.asarray(body.get("actions"))
            else:
                raise HTTPException(400, "Unknown simulation policy.")
            env.step(action)
            state.simulation_steps += 1
            return state.simulation_frame(length)

    @app.get("/api/visualization/history/{context}")
    def visual_history(context: Literal["training", "test"], step: int = Query(ge=0),
                       timeline_id: str = "", trail_length_m: float = Query(500, ge=0, le=1000000)):
        with state.lock:
            history = state.training_history if context == "training" else state.test_history
            if context == "test" and state.evaluation_history and timeline_id == state.evaluation_history.identity:
                history = state.evaluation_history
            if timeline_id and timeline_id != history.identity:
                raise HTTPException(409, "The visualization history has changed. Return to Live.")
            return history.frame(step, trail_length_m if context == "test" else 0)

    @app.post("/api/simulation/unload")
    def unload_model():
        with state.lock:
            state.simulation = state.simulation_model = state.simulation_source = None
            state.test_history.clear()
            return {"unloaded": True}

    @app.post("/api/training/start")
    def start(config: ExperimentConfig):
        with state.lock:
            if state.worker and state.worker.is_alive():
                raise HTTPException(409, "An experiment is already running.")
            # Validate all geometry and inventory before creating a background job.
            config.scenario.reward_mode = "individual"
            select_device(config)
            env = OpusTeamEnv(config.scenario)
            env.reset(seed=config.training.seed)
            approximate_bytes = config.training.rollout_steps * (
                3 * config.scenario.observation_size**2 * 2 + env.n_max**2 * 49 * 4 + 4096
            )
            if approximate_bytes > 3 * 1024**3:
                raise HTTPException(400, "Update interval exceeds the 3 GiB v1 memory budget; reduce the interval.")
            state.stop = threading.Event()
            state.samples.clear()
            state.training_history.clear()
            state.status = {"status": "starting", "steps": 0, "total_steps": config.training.total_steps,
                            "phase": "starting", "history": [], "frame": env.frame(), "frame_step": 0,
                            "telemetry": env.telemetry()}

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
    def status(after_step: int = -1, run_id: str = ""):
        with state.lock:
            result = state.status.copy()
            cursor = after_step if run_id == result.get("run_id") else -1
            result["samples"] = [row for row in state.samples if row["steps"] > cursor]
            result["timeline"] = state.training_history.metadata()
            return result

    @app.get("/api/runs")
    def runs():
        output = []
        if state.runs_root.exists():
            for path in sorted(state.runs_root.glob("*/summary.json"), reverse=True):
                try:
                    summary = json.loads(path.read_text(encoding="utf8"))
                    summary["run_id"] = path.parent.name
                    metadata = path.parent / "metadata.json"
                    summary["name"] = json.loads(metadata.read_text(encoding="utf8")).get("name", path.parent.name) \
                        if metadata.exists() else path.parent.name
                    summary["checkpoints"] = [p.name for p in sorted(path.parent.glob("*.pt"))]
                    manual = path.parent / "evaluation-manual.json"
                    if manual.exists():
                        summary["manual_evaluation"] = json.loads(manual.read_text(encoding="utf8"))
                    output.append(summary)
                except (OSError, json.JSONDecodeError):
                    continue
        return output

    @app.post("/api/runs/{run_id}/rename")
    def rename_run(run_id, body: dict):
        name = body.get("name")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80 or any(ord(c) < 32 for c in name):
            raise HTTPException(400, "Use a name of 1–80 characters without line breaks.")
        with state.lock:
            directory = state.run_directory(run_id)
            save_json(directory / "metadata.json", {"name": name.strip()})
        return {"run_id": run_id, "name": name.strip()}

    @app.post("/api/runs/{run_id}/delete")
    def delete_run(run_id):
        with state.lock:
            directory = state.run_directory(run_id)  # Resolved directory must be a direct child of runs_root.
            if state.worker and state.worker.is_alive() and state.status.get("run_id") in (None, run_id):
                raise HTTPException(409, "Stop training before deleting this run.")
            if state.evaluation_worker and state.evaluation_worker.is_alive() and \
                    state.evaluation_status.get("run_id") == run_id:
                raise HTTPException(409, "Stop evaluation before deleting this run.")
            shutil.rmtree(directory)
            if state.simulation_source == run_id:
                state.simulation = state.simulation_model = state.simulation_source = None
                state.test_history.clear()
            if state.evaluation_status.get("run_id") == run_id:
                state.evaluation_status = {"status": "idle"}
        return {"deleted": run_id}

    @app.get("/api/runs/{run_id}/config")
    def run_config(run_id):
        return json.loads((state.run_directory(run_id) / "config.json").read_text(encoding="utf8"))

    @app.post("/api/runs/{run_id}/load")
    def load(run_id, body: PlaybackRequest):
        with state.lock:
            directory = state.run_directory(run_id)
            filename = body.checkpoint
            if filename not in [p.name for p in directory.glob("*.pt")]:
                raise HTTPException(404, "Checkpoint does not exist.")
            model, config, _ = load_checkpoint(directory / filename)
            state.simulation_model = model
            state.simulation_source = run_id
            state.simulation = OpusTeamEnv(config.scenario)
            seed = body.seed if body.seed is not None else config.training.seed + 10000
            state.simulation.reset(seed=seed)
            state.simulation.action_space.seed(seed)
            state.simulation_episode = 1
            state.reset_trails()
            return {"config": config.model_dump(), "frame": state.simulation_frame(body.trail_length_m)}

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate_run(run_id, body: EvaluationRequest):
        directory = state.run_directory(run_id)
        filename = body.checkpoint
        if filename not in [p.name for p in directory.glob("*.pt")]:
            raise HTTPException(404, "Checkpoint does not exist.")
        model, config, _ = load_checkpoint(directory / filename)
        scenario = body.scenario or config.scenario
        scenario.reward_mode = config.scenario.reward_mode
        if model.algorithm == "mappo" and sum(t.available_count for t in scenario.station_types) != model.station_slots:
            raise HTTPException(400, "Evaluation station inventory slots must match the trained model.")
        seed = body.seed if body.seed is not None else config.training.seed + 10000
        result = evaluate(scenario, model, body.episodes, seed=seed, policy=body.policy,
                          include_frame=True, trail_length_m=body.trail_length_m)
        save_json(directory / "evaluation-manual.json", {k: v for k, v in result.items() if k != "frame"})
        return result

    def start_evaluation(scenario, model, body, directory=None):
        with state.lock:
            if state.evaluation_worker and state.evaluation_worker.is_alive():
                raise HTTPException(409, "An evaluation is already running.")
            state.evaluation_stop = threading.Event()
            state.evaluation_status = {"status": "running", "job_id": uuid4().hex,
                "run_id": directory.name if directory else None,
                "checkpoint": body.checkpoint if directory else None, "policy": body.policy,
                "episode": 1, "episodes": body.episodes, "step": 0, "episode_length": scenario.horizon}
            started = time.perf_counter()

            def progress(update):
                with state.lock:
                    state.evaluation_status.update(**update, elapsed_seconds=time.perf_counter() - started)

            def work():
                try:
                    visual_history = VisualHistory()
                    result = evaluate(scenario, model, body.episodes, seed=body.seed or 0, policy=body.policy,
                        stop_event=state.evaluation_stop, include_frame=True, callback=progress,
                        trail_length_m=body.trail_length_m, visual_history=visual_history)
                    if directory and result.get("episodes"):
                        save_json(directory / "evaluation-manual.json", {
                            **{key: value for key, value in result.items() if key != "frame"},
                            "checkpoint": body.checkpoint, "policy": body.policy})
                    with state.lock:
                        state.evaluation_history = visual_history if result.get("frame") else None
                        state.evaluation_status.update(status="stopped" if state.evaluation_stop.is_set()
                            else "completed", result=result, elapsed_seconds=time.perf_counter() - started)
                except Exception as error:
                    with state.lock:
                        state.evaluation_status.update(status="failed", error=str(error))

            state.evaluation_worker = threading.Thread(target=work, daemon=True, name="opus-evaluation")
            state.evaluation_worker.start()
            return {"status": "running", "job_id": state.evaluation_status["job_id"]}

    @app.post("/api/runs/{run_id}/evaluation/start")
    def evaluate_saved(run_id, body: EvaluationRequest):
        # Reserve the run against deletion before reading its checkpoint.
        with state.lock:
            if state.evaluation_worker and state.evaluation_worker.is_alive():
                raise HTTPException(409, "An evaluation is already running.")
            directory = state.run_directory(run_id)
            if body.checkpoint not in [p.name for p in directory.glob("*.pt")]:
                raise HTTPException(404, "Checkpoint does not exist.")
            model, config, _ = load_checkpoint(directory / body.checkpoint)
            scenario = body.scenario or config.scenario
            scenario.reward_mode = config.scenario.reward_mode
            if model.algorithm == "mappo" and sum(t.available_count for t in scenario.station_types) != model.station_slots:
                raise HTTPException(400, "Evaluation station inventory slots must match the trained model.")
            body = body.model_copy(update={"seed": body.seed if body.seed is not None else config.training.seed + 10000})
            return start_evaluation(scenario, model, body, directory)

    @app.post("/api/evaluation/start")
    def evaluate_baseline(body: EvaluationRequest):
        if body.policy == "model" or body.scenario is None:
            raise HTTPException(400, "Choose a baseline and define a mission, or select a saved model.")
        return start_evaluation(body.scenario, None, body)

    @app.get("/api/evaluation/status")
    def evaluation_status():
        with state.lock:
            return state.evaluation_status.copy()

    @app.post("/api/evaluation/stop")
    def stop_evaluation():
        with state.lock:
            state.evaluation_stop.set()
            if state.evaluation_worker and state.evaluation_worker.is_alive():
                state.evaluation_status["status"] = "stopping"
            return {"status": state.evaluation_status["status"]}

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
                raise HTTPException(502, "Place search is unavailable; navigate the map.") from error
            state.geocode_cache[query] = data
            return data

    @app.get("/")
    def index():
        revision = frontend_revision(FRONTEND)
        html = (FRONTEND / "index.html").read_text(encoding="utf8")
        for name in ("app.js", "style.css", "field-help.js"):
            html = html.replace(f'"/static/{name}"', f'"/static/{name}?v={revision}"')
        return HTMLResponse(html)

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
    return app
