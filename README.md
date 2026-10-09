# OPUS v1

Local research application for persistent multi-UAV area scanning. It combines a Gymnasium team environment, analytical motion/energy/camera models, a deterministic recharge scheduler, a shared MAPPO actor, a centralized critic, and a browser dashboard.

## Start

On this computer, double-click **launch.cmd**. It starts the Python service and opens **http://127.0.0.1:8765**. Close the launcher terminal or press Ctrl+C to stop the service. The service listens on loopback only.

From PowerShell in this directory:

```powershell
.\.conda\python.exe -m opus_rl serve
```

To recreate the isolated environment:

```powershell
.\setup.ps1
```

The environment lives in `.conda`, which Git ignores. All Python dependencies are installed into that prefix. The setup uses conda-forge for Python and the official CPU PyTorch wheel index; it does not change the base Conda installation. `requirements-lock.txt` records the installed package versions. To recreate those versions, install CPU PyTorch first, then run `.\.conda\python.exe -m pip install -r requirements-lock.txt` and `.\.conda\python.exe -m pip install -e ".[dev]"`.

## Workflow

1. Start with **Demo**, or search the map for a place. Choose **Draw area**, left-click vertices, and right-click to close the polygon. Completed vertices snap to the hidden metric coverage grid when validated.
2. Define UAV and dock types, inventory counts, fixed/variable modes, camera requirements, and flight limits. **Place docks** adds locations using the selected dock type. Fixed mode needs one location for each selected dock; each initial UAV needs its own dock.
3. Choose **Validate mission**. **Reset**, **Step**, and **Play** show a sweep baseline, random actions, or a loaded model. These controls do not train the actor.
4. In **Training**, set the step budget, environment count, PPO parameters, evaluation and checkpoint intervals, and seed. **Start training** starts a local background job. **Stop & save** finishes the current calculation and writes a final checkpoint.
5. **Runs** lists saved experiments. Select a checkpoint, replay it, run repeatable evaluation, or export a ZIP containing model/configuration/normalization/metrics/evaluation files.

Scenario JSON can be imported/exported locally. The application can simulate and train without internet; map tiles and user-triggered place search require a connection. Leaflet itself is bundled locally. No source code, checkpoints, or experiments are uploaded.

## Command line

```powershell
.\.conda\python.exe -m opus_rl config --output configs/default.json
.\.conda\python.exe -m opus_rl train --config configs/default.json
.\.conda\python.exe -m opus_rl evaluate runs/<run-id>/best.pt --episodes 3
.\.conda\python.exe -m pytest -q
.\.conda\python.exe -m ruff check opus_rl tests
```

`configs/learning-smoke.json` is a small, bounded learning experiment. `configs/default.json` retains the full 1024² coverage grid, 128² observation grid, 5-second decisions, λ=0.9, 20–100 m coverage altitude, 1,000-decision episodes, and agreed reward weights. Configurations also support a reduced grid for development.

## Local Git

This directory is its own Git repository on branch `main`. The research files in the parent OPUS directory and the old `Code` prototype are preserved. A copy of the specification is tracked under `docs/`. No remote is configured.

```powershell
git status
git log --oneline
git diff
```

Generated runs, Conda packages, caches, and model files are excluded from Git. Export important experiments separately if you need another copy.

## Model and validation limits

v1 is an analytical research simulator. The aircraft's empirical power/acceleration defaults require calibration. A working training pipeline does not establish real-flight safety, optimal coverage, or generalization to every area and inventory. Read `docs/IMPLEMENTATION.md` for the precise motion, camera, scheduling, masking, and reward assumptions, and `docs/VALIDATION.md` for measured checks.

Load checkpoints produced by this local project only. The camera quality map represents geometric capture quality, not rendered images. Full physics, terrain, NFZs, communication failures, learned scheduling, station-placement optimization, and segment capture are future extensions.
