# OPUS v1

Local research application for persistent multi-UAV area scanning. It combines a Gymnasium environment, analytical motion/energy/camera models, a deterministic recharge scheduler, parameter-sharing IPPO with individual UAV rewards, and a browser dashboard.

![OPUS mission dashboard with shared launch dock](docs/dashboard.jpg)

*Mission setup, numbered charging docks, and persistent coverage in the local dashboard.*

## Start

The setup and launch scripts target Windows and require Git and Conda. From PowerShell:

```powershell
git clone https://github.com/aura-laboratory/OPUS.git
cd OPUS
.\setup.ps1
.\launch.cmd
```

After setup, double-click **launch.cmd** to start the Python service and open **http://127.0.0.1:8765**. Close the launcher terminal or press Ctrl+C to stop the service. The service listens on loopback only.

If the service is already running, the launcher opens the existing application. To stop a service running in the background, use `.\stop.ps1` from this directory. During training, choose **Stop & save** and wait for **Stopped** first.

From PowerShell in this directory:

```powershell
conda activate OPUS
python -m opus_rl serve
```

To recreate the isolated environment:

```powershell
.\setup.ps1
```

The isolated environment is named **OPUS**. Activate it with `conda activate OPUS`. The launcher selects it automatically through `conda run --name OPUS`. All Python dependencies are installed into that environment. Setup uses conda-forge for Python and the official CPU PyTorch wheel index; it does not change the base Conda installation. `requirements-lock.txt` records the installed package versions. After creating and activating OPUS, restore those exact versions with `python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt`, followed by `python -m pip install -e ".[dev]"`.

## Workflow

The starter mission covers the waterfront part of Aristotelous Square in Thessaloniki, with two docks inside the area. **Draw area** and **Place docks** turn green and appear pressed while active; click again to leave that mode.

1. Search the map for a place. Choose **Draw area**, left-click vertices, and right-click to close the polygon. Completed vertices snap to the coverage grid when validated; the area and cell-size figures update with the mission.
2. Add each aircraft with **+UAV**. Each framed UAV entry has its own **Name**, **Preset**, and **Launch dock**, even when several aircraft use the same preset. Choose one charging-station preset and **Number of docks**, then place those numbered docks. Shared launch docks stage UAVs for sequential takeoff. Camera and energy-model settings remain under UAV **Properties**. Reference specifications and assumptions are in [PRESETS.md](docs/PRESETS.md).
3. Choose **Validate mission**. **Reset**, **Step**, and **Play** preview a hand-written cyclic sweep or random actions. These Mission controls do not train the actor. Saved-model replay is in **Runs**.
4. In **Training**, set **Step time interval ΔT**, **Episode length**, the training budget, **Update interval**, CPU threads and compute device. Training uses one environment. IPPO parameters and reward weights have their own dropdowns. Evaluation uses the same episode length. **Start training** starts a local background job. **Stop & save** finishes the current calculation and writes a final checkpoint. Grid settings are in **Mission → Area of interest**.
5. **Runs** lists saved experiments. Select a checkpoint, replay it, run repeatable evaluation, or export a ZIP containing model/configuration/normalization/metrics/evaluation files.

Every UAV uses the same policy and local value function, with its own reward, return and advantage. Coverage credit measures the difference made by that UAV's current capture, keeping previous coverage and other captures unchanged. Fairness can make this contribution negative. Energy, projection, collision and persistent crash penalties are individual. The graphs show mean coverage and mean UAV reward at each training step, with breaks between episodes. The latest 2,000 points are displayed; `steps.csv` retains all points and `uav-rewards.csv` records each UAV's reward, components and battery. Existing MAPPO checkpoints remain replayable with their original team score.

Hover over fields or live readings for 1.5 seconds for descriptions. Reward weights include an equation that updates with the values. CPU threads show the selection out of the detected logical CPU count. **Auto** uses a compatible GPU when estimated free memory permits, otherwise CPU; simulation remains on CPU. No drivers or libraries are installed automatically. Learning diagnostics appear after the first weight update. **Processing time** shows cumulative phase time, mean milliseconds per call and time share, including simulator subphases; `timings.csv` and the run summary retain the counters.

Progress, current episode, simulation time, batteries and rewards are captured every training step; the dashboard polls every 0.5 seconds and refreshes its map about once a second. Evaluation progress is shown separately and does not enter the training plots. Coverage and simulated episode time reset each episode, with batteries restored to 100%. Training has no manual simulation controls; place search is Mission-only, and loaded-model information is Runs-only.

Scenario configurations are saved with each run. The application can simulate and train without internet; map tiles and user-triggered place search require a connection. Leaflet itself is bundled locally. The application does not upload scenarios, checkpoints, or experiment data.

## Command line

```powershell
conda activate OPUS
python -m opus_rl config --output configs/default.json
python -m opus_rl train --config configs/default.json
python -m opus_rl evaluate runs/<run-id>/best.pt --episodes 3
python -m pytest -q
python -m ruff check opus_rl tests
```

`configs/learning-smoke.json` is a small, bounded learning experiment. `configs/default.json` uses the full 1024² coverage grid, 128² observation grid, 5-second decisions, λ=0.9, 20–100 m coverage altitude, 500-step episodes and agreed reward weights. Training defaults are 10,000 steps, a 100-step update interval, evaluation/checkpoints every 2,000 steps, 5 evaluation episodes, seed 42, 6 CPU threads (limited by availability) and Auto compute. Configurations also support a reduced grid for development.

## Git repository

The source repository is [aura-laboratory/OPUS](https://github.com/aura-laboratory/OPUS), with `master` as the default branch. The local checkout is connected through `origin`. A copy of the system specification is tracked under `docs/`.

```powershell
git status
git log --oneline
git diff
git pull --ff-only
git push
```

Generated runs, Conda packages, caches, and model files are excluded from Git. Export important experiments separately if you need another copy.

## Model and validation limits

v1 is an analytical research simulator. The aircraft's empirical power/acceleration defaults require calibration. A working training pipeline does not establish real-flight safety, optimal coverage, or generalization to every area and inventory. Read `docs/IMPLEMENTATION.md` for the precise motion, camera, scheduling, masking, and reward assumptions, and `docs/VALIDATION.md` for measured checks.

Load checkpoints produced by this local project only. The camera quality map represents geometric capture quality, not rendered images. Full physics, terrain, NFZs, communication failures, learned scheduling, station-placement optimization, and segment capture are future extensions.
