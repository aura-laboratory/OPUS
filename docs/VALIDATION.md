# OPUS v1 validation

Validated locally on 2026-10-09, using Python 3.11.17 and CPU PyTorch 2.14.1, on a Ryzen 5 2600X with 32 GB RAM. Machine-readable measurements are in `validation-results.json`. The source implementation used for the learning comparison is commit `9652b90e32d166cad7bc0f36391b533d65d25845`.

## Automated checks

**21 tests pass.** Ruff and JavaScript syntax checks pass. The suite covers:

- Gymnasium signatures, spaces, deterministic resets, observation ownership and vector environments.
- Fixed/variable typed inventory, padding and count limits.
- Rest-to-rest and incomplete motion, initial velocity handling, final acceleration, closed-form energy, temporal crossing collisions and concave boundary feasibility.
- Reversible geographic coordinates, grid snapping and supercover boundary cells.
- Nadir and oblique pinhole GSD, exact quality shaping, EMA half-life, block mean/p10 fusion and normalized reward penalties.
- Recharge reservation, charging/resumption, battery depletion and persistent crash penalties.
- Tanh log probabilities, executed actor masks, empty actor batches, teammate permutation invariance, isolated action RNG and time-limit bootstrapping.
- Local API validation, background training, checkpoint reload/replay, evaluation, ZIP export and cross-origin mutation rejection.

The test run emits one upstream Starlette/httpx deprecation warning. It does not affect these checks.

## Actual learning comparison

Run `20261009-160155-7747c6` used `configs/learning-smoke.json`: one UAV, one dock, a fixed 40 × 40 m area, a 64² coverage grid, a 16² observation grid, 64-decision episodes, and 8,192 training team steps. The agreed reward weights, ΔT=5 s and λ=0.9 were retained. Training took **96.0 seconds**, including scheduled evaluation. Three comparison episodes used seeds 20000–20002 and the same scenario for every policy.

| Policy | Episode return | Mean coverage over episode | Final p10 | Crashed UAVs |
| --- | ---: | ---: | ---: | ---: |
| Initial actor | 6.921 | 0.205 | 0.000 | 0 |
| Random actions | 16.871 | 0.115 | 0.095 | 0 |
| Sweep baseline | 24.413 | 0.346 | 0.163 | 0 |
| Trained best actor | **40.828** | **0.644** | **0.767** | **0** |

The actor improved over its initial policy and beat random actions and the sweep baseline on this small scenario. The initial actor already had an explicit near-nadir gimbal bias; the comparison uses that same initialization. The fixed one-UAV scenario makes deterministic-policy results identical across these evaluation seeds. These are pipeline acceptance results, not three independent training replicates or evidence of learned multi-UAV coordination, geographic generalization, or optimality.

The complete configurations, metrics, initial/best/final models, intermediate checkpoints and comparison are saved locally under `runs/20261009-160155-7747c6`. Runs are excluded from Git and can be exported through the application.

## Long episode and full-resolution checks

A two-UAV sweep run with the default 1,000-decision horizon and a reduced 64² grid reached **5,000 simulated seconds**, truncated at exactly decision 1,000, and had **zero crashes**. It exercised automatic takeoff, coverage, reserved return transit and charging. The status counts across both UAVs were 1,042 coverage, 944 charging, 10 recharge-transit and 4 takeoff samples. The episode took 6.1 seconds. The reduced grid changes spatial discretization; this is a battery/controller/horizon check.

At the full default **1024² coverage / 128² observation** resolution with two UAVs, 50 sweep decisions took **33.1 seconds** while other validation work was running. Reset took 0.30 seconds; mean final quality was 0.546, with no crashes. Full-resolution CPU training is substantially slower than the reduced-grid learning example. Through the browser, a full-resolution two-UAV training job stopped cleanly at 300 team steps and saved checkpoints; its observed throughput was approximately 4.1 team steps/s. A separate bounded 32-step full-resolution job completed with evaluation and model export. Neither short run establishes effective multi-UAV learning.

## Browser workflow

Verified in the Codex in-app browser:

- Map tiles, attribution and an explicitly submitted Thessaloniki place search.
- Polygon drawing with right-click closure, typed dock placement and mission validation.
- Fixed inventory and sampled fleet/station modes.
- Step, Play, Pause, reset, live movement and coverage overlays.
- Edited training values reaching the backend, live metrics, completion, graceful Stop & save, saved runs and learned-policy replay.

Browser testing found and corrected stale numeric form values, a disabled Pause control and stale dock locations after replay. Loading a checkpoint now preserves the training action RNG. The launcher reuses a running local service; `stop.ps1` targets only the named OPUS environment's Python service.

## Environment migration: 2026-10-15

The original project-local `.conda` environment was cloned into the named **OPUS** environment at `C:\Users\thano\miniconda3\envs\OPUS`, preserving installed versions. Pip-generated console launchers were updated for the new interpreter location. All 21 tests, dependency consistency, lint, `conda activate OPUS`, the application launcher and the stop script were checked in the named environment. Setup, launch instructions and `environment.yml` now use OPUS. The old environment was removed after verification; saved runs remain in the project.

## Remaining research limits

The environment is an analytical simulator with documented assumptions. Aircraft power coefficients and camera templates need calibration. Full-scale heterogeneous multi-UAV learning, statistical repeatability across training seeds, collision avoidance quality and transfer to other areas need separate experiments. Exact trajectory resumption, full dynamics, terrain, NFZs, communication loss, queued recharge planning and station-placement optimization are outside v1. See `IMPLEMENTATION.md` for the executable modeling choices.
