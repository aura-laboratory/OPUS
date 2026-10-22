# OPUS v1 validation

Validated locally on 2026-10-09, using Python 3.11.17 and CPU PyTorch 2.14.1, on a Ryzen 5 2600X with 32 GB RAM. Machine-readable measurements are in `validation-results.json`. The source implementation used for the learning comparison is commit `9652b90e32d166cad7bc0f36391b533d65d25845`.

## Automated checks

**49 tests pass** as of 2026-10-20. Ruff and JavaScript syntax checks pass. The suite covers:

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

This is the original MAPPO/team-reward benchmark, retained for historical comparison. It does not establish learning performance for the subsequent IPPO/individual-reward implementation.

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

A two-UAV sweep run with the then-default 1,000-decision horizon and a reduced 64² grid reached **5,000 simulated seconds**, truncated at exactly decision 1,000, and had **zero crashes**. It exercised automatic takeoff, coverage, reserved return transit and charging. The status counts across both UAVs were 1,042 coverage, 944 charging, 10 recharge-transit and 4 takeoff samples. The episode took 6.1 seconds. The reduced grid changes spatial discretization; this is a battery/controller/horizon check.

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

## Interface and launch revision: 2026-10-19

All 29 tests pass in the named OPUS environment. New checks cover per-inventory UAV launch assignments, shared-dock staging and sequential takeoff, full initial batteries, legacy configuration normalization, inline field validation, fixed-mode independence from sampling bounds, and fresh interface delivery. The API training/save/reload/evaluate/export workflow passes with both separate and shared launch docks. A three-UAV heterogeneous mission verifies independent names and preset IDs through simulation, training, saving and checkpoint reload.

Browser checks verified right-aligned units, nested energy/camera properties, reward weights under Training, red invalid fields with explanations, fixed/sampled selection, per-UAV dock selectors, numbered green battery markers, and two UAVs assigned to one placed dock. Changing the coverage grid from 1024 to 64 changed the displayed cell width from 0.156 m to 2.500 m for the same 2.56 ha area.

After an older interface was reported, the complete UI request was audited again in the running application. HTML, JavaScript, CSS and API responses now use `Cache-Control: no-store`. Asset URLs and launcher page URLs include a content revision. A regression test changes a served asset and verifies that both the revision and delivered content update.

The original saved best checkpoint still loads and evaluates to return 40.828 and mean coverage 0.644. This is a compatibility check; the new launch behavior has not undergone a new long training study.

The individual-UAV editor was checked with three separate Mavic 3E cards, per-aircraft name/preset changes, removal of the middle aircraft, dock-preset selection, and a one-dock mission. A 4-second step interval edited under Training produced a 4-second simulation step. Numbered blue drone symbols and green battery markers retain their geographic anchors. No inventory count controls or station Properties dropdown remain in the browser.

The shared section frames, nested cards, heading gaps, and input spacing were visually checked across Mission, Training and Runs. Training dropdowns, chart frames, progress controls and saved-run checkpoint/action rows use the same hierarchy.

## Starter area and map tools: 2026-10-20

The starter footprint was checked against the waterfront part of Aristotelous Square. The API verifies that it contains the square's center and both launch docks. All 27 non-training tests pass; the three training tests were excluded from this check. Ruff and JavaScript syntax checks pass.

Browser checks on the existing local server verified green, inset pressed states for Draw area and Place docks, toggling either mode off, switching between tools, restoring the original polygon after cancelling an unfinished drawing, and entering dock placement after closing a polygon. No server or training job was launched. Reward and MAPPO code are unchanged.

## Parameter-sharing IPPO: 2026-10-20

New runs use individual capture-difference rewards, local per-UAV values and separate GAE/return targets. Checks cover exact counterfactuals with history and overlap, redundant/tied captures, negative fairness contributions, individual collision/crash costs, per-slot reward ownership, and exclusion of padded slots from policy/value losses. A PPO check supplies opposing UAV advantages and verifies opposing updates.

Bounded API tests complete 32 team steps with separate and shared launch docks, then reload, evaluate and export the IPPO model and individual reward CSV. The privileged-state accessor is disabled during these tests. A two-environment sampled-fleet test crosses episode resets with different active fleet sizes, verifies agent counts and recorded reward averages, and replaces the Gymnasium scalar reward with an unrelated large value to ensure it cannot enter learner targets.

The original saved MAPPO best model still evaluates to return **40.82751992640401** and mean coverage **0.6439687173697166** in legacy team-reward mode. New checkpoints identify IPPO and individual reward semantics explicitly. There has been no new long learning study; historical team-return values cannot be compared directly to the new individual returns. No application server was launched or restarted for this revision.

## Training controls and hardware: 2026-10-20

All 48 tests pass after merging episode fields into Training, removing the independent evaluation cap, and adding hardware discovery. API evaluation reaches the configured episode horizon; legacy configurations containing `evaluation_horizon` remain readable. Hardware checks verify cached discovery, a CPU count displayed separately from GPU support, Auto fallback without a compatible backend or sufficient memory, and inline errors for excess CPU threads. GPU selection paths are tested with mocked availability and memory; actual GPU training is untested. The current machine reports 12 logical CPU threads and a Radeon RX 580 with a CPU-only PyTorch build.

Both frontend scripts pass syntax checks. A standalone UI preview was prepared without starting an OPUS server, but browser policy blocked local-file navigation. The new tooltip delay and MathML layout have not been visually verified in the browser.

## Single-environment update revision: 2026-10-20

All 49 tests pass. Training creates exactly one simulator and no environment pool. The update test crosses episode resets with sampled fleet sizes, verifies individual reward averages and zero-valued environment IDs, and completes an intentionally non-divisible 23-step budget exactly. The separate/shared launch-dock train/save/reload/evaluate/export checks still pass. Earlier `num_envs` configuration keys are discarded during loading.

Defaults match the selected screenshot: ΔT 5 s, 500-step episodes, 10,000 training steps, Update interval 100 steps, evaluation and checkpoints every 2,000 steps, 5 evaluation episodes, seed 42, 6 CPU threads (capped to available threads), and Auto device. The current architecture is unchanged; the width tooltip now describes all affected layers and uses neurons. Odd widths are rejected before training because the teammate encoder uses H/2.

The existing user-started server was used for browser verification without a restart or training launch. Checks confirmed the removed parallel selector, Update interval label, compact CPU/device row, neuron units, diagnostics placeholder, native MathML rendering, and live coefficient updates. The delayed description stayed hidden initially and appeared after about 1.6 seconds of focus. This completes visual verification of the previous field-help/equation changes. The older server's defaults were manually set to the requested values in this separate browser tab for the screenshot; new backend defaults are verified by tests and require a server restart.

## Live training telemetry and profiling: 2026-10-20

All 52 tests pass. New checks verify that every training transition publishes progress before optimizer updates, preserves the final episode readings before reset, reports episode/time/reward/battery correctly, and keeps scheduled evaluation out of training samples. Stopping mid-rollout preserves the last step, performs the partial update, and saves a final checkpoint. API/export checks include complete step logs, timing logs and per-UAV battery readings. Cursor-based polling returns only unseen training samples; the server retains a bounded 2,000-point window and keeps update-level learning diagnostics separate.

Timing checks verify call counts for inference, simulation, bootstrap, optimization and evaluation; simulator subphase totals fit within their parent; measured top-level percentages sum to 100%. CUDA synchronization paths exist for profiling but have not been exercised on GPU hardware. Processing-time counters are instrumentation, not a speed or convergence benchmark.

Ruff, both JavaScript syntax checks and whitespace checks pass. The OPUS server was closed during this revision and was not launched or restarted. Tab-specific controls, new metric descriptions and profiling layout have not been visually verified in a running browser. Start OPUS again to load the new backend and interface.

## Test workspace and measured bottlenecks: 2026-10-21

The saved run `20261021-111646-970e30` collected 1,294 steps in 415.56 s with two UAVs, a 1024² coverage grid, 128² observation grid, ΔT = 5 s and six CPU threads. It averaged 3.11 team steps/s. Cumulative phase counters identify camera geometry as the largest cost: **71.71%** (230.19 ms/step), coverage/rewards **15.83%** (50.80 ms/step), observation construction **5.09%** (16.35 ms/step), policy inference **1.39%** (4.46 ms/step), next-state value inference **0.86%**, weight updates **1.00%**, and frame preparation/live reporting **1.45%**. These nested simulator rows are parts of the 94.19% simulation total, not additional costs. This run stopped before scheduled evaluation. Browser paint, tile downloads and HTTP serialization are not included in the frame-reporting measurement.

Map updates target a wall-clock interval, not a fixed number of steps. At this throughput snapshots generally advance several steps (roughly 3–4, or 15–20 simulated seconds); episode/final frames and pauses can differ. New publications expose the exact interval, and the frontend records synchronous drawing times for future browser measurements. No camera or policy computations were changed as a performance optimization.

All **58 Python tests** and **4 Node interface logic tests** pass, along with Ruff and JavaScript syntax checks. Backend checks cover run display-name persistence/export, deletion isolation, active training/evaluation deletion guards, bounded trails, trail reset at episode boundaries, invalid requests that must not advance the simulator, repeatable asynchronous evaluations, progress reporting, cancellation, and exclusion of incomplete episodes. Node tests (`node --test tests/frontend.test.cjs`) exercise Mission/Test/Training visibility, deferred map fitting and equal padding, trail length, and persistent evaluation feedback. These are logic tests, not rendered-browser layout verification. The application server was not launched, and actual user runs were only read for profiling.

## Camera subprofiling and read-only time history: 2026-10-21

A bounded saved-policy replay used `20261021-111646-970e30/final.pt`, two UAVs and the original 1024² grid. Five warm-up steps preceded 30 measured sampled-action interactions. No weights or user runs were modified. Mean camera-stage time was **336.99 ms per environment step** in this separate pose sample: GSD/ray geometry **33.75%** (113.74 ms), full-grid projection **27.38%** (92.25 ms), visibility/index selection **19.51%** (65.76 ms), angle/quality **5.59%** (18.84 ms), statistics **4.61%** (15.54 ms), fusion **3.86%** (13.01 ms), allocation **2.17%** (7.31 ms), pose setup **0.05%**, and residual cleanup/other **3.08%**. This is a camera-stage breakdown for the new replay, not a direct comparison with the prior full-run average. The raw measurement is in the local ignored `build/camera-profile.json`. No camera formula or candidate-cell optimization was introduced.

All **64 Python tests** and **8 Node interface logic tests** pass. New checks cover camera subphase accounting, rejection of negative-depth ground points, metric distance clipping with a partial segment, bounded episode-separated visual buffers, exact historical UAV readings with nearest-earlier heatmaps, generation changes, and historical API reads that do not alter live simulator state or training stop flags. Evaluation's first episode also supports historical access. Interface tests verify serialized automatic renaming, named Load/Unload state, stable historical display while live data advances, Live restoration, and history-only requests during scrubbing.

Ruff and JavaScript syntax checks pass. OPUS was not launched. Rendering/layout and interactive slider motion still require a running-browser check; Node tests validate the control logic only. Visual history is session-only and capped at 10,000 completed steps, with map refresh granularity separate from per-step UAV readings.

## Training pause and policy controls: 2026-10-21

All **72 Python tests** and **12 Node interface logic tests** pass, with Ruff, both JavaScript syntax checks and whitespace checks. Training-control checks cover one-interaction permits across update/evaluation boundaries, preserving rollout data, exact final weights versus uninterrupted training, stop/save while paused, shutdown waking a paused worker, final-budget completion by single-stepping, current single-step map frames, and pausing/resuming scheduled evaluation including the final evaluation. A controlled-clock check confirms that an idle pause inside evaluation does not inflate its measured duration or processing-time share.

Interface checks cover Pause/Resume/Step availability and command routing, evaluation of each selected policy (circular, random, named saved model), and removal of percentile readouts from live/evaluation results. Reward coefficients and observation encoding are unchanged pending the reward-design decision. Processing-time percentages use cumulative measured top-level processing, excluding paused waiting; nested camera/simulator rows are parts of that total.

The application server was not launched or restarted. These checks use bounded training, TestClient and Node logic tests; rendered-browser layout remains unverified. Changes are recorded in progressive Git commits with comments on control boundaries, timing exclusion and lock ordering.

## P₁₀ removal and evolving evaluation charts: 2026-10-21

All **80 Python tests** and **15 Node interface logic tests** pass, with Ruff, both JavaScript syntax checks and whitespace checks. Current policies use two map channels and no percentile reward or step-log field. Coverage tests forbid percentile calculation/sorting on the current path, verify the remaining weighted objective, preserve original actions/rewards/channels when loading an earlier IPPO checkpoint, and upgrade an older scenario when starting fresh training. Earlier MAPPO compatibility remains covered.

Evaluation checks cover saved-model, circular and random policies: pause after a known interaction, continue the same episode, preserve repeatable returns and graph samples, report exact episode boundaries, return only unseen cursor samples, and keep evaluation separate from training. Stop and application shutdown wake a paused evaluator; incomplete episodes are excluded from aggregate results. Live visual history is available during evaluation, begins at step zero, and follows every episode. Exported result frames retain the first completed episode. Deleting a completed evaluated run clears its sample/history buffers.

Interface checks cover per-step playback charts, episode changes, new-playback resets, evaluation sample deduplication, separate training/playback/evaluation histories, and Pause changing to Continue without clearing graphs. Checkpoint compatibility was verified using controlled fixtures. The application server was not launched, and rendered-browser layout remains unverified.

## Training paths and stable history scrubbing: 2026-10-22

All **81 Python tests** and **16 Node interface logic tests** pass, with Ruff, JavaScript syntax and whitespace checks. Training snapshots and history now include the displayed episode's retained UAV trajectories, stopping at the selected timestamp; paths do not depend on Test's distance limit. Checks cover episode separation, live-frame alignment, past-path truncation and unchanged trainer state during history reads.

The history slider spans its own full-width row, with timestamp and Live below it. Its domain stays fixed while dragging; the displayed right edge returns to Live even if steps arrive during the drag, and releasing elsewhere preserves the selected step. Interface checks exercise both cases. Coverage/CNN grid descriptions now state N × N dimensions and explain pre-CNN block averaging. The camera geometry and quality model are unchanged. The application was not launched; rendered-browser layout remains unverified.

## Remaining research limits

The environment is an analytical simulator with documented assumptions. Aircraft power coefficients and camera templates need calibration. Full-scale heterogeneous multi-UAV learning, statistical repeatability across training seeds, collision avoidance quality and transfer to other areas need separate experiments. Exact trajectory resumption, full dynamics, terrain, NFZs, communication loss, queued recharge planning and station-placement optimization are outside v1. See `IMPLEMENTATION.md` for the executable modeling choices.
