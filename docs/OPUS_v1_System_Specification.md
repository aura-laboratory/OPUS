# OPUS v1 System Specification

Persistent Multi-UAV Area Scanning with Gymnasium and MAPPO

> **Design revision: 2026-10-09.** This revision records the confirmed defaults for ΔT, coverage-memory λ, GSD quality, reward terms and normalization, flight envelope, episode horizon and local map implementation. These values remain configurable and are initial research settings, not calibrated operational guarantees.

> **Superseding user revision: 2026-10-20.** New runs use parameter-sharing IPPO: one shared actor and local value function, with per-UAV rewards, returns and GAE advantages. Coverage credit is the signed marginal difference between next-map quality with and without that UAV's current capture, retaining all history and other captures. Jain fairness remains included and may make the contribution negative. Energy, projection, collisions and persistent crash costs apply individually. The scalar Gymnasium reward is a monitoring average only. This replaces the MAPPO/team-reward design below; see [IMPLEMENTATION.md](IMPLEMENTATION.md#individual-rewards) for the current formula, masks and checkpoint compatibility. Earlier user revisions concerning mission editing, shared launch docks and full initial batteries are also recorded there.

## 1. Document purpose and project context

The current user-selected training defaults supersede the baseline tables below: one environment, ΔT 5 s, Episode length 500 steps, 10,000 total training steps, Update interval 100 steps, evaluation/checkpoints every 2,000 steps, 5 evaluation episodes, seed 42, 6 CPU threads and Auto compute. Evaluation uses the same episode length. Current executable choices are recorded in `IMPLEMENTATION.md`.

This document consolidates the current agreed design of the OPUS reinforcement-learning loop. It is intended to be self-contained context for architecture refinement and for a later prompt that generates a first working implementation. OPUS addresses the limited endurance of individual UAVs through cooperative multi-UAV operation, automatic rotation through charging stations, and high-level decision-making that is lightweight enough for extensive RL training. The immediate research target is persistent area scanning: a team must keep a georeferenced area well and uniformly observed for an effectively unbounded mission rather than cover it once and stop.

> **Central design principle:** The v1 environment abstracts low-level rotor physics but preserves the variables that matter for planning: position, velocity, acceleration, yaw, camera pitch, elapsed time, energy expenditure, battery, charging, camera geometry, local GSD and persistent spatial coverage quality.


- a multi-UAV OpenAI Gymnasium environment focused on high-level decisions rather than motor commands;

- RL-based collaborative and fault-tolerant area coverage;

- optimal placement of docking stations;

- a recharge scheduler for persistent and unattended monitoring;

- later integration of real-world sensing, distributed coordination and fault tolerance.

## 2. Problem formulation

The task is modeled as a cooperative, partially observable, multi-agent control problem with finite training episodes that approximate an infinite-horizon mission. The physical area is fixed in v1, but selected fleet and station parameters can be fixed or randomized by the user.

| Element | Current definition |
| --- | --- |
| Operational objective | Maintain high and spatially balanced recent observation quality over every cell of the area of interest. |
| Persistence | Coverage must continuously be renewed. Historical observations gradually lose relevance. |
| Team | Multiple homogeneous or heterogeneous rotary-wing UAVs, with one shared actor applied to every active UAV. |
| Action | A local 3D waypoint displacement, UAV yaw target and camera-gimbal pitch target for each active UAV. |
| Decision interval | Fixed ΔT, **5 simulated seconds by default**. The environment analytically computes the state after ΔT without time-stepped physics or real-time waiting. |
| Energy | An empirical power model is integrated over the analytically calculated motion profile. |
| Recharge | A deterministic external scheduler overrides the coverage actor when battery safety requires a recharge mission. |
| Camera requirement | A cell observation is accepted only if its local projected GSD satisfies the user-defined maximum GSD; accepted captures receive the agreed continuous GSD- and incidence-angle-based quality score. |
| Training algorithm | MAPPO with parameter sharing, a shared per-UAV actor and a centralized critic. |
| Coverage memory | Per-cell EMA with **λ = 0.9** by default, independently configurable from PPO/GAE parameters. |
| Episode horizon | **1,000 environment decisions**, i.e. 5,000 seconds (83 min 20 sec) of simulated time at the default ΔT. |



## 3. Current v1 scope and deliberate exclusions

- Flat ground plane and a georeferenced 2D polygonal area of interest.

- A fixed high-resolution square-cell coverage grid, initially 1024 × 1024.

- Straight-line Euclidean motion between the current pose and the requested waypoint.

- Analytical motion over ΔT with linear/rotational speed and acceleration limits.

- Automatic endpoint image capture in coverage Mode A.

- Optional later coverage Mode B that samples camera poses along the straight segment.

- Continuous per-cell observation quality, bounded persistent quality memory, and a scalar cooperative reward.

- One UAV charging at each station; stations have an unlimited external energy supply.

- No communication loss during v1 training.

- Deferred: Rotor-level actions, full six-degree-of-freedom rigid-body dynamics, wind, drag and detailed propulsion.

- Deferred: PID motor control in v1; the backend interface must permit this replacement later.

- Deferred: No-fly zones and terrain elevation models.

- Deferred: Learned recharge scheduling; recharge is deterministic in v1.

- Deferred: Procedurally variable AoI geometry in v1.

- Deferred: Photogrammetric overlap and multi-camera UAVs in v1.

## 4. End-to-end system architecture

The whole application consists of a local user interface, configuration objects, a Gymnasium team environment, a deterministic recharge scheduler, an analytical transition backend, a shared MAPPO actor/critic stack, and a live training dashboard.

```text
User configuration → ScenarioConfig / TrainConfig
Scenario reset → active UAV/station inventory + initial global state
Per-UAV observation → shared actor → candidate action A(t)
Recharge scheduler → optional per-UAV override → executed action A′(t)
Gymnasium step → analytic next state + energy + camera/coverage update
Environment output → O(t+1), R(t), terminated, truncated, info
MAPPO update → centralized critic + shared actor
Dashboard/checkpoints → live metrics and trained model export
```

## 5. User interface and configuration workflow

The recommended v1 UI is a local browser application rather than a native desktop window. Running the Python launcher starts a local backend and opens the browser. This keeps the map interaction, plots and forms simple while the simulation and learning remain in Python. **Selected v1 stack:** Leaflet for the map widget, OpenStreetMap tiles for basemap imagery, Nominatim for user-submitted location searches, FastAPI for the local Python backend, and pyproj for reversible geographic-to-local-metric coordinate conversion. Respect tile/geocoding service usage policies and avoid unthrottled search-as-you-type requests.

### 5.1 Layout

- Right panel (larger): interactive world map, place search, AoI drawing, station placement and optional overlays.

- Left panel: mission, UAV inventory, station inventory, variability, reward, training and evaluation controls.

- Training view: the map can be replaced by or tabbed with live charts and current evaluation rollouts.

### 5.2 Area-of-interest drawing

- The user searches or navigates to a location on the world map.

- Left-click adds polygon vertices. The unfinished polygon and prospective next edge remain visible under the pointer.

- Right-click closes the polygon by connecting the final vertex to the first.

- Vertices snap to vertices of the hidden 1024 × 1024 grid.

- The UI displays straight geometric edges. Internally, the polygon is rasterized using a supercover/digital-line approximation and point-in-polygon logic; A* is not used.

- The physical size represented by each grid cell depends on the AoI extent. This is map discretization and is separate from camera GSD.

### 5.3 UAV inventory and variability

The user defines one or more UAV types and the number physically available for each type. Every type contains flight, battery, power and camera properties. The interface supports two operating modes:

| Mode | Behavior |
| --- | --- |
| Fixed fleet | The exact number of active UAVs of every type is selected and remains unchanged during all resets. |
| Variable fleet | Each reset samples a subset without replacement from the available inventory. The sampled count of each type can never exceed the number owned by the user. |



> Example inventory: Available: 2 × Type A and 1 × Type B. Valid sampled fleets include [A], [B], [A,A], [A,B], and [A,A,B]. Invalid fleets include [A,A,A] and [B,B].


### 5.4 Station inventory and variability

- The user defines station types, charge power and available count.

- Fixed mode: the user clicks map locations for every active station.

- Variable mode: reset samples a valid number and valid locations inside the AoI from the available inventory.

- Stations also serve as initial UAV positions in v1.

- One UAV may occupy/charge at one station at a time. The station energy reservoir itself is unlimited.

- A station is reserved before a UAV is dispatched to it, preventing two recharge missions from selecting the same station.

- Current v1 consistency rule: the active fleet cannot exceed the number of active stations because every UAV starts docked.

### 5.5 Training controls and live monitoring

| Control | v1 behavior |
| --- | --- |
| Algorithm | MAPPO |
| Decision interval ΔT | **5 s of simulated time** (default, configurable; no wall-clock waiting) |
| Flight envelope | Horizontal AoI polygon; coverage-policy altitude **20–100 m AGL** (configurable) |
| Episode horizon | **1,000 decision steps** (default, configurable) |
| Total environment steps | User defined |
| Parallel environments | User defined |
| Evaluation frequency | User defined |
| Evaluation episodes | User defined |
| Checkpoint frequency | User defined |
| Seed | User defined |
| Core PPO values | learning rate, γ, GAE λ, clip range, entropy coefficient, value coefficient, batch size, epochs |
| Coverage memory | **EMA λ = 0.9** (default, configurable; distinct from PPO/GAE λ); equivalent half-life about **32.9 s** at ΔT = 5 s |
| Coverage reward weights | mean **0.5**; Jain fairness **0.4**; p10 quality **0.1** |
| Other reward weights | energy **0.2**; action projection **0.1**; collision **0.3**; persistent crash fraction **0.4** |
| PPO rollout length | Can be shorter than episode horizon (e.g. 128 or 256 steps), configurable |



## 6. Gymnasium environment interface

Gymnasium can represent the whole swarm as one team-level environment. The environment still uses a shared per-UAV actor internally. To keep Gymnasium spaces fixed while the active fleet is variable, every run uses the maximum available fleet size Nmax and padding plus active masks.

```text
action_space: Box(-1, 1, shape=(N_max, 5))
observation_space: Dict({
  'uav_own': Box(..., shape=(N_max, F_own)),
  'other_uavs': Box(..., shape=(N_max, N_max-1, F_other)),
  'other_uav_mask': MultiBinary((N_max, N_max-1)),
  'map': Box(0, 1, shape=(N_max, C_map, H_obs, W_obs)),
  'active_mask': MultiBinary(N_max),
  'actor_training_mask': MultiBinary(N_max),
  'mission_context': Box(..., shape=(N_max, F_context))
})
```

Inactive padded slots are ignored. The MAPPO implementation applies the same actor to each active slot, and the centralized critic receives the full padded state plus masks. A later PettingZoo ParallelEnv wrapper may expose the same core simulator through a native multi-agent API, but it is not required for v1.

### 6.1 Standard step contract

```text
observation, reward, terminated, truncated, info = env.step(team_action)

reward: one cooperative scalar
terminated: all UAVs have crashed
truncated: 1,000 decision steps reached
info: decomposed reward terms, coverage metrics, energy, scheduler masks, projections, collisions, per-UAV status and diagnostics
```

## 7. State and observation design

### 7.1 Global simulator state

| State group | Contents |
| --- | --- |
| Mission/time | Current decision index, modeled time t, ΔT, horizon and configuration identifiers. |
| Area | Georeferenced polygon, local metric transform, high-resolution AoI mask, cell geometry and physical cell size. |
| Coverage | High-resolution persistent quality map C(t), latest per-cell local GSD and optional last-observation diagnostics. |
| UAVs | All pose, motion, battery, status, type and active-mask values. |
| Stations | Position, type, charge power, occupied/reserved state and current UAV. |
| Scheduler | Recharge override mask, station assignment, predicted energy-to-station and training mask. |



### 7.2 Per-UAV physical state

| Group | Fields |
| --- | --- |
| Pose | Global local-frame x, y, z; UAV yaw ψ; camera-gimbal pitch θg. UAV roll and pitch are not modeled in v1. |
| Motion | Linear velocity vx, vy, vz; linear acceleration ax, ay, az; yaw rate and yaw acceleration; gimbal pitch rate. |
| Energy | Battery energy in Wh, normalized battery ratio and reserve margin. |
| Status | inactive, coverage-active, recharge-transit, landing, charging or crashed. |
| Type/intrinsics | Normalized flight limits, battery capacity, energy coefficients and camera properties, or a learned type embedding derived from them. |



### 7.3 Actor observation

The coverage actor does not receive station positions in v1. Station information belongs to the external recharge scheduler and to the centralized critic. Each active UAV actor receives:

- its own normalized state and intrinsic/type properties;

- a fused fixed-size raster representation of the AoI and persistent quality map;

- a variable-cardinality set of the other active UAVs, using global normalized values;

- a small mission-context vector only for parameters that may vary across training instances.

| Other-UAV feature group | Values |
| --- | --- |
| Position | global normalized x, y, z |
| Velocity | global normalized vx, vy, vz |
| Orientation | sin(yaw), cos(yaw), gimbal pitch |
| Energy | battery ratio |
| Status | one-hot operational state |
| Type | type ID embedding or normalized properties |
| Mask | whether this entity exists in the current reset |



### 7.4 Mission context

Mission context is not a vague extra state. It is a compact vector of parameters whose variation changes the task but is not otherwise obvious from the observation. Candidate fields are required GSD, ΔT, physical cell size, coverage-memory half-life, normalized time within the 1,000-step training window, and the current UAV type properties. Any value that is constant for all training runs may be omitted.

### 7.5 Coverage-grid fusion for the neural map

The 1024 × 1024 environment grid is not merely subsampled. It is fused into a lower-resolution map, initially 128 × 128, by aggregating every high-resolution block. This protects poorly covered cells from being hidden by naïve nearest-neighbor resizing.

| Raster channel | Block-fusion operation |
| --- | --- |
| AoI occupancy fraction | Fraction of the high-resolution block that belongs to the AoI. |
| Mean persistent quality | Average C_i in the block. |
| Low-tail quality | 10th percentile or minimum-like robust statistic in the block. |
| Optional quality spread | Standard deviation or high-minus-low percentile. |



## 8. Action space, ΔT and bounded displacement

For every active UAV, the shared actor outputs five normalized continuous values:

```text
a_i(t) = [Δx_norm, Δy_norm, Δz_norm, yaw_target_norm, gimbal_pitch_target_norm]
```

The decision interval is fixed at **ΔT = 5 seconds of simulated time** by default (user-configurable). ΔT defines the modeled time represented by one environment step. A corresponding maximum reachable displacement ΔDmax is derived from the current velocity, maximum velocity and maximum acceleration. Therefore, ΔD is a bound induced by ΔT, not a separate fixed-distance formulation. **1,000 steps represent 5,000 simulated seconds (83 min 20 sec), not real-world training/wall-clock duration.**

> The environment executes immediately in computation time, but its transition represents the physical state after ΔT seconds. In later real-world execution, policy inference and communication must complete before the current waypoint segment finishes.


### 8.1 Reachable set and action projection

- The three displacement components are mapped to a reachable ellipsoid centered at the current UAV position; horizontal and vertical limits may differ.

- The target is constrained by velocity, acceleration, yaw-rate, yaw-acceleration and gimbal-rate limits.

- The **coverage-policy** horizontal target must lie inside the AoI polygon, and altitude must satisfy **20 m ≤ z ≤ 100 m above the flat ground plane** by default (both user-configurable); gimbal pitch must respect UAV limits. Together the polygon and altitude range define the permitted 3D coverage envelope.

- The external takeoff/recharge/landing controller is exempt from the 20 m coverage minimum when descending to a ground station; it remains subject to its own safety and physical constraints.

- If the raw action is infeasible, the environment projects it to the nearest feasible action and records the projection magnitude.

- The executed transition is always valid, while a scalar penalty teaches the actor to stop requesting corrections.

## 9. Analytical UAV transition backend

The v1 internal controller is an analytical kinodynamic controller, not a PID and not a time-stepped physics simulator. It maps the current state, requested waypoint and UAV limits directly to the state after ΔT.

```text
next_uav_state, transition_info = motion_backend.propagate(current_state, projected_action, uav_intrinsics, delta_t)

transition_info includes: Euclidean path length, estimated reach time, velocity profile summary, final velocity/acceleration, yaw/gimbal changes, energy and feasibility data
```

### 9.1 Translation

- Direction is the straight Euclidean vector from current position to projected waypoint.

- The backend uses an analytically defined accelerate/cruise/decelerate profile subject to current velocity, vmax and amax.

- If the waypoint is reachable within ΔT, the UAV reaches it and finishes at zero translational speed; remaining modeled time is stationary airborne time.

- If it is not reachable, the UAV finishes at the furthest feasible point along the line with the analytically calculated final velocity and acceleration.

- No intermediate state array is produced in v1.

### 9.2 Yaw and camera gimbal

- Yaw moves toward the requested angle under angular-rate and angular-acceleration limits.

- Camera pitch moves toward its target under gimbal-rate limits.

- Translation, yaw and gimbal changes are modeled as simultaneous over the same ΔT.

- UAV roll and pitch are omitted from the high-level state.

### 9.3 Collision detection without substeps

Because all actions represent synchronized motion during the same ΔT, UAV-UAV collision checking should use the minimum distance between their two straight motion segments over normalized time, rather than checking only final positions. If the minimum separation is below the sum of collision radii, the affected UAVs are marked crashed. This remains analytical and does not require a simulation substep.

### 9.4 Future PID/real-world backend

The environment must depend on an abstract MotionBackend interface. A later PIDQuadrotorBackend or external PX4/AirSim/SITL bridge can replace the analytical controller without changing the agent API. That backend would require additional properties such as mass, inertia matrix, motor and propeller coefficients, drag/aerodynamic terms, thrust limits, estimator state and controller gains, and would cost substantially more computation.

## 10. Energy, battery and charging model

v1 deliberately avoids mass, drag and rotor parameters. Power is calculated from the analytical velocity and acceleration profile with an empirical model whose coefficients belong to each UAV type:

```text
P(t) = P_airborne_base + k_vxy · ||v_xy(t)||² + k_vz_up · max(v_z(t), 0) + k_vz_down · max(-v_z(t), 0) + k_acc · ||a(t)||²
E_Wh = (1 / 3600) · ∫₀^ΔT P(t) dt
battery_next = battery_current − E_Wh
```

The integral is evaluated from the piecewise analytical motion profile. If the UAV reaches its waypoint before ΔT, the remaining interval uses the airborne baseline power. Takeoff and landing may use fixed time and energy costs configured per UAV type. The coefficients are initial approximations and must later be calibrated from flight logs or a higher-fidelity backend.

### 10.1 Charging

```text
battery_next = min(battery_capacity_Wh, battery_current + station_charge_power_W · ΔT / 3600)
```

- A charging UAV does not execute the coverage actor.

- One UAV can charge at a station at a time.

- The station never runs out of source energy.

- The default v1 policy is to recharge to full before returning the UAV to coverage.

- If battery reaches zero before successful station landing, the UAV is considered crashed.

## 11. Deterministic recharge scheduler

The recharge scheduler is an external safety/mission module. It is not learned in v1 and it is the only module that receives station locations as decision input. The coverage actor remains focused on spatial observation.

```text
For every coverage-active UAV i:
  1. Find the nearest station that can be reserved.
  2. Estimate E_travel(i, station) with the same motion/energy backend.
  3. Add landing energy, takeoff-independent reserve and safety margin.
  4. If battery_i <= E_required + margin: reserve station; override actor action; set actor_training_mask[i] = 0.
  5. Otherwise execute the actor action and set mask = 1.
```

The overridden UAV remains part of the environment and remains visible to other UAVs. Its transition can train the critic because it changes the global state, but its actor log-probability and policy loss are masked out while the scheduler controls it. Other active UAVs continue normal learning in the same step.

## 12. Camera, local GSD and coverage update

The camera backend uses a pinhole model on a flat ground plane. Camera pose is determined from UAV position, UAV yaw and camera-gimbal pitch; camera roll is fixed to zero. The user supplies camera intrinsics or equivalent FOV/resolution values and the maximum acceptable GSD.

### 12.1 Coverage modes

| Mode | Behavior |
| --- | --- |
| Mode A — endpoint | Default v1. Capture and update coverage only from the final camera pose calculated for the step. |
| Mode B — segment sampling | Later switch. Analytically sample K poses along the straight motion segment and combine their observations. No dynamics substeps are required. |



### 12.2 Per-cell local GSD

- Project the camera footprint to the ground plane.

- Find AoI cells whose centers lie inside the image footprint.

- For each candidate center, calculate local GSD through the image-to-ground projection Jacobian or by projecting small neighboring ground offsets.

- Oblique views naturally produce worse and spatially nonuniform GSD because cells are farther from the camera and more foreshortened.

- A candidate is accepted only when local_GSD ≤ required_GSD.

### 12.3 Continuous observation quality

A valid cell receives continuous instantaneous quality `q_i(t)` rather than a binary visit count. The **agreed v1 default** (also configurable) rewards lower local GSD and more perpendicular viewing angles:

```text
valid_i(t) = 1 if cell center is visible AND GSD_i(t) <= GSD_required, else 0
q_i(t) = 0                                                 if valid_i(t) = 0
q_i(t) = (1 − GSD_i(t)/(2 · GSD_required)) · sqrt(max(0, cos(θ_i(t))))
                                                         if valid_i(t) = 1
0 <= q_i(t) <= 1
```

`θ_i` is the incidence angle between the viewing ray and the ground-surface normal (0° is perpendicular/nadir). In the valid interval, the GSD factor ranges from 0.5 at the threshold toward 1 for increasingly fine GSD. The square-root angle factor is intentionally mild because oblique local GSD already captures part of the quality degradation. Cells failing GSD, lying outside the image, or outside the AoI receive `q_i=0`. If multiple UAVs validly observe the same cell in one decision interval, the **v1 combination rule** is the best observation (`q_i = max_uav q_i,uav`) rather than summing beyond one.

## 13. Persistent coverage-quality memory and reward

The environment maintains a bounded persistent quality score `C_i(t) ∈ [0,1]` for every AoI cell. It is an exponential moving average (EMA) of recent observation quality:

```text
C_i(t+1) = λ · C_i(t) + (1 − λ) · q_i(t)
DEFAULT: λ = 0.9     (configurable independently of PPO's GAE lambda)
```

Unobserved cells have `q_i(t)=0`, so their score decays; repeated high-quality observation raises the score toward 1. No per-step cross-cell min-max normalization is performed, so scores remain comparable over time and cannot explode. With `ΔT=5 s` and `λ=0.9`, an unobserved cell's score has a half-life of `5 · ln(0.5)/ln(0.9) ≈ 32.9 s`. This is a deliberately user-selected starting value, subject to later sensitivity tests; λ is **not** a network learning rate.

### 13.1 Coverage equilibrium

Equal coverage alone is insufficient because a map of uniformly zero quality is perfectly equal. The reward combines absolute coverage level, spatial fairness, and protection of the poorest-covered cells.

```text
M              = number of AoI cells
μ_C(t)         = (1/M) · Σ_i C_i(t)                      # mean quality
J_C(t)         = (Σ_i C_i(t))² / (M · Σ_i C_i(t)² + ε)   # Jain fairness
P10_C(t)       = 10th percentile of {C_i(t)}           # low-tail quality
R_coverage(t)  = 0.5 · μ_C(t) + 0.4 · J_C(t) + 0.1 · P10_C(t)
```

All three coverage components lie in `[0,1]` (up to numerical tolerance); the internal weights sum to 1. For an all-zero coverage map, define Jain fairness as 0 (the `ε` safeguard also prevents zero division). The **external coverage weight remains 1.0** for v1.

### 13.2 Final scalar v1 reward and normalization

```text
R(t) = R_coverage(t) − 0.2 · E_norm(t) − 0.1 · P_norm(t)
                     − 0.3 · C_collision(t) − 0.4 · C_crash(t)
```

Every penalty is normalized to `[0,1]` so the selected weights have interpretable relative scales, including when the number/types of active UAVs vary. Let `N0` denote the number of real UAVs sampled at the beginning of the episode (not padded inactive slots). The proposed **v1 normalization definitions** are:

| Term | Exact v1 definition | Interpretation |
| --- | --- | --- |
| `E_norm(t)` | `clip((1/N0) · Σ_i clip(E_i,used,Wh(t) / (2 · P_airborne_base,i · ΔT / 3600), 0, 1), 0, 1)` | Mean normalized UAV energy use relative to twice each type's baseline airborne consumption over ΔT; a charging/inactive/crashed UAV has zero flight energy that step. |
| `P_norm(t)` | `(1/N0) · Σ_{i controlled by actor} clip(||a_i,raw − a_i,projected||₂ / (2√5), 0, 1)` | Magnitude of correction made to each normalized five-component actor action. Ignore padded, crashed, and recharge-override actions. |
| `C_collision(t)` | `min(1, number of UAVs involved in collision during this step / N0)` | Collision penalty scaled to fleet size. |
| `C_crash(t)` | `number of UAVs currently crashed / N0` | **Persistent** crash penalty on every subsequent step, not only the crash event. Running out of battery before successful landing counts as a crash. |

The UAV energy denominator is a **heuristic normalization reference**, not a claim that `2·P_airborne_base` is a physical maximum; values are capped for stability. The same note applies to the size of the projection penalty. All component values, including raw energy and raw projection distances, are logged separately; reward decomposition, multi-head critics, constrained RL and multi-objective approaches are later experiments.

## 14. MAPPO agent architecture

MAPPO is the initial learning algorithm. The actor is shared across all active UAVs, while the critic is centralized during training. Parameter sharing is used whether the fleet size is fixed or variable.

### 14.1 Shared actor

```text
own_embedding_i = MLP_own(own_state_i + own_intrinsics_i)
map_embedding_i = CNN(fused_map_i)
entity_j = MLP_other(other_uav_j)
team_embedding_i = masked invariant pooling/attention({entity_j | j != i})
actor_input_i = concat(own_embedding_i, map_embedding_i, team_embedding_i, mission_context_i)
action_distribution_i = GaussianActor(actor_input_i)
```

The variable set of other UAVs is converted into one fixed-size permutation-invariant team embedding. Applying the same actor once per active UAV yields one action for each UAV. If variable fleet size is enabled, the network must train across sampled team sizes; architectural compatibility alone does not guarantee statistical generalization.

### 14.2 Centralized critic

The critic receives privileged global state during training: full high-level swarm state, global fused map, active masks, station positions/status, recharge assignments and mission context. UAV and station entity sets are aggregated with mask-aware invariant encoders. The critic is not required during decentralized real-world execution.

### 14.3 Training masks

| Mask | Purpose |
| --- | --- |
| active_mask | 1 only for UAV slots that exist in the sampled fleet. |
| actor_training_mask | 0 while a UAV is controlled by recharge/landing/charging logic or is crashed; 1 while its action came from the actor. |
| entity mask | Marks valid members of each other-UAV set. |
| critic state mask | Marks valid UAV/station slots for centralized aggregation. |



## 15. Optional variability and scenario sampling

The user may choose case-specific training or allow selected parameters to vary. The application creates a scenario distribution only over fields marked variable; every other field remains fixed.

| Parameter | v1 variability behavior |
| --- | --- |
| AoI polygon | Fixed in v1. Variable geometry is a future extension. |
| UAV fleet | Exact counts or subset sampled without replacement from the typed inventory. |
| Station fleet | Exact counts or subset sampled from typed inventory. |
| Station positions | User-clicked fixed positions or valid random cells. |
| Initial assignment | Derived from active stations; may be fixed or randomly permuted. |
| UAV properties | Fixed per type in v1; bounded per-property noise may be added later. |
| Communication | Full and reliable during v1 training. |



> Entity encoders and masks make variable fleet sizes executable. Robust behavior requires training and evaluation across the fleet sizes and type combinations that the policy is expected to handle.


## 16. Training and evaluation loop

- Read and validate UserConfig, ScenarioConfig, VariabilityConfig, RewardConfig and TrainConfig.

- Create N parallel Gymnasium TeamEnv instances.

- At reset, construct an active fleet and stations while respecting inventory counts, station capacity and initial-position rules.

- Produce padded observations and masks.

- Run the shared actor for each active coverage-controlled UAV.

- Apply the recharge scheduler and produce executed actions plus actor-training masks.

- Analytically calculate next UAV states, energy, collisions and charging.

- Evaluate camera coverage, local GSD, angle quality and update C(t).

- Compute scalar reward, termination/truncation and rich info metrics.

- Store rollout data; mask policy terms for scheduler-controlled UAVs.

- Compute GAE and perform MAPPO actor/critic updates. A rollout batch may be shorter than the 1,000-step environment episode (e.g. 128 or 256 steps); do not wait for the full episode to collect a PPO update.

- At evaluation frequency, run deterministic policies on fixed and sampled scenarios.

- Update live dashboard and save checkpoints/configuration snapshots.

## 17. Live metrics and experiment outputs

| Category | Metrics |
| --- | --- |
| Learning | episode return, evaluation return, actor loss, value loss, entropy, approximate KL, clip fraction, explained variance |
| Coverage level | mean persistent quality μC, p10 quality, fraction above selected quality thresholds; λ setting / effective half-life |
| Coverage equilibrium | Jain fairness JC, quality standard deviation and coefficient of variation |
| Camera | valid-GSD cell observations, mean local GSD, incidence-angle distribution, invalid-GSD attempts |
| Energy | team energy per step/episode, normalized energy penalty E_norm, energy per useful refresh, minimum battery ratio |
| Recharge | override count, recharge-transit time, station utilization and denied/no-station events |
| Safety | action projection rate/magnitude and P_norm, collision count and normalized collision fraction, currently crashed UAVs and persistent C_crash, active UAV count |
| Variability | sampled UAV/station counts and performance grouped by scenario size/type composition |



Every trained model export should include network weights, all configuration files, source-code version, random seeds, normalization statistics and evaluation summaries. This is necessary because a fixed-case model is meaningful only together with the case that produced it.

## 18. Default UAV template and required fields

The initial UI template is based on published DJI Mavic 3 Enterprise values where available. Values labeled “implementation default” are not manufacturer specifications; they are placeholders required to make the simulator runnable and must be calibrated.

| Field | Default | Basis |
| --- | --- | --- |
| max_horizontal_speed_mps | 15 | DJI normal-mode specification |
| max_climb_speed_mps | 6 | DJI normal-mode specification |
| max_descent_speed_mps | 6 | DJI normal-mode specification |
| max_horizontal_accel_mps2 | 4 | Implementation default; calibrate |
| max_vertical_accel_mps2 | 3 | Implementation default; calibrate |
| max_yaw_rate_degps | 200 | DJI maximum angular velocity |
| max_yaw_accel_degps2 | 90 | Implementation default; calibrate |
| max_gimbal_pitch_rate_degps | 100 | DJI controllable tilt speed |
| gimbal_pitch_min_deg | -90 | DJI controllable range |
| gimbal_pitch_max_deg | 35 | DJI controllable range |
| battery_capacity_Wh | 77 | DJI battery energy |
| initial_battery_ratio | 1.0 | Implementation default |
| reserve_battery_ratio | 0.15 | Implementation safety default |
| charge_power_W | 100 | DJI charger/station nominal power reference |
| P_airborne_base_W | 121.6 | Derived from 77 Wh / 38 min maximum hover time |
| k_vxy | 0.05 | Implementation default; empirical coefficient |
| k_vz_up | 8.0 | Implementation default; empirical coefficient |
| k_vz_down | 3.0 | Implementation default; empirical coefficient |
| k_acc | 1.0 | Implementation default; empirical coefficient |
| takeoff_energy_Wh | 0.30 | Implementation default; calibrate |
| landing_energy_Wh | 0.25 | Implementation default; calibrate |
| collision_radius_m | 1.0 | Implementation default |
| camera_width_px | 5280 | DJI maximum wide-camera image width |
| camera_height_px | 3956 | DJI maximum wide-camera image height |
| camera_horizontal_fov_deg | 84 | DJI wide-camera FOV |
| camera_equiv_focal_length_mm | 24 | DJI 35-mm-equivalent focal length; actual pinhole intrinsics should be derived/calibrated |



> **Important:** The empirical power coefficients are not trustworthy physical specifications. Their purpose is to provide a stable v1 transition. Evaluation must include sensitivity analysis, and real logs or a higher-fidelity controller should later replace them.


### 18.1 Station type fields

| Field | Meaning/default |
| --- | --- |
| type_name | User-defined label |
| available_count | Positive integer |
| charge_power_W | 100 W default |
| landing_radius_m | 1.5 m implementation default |
| capacity | Fixed to one UAV in v1 |
| energy_supply | Infinite in v1 |
| position | Fixed map click or reset-sampled valid location |
| occupied/reserved | Dynamic state |



### 18.2 Agreed v1 mission/training defaults (separate from UAV hardware)

| Setting | Value | Notes |
| --- | --- | --- |
| Decision interval `ΔT` | **5 simulated seconds** | Simulator computes the next state immediately in wall-clock time. |
| Environment episode horizon | **1,000 decisions** | 5,000 s = 83 min 20 sec of **simulated** time. |
| Coverage memory `λ` | **0.9** | Equivalent 32.9 s half-life at 5 s steps; adjustable. |
| Coverage altitude range | **20–100 m** | Above flat terrain, coverage actor only; user-adjustable. Landing/recharge may descend below 20 m. |
| Horizontal flight envelope | AoI polygon | Invalid target actions are projected back to feasible positions and penalized. |
| Observation-map fusion | 128 × 128 default | Fused mean/low-tail quality/occupancy from 1024 × 1024 high-resolution grid. |
| Coverage weights | **0.5 mean, 0.4 Jain fairness, 0.1 p10** | Sum to 1; overall coverage term weight 1. |
| Penalty weights | **0.2 energy, 0.1 projection, 0.3 collision, 0.4 crash** | Each penalty normalized to [0,1] as specified in §13.2. |
| Camera quality | `(1−GSD/(2·GSD_required))·sqrt(max(0,cos θ))` if valid; otherwise 0 | Applies only to visible cells meeting required GSD. |
| Map/backend libraries | Leaflet, OpenStreetMap, Nominatim, pyproj, FastAPI | Local browser UI and Python service. |
| MAPPO rollout collection | e.g. 128 or 256 transitions | Configurable; shorter than full episode if desired. |

**Note:** All are initial, user-adjustable research defaults. They should be retained in the saved scenario/training configuration and checked by unit tests.

## 19. Recommended software structure

```text
opus_rl/
  app/
    server.py
    frontend/
  configs/
    models.py
    defaults/
  envs/
    opus_team_env.py
    spaces.py
    rewards.py
  simulation/
    motion_backend.py
    analytical_backend.py
    energy.py
    recharge_scheduler.py
    camera.py
    coverage.py
    geometry.py
    collision.py
  marl/
    encoders.py
    actor.py
    critic.py
    mappo.py
    rollout_buffer.py
  training/
    train.py
    evaluate.py
    callbacks.py
    metrics.py
  tests/
    test_env_api.py
    test_motion_energy.py
    test_camera_gsd.py
    test_coverage_reward.py
    test_variable_inventory.py
```

## 20. Required validation and tests

| Test group | Acceptance focus |
| --- | --- |
| Gymnasium compliance | reset/step signatures, deterministic seeding, space containment, padded masks, vectorized environments and 1,000-step horizon truncation. |
| Motion | Known 1D/3D cases for reachable distance in ΔT=5 s, arrival time, final velocity and acceleration limits, and correct 20–100 m altitude projection for coverage actions. |
| Energy | Closed-form sanity cases, nonnegative energy, battery bounds and charging conservation. |
| Camera/GSD | Nadir analytical GSD checks, exact agreed GSD/angle shaping, invalid-GSD→0, oblique monotonicity, footprint and cell-center inclusion. |
| Coverage EMA | Bounds [0,1], default λ=0.9, 32.9 s half-life at ΔT=5 s, expected decay and convergence to constant quality. |
| Reward | Exact 0.5/0.4/0.1 and 0.2/0.1/0.3/0.4 weights; normalized penalty bounds, fleet-size scaling, sustained crash penalty; equal-zero map is not optimal. |
| Inventory variability | No sampled count exceeds user inventory; padded slots remain inactive; type mixtures are valid. |
| Recharge | Correct station reservation, threshold trigger, actor mask and crash on battery depletion before landing. |
| Collision | Analytical crossing trajectories detect minimum separation even if final positions are safe. |
| MAPPO smoke test | Small deterministic map learns a measurable improvement over random actions. |



## 21. Remaining decisions and risks

| Decision/risk | Current handling |
| --- | --- |
| Decision interval ΔT | **Resolved default: 5 simulated seconds.** Validate reachable actions and inference latency before deployment; remains configurable. |
| Coverage-memory sensitivity | **Resolved default: λ=0.9** (~32.9 s half-life). May require tuning for large AoIs/slow revisit times, but do not silently replace the agreed value. |
| GSD-quality function | **Resolved v1 formula in §12.3.** Camera model/angle quality should still be tested against expected geometry. |
| Reward tuning | **Resolved initial weights and normalization in §13.2.** Test empirical component scales, stability and the consequences of persistent crash penalties. |
| Power calibration | Current coefficients are placeholders. Energy results are comparative until calibrated. |
| No free station | v1 uses reservations. A scenario with no safely reachable station may cause a crash; future queue/earliest-release planning is needed. |
| Initial fleet/station relation | Current v1 rule requires at least one active station per active UAV. Relaxing this needs separate launch/parking logic. |
| Map services | **Resolved stack:** Leaflet + OpenStreetMap tiles + Nominatim search + pyproj + FastAPI. Tile/geocoding usage limits, attribution and offline fallback still require implementation handling. |
| Coordinate conversion | **pyproj** performs reversible WGS84/geographic-to-local-metric coordinate conversion; specific local projection must be selected from the AoI location. |
| Scalability | A 1024² grid and per-UAV map tensors can be memory-heavy under many parallel environments. Fusion should be vectorized and map observations may be shared rather than duplicated. |



## 22. Minimum acceptance criteria for generated v1

- A local browser UI opens from the Python launcher (Leaflet/OpenStreetMap, Nominatim, FastAPI, pyproj) and allows place search, polygon drawing, UAV/station inventories, fixed/variable selection and training settings.

- The AoI is converted to a 1024 × 1024 hidden mask and a local metric coordinate frame.

- A Gymnasium TeamEnv passes environment checks and supports padded variable fleets through masks.

- The action is [relative 3D waypoint displacement, yaw, gimbal pitch] per UAV, with ΔDmax induced by default ΔT=5 simulated seconds and with 20–100 m default coverage altitude bounds plus projection penalties.

- The environment analytically produces next pose, velocity, acceleration, yaw/gimbal state, elapsed time and energy without simulation substeps.

- Endpoint camera projection computes per-cell local GSD and **q=(1−GSD/(2·GSD_required))·sqrt(max(0,cos θ))** for valid cells, zero otherwise.

- Persistent cell quality uses bounded EMA with **λ=0.9**; `R_coverage=0.5·mean+0.4·Jain+0.1·p10`, with normalized penalties and agreed 0.2/0.1/0.3/0.4 weights.

- The deterministic recharge scheduler reserves stations, overrides unsafe UAV actions and masks those actor samples.

- MAPPO uses one shared actor and a centralized critic with entity encoders/masks.

- Training episodes truncate at **1,000 decisions** (83 min 20 sec simulated time at ΔT=5 s) or terminate when all UAVs have crashed; PPO rollout updates may use shorter chunks. Live learning/coverage/energy/safety metrics are shown.

- Fixed and variable inventory modes both run, and no sampled scenario violates available counts.

- The system saves model, configuration, seed, normalization and evaluation artifacts.

## 23. Planned extensions

- Coverage Mode B with sampled camera poses along the trajectory segment.

- Smooth trajectory generation from successive learned waypoints for actual execution.

- PID/full dynamic backend, PX4/AirSim/SITL integration and real flight-log calibration.

- A learned recharge scheduler and joint coverage/recharge optimization.

- Variable/procedural AoI geometry, terrain elevation and no-fly zones.

- Communication loss, stale teammate states, prediction and distributed/fault-tolerant execution.

- Station-placement optimization, multiple dock capacities and charging queues.

- Multiple cameras, overlap/sidelap constraints and 3D scanning.

- Reward decomposition, constrained RL, multi-objective RL and hierarchical policies.

- PettingZoo/RLlib wrappers and alternative MARL baselines.

## 24. Compact formal summary

```text
Defaults: ΔT=5 simulated seconds; λ=0.9; horizon=1000 decisions;
          coverage altitude 20–100 m; all configurable.

At reset:
  sample permitted fixed/variable fleet and station instance
  initialize all active UAVs at distinct stations with configured battery
  set C_i(0) = 0 for every AoI cell

At step t:
  A(t) = shared_actor(O(t))
  A′(t), m_actor(t) = recharge_scheduler(A(t), S(t))
  S_uav(t+1), E(t) = analytical_backend(S_uav(t), A′(t), ΔT)
  valid_i = visible_i AND (GSD_i <= GSD_required)
  q_i = max_per_UAV { valid_i · (1−GSD_i/(2·GSD_required)) · sqrt(max(0,cos θ_i)) }
  C_i(t+1) = 0.9·C_i(t) + 0.1·q_i(t)
  R_coverage = 0.5·μ_C + 0.4·J_C + 0.1·P10_C
  R = R_coverage − 0.2·E_norm − 0.1·P_norm − 0.3·C_collision − 0.4·C_crash
  return O(t+1), R, all_crashed, step==1000, info

Train MAPPO:
  shared actor uses own state + fused map + invariant other-UAV embedding
  centralized critic uses global UAV/station/map state
  actor loss is masked for inactive and scheduler-controlled UAVs
  collect PPO rollouts in configurable shorter batches (e.g. 128–256 steps)
```

## 25. Source context and references

[1] OPUS Part B2 research proposal (user-provided). Project objectives, high-level Gym environment, persistent monitoring, docking stations and recharge scheduling.

[2] OPUS Problem Formulation notes (user-provided). Persistent horizon, user inputs, state/action concepts, GSD and waypoint decisions.

[3] Gymnasium documentation: custom Env API and Box/Dict spaces. https://gymnasium.farama.org/api/env/

[4] PettingZoo Parallel API: simultaneous multi-agent actions and observations. https://pettingzoo.farama.org/api/parallel/

[5] Yu et al., “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games,” MAPPO baseline. https://arxiv.org/abs/2103.01955

[6] DJI Mavic 3 Enterprise official specifications. https://enterprise.dji.com/mavic-3-enterprise/specs

[7] Deep Sets: permutation-invariant set representation. https://arxiv.org/abs/1703.06114

> This is the consolidated design baseline, not a claim that every numeric default is final. The fundamental architecture is sufficiently specified to generate v1. The items in Section 21 should remain explicit configuration values or documented assumptions rather than hidden implementation choices.

