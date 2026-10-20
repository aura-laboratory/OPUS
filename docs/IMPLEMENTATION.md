# OPUS v1 implementation decisions

The tracked system specification dated 2026-10-09 is the baseline. The following decisions and user revisions supersede it where stated. Coverage, altitude, interval, horizon and reward weights retain their defaults. The 2026-10-20 revision replaces team-reward MAPPO with parameter-sharing IPPO and individual marginal rewards, including signed fairness contributions.

The user revisions of 2026-10-19 supersede the baseline's one-UAV-per-dock constraint and configurable initial charge: launch docks are assigned per UAV, shared launches are staged, and every UAV starts fully charged. Browser missions now define individual named UAVs with independent presets and properties, and one dock preset with a dock count. The inventory selection UI is removed. Step interval and episode length are in Training. See `PRESETS.md` for reference data and assumptions.

## Motion and boundary feasibility

Policy actions map to a rest-to-rest displacement ellipsoid. Horizontal and vertical radii come from the acceleration-limited triangular/trapezoidal motion profile over ΔT. The backend additionally enforces the limits along the requested 3D direction. Every feasible coverage waypoint is reached within the decision interval and ends with zero velocity. Any remaining time incurs airborne baseline power. This deliberately permits stopped endpoints rather than instant direction changes at nonzero velocity.

The backend can propagate an unreachable waypoint over a shorter interval, producing a nonzero final velocity, and can start from a feasible velocity aligned with its segment. An arbitrary noncollinear initial velocity is rejected; it requires a curved-path backend. The environment itself only dispatches feasible rest-to-rest waypoints, including controller transit chunks.

Concave polygons constrain the whole straight segment. When a segment leaves the polygon, its target is shortened before the first exit. This follows the reachable line rather than claiming a globally nearest projection onto a potentially nonconvex feasible set. Coverage actions respect the altitude envelope. Takeoff/landing may use lower altitude.

Translation has analytically timed acceleration, cruise, deceleration and stationary phases. Collision detection divides time at the union of phase boundaries and solves the cubic stationary points of squared separation. No physics substeps or arrays of simulated intermediate states are used. Crashes are recorded at the estimated closest-approach time; impact/debris dynamics and post-impact secondary collisions are not modeled. Energy depletion time is a root of the analytical cumulative energy integral.

Yaw uses shortest-angle, rate/acceleration-limited motion; gimbal pitch uses its rate limit. Projected targets arrive within ΔT and their endpoint rates are zero. Rotation does not incur extra energy beyond the specified empirical translation/baseline model.

## Camera and coverage

A flat-plane pinhole projection tests cell centers against the image. Local GSD is the maximum singular value of the image-to-ground Jacobian, so both pixel directions must meet the required GSD. Incidence uses the viewing ray and ground normal. Invalid cells receive zero quality; concurrent captures use their maximum quality.

The Mavic reference uses the published 84° camera FOV with an explicit **diagonal** axis assumption and image aspect ratio. The earlier specification called it horizontal; the public DJI page states FOV without labeling its axis. `camera_fov_axis` allows horizontal values for calibrated cameras. The 35-mm-equivalent focal length is not treated as a physical sensor focal length. This is documented camera-template interpretation, not manufacturer calibration.

Required GSD defaults to 0.05 m/pixel, a new configurable mission default. EMA, observation quality and reward weights follow the specification; the individual reward definition below supersedes its team averaging. Fusion computes occupancy, mean, and the exact p10 of valid cells in every block, excluding outside-area padding.

At λ=0.9 and ΔT=5 s, coverage half-life is about 32.9 s. Large-area performance depends on attainable revisit rate. Jain fairness is scale-invariant away from its numerical floor: very weak but uniform coverage can earn a large fairness term. Both effects remain visible research risks; the user explicitly retained signed fairness contributions.

## Inventory and recharge

Browser missions store each UAV as a separate entry with both internal counts equal to one; the dock count sets both internal station counts. Preset IDs and individual names persist in configurations and checkpoints. Older grouped definitions are expanded into individual cards when opened in the browser. The legacy typed-inventory/sampling API remains readable for earlier runs and command-line experiments. Launch assignments use stable inventory-slot IDs, with zero-based dock IDs in JSON and one-based numbers in the UI. Sampled station counts respect the launch docks required by that sampled fleet. There is no one-UAV-per-dock fleet limit.

All UAVs start at 100% charge. Docks have an unlimited external supply. Multiple UAVs assigned to one dock are staged off the physical berth: one occupies the pad and the others wait with no actor actions or energy draw. Queued UAVs are excluded from collision trajectories and map markers until launch; their observation uses the existing charging status encoding to retain checkpoint dimensions. Takeoff waits for the berth and launch column to clear. Each dock still charges one aircraft at a time.

The single dock-type default is **DJI Dock 3 reference**. DJI reports a 149.9 Wh aircraft battery and 15–95% charging in 27 minutes. The simulator uses a derived constant battery-side average of **266.49 W**: 149.9 × 0.80 / (27/60). This is an approximation, distinct from the dock's 800 W maximum mains input. Hardware compatibility, charge taper and cooling are not modeled. Source: [DJI Dock 3 specifications](https://enterprise.dji.com/dock-3/specs).

The scheduler considers available, unreserved, directly visible docks, chooses the nearest, and estimates return energy using the same analytical rest-to-rest chunks as executed transit. It includes landing overhead and the configured reserve ratio. A proposed actor action is overridden if the resulting battery/position cannot preserve a safe return reserve. Selection happens before movement, in battery-ratio order; reservations prevent concurrent assignment conflicts.

When no unreserved visible dock exists, coverage continues and a no-station diagnostic is emitted; eventual battery depletion crashes the UAV. Recharge queuing, earliest-release planning, and paths around concave corners remain future extensions. Landing approach tolerance uses the current UAV's effective radius; charging still requires reaching the exact dock. Constant one-time takeoff/landing energy costs supplement analytic flight consumption. Transit and takeoff operate under the same velocity/acceleration limits, with altitude exemption.

## Individual rewards

Let `C′ = λC + (1−λ)maxⱼ(qⱼ)` be the next coverage map. For UAV i, build `C′₋ᵢ` using the same previous map C and the same current captures from all other UAVs, omitting only i's current capture. A missing capture contributes zero. Past coverage is never removed.

With `F(C) = 0.5 mean(C) + 0.4 Jain(C) + 0.1 p10(C)` by default, the coverage contribution is `dᵢ = F(C′) − F(C′₋ᵢ)`. It is not clipped: improving coverage in an already better-observed region can reduce fairness enough to produce a negative dᵢ. A UAV with no capture earns zero coverage contribution; tied, fully redundant captures also earn zero. Marginal contributions need not add up to the global score. The strongest and second-strongest captures per cell permit exact counterfactuals without retaining every camera raster.

The individual reward is `rᵢ = dᵢ − 0.2eᵢ − 0.1pᵢ − 0.3cᵢ − 0.4xᵢ`, using the configurable weights. Energy is capped at one after dividing the UAV's own consumed Wh by `2 × P₀ᵢ × ΔT / 3600`. Projection is that UAV's normalized action correction; controlled actions have zero projection penalty. Collision is one for each involved UAV, regardless of pair count; uninvolved UAVs receive no collision cost. Crash is one for that UAV on every remaining decision while crashed, retaining the persistent-penalty convention. No individual penalty is divided by fleet size. Controller phases also receive their own energy costs, while their actor losses remain masked.

`info.uav_rewards` and `info.uav_reward_terms` contain padded per-slot results. Padding is zero and excluded from learning. Gymnasium's scalar reward and `info.reward` are the mean over actual UAVs for monitoring, and are never used as IPPO targets. Each frame also exposes the current reward beside its UAV. `uav-rewards.csv` records every actual UAV transition with environment, episode, stable inventory identifier, reward components and executed-actor flag. Evaluation saves per-UAV episode returns. The reward chart averages individual rewards across valid UAV transitions within the rollout, rather than showing evaluation return or accumulated reward.

## IPPO and reproducibility

One actor and one local value function are shared across UAVs. Both receive the same per-UAV observation: own state/intrinsics, a fused coverage map, observed teammate features and mission context. Neither receives privileged dock state, assignments or predicted return energy. Masked mean/max teammate pooling is permutation invariant. Other UAVs' simultaneous actions are not inputs.

A tanh-transformed Gaussian bounds actor actions. PPO stores the original latent action and its corrected log probability, not a projected or scheduler-generated action. The executed per-step actor mask comes from `info`; masks in the next observation are not substituted for it. Rewards, values, next values, returns and GAE advantages retain separate UAV axes throughout rollout collection and optimization. Coverage policy/entropy terms exclude scheduler-controlled, inactive or already-crashed slots. The value loss retains every actual UAV transition, including controller and persistent crashed states, and excludes padding. Empty actor batches remain valid value updates.

GAE bootstraps horizon truncations and stops its recurrence at any reset; actual termination receives no bootstrap. The rollout stores one shared map per environment in float16 and converts it to float32 for learning. Policy inference batches independent environments; analytical environment steps execute through a thread pool. The final batch may round the requested total steps up by fewer than `num_envs` team steps.

All feature normalization is fixed and explicit; no running observation statistics are used. Training uses private NumPy and PyTorch action generators. Model initialization/loading preserves the shared PyTorch initialization RNG, so browser replay does not disturb training action sampling. Checkpoints contain the model, optimizer, configuration, seeds/RNG state, source version, and step count. A checkpoint supports evaluation and replay; exact resumption of simulator trajectories is not implemented. Evaluation uses separate seeded environments. Statistical generalization and full-scale performance need additional experiments.

New checkpoints identify IPPO explicitly. Earlier checkpoints load the legacy centralized critic and team-reward mode for replay/evaluation, preserving historical scores. Starting a new training run from any configuration always creates a fresh IPPO model with individual rewards. Progress still counts environment/team steps; `agent_steps` separately counts actual UAV transitions. PPO minibatches group environment frames to share map encoding, with separate masked losses for every UAV within each frame.

## Local application

The browser and generated default configuration use the same bundled `default-area.json` starter footprint over the waterfront part of Aristotelous Square. Its approximate boundary follows the plaza footpaths mapped in [OpenStreetMap](https://www.openstreetmap.org/way/1232262919). Both docks are inside the footprint. The underlying `ScenarioConfig` metric defaults remain available for analytical tests; saved run configurations are unchanged.

FastAPI binds to 127.0.0.1. The application runs locally; source code is hosted in the aura-laboratory/OPUS GitHub repository, while checkpoints and experiments remain local. Basemap tiles come from OpenStreetMap, and submitted place searches use cached, rate-limited Nominatim requests (at most one per 1.05 s). No automatic search-as-you-type or bulk tile download is performed. Public service availability is not required for local-coordinate simulation/training.

The map overlay is for visualization; metric geometry and camera calculations use pyproj's local azimuthal-equidistant frame. Vertices snap to grid vertices. The raster combines center-based point-in-polygon membership with a supercover boundary, retaining every cell touched by an edge or vertex. Thus the raster boundary approximates the polygon to within a cell; camera projection evaluates those cells at their centers. Date-line-crossing areas and geographic latitudes beyond ±85° require preprocessing. API validation caps the area extent and inventory, and rejects excessive rollout buffer settings.

Primary references: [IPPO paper](https://arxiv.org/abs/2011.09533), [Gymnasium API](https://gymnasium.farama.org/api/env/), [MAPPO reference implementation](https://github.com/marlbenchmark/on-policy), [DJI camera specifications](https://enterprise.dji.com/mavic-3-enterprise/specs), [Nominatim policy](https://operations.osmfoundation.org/policies/nominatim/).
