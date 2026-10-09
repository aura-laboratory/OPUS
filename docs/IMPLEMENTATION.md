# OPUS v1 implementation decisions

The tracked system specification dated 2026-10-09 is the baseline. The following decisions make its remaining ambiguities executable. The agreed coverage, altitude, interval, horizon, and reward defaults remain unchanged.

## Motion and boundary feasibility

Policy actions map to a rest-to-rest displacement ellipsoid. Horizontal and vertical radii come from the acceleration-limited triangular/trapezoidal motion profile over ΔT. The backend additionally enforces the limits along the requested 3D direction. Every feasible coverage waypoint is reached within the decision interval and ends with zero velocity. Any remaining time incurs airborne baseline power. This deliberately permits stopped endpoints rather than instant direction changes at nonzero velocity.

The backend can propagate an unreachable waypoint over a shorter interval, producing a nonzero final velocity, and can start from a feasible velocity aligned with its segment. An arbitrary noncollinear initial velocity is rejected; it requires a curved-path backend. The environment itself only dispatches feasible rest-to-rest waypoints, including controller transit chunks.

Concave polygons constrain the whole straight segment. When a segment leaves the polygon, its target is shortened before the first exit. This follows the reachable line rather than claiming a globally nearest projection onto a potentially nonconvex feasible set. Coverage actions respect the altitude envelope. Takeoff/landing may use lower altitude.

Translation has analytically timed acceleration, cruise, deceleration and stationary phases. Collision detection divides time at the union of phase boundaries and solves the cubic stationary points of squared separation. No physics substeps or arrays of simulated intermediate states are used. Crashes are recorded at the estimated closest-approach time; impact/debris dynamics and post-impact secondary collisions are not modeled. Energy depletion time is a root of the analytical cumulative energy integral.

Yaw uses shortest-angle, rate/acceleration-limited motion; gimbal pitch uses its rate limit. Projected targets arrive within ΔT and their endpoint rates are zero. Rotation does not incur extra energy beyond the specified empirical translation/baseline model.

## Camera and coverage

A flat-plane pinhole projection tests cell centers against the image. Local GSD is the maximum singular value of the image-to-ground Jacobian, so both pixel directions must meet the required GSD. Incidence uses the viewing ray and ground normal. Invalid cells receive zero quality; concurrent captures use their maximum quality.

The Mavic reference uses the published 84° camera FOV with an explicit **diagonal** axis assumption and image aspect ratio. The earlier specification called it horizontal; the public DJI page states FOV without labeling its axis. `camera_fov_axis` allows horizontal values for calibrated cameras. The 35-mm-equivalent focal length is not treated as a physical sensor focal length. This is documented camera-template interpretation, not manufacturer calibration.

Required GSD defaults to 0.05 m/pixel, a new configurable mission default. EMA, observation quality, reward weights and penalty normalization follow the specification. Fusion computes occupancy, mean, and the exact p10 of valid cells in every block, excluding outside-area padding.

At λ=0.9 and ΔT=5 s, coverage half-life is about 32.9 s. Large-area performance depends on attainable revisit rate. Jain fairness is scale-invariant away from its numerical floor: very weak but uniform coverage can earn a large fairness term. Both effects remain visible research risks; v1 does not silently change the agreed reward.

## Inventory and recharge

Sampling is without replacement from typed inventory. Station counts are sampled conditional on fleet size; this avoids silently creating impossible initial assignments. Station locations need sufficient collision separation. All UAVs initially occupy separate stations, and begin automatic takeoff when fully charged. Charging uses the specified linear power model with an unlimited external supply.

The scheduler considers available, unreserved, directly visible docks, chooses the nearest, and estimates return energy using the same analytical rest-to-rest chunks as executed transit. It includes landing overhead and the configured reserve ratio. A proposed actor action is overridden if the resulting battery/position cannot preserve a safe return reserve. Selection happens before movement, in battery-ratio order; reservations prevent concurrent assignment conflicts.

When no unreserved visible dock exists, coverage continues and a no-station diagnostic is emitted; eventual battery depletion crashes the UAV. Queuing, earliest-release planning, and paths around concave corners remain future extensions. Dock landing radius describes the approach transition; charging still requires reaching the exact dock. Constant one-time takeoff/landing energy costs supplement analytic flight consumption. Transit and takeoff operate under the same velocity/acceleration limits, with altitude exemption.

## MAPPO and reproducibility

The shared actor receives own state/intrinsics, a fused map, globally normalized teammate features, and mission context. It receives no dock positions or assignments. Masked mean/max entity pooling is permutation invariant. The centralized team critic also receives dock state, assignments and predicted return energy.

A tanh-transformed Gaussian bounds actor actions. PPO stores the original latent action and its corrected log probability, not a projected or scheduler-generated action. The executed per-step actor mask comes from `info`; masks in the next observation are not substituted for it. Critic and GAE retain every transition while coverage policy/entropy terms exclude scheduler-controlled, inactive, or crashed slots. Empty actor batches remain valid critic updates.

GAE bootstraps horizon truncations and stops its recurrence at any reset; actual termination receives no bootstrap. The rollout stores one shared map per environment in float16 and converts it to float32 for learning. Policy inference batches independent environments; analytical environment steps execute through a thread pool. The final batch may round the requested total steps up by fewer than `num_envs` team steps.

All feature normalization is fixed and explicit; no running observation statistics are used. Training uses private NumPy and PyTorch action generators. Model initialization/loading preserves the shared PyTorch initialization RNG, so browser replay does not disturb training action sampling. Checkpoints contain the model, optimizer, configuration, seeds/RNG state, source version, and step count. A checkpoint supports evaluation and replay; exact resumption of simulator trajectories is not implemented. Evaluation uses separate seeded environments. Statistical generalization and full-scale performance need additional experiments.

## Local application

FastAPI binds to 127.0.0.1. No cloud hosting or Git remote is set up. Basemap tiles come from OpenStreetMap, and submitted place searches use cached, rate-limited Nominatim requests (at most one per 1.05 s). No automatic search-as-you-type or bulk tile download is performed. Public service availability is not required for local-coordinate simulation/training.

The map overlay is for visualization; metric geometry and camera calculations use pyproj's local azimuthal-equidistant frame. Vertices snap to grid vertices. The raster combines center-based point-in-polygon membership with a supercover boundary, retaining every cell touched by an edge or vertex. Thus the raster boundary approximates the polygon to within a cell; camera projection evaluates those cells at their centers. Date-line-crossing areas and geographic latitudes beyond ±85° require preprocessing. API validation caps the area extent and inventory, and rejects excessive rollout buffer settings.

Primary references: [Gymnasium API](https://gymnasium.farama.org/api/env/), [MAPPO reference implementation](https://github.com/marlbenchmark/on-policy), [DJI camera specifications](https://enterprise.dji.com/mavic-3-enterprise/specs), [Nominatim policy](https://operations.osmfoundation.org/policies/nominatim/).
