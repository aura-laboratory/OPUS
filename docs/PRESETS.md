# Reference presets

Checked against manufacturer specifications on 2026-10-19. Presets fill editable simulator properties and are saved with each mission. Each UAV has its own name and preset; all docks use one selected preset. These templates model the wide RGB camera only.

| UAV preset | Horizontal limit | Climb / descent | Battery | RGB image | FOV | Airborne baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Mavic 3E reference | 15 m/s | 6 / 6 m/s | 77 Wh | 5280 × 3956 | 84° | 121.6 W |
| Mavic 3T reference | 15 m/s | 6 / 6 m/s | 77 Wh | 8000 × 6000 | 84° | 121.6 W |
| Matrice 4E reference | 18 m/s | 10 / 8 m/s | 99.5 Wh | 5280 × 3956 | 84° | 142.14 W |
| Matrice 3D reference | 10 m/s | 6 / 6 m/s | 115.2 Wh | 5280 × 3956 | 84° | 172.8 W |

Sources: [Mavic 3 Enterprise](https://enterprise.dji.com/mavic-3-enterprise/specs), [Matrice 4](https://enterprise.dji.com/matrice-4-series/specs), [Matrice 3D / Dock 2](https://enterprise.dji.com/dock-2/specs).

Mavic limits use Normal mode and the original 77 Wh battery. The Matrice 4E horizontal envelope uses the lowest published directional maximum (18 m/s backward); Matrice 3D uses the lowest Normal-mode limit (10 m/s sideways). The simulator uses a symmetric horizontal envelope. Gimbal pitch is −90° to +35° at 100°/s. Published yaw limits are 200°/s for Mavic and 250°/s for Matrice 3D; Matrice 4E retains the simulator's assumed 200°/s limit.

Baseline airborne power is approximated by battery energy divided by published maximum hover duration: 38, 42 and 40 minutes respectively. Acceleration limits, yaw acceleration, effective radius, reserve, takeoff/landing energy and motion-energy coefficients remain editable modeling assumptions. FOV is interpreted as diagonal. Manufacturer data does not calibrate this analytical energy or collision model.

| Dock preset | Effective charging power | Derivation |
| --- | --- | --- |
| DJI Dock 3 reference | 266.49 W | 149.9 Wh × 80% / (27/60 h) |
| DJI Dock 2 reference | 151.2 W | 115.2 Wh × 70% / (32/60 h) |

Sources: [DJI Dock 3](https://enterprise.dji.com/dock-3/specs), [DJI Dock 2](https://enterprise.dji.com/dock-2/specs). These are derived constant battery-side charging averages, not mains input ratings. The model does not enforce manufacturer hardware compatibility or simulate charging taper/cooling. Dock presets do not restrict UAV preset selection.
