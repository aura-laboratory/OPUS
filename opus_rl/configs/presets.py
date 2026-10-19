"""Reference templates; manufacturer data and modeling assumptions are in docs/PRESETS.md."""

from opus_rl.configs.models import StationType, UAVType


def reference_presets():
    aircraft = [
        ("mavic-3e", "Mavic 3E reference", {}, "https://enterprise.dji.com/mavic-3-enterprise/specs"),
        ("mavic-3t", "Mavic 3T reference", {"camera_width_px": 8000, "camera_height_px": 6000},
         "https://enterprise.dji.com/mavic-3-enterprise/specs"),
        ("matrice-4e", "Matrice 4E reference", {
            "max_horizontal_speed_mps": 18, "max_climb_speed_mps": 10, "max_descent_speed_mps": 8,
            "battery_capacity_Wh": 99.5, "P_airborne_base_W": 142.14,
        }, "https://enterprise.dji.com/matrice-4-series/specs"),
        ("matrice-3d", "Matrice 3D reference", {
            "max_horizontal_speed_mps": 10, "max_yaw_rate_degps": 250,
            "battery_capacity_Wh": 115.2, "P_airborne_base_W": 172.8,
        }, "https://enterprise.dji.com/dock-2/specs"),
    ]
    docks = [
        ("dji-dock-3", "DJI Dock 3 reference", 266.49, "https://enterprise.dji.com/dock-3/specs"),
        ("dji-dock-2", "DJI Dock 2 reference", 151.2, "https://enterprise.dji.com/dock-2/specs"),
    ]
    return {
        "uavs": [{"id": key, "label": label, "source": source, "values": UAVType(
            name=label, preset_id=key, available_count=1, fixed_count=1, **values).model_dump()}
            for key, label, values, source in aircraft],
        "docks": [{"id": key, "label": label, "source": source, "values": StationType(
            name=label, preset_id=key, available_count=1, fixed_count=1, charge_power_W=power).model_dump()}
            for key, label, power, source in docks],
    }
