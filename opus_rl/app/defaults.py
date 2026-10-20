"""The user-facing starter mission, separate from metric simulator defaults."""

import json

from opus_rl.app import FRONTEND
from opus_rl.configs import ExperimentConfig


def default_experiment() -> ExperimentConfig:
    data = ExperimentConfig().model_dump()
    data["scenario"].update(json.loads((FRONTEND / "default-area.json").read_text(encoding="utf8")))
    return ExperimentConfig.model_validate(data)
