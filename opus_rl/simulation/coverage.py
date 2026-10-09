import numpy as np


def update_coverage(current, instantaneous, mask, memory):
    return np.where(mask, memory * current + (1 - memory) * instantaneous, 0).astype(np.float32)


def metrics(current, mask):
    values = current[mask]
    total = float(values.sum(dtype=np.float64))
    squared = float(np.square(values.astype(np.float64)).sum())
    fairness = total**2 / (len(values) * squared + 1e-12) if squared > 0 else 0.0
    return {"mean": float(values.mean()), "fairness": min(1.0, fairness),
            "p10": float(np.percentile(values, 10)), "std": float(values.std()),
            "fraction_above_0_1": float((values >= 0.1).mean())}


def fuse(current, mask, observation_size):
    blocks = current.shape[0] // observation_size
    # Each row contains every cell of one block, so no poor-quality cell disappears by subsampling.
    values = current.reshape(observation_size, blocks, observation_size, blocks).transpose(0, 2, 1, 3)
    valid = mask.reshape(observation_size, blocks, observation_size, blocks).transpose(0, 2, 1, 3)
    values, valid = values.reshape(observation_size, observation_size, -1), valid.reshape(
        observation_size, observation_size, -1
    )
    count = valid.sum(axis=-1)
    mean = np.sum(values * valid, axis=-1) / np.maximum(1, count)
    # Sort padded cells to the end and linearly interpolate the exact within-AoI p10.
    ordered = np.sort(np.where(valid, values, np.inf), axis=-1)
    location = np.maximum(0, (count - 1) * 0.1)
    low, high = np.floor(location).astype(int), np.ceil(location).astype(int)
    lower = np.take_along_axis(ordered, low[..., None], axis=-1)[..., 0]
    upper = np.take_along_axis(ordered, high[..., None], axis=-1)[..., 0]
    lower, upper = np.where(count > 0, lower, 0), np.where(count > 0, upper, 0)
    p10 = lower + (upper - lower) * (location - low)
    return np.stack([count / valid.shape[-1], mean, p10]).astype(np.float32)


def reward(config, coverage, energy, projection, collision, crash):
    coverage_reward = config.mean_weight * coverage["mean"] + config.fairness_weight * coverage["fairness"]
    coverage_reward += config.p10_weight * coverage["p10"]
    terms = {"coverage": float(coverage_reward), "energy": float(np.clip(energy, 0, 1)),
             "projection": float(np.clip(projection, 0, 1)), "collision": float(np.clip(collision, 0, 1)),
             "crash": float(np.clip(crash, 0, 1))}
    total = coverage_reward - config.energy_weight * terms["energy"]
    total -= config.projection_weight * terms["projection"] + config.collision_weight * terms["collision"]
    total -= config.crash_weight * terms["crash"]
    return float(total), terms
