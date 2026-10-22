"""Compare frozen dense camera with footprint-first capture on identical inputs.

Run from the repository root: python -m benchmarks.camera_speed
No service, browser, optimizer, saved model or user run is started/modified.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.camera_reference import scan as dense_scan
from opus_rl.app.defaults import default_experiment
from opus_rl.configs.models import ScenarioConfig
from opus_rl.envs import team_env
from opus_rl.simulation.camera import scan
from opus_rl.simulation.geometry import Area
from opus_rl.training.evaluate import heuristic_action


def scenarios(size):
    return {
        'starter': default_experiment().scenario.model_copy(update={'grid_size': size}),
        'large_square': ScenarioConfig(grid_size=size,
            polygon_xy=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),
        'small_square': ScenarioConfig(grid_size=size,
            polygon_xy=[(0, 0), (40, 0), (40, 40), (0, 40)], station_positions_xy=[(5, 5), (35, 35)]),
    }


def cameras(size, repeats):
    rows = []
    for name, config in scenarios(size).items():
        area = Area(config)
        for pitch in (-90, -60, -25):
            uav = SimpleNamespace(spec=config.uav_types[0], position=np.r_[area.low + area.side * .5, 50.],
                                  yaw=.5, gimbal=np.deg2rad(pitch), status='coverage')
            expected, actual = dense_scan(area, uav, .05), scan(area, uav, .05)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            del expected, actual
            for function in (dense_scan, scan):
                function(area, uav, .05)
            samples = [[], []]
            for repeat in range(repeats):
                for index in ((0, 1) if repeat % 2 else (1, 0)):
                    started = perf_counter()
                    (dense_scan, scan)[index](area, uav, .05)
                    samples[index].append(1000 * (perf_counter() - started))
            before, after = map(median, samples)
            row = {'grid_size': size, 'scenario': name, 'pitch_deg': pitch,
                   'dense_ms': before, 'footprint_ms': after, 'speedup': before / after,
                   'samples_ms': {'dense': samples[0], 'footprint': samples[1]}}
            rows.append(row)
            print(f'{size} {name:12} {pitch:3}: {before:.2f} -> {after:.2f} ms ({before / after:.2f}x)', flush=True)
    return rows


def interaction_trace(config, function, steps, actions=None):
    previous = team_env.scan
    team_env.scan = function
    env = team_env.OpusTeamEnv(config)
    durations, tape, digest = [], [], hashlib.sha256()
    try:
        env.reset(seed=42)
        for index in range(steps):
            action = actions[index] if actions is not None else heuristic_action(env)
            tape.append(action.copy())
            started = perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            durations.append(1000 * (perf_counter() - started))
            # Correctness hashing is outside the timed interaction.
            digest.update(env.coverage.tobytes())
            digest.update(observation['map'].tobytes())
            digest.update(json.dumps(info, sort_keys=True).encode())
            if terminated or truncated:
                env.reset()
        return sum(durations), digest.hexdigest(), tape
    finally:
        env.close()
        team_env.scan = previous


def interactions(size, repeats, steps):
    config = scenarios(size)['starter']
    _, expected, tape = interaction_trace(config, dense_scan, steps)
    samples = [[], []]
    for repeat in range(repeats):
        for index in ((0, 1) if repeat % 2 else (1, 0)):
            elapsed, digest, _ = interaction_trace(config, (dense_scan, scan)[index], steps, tape)
            assert digest == expected, 'Environment coverage, observations or rewards changed.'
            samples[index].append(elapsed)
    before, after = (median(values) / steps for values in samples)
    print(f'{size} environment: {before:.2f} -> {after:.2f} ms/step ({before / after:.2f}x)', flush=True)
    return {'grid_size': size, 'steps_per_replay': steps, 'replays_per_version': repeats,
            'dense_ms_per_step': before, 'footprint_ms_per_step': after, 'speedup': before / after,
            'replay_total_ms': {'dense': samples[0], 'footprint': samples[1]}, 'digest': expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/camera-footprint.json'))
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(6)
    report = {'platform': platform.platform(), 'numpy': np.__version__, 'cpu_threads': torch.get_num_threads(),
              'camera_repeats': args.repeats, 'cameras': [], 'interactions': []}
    for size in (512, 1024):
        report['cameras'].extend(cameras(size, args.repeats))
        report['interactions'].append(interactions(size, 3, args.steps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(f'Raw results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
