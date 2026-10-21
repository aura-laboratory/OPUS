"""Cached hardware discovery and conservative automatic device selection."""

import json
import os
import subprocess
import sys
from functools import lru_cache

import torch

from opus_rl.envs.team_env import CONTEXT_DIM, OWN_DIM


def cpu_threads():
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def windows_gpu_names():
    if sys.platform != "win32":
        return []
    try:
        output = subprocess.check_output([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -like 'PCI*' } | "
            "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress",
        ], text=True, encoding="utf8", timeout=8, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        names = json.loads(output) if output.strip() else []
        return [names] if isinstance(names, str) else (names or [])
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


@lru_cache(maxsize=1)
def hardware_info():
    available = torch.cuda.is_available()
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else windows_gpu_names()
    return {"cpu_threads": cpu_threads(), "gpu_names": names, "cuda_available": available,
            "torch_cuda_build": torch.version.cuda}


def estimated_gpu_bytes(config):
    """Allowance for float32 rollout tensors, model/optimizer and minibatch activations.

    This is a preflight estimate, not a reservation or guarantee against later GPU use.
    The analytical simulator continues to run on the CPU.
    """
    cfg, scenario = config.training, config.scenario
    agents = sum(kind.available_count for kind in scenario.uav_types)
    others = max(1, agents - 1)
    frames = min(cfg.rollout_steps, cfg.total_steps)
    frame_values = (3 if scenario.observation_version == 1 else 2) * scenario.observation_size**2 + agents * (
        OWN_DIM * (1 + others) + others + CONTEXT_DIM + 13)
    batch = min(cfg.batch_size, frames)
    activations = batch * (16 * scenario.observation_size**2 + agents * others * cfg.hidden_size)
    return 256 * 1024**2 + 4 * frames * frame_values + 16 * activations


def select_device(config):
    requested = config.training.device
    if requested == "cpu":
        return "cpu", "CPU selected."
    if not hardware_info()["cuda_available"]:
        if requested == "cuda":
            raise ValueError("A compatible GPU backend is unavailable in this OPUS environment.")
        return "cpu", "No compatible GPU backend; using CPU."
    required = estimated_gpu_bytes(config)
    try:
        free, _ = torch.cuda.memory.mem_get_info()
    except (RuntimeError, OSError) as error:
        if requested == "cuda":
            raise ValueError("GPU memory could not be checked; choose CPU or Auto.") from error
        return "cpu", "GPU memory check unavailable; using CPU."
    if required > free * 0.8:
        if requested == "cuda":
            raise ValueError("Estimated training memory exceeds available GPU memory; choose CPU or reduce the rollout.")
        return "cpu", "Estimated training memory exceeds available GPU memory; using CPU."
    return "cuda", "Compatible GPU with sufficient estimated free memory."
