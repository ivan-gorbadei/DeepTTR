"""What the trainer and its workers share: the devices they run on and the files of the run
directory through which they talk."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import time
from pathlib import Path

import torch

from deepttr import net as netmod
from deepttr.device import autocast_dtype, pick_device


def logger(run: Path, who: str):
    """`say(**event)`: one JSON line in RUN/log.jsonl, `{"at", "by", "event", ...}`, written in one
    call so that the processes' lines do not interleave; also printed."""
    fd = os.open(run / "log.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o664)

    def say(**kw) -> None:
        line = json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": who, **kw})
        os.write(fd, (line + "\n").encode())
        print(line, flush=True)

    return say


def read_events(path: Path) -> list[dict]:
    """A log's events in order; none without the file."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for s in f:
                try:
                    out.append(json.loads(s))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def replace(tmp: Path, path: Path, tries: int = 50) -> None:
    """`os.replace`, again for a while when another process holds the file open (Windows refuses
    to replace a file that is open)."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)


def write_text(path: Path, text: str) -> None:
    """A small file, replaced whole."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    replace(tmp, path)


def read_float(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_net(path: Path, net, step: int) -> None:
    """A checkpoint (`net.save`) with its step, replaced whole."""
    tmp = path.with_name(path.name + ".tmp")
    netmod.save(tmp, net, step=step)
    replace(tmp, path)


def run_devices(names) -> list[torch.device]:
    """The devices of a run: those named, else every CUDA card, else Apple's MPS, else the CPU."""
    if names:
        return [pick_device(n) for n in names]
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [pick_device()]


def train_dtype(device: torch.device) -> torch.dtype | None:
    """The precision a training step runs in: bfloat16 where the device computes it natively, else
    float32 (float16 would need a gradient scaler)."""
    return torch.bfloat16 if autocast_dtype(device) == torch.bfloat16 else None


def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def compiles(setting: bool | None, device: torch.device) -> bool:
    """Whether to `torch.compile` a network: as set, else on a CUDA card where Triton is installed
    (the CPU, Apple's MPS and Windows without Triton run it as it is)."""
    if setting is not None:
        return setting
    return device.type == "cuda" and importlib.util.find_spec("triton") is not None


def set_threads(device: torch.device, processes: int) -> None:
    """PyTorch's threads: a few where the device computes, a share of the cores on the CPU."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4 if device.type != "cpu" else max(1, (os.cpu_count() or 2) // max(1, processes)))
