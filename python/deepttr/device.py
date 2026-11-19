"""Where a network runs: the best device the machine has, and reduced precision only where the
device computes it natively."""
from __future__ import annotations

import os

import torch


def pick_device(name: str | torch.device | None = None) -> torch.device:
    """The device to run on. `name` (or the environment variable DEEPTTR_DEVICE) names one - "cuda",
    "cuda:1", "mps", "cpu"; "auto" or nothing picks a CUDA card if there is one, else Apple's MPS,
    else the CPU."""
    if isinstance(name, torch.device):
        return name
    name = name or os.environ.get("DEEPTTR_DEVICE") or "auto"
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """The reduced precision to run a network in on `device`, or None to stay in float32: bfloat16
    on a CUDA card that computes it natively, float16 on any other CUDA card. The CPU and MPS stay
    in float32: reduced precision there is slower or incomplete."""
    if device.type != "cuda":
        return None
    try:
        native = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # a torch without the argument reports native support only
        native = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native else torch.float16
