from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """
    Resolve a device string for training / inference.

    - ``auto`` — CUDA if available, else MPS if available, else CPU.
    - Otherwise passed to ``torch.device`` (e.g. ``cpu``, ``cuda``, ``cuda:1``, ``mps``).
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
