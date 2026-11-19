"""Shared pieces of the tests.

The tests of the engine need the compiled module (`pip install -e .` or `maturin develop --release`);
without it they are skipped, unless DEEPTTR_REQUIRE_ENGINE is set (continuous integration sets it,
so a broken build cannot pass as a skip). The tests of the published network read the checkpoint
that comes with the package; DEEPTTR_SNAPSHOT names another file to read instead.
"""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def engine():
    try:
        from deepttr import _engine
    except ImportError as exc:
        if os.environ.get("DEEPTTR_REQUIRE_ENGINE"):
            raise
        pytest.skip(f"the engine is not built: {exc}")
    return _engine


@pytest.fixture(scope="session")
def snapshot() -> Path:
    from deepttr.net import PUBLISHED

    path = Path(os.environ.get("DEEPTTR_SNAPSHOT") or PUBLISHED)
    assert path.is_file(), f"{path}: the checkpoint is not there"
    return path


class StubNet:
    """A stand-in for a network: policy logits that depend on each position alone, so the same
    position always gets the same move whatever else is in the batch; `phase` makes two of them
    play differently."""

    def __init__(self, phase: float = 0.0) -> None:
        self.phase = phase
        self.rows = 0

    def __call__(self, obs: dict) -> np.ndarray:
        rows = len(obs["legal"])
        self.rows += rows
        h = (obs["global"].reshape(rows, -1).astype(np.float64).sum(axis=1)
             + obs["players"].reshape(rows, -1).astype(np.float64).sum(axis=1)) * 0.37 + self.phase
        actions = np.arange(obs["legal"].shape[1])
        return np.sin(h[:, None] * (actions % 11 + 1)[None, :] + actions[None, :]).astype(np.float32)


@pytest.fixture
def stub():
    return StubNet
