"""Positions of self-play: the shards the workers write, and the window of positions the trainer
samples its batches from.

A shard, `replay/<pool>-w<worker>-<seq>.npz`, holds the records of whole games of one pool of a
worker (`write_shard`: `np.savez_compressed`, each key in the dtype the window keeps it in). The
pool is `self` for self-play, `L-<opponent>-n<table>` for the games against an opponent of the
league (its name with '-' for a space; `mixed`: a table of several opponents), and either of those
after a `k` for the look-ahead's games (`pool_prefix`, `shard_pool`). Every 25th shard of self-play
and every 10th of the look-ahead's is held out for validation; the league's never are (`held_out`).

A record with `lesson` 1 is a lesson, not a move of a game: the best option of a choice of
tickets tried in playouts, or a move the look-ahead proved. Lessons are rare, so they have a store
of their own and a share of every batch (`mixed_batch`).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from deepttr.train.process import replace

# the keys of a record the trainer reads, in the dtype the window keeps them in
STORE = {
    "routes": np.float16, "cities": np.float16, "tickets": np.float16, "players": np.float16, "global": np.float16,
    "slots": np.int8, "offer": np.int8, "events": np.int16, "legal": np.uint8,
    "policy": np.float16, "score": np.float16, "n_players": np.uint8, "route_owner": np.int8, "opp_tickets": np.uint8,
    "ticket_outcome": np.int8, "hands": np.uint8, "path_cost": np.uint8, "trail": np.float16, "trail_best": np.uint8,
}
# one value a record, with what a record without it holds: the opponent of its game (the trainer
# sets it by the shard's name), the probability of the move when it was played, PPO's TD target
# (NaN: none) and the penalty of a large hand
DEFAULTS = {"opponent": (np.uint8, 0), "root_prob": (np.float32, 1.0), "adv_td": (np.float32, np.nan),
            "hand_penalty": (np.float32, 0.0)}
# the moves a look tried at a decision (-1 past them) and the margins their playouts ended with,
# points (NaN past them): the move head's lessons, kept by a window with `looks`
LOOK_K, LOOK_W = 5, 32
LOOKS = {"look": (np.int16, -1, (LOOK_K,)), "look_table": (np.float16, np.nan, (LOOK_K, LOOK_W))}
# the other keys of a shard, in their dtypes: the keys of the trainer's bookkeeping
EXTRA = {"game": np.uint64, "moves": np.uint16, "ticket_start": np.uint8, "lesson": np.uint8, "value": np.float32,
         "root_forecast": np.float32}
SHARD, LOOK_SHARD = 20_000, 1_000  # records a shard; the look-ahead's games are few and slow
VAL_EVERY, LOOK_VAL_EVERY = 25, 10
NAME = re.compile(r"(?P<look>k?)(?:self|L-(?P<opponent>.+)-n(?P<table>\d))-w(?P<worker>\d+)-(?P<seq>\d+)")


class Replay:
    """A store of positions: the newest `capacity` records added, each kept in the dtypes of
    `STORE`, `DEFAULTS` and, with `looks`, `LOOKS`."""

    def __init__(self, capacity: int, looks: bool = False) -> None:
        self.capacity = capacity
        self.looks = looks
        self.arrays: dict[str, np.ndarray] = {}
        self.size = 0
        self.head = 0

    def add(self, recs: dict) -> None:
        n = len(recs["policy"])
        if not self.arrays:
            for k, dt in STORE.items():
                self.arrays[k] = np.zeros((self.capacity,) + np.shape(recs[k])[1:], dtype=dt)
            for k, (dt, none) in DEFAULTS.items():
                self.arrays[k] = np.full(self.capacity, none, dtype=dt)
            if self.looks:
                for k, (dt, none, shape) in LOOKS.items():
                    self.arrays[k] = np.full((self.capacity,) + shape, none, dtype=dt)
        if n > self.capacity:
            recs = {k: v[n - self.capacity:] for k, v in recs.items()}
            n = self.capacity
        idx = (self.head + np.arange(n)) % self.capacity
        for k in STORE:
            self.arrays[k][idx] = recs[k]
        for k, (_, none) in DEFAULTS.items():
            self.arrays[k][idx] = recs[k] if k in recs else none
        if self.looks:
            for k, (_, none, shape) in LOOKS.items():
                self.arrays[k][idx] = fit_looks(k, recs[k]) if k in recs else none
        self.head = int((self.head + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, n: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.size, size=n)
        return {k: v[idx] for k, v in self.arrays.items()}


def fit_looks(key: str, v: np.ndarray) -> np.ndarray:
    """A batch's `look` [N, K] or `look_table` [N, K, W] in the window's shape: -1 and NaN past what
    it holds, its first LOOK_K moves and LOOK_W playouts when it holds more."""
    _, none, shape = LOOKS[key]
    if v.shape[1:] == shape:
        return v
    out = np.full((len(v),) + shape, none, dtype=np.float32 if key == "look_table" else v.dtype)
    k = min(shape[0], v.shape[1])
    if key == "look":
        out[:, :k] = v[:, :k]
    else:
        w = min(shape[1], v.shape[2])
        out[:, :k, :w] = v[:, :k, :w]
    return out


def stored_dtype(key: str, dtype) -> np.dtype:
    if key in STORE:
        return STORE[key]
    if key in LOOKS:
        return LOOKS[key][0]
    if key in DEFAULTS:
        return DEFAULTS[key][0]
    return EXTRA.get(key, dtype)


def write_shard(parts: list[dict], path: Path) -> None:
    """Records (a list of dicts of arrays, a row a record) -> one shard, renamed into place whole."""
    shard = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    shard = {k: v.astype(stored_dtype(k, v.dtype), copy=False) for k, v in shard.items()}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **shard)
    replace(tmp, path)


def read_shard(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def split_lessons(recs: dict) -> tuple[dict, dict | None]:
    """A shard's records -> (the moves of games, the lessons or None)."""
    lesson = recs.get("lesson")
    if lesson is None or not (lesson >= 1).any():
        return recs, None
    t = lesson >= 1
    return {k: v[~t] for k, v in recs.items()}, {k: v[t] for k, v in recs.items()}


def mixed_batch(window: Replay, lessons: Replay, n_lessons: int, batch: int, rng: np.random.Generator) -> dict:
    """A training batch: `n_lessons` lessons from their store as its last rows, the rest from the
    window."""
    if n_lessons <= 0:
        return window.sample(batch, rng)
    a, b = window.sample(batch - n_lessons, rng), lessons.sample(n_lessons, rng)
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def spelling(name: str) -> str:
    """An opponent's name as its shards' names spell it."""
    return name.replace(" ", "-")


def pool_prefix(opponent: str, table: int = 0, look: bool = False) -> str:
    """The beginning of the names of a pool's shards: `opponent` "self" (self-play), "mixed" (a
    table of several opponents) or an opponent's name, at a table of `table`."""
    return ("k" if look else "") + ("self" if opponent == "self" else f"L-{spelling(opponent)}-n{table}")


def shard_name(prefix: str, worker: int, seq: int) -> str:
    return f"{prefix}-w{worker}-{seq:07d}.npz"


def shard_pool(name: str) -> dict | None:
    """What a shard's file name says: `look` (the look-ahead's games), `prefix`, `opponent` ("self"
    for self-play, "mixed" for a table of several opponents, else the opponent's name as `spelling`
    spells it), `table` (0 for self-play), `worker` and `seq`. None: not a shard's name."""
    m = NAME.fullmatch(name[:-4] if name.endswith(".npz") else name)
    if not m:
        return None
    league = m["opponent"] is not None
    return {"look": bool(m["look"]), "prefix": name[:name.rindex("-w")], "opponent": m["opponent"] if league else "self",
            "table": int(m["table"]) if league else 0, "worker": int(m["worker"]), "seq": int(m["seq"])}


def held_out(name: str) -> bool:
    """Whether a shard is held out for validation: every 25th of self-play and every 10th of the
    look-ahead's games; never one of the league's own pools."""
    s = shard_pool(name)
    if s is None or (s["opponent"] != "self" and not s["look"]):
        return False
    return s["seq"] % (LOOK_VAL_EVERY if s["look"] else VAL_EVERY) == 0
