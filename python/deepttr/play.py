"""Running a network for the engine, and the duplicate match that measures one player against another.

`Evaluator(net)(obs)` -> the policy logits of a batch of observations, float32 numpy.
`Bot(name)`: a baseline bot of the engine - `heuristic` (a ticket builder), `hoard` (a card
hoarder), `random`.
`duplicate_match(a, b, deals=200, players=2)` plays every deal once with A in each seat and B in all
the others (two players: twice, the seats swapped) and returns A's mean duplicate utility with its
95 % interval, the wins and losses, and the mean scores: +1 = A wins every game, 0 = even. A network
plays its first choice, the legal move its policy rates highest; wrapped in `look.Looker` it looks
ahead before it moves.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from deepttr import _engine as engine
from deepttr.device import autocast_dtype, pick_device
from deepttr.net import DeepTTRNet, to_device

BOTS = ("heuristic", "hoard", "random")


class Evaluator:
    """A network ready to answer the engine. It runs on `device` (default: the best the machine has,
    `device.pick_device`) and in reduced precision where the device computes it natively
    (`amp=False`: float32 everywhere). `compile=True` compiles it with `torch.compile`: faster over
    a long match, slow to start, and not available on every platform. Batches of more than
    `max_batch` positions are evaluated in parts, so memory stays bounded whatever is asked."""

    def __init__(self, net: DeepTTRNet, device: str | torch.device | None = None, amp: bool = True,
                 compile: bool = False, max_batch: int = 1024) -> None:
        self.device = pick_device(device)
        self.net = net.to(self.device).eval()
        self.dtype = autocast_dtype(self.device) if amp else None
        self.fwd = torch.compile(self.net) if compile else self.net
        self.max_batch = max_batch

    @torch.inference_mode()
    def outputs(self, obs: dict) -> dict:
        """Every output of the network (`net.DeepTTRNet.forward`) for a batch of observations, as
        tensors on the device."""
        x = to_device(obs, self.device)
        if self.dtype is None:
            return self.fwd(x)
        with torch.autocast(self.device.type, dtype=self.dtype):
            return self.fwd(x)

    def __call__(self, obs: dict) -> np.ndarray:
        """The policy logits, [rows, actions] float32."""
        rows = len(obs["legal"])
        if rows <= self.max_batch:
            return self.outputs(obs)["logits"].float().cpu().numpy()
        parts = [self({k: v[i:i + self.max_batch] for k, v in obs.items()}) for i in range(0, rows, self.max_batch)]
        return np.concatenate(parts)

    def move(self, game) -> int:
        """The network's first choice for the player to act in `game` (an engine `Game`); -1: a pass."""
        legal = game.legal()
        if len(legal) <= 1:
            return legal[0] if legal else -1
        return engine.first_choice(self(game.observation(game.current))[0], legal)


class Bot:
    """A baseline bot of the engine as a contestant of a match."""

    def __init__(self, name: str) -> None:
        if name not in BOTS:
            raise ValueError(f"unknown bot {name!r}: one of {', '.join(BOTS)}")
        self.name = name


def _side(player) -> str:
    return player.name if isinstance(player, Bot) else "net"


def win_rate(wins: int, losses: int, games: int, players: int) -> float:
    """Two players: A's wins over the decided games; more: the share of games A finished first."""
    return round(wins / max(1, wins + losses), 4) if players == 2 else round(wins / max(1, games), 4)


def summary(res: dict, players: int) -> dict:
    """The numbers of a match from its games (`Arena.results`): A's mean duplicate utility over the
    deals with its 95 % interval (cut at the utility's bounds, -1 and 1), games won (first alone)
    and lost (last alone; two players: the other won), the win rate, and the mean scores of A's
    seat and of B's seats."""
    games = len(res["a_seat"])
    rows = np.arange(games)
    a_utility = res["utility"][rows, res["a_seat"]]
    per_deal = a_utility.reshape(-1, players).mean(axis=1)  # a deal's seatings are consecutive games
    mean = float(per_deal.mean())
    se = float(per_deal.std(ddof=1) / math.sqrt(len(per_deal))) if len(per_deal) > 1 else float("inf")
    wins, losses = int((a_utility > 0.999).sum()), int((a_utility < -0.999).sum())
    total = res["total"][:, :players].astype(np.float64)
    a_score = total[rows, res["a_seat"]]
    return {
        "players": players, "deals": len(per_deal), "games": games, "a_wins": wins, "a_losses": losses,
        "a_win_rate": win_rate(wins, losses, games, players), "a_mean": round(mean, 4),
        "ci95": [round(max(-1.0, mean - 1.96 * se), 4), round(min(1.0, mean + 1.96 * se), 4)],
        "a_score": round(float(a_score.mean()), 2),
        "b_score": round(float((total.sum(axis=1) - a_score).mean() / (players - 1)), 2),
    }


def play_match(a, b, deals: int = 200, players: int = 2, seed: int = 1, first_deal: int = 0,
               threads: int = 0, rules: dict | None = None) -> dict:
    """Play A against B and return every game (`Arena.results`: a dict of arrays, a row a game).
    A and B are each a `Bot`, an `Evaluator` (any callable from observations to policy logits), which
    plays its first choice, or a player that chooses its own moves (`look.Looker`)."""
    arena = engine.Arena(_side(a), _side(b), deals=deals, players=players, seed=seed, first_deal=first_deal,
                         threads=threads, rules=rules)
    n_actions = engine.dims()["n_actions"]
    chooses = [hasattr(player, "choose") for player in (a, b)]
    while arena.prepare():
        obs, sides = arena.observations(), arena.sides()
        logits = np.empty((len(sides), n_actions), np.float32)
        moves = np.empty(len(sides), np.int64)
        for side, player in enumerate((a, b)):
            idx = np.flatnonzero(sides == side)
            if not idx.size:
                continue
            rows = {k: v[idx] for k, v in obs.items()}
            if chooses[side]:
                moves[idx] = player.choose(arena, idx, rows)
            else:
                logits[idx] = player(rows)
        if not any(chooses):
            arena.feed(logits)
            continue
        for i in np.flatnonzero(~np.isin(sides, np.flatnonzero(chooses))):
            moves[i] = engine.first_choice(logits[i], np.flatnonzero(obs["legal"][i]).tolist())
        arena.play(moves.tolist())
    return arena.results()


def duplicate_match(a, b, deals: int = 200, players: int = 2, seed: int = 1, first_deal: int = 0,
                    threads: int = 0, rules: dict | None = None) -> dict:
    """A against B on `deals` deals (deals `first_deal`.. of the seed's sequence), each played
    `players` times with A in each seat in turn and B in all the others. A's utility of a game is
    1 - 2 rank / (players - 1): +1 first, -1 last; averaged over a deal's seatings it is 0 for equal
    players, whatever the deal. `threads`: the engine's worker threads (0: all the cores). `rules`:
    the settings that differ from the official rules (`_engine.Game`)."""
    t0 = time.perf_counter()
    res = play_match(a, b, deals, players, seed, first_deal, threads, rules)
    return {**summary(res, players), "seed": seed, "first_deal": first_deal, "seconds": round(time.perf_counter() - t0, 1)}
