"""One rating scale from many matches: a Bradley-Terry model fitted by Newton's method.

usage: python -m deepttr.elo GAMES.jsonl [--anchor heuristic] [--players 2] [--out RATINGS.json]

GAMES.jsonl holds a match a line, as `python -m deepttr.match --append` writes them; the fit reads
"a", "b", "a_wins", "a_losses" and "games" (and "players", when present, to keep to one table size).

The model: P(a beats b) = 1 / (1 + 10^((R_b - R_a) / 400)); a game neither side won counts as half a
win for each. The anchor's rating is 0. Every other player has `prior_draws` virtual draws against
the anchor, so that a record like 400-0 still gives a finite - and wide - estimate. The 95 %
intervals come from the curvature of the likelihood at its maximum (the Laplace approximation).
Players must be fixed in strength: a network with its way of choosing moves is one player.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

GAMMA = math.log(10) / 400


def fit(games: list[dict], anchor: str = "heuristic", prior_draws: float = 2.0, iters: int = 100) -> dict:
    """Ratings (the anchor at 0) with their 95 % intervals: {name: {"elo", "lo", "hi", "games"}}."""
    if not games:
        return {anchor: {"elo": 0.0, "lo": 0.0, "hi": 0.0, "games": 0}}
    names = sorted({g["a"] for g in games} | {g["b"] for g in games} | {anchor})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    n = np.zeros((k, k))  # games between i and j
    w = np.zeros((k, k))  # i's score against j: wins + draws / 2
    for g in games:
        i, j = idx[g["a"]], idx[g["b"]]
        draws = g["games"] - g["a_wins"] - g["a_losses"]
        s = g["a_wins"] + draws / 2
        n[i, j] += g["games"]
        n[j, i] += g["games"]
        w[i, j] += s
        w[j, i] += g["games"] - s
    a0 = idx[anchor]
    for i in range(k):  # the prior: virtual draws against the anchor
        if i != a0:
            n[i, a0] += prior_draws
            n[a0, i] += prior_draws
            w[i, a0] += prior_draws / 2
            w[a0, i] += prior_draws / 2
    free = [i for i in range(k) if i != a0]
    r = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-GAMMA * (r[:, None] - r[None, :])))
        grad = GAMMA * (w - n * p).sum(1)
        h = GAMMA ** 2 * n * p * (1 - p)
        hess = np.diag(h.sum(1)) - h  # minus the Hessian of the log-likelihood
        step = np.linalg.solve(hess[np.ix_(free, free)], grad[free])
        r[free] += step
        if np.abs(step).max() < 1e-6:
            break
    p = 1 / (1 + np.exp(-GAMMA * (r[:, None] - r[None, :])))
    h = GAMMA ** 2 * n * p * (1 - p)
    cov = np.linalg.inv((np.diag(h.sum(1)) - h)[np.ix_(free, free)])
    se = np.zeros(k)
    se[free] = np.sqrt(np.diag(cov))
    played = {name: 0 for name in names}
    for g in games:
        played[g["a"]] += g["games"]
        played[g["b"]] += g["games"]
    return {name: {"elo": round(float(r[i]), 1), "lo": round(float(r[i] - 1.96 * se[i]), 1),
                   "hi": round(float(r[i] + 1.96 * se[i]), 1), "games": played[name]} for name, i in idx.items()}


def read_games(path: str | Path, players: int | None = 2) -> list[dict]:
    """The matches of a file, a JSON object a line; with `players`, only those at tables of that
    size (a line that does not say is taken as one)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        if players is None or g.get("players", players) == players:
            out.append(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ratings on one scale from a file of matches.")
    ap.add_argument("games", help="a file of matches, a JSON line each (python -m deepttr.match --append)")
    ap.add_argument("--anchor", default="heuristic", help="the player whose rating is 0")
    ap.add_argument("--players", type=int, default=2, help="the table size of the scale; 0: every match of the file")
    ap.add_argument("--prior-draws", type=float, default=2.0, help="virtual draws of every player against the anchor")
    ap.add_argument("--out", default=None, metavar="RATINGS.json")
    x = ap.parse_args()
    games = read_games(x.games, x.players or None)
    ratings = fit(games, x.anchor, x.prior_draws)
    table = sorted(ratings.items(), key=lambda kv: -kv[1]["elo"])
    for name, v in table:
        print(f"{name:24s} {v['elo']:+8.0f}  [{v['lo']:+.0f}, {v['hi']:+.0f}]  games {v['games']}")
    if x.out:
        Path(x.out).write_text(json.dumps({"anchor": x.anchor, "matches": len(games), "players": dict(table)}, indent=1) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
