"""How two players play against each other: statistics of whole games, not just who won.

usage: python -m deepttr.versus A B [--deals 100] [--players 2] [--seed 1] [--device auto]

A and B as in `python -m deepttr.match`: `net` (the published network), a checkpoint file or a
bot. Plays the duplicate match and prints, for each side, the means per game and seat: the score
and its parts (route points, ticket points), how often it held the longest trail and how long the
trail was, tickets held and completed, ticket draws, cards taken from the deck and from the face-up
row, trains left, and the routes claimed by length; and the match's summary. A diagnostic of style:
the match is what measures strength.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from deepttr import _engine as engine
from deepttr.device import pick_device
from deepttr.match import NET, add_run_options, name_of, player
from deepttr.play import BOTS, play_match, summary

SEAT_KEYS = ("total", "route", "tickets", "trail", "completed", "held", "ticket_draws", "blind_draws", "face_up_draws",
             "trains")


def side_stats(res: dict, players: int, a_side: bool) -> dict:
    """The means over the seats of one side - A's seat of every game, or all of B's - of a match's
    games (`Arena.results`)."""
    seats = np.arange(players)[None, :]
    mine = (seats == res["a_seat"][:, None]) == a_side  # [games, players]
    out = {"seats": int(mine.sum())}
    for k in SEAT_KEYS:
        out[k] = round(float(res[k][:, :players][mine].mean()), 2)
    # the longest trail is shared by every seat that has it
    trail = res["trail"][:, :players]
    best = trail.max(axis=1, keepdims=True)
    out["longest_share"] = round(float(((trail == best) & (best > 0))[mine].mean()), 3)
    lengths = np.array([r["length"] for r in json.loads(engine.map_json())["routes"]])
    owner = res["route_owner"]  # [games, routes], -1: nobody
    of_side = (owner >= 0) & ((owner == res["a_seat"][:, None]) == a_side)
    out["claims_by_length"] = {int(n): round(float(of_side[:, lengths == n].sum() / mine.sum()), 2) for n in np.unique(lengths)}
    out["claims"] = round(float(of_side.sum() / mine.sum()), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistics of how two players play against each other.")
    ap.add_argument("a", help=f"{NET} (the published network), a checkpoint file, or a bot: {', '.join(BOTS)}")
    ap.add_argument("b", help="the same")
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    add_run_options(ap)
    x = ap.parse_args()
    device = pick_device(x.device)
    a, b = (player(s, device, not x.no_amp, x.compile) for s in (x.a, x.b))
    res = play_match(a, b, deals=x.deals, players=x.players, seed=x.seed, threads=x.threads)
    print(json.dumps({
        "a": name_of(x.a), "b": name_of(x.b), **summary(res, x.players), "turns": round(float(res["turns"].mean()), 1),
        "a_stats": side_stats(res, x.players, True), "b_stats": side_stats(res, x.players, False),
    }, indent=1))


if __name__ == "__main__":
    main()
