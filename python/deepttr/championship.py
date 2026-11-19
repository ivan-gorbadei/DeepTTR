"""The championship of a training run: the end of every phase and the two scripted bots, each pair
in a duplicate match at a table of two, and one Elo scale fitted to all the matches.

usage: python -m deepttr.championship RUN_ROOT [--deals 200] [--device auto]

The players are RUN_ROOT/phaseNN/end.pt for every phase that has ended, named "phase N", and the
bots "ticket builder" and "card hoarder". A network plays its first choice. Every pair meets once
in `deals` deals, each dealt twice with the seats swapped (400 games by default). The matches go
one a line into RUN_ROOT/championship.jsonl; a pair already there is not played again, so a
stopped championship goes on where it stopped. The ratings (`elo.fit`, the ticket builder at 0)
go into RUN_ROOT/elo.json and are printed as a table.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from deepttr import elo, play

BOTS = {"ticket builder": "heuristic", "card hoarder": "hoard"}
ANCHOR = "ticket builder"


def players_of(root: Path) -> dict[str, Path | str]:
    """The run's players: each ended phase's end network by its phase, then the bots."""
    out: dict[str, Path | str] = {}
    for end in sorted(root.glob("phase*/end.pt")):
        out[f"phase {int(end.parent.name[5:])}"] = end
    return {**out, **BOTS}


def contestant(source: Path | str, device: str | None):
    if isinstance(source, str):
        return play.Bot(source)
    from deepttr import net as netmod

    network, _ = netmod.load(source)
    return play.Evaluator(network, device)


def play_all(root: Path, players: dict, deals: int = 200, device: str | None = None, log=print) -> list[dict]:
    """Every pair's match, those of the file first; the new ones appended to it as they finish."""
    path = root / "championship.jsonl"
    done = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    met = {frozenset((m["a"], m["b"])) for m in done}
    loaded: dict = {}
    for a, b in itertools.combinations(players, 2):
        if frozenset((a, b)) in met:
            continue
        for name in (a, b):
            if name not in loaded:
                loaded[name] = players[name] if callable(players[name]) else contestant(players[name], device)
        res = play.duplicate_match(loaded[a], loaded[b], deals=deals, players=2)
        line = {"a": a, "b": b, **res}
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(line) + "\n")
        done.append(line)
        log(f"{a} - {b}: {res['a_wins']} - {res['a_losses']} of {res['games']}")
    return done


def ratings(matches: list[dict]) -> dict:
    return elo.fit(matches, ANCHOR)


def main() -> None:
    ap = argparse.ArgumentParser(description="The championship of a training run's phase ends and the bots.")
    ap.add_argument("root", help="the run's folder, with a folder phaseNN for every phase")
    ap.add_argument("--deals", type=int, default=200, help="deals a pair; each is played twice, the seats swapped")
    ap.add_argument("--device", default=None, help="where the networks run (default: the best the machine has)")
    a = ap.parse_args()
    root = Path(a.root)
    matches = play_all(root, players_of(root), a.deals, a.device)
    table = sorted(ratings(matches).items(), key=lambda kv: -kv[1]["elo"])
    (root / "elo.json").write_text(json.dumps({"anchor": ANCHOR, "matches": len(matches), "players": dict(table)}, indent=1) + "\n",
                                   encoding="utf-8")
    for name, v in table:
        print(f"{name:16s} {v['elo']:+6.0f}  [{v['lo']:+.0f}, {v['hi']:+.0f}]")


if __name__ == "__main__":
    main()
