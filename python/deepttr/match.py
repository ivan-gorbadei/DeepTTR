"""A duplicate match between two players, from the command line.

usage: python -m deepttr.match A B [--deals 200] [--players 2] [--seed 1] [--device auto]
                                   [--look a|b|ab] [--look-worlds 8] [--look-check 24]
                                   [--name-a NAME] [--name-b NAME] [--append GAMES.jsonl]

A and B are each `net` (the published network, which comes with the package), a network checkpoint
(a file) or a baseline bot: `heuristic`, `hoard`, `random`; `python -m deepttr.match net hoard` is
the network against the card hoarder.
Every deal is played once with A in each seat and B in all the others. A network plays its first
choice; with `--look` the named side looks ahead before every move (`deepttr.look`), which costs
thousands of evaluations a move. Prints one JSON line: A's mean duplicate utility with its 95 %
interval, wins, losses, mean scores. `--append` adds the line to a file of matches, the input of
`python -m deepttr.elo`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from deepttr import net as netmod
from deepttr.device import pick_device
from deepttr.look import CHECK, WORLDS, Looker
from deepttr.play import BOTS, Bot, Evaluator, duplicate_match


NET = "net"  # the published network's name on the command line


def checkpoint(spec: str | None) -> Path:
    """The file a command-line name stands for: `net`, or no name, is the published network."""
    return netmod.PUBLISHED if spec in (None, NET) else Path(spec)


def player(spec: str, device, amp: bool = True, compile: bool = False):
    """A contestant from its command-line name: a bot, or a checkpoint's network on `device`."""
    if spec in BOTS:
        return Bot(spec)
    path = checkpoint(spec)
    if not path.is_file() and spec == NET:
        raise SystemExit(f"the published network is not in this installation: {path}")
    if not path.is_file():
        raise SystemExit(f"{spec}: not a checkpoint file, not `{NET}` (the published network) and not a bot ({', '.join(BOTS)})")
    net, _ = netmod.load(path)
    return Evaluator(net, device, amp=amp, compile=compile)


def name_of(spec: str) -> str:
    return spec if spec in BOTS else checkpoint(spec).stem


def add_run_options(ap: argparse.ArgumentParser) -> None:
    """The options every command that runs a network shares."""
    ap.add_argument("--device", default="auto", help="auto (CUDA, else Apple MPS, else CPU), cpu, cuda, cuda:1, mps")
    ap.add_argument("--no-amp", action="store_true", help="float32 even where the device has a native reduced precision")
    ap.add_argument("--compile", action="store_true", help="torch.compile the networks (long matches)")
    ap.add_argument("--threads", type=int, default=0, help="the engine's worker threads (0: all the cores)")


def main() -> None:
    ap = argparse.ArgumentParser(description="A duplicate match between two players.")
    ap.add_argument("a", help=f"{NET} (the published network), a checkpoint file, or a bot: {', '.join(BOTS)}")
    ap.add_argument("b", help="the same")
    ap.add_argument("--deals", type=int, default=200, help="each deal is played once per seat")
    ap.add_argument("--players", type=int, default=2, help="seats at the table, 2 to 5: A in one, B in the others")
    ap.add_argument("--seed", type=int, default=1, help="the sequence of deals")
    ap.add_argument("--first-deal", type=int, default=0, help="start at this deal of the sequence")
    ap.add_argument("--look", default="", choices=("", "a", "b", "ab"), help="the sides that look ahead before a move")
    ap.add_argument("--look-worlds", type=int, default=WORLDS, help="the first look: copies of the game per candidate move")
    ap.add_argument("--look-check", type=int, default=CHECK, help="the second look: copies per move")
    ap.add_argument("--name-a", default=None, help="A's name in the output (default: the file's or the bot's name)")
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--append", default=None, metavar="GAMES.jsonl", help="add the result to this file of matches")
    add_run_options(ap)
    x = ap.parse_args()
    device = pick_device(x.device)
    sides = {"a": player(x.a, device, not x.no_amp, x.compile), "b": player(x.b, device, not x.no_amp, x.compile)}
    names = {"a": x.name_a or name_of(x.a), "b": x.name_b or name_of(x.b)}
    for side in x.look:
        if isinstance(sides[side], Bot):
            raise SystemExit(f"--look {side}: a bot has no policy to look ahead with")
        sides[side] = Looker(sides[side], x.look_worlds, x.look_check, seconds=None)
        names[side] += "+look"
    m = duplicate_match(sides["a"], sides["b"], deals=x.deals, players=x.players, seed=x.seed, first_deal=x.first_deal,
                        threads=x.threads)
    line = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "a": names["a"], "b": names["b"], **m, "device": str(device)}
    if x.look:
        line["look"] = {"worlds": x.look_worlds, "check": x.look_check, **{side: sides[side].stats for side in x.look}}
    print(json.dumps(line))
    if x.append:
        with open(x.append, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
