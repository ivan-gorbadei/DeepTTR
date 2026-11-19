"""Strength measures of a training run's networks.

`table_match(a, b, players)` plays network A against opponent B in duplicate matches at a table of
`players` (every deal once with A in each seat, B in all the others) and returns one line: the
share of games A finished first, its points, the best other seat's points, its ticket points and
its completed tickets. A plays its first choice, or with `sample=True` a sample of its policy - the
moves its training plays; B, a network or a bot, plays its first choice. About 400 games at every
table, on deals of their own that training never deals.

`ticket_measure(a)` puts the network in both seats of two-player games by the second phase's
rules (only tickets score, and a failed ticket costs its points) and returns the ticket points a
seat: the skill the first two phases teach.

`goal(lines)` is the goal of a phase against its league: the mean, over the opponents and the
tables, of the share of games won.

`watch(run, plan)` measures a phase's snapshots as they appear in RUN/snapshots, always the newest
(the plan names the groups of matches - opponents, tables, move - and the ticket measures) and appends
the lines to RUN/measures.jsonl (matches) or RUN/tickets.jsonl (the ticket measure), each with the
snapshot's step and the training games played by then; it ends once the phase has ended
(RUN/phase.json) and its last snapshot is measured. The curriculum runs it beside the trainer:

    python -m deepttr.measure RUN_DIR PLAN.json [--device cuda:1]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deepttr import play

DEALS = {2: 200, 3: 134, 4: 100, 5: 80}  # about 400 games at every table
SEED = 1_000_003  # the measures' deals: a sequence of their own
TICKET_RULES = {"route_points": False, "longest_bonus": False}
TICKET_DEALS = 1024


class Sampler:
    """A network that plays a sample of its policy: Gumbel noise is added to its logits, so the
    first choice of the noisy logits is a draw from the policy over the legal moves."""

    def __init__(self, evaluator, seed: int) -> None:
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: dict) -> np.ndarray:
        logits = self.evaluator(obs)
        u = self.rng.random(logits.shape).clip(1e-12, 1 - 1e-12)
        return logits - np.log(-np.log(u)).astype(np.float32)


def table_match(a, b, players: int, deals: int | None = None, seed: int = SEED, sample: bool = False,
                threads: int = 0, rules: dict | None = None) -> dict:
    """A against B at tables of `players`: the counts of one measure line. A first place is a game A
    finished first alone."""
    deals = deals or DEALS[players]
    t0 = time.perf_counter()
    res = play.play_match(Sampler(a, seed + 7919) if sample else a, b, deals=deals, players=players, seed=seed,
                          threads=threads, rules=rules)
    games = len(res["a_seat"])
    rows = np.arange(games)
    seat = res["a_seat"].astype(np.int64)
    total = res["total"][:, :players].astype(np.float64)
    others = np.where(np.arange(players)[None, :] == seat[:, None], -np.inf, total)
    return {"players": players, "move": "sample" if sample else "first", "deals": deals, "games": games,
            "first_places": int((res["utility"][rows, seat] > 0.999).sum()),
            "points": round(float(total[rows, seat].mean()), 2),
            "opponent_points": round(float(others.max(axis=1).mean()), 2),
            "ticket_points": round(float(res["tickets"][rows, seat].mean()), 2),
            "completed": round(float(res["completed"][rows, seat].mean()), 3),
            "seconds": round(time.perf_counter() - t0, 1)}


def ticket_measure(a, deals: int | None = None, seed: int = SEED, threads: int = 0) -> dict:
    """The network in both seats of two-player games where only tickets score, by its first choice:
    the ticket points, the tickets held and those completed, a seat."""
    deals = deals or TICKET_DEALS
    t0 = time.perf_counter()
    res = play.play_match(a, a, deals=deals, players=2, seed=seed, threads=threads, rules=TICKET_RULES)
    per_seat = lambda key: round(float(res[key][:, :2].astype(np.float64).mean()), 3)  # noqa: E731
    return {"deals": deals, "move": "first", "ticket_points_per_seat": round(per_seat("tickets"), 2),
            "held_per_seat": per_seat("held"), "completed_per_seat": per_seat("completed"),
            "seconds": round(time.perf_counter() - t0, 1)}


def goal(lines: list[dict], move: str = "sample") -> float | None:
    """The mean share of games won over the measured cells (opponent and table) of one snapshot,
    or None without any."""
    shares = [line["first_places"] / line["games"] for line in lines if line.get("move") == move and line.get("games")]
    return sum(shares) / len(shares) if shares else None


def read_lines(path: Path) -> list[dict]:
    """A file of JSON lines; a line cut short by a writer under way is skipped."""
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def games_by_step(run: Path) -> list[tuple[int, int]]:
    """(step, training games played by then) of every train line of the run's log: self-play games
    and league games together."""
    total, out = 0, []
    for e in read_lines(run / "log.jsonl"):
        if e.get("event") == "train":
            total += int(e.get("g_games", 0)) + sum(int(p.get("g_games", 0)) for p in (e.get("league") or {}).values())
            out.append((int(e["step"]), total))
    return out


def snapshot_step(path: Path) -> int:
    return int(path.stem)


def opponent(spec: dict, device: str | None):
    """A measured opponent: {"name", "bot"} or {"name", "path"} (a network, by its first choice)."""
    if "bot" in spec:
        return play.Bot(spec["bot"])
    from deepttr import net as netmod

    network, _ = netmod.load(spec["path"])
    return play.Evaluator(network, device)


def measure_snapshot(network, plan: dict, rivals: dict) -> tuple[list[dict], list[dict]]:
    """One snapshot's lines by the plan: {"matches": [{"opponents": [...], "tables": [2, 3, 4, 5],
    "move": "sample" | "first"}, ...], "tickets": ["first", "sample"]}."""
    matches, tickets = [], []
    for group in plan.get("matches", []):
        for spec in group["opponents"]:
            for players in group["tables"]:
                matches.append({"opponent": spec["name"], **table_match(network, rivals[spec["name"]], players,
                                                                        sample=group["move"] == "sample")})
    for move in plan.get("tickets", []):
        if move == "first":
            tickets.append(ticket_measure(network))
        else:
            res = play.play_match(Sampler(network, SEED + 7919), Sampler(network, SEED + 104729), deals=TICKET_DEALS, players=2,
                                  seed=SEED, rules=TICKET_RULES)
            tickets.append({"deals": TICKET_DEALS, "move": "sample",
                            "ticket_points_per_seat": round(float(res["tickets"][:, :2].astype(np.float64).mean()), 2)})
    return matches, tickets


def watch(run: Path, plan: dict, device: str | None = None, every: float = 30.0) -> None:
    """Measure the newest snapshot not measured yet, again and again; the older ones it passed by
    while it measured are left out, so the measures keep up with a fast phase."""
    from deepttr import net as netmod

    rivals = {spec["name"]: opponent(spec, device) for group in plan.get("matches", []) for spec in group["opponents"]}
    lines = read_lines(run / "measures.jsonl") + read_lines(run / "tickets.jsonl")
    last = max((line["step"] for line in lines), default=-1)
    while True:
        ended = (run / "phase.json").exists()
        newer = [s for s in run.glob("snapshots/*.pt") if snapshot_step(s) > last]
        if not newer:
            if ended:
                return
            time.sleep(every)
            continue
        snap = max(newer, key=snapshot_step)
        try:
            network, _ = netmod.load(snap)
        except (OSError, RuntimeError, EOFError):  # a snapshot still being written: the next round
            time.sleep(every)
            continue
        step = snapshot_step(snap)
        matches, tickets = measure_snapshot(play.Evaluator(network, device), plan, rivals)
        played = [g for s, g in games_by_step(run) if s <= step]
        stamp = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "train_games": played[-1] if played else 0}
        for name, out_lines in (("measures.jsonl", matches), ("tickets.jsonl", tickets)):
            if out_lines:
                with (run / name).open("a", encoding="utf-8") as out:
                    out.writelines(json.dumps({**stamp, **line}) + "\n" for line in out_lines)
        last = step


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure every snapshot of a phase as it appears.")
    ap.add_argument("run", help="the phase's folder")
    ap.add_argument("plan", help="a JSON file: the opponents, tables and measures of the phase")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    watch(Path(a.run), json.loads(Path(a.plan).read_text(encoding="utf-8")), a.device)


if __name__ == "__main__":
    main()
