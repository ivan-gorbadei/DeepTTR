"""The eleven phases of the network's training, run in order.

usage: python -m deepttr.curriculum RUN_ROOT [--phases 1-11] [--devices cuda:0,cuda:1] [--measure-device cuda:1]
                                    [--check STEPS]

Each phase is a recipe (deepttr/recipes/phaseNN.toml): what it trains, how, against whom, and when it
ends. Phase N runs in RUN_ROOT/phaseNN - the trainer and its workers, and beside them the measures of
its snapshots (`deepttr.measure`). It starts from the network the phase before it ended with, and its
league holds the bots and the ends of the earlier phases its recipe names. It ends by its rule:

- `built`: the players complete this share of the tickets they keep at the start (the training's
  own games, its newest line); it fails after `most_games` games short of it;
- `steps`: after this many training steps - as long as the phase of the original run trained;
- `full_price`: once the route price has risen to the whole;
- `plateau`: once its goal against its league - the share of games won, the mean over the
  opponents and the tables - has stopped rising: the last `n` measured snapshots no more than
  `gain` points above the `n` before them.

The end network is RUN_ROOT/phaseNN/end.pt, and RUN_ROOT/phaseNN/phase.json records how the phase
ended, its numbers and what it cost. A phase with a phase.json is not run again, so the command goes
on where it stopped. In the championship phases the share of games against each network of the
league follows (1 - w)^2 within a table, w the share of games the network wins against it, so that it
meets most the opponents it loses to.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from deepttr.measure import goal, read_lines

TABLES = (2, 3, 4, 5)
PHASES = 11
RECIPES = Path(__file__).resolve().parent / "recipes"
PLACES_LINES = 12  # the training lines the next phase's places are read from, about an hour
PLACES_GAMES = 120  # games a table of the measure that sets the first championship's places
POLL = 60.0  # seconds between two looks at a phase under way
BOTS = {"heuristic": "ticket builder", "hoard": "card hoarder"}


def built_share(line: dict) -> float | None:
    """A training line's share of the starting tickets its players completed, of those they kept."""
    done, failed = line.get("g_player_completed_start"), line.get("g_player_failed_start")
    if done is None or failed is None or done + failed <= 0:
        return None
    return float(done) / float(done + failed)


def trained(run: Path) -> dict:
    """What a phase has trained so far: steps, games (the league's too) and hours since its start."""
    log = read_lines(run / "log.jsonl")
    train = [e for e in log if e.get("event") == "train"]
    starts = [e["at"] for e in log if e.get("event") == "start" and "at" in e]
    games = sum(int(e.get("g_games", 0)) + sum(int(p.get("g_games", 0)) for p in (e.get("league") or {}).values()) for e in train)
    hours = (time.time() - time.mktime(time.strptime(starts[0][:19], "%Y-%m-%dT%H:%M:%S"))) / 3600 if starts else 0.0
    steps = [int(e["step"]) for e in train] + [int(e.get("step", 0)) for e in log if e.get("event") == "end"]
    return {"steps": max(steps, default=0), "games": int(games), "hours": round(hours, 2)}


def goals(run: Path, cells: int, move: str = "sample") -> list[tuple[int, float]]:
    """(step, goal) of every snapshot whose `cells` measures (opponents x tables) are all in."""
    by_step: dict[int, list[dict]] = {}
    for line in read_lines(run / "measures.jsonl"):
        if line.get("move") == move:
            by_step.setdefault(int(line["step"]), []).append(line)
    return [(step, goal(lines, move)) for step, lines in sorted(by_step.items()) if len(lines) >= cells]


def verdict(run: Path, end: dict, cells: int = 0) -> tuple[str | None, dict]:
    """("ended" | "failed" | None, the numbers): the phase's end rule on the run so far."""
    rule = end["rule"]
    if rule == "built":
        shares = [(e, built_share(e)) for e in read_lines(run / "log.jsonl") if e.get("event") == "train"]
        shares = [(e, s) for e, s in shares if s is not None]
        if shares and shares[-1][1] >= end["share"]:
            return "ended", {"built": round(shares[-1][1], 4), "step": shares[-1][0]["step"]}
        if trained(run)["games"] >= end["most_games"]:
            return "failed", {"built": round(shares[-1][1], 4) if shares else None, "games": trained(run)["games"]}
        return None, {}
    if rule == "steps":
        ends = [e.get("step", 0) for e in read_lines(run / "log.jsonl") if e.get("event") == "end"]
        if max(ends + [trained(run)["steps"]]) >= end["steps"]:
            return "ended", {"steps": end["steps"]}
        return None, {}
    if rule == "full_price":
        done = run / "full_price.json"
        return ("ended", json.loads(done.read_text(encoding="utf-8"))) if done.exists() else (None, {})
    if rule == "plateau":
        k, gain = int(end.get("n", 4)), float(end.get("gain", 1.0))
        vals = [g for _, g in goals(run, cells)]
        if len(vals) >= 2 * k:
            rise = 100 * (sum(vals[-k:]) - sum(vals[-2 * k:-k])) / k
            if rise <= gain:
                return "ended", {"goal": round(100 * sum(vals[-k:]) / k, 2), "rise": round(rise, 2), "checks": len(vals)}
        return None, {}
    raise ValueError(f"unknown end rule {rule!r}")


def training_wins(run: Path, lines: int = PLACES_LINES) -> dict[tuple[str, int], float]:
    """The share of games the network won in training against each opponent at each table, over a
    phase's last `lines` training lines (its `league` blocks, a pool keyed "opponent@table")."""
    won: dict[tuple[str, int], list[float]] = {}
    train = [e for e in read_lines(run / "log.jsonl") if e.get("event") == "train"][-lines:]
    for e in train:
        for key, pool in (e.get("league") or {}).items():
            name, _, table = key.rpartition("@")
            games = float(pool.get("g_games", 0))
            if games > 0 and pool.get("g_won") is not None:
                acc = won.setdefault((name, int(table)), [0.0, 0.0])
                acc[0] += games * float(pool["g_won"])
                acc[1] += games
    return {key: w / g for key, (w, g) in won.items()}


def places(names: list[str], wins: dict[tuple[str, int], float], totals: dict[int, float]) -> dict[str, dict[int, float]]:
    """Each network's share of a worker's games at each table: (1 - w)^2 among the networks of the
    league, scaled to the table's total. An opponent not met yet counts as an even one, w = 1/table."""
    out: dict[str, dict[int, float]] = {name: {} for name in names}
    for table, total in totals.items():
        weight = {name: (1 - wins.get((name, table), 1 / table)) ** 2 for name in names}
        norm = sum(weight.values()) or 1.0
        for name in names:
            out[name][table] = round(total * weight[name] / norm, 4)
    return out


def recipe(n: int) -> dict:
    """Phase `n`'s recipe."""
    with (RECIPES / f"phase{n:02d}.toml").open("rb") as fh:
        return tomllib.load(fh)


def phase_dir(root: Path, n: int) -> Path:
    return root / f"phase{n:02d}"


def end_of(root: Path, n: int) -> Path:
    """The network phase `n` ended with; it must be there."""
    path = phase_dir(root, n) / "end.pt"
    if not path.is_file():
        raise SystemExit(f"{path}: phase {n} has not ended, and a later phase needs its network")
    return path


def player_of(root: Path, player: str) -> str:
    """A recipe's player: a bot of the engine as it is, "phase N" the end of phase N of this run."""
    if player in BOTS:
        return player
    if player.startswith("phase "):
        return str(end_of(root, int(player.split()[1])))
    raise ValueError(f"unknown player {player!r}")


def measured_wins(root: Path, start: Path, names: dict[str, str], device: str | None) -> dict[tuple[str, int], float]:
    """The share of games the starting network wins against each network at each table, by a sample
    of its policy as it trains: the measure behind the first championship's places."""
    from deepttr import measure
    from deepttr import net as netmod
    from deepttr.play import Evaluator

    me = Evaluator(netmod.load(start)[0], device)
    wins = {}
    for name, path in names.items():
        rival = Evaluator(netmod.load(path)[0], device)
        for table in TABLES:
            line = measure.table_match(me, rival, table, deals=PLACES_GAMES // table, sample=True)
            wins[(name, table)] = line["first_places"] / line["games"]
    return wins


def config_of(root: Path, rec: dict, devices: tuple[str, ...], device: str | None):
    """The phase's TrainConfig, its league resolved: the earlier phases' ends, and the places of a
    championship phase. Also returns what set the places, for the phase's record."""
    from deepttr.train.config import Opponent, TrainConfig

    n = rec["phase"]
    train = dict(rec.get("train", {}))
    if "tables" in train:
        train["tables"] = tuple(train["tables"])
    league = rec.get("league", [])
    placed = [o["name"] for o in league if o["shares"] == "places"]
    shares, chosen = {}, {}
    if placed:
        spec = rec["places"]
        totals = {int(k): float(v) for k, v in spec["totals"].items()}
        if spec["from"] == "measured":
            wins = measured_wins(root, end_of(root, n - 1), {o["name"]: player_of(root, o["player"]) for o in league
                                                             if o["shares"] == "places"}, device)
        else:
            wins = training_wins(phase_dir(root, n - 1))
        shares = places(placed, wins, totals)
        chosen = {"from": spec["from"], "wins": {f"{k}@{t}": round(w, 4) for (k, t), w in sorted(wins.items())},
                  "shares": shares}
    opponents = tuple(Opponent(o["name"], player_of(root, o["player"]),
                               shares[o["name"]] if o["shares"] == "places" else {int(k): float(v) for k, v in o["shares"].items()})
                      for o in league)
    end = rec["end"]
    cfg = TrainConfig(init=None if n == 1 else str(end_of(root, n - 1)), steps=end.get("steps", 0) if end["rule"] == "steps" else 0,
                      league=opponents, devices=devices, **train)
    return cfg.check(), chosen


def measure_plan(cfg, rec: dict) -> dict:
    """What is measured of the phase's snapshots: every phase against the two bots at a table of two by
    its first choice (a line across the whole training); a phase with a league against each of its
    opponents at every table by a sample of its policy (its goal); the ticket measures it names."""
    def spec(o):
        return {"name": o.name, "bot": o.player} if o.bot else {"name": o.name, "path": o.player}

    groups = [{"opponents": [{"name": name, "bot": bot} for bot, name in BOTS.items()], "tables": [2], "move": "first"}]
    if cfg.league:
        groups.append({"opponents": [spec(o) for o in cfg.league], "tables": list(TABLES), "move": "sample"})
    return {"matches": groups, "tickets": rec.get("measure", {}).get("tickets", [])}


def start(cmd: list[str], out: Path) -> subprocess.Popen:
    log = out.open("ab")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def end_network(run: Path, rec: dict) -> Path:
    """The network the phase ends with: at full price the snapshot of that step, else the last."""
    if rec["end"]["rule"] == "full_price":
        return Path(json.loads((run / "full_price.json").read_text(encoding="utf-8"))["end"])
    return max(run.glob("snapshots/*.pt"), key=lambda p: int(p.stem))


def run_phase(root: Path, n: int, devices: tuple[str, ...], measure_device: str | None, log=print, check: int = 0) -> dict:
    """Phase `n`, from its start or where it stopped, to its end; its record. `check`: a check of the
    recipe - the phase trains this many steps and ends, whatever its rule."""
    run, rec = phase_dir(root, n), recipe(n)
    if check:
        rec = {**rec, "end": {"rule": "steps", "steps": check}}
    done = run / "phase.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():  # a phase to resume keeps the league and places it started with
        from deepttr.train.config import TrainConfig

        cfg = TrainConfig.load(run / "config.json")
        chosen = json.loads((run / "places.json").read_text(encoding="utf-8")) if (run / "places.json").exists() else {}
    else:
        cfg, chosen = config_of(root, rec, devices, measure_device)
        cfg.save(run / "config.json")
        if chosen:
            (run / "places.json").write_text(json.dumps(chosen, indent=1) + "\n", encoding="utf-8")
    plan = measure_plan(cfg, rec)
    (run / "measure.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(RECIPES / f"phase{n:02d}.toml", run / "recipe.toml")
    cells = len(plan["matches"][-1]["opponents"]) * len(TABLES) if cfg.league else 0
    log(f"phase {n}, {rec['name']}: {rec['end']['rule']}")
    trainer = start([sys.executable, "-m", "deepttr.train.trainer", str(run), "--config", str(run / "config.json")], run / "trainer.out")
    measurer = start([sys.executable, "-m", "deepttr.measure", str(run), str(run / "measure.json")]
                     + (["--device", measure_device] if measure_device else []), run / "measure.out")
    state, numbers, stopping = None, {}, False
    while trainer.poll() is None:
        state, numbers = verdict(run, rec["end"], cells)
        if state and not stopping:
            (run / "STOP").write_text(state, encoding="utf-8")
            stopping = True
        time.sleep(POLL)
    if state is None:
        state, numbers = verdict(run, rec["end"], cells)
    if state is None:
        measurer.terminate()
        raise SystemExit(f"phase {n}: the trainer stopped (exit code {trainer.returncode}) before the phase's end; "
                         f"see {run / 'trainer.out'} - the command goes on from here when run again")
    end = end_network(run, rec)
    shutil.copyfile(end, run / "end.pt")
    lines = read_lines(run / "log.jsonl")
    record = {"phase": n, "name": rec["name"], "about": " ".join(rec["about"].split()),
              "start": next((e["at"] for e in lines if e.get("event") == "start"), None), "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "rule": rec["end"]["rule"], "verdict": state, "numbers": numbers, **trained(run),
              "end_snapshot": str(end.relative_to(run)).replace("\\", "/"), **({"places": chosen} if chosen else {})}
    done.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    measurer.wait()  # it measures the last snapshot and ends
    log(f"phase {n} {state}: {numbers}; {record['steps']} steps, {record['games']} games, {record['hours']} h")
    if state == "failed":
        raise SystemExit(f"phase {n} failed: {numbers}")
    return record


def phases_of(spec: str) -> list[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m deepttr.curriculum", description="Run the phases of the training in order.")
    ap.add_argument("root", help="the run's folder: a folder phaseNN for each phase")
    ap.add_argument("--phases", default=f"1-{PHASES}", help="the phases to run, e.g. 1-11 or 5")
    ap.add_argument("--devices", default="", help="the training's devices, e.g. cuda:0,cuda:1 (default: every CUDA card, else MPS, else the CPU)")
    ap.add_argument("--measure-device", default=None, help="where the measures run (default: the best the machine has)")
    ap.add_argument("--check", type=int, default=0, metavar="STEPS",
                    help="a check of the recipes: every phase trains this many steps and ends, the next starting from it")
    a = ap.parse_args()
    root = Path(a.root)
    devices = tuple(d for d in a.devices.split(",") if d)
    for n in phases_of(a.phases):
        run_phase(root, n, devices, a.measure_device, check=a.check)


if __name__ == "__main__":
    main()
