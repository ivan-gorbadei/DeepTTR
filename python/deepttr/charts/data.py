"""What the chart page and the static charts draw: a run root read into views and their charts.

A run root holds a folder a phase and the ratings of the phase ends:

    RUN_ROOT/phaseNN/log.jsonl       the trainer's lines ("train", every few minutes: step, losses,
                                     statistics of the games played since the line before, the
                                     league's pools keyed "<opponent>@<table>") and the workers'
                                     ("selfplay": counters since the worker started)
    RUN_ROOT/phaseNN/measures.jsonl  a line per measured snapshot, opponent, table and way of moving
    RUN_ROOT/phaseNN/tickets.jsonl   a line per snapshot and way of moving: ticket points a seat on
                                     the same deals of two players
    RUN_ROOT/phaseNN/phase.json      how the phase ended: its rule, verdict, numbers and cost
    RUN_ROOT/phaseNN/recipe.toml     its settings
    RUN_ROOT/elo.json                the ratings of the phase ends and the bots on one scale

`load(root)` reads it into views: the curriculum (all the phases on one time line) and each phase.
A view's charts are dicts drawn as they are. A point of a time chart is [games, step, hours, value],
and for a measured share also the ends of its 95 % interval; the reader picks which of the first
three is the x axis. Games are the training games played since the phase began, the league's
included; hours count the time the training ran, the stops before a restart left out.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python before 3.11
    tomllib = None

TABLES = (2, 3, 4, 5)
BOTS = ("card hoarder", "ticket builder")
INK = "ink"  # the colour of a summary line: the page's primary ink, not a series hue
Z = 1.96  # the 95 % interval
BIN = 1 / 6  # hours: the workers' counters are summed over ten minutes
MIN_GAMES = 100  # the training games behind a point of first places at a table
PHASE = re.compile(r"phase (\d+)$")
EPOCH = datetime(2000, 1, 1)
FILES = ("log.jsonl", "measures.jsonl", "tickets.jsonl", "phase.json", "recipe.toml")

# the charts of each phase, in order: what shows its skill being learned
CHARTS = {
    1: ("built",),
    2: ("built", "tickets", "fixed"),
    3: ("built", "tickets", "fixed", "tables"),
    4: ("price", "score", "tickets", "built", "bots"),
    5: ("goal", "vs", "score"),
    6: ("goal", "vs", "score"),
    7: ("goal", "vs", "score"),
    8: ("goal", "vs", "score"),
    9: ("goal", "elo", "vs", "score"),
    10: ("goal", "elo", "teacher", "critic", "speed", "vs", "score"),
    11: ("goal", "forecast", "proven", "score", "elo"),
}
OTHER = ("built", "tickets", "goal", "score")  # a phase the table does not name


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_lines(path: Path) -> list[dict]:
    """The JSON objects of a file, one a line; a line still being written or broken is passed over."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except OSError:
        pass
    return out


def seconds(at) -> float | None:
    """A line's time ("YYYY-MM-DDTHH:MM:SS"), in seconds; only differences are used."""
    try:
        return (datetime.strptime(str(at)[:19], "%Y-%m-%dT%H:%M:%S") - EPOCH).total_seconds()
    except ValueError:
        return None


def running_hours(lines: list[dict]) -> list[float]:
    """Each line's hours since the first: the time between consecutive lines, except the gap that
    ends at a start or resume line - the training was stopped then."""
    out, total, last = [], 0.0, None
    for e in lines:
        t = seconds(e.get("at"))
        if t is not None:
            if last is not None and t > last and e.get("event") not in ("start", "resume"):
                total += t - last
            last = t if last is None else max(last, t)
        out.append(total / 3600)
    return out


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    """y at x on the broken line through (xs, ys), xs rising; past the ends the end segments go on."""
    if len(xs) < 2:
        return ys[0] if ys else 0.0
    i = min(max(bisect.bisect_left(xs, x), 1), len(xs) - 1)
    a, b = i - 1, i
    if x < xs[0]:
        return ys[0]
    dx = xs[b] - xs[a]
    return ys[b] if dx <= 0 else ys[a] + (ys[b] - ys[a]) * (x - xs[a]) / dx


def pools(line: dict) -> dict[str, dict]:
    """A training line's league pools that played games since the line before."""
    return {k: v for k, v in (line.get("league") or {}).items() if isinstance(v, dict) and v.get("g_games")}


def line_games(line: dict) -> int:
    return int(line.get("g_games") or 0) + sum(int(v["g_games"]) for v in pools(line).values())


def split_key(key: str) -> tuple[str, int] | None:
    name, _, table = key.rpartition("@")
    return (name, int(table)) if name and table.isdigit() else None


def first_of(d: dict, *keys):
    for k in keys:
        if isinstance(d.get(k), (int, float)):
            return d[k]
    return None


class Clock:
    """Where a step or an hour of a phase falls on the other axes: the broken line through the
    phase's lines that name a step - its training lines, and its start, resume and end lines, which
    carry the games played by then."""

    def __init__(self, marks: list[tuple[float, float, float]]) -> None:
        self.s, self.n, self.h = [0.0], [0.0], [0.0]
        for s, n, h in marks:
            if s >= self.s[-1]:
                self.s.append(float(s))
                self.n.append(float(n))
                self.h.append(float(h))

    def at_step(self, step: float, games: float | None = None) -> tuple[float, float, float]:
        n = games if isinstance(games, (int, float)) else interp(step, self.s, self.n)
        return n, step, interp(step, self.s, self.h)

    def at_hour(self, h: float) -> tuple[float, float, float]:
        return interp(h, self.h, self.n), interp(h, self.h, self.s), h


def rnd(v: float | None, k: int = 4):
    return None if v is None or not math.isfinite(v) else round(v, k) + 0.0


def pt(x: tuple, y: float, lo: float | None = None, hi: float | None = None) -> list:
    n, s, h = x
    p = [int(round(n)), int(round(s)), rnd(h), rnd(y)]
    return p + [rnd(lo), rnd(hi)] if lo is not None else p


def series(name: str, color, points: list, style: str = "line", smooth: bool = False, width: float = 2) -> dict:
    return {"name": name, "color": color, "style": style, "smooth": smooth, "width": width, "points": points}


def chart(cid: str, title: str, about: str, unit: str, ser: list[dict], refs: list[dict] | None = None,
          kind: str = "time") -> dict | None:
    ser = [s for s in ser if s["points"]]
    if not ser:
        return None
    return {"id": cid, "kind": kind, "title": title, "about": about, "unit": unit, "series": ser, "refs": refs or []}


def wilson(k: float, n: float) -> tuple[float, float]:
    """The 95 % Wilson interval of a share k / n."""
    p, z2 = k / n, Z * Z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return mid - half, mid + half


def mean_share(cells: list[dict]) -> tuple[float, float, float]:
    """The mean first-place share of several cells and its 95 % interval (normal, cells independent)."""
    shares = [(c["first_places"] / c["games"], c["games"]) for c in cells]
    m = sum(p for p, _ in shares) / len(shares)
    se = math.sqrt(sum(p * (1 - p) / g for p, g in shares)) / len(shares)
    return m, max(0.0, m - Z * se), min(1.0, m + Z * se)


def opponent_order(names) -> list[str]:
    """The bots, then the phase ends from the newest, then any other opponent."""
    def key(name: str):
        m = PHASE.match(name)
        return (0, BOTS.index(name), "") if name in BOTS else (1, -int(m.group(1)), "") if m else (2, 0, name)
    return sorted(names, key=key)


class Phase:
    """One phase folder read: its training lines on the three axes, its measures, and its charts."""

    def __init__(self, folder: Path, number: int) -> None:
        self.folder, self.number = folder, number
        self.info = read_json(folder / "phase.json") or {}
        try:
            self.recipe = (folder / "recipe.toml").read_text(encoding="utf-8")
        except OSError:
            self.recipe = None
        lines = read_lines(folder / "log.jsonl")
        self.lines, self.hours = lines, running_hours(lines)
        self.rows, games, marks = [], 0, []
        for e, h in zip(lines, self.hours):
            if not isinstance(e.get("step"), (int, float)) or e.get("event") not in ("train", "start", "resume", "end"):
                continue
            if e["event"] == "train":
                games += line_games(e)
                self.rows.append({"n": games, "s": e["step"], "h": h, "line": e, "pools": pools(e)})
            marks.append((e["step"], games, h))
        self.clock = Clock(marks)
        self.measures = [m for m in read_lines(folder / "measures.jsonl")
                         if isinstance(m.get("step"), (int, float)) and m.get("games") and isinstance(m.get("first_places"), (int, float))]
        self.tickets = [t for t in read_lines(folder / "tickets.jsonl")
                        if isinstance(t.get("step"), (int, float)) and isinstance(t.get("ticket_points_per_seat"), (int, float))]
        if not self.info.get("name") and self.recipe and tomllib:
            try:
                about = tomllib.loads(self.recipe)
                self.info = {**self.info, **{k: about[k] for k in ("name", "about") if isinstance(about.get(k), str)}}
            except (ValueError, TypeError):
                pass
        self.done = bool(self.info.get("verdict"))
        ends = [self.clock.at_step(self.rows[-1]["s"])] if self.rows else []
        ends += [self.x_of(m) for m in self.measures + self.tickets]
        if self.done and isinstance(self.info.get("steps"), (int, float)):
            ends.append(self.clock.at_step(self.info["steps"]))
        self.extent = {"n": max([e[0] for e in ends], default=0.0), "s": max([e[1] for e in ends], default=0.0),
                       "h": max([e[2] for e in ends] + self.hours[-1:], default=0.0)}

    def x_of(self, m: dict) -> tuple[float, float, float]:
        return self.clock.at_step(m["step"], m.get("train_games"))

    def header(self) -> dict:
        i = self.info
        return {"phase": self.number, "name": i.get("name") or f"Phase {self.number}", "about": i.get("about") or "",
                "rule": i.get("rule"), "verdict": i.get("verdict"), "numbers": i.get("numbers") or {},
                "start": i.get("start") or (self.lines[0].get("at") if self.lines else None), "end": i.get("end"),
                "steps": i.get("steps", self.rows[-1]["s"] if self.rows else None),
                "games": i.get("games", self.rows[-1]["n"] if self.rows else None),
                "hours": i.get("hours", rnd(self.hours[-1], 2) if self.hours else None),
                "end_snapshot": i.get("end_snapshot"), "done": self.done, "last": self.lines[-1].get("at") if self.lines else None}

    # --- the training lines

    def at_table(self, row: dict, table: int, *keys):
        """A statistic of the network's games at tables of `table` in one training line: its league's
        pools there, weighted by their games, else the line's self-play at that table, else - a run of
        two-player self-play alone - the line itself."""
        got = [(p["g_games"], first_of(p, *keys)) for k, p in row["pools"].items() if (split_key(k) or ("", 0))[1] == table]
        got = [(g, v) for g, v in got if v is not None]
        if got:
            return sum(g * v for g, v in got) / sum(g for g, _ in got)
        line = row["line"]
        own = (line.get("selfplay_tables") or {}).get(str(table))
        if isinstance(own, dict) and own.get("g_games"):
            return first_of(own, *keys)
        if table == 2 and not line.get("selfplay_tables") and not row["pools"]:
            return first_of(line, *keys)
        return None

    def pooled(self, row: dict, *keys):
        """A statistic over all the line's games: its self-play and every pool of its league."""
        line = row["line"]
        got = [(line.get("g_games") or 0, first_of(line, *keys))] + [(p["g_games"], first_of(p, *keys)) for p in row["pools"].values()]
        got = [(g, v) for g, v in got if g and v is not None]
        return sum(g * v for g, v in got) / sum(g for g, _ in got) if got else None

    def line_series(self, name: str, color, value, smooth: bool = True, width: float = 2) -> dict:
        pts = []
        for r in self.rows:
            v = value(r)
            if v is not None:
                pts.append(pt((r["n"], r["s"], r["h"]), v))
        return series(name, color, pts, smooth=smooth, width=width)

    def tables_series(self, value) -> list[dict]:
        """A line a table size, and their mean where a line has all four."""
        out = [self.line_series(f"table of {t}", i, lambda r, t=t: value(r, t)) for i, t in enumerate(TABLES)]

        def mean(r):
            vals = [value(r, t) for t in TABLES]
            return None if any(v is None for v in vals) else sum(vals) / len(vals)
        return out + [self.line_series("mean of the tables", INK, mean, width=3)]

    # --- the charts

    def charts(self, elo: dict) -> list[dict]:
        out = []
        for what in CHARTS.get(self.number, OTHER):
            made = getattr(self, "chart_" + what)(elo) if what == "elo" else getattr(self, "chart_" + what)()
            out += [c for c in (made if isinstance(made, list) else [made]) if c]
        return out

    def chart_built(self):
        def share(r):
            e = r["line"]
            done, fail = first_of(e, "g_player_completed_start", "g_completed_start"), first_of(e, "g_player_failed_start", "g_failed_start")
            return 100 * done / (done + fail) if done is not None and fail is not None and done + fail > 0 else None
        th = (self.info.get("numbers") or {}).get("threshold")
        refs = [{"y": 100 * th if th <= 1 else th, "label": "the phase's threshold"}] if isinstance(th, (int, float)) else []
        return chart("built", "Starting tickets completed",
                     "Of the tickets the players kept at the start of their training games, the share they completed; "
                     "a point a training line.", "%", [self.line_series("completed", 0, share)], refs)

    def chart_tickets(self):
        def get(*keys):
            return lambda r: first_of(r["line"], *keys)
        return chart("tickets", "Tickets taken and completed",
                     "Per player and training game: all the tickets kept, those completed and those failed.", "tickets",
                     [self.line_series("kept", 0, get("g_player_tickets", "g_tickets")),
                      self.line_series("completed", 1, get("g_player_completed", "g_completed")),
                      self.line_series("failed", 2, get("g_player_failed", "g_failed"))])

    def chart_fixed(self):
        by = defaultdict(dict)
        for t in self.tickets:
            by[t.get("move")][t["step"]] = t
        ser = [series(label, i, [pt(self.x_of(t), t["ticket_points_per_seat"]) for _, t in sorted(by.get(move, {}).items())], "dots")
               for i, (move, label) in enumerate((("first", "first choice"), ("sample", "sampled, as in training")))]
        ser.append(self.line_series("training games of two", 2, lambda r: self.at_table(r, 2, "g_player_ticket_points", "g_ticket_points")))
        deals = max((t.get("deals") or 0 for t in self.tickets), default=0)
        return chart("fixed", "Ticket points on fixed deals",
                     f"Every snapshot plays the same {deals:,} deals of two players in both seats, only tickets scoring "
                     "and a failed ticket costing its points: the ticket points a seat. The training games of two "
                     "for comparison.", "points", ser)

    def chart_tables(self):
        return chart("tables", "Ticket points by table",
                     "Per player and training game at each table size: completed tickets minus failed ones.", "points",
                     self.tables_series(lambda r, t: self.at_table(r, t, "g_player_ticket_points", "g_ticket_points")),
                     [{"y": 0, "label": "zero"}])

    def chart_price(self):
        return chart("price", "Route price",
                     "The share of their points the routes score in the training games; it rises during the phase "
                     "until the scoring is the official one.", "%",
                     [series("route price", 0, [pt((r["n"], r["s"], r["h"]), 100 * r["line"]["route_price"]) for r in self.rows
                                                if isinstance(r["line"].get("route_price"), (int, float))], "step")])

    def chart_score(self):
        out = []
        for t in TABLES:
            total = lambda r, t=t: self.at_table(r, t, "g_player_score", "g_score")  # noqa: E731
            tickets = lambda r, t=t: self.at_table(r, t, "g_player_ticket_points", "g_ticket_points")  # noqa: E731

            def routes(r, t=t):
                a, b = total(r), tickets(r)
                return None if a is None or b is None else a - b
            out.append(chart(f"score{t}", f"Points a game at a table of {t}",
                             "The network's points in its training games at this table: all of them, those of the routes with "
                             "the longest trail's bonus, and those of the tickets.", "points",
                             [self.line_series("all", 0, total), self.line_series("routes and longest trail", 1, routes),
                              self.line_series("tickets", 2, tickets)]))
        return out

    def chart_bots(self):
        ser = []
        for i, bot in enumerate(BOTS):
            cells = sorted((m for m in self.measures if m.get("opponent") == bot and m.get("players") == 2 and m.get("move") == "first"),
                           key=lambda m: m["step"])
            ser.append(series(f"against the {bot}", i, [pt(self.x_of(m), 100 * m["first_places"] / m["games"],
                                                          *(100 * v for v in wilson(m["first_places"], m["games"]))) for m in cells], "dots"))
        return chart("bots", "Wins against the bots",
                     "Matches at a table of two on deals of their own, about 400 games a snapshot, the network by its "
                     "first choice; band: 95 % interval.", "%", ser, [{"y": 50, "label": "even"}])

    def goal_points(self, move: str = "sample", opponent: str | None = None) -> tuple[list, list[int]]:
        """The mean first-place share over the measured cells (opponent and table) of every snapshot
        that has them all, with its interval; and the tables of those cells."""
        lines = [m for m in self.measures if m.get("move") == move and (opponent is None or m.get("opponent") == opponent)]
        cells = {(m.get("opponent"), m.get("players")) for m in lines}
        by = defaultdict(dict)
        for m in lines:
            by[m["step"]].setdefault((m.get("opponent"), m.get("players")), m)
        pts = []
        for step in sorted(by):
            got = by[step]
            if len(got) == len(cells):
                mean, lo, hi = mean_share(list(got.values()))
                pts.append(pt(self.x_of(next(iter(got.values()))), 100 * mean, 100 * lo, 100 * hi))
        return pts, sorted({c[1] for c in cells if isinstance(c[1], int)})

    def chart_goal(self):
        pts, tables = self.goal_points()
        opps = {m.get("opponent") for m in self.measures if m.get("move") == "sample"}
        even = [{"y": 100 * sum(1 / t for t in tables) / len(tables), "label": "an even player"}] if tables else []
        return chart("goal", "The phase's goal: first places",
                     f"The share of games the network finished first, the mean over its {len(opps)} opponents and the "
                     f"tables of {', '.join(map(str, tables))}: matches at every measured snapshot, about 400 games a "
                     "table, the network moving as it trains (a sample of its policy); band: 95 % interval.",
                     "%", [series("goal", 0, pts, "dots")], even)

    def chart_vs(self):
        opps = defaultdict(set)
        for r in self.rows:
            for k in r["pools"]:
                got = split_key(k)
                if got and got[0] != "mixed":
                    opps[got[0]].add(got[1])
        out = []
        for name in opponent_order(opps):
            ser = self.won_series(name, sorted(opps[name]))
            measured, _ = self.goal_points(opponent=name)
            ser.append(series("mean of the tables, measured", 4, measured, "dots"))
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            out.append(chart(f"vs-{slug}", f"Against {name}: first places by table",
                             f"The share of the training games against {name} the network finished first at each table "
                             f"size, a point every {MIN_GAMES} games or more, and the mean of the tables; the dots are the "
                             "measured matches (about 400 games a table, the network moving as it trains) with their "
                             "95 % interval.", "%", ser,
                             [{"y": 100 * sum(1 / t for t in TABLES) / len(TABLES), "label": "an even player, mean"}]))
        return out

    def won_series(self, name: str, tables: list[int]) -> list[dict]:
        """The share of the training games against `name` the network finished first, a line a table
        and the mean of the tables. A point holds at least MIN_GAMES games of its table (of every
        table, for the mean): consecutive training lines are added up until they do, so a rare
        opponent gets fewer points, not noisier ones."""
        acc = {t: [0.0, 0.0] for t in tables}  # games and first places since the table's last point
        both = {t: [0.0, 0.0] for t in tables}  # the same since the mean's last point
        lines, mean = {t: [] for t in tables}, []
        for r in self.rows:
            x = (r["n"], r["s"], r["h"])
            for t in tables:
                p = r["pools"].get(f"{name}@{t}")
                if p and isinstance(p.get("g_won"), (int, float)):
                    for a in (acc[t], both[t]):
                        a[0] += p["g_games"]
                        a[1] += p["g_games"] * p["g_won"]
                if acc[t][0] >= MIN_GAMES:
                    lines[t].append(pt(x, 100 * acc[t][1] / acc[t][0]))
                    acc[t] = [0.0, 0.0]
            if all(both[t][0] >= MIN_GAMES for t in tables):
                mean.append(pt(x, 100 * sum(both[t][1] / both[t][0] for t in tables) / len(tables)))
                both = {t: [0.0, 0.0] for t in tables}
        return ([series(f"table of {t}", TABLES.index(t), lines[t], smooth=True) for t in tables]
                + [series("mean of the tables, training", INK, mean, smooth=True, width=3)])

    def worker_bins(self) -> list[dict]:
        """The workers' look-ahead counters, differenced per worker and summed over ten minutes."""
        keys = ("look_games", "look_checked", "look_proven", "forecast_flagged", "forecast_proven")
        was, when, bins, began = {}, {}, {}, 0.0
        for e, h in zip(self.lines, self.hours):
            if e.get("event") in ("start", "resume"):
                began = h  # the workers start again with the trainer: their counters begin at 0
            if e.get("event") != "selfplay" or not isinstance(e.get("look_proven"), (int, float)):
                continue
            by = str(e.get("by", ""))
            cur = {k: float(e.get(k) or 0) for k in keys}
            prev = was.get(by)
            restarted = prev is None or any(cur[k] < prev[k] for k in keys)
            diff = cur if restarted else {k: cur[k] - prev[k] for k in keys}
            dt = h - (began if restarted else when[by])
            was[by], when[by] = cur, h
            b = bins.setdefault(int(h / BIN), {"h": h, **{k: 0.0 for k in keys}, "rate": 0.0, "frate": 0.0, "timed": 0})
            b["h"] = max(b["h"], h)
            for k in keys:
                b[k] += diff[k]
            if dt > 0:  # lessons an hour: each worker's own rate, added up
                b["rate"] += diff["look_proven"] / dt
                b["frate"] += diff["forecast_proven"] / dt
                b["timed"] += 1
        return [bins[k] for k in sorted(bins)]

    def bin_series(self, name: str, color, bins: list[dict], value) -> dict:
        pts = []
        for b in bins:
            v = value(b)
            if v is not None:
                pts.append(pt(self.clock.at_hour(b["h"]), v))
        return series(name, color, pts, smooth=True)

    def chart_teacher(self):
        pts, games, proven = [], 0.0, 0.0  # a find is counted at its decision, a game at its end: a point holds MIN_GAMES games
        for b in self.worker_bins():
            games, proven = games + b["look_games"], proven + b["look_proven"]
            if games >= MIN_GAMES:
                pts.append(pt(self.clock.at_hour(b["h"]), proven / games))
                games = proven = 0.0
        finds = chart("teacher", "Look-ahead teacher: proven finds a game",
                      "In the teacher's games the look-ahead plays the network's first choice and its alternatives out "
                      "to the end in sampled deals; a find is a move proved better by 3 points and 2 standard errors, "
                      f"and becomes a lesson. Finds per teacher's game, a point every {MIN_GAMES} games or more.", "finds",
                      [series("proven finds a game", 0, pts, smooth=True)])
        agree = chart("agree", "First choice agrees with the lesson",
                      "On held-out positions of lessons, how often the network's own first choice is already the "
                      "lesson's move.", "%",
                      [self.line_series("agrees", 0, lambda r: 100 * r["line"]["val_agree_tickets"]
                                        if isinstance(r["line"].get("val_agree_tickets"), (int, float)) else None)])
        return [finds, agree]

    def chart_critic(self):
        return chart("critic", "Critic: the share of the rewards' spread it explains",
                     "On the training batches: 0 % - it foresees nothing, 100 % - every reward.", "%",
                     [self.line_series("explained", 0, lambda r: 100 * r["line"]["loss_critic_explained"]
                                       if isinstance(r["line"].get("loss_critic_explained"), (int, float)) else None)])

    def chart_speed(self):
        return chart("speed", "Training speed", "New positions a second from the games of the workers.", "positions/s",
                     [self.line_series("positions a second", 0, lambda r: first_of(r["line"], "positions_per_s"))])

    def chart_forecast(self):
        bins = self.worker_bins()
        return chart("forecast", "Move forecast: hints the check confirms",
                     "A hint is a move the move forecast rates well above the network's first choice; it goes straight "
                     "to the check in fresh deals. The share of hints the check proves better, against the share of the "
                     "ordinary checks; ten minutes a point.", "%",
                     [self.bin_series("hints confirmed", 0, bins, lambda b: 100 * b["forecast_proven"] / b["forecast_flagged"] if b["forecast_flagged"] else None),
                      self.bin_series("ordinary checks confirmed", 1, bins,
                                      lambda b: 100 * (b["look_proven"] - b["forecast_proven"]) / b["look_checked"] if b["look_checked"] else None)])

    def chart_proven(self):
        bins = [b for b in self.worker_bins() if b["timed"]]
        return chart("proven", "Lessons from proven finds an hour",
                     "Moves the look-ahead's check proved better than the network's first choice, an hour, all the "
                     "workers together; those its move forecast hinted at among them.", "lessons/h",
                     [self.bin_series("all", 0, bins, lambda b: b["rate"]), self.bin_series("from hints", 1, bins, lambda b: b["frate"])])

    def chart_elo(self, elo: dict):
        return elo_chart(elo, self.number)


def elo_chart(elo: dict, mark: int | None = None) -> dict | None:
    """The rating of every phase's end, the bots as lines across."""
    players = elo.get("players") or {}
    ends = sorted((int(m.group(1)), p) for name, p in players.items() if (m := PHASE.match(str(name))))
    if not ends:
        return None
    pts = [[k, rnd(p["elo"], 1), rnd(p.get("lo", p["elo"]), 1), rnd(p.get("hi", p["elo"]), 1)] for k, p in ends]
    games = elo.get("games")
    out = chart("elo", "Rating of each phase's end",
                "Every phase's end network and the two bots on one Elo scale, the ticket builder at 0, from duplicate "
                f"matches at a table of two{f' ({games:,} games)' if isinstance(games, int) else ''}, everyone by its "
                "first choice; bars: 95 % interval.", "Elo",
                [series("end of the phase", 0, pts, "dots")],
                [{"y": players[b]["elo"], "label": b} for b in BOTS if b in players], kind="phases")
    if out and mark is not None:
        out["mark"] = mark
    return out


def ratings(root: Path) -> dict:
    """elo.json as {"players": {name: {"elo", "lo", "hi", "games"}}, "games"}; empty without one."""
    data = read_json(root / "elo.json")
    if not isinstance(data, dict):
        return {}
    players = data.get("players") or {}
    if isinstance(players, list):
        players = {p.get("name"): p for p in players if isinstance(p, dict)}
    out = {}
    for name, p in players.items():
        r = first_of(p, "elo", "rating") if isinstance(p, dict) else None
        if r is not None:
            out[str(name)] = {"elo": r, "lo": first_of(p, "lo") if first_of(p, "lo") is not None else r,
                              "hi": first_of(p, "hi") if first_of(p, "hi") is not None else r, "games": p.get("games")}
    games = data.get("games")
    if not isinstance(games, int):
        played = [p.get("games") for p in out.values() if isinstance(p.get("games"), int)]
        games = sum(played) // 2 if played else None
    return {"players": out, "games": games}


def phase_folders(root: Path) -> list[tuple[int, Path]]:
    found = []
    for d in root.glob("phase[0-9][0-9]"):
        if d.is_dir():
            found.append((int(d.name[5:]), d))
    return sorted(found)


def version(root: Path) -> str:
    """Changes whenever a file the views are read from changes."""
    parts = []
    for path in [root / "elo.json"] + [d / f for _, d in phase_folders(root) for f in FILES]:
        try:
            st = path.stat()
            parts.append(f"{path.name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{path.name}:-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def shifted(points: list, by: dict) -> list:
    return [[p[0] + by["n"], p[1] + by["s"], rnd(p[2] + by["h"])] + p[3:] for p in points]


def curriculum(phases: list[Phase], elo: dict) -> dict:
    """All the phases on one time line: a band a phase, and the charts that run across them."""
    bands, start = [], {"n": 0.0, "s": 0.0, "h": 0.0}
    for ph in phases:
        bands.append({"phase": ph.number, "name": ph.header()["name"], "from": dict(start),
                      "to": {k: start[k] + ph.extent[k] for k in start}})
        start = bands[-1]["to"]
    bots, points = [[] for _ in BOTS], []
    for ph, band in zip(phases, bands):
        off = band["from"]
        for i, bot in enumerate(BOTS):
            cells = sorted((m for m in ph.measures if m.get("opponent") == bot and m.get("players") == 2 and m.get("move") == "first"),
                           key=lambda m: m["step"])
            bots[i] += shifted([pt(ph.x_of(m), 100 * m["first_places"] / m["games"],
                                   *(100 * v for v in wilson(m["first_places"], m["games"]))) for m in cells], off)
            if cells:
                bots[i].append(None)  # no line from one phase into the next
        mine = [pt((r["n"], r["s"], r["h"]), v) for r in ph.rows if (v := ph.pooled(r, "g_player_score", "g_score")) is not None]
        points += shifted(mine, off) + ([None] if mine else [])
    charts = [elo_chart(elo),
              chart("bots", "Wins against the bots, all phases",
                    "Matches at a table of two on deals of their own, about 400 games a measured snapshot, the network "
                    "by its first choice; band: 95 % interval. The rules of the first phases are not the official ones "
                    "the matches are played by.", "%",
                    [series(f"against the {bot}", i, [p for p in bots[i]], "dots") for i, bot in enumerate(BOTS)],
                    [{"y": 50, "label": "even"}]),
              chart("points", "Points a game in training, all phases",
                    "The network's mean final score in its training games, all tables and opponents together; the rules "
                    "and the opponents change from phase to phase.", "points",
                    [series("points a game", 0, points, smooth=True)])]
    for c in charts:
        if c:
            for s in c["series"]:
                while s["points"] and s["points"][-1] is None:
                    s["points"].pop()
    return {"bands": bands, "extent": dict(start), "charts": [c for c in charts if c]}


def load(root: str | Path) -> dict:
    """Every view of a run root: {"version", "phases": [headers], "views": {"curriculum": ..., "1": ...}}."""
    root = Path(root)
    elo = ratings(root)
    phases = [Phase(d, k) for k, d in phase_folders(root)]
    views = {"curriculum": {"view": "curriculum", **curriculum(phases, elo)}}
    for ph in phases:
        end = None
        if ph.done and isinstance(ph.info.get("steps"), (int, float)):
            n, s, h = ph.clock.at_step(ph.info["steps"])
            end = {"n": int(round(n)), "s": int(s), "h": rnd(h)}
        views[str(ph.number)] = {"view": ph.number, "header": {**ph.header(), "recipe": ph.recipe},
                                 "extent": {k: rnd(v) for k, v in ph.extent.items()}, "end": end, "charts": ph.charts(elo)}
    return {"version": version(root), "root": root.name, "phases": [ph.header() for ph in phases], "views": views}
