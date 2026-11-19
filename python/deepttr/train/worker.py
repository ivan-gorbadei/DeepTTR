"""A self-play worker: games on one device, written as shards for the trainer.

The trainer starts one worker a device (`run`). A worker keeps pools of games of the engine
(`_engine.SelfPlay`), laid out by `plan`: self-play, a pool for each opponent of the league at each
table size, one for each table of several opponents, and the look-ahead's pools. Each round
(`Worker.round`) every pool advances its games to their next decisions; the learner's decisions of
all the pools are evaluated in one call of the newest network and each frozen network's in one call
of its own; the moves go back to the pools, and the records of finished games to the pools' shards.
Every 30 seconds the worker takes the trainer's newest network and route price; it waits while the
trainer owes more than MAX_OWED training samples, so that no more positions are made than are
trained on. Every ten minutes it writes a `selfplay` line: its pace and the pools' counters.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from deepttr import net as netmod
from deepttr.play import Evaluator
from deepttr.train import losses as L
from deepttr.train import replay as R
from deepttr.train.config import MIXED_TABLES, TrainConfig
from deepttr.train.process import compiles, logger, read_float, set_threads

MAX_OWED = 200_000
RELOAD_SECONDS, LINE_SECONDS, OWED_SECONDS = 30, 600, 5
CHUNK = 4096  # rows a network evaluates at once
TICKET_SLOTS = 2048  # the teacher's playouts of the choices of tickets under way at once
LOOK_GAMES, LOOK_SLOTS = 256, 8192  # the look-ahead's games a worker, and its playouts under way at once
# the engine's threads a pool, unless the run sets them: the pools of a round take their turns, and
# with many more threads a pool they spend their time waking and waiting on them
THREADS = 12
# at a table of three or more of several opponents the engine seats the card hoarder at one seat
# in half the games: it is not drawn there
FIXED_BOT = "hoard"


@dataclass
class Pool:
    """A pool of games of a worker, as `plan` lays it out: against `opponent` ("self", an
    opponent's name, or "mixed" for a table of several); `nets`, the frozen networks of its seats
    (indexes into the worker's list); `draws`, at a table of several opponents, the players of its
    other seats with their shares ("hoard", "net:0" - the pool's first network)."""

    opponent: str
    games: int
    players: int = 0  # 0: the run's tables
    bot: str = ""
    nets: tuple[int, ...] = ()
    draws: tuple[tuple[str, float], ...] = ()
    look: bool = False
    slots: int = 0  # playouts under way at once

    @property
    def prefix(self) -> str:
        """The beginning of its shards' names (`replay.pool_prefix`)."""
        return R.pool_prefix(self.opponent, self.players, self.look)

    @property
    def key(self) -> str:
        """Its name in the log: "self", "look", `<opponent>@<table>` or "look@<table>"."""
        if self.opponent == "self":
            return "look" if self.look else "self"
        return f"{'look' if self.look else self.opponent}@{self.players}"


def opponents_at(cfg: TrainConfig, n: int, nets: list[str]) -> dict:
    """A table of n where every other seat is drawn from the league: the draws, by the opponents'
    shares there, bots first, and the pool's networks."""
    draws, own = [], []
    for o in [o for o in cfg.league if o.bot] + [o for o in cfg.league if not o.bot]:
        share = o.shares.get(n, 0.0)
        if share <= 0 or (n >= 3 and o.player == FIXED_BOT):
            continue
        if o.bot:
            draws.append((o.player, share))
        else:
            draws.append((f"net:{len(own)}", share))
            own.append(nets.index(o.player))
    return {"draws": tuple(draws), "nets": tuple(own)}


def plan(cfg: TrainConfig) -> tuple[list[Pool], list[str]]:
    """A worker's pools and the frozen networks they need (their files). An opponent's pool at a
    table of n plays whole deals, each once from every seat: its games are its share of `games`
    rounded to a multiple of n, n at least; a table of several opponents gets as many as all their
    shares there; self-play the rest."""
    def units(share: float, n: int) -> int:
        return max(n, int(round(cfg.games * share / n)) * n)

    nets = list(dict.fromkeys(o.player for o in cfg.league if not o.bot))
    mixed = set(MIXED_TABLES) if cfg.mixed_tables else set()
    pools = []
    for o in [o for o in cfg.league if o.bot] + [o for o in cfg.league if not o.bot]:
        for n in cfg.tables:
            share = o.shares.get(n, 0.0)
            if share > 0 and n not in mixed:
                pools.append(Pool(o.name, units(share, n), n, bot=o.player if o.bot else "", nets=() if o.bot else (nets.index(o.player),)))
    for n in sorted(mixed):
        pools.append(Pool("mixed", units(sum(o.shares.get(n, 0.0) for o in cfg.league), n), n, **opponents_at(cfg, n, nets)))
    pools.insert(0, Pool("self", max(2, cfg.games - sum(p.games for p in pools)), slots=TICKET_SLOTS if cfg.ticket_playouts else 0))
    if cfg.lookahead and cfg.look_league:
        k = len(cfg.tables)
        for n in cfg.tables:
            pools.append(Pool("mixed", max(1, round(LOOK_GAMES / k)), n, look=True, slots=LOOK_SLOTS // k, **opponents_at(cfg, n, nets)))
    elif cfg.lookahead:
        pools.append(Pool("self", LOOK_GAMES, look=True, slots=LOOK_SLOTS))
    return pools, nets


def make_pool(p: Pool, cfg: TrainConfig, seed: int):
    """The engine's pool of games for `p`. Every pool plays by the run's rules. The learner plays a
    sample of its policy, the teacher's target being recorded where there is one; a frozen network
    plays its first choice; a pool against the league plays duplicate deals unless it looks ahead."""
    from deepttr import _engine as engine

    threads = cfg.threads or min(THREADS, os.cpu_count() or 1)
    kw = dict(games=p.games, seed=seed, threads=threads, rules=dict(cfg.rules) or None)
    if p.players:
        kw["players"] = p.players
    else:
        kw["tables"] = list(cfg.tables)
    if cfg.method == "teacher":
        kw.update(teacher=True, joint_paths=cfg.ticket_playouts, ticket_playouts=cfg.ticket_playouts)
    if p.bot:
        kw["bot"] = p.bot
    elif p.draws:
        kw["mixed"] = list(p.draws)
    elif p.nets:
        kw["frozen"] = True
    if p.look:
        kw.update(lookahead=True, look_check_keep=cfg.keep_check, look_forecast=cfg.move_forecast)
    if p.slots:
        kw["playout_slots"] = p.slots
    return engine.SelfPlay(**kw)


class Sink:
    """The records of one pool's finished games -> its shards, `size` records each, one shard
    being written at a time."""

    def __init__(self, replay: Path, prefix: str, worker: int, size: int) -> None:
        self.dir, self.prefix, self.worker, self.size = replay, prefix, worker, size
        found = [R.shard_pool(p.name) for d in (replay, replay.parent / "ticket_lessons") for p in d.glob(f"{prefix}-w{worker}-*.npz")]
        self.seq = max([s["seq"] for s in found if s is not None and s["prefix"] == prefix] + [-1]) + 1
        self.buf: list[dict] = []
        self.held = 0
        self.writer: threading.Thread | None = None

    def add(self, recs: dict) -> None:
        self.buf.append(recs)
        self.held += len(recs["policy"])
        if self.held >= self.size:
            self.close()
            path = self.dir / R.shard_name(self.prefix, self.worker, self.seq)
            self.writer = threading.Thread(target=R.write_shard, args=(self.buf, path))
            self.writer.start()
            self.seq += 1
            self.buf, self.held = [], 0

    def close(self) -> None:
        if self.writer is not None:
            self.writer.join()


def outputs(ev, obs: dict) -> dict:
    """A network's outputs for a batch of observations, CHUNK rows at a time: `logits` [rows,
    actions] and `forecast`, the critic's [rows] (None without the head), float32 numpy arrays;
    `move`, the move head's, a tensor on the device (None without the head)."""
    rows = len(obs["legal"])
    if not rows:
        return {"logits": np.zeros((0, obs["legal"].shape[1]), np.float32), "forecast": None, "move": None}
    parts = [ev.outputs({k: v[i:i + CHUNK] for k, v in obs.items()}) for i in range(0, rows, CHUNK)]
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    return {"logits": out["logits"].float().cpu().numpy(), "forecast": out["adv"].float().cpu().numpy() if "adv" in out else None,
            "move": out.get("move")}


def rows_of(o: dict, m: np.ndarray) -> dict:
    return o if len(m) == len(o["legal"]) else {k: v[m] for k, v in o.items()}


class Worker:
    """A worker's pools with their shards, and the networks that move in them: `learner`, the
    network being trained, and `frozen`, the league's (`Evaluator`s, or anything with their
    `outputs`)."""

    def __init__(self, run: Path, cfg: TrainConfig, index: int, pools: list[tuple[Pool, object]], learner, frozen: list, say) -> None:
        self.run, self.cfg, self.index, self.say = run, cfg, index, say
        self.pools, self.learner, self.frozen = pools, learner, frozen
        self.sinks = [Sink(run / "replay", p.prefix, index, R.LOOK_SHARD if p.look else R.SHARD) for p, _ in pools]
        self.hand = [0, 0.0]  # the turns the hand penalty fell on since the last line, and its points

    def round(self) -> int:
        """One round of every pool -> the rows the learner evaluated. The engine says who answers
        each row of a pool (`row_nets`): 0 the learner, k + 1 the pool's k-th frozen network. The
        learner's forecast of each of its decisions goes with its move: the records keep it."""
        obs, who = [], []
        for _, pool in self.pools:
            pool.prepare()
            obs.append(pool.observations())
            who.append(np.asarray(pool.row_nets()).astype(np.int64))
        mine = [np.flatnonzero(w == 0) for w in who]
        parts = [rows_of(o, m) for o, m in zip(obs, mine)]
        batch = parts[0] if len(parts) == 1 else {k: np.concatenate([o[k] for o in parts]) for k in parts[0]}
        own = outputs(self.learner, batch)
        theirs: dict[int, list] = {}  # a pool's rows of each frozen network: [(rows, logits)]
        by_net: dict[int, list] = {}
        for i, ((p, _), w) in enumerate(zip(self.pools, who)):
            for k in np.unique(w[w > 0]):
                by_net.setdefault(p.nets[k - 1], []).append((i, np.flatnonzero(w == k)))
        for g, items in by_net.items():
            lg = outputs(self.frozen[g], {k: np.concatenate([obs[i][k][r] for i, r in items]) for k in obs[items[0][0]]})["logits"]
            at = 0
            for i, r in items:
                theirs.setdefault(i, []).append((r, lg[at:at + len(r)]))
                at += len(r)
        at = 0
        for i, ((p, pool), o, m) in enumerate(zip(self.pools, obs, mine)):
            n = len(o["legal"])
            if p.look and self.cfg.move_forecast:
                self.forecast(pool, own["move"], at, m)
            fc = None if own["forecast"] is None else own["forecast"][at:at + len(m)]
            if len(m) == n:
                lg = own["logits"][at:at + n]
            else:
                lg = np.empty((n, o["legal"].shape[1]), np.float32)
                lg[m] = own["logits"][at:at + len(m)]
                for r, theirs_lg in theirs.get(i, []):
                    lg[r] = theirs_lg
                if fc is not None:
                    full = np.full(n, np.nan, np.float32)
                    full[m] = fc
                    fc = full
            at += len(m)
            pool.feed(lg, fc)
            recs = pool.take_records()
            if recs is not None:
                self.finished(recs)
                self.sinks[i].add(recs)
        return len(batch["legal"])

    @staticmethod
    def forecast(pool, move, at: int, mine: np.ndarray) -> None:
        """The move head's forecast for the rows of a look-ahead pool that ask for it (its learner's
        decisions), from this round's call - the pool's learner rows `mine` sit at `at` of the call's
        batch."""
        ask = np.asarray(pool.forecast_rows(), dtype=np.int64)
        if ask.size == 0 or move is None or not len(mine):
            return
        j = np.searchsorted(mine, ask)
        ok = (j < len(mine)) & (mine[np.minimum(j, len(mine) - 1)] == ask)
        if ok.any():
            pos = torch.as_tensor(at + j[ok], device=move.device)
            pool.feed_forecast(ask[ok].tolist(), np.ascontiguousarray(move[pos].float().cpu().numpy()))

    def finished(self, recs: dict) -> None:
        """A pool's records of whole games, before they go to a shard."""
        if self.cfg.hand_penalty:
            recs["hand_penalty"], turns, points = L.hand_penalty(recs)
            self.hand[0] += turns
            self.hand[1] += points

    def line(self, evals: int, seconds: float, waited: float) -> None:
        st = self.pools[0][1].stats()
        line = {"worker": self.index, "evals_per_s": round(evals / max(seconds, 1e-9)), "games_finished": st.get("games_finished"),
                "moves_per_game": round(float(st.get("moves_per_game", 0.0)), 1), "waited_s": round(waited)}
        counted = [pool.stats() for p, pool in self.pools if p.look] if self.cfg.lookahead else [st] if self.cfg.ticket_playouts else []
        if counted:
            sums = {k: sum(s[k] for s in counted) for k in counted[0] if all(isinstance(s.get(k), (int, float)) for s in counted)}
            if self.cfg.lookahead:
                sums["look_games"] = sums.pop("games_finished", 0)
                sums.pop("moves_per_game", None)
            line.update({k: v for k, v in sums.items() if k not in line})
        if self.cfg.hand_penalty:
            line.update(hand_turns=self.hand[0], hand_points=round(self.hand[1], 1))
            self.hand = [0, 0.0]
        league = {p.key: pool.stats().get("games_finished") for p, pool in self.pools if p.players and not p.look}
        if league:
            line["league_games"] = league
        self.say(event="selfplay", **line)

    def loop(self, stop, net: netmod.DeepTTRNet, device: torch.device) -> None:
        """Rounds until `stop()`; every RELOAD_SECONDS the newest network and route price."""
        latest = self.run / "latest.pt"
        stamp = latest.stat().st_mtime if latest.exists() else 0.0
        price = None
        t_reload = t_owed = t_line = time.time()
        evals, waited, held = 0, 0.0, False
        while not stop():
            if time.time() - t_owed > OWED_SECONDS:
                t_owed = time.time()
                held = (read_float(self.run / "owed") or 0.0) > MAX_OWED
            if held:
                time.sleep(1)
                waited += 1
                continue
            evals += self.round()
            if time.time() - t_reload > RELOAD_SECONDS:
                t_reload = time.time()
                try:
                    m = latest.stat().st_mtime
                    if m != stamp:
                        fresh, _ = netmod.load(latest, device)
                        net.load_state_dict({k: v for k, v in fresh.state_dict().items() if k in net.state_dict()})
                        stamp = m
                except Exception as exc:  # noqa: BLE001 - a file being replaced: the next time
                    print(f"worker {self.index}: reload failed: {exc}", flush=True)
                p = read_float(self.run / "route_price")
                if p is not None and p != price:
                    price = p
                    for _, pool in self.pools:
                        pool.set_route_price(p)
            if time.time() - t_line > LINE_SECONDS:
                self.line(evals, time.time() - t_line, waited)
                t_line, evals, waited = time.time(), 0, 0.0
        for s in self.sinks:
            s.close()


def run(run_dir: str, cfg: TrainConfig, index: int, device: str, stop=None) -> None:
    """A worker's process, started by the trainer: self-play on `device` until `stop` (an event)
    is set or the trainer is gone."""
    run_path = Path(run_dir)
    dev = torch.device(device)
    set_threads(dev, 2)
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # a worker may not start processes of its own
    say = logger(run_path, f"worker{index}")
    parent = mp.parent_process()
    checked = [time.time(), True]

    def stopped() -> bool:
        if stop is not None and stop.is_set():
            return True
        if parent is not None and time.time() - checked[0] > RELOAD_SECONDS:
            checked[:] = [time.time(), parent.is_alive()]
        return not checked[1]

    latest = run_path / "latest.pt"
    while not latest.exists():
        if stopped():
            return
        time.sleep(1)
    net, _ = netmod.load(latest, dev)
    learner = Evaluator(net, dev, compile=compiles(cfg.compile, dev))
    specs, nets = plan(cfg)
    frozen = [Evaluator(netmod.load(path, dev)[0], dev) for path in nets]
    seed = int(time.time()) * 10 + index
    pools = [(p, make_pool(p, cfg, seed + 1000 * k)) for k, p in enumerate(specs)]
    price = read_float(run_path / "route_price")
    if price is not None:
        for _, pool in pools:
            pool.set_route_price(price)
    say(event="league", worker=index, pools={p.key: p.games for p in specs}, nets=nets)
    Worker(run_path, cfg, index, pools, learner, frozen, say).loop(stopped, net, dev)
