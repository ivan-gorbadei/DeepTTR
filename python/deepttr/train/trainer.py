"""The trainer of a phase: it reads the workers' shards into a window of positions, trains the
network on batches drawn from it, and writes the network the workers play with.

    python -m deepttr.train.trainer RUN_DIR [--config FILE]

`train(run_dir, cfg)` runs a phase in RUN_DIR, a new one or one to resume. The directory holds what
its processes share:

- `config.json`: the phase (`TrainConfig`), read when no other file is given;
- `latest.pt`, `opt.pt`: the newest network and the optimizer's state, every `save_every` steps; a
  run that finds them resumes where they were;
- `snapshots/<step>.pt`: the network at the start, every `snapshot_steps` steps and at the end;
- `replay/`, `ticket_lessons/`: the workers' shards (`replay`) and the lessons kept from them while
  their store holds them, so that a restart refills it;
- `log.jsonl`: one JSON object a line, `{"at", "by", "event", ...}`; every five minutes a `train`
  line - the losses, the held-out losses (`val_*`), the self-play games' statistics (`g_*`), and
  the league's by opponent and table (`league`, keyed `<opponent>@<table>`);
- `owed`: the training samples owed to the new positions; `route_price`: the price of a route in
  new games while it rises; `ppo_controls.json`: PPO's two adaptive weights; `full_price.json`: the
  step at which the price became whole and the snapshot of that step, the phase's end;
- `STOP`: written by whoever wants the run to end. The trainer saves, ends the workers, writes its
  `end` line and removes the file; so it does at the end of its `steps`.

A step trains on BATCH positions: the window's, and up to a tenth lessons from their own store,
each lesson about eight times. The trainer owes `ratio` samples to each new position and trains
while it owes a batch; the workers wait while it owes too much.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import torch

from deepttr import net as netmod
from deepttr.net import OBS_KEYS, DeepTTRNet, NetConfig
from deepttr.train import losses as L
from deepttr.train import replay as R
from deepttr.train import worker as W
from deepttr.train.config import TrainConfig
from deepttr.train.process import (autocast, compiles, logger, read_events, replace, run_devices, save_net,
                                   set_threads, train_dtype, write_text)

BATCH = 1024
WARMUP = 500
BETAS, WEIGHT_DECAY, CLIP_NORM = (0.9, 0.95), 1e-4, 1.0
LOG_SECONDS = 300
VAL_WINDOW, VAL_LESSONS = 50_000, 10_000
LESSON_WINDOW, LESSON_SHARE, LESSON_REUSE, LESSON_MIN = 50_000, 0.1, 8.0, 64
ADAPT_STEPS = 100  # PPO's weights follow the entropy and the KL measured over this many steps
# a record's `opponent`, by which PPO compares a move with those of its kind of game; a network
# of the league: NETS_FROM + its place among the league's networks
BOT_IDS = {"hoard": 1, "heuristic": 2}
SELF_ID, MIXED_ID, NETS_FROM = 0, 4, 9


def opponent_ids(cfg: TrainConfig) -> dict[str, int]:
    """Each opponent's number by the shards' spelling of its name."""
    ids = {"self": SELF_ID, "mixed": MIXED_ID}
    nets = [o for o in cfg.league if not o.bot]
    ids.update({R.spelling(o.name): BOT_IDS[o.player] if o.bot else NETS_FROM + nets.index(o) for o in cfg.league})
    return ids


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = (t.float() if t.dtype == torch.float16 else t).to(device, non_blocking=True)
    return out


def loss_weights(cfg: TrainConfig) -> dict:
    return {**L.WEIGHTS, "trail": cfg.trail_weight, "critic": L.CRITIC_WEIGHT, "move": L.MOVE_WEIGHT}


def player_last(recs: dict) -> np.ndarray:
    """The last record of every player of every game; a player is told by its starting tickets."""
    g = recs["game"].astype(np.int64)
    st = recs["ticket_start"].astype(np.int64)
    who = (st << np.arange(st.shape[1], dtype=np.int64)).sum(1)
    order = np.lexsort((recs["moves"], who, g))
    kg, kw = g[order], who[order]
    return order[np.r_[(kg[1:] != kg[:-1]) | (kw[1:] != kw[:-1]), True]]


def game_stats(recs: dict, points: np.ndarray, n_claim: int) -> dict:
    """The statistics of the games of a shard: from each game's last record, the scores of its
    player and of the others, its tickets completed and failed (those kept at the opening apart:
    `_start`), the game's length and whether that player won; from every player's last record, its
    score, tickets and ticket points (`player_*`); over all the records, the policy's mass on claims,
    on drawing cards and on drawing tickets."""
    g, mv = recs["game"], recs["moves"]
    order = np.lexsort((mv, g))
    last = order[np.r_[g[order][1:] != g[order][:-1], True]]
    out = recs["ticket_outcome"][last]
    s = recs["ticket_start"][last].astype(bool)
    score = recs["score"].astype(np.float32)
    n = recs["n_players"].astype(np.int64)
    pol = recs["policy"].astype(np.float32)
    others = np.where(np.arange(1, score.shape[1])[None, :] < n[last, None], score[last, 1:], -1e9)
    st = {
        "games": int(len(last)), "score": float(score[last, 0].mean() * 100), "opp_score": float(score[last, 1].mean() * 100),
        "best_opp_score": float(others.max(1).mean() * 100),
        "completed": float((out == 1).sum(1).mean()), "failed": float((out == -1).sum(1).mean()), "moves": float(mv[last].mean()),
        "p_claim": float(pol[:, :n_claim].sum(1).mean()), "p_cards": float(pol[:, n_claim:n_claim + 6].sum(1).mean()),
        "p_tickets": float(pol[:, n_claim + 6].mean()),
        "completed_start": float(((out == 1) & s).sum(1).mean()), "failed_start": float(((out == -1) & s).sum(1).mean()),
        "completed_drawn": float(((out == 1) & ~s).sum(1).mean()), "failed_drawn": float(((out == -1) & ~s).sum(1).mean()),
        "tickets": float((out != 0).sum(1).mean()), "ticket_points": float((out.astype(np.float32) * points[None, :]).sum(1).mean()),
    }
    if "value" in recs:
        st["won"] = float((recs["value"][last, 0] >= 0.999).mean())
    pl = player_last(recs)
    o = recs["ticket_outcome"][pl].astype(np.float32)
    ps = recs["ticket_start"][pl].astype(bool)
    st.update({"player_score": float(score[pl, 0].mean() * 100), "player_tickets": float((o != 0).sum(1).mean()),
               "player_completed": float((o == 1).sum(1).mean()), "player_failed": float((o == -1).sum(1).mean()),
               "player_ticket_points": float((o * points[None, :]).sum(1).mean()),
               "player_completed_start": float(((o == 1) & ps).sum(1).mean()),
               "player_failed_start": float(((o == -1) & ps).sum(1).mean())})
    return st


def pooled(stats: list[tuple[int, dict]]) -> dict:
    """The statistics of several shards, weighted by their games, as `g_*` fields."""
    n = sum(k for k, _ in stats)
    if not n:
        return {}
    keys = [k for k in stats[0][1] if k != "games" and all(k in st for _, st in stats)]
    return {"g_games": n, **{f"g_{k}": round(sum(c * st[k] for c, st in stats) / n, 3) for k in keys}}


class Data:
    """The trainer's positions: the window, the lessons' store, the held-out stores, the samples
    owed, and the statistics of the games read since the last training line."""

    def __init__(self, run: Path, cfg: TrainConfig) -> None:
        from deepttr import _engine

        self.run, self.cfg = run, cfg
        looks = cfg.move_forecast
        self.window, self.val = R.Replay(cfg.window, looks), R.Replay(VAL_WINDOW, looks)
        self.lessons, self.val_lessons = R.Replay(LESSON_WINDOW, looks), R.Replay(VAL_LESSONS, looks)
        self.seen: set[str] = set()
        self.sizes: dict[str, int] = {}  # the records of each shard read
        self.kept: deque = deque()  # (a lesson file, its lessons), oldest first
        self.owed = self.owed_lessons = 0.0
        self.ids = opponent_ids(cfg)
        self.names = {R.spelling(o.name): o.name for o in cfg.league}
        self.points = np.array([t["points"] for t in json.loads(_engine.map_json())["tickets"]], np.float32)
        self.n_claim = _engine.dims()["n_claim"]
        self.new_line()

    def new_line(self) -> None:
        self.new = self.new_lessons = self.in_batches = 0
        self.selfplay: list = []
        self.league: dict[str, list] = {}
        self.tables: dict[str, list] = {}

    def shards(self) -> list[Path]:
        return sorted((self.run / "replay").glob("*.npz"), key=lambda p: p.stat().st_mtime)

    def take(self, path: Path, recs: dict, shares: tuple[float, float], restoring: bool = False) -> None:
        """A shard's records into the stores: its moves of games into the window (or the held-out
        store), marked by their opponent and, with TD targets, given them; its lessons into theirs
        and into `ticket_lessons/`. A shard read on a restart was trained on already: it owes nothing."""
        s = R.shard_pool(path.name)
        self.sizes[path.name] = len(recs["policy"])
        if self.cfg.td:
            recs["adv_td"] = L.td_targets(recs, *shares)
        games, lessons = R.split_lessons(recs)
        games["opponent"] = np.full(len(games["policy"]), self.ids.get(s["opponent"], NETS_FROM), np.uint8)
        held = R.held_out(path.name)
        if lessons is not None and not (self.run / "ticket_lessons" / path.name).exists():
            try:
                kept = self.keep_lessons(path, lessons)
                if not restoring:  # a restart reads the lesson files afterwards
                    self.kept.append(kept)
            except OSError:  # a full disk: into their store straight away
                if restoring:
                    (self.val_lessons if held else self.lessons).add(lessons)
        (self.val if held else self.window).add(games)
        if restoring:
            return
        if held:
            if lessons is not None:
                self.val_lessons.add(lessons)
            return
        self.owed += len(games["policy"]) * self.cfg.ratio
        self.new += len(games["policy"])
        if lessons is not None:
            self.lessons.add(lessons)
            self.owed_lessons += len(lessons["policy"]) * LESSON_REUSE
            self.new_lessons += len(lessons["policy"])
        self.note(s, recs)

    def keep_lessons(self, shard: Path, lessons: dict) -> tuple[Path, int]:
        path = self.run / "ticket_lessons" / shard.name
        R.write_shard([lessons], path)
        st = shard.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        return path, len(lessons["policy"])

    def note(self, s: dict, recs: dict) -> None:
        st = game_stats(recs, self.points, self.n_claim)
        if s["opponent"] != "self":  # the league's games, the look-ahead's against it too
            key = f"{self.names.get(s['opponent'], s['opponent'])}@{s['table']}"
            self.league.setdefault(key, []).append((st["games"], st))
            return
        st.pop("won", None)
        self.selfplay.append((st["games"], st))
        if len(self.cfg.tables) > 1:
            for t in np.unique(recs["n_players"]):
                sel = recs["n_players"] == t
                sv = game_stats({k: v[sel] for k, v in recs.items()}, self.points, self.n_claim)
                sv.pop("won", None)
                self.tables.setdefault(str(int(t)), []).append((sv["games"], sv))

    def read_new(self, shares: tuple[float, float], restoring: bool = False) -> None:
        """The shards not read yet; one that cannot be read now is tried again the next time."""
        for p in self.shards():
            if p.name in self.seen or R.shard_pool(p.name) is None:
                continue
            try:
                recs = R.read_shard(p)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                continue
            self.take(p, recs, shares, restoring)
            self.seen.add(p.name)

    def restore(self, shares: tuple[float, float]) -> None:
        """A restart: the window from the shards on disk, the lessons' store from their files."""
        self.read_new(shares, restoring=True)
        for p in sorted((self.run / "ticket_lessons").glob("*.npz"), key=lambda q: q.stat().st_mtime):
            try:
                tk = R.read_shard(p)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):  # a broken file: its lessons are lost
                continue
            (self.val_lessons if R.held_out(p.name) else self.lessons).add(tk)
            self.kept.append((p, len(tk["policy"])))

    def prune(self) -> None:
        """The oldest shards beyond the window and the oldest lesson files beyond their store."""
        files = [p for p in self.shards() if p.name in self.sizes]
        n = sum(self.sizes[p.name] for p in files)
        while len(files) > 1 and n > self.cfg.window * 1.2:
            p = files.pop(0)
            if unlink(p):
                n -= self.sizes.pop(p.name)
        n = sum(k for _, k in self.kept)
        while len(self.kept) > 1 and n > LESSON_WINDOW * 1.2:
            p, k = self.kept.popleft()
            unlink(p)
            n -= k

    def batch(self, rng: np.random.Generator) -> tuple[dict, int]:
        """A batch's arrays and how many lessons are its last rows."""
        nt = min(int(self.owed_lessons), int(LESSON_SHARE * BATCH)) if self.lessons.size >= LESSON_MIN else 0
        return R.mixed_batch(self.window, self.lessons, nt, BATCH, rng), nt


def unlink(p: Path) -> bool:
    try:
        p.unlink(missing_ok=True)
        return True
    except OSError:  # held open by another process: the next time
        return False


class Controls:
    """PPO's two adaptive weights - the KL penalty's and the entropy hold's - and the hold's target,
    the policy's entropy over the phase's first ADAPT_STEPS steps; they go on over a restart
    (`ppo_controls.json`, the log's `entropy_hold` line)."""

    def __init__(self, run: Path, step: int, say) -> None:
        self.path, self.say = run / "ppo_controls.json", say
        self.kl_beta, self.hold, self.target = L.KL_BETA, 0.0, None
        self.entropy = self.kl = 0.0
        self.n = 0
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.kl_beta = min(1e3, max(1e-3, float(d["kl_beta"])))
            self.hold = min(1.0, max(0.0, float(d["entropy_hold"])))
            say(event="ppo_controls", step=step, kl_beta=round(self.kl_beta, 4), entropy_hold=round(self.hold, 4), as_of=d.get("step"))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        for e in read_events(run / "log.jsonl"):
            if e.get("event") == "entropy_hold" and e.get("target") is not None:
                self.target = float(e["target"])
                say(event="entropy_hold", step=step, target=round(self.target, 4), source="the log")
                break

    def note(self, step: int, stats: dict) -> None:
        """After a step's loss: the weights it used into `stats`; every ADAPT_STEPS steps the hold
        follows the entropy and the KL penalty the KL measured over them."""
        stats["entropy_hold"], stats["ppo_beta"] = self.hold, self.kl_beta
        self.entropy += stats["entropy"]
        self.kl += stats["ppo_kl"]
        self.n += 1
        if self.n < ADAPT_STEPS:
            return
        if self.target is None:
            self.target = self.entropy / self.n
            self.say(event="entropy_hold", step=step, target=round(self.target, 4), source="the phase's first steps")
        else:
            self.hold = L.adapt_entropy_hold(self.hold, self.entropy / self.n, self.target)
        self.kl_beta = L.adapt_kl_beta(self.kl_beta, self.kl / self.n)
        self.entropy = self.kl = 0.0
        self.n = 0
        write_text(self.path, json.dumps({"step": step, "kl_beta": self.kl_beta, "entropy_hold": self.hold}))


def start_network(run: Path, cfg: TrainConfig, device: torch.device, say) -> tuple[DeepTTRNet, int, bool]:
    """The network a run trains: `latest.pt` when the run resumes, else `cfg.init` or random
    weights; with the heads the phase needs added (`net.grow`). -> (it, the step, resumed)."""
    latest = run / "latest.pt"
    if latest.exists():
        net, meta = netmod.load(latest, device)
        step, resumed = int(meta.get("step", 0)), True
        say(event="resume", step=step, args=cfg.to_dict(), loss_weights=loss_weights(cfg))
    else:
        if cfg.init:
            net, _ = netmod.load(cfg.init, device)
        else:
            from deepttr import _engine

            config = NetConfig(outcome_heads=cfg.outcome_heads, adv_head=cfg.critic, move_head=cfg.move_forecast)
            net = DeepTTRNet(_engine.dims(), config).to(device)
        step, resumed = 0, False
        say(event="start", params=net.n_params(), config=net.cfg.as_dict(), args=cfg.to_dict(), loss_weights=loss_weights(cfg))
    grown = False
    for event, on, heads in (("outcome_heads", cfg.outcome_heads == "tokens", {"outcome_heads": "tokens"}),
                             ("trail_head", True, {"trail_head": True}), ("adv_head", cfg.critic, {"adv_head": True}),
                             ("move_head", cfg.move_forecast, {"move_head": True})):
        new = netmod.grow(net, **heads) if on else net
        if new is not net:
            net, grown = new, True
            say(event=event, params=net.n_params())
    if not resumed or grown:
        save_net(latest, net, step)
    if not resumed:
        save_net(run / "snapshots" / f"{step:08d}.pt", net, step)
    return net.train(), step, resumed


def make_optimizer(run: Path, net: DeepTTRNet, step: int, device: torch.device, say) -> tuple[torch.optim.Optimizer, int]:
    """AdamW, its state from `opt.pt` when the run has one that fits -> (it, the step from which a
    fresh one warms up)."""
    opt = torch.optim.AdamW(net.parameters(), lr=1e-9, betas=BETAS, weight_decay=WEIGHT_DECAY)
    path = run / "opt.pt"
    if path.exists():
        try:
            opt.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            return opt, 0
        except Exception:  # noqa: BLE001 - another set of weights (a head added), or a broken file
            say(event="optimizer_reset", warmup_from=step)
    elif step:
        say(event="optimizer_fresh", warmup_from=step)
    return opt, step


def save_optimizer(run: Path, opt: torch.optim.Optimizer) -> None:
    tmp = run / "opt.pt.tmp"
    torch.save(opt.state_dict(), tmp)
    replace(tmp, run / "opt.pt")


def validate(net: DeepTTRNet, data: Data, cfg: TrainConfig, device: torch.device, dtype) -> dict:
    """The losses on held-out games (four batches) and the share of their positions where the
    network's first choice is the target's (`agree`); of held-out lessons, the same share
    (`agree_tickets`)."""
    if data.val.size < BATCH:
        return {}
    acc: dict[str, float] = {}
    rng = np.random.default_rng(0)
    with torch.no_grad(), autocast(device, dtype):
        for _ in range(4):
            vb = to_device({k: v for k, v in data.val.sample(BATCH, rng).items() if k not in R.LOOKS}, device)
            out = net({k: vb[k] for k in OBS_KEYS})
            total, parts, _ = L.batch_loss(out, vb, "teacher", trail_weight=cfg.trail_weight)
            for k, v in {**parts, "total": total}.items():
                acc[k] = acc.get(k, 0.0) + float(v) / 4
            acc["agree"] = acc.get("agree", 0.0) + float((out["logits"].argmax(-1) == vb["policy"].argmax(-1)).float().mean()) / 4
        if data.val_lessons.size >= (32 if cfg.lookahead else 256):
            tb = to_device(data.val_lessons.sample(min(BATCH, data.val_lessons.size), rng), device)
            lg = net({k: tb[k] for k in OBS_KEYS})["logits"]
            acc["agree_tickets"] = float((lg.argmax(-1) == tb["policy"].argmax(-1)).float().mean())
    return {f"val_{k}": round(v, 4) for k, v in acc.items()}


def start_workers(run: Path, cfg: TrainConfig, devices: list[torch.device], say):
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    procs = [ctx.Process(target=W.run, args=(str(run), cfg, i, str(d), stop), daemon=True) for i, d in enumerate(devices)]
    for p in procs:
        p.start()
    say(event="processes", devices=[str(d) for d in devices], workers=[p.pid for p in procs], trainer=os.getpid())
    return procs, stop


def train(run_dir, cfg: TrainConfig, workers: bool = True) -> int:
    """Run a phase in `run_dir` until its `steps`, a file STOP, the whole route price, or a worker's
    death -> the step it ended at. `workers=False`: no self-play; the trainer trains on the shards
    the run directory already holds and ends when they are used up."""
    cfg.check()
    run = Path(run_dir).resolve()
    for d in ("replay", "snapshots", "ticket_lessons"):
        (run / d).mkdir(parents=True, exist_ok=True)
    for p in (run / "replay").glob("*.tmp"):  # shards cut short by the last stop
        unlink(p)
    say = logger(run, "trainer")
    devices = run_devices(cfg.devices)
    device = devices[0]
    set_threads(device, 1 + sum(d.type == "cpu" for d in devices) if workers else 1)
    dtype = train_dtype(device)
    net, step, resumed = start_network(run, cfg, device, say)
    opt, fresh_at = make_optimizer(run, net, step, device, say)
    fwd = torch.compile(net) if compiles(cfg.compile, device) else net

    def shares(at: int) -> tuple[float, float]:
        return L.bonus_shares(at, cfg.bonus_ramp) if cfg.bonus else (0.0, 0.0)

    data = Data(run, cfg)
    if resumed:
        data.restore(shares(step))
        if data.seen:
            say(event="window_reloaded", shards=len(data.seen), positions=data.window.size, ticket_lessons=data.lessons.size)
    ctl = Controls(run, step, say) if cfg.method == "ppo" else None
    start_price = float(cfg.rules.get("route_price", 1.0))

    def price_at(at: int) -> float:
        return round(min(1.0, start_price + cfg.price_step * (at // cfg.price_steps)), 6) if cfg.price_steps else start_price

    price = price_at(step)
    if cfg.price_steps:
        write_text(run / "route_price", f"{price:.6f}")
    procs, stop = start_workers(run, cfg, devices, say) if workers else ([], None)
    rng = np.random.default_rng(int(time.time()))
    t_line, t_owed, acc, n_acc = time.time(), 0.0, {}, 0
    reason = ""
    while not reason:
        if (run / "STOP").exists():
            reason = "stopped"
        elif cfg.steps and step >= cfg.steps:
            reason = "steps"
        elif cfg.price_steps and price >= 1.0:
            reason = "full price"
        if reason:
            break
        data.read_new(shares(step))
        data.prune()
        if time.time() - t_owed > 2:
            t_owed = time.time()
            write_text(run / "owed", str(int(data.owed)))
        if data.window.size < 4 * BATCH or data.owed < BATCH:
            if not workers:
                reason = "positions used"
            elif any(not p.is_alive() for p in procs):
                say(event="process_died", alive=[p.is_alive() for p in procs])
                reason = "a worker died"
            else:
                time.sleep(1)
            continue
        raw, nt = data.batch(rng)
        if cfg.method == "ppo":
            raw["advantage"] = L.reward(raw["score"], raw["n_players"], raw["ticket_outcome"], *shares(step)).astype(np.float32)
        b = to_device(raw, device)
        end_rows = None
        if nt and cfg.lookahead:  # the look-ahead's lessons teach no forecast of the game's end
            end_rows = torch.ones(BATCH, device=device)
            end_rows[-nt:] = 0.0
        for g in opt.param_groups:
            g["lr"] = L.warm_lr(cfg.lr, step, fresh_at, WARMUP)
        with autocast(device, dtype):
            out = fwd({k: b[k] for k in OBS_KEYS})
        total, parts, stats = L.batch_loss(out, b, cfg.method, lessons=nt, trail_weight=cfg.trail_weight, critic=cfg.critic,
                                           hold=ctl.hold if ctl else 0.0, kl_beta=ctl.kl_beta if ctl else 0.0, end_rows=end_rows)
        if ctl is not None:
            ctl.note(step, stats)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_NORM)
        opt.step()
        step += 1
        data.owed -= BATCH - nt
        data.owed_lessons -= nt
        data.in_batches += nt
        for k, v in {"total": total, **parts}.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        for k, v in stats.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        if step % cfg.save_every == 0:
            save_net(run / "latest.pt", net, step)
            save_optimizer(run, opt)
        if step % cfg.snapshot_steps == 0:
            save_net(run / "snapshots" / f"{step:08d}.pt", net, step)
        if price_at(step) != price:
            price = price_at(step)
            write_text(run / "route_price", f"{price:.6f}")
            say(event="route_price", step=step, route_price=price)
            if price >= 1.0:  # the phase's end: the network at the step the price became whole
                end = run / "snapshots" / f"{step:08d}.pt"
                save_net(end, net, step)
                write_text(run / "full_price.json", json.dumps({"step": step, "end": str(end)}))
        if time.time() - t_line > LOG_SECONDS:
            el = time.time() - t_line
            line = {"step": step, "window": data.window.size, "new_positions": data.new, "positions_per_s": round(data.new / el, 1),
                    "steps_per_s": round(n_acc / el, 2), "owed": int(data.owed), "lr": float(f"{opt.param_groups[0]['lr']:.3g}")}
            if cfg.ticket_playouts or cfg.lookahead:
                line.update(ticket_lessons=data.new_lessons, ticket_window=data.lessons.size, ticket_owed=int(data.owed_lessons),
                            tickets_per_batch=round(data.in_batches / max(1, n_acc), 1))
            line.update(shards=len(data.seen), **{f"loss_{k}": round(v / max(1, n_acc), 4) for k, v in acc.items()},
                        val_positions=data.val.size, **validate(net, data, cfg, device, dtype), **pooled(data.selfplay))
            if cfg.price_steps:
                line["route_price"] = round(price, 4)
            if cfg.bonus:
                line["score_goal_share"], line["ticket_goal_share"] = (round(x, 4) for x in shares(step))
            if data.league:
                line["league"] = {k: pooled(v) for k, v in data.league.items()}
            if data.tables:
                line["selfplay_tables"] = {k: pooled(v) for k, v in data.tables.items()}
            say(event="train", **line)
            t_line, acc, n_acc = time.time(), {}, 0
            data.new_line()
    save_net(run / "latest.pt", net, step)
    save_optimizer(run, opt)
    end = run / "snapshots" / f"{step:08d}.pt"
    if not end.exists():
        save_net(end, net, step)
    if stop is not None:
        stop.set()
        for p in procs:
            p.join(timeout=60)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
    unlink(run / "STOP")
    say(event="end", step=step, reason=reason)
    return step


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m deepttr.train.trainer", description="Train a phase in a run directory.")
    ap.add_argument("run_dir")
    ap.add_argument("--config", help="the phase as a TrainConfig in JSON (default: RUN_DIR/config.json; kept there)")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.load(a.config or run / "config.json")
    if a.config:
        cfg.save(run / "config.json")
    train(run, cfg)


if __name__ == "__main__":
    main()
