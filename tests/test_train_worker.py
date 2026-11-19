"""A self-play worker: the pools it lays out for a phase; one round of its loop with stand-ins for
the engine's pools and for the networks - whose rows each network evaluates, what goes back to the
pools, the shards and the `selfplay` line; and its loop with the engine's pools, whose shards the
trainer then trains on."""
import json
import shutil

import numpy as np
import pytest
import torch

from deepttr import net as netmod
from deepttr.net import DeepTTRNet, NetConfig
from deepttr.train import replay as R
from deepttr.train import trainer as T
from deepttr.train import worker as W
from deepttr.train.config import Opponent, TrainConfig
from test_train_replay import synthetic

HOARD, BUILDER = "card hoarder", "ticket builder"


def shares(*values):
    return dict(zip((2, 3, 4, 5), values))


def sizes(cfg):
    pools, _ = W.plan(cfg)
    return {p.key: p.games for p in pools}


def test_the_pools_of_the_first_league_phase():
    league = (Opponent(HOARD, "hoard", shares(0.0151, 0.0148, 0.0141, 0.0146)),
              Opponent(BUILDER, "heuristic", shares(0.0161, 0.0154, 0.0152, 0.0146)),
              Opponent("phase 4", "nets/four.pt", shares(0.0628, 0.1384, 0.2530, 0.4161)))
    got = sizes(TrainConfig(league=league))
    want = {"self": 10, **{f"{HOARD}@{n}": g for n, g in zip((2, 3, 4, 5), (16, 15, 16, 15))},
            **{f"{BUILDER}@{n}": g for n, g in zip((2, 3, 4, 5), (16, 15, 16, 15))},
            **{f"phase 4@{n}": g for n, g in zip((2, 3, 4, 5), (64, 141, 260, 425))}}
    assert got == want


def test_the_pools_of_the_next_two_phases():
    nets = shares(0.0358, 0.0796, 0.1415, 0.2219)
    six = (Opponent(HOARD, "hoard", shares(0.0083, 0.0080, 0.0084, 0.0080)), Opponent("phase 5", "five.pt", nets), Opponent("phase 4", "four.pt", nets))
    got = sizes(TrainConfig(league=six))
    assert got["self"] == 17 and [got[f"{HOARD}@{n}"] for n in (2, 3, 4, 5)] == [8, 9, 8, 10]
    assert [got[f"phase 5@{n}"] for n in (2, 3, 4, 5)] == [36, 81, 144, 225] == [got[f"phase 4@{n}"] for n in (2, 3, 4, 5)]
    each = shares(0.0473, 0.0707, 0.0943, 0.1180)
    seven = tuple(Opponent(f"phase {k}", f"{k}.pt", each) for k in (6, 5, 4))
    got = sizes(TrainConfig(league=seven))
    assert got["self"] == 16 and all([got[f"phase {k}@{n}"] for n in (2, 3, 4, 5)] == [48, 72, 96, 120] for k in (6, 5, 4))


def test_the_pools_of_the_championship():
    bots = shares(0.005, 0.005, 0.005, 0.005)
    nets = {"older a": (0.0082, 0.0221, 0.0284, 0.0384), "older b": (0.0092, 0.0161, 0.0249, 0.0264),
            "older c": (0.0103, 0.0172, 0.0223, 0.0350), "older d": (0.0220, 0.0195, 0.0243, 0.0429),
            "phase 5": (0.0153, 0.0150, 0.0320, 0.0310), "phase 6": (0.0212, 0.0234, 0.0305, 0.0384),
            "phase 7": (0.0220, 0.0337, 0.0343, 0.0411), "side a": (0.0115, 0.0298, 0.0426, 0.0496),
            "side b": (0.0121, 0.0254, 0.0335, 0.0411)}
    league = (Opponent(BUILDER, "heuristic", bots), Opponent(HOARD, "hoard", bots),
              *(Opponent(name, f"{name}.pt", shares(*s)) for name, s in nets.items()))
    got = sizes(TrainConfig(league=league))
    assert len(got) == 45 and got["self"] == 13
    at2 = [got[f"{name}@2"] for name in (BUILDER, HOARD, *nets)]
    assert at2 == [6, 6, 8, 10, 10, 22, 16, 22, 22, 12, 12]


def test_mixed_tables_and_the_look_ahead_against_the_league():
    league = (Opponent(BUILDER, "heuristic", shares(0.005, 0.005, 0.005, 0.005)), Opponent(HOARD, "hoard", shares(0.005, 0.005, 0.005, 0.005)),
              Opponent("phase 9", "nine.pt", shares(0.06, 0.1, 0.13, 0.17)), Opponent("phase 8", "eight.pt", shares(0.07, 0.1, 0.14, 0.17)))
    cfg = TrainConfig(league=league, mixed_tables=True, lookahead=True, look_league=True, move_forecast=True)
    pools, nets = W.plan(cfg)
    by = {p.key: p for p in pools}
    assert nets == ["nine.pt", "eight.pt"]
    assert {k for k in by if k.endswith("@2") and not k.startswith("look")} == {f"{BUILDER}@2", f"{HOARD}@2", "phase 9@2", "phase 8@2"}
    for n, total in ((3, 0.21), (4, 0.28), (5, 0.35)):
        mixed = by[f"mixed@{n}"]
        assert mixed.games == max(n, int(round(1024 * total / n)) * n) and mixed.prefix == f"L-mixed-n{n}"
        assert [d for d, _ in mixed.draws] == ["heuristic", "net:0", "net:1"], "the engine seats the card hoarder: it is not drawn"
        assert [nets[k] for k in mixed.nets] == ["nine.pt", "eight.pt"]
    looks = [p for p in pools if p.look]
    assert [(p.key, p.games, p.slots) for p in looks] == [(f"look@{n}", 64, 2048) for n in (2, 3, 4, 5)]
    assert looks[0].prefix == "kL-mixed-n2" and [d for d, _ in looks[0].draws] == ["heuristic", "hoard", "net:0", "net:1"]
    assert by["self"].games == 1024 - sum(p.games for p in pools if p.players and not p.look)


def test_the_teachers_and_the_look_aheads_own_pools():
    assert [(p.key, p.games, p.slots) for p in W.plan(TrainConfig(method="teacher", ticket_playouts=True, games=512))[0]] == [("self", 512, 2048)]
    assert [(p.key, p.games, p.look) for p in W.plan(TrainConfig(lookahead=True))[0]] == [("self", 1024, False), ("look", 256, True)]


class Stub:
    """A network's stand-in: every output of a row is `tag` (the critic's ten times it)."""

    def __init__(self, tag):
        self.tag, self.rows = tag, 0

    def outputs(self, obs):
        n = len(obs["legal"])
        self.rows += n
        return {"logits": torch.full((n, 422), self.tag), "value": torch.zeros(n, 5), "belief": torch.full((n, 5, 30), self.tag),
                "adv": torch.full((n,), 10 * self.tag), "move": torch.full((n, 422), self.tag)}


class FakePool:
    """A stand-in for an engine pool: fixed observations, who decides at each row, and a game's
    records every round."""

    def __init__(self, rows, seed, nets=None, ask=()):
        self.rng = np.random.default_rng(seed)
        recs = synthetic(self.rng, games=1, turns=rows)
        self.obs = {k: recs[k][:rows] for k in ("routes", "cities", "tickets", "players", "global", "slots", "offer", "events", "legal")}
        self.nets = np.zeros(rows, np.uint8) if nets is None else nets
        self.ask = list(ask)
        self.fed, self.forecasts, self.price, self.rounds = [], [], None, 0

    def prepare(self):
        self.rounds += 1

    def observations(self):
        return self.obs

    def row_nets(self):
        return self.nets

    def feed(self, logits, forecast=None):
        self.fed.append((logits.copy(), None if forecast is None else forecast.copy()))

    def take_records(self):
        return synthetic(self.rng, games=1, tables=(2,), turns=6, game0=self.rounds)

    def forecast_rows(self):
        return self.ask

    def feed_forecast(self, rows, move):
        self.forecasts.append((rows, move))

    def set_route_price(self, p):
        self.price = p

    def stats(self):
        return {"games_finished": self.rounds, "moves_per_game": 50.0, "look_tried": 3, "look_proven": 1}


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SHARD", 20)
    monkeypatch.setattr(R, "LOOK_SHARD", 5)
    (tmp_path / "replay").mkdir()
    specs = [W.Pool("self", 6), W.Pool("phase 5", 4, 2, nets=(0,)),
             W.Pool("mixed", 3, 3, nets=(1, 0), draws=(("heuristic", 0.1), ("net:0", 0.1), ("net:1", 0.1))),
             W.Pool("mixed", 2, 2, nets=(0,), draws=(("net:0", 0.1),), look=True), W.Pool(HOARD, 2, 2, bot="hoard")]
    pools = [FakePool(6, 1), FakePool(4, 2, nets=np.array([0, 1, 0, 0])), FakePool(3, 3, nets=np.array([0, 2, 1])),
             FakePool(2, 4, nets=np.array([1, 0]), ask=[0, 1]), FakePool(2, 5)]
    lines = []
    cfg = TrainConfig(lookahead=True, move_forecast=True, hand_penalty=True)
    w = W.Worker(tmp_path, cfg, 0, list(zip(specs, pools)), Stub(1.0), [Stub(2.0), Stub(3.0)], lambda **e: lines.append(e))
    return w, pools, lines


def test_each_network_evaluates_its_own_rows(worker):
    w, pools, _ = worker
    assert w.round() == 6 + 3 + 1 + 1 + 2
    lg = [p.fed[-1][0][:, 0].tolist() for p in pools]
    assert lg == [[1.0] * 6, [1.0, 2.0, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 1.0], [1.0, 1.0]]
    assert (pools[0].fed[-1][1] == 10.0).all(), "the learner's critic goes with its moves, for the records"
    fc = pools[1].fed[-1][1]
    assert np.isnan(fc[1]) and (fc[[0, 2, 3]] == 10.0).all(), "a frozen network's rows get none"
    assert w.frozen[0].rows == 3 and w.frozen[1].rows == 1 and w.learner.rows == 13, "one call a network a round"
    rows, move = pools[3].forecasts[-1]
    assert rows == [1] and move.shape == (1, 422) and (move == 1.0).all(), "the move head's forecast at the learner's decisions"


def test_finished_games_go_to_shards_with_their_hand_penalty(worker):
    w, pools, lines = worker
    for _ in range(5):
        w.round()
    for s in w.sinks:
        s.close()
    names = sorted(p.name for p in (w.run / "replay").glob("*.npz"))
    for prefix in ("self", "kL-mixed-n2", "L-card-hoarder-n2", "L-phase-5-n2", "L-mixed-n3"):
        assert R.shard_name(prefix, 0, 0) in names, prefix
    shard = R.read_shard(w.run / "replay" / R.shard_name("self", 0, 0))
    assert "hand_penalty" in shard and shard["hand_penalty"].dtype == np.float32 and (shard["hand_penalty"] <= 0).all()
    w.line(evals=130, seconds=10.0, waited=2.0)
    line = lines[-1]
    assert line["event"] == "selfplay" and line["evals_per_s"] == 13 and line["games_finished"] == 5 and line["waited_s"] == 2
    assert line["look_games"] == 5 and line["look_tried"] == 3 and line["look_proven"] == 1, "the look-ahead pools' counters"
    assert {"hand_turns", "hand_points"} <= set(line) and line["league_games"] == {"phase 5@2": 5, "mixed@3": 5, f"{HOARD}@2": 5}
    w.line(evals=0, seconds=10.0, waited=0.0)
    assert lines[-1]["hand_turns"] == 0, "counted since the last line"


class StopAfter:
    """A stop event that is set once the run directory holds `shards` shards."""

    def __init__(self, replay, shards):
        self.replay, self.shards, self.calls = replay, shards, 0

    def is_set(self):
        self.calls += 1
        return len(list(self.replay.glob("*.npz"))) >= self.shards or self.calls > 20_000


def test_the_engines_games_feed_the_trainer(engine, tmp_path, monkeypatch):
    """Every kind of pool of the last phase - self-play, a bot, a frozen network, mixed tables, the
    look-ahead against the league with the move forecast - played by the engine with small networks
    on the CPU; then the trainer takes their shards."""
    for module, name, value in ((R, "SHARD", 400), (R, "LOOK_SHARD", 100), (W, "RELOAD_SECONDS", 0), (W, "LINE_SECONDS", 0),
                                (W, "LOOK_GAMES", 8), (W, "LOOK_SLOTS", 4 * 2 * 24), (T, "BATCH", 64), (T, "LOG_SECONDS", 0)):
        monkeypatch.setattr(module, name, value)
    torch.manual_seed(0)
    play = tmp_path / "play"
    (play / "replay").mkdir(parents=True)
    small = NetConfig(width=16, depth=1, heads=2)
    netmod.save(play / "latest.pt", DeepTTRNet(engine.dims(), small), step=0)
    netmod.save(tmp_path / "frozen.pt", DeepTTRNet(engine.dims(), small), step=0)
    league = (Opponent(BUILDER, "heuristic", shares(0.01, 0.01, 0.01, 0.01)), Opponent(HOARD, "hoard", shares(0.05, 0.01, 0.01, 0.01)),
              Opponent("phase 10", str(tmp_path / "frozen.pt"), shares(0.1, 0.15, 0.2, 0.25)))
    cfg = TrainConfig(steps=3, games=64, league=league, mixed_tables=True, lookahead=True, look_league=True, move_forecast=True,
                      hand_penalty=True, critic=True, td=True, bonus=True, ratio=1.0, window=50_000, devices=("cpu",), threads=2)
    W.run(str(play), cfg.check(), 0, "cpu", stop=StopAfter(play / "replay", 5))
    log = [json.loads(x) for x in (play / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    pools = log[0]["pools"]
    assert log[0]["event"] == "league" and set(pools) == {"self", f"{BUILDER}@2", f"{HOARD}@2", "phase 10@2", "mixed@3", "mixed@4",
                                                          "mixed@5", "look@2", "look@3", "look@4", "look@5"}
    line = [e for e in log if e["event"] == "selfplay"][-1]
    assert {"hand_turns", "hand_points", "look_games", "look_tried", "forecast_flagged"} <= set(line)
    shards = sorted((play / "replay").glob("*.npz"))
    kinds = {R.shard_pool(x.name)["prefix"] for x in shards}
    assert any(k.startswith("L-") for k in kinds), kinds
    recs = R.read_shard(shards[0])
    assert set(R.STORE) <= set(recs) and "hand_penalty" in recs and np.isfinite(recs["root_forecast"]).all()
    run = tmp_path / "train"
    shutil.copytree(play / "replay", run / "replay")
    cfg.init = str(play / "latest.pt")
    assert T.train(run, cfg, workers=False) == 3
    train_lines = [json.loads(x) for x in (run / "log.jsonl").read_text(encoding="utf-8").splitlines() if '"train"' in x]
    assert train_lines and "loss_critic" in train_lines[-1]
