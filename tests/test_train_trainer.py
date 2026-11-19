"""The trainer on the CPU: a few steps of each way of learning on synthetic shards, with a small
network and small batches - the run directory it leaves, its log, a resume, a stop by file, the
price ramp that ends a phase, and the start and stop of its worker processes."""
import json

import numpy as np
import pytest
import torch

from deepttr import net as netmod
from deepttr.net import DeepTTRNet, NetConfig
from deepttr.train import losses as L
from deepttr.train import replay as R
from deepttr.train import trainer as T
from deepttr.train.config import Opponent, TrainConfig
from test_train_replay import synthetic


@pytest.fixture(autouse=True)
def small_batches(monkeypatch):
    monkeypatch.setattr(T, "BATCH", 64)
    monkeypatch.setattr(T, "LOG_SECONDS", 0)  # a training line every step
    monkeypatch.setattr(T, "ADAPT_STEPS", 2)
    monkeypatch.setattr(T, "LESSON_MIN", 1)


def init_net(engine, path, **heads):
    torch.manual_seed(0)
    cfg = {"outcome_heads": "global", "adv_head": False, "move_head": False, **heads}
    netmod.save(path, DeepTTRNet(engine.dims(), NetConfig(width=16, depth=1, heads=2, **cfg)), step=77)
    return str(path)


def write_shards(run, names, seed=0, **kw):
    rng = np.random.default_rng(seed)
    (run / "replay").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        recs = synthetic(rng, games=8, game0=100 * i, **kw)
        recs["hand_penalty"] = L.hand_penalty(recs)[0]
        R.write_shard([recs], run / "replay" / name)


def events(run):
    return [json.loads(s) for s in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_the_teacher_trains_and_leaves_its_run(engine, tmp_path):
    run = tmp_path / "run"
    write_shards(run, [R.shard_name("self", 0, s) for s in range(4)], lesson_share=0.1, tables=(2,))
    cfg = TrainConfig(init=init_net(engine, tmp_path / "init.pt"), steps=3, method="teacher", ticket_playouts=True, tables=(2,),
                      outcome_heads="global", rules={"route_points": False, "longest_bonus": False}, lr=2e-4, ratio=1.0,
                      window=10_000, trail_weight=0.0, save_every=2, snapshot_steps=2, devices=("cpu",))
    assert T.train(run, cfg, workers=False) == 3
    net, meta = netmod.load(run / "latest.pt")
    assert meta == {"step": 3} and net.cfg.outcome_heads == "global" and not net.cfg.adv_head
    assert sorted(p.name for p in (run / "snapshots").glob("*.pt")) == ["00000000.pt", "00000002.pt", "00000003.pt"]
    assert (run / "opt.pt").exists() and (run / "owed").exists() and not (run / "route_price").exists()
    assert sorted(p.name for p in (run / "ticket_lessons").glob("*.npz")) == [R.shard_name("self", 0, s) for s in range(4)]
    log = events(run)
    assert [e["event"] for e in log if e["event"] != "train"] == ["start", "end"]
    assert log[0]["args"]["method"] == "teacher" and log[0]["loss_weights"]["trail"] == 0.0
    trains = [e for e in log if e["event"] == "train"]
    assert [e["step"] for e in trains] == [1, 2, 3] and all(e["by"] == "trainer" for e in log)
    for k in ("loss_total", "loss_policy", "loss_belief", "loss_owner", "loss_ticket", "loss_hand", "loss_path", "loss_trail",
              "val_total", "val_agree", "tickets_per_batch", "lr"):
        assert k in trains[-1], k
    first = trains[0]  # the games' statistics come with the line after their shards were read
    assert first["g_games"] == 24 and first["ticket_lessons"] > 0 and "g_player_ticket_points" in first
    assert first["tickets_per_batch"] > 0 and "league" not in first and "selfplay_tables" not in first
    assert log[-1] == {**log[-1], "event": "end", "step": 3, "reason": "steps"}


def test_a_run_resumes_and_stops_by_file(engine, tmp_path):
    run = tmp_path / "run"
    write_shards(run, [R.shard_name("self", 0, s) for s in range(1, 5)], tables=(2, 3))
    cfg = TrainConfig(init=init_net(engine, tmp_path / "init.pt"), steps=2, method="imitate", outcome_heads="global", ratio=1.0,
                      window=10_000, devices=("cpu",))
    assert T.train(run, cfg, workers=False) == 2
    assert "selfplay_tables" in [e for e in events(run) if e["event"] == "train"][0], "a mix of tables: each table's games apart"
    before, _ = netmod.load(run / "latest.pt")
    cfg.steps = 4
    # a restart refills the window from the shards on disk; they were trained on, so they owe
    # nothing, and without workers there is nothing new to train on
    assert T.train(run, cfg, workers=False) == 2
    log = events(run)
    resume = [e for e in log if e["event"] == "resume"]
    assert len(resume) == 1 and resume[0]["step"] == 2 and resume[0]["args"]["steps"] == 4
    reloaded = [e for e in log if e["event"] == "window_reloaded"]
    assert len(reloaded) == 1 and reloaded[0]["shards"] == 4 and reloaded[0]["positions"] > 0
    assert not [e for e in log if e["event"] in ("optimizer_reset", "optimizer_fresh")], "the optimizer goes on where it was"
    assert log[-1]["reason"] == "positions used"
    after, meta = netmod.load(run / "latest.pt")
    assert meta["step"] == 2 and all(torch.equal(a, b) for a, b in zip(before.state_dict().values(), after.state_dict().values()))
    (run / "STOP").write_text("")
    cfg.steps = 0
    assert T.train(run, cfg, workers=False) == 2
    assert events(run)[-1]["reason"] == "stopped" and not (run / "STOP").exists()


def test_the_price_ramp_ends_the_phase_at_the_whole_price(engine, tmp_path):
    run = tmp_path / "run"
    write_shards(run, [R.shard_name("self", 0, s) for s in range(1, 9)])
    cfg = TrainConfig(init=init_net(engine, tmp_path / "init.pt", outcome_heads="tokens"), method="imitate", rules={"route_price": 0.0},
                      price_step=0.5, price_steps=2, ratio=1.0, window=10_000, devices=("cpu",))
    assert T.train(run, cfg, workers=False) == 4
    assert json.loads((run / "full_price.json").read_text()) == {"step": 4, "end": str((run / "snapshots" / "00000004.pt").resolve())}
    assert (run / "route_price").read_text() == "1.000000"
    log = events(run)
    assert [(e["step"], e["route_price"]) for e in log if e["event"] == "route_price"] == [(2, 0.5), (4, 1.0)]
    assert [e["route_price"] for e in log if e["event"] == "train"] == [0.0, 0.5, 0.5, 1.0]
    assert log[-1]["reason"] == "full price"


def test_ppo_trains_with_every_mechanism_of_the_later_phases(engine, tmp_path):
    run = tmp_path / "run"
    hoarder = [R.pool_prefix("card hoarder", n) for n in (2, 2, 3)]
    names = [R.shard_name("self", 0, 0)] + [R.shard_name(p, 0, s) for p, s in zip(hoarder, (1, 2, 1))]
    names += [R.shard_name(R.pool_prefix("self", look=True), 0, s) for s in range(1, 4)]
    write_shards(run, names, lesson_share=0.1, looks=True)
    league = (Opponent("card hoarder", "hoard", {2: 0.1, 3: 0.1}),)
    cfg = TrainConfig(init=init_net(engine, tmp_path / "init.pt"), steps=4, league=league, critic=True, td=True, bonus=True, bonus_ramp=True,
                      lookahead=True, move_forecast=True, hand_penalty=True, window=10_000, devices=("cpu",))
    assert T.train(run, cfg, workers=False) == 4
    net, _ = netmod.load(run / "latest.pt")
    assert net.cfg == NetConfig(width=16, depth=1, heads=2), "the phase's heads grown at its start"
    log = events(run)
    assert [e["event"] for e in log if e["event"] in ("outcome_heads", "adv_head", "move_head")] == ["outcome_heads", "adv_head", "move_head"]
    assert [e["target"] is not None for e in log if e["event"] == "entropy_hold"] == [True]
    controls = json.loads((run / "ppo_controls.json").read_text())
    assert controls["step"] == 3 and controls["kl_beta"] > 0
    trains = [e for e in log if e["event"] == "train"]
    line = trains[-1]
    for k in ("loss_critic", "loss_critic_explained", "loss_ppo_kl", "loss_entropy", "loss_td_rows", "loss_move", "loss_move_pairs",
              "loss_ppo_beta", "loss_entropy_hold", "score_goal_share", "ticket_goal_share"):
        assert k in line, k
    assert line["loss_td_rows"] == 1.0 and line["score_goal_share"] == 0.0004 and line["ticket_goal_share"] == 0.0
    league = trains[0]["league"]
    assert set(league) == {"card hoarder@2", "card hoarder@3"} and {"g_won", "g_games"} <= set(league["card hoarder@2"])


def test_the_trainer_starts_and_ends_its_workers(engine, tmp_path):
    """The worker processes of a run start and end with the trainer (here at once, by a file STOP
    already there) - whether or not the engine they need can make their games."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "STOP").write_text("")
    cfg = TrainConfig(init=init_net(engine, tmp_path / "init.pt", outcome_heads="tokens"), method="imitate", devices=("cpu",))
    cfg.save(run / "config.json")
    T.main([str(run)])
    log = events(run)
    procs = [e for e in log if e["event"] == "processes"]
    assert len(procs) == 1 and procs[0]["devices"] == ["cpu"] and len(procs[0]["workers"]) == 1
    assert log[-1]["event"] == "end" and log[-1]["reason"] == "stopped" and not (run / "STOP").exists()
