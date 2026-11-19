"""The measures of a phase's snapshots as they appear."""
import json

import torch


def test_the_newest_snapshot_is_measured_with_the_training_games_played_by_then(engine, tmp_path, monkeypatch):
    from deepttr import measure
    from deepttr import net as netmod

    monkeypatch.setattr(measure, "DEALS", {2: 3, 3: 2})
    monkeypatch.setattr(measure, "TICKET_DEALS", 4)
    (tmp_path / "snapshots").mkdir()
    torch.manual_seed(3)

    def snapshot(step):
        netmod.save(tmp_path / "snapshots" / f"{step:08d}.pt",
                    netmod.DeepTTRNet(engine.dims(), netmod.NetConfig(width=32, depth=1, heads=2)), step=step)

    for step in (0, 2000):
        snapshot(step)
    lines = [{"event": "train", "step": 1000, "g_games": 50, "league": {"card hoarder@2": {"g_games": 10}}},
             {"event": "train", "step": 2100, "g_games": 40}]
    (tmp_path / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    (tmp_path / "phase.json").write_text("{}", encoding="utf-8")  # the phase has ended: measure what is newest, then out
    plan = {"matches": [{"opponents": [{"name": "card hoarder", "bot": "hoard"}], "tables": [2, 3], "move": "first"}],
            "tickets": ["first"]}
    measure.watch(tmp_path, plan, device="cpu", every=0)
    matches = measure.read_lines(tmp_path / "measures.jsonl")
    tickets = measure.read_lines(tmp_path / "tickets.jsonl")
    assert [(m["step"], m["players"], m["opponent"]) for m in matches] == [(2000, 2, "card hoarder"), (2000, 3, "card hoarder")]
    assert [m["train_games"] for m in matches] == [60, 60] and matches[1]["games"] == 6
    assert [t["step"] for t in tickets] == [2000] and tickets[0]["deals"] == 4
    measure.watch(tmp_path, plan, device="cpu", every=0)
    assert len(measure.read_lines(tmp_path / "measures.jsonl")) == 2, "the older snapshot passed by is not measured later"
    snapshot(4000)
    measure.watch(tmp_path, plan, device="cpu", every=0)
    assert [m["step"] for m in measure.read_lines(tmp_path / "measures.jsonl")] == [2000, 2000, 4000, 4000]
