"""The duplicate match and the evaluator."""
import numpy as np
import pytest
import torch

from deepttr.device import autocast_dtype, pick_device


def test_the_device_is_picked_by_itself_and_can_be_named(monkeypatch):
    monkeypatch.delenv("DEEPTTR_DEVICE", raising=False)
    auto = pick_device()
    assert auto.type in ("cuda", "mps", "cpu")
    if not torch.cuda.is_available():
        assert auto.type != "cuda"
    assert pick_device("cpu") == torch.device("cpu") == pick_device(torch.device("cpu"))
    monkeypatch.setenv("DEEPTTR_DEVICE", "cpu")
    assert pick_device() == torch.device("cpu") and pick_device("auto") == auto
    assert autocast_dtype(torch.device("cpu")) is None and autocast_dtype(torch.device("mps")) is None


def test_win_rate_counts_decided_games_for_two_and_first_places_for_more():
    from deepttr.play import win_rate

    assert win_rate(30, 10, 50, 2) == 0.75
    assert win_rate(30, 10, 50, 4) == 0.6
    assert win_rate(0, 0, 0, 2) == 0.0


def test_summary_averages_a_deals_seatings(engine):
    from deepttr.play import summary

    # two deals at a table of 3: A first in every seating of the first deal, last in every one of the second
    a_seat = np.array([0, 1, 2, 0, 1, 2], np.uint8)
    utility = np.zeros((6, 5), np.float32)
    total = np.zeros((6, 5), np.int32)
    for g in range(6):
        utility[g, :3] = [0.0, 0.0, 0.0]
        utility[g, a_seat[g]] = 1.0 if g < 3 else -1.0
        total[g, :3] = 50
        total[g, a_seat[g]] = 120 if g < 3 else 20
    m = summary({"a_seat": a_seat, "utility": utility, "total": total}, 3)
    assert (m["deals"], m["games"], m["a_wins"], m["a_losses"], m["a_win_rate"]) == (2, 6, 3, 3, 0.5)
    assert m["a_mean"] == 0.0 and m["a_score"] == 70.0 and m["b_score"] == 50.0
    assert m["ci95"][0] < 0 < m["ci95"][1]


@pytest.mark.parametrize("players", [2, 3, 5])
def test_identical_players_draw_exactly(engine, stub, players):
    from deepttr.play import duplicate_match

    m = duplicate_match(stub(0.4), stub(0.4), deals=3, players=players, seed=11)
    assert (m["deals"], m["games"], m["players"]) == (3, 3 * players, players)
    assert m["a_mean"] == 0.0, m
    assert m["a_score"] == m["b_score"]


def test_a_match_is_the_same_whatever_the_threads_and_continues_by_first_deal(engine, stub):
    from deepttr.play import play_match

    whole = play_match(stub(0.1), stub(2.0), deals=4, players=2, seed=5, threads=1)
    again = play_match(stub(0.1), stub(2.0), deals=4, players=2, seed=5, threads=3)
    assert all((whole[k] == again[k]).all() for k in whole)
    tail = play_match(stub(0.1), stub(2.0), deals=2, players=2, seed=5, first_deal=2)
    assert all((whole[k][4:] == tail[k]).all() for k in whole)
    other = play_match(stub(0.1), stub(2.0), deals=4, players=2, seed=6)
    assert (whole["total"] != other["total"]).any(), "another seed, other deals"


def test_a_bot_plays_without_being_asked(engine, stub):
    from deepttr.play import Bot, duplicate_match

    net = stub(0.0)
    m = duplicate_match(net, Bot("hoard"), deals=2, players=4, seed=3)
    assert m["games"] == 8 and net.rows > 0
    m = duplicate_match(Bot("heuristic"), Bot("random"), deals=30, players=2, seed=3)
    assert m["a_win_rate"] > 0.9 and m["a_mean"] > 0.8 and m["a_score"] > m["b_score"]
    with pytest.raises(ValueError):
        Bot("nobody")


def test_an_evaluator_runs_a_network_for_a_match(engine):
    from deepttr.net import DeepTTRNet, NetConfig
    from deepttr.play import Bot, Evaluator, duplicate_match

    torch.manual_seed(1)
    ev = Evaluator(DeepTTRNet(engine.dims(), NetConfig(width=32, depth=1, heads=2)), "cpu")
    assert ev.device.type == "cpu" and ev.dtype is None
    g = engine.Game(players=2, seed=2)
    logits = ev(g.observation(g.current))
    assert logits.shape == (1, 422) and logits.dtype == np.float32
    assert ev.move(g) in g.legal() and ev.move(g) == engine.first_choice(logits[0], g.legal())
    assert set(ev.outputs(g.observation(g.current))) >= {"logits", "value", "score", "belief"}
    # a batch larger than the evaluator's limit is evaluated in parts, to the same numbers
    batch = {k: np.concatenate([v] * 5) for k, v in g.observation(g.current).items()}
    small = Evaluator(ev.net, "cpu", max_batch=2)
    assert np.allclose(small(batch), np.concatenate([logits] * 5), atol=1e-5)
    # short games (12 trains): an untrained network still plays legal games to the end
    m = duplicate_match(ev, Bot("random"), deals=2, players=3, seed=1, rules={"trains": 12})
    assert m["games"] == 6 and -1 <= m["a_mean"] <= 1
