"""Growing a network's heads at a phase's start (`net.grow`): the network keeps every output it had,
a new critic and move forecast start at zero, and the lineage's sizes come out as published."""
import numpy as np
import pytest
import torch

from deepttr import net as netmod
from deepttr.net import DeepTTRNet, NetConfig


@pytest.fixture(scope="module")
def dims(engine):
    return engine.dims()


@pytest.fixture(scope="module")
def positions(engine):
    """Observations of a game of three by the heuristic bot, every eleventh position."""
    import random

    g, r, rows = engine.Game(players=3, seed=5), random.Random(5), []
    while not g.is_over():
        if len(g.moves()) % 11 == 0:
            rows.append(g.observation(g.current))
        g.step(g.bot_move("heuristic", r.getrandbits(62)))
    return {k: np.concatenate([o[k] for o in rows]) for k in rows[0]}


def small(dims, **kw):
    torch.manual_seed(0)
    return DeepTTRNet(dims, NetConfig(width=32, depth=2, heads=2, **kw)).eval()


def run(net, positions):
    with torch.inference_mode():
        return net(netmod.to_device(positions, torch.device("cpu")))


def test_a_grown_network_keeps_its_outputs(dims, positions):
    net = small(dims, outcome_heads="global", trail_head=False, adv_head=False, move_head=False)
    before = run(net, positions)
    grown = netmod.grow(net, trail_head=True, adv_head=True, move_head=True)
    assert grown.cfg == NetConfig(width=32, depth=2, heads=2, outcome_heads="global") and not grown.training
    after = run(grown, positions)
    for k in before:
        assert torch.equal(before[k], after[k]), k
    assert (after["adv"] == 0).all() and (after["move"] == 0).all(), "the critic and the move forecast start at zero"
    assert "trail" in after


def test_new_outcome_heads_replace_the_old_and_nothing_else(dims, positions):
    net = small(dims, outcome_heads="global", adv_head=True, move_head=False)
    before = run(net, positions)
    grown = netmod.grow(net, outcome_heads="tokens")
    assert grown.cfg.outcome_heads == "tokens" and isinstance(grown.value_head, netmod.OutcomeHead)
    after = run(grown, positions)
    for k in ("logits", "belief", "owner", "ticket", "hand", "path", "trail"):
        assert torch.equal(before[k], after[k]), k
    assert (after["adv"] == 0).all(), "a critic of the new kind starts afresh"


def test_growing_what_a_network_has_changes_nothing(dims, snapshot):
    net = small(dims)
    assert netmod.grow(net, trail_head=True, adv_head=True, move_head=True, outcome_heads="tokens") is net
    assert netmod.grow(net, adv_head=False) is net, "a head is never taken away"
    published, _ = netmod.load(snapshot)
    assert netmod.grow(published, trail_head=True, adv_head=True, move_head=True) is published
    with pytest.raises(ValueError):
        netmod.grow(net, see_all=True)


def test_the_lineage_grows_to_the_published_size(dims):
    torch.manual_seed(1)
    net = DeepTTRNet(dims, NetConfig(outcome_heads="global", adv_head=False, move_head=False))
    sizes = [net.n_params()]
    for heads in ({"outcome_heads": "tokens"}, {"adv_head": True}, {"move_head": True}):
        net = netmod.grow(net, **heads)
        sizes.append(net.n_params())
    assert sizes == [5_342_325, 6_790_261, 7_579_254, 7_711_871]
    assert net.cfg == NetConfig()
