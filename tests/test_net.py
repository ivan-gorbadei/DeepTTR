"""The network: its tokens, its sizes, its outputs. Random weights on the CPU - no checkpoint needed."""
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
    """Observations of a few positions of every table size, from the opening to the end game."""
    import random

    rows = []
    for players in (2, 3, 4, 5):
        g = engine.Game(players=players, seed=100 + players)
        r = random.Random(players)
        while not g.is_over():
            if len(g.moves()) % 17 == 0:
                rows.append(g.observation(g.current))
            g.step(g.bot_move("heuristic", r.getrandbits(62)))
    return {k: np.concatenate([o[k] for o in rows]) for k in rows[0]}


def small(dims, **kw):
    torch.manual_seed(0)
    return DeepTTRNet(dims, NetConfig(width=32, depth=2, heads=2, **kw)).eval()


def test_the_tokens_are_209(dims):
    lay = netmod.token_layout(dims)
    assert lay == {"routes": 0, "cities": 100, "tickets": 136, "players": 166, "global": 171, "slots": 172, "events": 177,
                   "total": 209}


def test_the_relations_follow_the_map(dims):
    rel = netmod.relation_matrix(dims)
    assert rel.shape == (209, 209) and (np.diag(rel) == netmod.REL_SELF).all()
    assert (rel == rel.T).all(), "every relation is mutual"
    routes = rel[:100, :100]
    assert (routes == netmod.REL_ROUTE_TWIN).sum() == 2 * 22, "22 double routes"
    assert ((rel[:100, 100:136] == netmod.REL_ROUTE_CITY).sum(axis=1) == 2).all(), "a route has two cities"
    assert ((rel[136:166, 100:136] == netmod.REL_TICKET_CITY).sum(axis=1) == 2).all(), "a ticket has two ends"
    cities = rel[100:136, 100:136]
    assert cities.max() <= netmod.REL_CITY_DIST + 8 and (cities[~np.eye(36, dtype=bool)] > netmod.REL_CITY_DIST).all()
    neighbours = (cities == netmod.REL_CITY_DIST + 1).sum() // 2
    assert neighbours == 78, "78 pairs of cities are joined by a route"
    assert (rel[166:, :] == netmod.REL_NONE).sum() == (rel[166:, :] != netmod.REL_SELF).sum(), "seats, slots and events have no place on the map"


def test_the_default_network_is_the_published_size(dims):
    assert DeepTTRNet(dims).n_params() == 7_711_871
    assert NetConfig().as_dict() == {"width": 256, "depth": 6, "heads": 4, "mlp_ratio": 4, "outcome_heads": "tokens",
                                     "trail_head": True, "move_head": True, "adv_head": True}


@pytest.mark.parametrize("config,params", [
    ({"outcome_heads": "global", "move_head": False, "adv_head": False}, 5_342_325),
    ({"outcome_heads": "tokens", "move_head": False, "adv_head": False}, 6_790_261),
    ({"outcome_heads": "tokens", "move_head": False, "adv_head": True}, 7_579_254),
])
def test_the_networks_of_the_earlier_phases_can_be_built(dims, config, params):
    assert DeepTTRNet(dims, NetConfig(**config)).n_params() == params


def test_the_outputs_have_their_shapes_and_illegal_moves_are_masked(dims, positions):
    net = small(dims)
    with torch.inference_mode():
        out = net(netmod.to_device(positions, torch.device("cpu")))
    n = len(positions["legal"])
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "logits": (n, 422), "value": (n, 5), "score": (n, 5), "belief": (n, 5, 30), "owner": (n, 100, 6), "ticket": (n, 30, 3),
        "hand": (n, 5, 9), "path": (n, 5, 30, 2), "trail": (n, 5, 2), "adv": (n,), "move": (n, 422)}
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in out.values())
    legal = torch.from_numpy(positions["legal"]) > 0
    assert (out["logits"][~legal] == -1e9).all() and (out["logits"][legal] > -1e8).all()
    assert out["value"].abs().max() <= 1 and (out["hand"] >= 0).all()
    assert (out["adv"] == 0).all() and (out["move"] == 0).all(), "the critic and the move forecast start at zero"


def test_the_optional_heads_are_optional(dims, positions):
    net = small(dims, outcome_heads="global", trail_head=False, move_head=False, adv_head=False)
    with torch.inference_mode():
        out = net(netmod.to_device(positions, torch.device("cpu")))
    assert set(out) == {"logits", "value", "score", "belief", "owner", "ticket", "hand", "path"}


def test_a_position_is_evaluated_the_same_alone_and_in_a_batch(dims, positions):
    net = small(dims)
    x = netmod.to_device(positions, torch.device("cpu"))
    with torch.inference_mode():
        whole = net(x)
        one = net({k: v[3:4] for k, v in x.items()})
    for k in whole:
        assert torch.allclose(whole[k][3:4], one[k], atol=1e-5), k


def test_a_checkpoint_round_trips(dims, positions, tmp_path):
    net = small(dims, outcome_heads="global", move_head=False)
    netmod.save(tmp_path / "net.pt", net, step=123)
    again, meta = netmod.load(tmp_path / "net.pt")
    assert meta == {"step": 123} and again.cfg == net.cfg and not again.training
    x = netmod.to_device(positions, torch.device("cpu"))
    with torch.inference_mode():
        a, b = net(x), again(x)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_a_checkpoint_names_the_heads_it_has(dims, tmp_path):
    net = small(dims, outcome_heads="global", trail_head=False, move_head=False, adv_head=False)
    blob = {"config": {"width": 32, "depth": 2, "heads": 2, "mlp_ratio": 4}, "dims": net.dims, "state": net.state_dict()}
    torch.save(blob, tmp_path / "old.pt")
    again, _ = netmod.load(tmp_path / "old.pt")
    assert again.cfg == net.cfg, "a head the checkpoint does not name is absent"
    torch.save({**blob, "config": {**blob["config"], "some_new_input": True}}, tmp_path / "new.pt")
    with pytest.raises(ValueError):
        netmod.load(tmp_path / "new.pt")
    torch.save({**blob, "config": {**blob["config"], "some_new_input": False}}, tmp_path / "off.pt")
    netmod.load(tmp_path / "off.pt")
