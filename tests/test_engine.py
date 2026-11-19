"""The engine as Python sees it: games, observations, views, bots, matches."""
import hashlib
import json
import random

import numpy as np
import pytest


def play_out(game, seed: int = 0, bots=("heuristic", "hoard")) -> None:
    r = random.Random(seed)
    while not game.is_over():
        game.step(game.bot_move(bots[game.current % len(bots)], r.getrandbits(62)))


def test_the_sizes_are_the_usa_maps(engine):
    d = engine.dims()
    assert (d["n_routes"], d["n_cities"], d["n_tickets"], d["n_actions"], d["n_claim"]) == (100, 36, 30, 422, 408)
    assert (d["route_f"], d["city_f"], d["ticket_f"], d["player_f"], d["global_f"], d["event_i"]) == (36, 8, 5, 40, 58, 5)
    assert (d["max_players"], d["events"], d["card_types"], d["legal"]) == (5, 32, 9, 422)
    assert len(d["route_a"]) == len(d["route_b"]) == len(d["route_pair"]) == 100
    assert len(d["ticket_a"]) == len(d["ticket_b"]) == 30
    assert len(d["claim_route"]) == len(d["claim_color"]) == 408
    assert max(d["claim_route"]) == 99 and set(d["claim_color"]) == set(range(8))


def test_the_board_has_every_city_with_a_place(engine):
    board = json.loads(engine.map_json())
    assert len(board["cities"]) == len(board["layout"]) == 36
    assert all(set(p) == {"lat", "lon"} for p in board["layout"])
    assert len(board["routes"]) == 100 and len(board["tickets"]) == 30 and len(board["claims"]) == 408
    assert board["actions"] == {"n_claim": 408, "draw0": 408, "blind": 413, "tickets": 414, "keep1": 415, "n": 422}
    assert sum(r["length"] for r in board["routes"] if r["color"] == "red") == 27


def test_a_game_replays_from_its_seed_and_moves(engine):
    g = engine.Game(players=3, seed=42)
    play_out(g, seed=1)
    again = engine.Game(players=3, seed=42)
    for seat, action in g.moves():
        assert again.current == seat
        again.step(action)
    assert again.is_over() and again.view(0) == g.view(0)
    assert engine.Game(players=3, seed=43).view(0) != engine.Game(players=3, seed=42).view(0)


def test_a_copy_goes_its_own_way(engine):
    g = engine.Game(players=2, seed=3)
    c = g.copy()
    c.step(c.legal()[0])
    assert len(g.moves()) == 0 and len(c.moves()) == 1 and g.seed == c.seed == 3


def test_only_legal_moves_are_played(engine):
    g = engine.Game(players=2, seed=7)
    assert g.legal(), "the game opens with a choice of tickets"
    with pytest.raises(ValueError):
        g.step(0)  # a claim before the tickets are chosen
    with pytest.raises(ValueError):
        g.step(-1)  # a pass while a move is legal
    with pytest.raises(ValueError):
        g.step(100000)
    play_out(g)
    with pytest.raises(ValueError):
        g.step(0)  # the game is over
    assert g.bot_move("hoard") == -1
    with pytest.raises(ValueError):
        g.bot_move("nobody")
    with pytest.raises(ValueError):
        engine.Game(players=6)
    with pytest.raises(ValueError):
        g.observation(2)


def test_an_observation_marks_the_legal_moves_of_the_player_to_act(engine):
    g = engine.Game(players=4, seed=9)
    r = random.Random(2)
    for _ in range(80):
        if g.is_over():
            break
        legal = g.legal()
        for seat in range(4):
            o = g.observation(seat)
            assert {k: v.shape for k, v in o.items()} == {
                "routes": (1, 100, 36), "cities": (1, 36, 8), "tickets": (1, 30, 5), "players": (1, 5, 40), "global": (1, 58),
                "slots": (1, 5), "offer": (1, 3), "events": (1, 32, 5), "legal": (1, 422)}
            marked = np.flatnonzero(o["legal"][0]).tolist()
            assert marked == (legal if seat == g.current else [])
            assert o["players"][0, :4, 0].tolist() == [1, 1, 1, 1] and o["players"][0, 4, 0] == 0, "four seats are present"
        g.step(r.choice(legal) if legal else -1)


def test_a_view_shows_a_seat_its_own_hand_only(engine):
    g = engine.Game(players=2, seed=11)
    v0, v1 = json.loads(g.view(0)), json.loads(g.view(1))
    assert sum(v0["hand"]) == sum(v1["hand"]) == 4 and v0["seats"][1]["cards"] == 4
    assert "final" not in v0
    assert (v0["legal"] == []) != (v1["legal"] == []), "only the player to act has moves"
    play_out(g)
    end = json.loads(g.view(0))
    assert end["phase"] == "over" and len(end["final"]["hands"]) == 2, "at the end every hand of tickets is shown"
    assert sorted(end["final"]["ranks"]) in ([0.0, 1.0], [0.5, 0.5])


def test_rules_can_differ_from_the_official_ones(engine):
    assert engine.Game().rules == {"route_points": True, "longest_bonus": True, "partial_tickets": False, "trains": 45,
                                   "route_price": 1.0}
    g = engine.Game(players=2, seed=1, rules={"trains": 12, "route_points": False, "partial_tickets": True, "route_price": 0.5})
    assert g.rules == {"route_points": False, "longest_bonus": True, "partial_tickets": True, "trains": 12, "route_price": 0.5}
    assert [s["trains"] for s in json.loads(g.view(0))["seats"]] == [12, 12]
    play_out(g)
    final = json.loads(g.view(0))["final"]
    assert sum(final["route"]) > 0, "routes were claimed"
    for seat in range(2):  # and do not score: the total is the tickets and the longest-trail bonus
        assert final["total"][seat] - final["tickets"][seat] in (0, 10)
    for bad in ({"trains": 2}, {"route_price": 1.5}, {"blocking": True}):
        with pytest.raises(ValueError):
            engine.Game(rules=bad)


def test_the_first_choice_is_the_best_legal_logit(engine):
    logits = np.zeros(422, np.float32)
    logits[[3, 10, 400]] = [2.0, 5.0, 5.0]
    assert engine.first_choice(logits, [3, 7]) == 3
    assert engine.first_choice(logits, [3, 10]) == 10
    assert engine.first_choice(logits, [3, 10, 400]) == 400, "of equal logits the higher action id"
    with pytest.raises(ValueError):
        engine.first_choice(logits, [])
    with pytest.raises(ValueError):
        engine.first_choice(logits[:10], [1])


def test_a_match_of_bots_needs_no_network(engine):
    arena = engine.Arena("heuristic", "random", deals=10, players=3, seed=5)
    assert arena.games == 30 and not arena.done
    assert arena.prepare() == 0 and arena.done
    res = arena.results()
    assert res["a_seat"].tolist() == [0, 1, 2] * 10 and res["deal"].tolist() == [d for d in range(10) for _ in range(3)]
    assert res["utility"].shape == (30, 5) and res["route_owner"].shape == (30, 100)
    assert np.allclose(res["utility"][:, :3].sum(axis=1), 0, atol=1e-5), "a game's utilities sum to zero"
    assert (res["utility"][:, 3:] == 0).all() and (res["total"][:, 3:] == 0).all(), "seats past the table's stay 0"
    built = np.array([[(res["route_owner"][g] == q).sum() for q in range(3)] for g in range(30)])
    assert (built > 0).any() and res["route_owner"].max() <= 2 and res["route_owner"].min() == -1
    assert (res["trains"][:, :3] <= 45).all() and (res["held"][:, :3] >= 2).all()
    assert (res["completed"] <= res["held"]).all()


def test_a_match_asks_each_network_for_its_own_seats(engine, stub):
    arena = engine.Arena("net", "net", deals=3, players=2, seed=8)
    a, b = stub(0.0), stub(1.3)
    steps = 0
    while (rows := arena.prepare()):
        obs, sides = arena.observations(), arena.sides()
        assert len(sides) == rows == len(obs["legal"]) and set(sides.tolist()) <= {0, 1}
        assert (obs["legal"].sum(axis=1) > 1).all(), "a network is asked only when it has a choice"
        logits = np.empty((rows, 422), np.float32)
        for side, net in enumerate((a, b)):
            idx = np.flatnonzero(sides == side)
            if idx.size:
                logits[idx] = net({k: v[idx] for k, v in obs.items()})
        with pytest.raises(ValueError):
            arena.feed(logits[:, :10])
        arena.feed(logits)
        steps += 1
    assert arena.done and len(arena.results()["deal"]) == 6 and a.rows > 0 and b.rows > 0 and steps > 20


def test_arena_arguments_are_checked(engine):
    for kw in ({"a": "nobody"}, {"deals": 0}, {"players": 1}, {"rules": {"trains": 200}}):
        with pytest.raises(ValueError):
            engine.Arena(**{"a": "net", "b": "hoard", "deals": 2, **kw})


def digest(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def test_the_engine_plays_the_same_games_on_every_platform(engine):
    """Pinned results: the deals, the rules, the bots and the observation are deterministic, with no
    floating-point operation whose result depends on the platform. A change here means the engine
    plays or encodes differently, and trained networks no longer read it as they were trained to."""
    arena = engine.Arena("heuristic", "hoard", deals=50, players=2, seed=1)
    arena.prepare()
    res = arena.results()
    assert (int(res["total"].sum()), int(res["turns"].sum()), int(res["completed"].sum())) == GOLDEN["bots_match"]
    obs_hash = []
    for players, seed in ((2, 1), (3, 2), (4, 3), (5, 4)):
        g = engine.Game(players=players, seed=seed)
        r = random.Random(seed)
        rows = []
        while not g.is_over():
            if len(g.moves()) % 5 == 0:
                o = g.observation(g.current)
                rows += [o[k] for k in ("routes", "cities", "tickets", "players", "global", "slots", "offer", "events", "legal")]
            g.step(g.bot_move(("heuristic", "hoard")[g.current % 2], r.getrandbits(62)))
        final = json.loads(g.view(0))["final"]
        obs_hash.append((digest(rows), final["total"]))
    assert obs_hash == GOLDEN["observations"]


GOLDEN = {
    # 50 deals of the ticket builder against the card hoarder at a table of two: every point scored, turn played and ticket completed
    "bots_match": (14409, 6576, 223),
    # per table of 2 to 5: a hash of the observations along one game, and the game's final scores
    "observations": [
        ("c0d34e0885189640", [68, 97]),
        ("ae6b65f3d0b127e0", [33, 91, 100]),
        ("c04cb706302e535d", [63, 72, 2, 84]),
        ("a4ccb58d37f4f360", [18, 78, -1, 91, 32]),
    ],
}
