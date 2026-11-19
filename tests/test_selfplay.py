"""Self-play for training through the engine: the pools of the training phases, a few games each, and
what their records hold."""
import numpy as np
import pytest

POSITION = {"routes", "cities", "tickets", "players", "global", "slots", "offer", "events", "legal"}
RECORD = POSITION | {
    "policy", "value", "score", "trail", "root_forecast", "root_prob", "advantage", "route_owner", "opp_tickets",
    "ticket_outcome", "ticket_start", "trail_best", "hands", "path_cost", "n_players", "game", "moves", "lesson",
}
TYPES = {
    "policy": np.float32, "value": np.float32, "score": np.float32, "trail": np.float32, "root_forecast": np.float32,
    "root_prob": np.float32, "advantage": np.float32, "route_owner": np.int8, "opp_tickets": np.uint8,
    "ticket_outcome": np.int8, "ticket_start": np.uint8, "trail_best": np.uint8, "hands": np.uint8,
    "path_cost": np.uint8, "n_players": np.uint8, "game": np.uint64, "moves": np.uint16, "lesson": np.uint8,
    "look": np.int16, "look_table": np.float32,
}


def logits_of(obs: dict, nets: np.ndarray) -> np.ndarray:
    """Stand-in networks: logits that depend on the position and on the network that answers it."""
    rows = len(obs["legal"])
    h = (obs["global"].reshape(rows, -1).astype(np.float64).sum(axis=1)
         + obs["players"].reshape(rows, -1).astype(np.float64).sum(axis=1)) * 0.37 + 0.7 * nets
    actions = np.arange(obs["legal"].shape[1])
    return np.sin(h[:, None] * (actions % 11 + 1)[None, :] + actions[None, :]).astype(np.float32)


def play(pool, games: int, move_forecast=None, each=None) -> dict:
    """Answer every row until `games` games have ended; the records, concatenated."""
    parts = []
    for _ in range(200_000):
        rows = pool.prepare()
        obs = pool.observations()
        assert len(obs["legal"]) == rows == len(pool.row_nets()) == len(pool.decider_seats())
        if each is not None:
            each(pool, obs)
        if move_forecast is not None:
            ask = pool.forecast_rows()
            pool.feed_forecast(ask.tolist(), move_forecast(obs, ask))
        pool.feed(logits_of(obs, pool.row_nets()), np.full(rows, 0.25, np.float32))
        recs = pool.take_records()
        if recs is not None:
            parts.append(recs)
        if pool.stats()["games_finished"] >= games:
            break
    assert parts, pool.stats()
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def check_records(engine, recs: dict, looks: bool = False) -> None:
    d = engine.dims()
    assert set(recs) == RECORD | ({"look", "look_table"} if looks else set())
    n = len(recs["policy"])
    seats, tickets = d["max_players"], d["n_tickets"]
    shapes = {
        "policy": (n, d["n_actions"]), "value": (n, seats), "score": (n, seats), "trail": (n, seats),
        "trail_best": (n, seats), "route_owner": (n, d["n_routes"]), "opp_tickets": (n, seats, tickets),
        "ticket_outcome": (n, tickets), "ticket_start": (n, tickets), "hands": (n, seats, d["card_types"]),
        "path_cost": (n, seats, tickets), "look": (n, 5), "look_table": (n, 5, 32),
    }
    for key, want in shapes.items():
        if key in recs:
            assert recs[key].shape == want, key
    for key, kind in TYPES.items():
        if key in recs:
            assert recs[key].dtype == kind, key
    assert np.allclose(recs["policy"].sum(axis=1), 1.0, atol=1e-3)
    assert not (recs["policy"] > 0)[recs["legal"] == 0].any(), "mass on an illegal move"
    assert set(np.unique(recs["lesson"])) <= {0, 1}
    assert np.all((recs["root_prob"] > 0) & (recs["root_prob"] <= 1))
    players = recs["n_players"].astype(int)
    seat = np.arange(seats)[None, :]
    assert not recs["value"][seat >= players[:, None]].any(), "nothing past the table's seats"
    assert np.all(recs["route_owner"] < players[:, None])
    # a game's records agree on its end: the scores each sees, turned to the deciding seat
    for g in np.unique(recs["game"])[:20]:
        rows = recs["game"] == g
        totals = {tuple(np.sort(s[:p])) for s, p in zip(recs["score"][rows], players[rows])}
        assert len(totals) == 1, f"game {g}: {totals}"


def test_the_teachers_games_record_its_targets(engine):
    pool = engine.SelfPlay(games=8, tables=[2], rules={"route_points": False, "longest_bonus": False,
                                                       "partial_tickets": True}, seed=3, teacher=True)
    recs = play(pool, 4)
    check_records(engine, recs)
    assert not recs["lesson"].any()
    legal = recs["legal"].sum(axis=1)
    top = recs["policy"].max(axis=1)
    best = (np.abs(recs["policy"] - top[:, None]) < 1e-6).sum(axis=1)
    assert np.allclose(top, 0.1 / legal + 0.9 / best, atol=1e-4), "the best moves share 0.9, all of them 0.1"
    assert np.all(recs["root_forecast"] == np.float32(0.25))


def test_ticket_playouts_teach_a_choice_of_tickets_by_its_copies(engine):
    # short games (20 trains): every copy is a whole game to its end
    pool = engine.SelfPlay(games=4, tables=[2], rules={"route_points": False, "longest_bonus": False, "trains": 20},
                           seed=5, teacher=True, joint_paths=True, ticket_playouts=True, playout_slots=512)
    copies = []
    recs = play(pool, 2, each=lambda p, obs: copies.append(len(obs["legal"]) - p.games))
    check_records(engine, recs)
    st = pool.stats()
    assert max(copies) > 0 and st["playouts"] > 0 and st["playout_evals"] >= st["playouts"]
    assert st["ticket_lessons"] == sum(st[k] for k in ("lessons_opening", "lessons_keep", "lessons_draw",
                                                       "lessons_early"))
    assert st["lessons_opening"] > 0 and recs["lesson"].any()


def test_imitation_records_the_moves_played_at_every_table(engine):
    pool = engine.SelfPlay(games=16, tables=[2, 3, 4, 5], rules={"route_points": False, "longest_bonus": False},
                           seed=7)
    recs = play(pool, 24)
    check_records(engine, recs)
    tables = set(np.unique(recs["n_players"]).tolist())
    assert {2, 3, 4} <= tables <= {2, 3, 4, 5}
    assert np.all(recs["policy"].max(axis=1) == 1.0), "the move played is the target"
    # a game's rewards sum to zero over its seats
    for g in np.unique(recs["game"])[:10]:
        rows = recs["game"] == g
        seats = {}
        for adv, start in zip(recs["advantage"][rows], recs["ticket_start"][rows]):
            seats[start.tobytes()] = adv
        if len(seats) == recs["n_players"][rows][0]:
            assert abs(sum(seats.values())) < 1e-3


def test_the_route_price_of_new_games_follows_the_pool(engine):
    at = engine.dims()["rules_at"]
    pool = engine.SelfPlay(games=8, tables=[2, 3], rules={"route_price": 0.0}, seed=9)
    first = play(pool, 8)
    assert np.all(first["global"][:, at] == 0.0)
    pool.set_route_price(0.6)
    later = play(pool, 40)
    assert np.isclose(later["global"][:, at], 0.6).any(), "the games started later play at the new price"
    with pytest.raises(ValueError):
        pool.set_route_price(1.5)


def test_a_league_pool_plays_duplicate_deals_and_records_the_trained_seat(engine):
    for players in (2, 4):
        pool = engine.SelfPlay(games=2 * players, players=players, bot="hoard", seed=11)

        def trained(p, obs):
            assert np.all(p.decider_seats() == np.arange(len(obs["legal"])) % players), "a bot never asks"
            assert not p.row_nets().any()

        recs = play(pool, 2 * players, each=trained)
        check_records(engine, recs)
        assert np.all(recs["n_players"] == players)
        slot = recs["game"] % (2 * players)
        deal = (slot // players) * 10_000 + recs["game"] // (2 * players)
        for d in np.unique(deal):
            assert len(np.unique(recs["advantage"][deal == d])) == 1, "one reward a deal"


def test_a_frozen_network_answers_its_own_seats(engine):
    players = 3
    pool = engine.SelfPlay(games=6, players=players, frozen=True, seed=13)
    seen = set()

    def nets(p, obs):
        seats = p.decider_seats()
        frozen = seats != np.arange(len(seats)) % players
        assert np.array_equal(p.row_nets(), frozen.astype(np.uint8))
        seen.update(frozen.tolist())

    recs = play(pool, 6, each=nets)
    check_records(engine, recs)
    assert seen == {False, True}


def test_the_look_ahead_teaches_the_moves_its_check_proves(engine):
    pool = engine.SelfPlay(games=4, tables=[2, 3], rules={"trains": 20}, seed=15, lookahead=True, playout_slots=1024)
    recs = play(pool, 4)
    check_records(engine, recs)
    st = pool.stats()
    assert st["look_tried"] > 0 and st["playouts"] > 0 and st["look_checked"] <= st["look_tried"]
    assert st["ticket_lessons"] == st["look_proven"] and st["forecast_flagged"] == 0
    lessons = recs["lesson"] == 1
    assert np.all(recs["policy"][~lessons].max(axis=1) == 1.0)
    assert np.all(recs["policy"][lessons].max(axis=1) < 1.0), "a lesson is a policy"


def test_a_mixed_tables_look_ahead_with_the_move_forecast_keeps_the_copies(engine):
    na = engine.dims()["n_actions"]

    def move_forecast(obs, rows):
        # every legal move but the lowest 20 points above it
        f = np.where(obs["legal"][rows] == 1, 0.2, 0.0).astype(np.float32)
        lowest = obs["legal"][rows].argmax(axis=1)
        f[np.arange(len(rows)), lowest] = 0.0
        return f.reshape(len(rows), na)

    pool = engine.SelfPlay(games=6, players=3, mixed=[("heuristic", 0.5), ("net:0", 0.3), ("net:1", 0.2)], seed=17,
                           lookahead=True, look_check_keep=48, look_forecast=True, playout_slots=1024)
    nets = set()
    recs = play(pool, 3, move_forecast=move_forecast, each=lambda p, obs: nets.update(p.row_nets().tolist()))
    check_records(engine, recs, looks=True)
    assert nets == {0, 1, 2}, "the trained network and both frozen ones answer rows"
    st = pool.stats()
    assert st["forecast_flagged"] > 0 and st["forecast_proven"] <= st["forecast_flagged"]
    kept = recs["look"][:, 0] >= 0
    assert kept.any() and not recs["lesson"][kept].any(), "the decisions' own records keep the copies"
    assert np.isfinite(recs["look_table"][kept]).any(axis=(1, 2)).all()
    assert np.isnan(recs["look_table"][~kept]).all()


def test_a_mixed_table_of_three_seats_the_card_hoarder_itself(engine):
    pool = engine.SelfPlay(games=6, players=3, mixed=[("heuristic", 1.0), ("net:0", 1.0)], seed=19)
    recs = play(pool, 6)
    check_records(engine, recs)
    with pytest.raises(ValueError):
        engine.SelfPlay(games=6, players=3, mixed=[("hoard", 1.0)])


@pytest.mark.parametrize("words", [
    dict(games=4, teacher=True, bot="hoard"),
    dict(games=4, ticket_playouts=True),
    dict(games=4, teacher=True, ticket_playouts=True),
    dict(games=4, lookahead=True, frozen=True, playout_slots=1024),
    dict(games=4, lookahead=True, playout_slots=40),
    dict(games=4, look_forecast=True),
    dict(games=5, players=2, bot="heuristic"),
    dict(games=4, bot="nobody"),
    dict(games=4, frozen=True, bot="hoard"),
    dict(games=4, players=2, mixed=[("net:x", 1.0)]),
    dict(games=4, players=2, mixed=[("heuristic", 0.0)]),
    dict(games=4, tables=[2, 6]),
    dict(games=4, tables=[2, 3], bot="hoard"),
])
def test_settings_that_do_not_go_together_are_refused(engine, words):
    with pytest.raises(ValueError):
        engine.SelfPlay(**words)


def test_answers_of_the_wrong_shape_are_refused(engine):
    pool = engine.SelfPlay(games=2, tables=[2])
    rows = pool.prepare()
    na = engine.dims()["n_actions"]
    with pytest.raises(ValueError):
        pool.feed(np.zeros((rows + 1, na), np.float32))
    with pytest.raises(ValueError):
        pool.feed(np.zeros((rows, na), np.float32), np.zeros(rows + 1, np.float32))
    pool.feed(np.zeros((rows, na), np.float32))
    assert pool.take_records() is None and pool.games == 2
