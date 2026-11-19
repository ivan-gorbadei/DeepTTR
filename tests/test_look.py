"""The look-ahead: the engine's copies of a decision, the rule that compares them, and the player
that moves by them."""
import random
import time

import numpy as np
import pytest


def midgame(engine, players: int = 2, seed: int = 3, moves: int = 30, rules: dict | None = None):
    g = engine.Game(players=players, seed=seed, rules=rules)
    r = random.Random(seed)
    for _ in range(moves):
        legal = g.legal()
        if g.is_over() or len(legal) < 1:
            break
        g.step(r.choice(legal))
    while not g.is_over() and len(g.legal()) < 2:  # on to a decision with a choice
        legal = g.legal()
        g.step(legal[0] if legal else -1)
    return g


def run(look, net) -> np.ndarray:
    while look.prepare():
        look.feed(net(look.observations()))
    return look.table()


def test_the_options_start_with_the_first_choice_and_cover_the_kinds_of_move(engine):
    logits = np.full(422, -9, np.float32)
    claims, face, blind, tickets = [3, 7], 409, 413, 414
    logits[[3, 7, face, blind, tickets]] = [1, 5, 2, 3, 0.5]
    assert engine.look_options(logits, [3, 7, face, blind, tickets]) == [7, blind, tickets, face]
    assert engine.look_options(logits, claims) == [7, 3]
    keeps = list(range(415, 422))
    logits[keeps] = np.arange(7)
    assert engine.look_options(logits, keeps) == [421, 420, 419], "a draw of tickets: the best three sets"
    assert engine.look_options(logits, [417, 419, 420, 421]) == [421, 420, 419, 417], "the opening: every set"
    flat = np.zeros(422, np.float32)
    for legal in ([2, 9, blind, tickets], keeps, claims):
        assert engine.look_options(flat, legal)[0] == engine.first_choice(flat, legal)
    with pytest.raises(ValueError):
        engine.look_options(logits, [])


def test_a_candidate_is_better_by_three_points_and_two_standard_errors(engine):
    first = [0, 10, -5, 3]
    better = lambda other: engine.look_better(np.array([first, other], np.float32))  # noqa: E731
    assert better([5, 15, 0, 8]), "5 points in every world"
    assert not better([2, 12, -3, 5]), "2 points in every world: too few"
    assert not better([40, -20, 30, -20]), "6 points on average, the noise far above"
    assert not better([-5, 5, -10, -2]), "worse"
    with pytest.raises(ValueError):
        engine.look_better(np.zeros((3, 4), np.float32))


def test_a_look_plays_whole_games_and_leaves_its_game_alone(engine, stub):
    g = midgame(engine, players=3, seed=4)
    before, legal = (g.view(g.current), g.moves()), g.legal()
    options = legal[:3]
    net = stub(0.2)
    table = run(engine.Lookahead(g, options, worlds=3, seed=5), net)
    assert table.shape == (len(options), 3) and table.dtype == np.float32
    assert net.rows > 100, "the copies were played to their ends"
    assert (np.abs(table) < 400).all() and table.any()
    assert (g.view(g.current), g.moves()) == before
    assert (run(engine.Lookahead(g, options, worlds=3, seed=5), stub(0.2)) == table).all(), "the same seed, the same worlds"
    assert (run(engine.Lookahead(g, options, worlds=3, seed=6), stub(0.2)) != table).any(), "another seed, other worlds"
    same = run(engine.Lookahead(g, [legal[0], legal[0]], worlds=4, seed=1), stub(0.2))
    assert (same[0] == same[1]).all(), "the copies of a world differ by the candidate alone"


def test_a_look_checks_what_it_is_given(engine, stub):
    g = midgame(engine)
    legal = g.legal()
    illegal = next(a for a in range(422) if a not in legal)
    for options, worlds in (([illegal], 2), ([], 2), (legal[:2], 0)):
        with pytest.raises(ValueError):
            engine.Lookahead(g, options, worlds)
    look = engine.Lookahead(g, legal[:2], worlds=2)
    rows = look.prepare()
    assert rows == len(look.observations()["legal"]) > 0
    with pytest.raises(ValueError):
        look.feed(np.zeros((rows + 1, 422), np.float32))


def test_a_match_gives_the_look_of_a_waiting_row_and_takes_the_moves(engine, stub):
    net = stub(0.5)
    arena = engine.Arena("net", "net", deals=2, players=2, seed=9)
    rows = arena.prepare()
    obs = arena.observations()
    logits = net(obs)
    legal = [np.flatnonzero(obs["legal"][i]).tolist() for i in range(rows)]
    options = engine.look_options(logits[0], legal[0])
    first = run(arena.look(0, options, 3), stub(0.5))
    assert first.shape == (len(options), 3)
    assert (run(arena.look(0, options, 3), stub(0.5)) == first).all(), "a row's look replays: its seed is the game's"
    assert (run(arena.look(0, options, 3, stage=1), stub(0.5)) != first).any(), "the second look deals other worlds"
    for bad in ([legal[0][0]], [10 ** 4] * rows, [next(a for a in range(422) if a not in legal[0])] + [m[0] for m in legal[1:]]):
        with pytest.raises((ValueError, OverflowError)):
            arena.play(bad)
    assert arena.prepare() == rows, "nothing was played by a refused call"
    with pytest.raises(ValueError):
        arena.look(rows, options, 3)
    arena.play([m[0] for m in legal])
    assert arena.prepare() > 0 and not arena.done


class FixedLook:
    """A look whose table is known beforehand: the decision logic can then be tested exactly."""

    def __init__(self, table) -> None:
        self.t = np.array(table, np.float32)

    def prepare(self) -> int:
        return 0

    def table(self) -> np.ndarray:
        return self.t


def position(tables: dict, asked: list):
    """A position with four candidates whose looks are `tables[stage]`; `asked` gets every look made."""
    logits = np.full(422, -9, np.float32)
    logits[[7, 413, 414, 409]] = [5, 3, 0.5, 2]

    def make(options, worlds, stage):
        asked.append((list(options), worlds, stage))
        return FixedLook(tables[stage])

    return logits, [7, 409, 413, 414], make


def test_another_move_is_played_only_when_the_second_look_proves_it(engine):
    from deepttr.look import decide

    ev = None  # the fixed looks never ask the network
    first = [[10] * 4, [12] * 4, [30] * 4, [5] * 4]  # the third candidate (drawing tickets) looks best
    asked: list = []
    proved = [[0, 10, -5, 3] * 2, [5, 15, 0, 8] * 2]
    (move, facts), = decide(ev, [position({0: first, 1: proved}, asked)], worlds=4, check=8)
    assert asked == [([7, 413, 414, 409], 4, 0), ([7, 414], 8, 1)], "the first choice and the best other, again"
    assert move == 414 and facts == {"options": [7, 413, 414, 409], "changed": True, "first": [10, 12, 30, 5], "check": 5.0}

    asked.clear()
    refused = [[0, 10, -5, 3] * 2, [2, 12, -3, 5] * 2]  # two points ahead: not enough
    (move, facts), = decide(ev, [position({0: first, 1: refused}, asked)], worlds=4, check=8)
    assert move == 7 and not facts["changed"] and facts["check"] == 2.0 and len(asked) == 2

    asked.clear()
    content = [[10] * 4, [9] * 4, [10] * 4, [5] * 4]  # nothing ends above the first choice: no second look
    (move, facts), = decide(ev, [position({0: content}, asked)], worlds=4, check=8)
    assert move == 7 and "check" not in facts and len(asked) == 1

    asked.clear()
    logits, _, make = position({}, asked)
    assert decide(ev, [(logits, [409], make)]) == [(409, {"options": [409], "changed": False})] and not asked


def test_out_of_time_the_first_choice_is_played(engine, stub):
    from deepttr.look import decide

    g = midgame(engine)
    net = stub(0.1)
    logits = net(g.observation(g.current))[0]
    make = lambda options, worlds, stage: engine.Lookahead(g, options, worlds, 1 + stage)  # noqa: E731
    (move, facts), = decide(net, [(logits, g.legal(), make)], worlds=2, check=2, deadline=time.perf_counter() - 1)
    assert move == engine.first_choice(logits, g.legal()) and facts["time"] is True and not facts["changed"]


def test_decisions_looked_at_together_come_out_as_alone(engine, stub):
    from deepttr.look import decide

    games = [midgame(engine, players=n, seed=20 + n, rules={"trains": 14}) for n in (2, 3, 4)]

    def positions(net):
        out = []
        for k, g in enumerate(games):
            make = lambda options, worlds, stage, g=g, k=k: engine.Lookahead(g, options, worlds, 100 * k + stage)  # noqa: E731
            out.append((net(g.observation(g.current))[0], g.legal(), make))
        return out

    together = decide(stub(0.3), positions(stub(0.3)), worlds=3, check=4)
    alone = [decide(stub(0.3), [p], worlds=3, check=4)[0] for p in positions(stub(0.3))]
    assert together == alone
    assert all(move in g.legal() and facts["options"][0] == engine.first_choice(p[0], g.legal())
               for (move, facts), g, p in zip(together, games, positions(stub(0.3))))


def test_a_looker_moves_in_a_game(engine, stub):
    from deepttr.look import Looker

    looker = Looker(stub(0.6), worlds=2, check=2, seconds=None)
    g = midgame(engine, rules={"trains": 14})
    move = looker.move(g, seed=5)
    assert move in g.legal() and looker.last["options"][0] == engine.first_choice(stub(0.6)(g.observation(g.current))[0], g.legal())
    assert looker.stats["decisions"] == 1 and looker.move(g, seed=5) == move, "the same seed, the same look"
    over = engine.Game(players=2, seed=1, rules={"trains": 10})
    while not over.is_over():
        over.step(over.bot_move("heuristic"))
    assert looker.move(over) == -1 and looker.last is None, "nothing to decide, nothing to look at"
    assert looker.stats["decisions"] == 2
    with pytest.raises(ValueError):
        Looker(stub(), worlds=0)


def test_a_looker_plays_a_duplicate_match(engine, stub):
    from deepttr.look import Looker
    from deepttr.play import Bot, duplicate_match

    short = {"trains": 10}  # short games: a look plays every one of its copies to the end
    looker = Looker(stub(0.3), worlds=2, check=2, seconds=None)
    m = duplicate_match(looker, stub(0.3), deals=2, players=2, seed=4, rules=short)
    assert m["games"] == 4 and -1 <= m["a_mean"] <= 1 and looker.stats["decisions"] > 10
    again = duplicate_match(Looker(stub(0.3), 2, 2, None), stub(0.3), deals=2, players=2, seed=4, rules=short)
    assert {k: v for k, v in again.items() if k != "seconds"} == {k: v for k, v in m.items() if k != "seconds"}
    even = duplicate_match(Looker(stub(0.3), 2, 2, None), Looker(stub(0.3), 2, 2, None), deals=2, players=2, seed=4, rules=short)
    assert even["a_mean"] == 0.0, "two identical players that look ahead still play a deal the same from both seats"
    three = duplicate_match(Looker(stub(0.3), 2, 2, None), Bot("random"), deals=1, players=3, seed=2, rules=short)
    assert three["games"] == 3
