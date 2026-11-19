"""The losses of training: the reward of a game and its TD targets, the hand penalty, the weights of
imitation, PPO's pieces, the critic, the move head's loss, and how a batch's loss is put together."""
import numpy as np
import pytest
import torch

from deepttr.train import losses as L
from test_train_replay import synthetic


def views(final):
    """Every seat's view of a table's final scores (hundreds of points), seat 0 the viewer's."""
    n = len(final)
    score = np.zeros((n, 5))
    for s in range(n):
        score[s, :n] = np.roll(final, -s)
    return score, np.full(n, n)


@pytest.mark.parametrize("final", [[1.2, 0.8], [0.5, 1.4, 0.9], [1.0, 1.0, 0.3, 0.7], [0.2, 0.9, 1.6, 1.1, 0.4]])
def test_the_reward_is_zero_sum_at_every_table(final):
    score, n = views(np.array(final))
    r = L.reward(score, n, np.zeros((len(final), 30)))
    assert abs(r.sum()) < 1e-12
    if len(final) == 2:
        assert r[0] == pytest.approx(final[0] - final[1]), "at a table of two: the margin"
    best = int(np.argmax(final))
    assert r[best] == r.max()


def test_the_bonuses_are_the_seats_own_and_stop_at_their_goals():
    score, n = views(np.array([1.2, 0.8]))
    outcome = np.zeros((2, 30))
    outcome[0, :3] = 1  # seat 0 completed three tickets, seat 1 seven
    outcome[1, 3:10] = 1
    plain = L.reward(score, n, outcome)
    both = L.reward(score, n, outcome, 1.0, 1.0)
    assert both[0] - plain[0] == pytest.approx(L.SCORE_WEIGHT * (1.2 - 1.6) + 0.1 * (3 - 5))
    assert both[1] - plain[1] == pytest.approx(L.SCORE_WEIGHT * (0.8 - 1.6)), "seven tickets: above the goal, no bonus"
    half = L.reward(score, n, outcome, 0.5, 0.0)
    assert half[0] - plain[0] == pytest.approx(0.5 * L.SCORE_WEIGHT * (1.2 - 1.6))
    above, _ = views(np.array([1.9, 1.7]))
    assert (L.reward(above, n, np.ones((2, 30)), 1.0, 1.0) == L.reward(above, n, np.ones((2, 30)))).all()


def test_the_bonuses_come_in_over_the_steps():
    assert L.bonus_shares(0, False) == (1.0, 1.0) == L.bonus_shares(123_456, False)
    assert L.bonus_shares(0, True) == (0.0, 0.0)
    assert L.bonus_shares(5_000, True) == (0.5, 0.0)
    assert L.bonus_shares(25_000, True) == (1.0, 0.0)
    assert L.bonus_shares(26_500, True) == (1.0, 0.5)
    assert L.bonus_shares(40_000, True) == (1.0, 1.0)


def one_game(final, moves_of, forecast):
    """Records of one game: seat s moved at `moves_of[s]`, with forecasts `forecast[s]`."""
    rows = [(s, m, f) for s in range(len(final)) for m, f in zip(moves_of[s], forecast[s])]
    score = np.zeros((len(rows), 5))
    for i, (s, _, _) in enumerate(rows):
        score[i, :len(final)] = np.roll(final, -s)
    return {"score": score, "n_players": np.full(len(rows), len(final)), "ticket_outcome": np.zeros((len(rows), 30), np.int8),
            "root_forecast": np.array([f for _, _, f in rows]), "lesson": np.zeros(len(rows), np.uint8),
            "game": np.zeros(len(rows), np.uint64), "moves": np.array([m for _, m, _ in rows])}


def test_td_targets_chain_a_seats_forecasts_back_from_its_last_move():
    recs = one_game(np.array([1.0, 0.6]), [[0, 2, 4], [1, 3, 5]], [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    t = L.td_targets(recs, lam=0.5)
    assert t[2] == pytest.approx(0.4) and t[5] == pytest.approx(-0.4), "the last move: the game's reward"
    assert t[1] == pytest.approx(0.5 * 0.3 + 0.5 * 0.4)
    assert t[0] == pytest.approx(0.5 * 0.2 + 0.5 * (0.5 * 0.3 + 0.5 * 0.4))
    plain = L.reward(recs["score"], recs["n_players"], None).astype(np.float32)
    assert t.dtype == np.float32 and (L.td_targets(recs, lam=1.0) == plain).all()


def test_td_targets_give_the_reward_where_there_is_no_chain():
    recs = one_game(np.array([1.0, 0.6]), [[0, 2, 4], [1, 3, 5]], [[0.1, 0.2, 0.3], [-0.1, np.nan, -0.3]])
    recs["lesson"][1] = 1
    t = L.td_targets(recs, lam=0.5)
    assert t[1] == pytest.approx(0.4), "a lesson"
    assert t[0] == pytest.approx(0.5 * 0.3 + 0.5 * 0.4), "the chain passes the lesson by"
    assert t[4] == pytest.approx(-0.4), "no forecast"
    even = one_game(np.array([0.8, 0.8]), [[0, 2], [1, 3]], [[0.1, 0.2], [0.3, 0.4]])
    assert (L.td_targets(even, lam=0.5) == 0).all(), "seats a turn of the table cannot tell apart"


@pytest.mark.parametrize("cards,points", [(0, 0), (14, 0), (15, 5), (16, 10), (19, 25), (20, 30), (25, 30), (45, 30)])
def test_the_hand_penalty_schedule(cards, points):
    assert float(L.hand_points(cards)) == points


def test_the_hand_penalty_falls_on_the_first_move_of_a_turn_that_ends_with_a_large_hand():
    # (seat, moves, cards in hand at the record): seat A's turns start at moves 0, 4, 6 and 10, seat B's
    # at 2, 5 and 8; a turn's hand at its end is the one at the seat's next turn
    rows = [(0, 0, 10), (0, 1, 11), (1, 2, 5), (1, 3, 6), (0, 4, 15), (1, 5, 19), (0, 6, 14), (0, 7, 16),
            (1, 8, 25), (1, 9, 25), (0, 10, 20)]
    hands = np.zeros((len(rows) + 1, 5, 9), np.uint8)
    for i, (_, _, c) in enumerate(rows):
        hands[i, 0, 0], hands[i, 0, 1] = c // 2, c - c // 2
    start = np.zeros((len(rows) + 1, 30), np.uint8)
    for i, (s, _, _) in enumerate(rows):
        start[i, [2 * s, 2 * s + 1]] = 1
    start[-1, [0, 1]] = 1  # a lesson of seat A's at its second turn, with a full hand
    hands[-1, 0, 0] = 30
    recs = {"game": np.zeros(len(rows) + 1, np.uint64), "moves": np.array([m for _, m, _ in rows] + [4], np.uint16),
            "ticket_start": start, "hands": hands, "lesson": np.array([0] * len(rows) + [1], np.uint8)}
    out, turns, points = L.hand_penalty(recs)
    want = np.zeros(len(rows) + 1, np.float32)
    want[0] = -0.05  # A's first turn ended with 15
    want[6] = -0.30  # A's third turn (moves 6-7) ended with 20; its second ended with 14: none
    want[2] = -0.25  # B's first turn ended with 19
    want[5] = -0.30  # B's second ended with 25; B's last turn and A's pay nothing
    np.testing.assert_allclose(out, want, atol=1e-7)
    assert turns == 4 and points == 90.0
    recs["game"][rows.index((1, 5, 19))] = 7  # that record now a game of its own: B's turns of game 0 no longer meet it
    out2, _, _ = L.hand_penalty(recs)
    assert out2[2] == pytest.approx(-0.30), "B's first turn now ends with the hand of its turn at move 8"


def test_imitation_weights_a_move_by_its_score_within_its_table_size():
    score = torch.tensor([1.0, 2.0, 3.0, 5.0, 0.0])
    n = torch.tensor([2, 2, 2, 3, 4])
    w = L.score_by_table(score, n)
    assert w.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0], abs=1e-5)


def test_the_advantage_is_centred_by_opponent_and_table():
    adv = torch.tensor([1.0, 3.0, 10.0, 20.0, 5.0, 7.0])
    opp = torch.tensor([0, 0, 1, 1, 1, 0])
    n = torch.tensor([2, 2, 2, 2, 3, 2])
    rows = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    out = L.advantage_by_opponent(adv, opp, n, rows)
    assert out.tolist() == [-1.0, 1.0, -5.0, 5.0, 0.0, 0.0]


def ppo_batch(logits, act, adv):
    b = len(act)
    p = torch.softmax(logits, -1)[torch.arange(b), act]
    return {"policy": torch.nn.functional.one_hot(act, logits.shape[1]).float(), "root_prob": p.detach(),
            "opponent": torch.zeros(b, dtype=torch.long), "n_players": torch.full((b,), 2), "advantage": adv}


def test_ppo_pushes_each_move_by_its_advantage():
    torch.manual_seed(0)
    logits = torch.randn(4, 6, requires_grad=True)
    act = torch.tensor([1, 2, 3, 4])
    b = ppo_batch(logits, act, torch.tensor([1.0, -1.0, 1.0, -1.0]))
    loss, st = L.ppo_policy(logits, b, b["advantage"], torch.ones(4))
    assert st["ppo_clip"] == 0 and st["ppo_ratio"] == pytest.approx(1.0) and st["ppo_kl"] == pytest.approx(0.0, abs=1e-6)
    assert st["adv_rms"] == pytest.approx(1.0)
    loss.backward()
    g = logits.grad[torch.arange(4), act]
    assert (g[[0, 2]] < 0).all() and (g[[1, 3]] > 0).all(), "a step against the gradient raises a good move, lowers a bad one"


def test_ppo_terms_kl_hold_and_lessons():
    torch.manual_seed(1)
    logits = torch.randn(6, 8)
    act = torch.tensor([0, 1, 2, 3, 4, 5])
    b = ppo_batch(logits, act, torch.tensor([1.0, -1.0, 2.0, -2.0, 0.0, 0.0]))
    b["root_prob"] = b["root_prob"] * 0.5  # the policy moved since the moves were played
    rows = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])  # the last two rows are lessons
    base, st = L.ppo_policy(logits, b, b["advantage"], rows)
    assert st["ppo_clip"] == 1.0, "ratio 2: outside the clip"
    with_kl, st2 = L.ppo_policy(logits, b, b["advantage"], rows, kl_beta=2.0)
    assert st2["ppo_kl"] > 0 and float(with_kl - base) == pytest.approx(2.0 * st2["ppo_kl"] * 4 / 6, rel=1e-5)
    held, st3 = L.ppo_policy(logits, b, b["advantage"], rows, hold=0.1)
    assert float(held - base) == pytest.approx(0.1 * st3["entropy"] * 4 / 6, rel=1e-5)
    lessons = torch.zeros(6, 8, requires_grad=True)
    L.ppo_policy(lessons, b, b["advantage"], rows)[0].backward()
    assert lessons.grad[4, 4] < 0 and lessons.grad[5, 5] < 0, "a lesson's target is raised by cross-entropy"


def test_the_critic_explains_what_it_forecasts():
    r = torch.tensor([0.5, -0.2, 0.1, 0.9, -0.7])
    rows = torch.ones(5)
    loss, st = L.critic_loss(r.clone(), r, rows)
    assert float(loss) == 0 and st["critic_explained"] == pytest.approx(1.0)
    _, st = L.critic_loss(torch.full((5,), float(r.mean())), r, rows)
    assert st["critic_explained"] == pytest.approx(0.0, abs=1e-6)
    far = torch.tensor([3.0, 0, 0, 0, 0])
    assert float(L.critic_loss(far, torch.zeros(5), rows)[0]) == pytest.approx(0.5 * (3.0 - 0.25) / 5), "Huber: linear far off"


def test_the_move_head_learns_the_playouts_means_and_the_pairs_they_tell_apart():
    look = torch.tensor([[5, 7, 9, -1, -1], [-1] * 5])
    table = torch.full((2, 5, 32), float("nan"))
    table[0, 0, :8] = 0.0
    table[0, 1, :8] = 10.0
    table[0, 2, :8] = torch.tensor([1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0])  # 1 point above move 5: not told apart
    move = torch.zeros(2, 422)
    move[0, 7], move[0, 9] = 0.10, 0.01
    loss, st = L.move_loss(move, look, table)
    assert st["move_rows"] == 1.0 and st["move_pairs"] == 2.0 and st["move_number"] == pytest.approx(0.0, abs=1e-7)
    move[0, 7] = -0.10
    wrong, _ = L.move_loss(move, look, table)
    assert float(wrong) > float(loss), "the wrong order costs more"
    assert L.move_loss(move, look[1:], table[1:])[0] is None


def random_outputs(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = len(b["policy"])
    r = lambda *s: torch.randn(*s, generator=g, requires_grad=True)  # noqa: E731
    legal = b["legal"] > 0
    logits = torch.where(legal, r(B, 422), torch.full((B, 422), -1e9))
    return {"logits": logits, "belief": r(B, 5, 30), "owner": r(B, 100, 6), "ticket": r(B, 30, 3), "hand": torch.nn.functional.softplus(r(B, 5, 9)),
            "path": r(B, 5, 30, 2), "trail": r(B, 5, 2), "adv": r(B), "move": r(B, 422)}


def tensors(recs):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).float() if v.dtype == np.float16 else torch.from_numpy(np.ascontiguousarray(v))
            for k, v in recs.items()}


def test_absent_seats_and_left_out_rows_change_nothing():
    recs = synthetic(np.random.default_rng(7), games=4, tables=(2, 3))
    b = tensors(recs)
    out = {k: v.detach() for k, v in random_outputs(b).items()}
    base = L.forecast_losses(out, b)
    absent = {k: v.detach().clone() for k, v in out.items()}
    two = b["n_players"] == 2
    for k in ("belief", "hand", "path", "trail"):
        absent[k][two, 2:] += 5.0
    moved = L.forecast_losses(absent, b)
    for k in base:
        assert float(moved[k]) == pytest.approx(float(base[k]), rel=1e-6), k
    end = torch.ones(len(b["policy"]))
    end[:10] = 0.0
    kept = L.forecast_losses(out, b, end)
    other = {k: v.detach().clone() for k, v in out.items()}
    for k in ("owner", "ticket", "trail", "belief"):
        other[k][:10] += 3.0
    left = L.forecast_losses(other, b, end)
    for k in ("owner", "ticket", "trail"):
        assert float(left[k]) == pytest.approx(float(kept[k]), rel=1e-6), k
    assert float(left["belief"]) != pytest.approx(float(kept["belief"])), "the belief learns from every row"


@pytest.mark.parametrize("method", ["teacher", "imitate", "ppo"])
def test_a_batch_loss_is_its_weighted_terms(method):
    recs = synthetic(np.random.default_rng(8), games=4, lesson_share=0.1, looks=True)
    recs["advantage"] = L.reward(recs["score"], recs["n_players"], recs["ticket_outcome"]).astype(np.float32)
    recs["adv_td"] = L.td_targets(recs)
    recs["hand_penalty"] = L.hand_penalty(recs)[0]
    recs["opponent"] = np.zeros(len(recs["policy"]), np.uint8)
    b = tensors(recs)
    out = random_outputs(b)
    total, parts, stats = L.batch_loss(out, b, method, lessons=8, trail_weight=0.1, critic=method == "ppo", hold=0.01, kl_beta=1.0)
    weights = {**L.WEIGHTS, "trail": 0.1, "move": L.MOVE_WEIGHT}
    want = sum(weights[k] * float(v.detach()) for k, v in parts.items())
    if method == "ppo":
        want += L.CRITIC_WEIGHT * stats["critic"]
        assert stats["td_rows"] == 1.0 and {"ppo_kl", "entropy", "critic_explained"} <= set(stats)
    assert set(parts) == {"policy", "belief", "owner", "ticket", "hand", "path", "trail", "move"}
    assert float(total.detach()) == pytest.approx(want, rel=1e-5)
    total.backward()
    assert out["adv"].grad is not None if method == "ppo" else out["adv"].grad is None


def test_ppo_target_takes_the_td_return_and_the_hand_penalty():
    recs = synthetic(np.random.default_rng(9), games=2, tables=(2,))
    n = len(recs["policy"])
    b = tensors({**recs, "advantage": np.zeros(n, np.float32), "adv_td": np.full(n, np.nan, np.float32),
                 "hand_penalty": np.zeros(n, np.float32), "opponent": np.zeros(n, np.uint8)})
    out = random_outputs(b)
    rows = torch.ones(n)
    target = torch.linspace(-1, 1, n)
    penalty = torch.where(torch.arange(n) % 3 == 0, -0.3, 0.0)  # a constant one would vanish in the centring
    want, _ = L.ppo_policy(out["logits"], b, target + penalty, rows, kl_beta=L.KL_BETA)
    got, _, _ = L.batch_loss(out, {**b, "adv_td": target, "hand_penalty": penalty}, "ppo", trail_weight=0.0)
    parts = L.forecast_losses(out, b)
    extra = sum(L.WEIGHTS[k] * float(v.detach()) for k, v in parts.items() if k in L.WEIGHTS)
    assert float(got.detach()) == pytest.approx(float(want.detach()) + extra, rel=1e-5)


def test_the_adaptive_weights():
    assert L.adapt_kl_beta(1.0, 0.04) == 1.5 and L.adapt_kl_beta(1.0, 0.01) == pytest.approx(1 / 1.5) and L.adapt_kl_beta(1.0, 0.02) == 1.0
    assert L.adapt_kl_beta(1e3, 1.0) == 1e3 and L.adapt_kl_beta(1e-3, 0.0) == 1e-3
    assert L.adapt_entropy_hold(0.0, 1.05, 1.0) == 1e-3, "on, from its least"
    assert L.adapt_entropy_hold(0.01, 1.05, 1.0) == pytest.approx(0.012)
    assert L.adapt_entropy_hold(0.012, 0.95, 1.0) == pytest.approx(0.01)
    assert L.adapt_entropy_hold(0.0011, 0.95, 1.0) == 0.0, "off under its least"
    assert L.adapt_entropy_hold(0.01, 1.01, 1.0) == 0.01 and L.adapt_entropy_hold(0.9, 2.0, 1.0) == 1.0


def test_the_learning_rate_warms_up_after_a_fresh_optimizer():
    assert L.warm_lr(5e-5, 0, 0) == pytest.approx(1e-7)
    assert L.warm_lr(5e-5, 499, 0) == 5e-5 == L.warm_lr(5e-5, 10_000, 0)
    assert L.warm_lr(5e-5, 1000, 1000) == pytest.approx(1e-7), "from the step it started fresh at"
