"""The losses of training, the reward of a game, and the pieces of PPO.

A batch is a dict of tensors in the keys of the window (`replay.STORE`): the observation; `policy`,
the target (the teacher's policy, or the move played as a one-hot); what the seat to act could not
see then - `opp_tickets` (every seat's tickets) and `hands` (every seat's cards by type) - and
`path_cost`, the trains each seat still needed to join each ticket's cities (`FAR`: it no longer
could); and how the game ended, seen from that seat, seat 0: `score` (every seat's final score /
100), `route_owner` (-1: nobody), `ticket_outcome` (0 not held, 1 completed, -1 failed), `trail`
(every seat's longest trail / 30) and `trail_best` (whether it held the longest).

`batch_loss` is the loss of a batch. The policy learns the teacher's targets by cross-entropy, or
the moves of its own games by cross-entropy weighted by `score_by_table`, or by PPO (`ppo_policy`)
on each move's reward (`reward`, `td_targets`, `hand_penalty`). Rewards, advantages and forecasts
are in hundreds of points, as the records keep scores.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SEATS = 5
FAR = 255  # path_cost: the seat can no longer join the ticket's cities
# the weights of the loss terms; the longest trail's is the phase's (`TrainConfig.trail_weight`)
WEIGHTS = {"policy": 1.0, "belief": 0.25, "owner": 0.1, "ticket": 0.1, "hand": 1.0, "path": 1.0}
CRITIC_WEIGHT, MOVE_WEIGHT = 1.0, 1.0

# the reward's bonuses (`reward`): the own points up to SCORE_GOAL, SCORE_WEIGHT a point short of
# it, and TICKET_POINTS a ticket completed short of TICKET_GOAL; brought in over the steps by
# `bonus_shares`
SCORE_GOAL, SCORE_WEIGHT = 160.0, 1.5
TICKET_GOAL, TICKET_POINTS = 5, 10.0
SCORE_RAMP = 10_000
TICKET_RAMP_FROM, TICKET_RAMP = 25_000, 3_000
TD_LAMBDA = 0.995
# the penalty of a turn that ends with a large hand (`hand_penalty`): HAND_POINTS points a train
# card above HAND_FREE, at most HAND_MAX
HAND_FREE, HAND_POINTS, HAND_MAX = 14, 5.0, 30.0

# PPO: the clip of the probability ratio, the target of the KL penalty and its first weight
CLIP = 0.2
KL_TARGET, KL_BETA = 0.02, 1.0
CRITIC_DELTA = 0.5  # the critic's Huber loss: quadratic up to 50 points of error, linear beyond
# the move head (`move_loss`): two moves are told apart when the mean of their paired playouts
# differs by LOOK_POINTS points or more, and by LOOK_Z standard errors
LOOK_POINTS, LOOK_Z = 3.0, 2.0


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp(min=1)


def seats_present(n_players: torch.Tensor) -> torch.Tensor:
    """[B, seats] 1 for the seats of each record's table."""
    return (torch.arange(SEATS, device=n_players.device)[None, :] < n_players.long()[:, None]).float()


def forecast_losses(out: dict, b: dict, end_rows: torch.Tensor | None = None, trail: bool = True) -> dict:
    """The auxiliary forecasts' losses, each a mean over the batch: `belief` (which tickets each other
    seat holds), `owner` (who holds each route at the end), `ticket` (how each ticket ends), `hand`
    (every other seat's cards), `path` (the trains each seat still needs for each ticket, and whether
    it still can) and `trail` (the longest trail at the end). `end_rows` (1 or 0 a row; None: all of
    them) are the rows whose game's end teaches `owner`, `ticket` and `trail`."""
    present = seats_present(b["n_players"])
    others = present.clone()
    others[:, 0] = 0.0
    nb = present.shape[0]
    end = torch.ones_like(present[:, 0]) if end_rows is None else end_rows.float()
    bel = F.binary_cross_entropy_with_logits(out["belief"], b["opp_tickets"].float(), reduction="none").mean(-1)
    parts = {"belief": ((bel * others).sum(-1) / others.sum(-1).clamp(min=1)).mean()}
    own = F.cross_entropy(out["owner"].reshape(-1, SEATS + 1), (b["route_owner"].long() + 1).reshape(-1), reduction="none")
    parts["owner"] = masked_mean(own.reshape(nb, -1).mean(-1), end)
    tk = b["ticket_outcome"].long()
    tk = torch.where(tk < 0, torch.full_like(tk, 2), tk)
    tic = F.cross_entropy(out["ticket"].reshape(-1, 3), tk.reshape(-1), reduction="none")
    parts["ticket"] = masked_mean(tic.reshape(nb, -1).mean(-1), end)
    hand = ((out["hand"] - b["hands"].float() / 6.0) ** 2).mean(-1)
    parts["hand"] = ((hand * others).sum(-1) / others.sum(-1).clamp(min=1)).mean()
    cost = b["path_cost"].long()
    can = (cost != FAR).float()
    seat = present.unsqueeze(-1).expand_as(can)
    reach = F.binary_cross_entropy_with_logits(out["path"][..., 1], can, reduction="none")
    dist = (out["path"][..., 0] - cost.clamp(max=90).float() / 45.0) ** 2 * can
    parts["path"] = ((reach + dist) * seat).sum() / seat.sum().clamp(min=1)
    if trail:
        m = present * end[:, None]
        best = F.binary_cross_entropy_with_logits(out["trail"][..., 0], b["trail_best"].float(), reduction="none")
        length = (out["trail"][..., 1] - b["trail"].float()) ** 2
        parts["trail"] = ((best + length) * m).sum() / m.sum().clamp(min=1)
    return parts


def score_by_table(score: torch.Tensor, n_players: torch.Tensor) -> torch.Tensor:
    """The weight of each move when the policy imitates its own games: how far its player's final
    score was above the mean of the batch's records at a table of the same size, in their standard
    deviations; 0 below the mean and for a table size alone in the batch. [B]"""
    sc = score.float()
    n = n_players.long()
    w = torch.zeros_like(sc)
    for t in n.unique():
        g = n == t
        if int(g.sum()) >= 2:
            w[g] = ((sc[g] - sc[g].mean()) / (sc[g].std() + 1e-6)).clamp(min=0)
    return w


def reward(score, n_players, ticket_outcome, score_share: float = 0.0, ticket_share: float = 0.0) -> np.ndarray:
    """Each record's reward, float64 [N]: its seat's margin over the best other seat less the mean of
    every seat's such margin at the table - a zero-sum game - and, with the shares of the bonuses in
    (`bonus_shares`), SCORE_WEIGHT a point the seat's own score fell short of SCORE_GOAL and
    TICKET_POINTS a ticket it completed short of TICKET_GOAL (none above the goals). The bonuses are
    the seat's own: centred on the table they would reward only the differences again."""
    score = np.asarray(score, dtype=np.float64)
    n = np.asarray(n_players).astype(np.int64)
    present = np.arange(score.shape[1])[None, :] < n[:, None]
    masked = np.where(present, score, -1e9)
    top = -np.sort(-masked, axis=1)
    best_other = np.where(masked == top[:, :1], top[:, 1:2], top[:, :1])
    margin = np.where(present, score - best_other, 0.0)
    r = margin[:, 0] - margin.sum(1) / n
    if score_share:
        goal = SCORE_GOAL / 100.0
        r = r + SCORE_WEIGHT * score_share * (np.minimum(score[:, 0], goal) - goal)
    if ticket_share:
        done = (np.asarray(ticket_outcome) == 1).sum(1).astype(np.float64)
        r = r + TICKET_POINTS / 100.0 * ticket_share * (np.minimum(done, TICKET_GOAL) - TICKET_GOAL)
    return r


def bonus_shares(step: int, ramp: bool) -> tuple[float, float]:
    """How much of the score bonus and of the ticket bonus is in at a training step: all of both,
    or with the ramp, the score's from none at the start to all at SCORE_RAMP steps and the
    tickets' from none at TICKET_RAMP_FROM to all TICKET_RAMP steps later."""
    if not ramp:
        return 1.0, 1.0
    return min(1.0, max(step, 0) / SCORE_RAMP), min(1.0, max(step - TICKET_RAMP_FROM, 0) / TICKET_RAMP)


def td_targets(recs: dict, score_share: float = 0.0, ticket_share: float = 0.0, lam: float = TD_LAMBDA) -> np.ndarray:
    """PPO's target of each move of whole games, float32 [N]: the TD(lambda) return of its game's
    reward over the critic's forecasts of the same seat's later positions (`root_forecast`, the
    forecast the network gave when it moved). The seat's last move gets the reward; each move
    before it (1 - lam) x the forecast at the seat's next move + lam x that move's target, so a move
    after which the forecast rose gets more than the moves before it. A seat of a game is told by
    its records' final scores, seen from it; where there is no chain - a lesson, a move without a
    forecast, seats that a turn of the table cannot tell apart (equal scores all round) - the
    target is the reward itself."""
    score = np.asarray(recs["score"], dtype=np.float64)
    n = np.asarray(recs["n_players"]).astype(np.int64)
    r = reward(score, n, recs["ticket_outcome"], score_share, ticket_share)
    target = r.astype(np.float32)
    if lam >= 1.0 or not len(n):
        return target
    forecast = np.asarray(recs["root_forecast"], dtype=np.float64)
    chained = np.isfinite(forecast) & (np.asarray(recs["lesson"]) == 0)
    game = np.asarray(recs["game"])
    seats: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(chained).tolist():
        seats.setdefault((int(game[i]), tuple(score[i, :n[i]].tolist())), []).append(i)
    moves = np.asarray(recs["moves"]).astype(np.int64)
    for (_, s), rows in seats.items():
        if any(s[k:] + s[:k] == s for k in range(1, len(s))):
            continue
        rows.sort(key=lambda i: moves[i])
        g = r[rows[-1]]
        for j in range(len(rows) - 2, -1, -1):
            g = (1.0 - lam) * forecast[rows[j + 1]] + lam * g
            target[rows[j]] = g
    return target


def hand_points(cards) -> np.ndarray:
    """The points a turn that ends holding `cards` train cards costs: HAND_POINTS a card above
    HAND_FREE, at most HAND_MAX - 0 up to 14 cards, 5 at 15, 25 at 19, 30 from 20 on."""
    return np.minimum(HAND_MAX, HAND_POINTS * np.maximum(np.asarray(cards, dtype=np.float64) - HAND_FREE, 0.0))


def hand_penalty(recs: dict) -> tuple[np.ndarray, int, float]:
    """The penalty of large hands for the records of whole games: on the first move of every turn a
    seat ends holding more than HAND_FREE train cards, minus `hand_points` of that hand in hundreds
    of points - that move chose to draw, to claim or to take tickets, so it alone decides what the
    hand holds after the turn; 0 on every other record. A seat's turn is a run of its records with
    consecutive `moves`; the hand it ended with is the one it holds at its next turn's first record
    (`hands`, seat 0), so a seat's last turn pays nothing; a seat of a game is told by its starting
    tickets. Lessons are left out. -> (the penalties [N] float32, the turns penalised, their points)"""
    game = np.asarray(recs["game"]).astype(np.int64)
    out = np.zeros(len(game), dtype=np.float32)
    start = np.asarray(recs["ticket_start"]).astype(np.int64)
    who = (start << np.arange(start.shape[1], dtype=np.int64)).sum(1)
    cards = np.asarray(recs["hands"])[:, 0, :].astype(np.int64).sum(-1)
    moves = np.asarray(recs["moves"]).astype(np.int64)
    own = np.flatnonzero(np.asarray(recs["lesson"]) == 0)
    order = own[np.lexsort((moves[own], who[own], game[own]))]
    seat_starts = np.flatnonzero((game[order][1:] != game[order][:-1]) | (who[order][1:] != who[order][:-1])) + 1
    turns, points = 0, 0.0
    for rows in np.split(order, seat_starts):
        first = np.r_[0, np.flatnonzero(np.diff(moves[rows]) > 1) + 1]  # each turn's first record, in `rows`
        for t in range(len(first) - 1):
            p = float(hand_points(cards[rows[first[t + 1]]]))
            if p > 0:
                out[rows[first[t]]] = -p / 100.0
                turns += 1
                points += p
    return out, turns, points


def advantage_by_opponent(adv: torch.Tensor, opponent: torch.Tensor, n_players: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Each move's advantage over the usual of its kind of game: `adv` less the mean over the
    batch's moves (`rows` 1) against the same opponent at a table of the same size; 0 for a move
    alone in its group and for the other rows. [B]"""
    adv = adv.float() * rows
    group = opponent.long() * 8 + n_players.long()
    out = torch.zeros_like(adv)
    game = rows > 0
    for g in group[game].unique():
        k = game & (group == g)
        if int(k.sum()) >= 2:
            out[k] = adv[k] - adv[k].mean()
    return out


def ppo_policy(logits: torch.Tensor, b: dict, target: torch.Tensor, rows: torch.Tensor, hold: float = 0.0,
               kl_beta: float = 0.0, baseline: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """The policy's loss by PPO over a batch -> (the loss, its numbers). The rows `rows` (1) are
    moves of games: the clipped objective on the probability of the move played now over its
    probability then (`root_prob`), the advantage `target` (less `baseline`, the critic's forecast)
    centred by opponent and table size (`advantage_by_opponent`) and divided by its root mean
    square; `kl_beta` times the KL divergence of the policy then from the policy now, by Schulman's
    k3 estimate on the moves played; and `hold` times the policy's entropy, a push against a wider
    policy. The other rows (0) are lessons, learned by cross-entropy to their target."""
    logp = F.log_softmax(logits.float(), dim=-1)
    m = rows.float()
    n = m.sum().clamp(min=1.0)
    lp = logp.gather(1, b["policy"].float().argmax(-1)[:, None]).squeeze(1)
    rp = b["root_prob"].float().clamp(min=1e-6)
    ratio = torch.exp(lp - torch.log(rp))
    adv = target.float() if baseline is None else target.float() - baseline.float()
    adv = advantage_by_opponent(adv, b["opponent"], b["n_players"], m)
    rms = torch.sqrt((adv ** 2).sum() / n)
    a = adv / rms.clamp(min=1e-6)
    surr = torch.minimum(ratio * a, ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * a)
    p = logp.exp()
    ent = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
    ce = -(b["policy"].float() * logp).sum(-1)
    log_r = (lp - torch.log(rp)).clamp(-5.0, 5.0)
    k3 = torch.exp(log_r) - 1.0 - log_r
    loss = (-(surr * m).sum() + hold * (ent * m).sum() + (ce * (1 - m)).sum() + kl_beta * (k3 * m).sum()) / float(m.numel())
    with torch.no_grad():
        stats = {"ppo_clip": float((((ratio - 1).abs() > CLIP).float() * m).sum() / n), "entropy": float((ent * m).sum() / n),
                 "ppo_ratio": float((ratio * m).sum() / n), "ppo_kl": float((k3 * m).sum() / n), "adv_rms": float(rms.detach())}
    return loss, stats


def critic_loss(pred: torch.Tensor, target: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """The critic's loss: its forecast (`adv`) against each game's reward, Huber over the moves of
    games (`rows` 1) -> (the loss, its numbers: `critic` the loss, `critic_explained` the share of
    the rewards' variance it takes off)."""
    m = rows.float()
    n = m.sum().clamp(min=1.0)
    r = target.float()
    loss = (F.huber_loss(pred.float(), r, reduction="none", delta=CRITIC_DELTA) * m).sum() / n
    with torch.no_grad():
        var_r = (((r - (r * m).sum() / n) ** 2) * m).sum() / n
        e = (r - pred.float()) * m
        var_e = (((e - e.sum() / n) ** 2) * m).sum() / n
        stats = {"critic": float(loss.detach()), "critic_explained": float(1.0 - var_e / var_r.clamp(min=1e-9))}
    return loss, stats


def move_loss(move: torch.Tensor, look: torch.Tensor, table: torch.Tensor) -> tuple[torch.Tensor | None, dict]:
    """The move head's loss on the records that keep their look's playouts: `look` [B, K] the moves
    tried (-1 past them), `table` [B, K, W] the margin each move's playouts ended with, points (NaN
    past them; the j-th playout of every move in the same world). The forecast of each move tried
    learns the mean of its playouts (Huber), and each pair of moves the playouts tell apart is
    ordered by a logistic loss on the forecasts' difference. -> (the loss, None without such a
    record; its numbers)."""
    rows = look[:, 0] >= 0
    count = int(rows.sum())
    if count == 0:
        return None, {"move_rows": 0.0}
    look, t = look[rows].long(), table[rows].float()
    fin = torch.isfinite(t)
    k = fin.sum(-1)
    valid = (look >= 0) & (k > 0)
    mean = torch.where(valid, torch.where(fin, t, 0.0).sum(-1) / k.clamp(min=1), 0.0)
    d = t[:, :, None, :] - t[:, None, :, :]
    dfin = torch.isfinite(d)
    nd = dfin.sum(-1)
    dm = torch.where(dfin, d, 0.0).sum(-1) / nd.clamp(min=1)
    dv = torch.where(dfin, (d - dm[..., None]) ** 2, 0.0).sum(-1) / (nd - 1).clamp(min=1)
    ok = valid[:, :, None] & valid[:, None, :] & (nd > 0) & ~torch.eye(look.shape[1], dtype=torch.bool, device=look.device)[None]
    pairs = torch.where(ok & (dm.abs() >= LOOK_POINTS) & (dm * dm * nd > LOOK_Z * LOOK_Z * dv), torch.sign(dm), 0.0)
    q = move[rows].float().gather(1, look.clamp(min=0))
    v = valid.float()
    number = (F.huber_loss(q, mean / 100.0, reduction="none", delta=0.1) * v).sum() / v.sum().clamp(min=1)
    dq = (q[:, :, None] - q[:, None, :]) * (100.0 / LOOK_POINTS)
    pm = (pairs != 0).float()
    order = (F.softplus(-dq * pairs) * pm).sum() / pm.sum().clamp(min=1)
    return number + order, {"move_number": float(number.detach()), "move_order": float(order.detach()), "move_pairs": float(pm.sum() / 2),
                            "move_rows": float(count)}


def batch_loss(out: dict, b: dict, method: str, *, lessons: int = 0, trail_weight: float = 0.1, critic: bool = False,
               hold: float = 0.0, kl_beta: float = KL_BETA, end_rows: torch.Tensor | None = None) -> tuple[torch.Tensor, dict, dict]:
    """The loss of a training batch -> (the total, its terms, PPO's and the critic's numbers).

    `method` (`config.METHODS`): "teacher" - cross-entropy to `policy` at every row; "imitate" - the
    same weighted by `score_by_table`; "ppo" - `ppo_policy` on the moves of games, the batch's last
    `lessons` rows being lessons (cross-entropy). PPO's target is `adv_td` where it is finite, else
    `advantage` (the reward), plus `hand_penalty` where the batch has it; with `critic` the critic
    head (`adv`) is the baseline and learns `advantage`. The move head learns from the rows that
    keep a look's playouts (`look`, `look_table`). `end_rows`: as `forecast_losses`."""
    weights = {**WEIGHTS, "trail": trail_weight}
    stats: dict = {}
    if method == "ppo":
        rows = torch.ones_like(b["advantage"])
        if lessons:
            rows[-lessons:] = 0.0
        target = b["advantage"].float()
        if "adv_td" in b:
            td = b["adv_td"].float()
            have = torch.isfinite(td)
            target = torch.where(have, td, target)
            stats["td_rows"] = float(have.float().mean())
        if "hand_penalty" in b:
            target = target + b["hand_penalty"].float()
        baseline = out["adv"].detach() if critic else None
        policy, stats_ppo = ppo_policy(out["logits"], b, target, rows, hold, kl_beta, baseline)
        stats.update(stats_ppo)
    else:
        ce = -(b["policy"].float() * F.log_softmax(out["logits"], dim=-1)).sum(-1)
        if method == "imitate":
            ce = ce * score_by_table(b["score"][:, 0], b["n_players"])
        policy = ce.mean()
    parts = {"policy": policy, **forecast_losses(out, b, end_rows, trail="trail" in out)}
    total = sum(weights[k] * v for k, v in parts.items())
    if method == "ppo" and critic:
        crit, crit_stats = critic_loss(out["adv"], b["advantage"], rows)
        total = total + CRITIC_WEIGHT * crit
        stats.update(crit_stats)
    if "move" in out and "look" in b:
        mv, mv_stats = move_loss(out["move"], b["look"], b["look_table"])
        if mv is not None:
            parts["move"] = mv
            total = total + MOVE_WEIGHT * mv
        stats.update(mv_stats)
    return total, parts, stats


def adapt_kl_beta(beta: float, kl: float, target: float = KL_TARGET, lo: float = 1e-3, hi: float = 1e3) -> float:
    """PPO's adaptive KL penalty (Schulman et al. 2017, section 4): its weight up by half when the
    measured KL is more than 1.5 times the target, down by a third when under the target / 1.5."""
    if kl > 1.5 * target:
        beta *= 1.5
    elif kl < target / 1.5:
        beta /= 1.5
    return min(hi, max(lo, beta))


def adapt_entropy_hold(weight: float, entropy: float, target: float, lo: float = 1e-3, hi: float = 1.0) -> float:
    """The weight of the push against a wider policy that holds its entropy at the phase's start:
    up by a fifth (from `lo` when off) while the entropy is more than 3 % above the target, down by
    a sixth while more than 3 % below it, and off once under `lo`. One-sided: it never widens."""
    if entropy > target * 1.03:
        weight = max(weight * 1.2, lo)
    elif entropy < target * 0.97:
        weight /= 1.2
        if weight < lo:
            weight = 0.0
    return min(hi, weight)


def warm_lr(lr: float, step: int, fresh_at: int, warmup: int = 500) -> float:
    """The learning rate at `step` of an optimizer that started fresh at step `fresh_at`: up from
    lr / warmup to lr over `warmup` steps, since a fresh Adam's first steps move every weight by
    about the rate whatever its gradient."""
    return lr * min(1.0, (step - fresh_at + 1) / warmup)
