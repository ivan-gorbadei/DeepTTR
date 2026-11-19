"""The look-ahead at play time: before it moves, the network tests its first choice against its
best alternatives by playing them out.

At a decision the candidates are the first choice and the best move of each other kind
(`_engine.look_options`). A first look plays every candidate to the end of the game in `worlds`
hidden worlds, dealt afresh from what the seat sees; the network plays every seat of those games
by its first choice. If some candidate ends ahead of the first choice on average, a second look
plays the two again in `check` fresh worlds, and the candidate is played only if it proves better
there: by 3 points of final margin on average and by 2 standard errors of the paired differences
(`_engine.look_better`). In every other case, and when the time runs out, the first choice stands.

It is a playout test of a few candidates, not a tree search: one decision branches, and each
branch is a whole game played by the policy. A look costs thousands of network evaluations, so it
wants a GPU; `seconds` bounds it.

`Looker(evaluator)` is a player that moves this way: `move(game)` for one game (the play server
uses it), and a contestant of `play.duplicate_match` in place of a bare evaluator.
"""
from __future__ import annotations

import random
import time

import numpy as np

from deepttr import _engine as engine

# the copies per candidate in the first look and in the second, and the seconds a move may take
WORLDS, CHECK, SECONDS = 8, 24, 8.0


def play_out(ev, looks: list, deadline: float | None = None) -> list[np.ndarray] | None:
    """Run look-aheads (`_engine.Lookahead`) to their ends together, all their waiting positions
    in one batch a step. -> each look's table, [options, worlds]; None if `deadline` (a
    `time.perf_counter` value) passes first."""
    active = list(looks)
    while active:
        waiting = [(look, n) for look in active if (n := look.prepare())]
        if not waiting:
            break
        if deadline is not None and time.perf_counter() > deadline:
            return None
        parts = [look.observations() for look, _ in waiting]
        logits = ev({k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        at = 0
        for look, n in waiting:
            look.feed(logits[at:at + n])
            at += n
        active = [look for look, _ in waiting]
    return [look.table() for look in looks]


def decide(ev, positions: list, worlds: int = WORLDS, check: int = CHECK, deadline: float | None = None) -> list[tuple[int, dict]]:
    """The moves of several decisions after looking ahead, their looks run together. A position is
    (the policy logits over every action, the legal moves, `make`), where `make(options, worlds,
    stage)` gives the look-ahead of the position; stage 1 is the second look, in other worlds.

    -> per position (the move, what the look saw): "options", the candidates, the first choice
    first; "first", each candidate's mean margin in the first look; "check", the tried candidate's
    mean lead over the first choice in the second look; "changed", whether another move than the
    first choice is played; "time", set when the deadline cut the look short."""
    options = [[int(a) for a in engine.look_options(np.ascontiguousarray(lg, dtype=np.float32), [int(a) for a in legal])]
               for lg, legal, _ in positions]
    moves = [o[0] for o in options]
    facts = [{"options": o, "changed": False} for o in options]
    looked = [i for i, o in enumerate(options) if len(o) > 1]
    tables = play_out(ev, [positions[i][2](options[i], worlds, 0) for i in looked], deadline)
    if tables is None:
        for i in looked:
            facts[i]["time"] = True
        return list(zip(moves, facts))
    tried = []  # (position, its best other candidate)
    for i, table in zip(looked, tables):
        means = table.mean(axis=1)
        facts[i]["first"] = [round(float(m), 1) for m in means]
        best = 1 + int(np.argmax(means[1:]))
        if means[best] > means[0]:
            tried.append((i, best))
    tables = play_out(ev, [positions[i][2]([options[i][0], options[i][best]], check, 1) for i, best in tried], deadline)
    if tables is None:
        for i, _ in tried:
            facts[i]["time"] = True
        return list(zip(moves, facts))
    for (i, best), table in zip(tried, tables):
        facts[i]["check"] = round(float((table[1] - table[0]).mean()), 1)
        if engine.look_better(np.ascontiguousarray(table, dtype=np.float32)):
            moves[i] = options[i][best]
            facts[i]["changed"] = True
    return list(zip(moves, facts))


class Looker:
    """A network that looks ahead before it moves. `evaluator` is the network (`play.Evaluator`);
    `worlds` and `check` are the copies per candidate in the first and in the second look;
    `seconds` bounds one decision of `move` (None: no bound). `stats` counts the decisions looked
    at, the moves the look changed and the looks cut short by the clock; `last` is what the look
    of the last `move` saw."""

    def __init__(self, evaluator, worlds: int = WORLDS, check: int = CHECK, seconds: float | None = SECONDS) -> None:
        if worlds < 1 or check < 2:
            raise ValueError("a look-ahead needs at least 1 world for the first look and 2 for the second")
        self.ev, self.worlds, self.check, self.seconds = evaluator, worlds, check, seconds
        self.stats = {"decisions": 0, "changed": 0, "timeouts": 0}
        self.last: dict | None = None

    def _count(self, facts: dict) -> None:
        self.stats["decisions"] += len(facts["options"]) > 1
        self.stats["changed"] += facts["changed"]
        self.stats["timeouts"] += bool(facts.get("time"))

    def move(self, game, seed: int | None = None) -> int:
        """The move for the player to act in `game` (an engine `Game`); -1: a pass. `seed` fixes
        the worlds of the look."""
        legal = game.legal()
        self.last = None
        if len(legal) <= 1:
            return legal[0] if legal else -1
        seed = random.getrandbits(62) if seed is None else seed
        deadline = None if self.seconds is None else time.perf_counter() + self.seconds
        logits = self.ev(game.observation(game.current))[0]
        make = lambda options, worlds, stage: engine.Lookahead(game, options, worlds, seed + stage)  # noqa: E731
        (move, self.last), = decide(self.ev, [(logits, legal, make)], self.worlds, self.check, deadline)
        self._count(self.last)
        return move

    def choose(self, arena, rows: np.ndarray, obs: dict) -> list[int]:
        """The moves for the waiting rows `rows` of a match (`_engine.Arena`), `obs` their
        observations: the looks of all the rows run together, with no clock."""
        logits = self.ev(obs)
        positions = [(logits[j], np.flatnonzero(obs["legal"][j]).tolist(),
                      lambda options, worlds, stage, row=int(row): arena.look(row, options, worlds, stage))
                     for j, row in enumerate(rows)]
        out = decide(self.ev, positions, self.worlds, self.check)
        for _, facts in out:
            self._count(facts)
        return [move for move, _ in out]
