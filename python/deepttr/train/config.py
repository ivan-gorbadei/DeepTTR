"""What a phase of training sets.

`TrainConfig` holds the settings that differ between the phases; what every phase that uses a
mechanism shares is a constant of the module that implements it. `Opponent` is a player of the
league. A run directory keeps its configuration as `config.json` (`TrainConfig.save`, `load`).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

BOTS = ("heuristic", "hoard")
METHODS = ("teacher", "imitate", "ppo")
TABLES = (2, 3, 4, 5)
MIXED_TABLES = (3, 4, 5)  # the tables of several opponents (`TrainConfig.mixed_tables`)
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]*")
RESERVED = ("self", "mixed", "look")


@dataclass
class Opponent:
    """A player of the league: a bot of the engine (`hoard`, `heuristic`) or a frozen network (the
    path of a checkpoint), with its share of a worker's games at each table size (a size it does
    not name: none). `name` (letters, digits, spaces, '-', '_') names the opponent's lines in the
    log, `<name>@<table>`, and its shards."""

    name: str
    player: str
    shares: dict[int, float]

    @property
    def bot(self) -> bool:
        return self.player in BOTS


@dataclass
class TrainConfig:
    """A phase of training. The defaults are those of the policy-gradient phases with every
    optional mechanism off."""

    # the network the phase starts from, a checkpoint; None: random weights, the published size
    init: str | None = None
    # the phase's training steps; 0: until the run directory holds a file STOP
    steps: int = 0

    # the games
    rules: dict = field(default_factory=dict)  # the settings that differ from the official rules (`_engine.Game`)
    tables: tuple[int, ...] = TABLES  # self-play's table sizes, each as often
    games: int = 1024  # the games a worker plays at once
    # how the policy learns: "teacher" - the scripted teacher's policy at every decision; "imitate" -
    # the moves of its own games, weighted by how far the player's score was above the usual at its
    # table size; "ppo" - PPO on each move's reward, in a league of opponents on duplicate deals
    method: str = "ppo"
    # with the teacher: a building lesson follows one joint plan for all the open tickets, and every
    # choice of tickets is played out from each of its options, the best option being the lesson
    ticket_playouts: bool = False
    league: tuple[Opponent, ...] = ()  # the opponents of the policy-gradient phases
    # at the tables of `MIXED_TABLES` every other seat is drawn from the league, by the
    # opponents' shares at that table, and the card hoarder sits at one seat in half the games
    mixed_tables: bool = False
    # the look-ahead teacher: a share of the learner's decisions and every choice of tickets are
    # played out, and a move that proves better than the first choice becomes a lesson
    lookahead: bool = False
    look_league: bool = False  # the look-ahead's games against the league, not against itself
    keep_check: int = 24  # the fresh worlds that check a choice of tickets the look-ahead found
    # the move head: its forecast sends a promising move to the look-ahead's check at once, and
    # every look's playouts teach it
    move_forecast: bool = False
    # a penalty on the PPO target of the first move of a turn that ends with a large hand
    # (`losses.hand_penalty`)
    hand_penalty: bool = False

    # the network and its training
    outcome_heads: str = "tokens"  # what the value, score and critic heads read: "global" or "tokens"
    window: int = 100_000  # the newest positions trained on
    ratio: float = 2.0  # training samples per new position
    lr: float = 5e-5
    trail_weight: float = 0.1  # the weight of the longest trail's forecast in the loss
    critic: bool = False  # the critic head learns each game's reward and is PPO's baseline
    td: bool = False  # PPO's target is the TD(lambda) return over the critic's forecasts
    bonus: bool = False  # the reward's bonuses for the own points and the tickets completed
    bonus_ramp: bool = False  # the bonuses come in over the first steps (`losses.bonus_shares`)
    # the route price rises by `price_step` every `price_steps` steps from the rules' price; the
    # network of the step it becomes whole ends the phase
    price_step: float = 0.0
    price_steps: int = 0
    save_every: int = 50  # steps between two networks for the workers
    snapshot_steps: int = 2000  # steps between two snapshots

    # where it runs, not part of a phase: devices (none: every CUDA card, else Apple's MPS, else the
    # CPU; one self-play worker each, the trainer on the first), the engine's threads a pool (0:
    # `worker.THREADS`, or the cores if fewer), `torch.compile` (None: on a CUDA card with Triton)
    devices: tuple[str, ...] = ()
    threads: int = 0
    compile: bool | None = None

    def check(self) -> TrainConfig:
        """Itself, if its settings fit together; else ValueError."""
        def need(ok: bool, what: str) -> None:
            if not ok:
                raise ValueError(f"TrainConfig: {what}")

        need(self.method in METHODS, f"method one of {METHODS}, not {self.method!r}")
        need(self.tables and set(self.tables) <= set(TABLES), f"tables among {TABLES}")
        need(self.outcome_heads in ("global", "tokens"), "outcome_heads 'global' or 'tokens'")
        need(self.steps >= 0 and self.games >= 2 and self.window > 0 and self.ratio > 0 and self.lr > 0, "positive sizes")
        need(self.save_every > 0 and self.snapshot_steps > 0, "positive step counts")
        need(not self.ticket_playouts or self.method == "teacher", "ticket_playouts with the teacher")
        ppo = self.method == "ppo"
        for name, on in (("league", bool(self.league)), ("critic", self.critic), ("bonus", self.bonus),
                         ("lookahead", self.lookahead), ("hand_penalty", self.hand_penalty)):
            need(not on or ppo, f"{name} with method 'ppo'")
        need(not self.td or self.critic, "td with the critic")
        need(not self.bonus_ramp or self.bonus, "bonus_ramp with bonus")
        need(not self.look_league or (self.lookahead and self.league), "look_league with lookahead and a league")
        need(not self.move_forecast or self.lookahead, "move_forecast with lookahead")
        need(not self.mixed_tables or (self.league and set(MIXED_TABLES) <= set(self.tables)),
             f"mixed_tables with a league at tables {MIXED_TABLES}")
        need(self.keep_check >= 2, "keep_check of 2 worlds or more")
        need((self.price_step > 0) == (self.price_steps > 0), "price_step and price_steps together")
        need(not self.price_steps or float(self.rules.get("route_price", 1.0)) < 1.0, "a price ramp from a route_price below 1")
        need(not self.mixed_tables or any(o.player == "hoard" for o in self.league), "mixed_tables with the card hoarder in the league")
        spelt = {o.name.replace(" ", "-").lower() for o in self.league}
        need(len(spelt) == len(self.league), "the league's names differ, also with '-' for a space and in any case")
        for o in self.league:
            need(bool(NAME.fullmatch(o.name)) and o.name == o.name.strip() and o.name not in RESERVED,
                 f"an opponent's name of letters, digits, spaces, '-' and '_', not {o.name!r}")
            need(set(o.shares) <= set(self.tables) and all(s >= 0 for s in o.shares.values()),
                 f"{o.name}: shares at the run's tables")
        for n in self.tables:
            need(sum(o.shares.get(n, 0.0) for o in self.league) < 1.0, f"the league's shares at a table of {n} below 1")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"TrainConfig: unknown settings {sorted(unknown)}")
        d = dict(d)
        if "league" in d:
            d["league"] = tuple(o if isinstance(o, Opponent) else
                                Opponent(o["name"], o["player"], {int(k): float(v) for k, v in o["shares"].items()})
                                for o in d["league"])
        for k in ("tables", "devices"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
