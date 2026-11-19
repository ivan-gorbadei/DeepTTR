"""A phase's configuration: its JSON round trip, and the settings that do not fit together."""
import json

import pytest

from deepttr.train.config import Opponent, TrainConfig

LEAGUE = (Opponent("card hoarder", "hoard", {2: 0.005, 3: 0.005}), Opponent("phase 5", "runs/five/latest.pt", {2: 0.1, 5: 0.3}))


def test_a_config_round_trips_through_json(tmp_path):
    cfg = TrainConfig(init="a.pt", steps=900, league=LEAGUE, critic=True, td=True, bonus=True, mixed_tables=True, devices=("cuda:0", "cuda:1"))
    cfg.check().save(tmp_path / "config.json")
    text = json.loads((tmp_path / "config.json").read_text())
    assert text["league"][1] == {"name": "phase 5", "player": "runs/five/latest.pt", "shares": {"2": 0.1, "5": 0.3}}
    again = TrainConfig.load(tmp_path / "config.json")
    assert again == cfg and again.league[0].bot and not again.league[1].bot
    assert TrainConfig.from_dict({}) == TrainConfig()
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"ppo_clip": 0.2})


@pytest.mark.parametrize("settings", [
    {"method": "search"},
    {"td": True},
    {"method": "teacher", "critic": True},
    {"method": "imitate", "ticket_playouts": True},
    {"method": "teacher", "league": LEAGUE},
    {"bonus_ramp": True},
    {"move_forecast": True},
    {"lookahead": True, "look_league": True},
    {"mixed_tables": True, "league": LEAGUE[1:]},
    {"price_step": 0.2},
    {"price_step": 0.2, "price_steps": 100},
    {"tables": (2, 6)},
    {"league": (Opponent("mixed", "hoard", {2: 0.1}),)},
    {"league": (Opponent("phase@5", "x.pt", {2: 0.1}),)},
    {"league": (Opponent("a", "hoard", {2: 0.6}), Opponent("b", "x.pt", {2: 0.4}))},
    {"tables": (2, 3), "league": (Opponent("a", "x.pt", {4: 0.1}),)},
])
def test_settings_that_do_not_fit_together(settings):
    with pytest.raises(ValueError):
        TrainConfig(**settings).check()


def test_the_phases_kinds_of_settings_fit():
    TrainConfig(method="teacher", ticket_playouts=True, tables=(2,), outcome_heads="global", trail_weight=0.0).check()
    TrainConfig(method="imitate", rules={"route_price": 0.0}, price_step=0.2, price_steps=5750).check()
    TrainConfig(league=LEAGUE, critic=True, td=True, bonus=True, bonus_ramp=True, hand_penalty=True).check()
    TrainConfig(league=LEAGUE, critic=True, td=True, bonus=True, lookahead=True, look_league=True, mixed_tables=True, keep_check=96,
                move_forecast=True, hand_penalty=True).check()
