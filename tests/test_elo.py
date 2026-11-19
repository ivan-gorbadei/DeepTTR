"""The rating fit: known ratings come back from their games, the anchor stays at 0."""
import json
import math

import numpy as np

from deepttr import elo


def games_of(true: dict, n: int = 4000, skip=()) -> list[dict]:
    rng = np.random.default_rng(0)
    out, names = [], list(true)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if {x, y} in skip:
                continue
            p = 1 / (1 + 10 ** ((true[y] - true[x]) / 400))
            w = int(rng.binomial(n, p))
            out.append({"a": x, "b": y, "a_wins": w, "a_losses": n - w, "games": n})
    return out


def test_fit_recovers_known_ratings():
    true = {"heuristic": 0.0, "a": 200.0, "b": -300.0, "c": 450.0}
    r = elo.fit(games_of(true, skip=[{"b", "c"}]), prior_draws=0.5)  # 750 apart: no games between them
    assert r["heuristic"]["elo"] == 0.0
    for k, v in true.items():
        assert abs(r[k]["elo"] - v) < 15, (k, r[k])
        assert r[k]["lo"] <= r[k]["elo"] <= r[k]["hi"]


def test_the_anchor_can_be_any_player():
    true = {"x": 0.0, "y": 150.0, "z": -100.0}
    r = elo.fit(games_of(true), anchor="y", prior_draws=0.0)
    assert r["y"] == {"elo": 0.0, "lo": 0.0, "hi": 0.0, "games": 8000}
    assert abs(r["x"]["elo"] + 150) < 15 and abs(r["z"]["elo"] + 250) < 15


def test_draws_count_as_half_a_win():
    r = elo.fit([{"a": "x", "b": "heuristic", "a_wins": 0, "a_losses": 0, "games": 100}])
    assert abs(r["x"]["elo"]) < 1e-6


def test_a_shutout_stays_finite_and_wide():
    r = elo.fit([{"a": "x", "b": "heuristic", "a_wins": 400, "a_losses": 0, "games": 400}])
    assert math.isfinite(r["x"]["elo"]) and r["x"]["elo"] > 400
    assert r["x"]["hi"] - r["x"]["lo"] > 200  # a shutout measures little
    assert elo.fit([]) == {"heuristic": {"elo": 0.0, "lo": 0.0, "hi": 0.0, "games": 0}}


def test_a_file_of_matches_gives_a_table(tmp_path, capsys, monkeypatch):
    lines = [{"a": "net", "b": "hoard", "a_wins": 300, "a_losses": 100, "games": 400, "players": 2},
             {"a": "hoard", "b": "heuristic", "a_wins": 380, "a_losses": 20, "games": 400},
             {"a": "net", "b": "hoard", "a_wins": 0, "a_losses": 800, "games": 800, "players": 4}]
    path = tmp_path / "games.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines) + "\n", encoding="utf-8")
    assert len(elo.read_games(path)) == 2, "one table size a scale"
    assert len(elo.read_games(path, None)) == 3
    monkeypatch.setattr("sys.argv", ["elo", str(path), "--out", str(tmp_path / "ratings.json")])
    elo.main()
    printed = capsys.readouterr().out.splitlines()
    assert [row.split()[0] for row in printed] == ["net", "hoard", "heuristic"]
    saved = json.loads((tmp_path / "ratings.json").read_text(encoding="utf-8"))
    assert saved["anchor"] == "heuristic" and saved["matches"] == 2
    assert saved["players"]["net"]["elo"] > saved["players"]["hoard"]["elo"] > 0 == saved["players"]["heuristic"]["elo"]
