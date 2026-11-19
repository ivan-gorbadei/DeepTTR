"""The phases' end rules and the championship's places."""
import json

import pytest


def write(run, name, lines):
    (run / name).write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")


def test_the_first_phase_ends_when_its_players_build_the_share_of_their_starting_tickets(tmp_path):
    from deepttr.curriculum import verdict

    end = {"rule": "built", "share": 0.975, "most_games": 1000}
    line = lambda step, done, failed, games=10: {"event": "train", "step": step, "g_games": games,  # noqa: E731
                                                 "g_player_completed_start": done, "g_player_failed_start": failed}
    write(tmp_path, "log.jsonl", [line(100, 90, 10)])
    assert verdict(tmp_path, end) == (None, {})
    write(tmp_path, "log.jsonl", [line(100, 90, 10), line(200, 98, 2)])
    assert verdict(tmp_path, end) == ("ended", {"built": 0.98, "step": 200})
    write(tmp_path, "log.jsonl", [line(100, 90, 10, games=600), line(200, 90, 10, games=600)])
    assert verdict(tmp_path, end)[0] == "failed", "short of the share after the most games it may play"


def test_a_phase_of_steps_ends_at_its_steps_and_one_of_price_at_full_price(tmp_path):
    from deepttr.curriculum import trained, verdict

    write(tmp_path, "log.jsonl", [{"event": "train", "step": 900}])
    assert verdict(tmp_path, {"rule": "steps", "steps": 1000})[0] is None
    write(tmp_path, "log.jsonl", [{"event": "train", "step": 900}, {"event": "end", "step": 950, "reason": "stop"}])
    assert verdict(tmp_path, {"rule": "steps", "steps": 1000})[0] is None, "a stop short of the steps is no end"
    write(tmp_path, "log.jsonl", [{"event": "train", "step": 900}, {"event": "end", "step": 1000, "reason": "steps"}])
    assert verdict(tmp_path, {"rule": "steps", "steps": 1000}) == ("ended", {"steps": 1000})
    write(tmp_path, "log.jsonl", [{"event": "start", "at": "2026-11-19T15:00:00"}, {"event": "end", "step": 200, "reason": "steps"}])
    assert trained(tmp_path)["steps"] == 200, "a phase too short for a training line counts the steps of its end"
    assert verdict(tmp_path, {"rule": "full_price"}) == (None, {})
    (tmp_path / "full_price.json").write_text('{"step": 28750}', encoding="utf-8")
    assert verdict(tmp_path, {"rule": "full_price"}) == ("ended", {"step": 28750})


def test_a_plateau_ends_the_phase_once_the_last_measures_rise_no_more_than_the_gain(tmp_path):
    from deepttr.curriculum import goals, verdict

    def snapshot(step, share):  # two opponents, both at the same share; a first-choice line is not counted
        return [{"step": step, "move": "sample", "games": 100, "first_places": int(100 * share), "opponent": o} for o in "ab"] + [
            {"step": step, "move": "first", "games": 100, "first_places": 0, "opponent": "a"}]

    rising = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54]
    write(tmp_path, "measures.jsonl", [x for i, s in enumerate(rising) for x in snapshot(i * 2000, s)])
    end = {"rule": "plateau", "n": 4, "gain": 1.0}
    assert verdict(tmp_path, end, cells=2) == (None, {}), "still rising 8 points"
    flat = rising + [0.53, 0.54, 0.53, 0.54]
    write(tmp_path, "measures.jsonl", [x for i, s in enumerate(flat) for x in snapshot(i * 2000, s)])
    assert verdict(tmp_path, end, cells=2) == (None, {}), "the last four 2.5 points above the four before"
    flat += [0.53, 0.55, 0.53, 0.54]
    write(tmp_path, "measures.jsonl", [x for i, s in enumerate(flat) for x in snapshot(i * 2000, s)])
    assert verdict(tmp_path, end, cells=2) == ("ended", {"goal": 53.75, "rise": 0.25, "checks": 16})
    write(tmp_path, "measures.jsonl", [x for i, s in enumerate(flat) for x in snapshot(i * 2000, s)][:-3])
    assert len(goals(tmp_path, 2)) == 15, "a snapshot with one of its cells missing is not counted yet"
    with pytest.raises(ValueError):
        verdict(tmp_path, {"rule": "no such rule"})


def test_every_recipe_makes_a_phase_that_fits_together_on_the_ends_of_the_phases_before(tmp_path, monkeypatch):
    from deepttr import curriculum

    monkeypatch.setattr(curriculum, "measured_wins", lambda root, start, names, device: {(k, t): 0.5 for k in names for t in (2, 3, 4, 5)})
    steps = {}
    for n in range(1, 12):
        rec = curriculum.recipe(n)
        assert rec["phase"] == n and rec["name"] and len(rec["about"].split()) > 15
        cfg, chosen = curriculum.config_of(tmp_path, rec, (), None)
        assert cfg.init == (None if n == 1 else str(tmp_path / f"phase{n - 1:02d}" / "end.pt"))
        assert all(o.bot or o.player.endswith("end.pt") for o in cfg.league)
        if chosen:  # a championship phase: the networks share out the table's total by their places
            totals = curriculum.recipe(n)["places"]["totals"]
            for table, total in totals.items():
                nets = sum(o.shares[int(table)] for o in cfg.league if not o.bot)
                assert abs(nets - total) < 1e-3
        plan = curriculum.measure_plan(cfg, rec)
        assert plan["matches"][0]["tables"] == [2] and len(plan["matches"]) == (2 if cfg.league else 1)
        steps[n] = cfg.steps
        (tmp_path / f"phase{n:02d}").mkdir()
        (tmp_path / f"phase{n:02d}" / "end.pt").write_bytes(b"")
    assert steps == {1: 0, 2: 38_994, 3: 57_177, 4: 0, 5: 0, 6: 0, 7: 39_989, 8: 89_577, 9: 502_512, 10: 31_883, 11: 97_894}
    with pytest.raises(SystemExit):
        curriculum.config_of(tmp_path / "empty", curriculum.recipe(5), (), None)  # phase 4 has not ended there


def test_a_phase_ends_with_its_last_snapshot_or_at_full_price_with_that_steps(tmp_path):
    from deepttr.curriculum import end_network

    (tmp_path / "snapshots").mkdir()
    for step in (0, 2000, 12000):
        (tmp_path / "snapshots" / f"{step:08d}.pt").write_bytes(b"")
    assert end_network(tmp_path, {"end": {"rule": "steps"}}).name == "00012000.pt"
    (tmp_path / "full_price.json").write_text(json.dumps({"step": 2000, "end": str(tmp_path / "snapshots" / "00002000.pt")}),
                                              encoding="utf-8")
    assert end_network(tmp_path, {"end": {"rule": "full_price"}}).name == "00002000.pt"


def test_the_places_follow_one_minus_the_share_won_squared_within_each_table(tmp_path):
    from deepttr.curriculum import places, training_wins

    write(tmp_path, "log.jsonl", [
        {"event": "train", "step": 1, "league": {"phase 5@2": {"g_games": 100, "g_won": 0.9}, "phase 6@2": {"g_games": 100, "g_won": 0.5}}},
        {"event": "train", "step": 2, "league": {"phase 5@2": {"g_games": 300, "g_won": 0.7}, "phase 6@3": {"g_games": 50, "g_won": 0.2}}}])
    wins = training_wins(tmp_path)
    assert wins == {("phase 5", 2): pytest.approx(0.75), ("phase 6", 2): 0.5, ("phase 6", 3): 0.2}
    shares = places(["phase 5", "phase 6", "phase 9"], wins, {2: 0.13, 3: 0.2})
    w2 = {"phase 5": 0.25 ** 2, "phase 6": 0.5 ** 2, "phase 9": 0.5 ** 2}  # phase 9 not met yet: even, 1/2
    assert shares["phase 5"][2] == round(0.13 * w2["phase 5"] / sum(w2.values()), 4)
    assert sum(s[3] for s in shares.values()) == pytest.approx(0.2, abs=1e-3)
    assert shares["phase 6"][3] > shares["phase 5"][3], "the network loses more to phase 6 at three: it meets it more"
