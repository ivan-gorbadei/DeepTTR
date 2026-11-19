"""The championship of a run's phase ends and the bots."""
import json


def test_every_pair_meets_once_and_a_stopped_championship_goes_on_where_it_stopped(engine, stub, tmp_path):
    from deepttr import championship

    players = {"phase 1": stub(), "phase 2": stub(0.5), **championship.BOTS}
    seen = []
    matches = championship.play_all(tmp_path, players, deals=2, log=seen.append)
    assert len(matches) == 6 and len(seen) == 6
    assert {frozenset((m["a"], m["b"])) for m in matches} == {frozenset(p) for p in
                                                             [("phase 1", "phase 2"), ("phase 1", "ticket builder"),
                                                              ("phase 1", "card hoarder"), ("phase 2", "ticket builder"),
                                                              ("phase 2", "card hoarder"), ("ticket builder", "card hoarder")]}
    assert all(m["games"] == 4 for m in matches)
    lines = (tmp_path / "championship.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["a"] == "phase 1"
    again = championship.play_all(tmp_path, players, deals=2, log=seen.append)
    assert len(again) == 6 and len(seen) == 6, "nothing is played twice"
    rated = championship.ratings(again)
    assert rated["ticket builder"]["elo"] == 0 and set(rated) == set(players)


def test_the_players_are_the_ended_phases_in_order_and_the_bots(tmp_path):
    from deepttr import championship

    for n in (2, 1, 11):
        (tmp_path / f"phase{n:02d}").mkdir()
        (tmp_path / f"phase{n:02d}" / "end.pt").write_bytes(b"")
    (tmp_path / "phase03").mkdir()  # a phase under way has no end yet
    assert list(championship.players_of(tmp_path)) == ["phase 1", "phase 2", "phase 11", "ticket builder", "card hoarder"]
