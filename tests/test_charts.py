"""The chart page and the static charts of a training run (deepttr.charts), on a small run root
written here: two phases and the ratings."""
import json
import re
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer

import pytest

from deepttr.charts import data, export
from deepttr.charts.server import Run, handler

T0 = datetime(2026, 1, 1, 8, 0, 0)


def at(minutes: float) -> str:
    return (T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S")


def write_lines(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def train(minutes, step, games, **fields) -> dict:
    return {"at": at(minutes), "by": "trainer", "event": "train", "step": step, "g_games": games, **fields}


def pool(won, games=300, score=100.0, tickets=40.0) -> dict:
    return {"g_games": games, "g_won": won, "g_player_score": score, "g_player_ticket_points": tickets}


def cell(step, games_so_far, opponent, players, move, first, games=400) -> dict:
    return {"at": at(step / 10), "step": step, "train_games": games_so_far, "opponent": opponent, "players": players,
            "move": move, "deals": 100, "games": games, "first_places": first, "points": 120.0, "opponent_points": 110.0}


@pytest.fixture
def root(tmp_path):
    one = tmp_path / "phase01"
    one.mkdir()
    write_lines(one / "log.jsonl", [
        {"at": at(0), "by": "trainer", "event": "start"},
        train(5, 100, 1000, g_player_completed_start=1.5, g_player_failed_start=0.5, g_player_tickets=2.0),
        train(10, 200, 1000, g_player_completed_start=1.8, g_player_failed_start=0.2, g_player_tickets=2.1),
        {"at": at(12), "by": "trainer", "event": "end", "step": 240},
        {"at": at(120), "by": "trainer", "event": "resume", "step": 240},  # stopped for 108 minutes
        train(125, 300, 1000, g_player_completed_start=1.9, g_player_failed_start=0.1, g_player_tickets=2.2),
    ])
    (one / "phase.json").write_text(json.dumps({"phase": 1, "name": "Completing a ticket", "about": "Two players.", "rule": "built",
                                                "verdict": "success", "numbers": {"built": 0.95, "threshold": 0.9},
                                                "steps": 300, "games": 3000, "hours": 0.29}), encoding="utf-8")
    (one / "recipe.toml").write_text('teacher = 0.1\n', encoding="utf-8")
    five = tmp_path / "phase05"
    five.mkdir()
    league = lambda w: {"card hoarder@2": pool(w), "card hoarder@3": pool(w), "phase 4@2": pool(w - 0.1), "phase 4@3": pool(w - 0.1)}  # noqa: E731
    write_lines(five / "log.jsonl", [
        {"at": at(0), "by": "trainer", "event": "start"},
        train(5, 50, 10, league=league(0.5)),
        train(10, 100, 10, league=league(0.6)),
        {"at": at(11), "by": "worker0", "event": "selfplay", "look_games": 10, "look_proven": 4, "look_checked": 20},
    ])
    write_lines(five / "measures.jsonl", [
        cell(50, 610, "card hoarder", 2, "sample", 200), cell(50, 610, "card hoarder", 3, "sample", 120),
        cell(50, 610, "phase 4", 2, "sample", 180), cell(50, 610, "phase 4", 3, "sample", 100),
        cell(100, 1220, "card hoarder", 2, "sample", 240),  # a snapshot not measured at every cell yet
        cell(50, 610, "card hoarder", 2, "first", 220), cell(50, 610, "ticket builder", 2, "first", 380),
    ])
    (tmp_path / "elo.json").write_text(json.dumps({"anchor": "ticket builder", "matches": 3, "players": {
        "phase 5": {"elo": 600.0, "lo": 560.0, "hi": 640.0, "games": 800}, "phase 1": {"elo": -400.0, "lo": -450.0, "hi": -350.0, "games": 800},
        "card hoarder": {"elo": 500.0, "lo": 470.0, "hi": 530.0, "games": 800}, "ticket builder": {"elo": 0.0, "lo": 0.0, "hi": 0.0, "games": 800}}}),
        encoding="utf-8")
    return tmp_path


def chart(view: dict, cid: str) -> dict:
    return next(c for c in view["charts"] if c["id"] == cid)


def test_hours_leave_out_a_stop_and_games_count_the_league(root):
    views = data.load(root)["views"]
    built = chart(views["1"], "built")
    pts = built["series"][0]["points"]
    assert [p[1] for p in pts] == [100, 200, 300] and [p[0] for p in pts] == [1000, 2000, 3000]
    assert pts[-1][2] == pytest.approx(17 / 60, abs=1e-3)  # 12 minutes, the stop of 108 left out, 5 more
    assert [p[3] for p in pts] == pytest.approx([75.0, 90.0, 95.0])
    assert built["refs"] == [{"y": 90.0, "label": "the phase's threshold"}]
    assert views["1"]["end"] == {"n": 3000, "s": 300, "h": pytest.approx(17 / 60, abs=1e-3)}
    games = [p[0] for p in chart(views["5"], "score2")["series"][0]["points"]]
    assert games == [1210, 2420]  # the self-play's 10 and the league's 1,200 a line


def test_the_goal_is_the_mean_of_complete_snapshots_with_its_interval(root):
    goal = chart(data.load(root)["views"]["5"], "goal")
    (p,) = goal["series"][0]["points"]  # step 100 lacks three of its four cells
    shares = [200 / 400, 120 / 400, 180 / 400, 100 / 400]
    mean = sum(shares) / 4
    se = sum(s * (1 - s) / 400 for s in shares) ** 0.5 / 4
    assert p[:4] == [610, 50, pytest.approx(5 / 60, abs=1e-3), pytest.approx(100 * mean)]
    assert p[4:] == pytest.approx([100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se)], abs=1e-3)
    assert goal["refs"][0]["y"] == pytest.approx(100 * (1 / 2 + 1 / 3) / 2)


def test_an_opponent_gets_its_chart_by_table(root):
    view = data.load(root)["views"]["5"]
    ids = [c["id"] for c in view["charts"]]
    assert ids[:3] == ["goal", "vs-card-hoarder", "vs-phase-4"]  # the bots first, then the phase ends
    vs = chart(view, "vs-card-hoarder")
    by = {s["name"]: [p[3] for p in s["points"]] for s in vs["series"]}
    assert by["table of 2"] == pytest.approx([50.0, 60.0]) and by["mean of the tables, training"] == pytest.approx([50.0, 60.0])
    assert len(by["mean of the tables, measured"]) == 1


def test_the_curriculum_lays_the_phases_end_to_end(root):
    cur = data.load(root)["views"]["curriculum"]
    one, five = cur["bands"]
    assert (one["phase"], five["phase"]) == (1, 5) and five["from"] == one["to"] and one["to"]["n"] == 3000
    bots = chart(cur, "bots")
    hoarder = next(s for s in bots["series"] if s["name"] == "against the card hoarder")
    assert hoarder["points"][0][0] == 3000 + 610 and hoarder["points"][0][3] == pytest.approx(55.0)
    lo, hi = data.wilson(220, 400)
    assert hoarder["points"][0][4:] == pytest.approx([100 * lo, 100 * hi], abs=1e-3)
    elo = chart(cur, "elo")
    assert elo["kind"] == "phases" and [p[0] for p in elo["series"][0]["points"]] == [1, 5]
    assert {r["label"] for r in elo["refs"]} == {"card hoarder", "ticket builder"}


def test_a_line_being_written_is_passed_over(root):
    log = root / "phase05" / "log.jsonl"
    log.write_text(log.read_text(encoding="utf-8") + '{"at": "2026-01-01T08:20:00", "event": "tra', encoding="utf-8")
    assert len(data.load(root)["views"]["5"]["charts"]) > 1
    assert data.wilson(50, 100) == pytest.approx((0.4038, 0.5962), abs=1e-4)


def test_the_server_serves_the_page_its_data_and_pictures(root):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(Run(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    get = lambda path: urllib.request.urlopen(base + path, timeout=10).read().decode()  # noqa: E731
    try:
        page = get("/")
        assert "<canvas" not in page and "drawPanel" in page  # canvases are made by the page's own script
        assert not re.search(r"""(src|href)=["']?(https?:)?//""", page)  # nothing loaded from elsewhere
        d = json.loads(get("/data?view=5"))
        assert [p["phase"] for p in d["phases"]] == [1, 5] and d["view"]["header"]["name"] == "Phase 5"
        assert json.loads(get("/data"))["view"]["view"] == "curriculum"
        svg = get("/svg?view=5&chart=goal&x=h&theme=dark&smooth=0.3")
        assert svg.startswith("<svg") and "prefers-color-scheme" not in svg
        for bad, code in (("/data?view=7", 404), ("/svg?view=5&chart=none", 404), ("/svg?view=5&chart=goal&x=q", 400)):
            with pytest.raises(urllib.error.HTTPError) as err:
                get(bad)
            assert err.value.code == code
    finally:
        httpd.shutdown()


def test_the_static_charts_need_nothing_from_elsewhere(root, tmp_path):
    written = export.write_all(root, tmp_path / "out")
    assert sorted(p.name for p in written) == ["curriculum-bots.svg", "curriculum-elo.svg", "phase01-built.svg", "phase05-goal.svg",
                                               "phase05-vs-card-hoarder.svg"]
    for path in written:
        text = path.read_text(encoding="utf-8")
        ET.fromstring(text)  # well-formed
        assert "@media (prefers-color-scheme: dark)" in text
        assert "<script" not in text and "@import" not in text and "@font-face" not in text
        assert re.findall(r"https?://[^\s\"']+", text) == ["http://www.w3.org/2000/svg"]
