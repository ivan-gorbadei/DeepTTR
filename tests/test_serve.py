"""The play server: whole games against an (untrained) network at tables of two to five, with and
without the record and the look-ahead."""
import json
import random
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch


@pytest.fixture(scope="module")
def checkpoint(engine, tmp_path_factory):
    from deepttr import net as netmod

    torch.manual_seed(2)
    path = tmp_path_factory.mktemp("net") / "tiny.pt"
    netmod.save(path, netmod.DeepTTRNet(engine.dims(), netmod.NetConfig(width=32, depth=1, heads=2)), step=7)
    return path


def play_to_the_end(srv, gid: str, seed: int = 0) -> dict:
    r = random.Random(seed)
    srv.wait(gid)
    state = srv.state(gid)
    while state["view"]["phase"] != "over":
        assert state["view"]["current"] == state["human_seat"] and not state["thinking"], "the network plays until the human's turn"
        srv.move(gid, r.choice(state["view"]["legal"]))
        srv.wait(gid)
        state = srv.state(gid)
    return state


def test_a_game_without_a_record_asks_no_consent_and_writes_nothing(engine, checkpoint, tmp_path):
    from deepttr.serve import Server

    srv = Server(checkpoint, "cpu", look_worlds=0, pace=0)
    assert json.loads(srv.board)["recording"] is False and srv.looker is None
    gid = srv.new("  Ada   Lovelace ", consent=False, seat=1)
    srv.wait(gid)
    state = srv.state(gid)
    assert state["name"] == "Ada Lovelace" and state["human_seat"] == 1 and state["ai"] == {"step": 7}
    assert state["players"] == 2 and state["names"] == ["AI 1", "Ada Lovelace"] and not state["thinking"]
    with pytest.raises(ValueError):
        srv.move(gid, 100000)
    end = play_to_the_end(srv, gid)
    assert len(end["view"]["final"]["total"]) == 2
    with pytest.raises(ValueError):
        srv.move(gid, 0)  # the game is over
    with pytest.raises(KeyError):
        srv.state("no-such-game")
    assert list(tmp_path.iterdir()) == []


def test_a_recorded_game_needs_consent_and_replays_from_its_record(engine, checkpoint, tmp_path):
    from deepttr.serve import Server

    srv = Server(checkpoint, "cpu", record=tmp_path / "games", look_worlds=0, pace=0)
    assert json.loads(srv.board)["recording"] is True
    with pytest.raises(ValueError):
        srv.new("x", consent=False, seat=0)
    gid = srv.new("x", consent=True, seat=None)
    end = play_to_the_end(srv, gid, seed=5)
    (line,) = (tmp_path / "games" / "games.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(line)
    assert rec["id"] == gid and rec["consent"] is True and rec["name"] == "x"
    assert rec["ai"]["checkpoint"] == "tiny.pt" and rec["ai"]["step"] == 7 and len(rec["ai"]["sha256"]) == 16
    assert rec["ai"]["look"] is None and rec["looks"] == {"decisions": 0, "changed": 0, "timeouts": 0}
    assert rec["final"]["total"] == end["view"]["final"]["total"]
    assert rec["result"] in ("human", "ai", "tie") and rec["players"] == 2 and len(rec["names"]) == 2
    assert (rec["human_score"], rec["ai_scores"]) == (rec["final"]["total"][rec["human_seat"]], [rec["final"]["total"][1 - rec["human_seat"]]])
    replay = engine.Game(players=2, seed=rec["seed"])
    for seat, action in rec["moves"]:
        assert replay.current == seat
        replay.step(action)
    assert replay.is_over() and json.loads(replay.view(0))["final"]["total"] == rec["final"]["total"]


def test_the_network_looks_ahead_before_it_answers(engine, checkpoint, tmp_path, capsys):
    from deepttr.serve import Server

    srv = Server(checkpoint, "cpu", record=tmp_path, look_worlds=2, look_check=2, look_seconds=120, pace=0)
    assert srv.about["look"] == {"worlds": 2, "check": 2, "seconds": 120}
    gid = srv.new("x", consent=True, seat=0)
    srv.wait(gid)
    looks = srv.games[gid]["meta"]["looks"]
    for _ in range(6):  # until the network has chosen its tickets and moved in its first turn
        if looks["decisions"] >= 2:
            break
        srv.move(gid, srv.state(gid)["view"]["legal"][0])
        srv.wait(gid)
    # every look played its copies to the end of the game
    assert looks["decisions"] >= 2 and looks["timeouts"] == 0 and looks["decisions"] == srv.looker.stats["decisions"]

    hurried = Server(checkpoint, "cpu", look_worlds=2, look_check=2, look_seconds=0, pace=0)
    gid = hurried.new("x", consent=False, seat=0)
    hurried.wait(gid)
    hurried.move(gid, hurried.state(gid)["view"]["legal"][0])
    hurried.wait(gid)
    looks = hurried.games[gid]["meta"]["looks"]
    assert looks["timeouts"] == looks["decisions"] > 0 and looks["changed"] == 0, "out of time, the first choice"
    capsys.readouterr()
    play_to_the_end(hurried, gid)
    assert (f"game {gid} is over: {looks['decisions']} looks ahead, {looks['changed']} changed the move, "
            f"{looks['timeouts']} ran out of time") in capsys.readouterr().out


def test_with_no_checkpoint_the_published_network_plays(engine):
    from deepttr import net as netmod
    from deepttr.serve import Server

    srv = Server(device="cpu", pace=0)
    assert srv.about["checkpoint"] == netmod.PUBLISHED.name and srv.about["step"] == 154189
    assert srv.looker is None and srv.about["look"] is None, "on a CPU the first choice, unless the look-ahead is asked for"
    assert srv.describe() == [f"network deepttr.pt, step 154189, sha256 {srv.about['sha256']}, on cpu",
                              "no look-ahead on a CPU: the network plays its first choice (--look-worlds 8 turns it on)"]
    gid = srv.new("x", consent=False, seat=0)
    for _ in range(2):
        srv.wait(gid)
        srv.move(gid, srv.state(gid)["view"]["legal"][0])
    srv.wait(gid)
    assert srv.state(gid)["view"]["moves"] >= 3, "the network has moved as well"
    with pytest.raises(SystemExit):
        Server("no-such-file.pt", "cpu")


def test_a_table_of_up_to_five_with_the_network_in_every_other_seat(engine, checkpoint):
    from deepttr.serve import Server

    srv = Server(checkpoint, "cpu", look_worlds=0, pace=0)
    gid = srv.new("Grace", consent=False, players=4)
    end = play_to_the_end(srv, gid, seed=3)
    me = end["human_seat"]
    assert end["players"] == 4 and len(end["view"]["final"]["total"]) == 4 and 0 <= me < 4
    assert end["names"][me] == "Grace" and [n for q, n in enumerate(end["names"]) if q != me] == ["AI 1", "AI 2", "AI 3"]
    assert srv.state(srv.new("x", consent=False, seat=4, players=5))["human_seat"] == 4
    for players, seat in ((1, None), (6, None), (3, 3), (2, -1), (True, None)):
        with pytest.raises(ValueError):
            srv.new("x", consent=False, seat=seat, players=players)


def test_the_network_plays_its_moves_at_a_pace_a_person_can_follow(engine, checkpoint, monkeypatch):
    import time

    from deepttr import serve
    from deepttr.serve import SHOW, Server

    # a deal in which seat 0 begins: the two seats of the network then move before the human's
    seed = next(s for s in range(100) if engine.Game(players=3, seed=s).current == 0)
    monkeypatch.setattr(serve.random, "getrandbits", lambda bits: seed)
    srv = Server(checkpoint, "cpu", look_worlds=0, pace=0.2)
    t0 = time.perf_counter()
    gid = srv.new("x", consent=False, seat=2, players=3)
    seen = [srv.state(gid)]
    while seen[-1]["thinking"]:
        time.sleep(0.05)
        seen.append(srv.state(gid))
    # the first move waits for the page to show the new table, the second for it to show the first
    assert time.perf_counter() - t0 >= (SHOW["start"] + SHOW["keep"]) * 0.2, "each move waits for the page to show the one before"
    assert len({s["view"]["moves"] for s in seen}) >= 3, "the page can see the moves one by one"
    with pytest.raises(ValueError):
        srv.move(gid, 10**6)
    # a claim is played out longer than a card, the longer the route the longer; at pace 0 nothing waits
    A = srv.actions
    claim6 = next(a for a in range(A["n_claim"]) if srv.routes[srv.claims[a][0]]["length"] == 6)
    assert srv.show_time(A["blind"]) == pytest.approx(SHOW["card"] * 0.2)
    assert srv.show_time(claim6) == pytest.approx((SHOW["claim"] + 6 * SHOW["train"]) * 0.2)
    assert srv.show_time(claim6) > srv.show_time(A["tickets"]) > srv.show_time(A["keep1"])
    assert Server(checkpoint, "cpu", look_worlds=0, pace=0).show_time(claim6) == 0


def test_the_command_without_arguments_serves_the_published_network(engine, monkeypatch, capsys):
    from deepttr import serve

    monkeypatch.setattr(sys, "argv", ["deepttr.serve", "--port", "0", "--device", "cpu"])
    monkeypatch.setattr(serve.ThreadingHTTPServer, "serve_forever", lambda self: self.server_close())
    serve.main()
    said = capsys.readouterr().out.splitlines()
    assert said[0].startswith("network deepttr.pt, step 154189, sha256 ") and said[0].endswith(", on cpu")
    assert said[1].startswith("no look-ahead on a CPU") and said[2].startswith("play at http://127.0.0.1:") and len(said) == 3


def test_the_look_ahead_is_the_default_on_a_gpu_alone(engine, checkpoint, tmp_path):
    from deepttr.serve import Server, look_worlds_on

    assert [look_worlds_on(torch.device(d)) for d in ("cuda", "cuda:1", "mps", "cpu")] == [8, 8, 8, 0]
    asked = Server(checkpoint, "cpu", record=tmp_path, look_worlds=8)
    assert asked.looker is not None and asked.about["look"] == {"worlds": 8, "check": 24, "seconds": 8.0}
    assert asked.describe()[1:] == [
        "look-ahead: 8 worlds a candidate, 24 to check, at most 8 s a move; a CPU seldom ends a look in that time, "
        "and the move is then the first choice",
        f"finished games are recorded in {tmp_path / 'games.jsonl'}, with the player's consent"]


def test_the_server_answers_over_http(engine, checkpoint):
    from deepttr.serve import Server, handler

    srv = Server(checkpoint, "cpu", look_worlds=0, pace=0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(srv))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers["Content-Type"], r.read()

    try:
        for path, ctype in (("/", "text/html"), ("/web/app.js", "text/javascript"), ("/web/style.css", "text/css"),
                            ("/web/board.webp", "image/webp"), ("/web/board.json", "application/json")):
            status, got, body = call(path)
            assert status == 200 and got.startswith(ctype) and len(body) > 500
        drawn = json.loads(call("/web/board.json")[2])
        assert drawn["width"] == 1600 and drawn["land"].startswith("M") and set(drawn["albers"]) == {"lat0", "lon0", "lat1", "lat2"}
        board = json.loads(call("/api/map")[2])
        assert len(board["cities"]) == 36 and board["recording"] is False
        game = json.loads(call("/api/new", {"name": "t", "seat": 0, "players": 3})[2])
        srv.wait(game["id"])
        game = json.loads(call(f"/api/game/{game['id']}")[2])
        assert game["human_seat"] == 0 and game["players"] == 3 and game["view"]["legal"] and not game["thinking"]
        after = json.loads(call(f"/api/game/{game['id']}/move", {"action": game["view"]["legal"][0]})[2])
        assert after["view"]["moves"] > game["view"]["moves"]
        srv.wait(game["id"])
        assert json.loads(call(f"/api/game/{game['id']}")[2])["view"]["moves"] >= after["view"]["moves"]
        for path, body, code in ((f"/api/game/{game['id']}/move", {"action": "x"}, 400), ("/api/game/none/move", {"action": 1}, 404),
                                 ("/api/nothing", {}, 404), ("/web/secret.txt", None, 404), ("/api/game/none", None, 404),
                                 ("/api/new", {"players": 7}, 400)):
            with pytest.raises(urllib.error.HTTPError) as err:
                call(path, body)
            assert err.value.code == code, path
    finally:
        httpd.shutdown()
        httpd.server_close()
