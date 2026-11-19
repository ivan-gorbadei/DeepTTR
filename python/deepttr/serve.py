"""Play against a network in a browser: Ticket to Ride, the USA map, two to five players.

usage: python -m deepttr.serve [CHECKPOINT] [--host 127.0.0.1] [--port 8080] [--device auto] [--record DIR]
                                [--look-worlds N] [--look-check 24] [--look-seconds 8] [--pace 1]

Without CHECKPOINT the published network plays, the one that comes with the package.

- GET  /                       the page (deepttr/web/index.html; /web/* are its files)
- GET  /api/map                the board: cities with coordinates, routes, tickets, the claim table
- POST /api/new                {"name": str, "consent": bool, "players": 2..5, "seat": int|null} -> a new
                               game; the network holds every seat but the human's
- GET  /api/game/ID            the game as the human sees it; "thinking" while the network moves
- POST /api/game/ID/move       {"action": int} -> the human's move. The network then plays its seats
                               one move at a time, each once the page has played out the move
                               before (`SHOW`, times `--pace`), until it is the human's turn again;
                               the page follows by asking for the game again

The network sees exactly what a player in its seat sees. On a GPU it looks ahead before it moves
(`deepttr.look`): its first choice and its best alternatives are played to the end of the game in
`--look-worlds` hidden worlds each (8), dealt afresh from what its seat has seen; an alternative
that comes out ahead is tried again in `--look-check` fresh worlds and is played only if it proves
better there. After `--look-seconds` the first choice is played as it is. A look takes thousands
of evaluations, a second or two on a fast CUDA card. A CPU seldom ends one within the time for a
move, so on a CPU the network plays its first choice, one evaluation a move, unless
`--look-worlds` is given. `--look-worlds 0` is the first choice on any device.

Games live in memory. With `--record DIR` every finished game is a line of DIR/games.jsonl - the
name the player gave, the network (file, training step, sha256) and its look-ahead, the seed,
every move and the final score, which replay the game - and a game then starts only with the
player's consent to that record. Without `--record` nothing is written and no consent is asked.
The server has no accounts and no access control: it is meant for a local machine or a trusted
network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from deepttr import _engine as engine
from deepttr import net as netmod
from deepttr.look import CHECK, SECONDS, WORLDS, Looker
from deepttr.play import Evaluator

WEB = Path(__file__).resolve().parent / "web"
STATIC = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
          "style.css": "text/css; charset=utf-8", "board.webp": "image/webp", "board.json": "application/json"}
# How long the page plays a move out, in seconds (web/app.js): a drawn card flies to its seat, drawn
# tickets fly one after another, a claim shows the cards paid and then sets the trains down, one
# every `train` seconds. The network's next move waits for it, so that every move can be followed;
# `--pace` multiplies these times (0: no waiting).
SHOW = {"start": 1.0, "card": 1.0, "tickets": 1.1, "keep": 0.8, "claim": 1.5, "train": 0.15, "pass": 1.2}
PACE = 1.0
MAX_BODY = 4096


def look_worlds_on(device) -> int:
    """The first look's copies where nothing was asked: the look-ahead on a GPU, none on a CPU,
    which seldom ends a look within the time for a move."""
    return 0 if device.type == "cpu" else WORLDS


class Server:
    """`checkpoint`: a file, or None for the published network. `look_worlds`: None for the
    device's default (`look_worlds_on`), 0 for no look-ahead."""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None, record: str | Path | None = None,
                 amp: bool = True, look_worlds: int | None = None, look_check: int = CHECK, look_seconds: float = SECONDS,
                 pace: float = PACE) -> None:
        checkpoint = netmod.PUBLISHED if checkpoint is None else Path(checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(f"{checkpoint}: no such checkpoint")
        net, meta = netmod.load(checkpoint)
        have = engine.dims()
        stale = {k: (net.dims.get(k), have[k]) for k in ("route_f", "city_f", "ticket_f", "player_f", "global_f", "n_actions")
                 if net.dims.get(k) != have[k]}
        if stale:  # a network built for another observation cannot read this engine's
            raise SystemExit(f"{checkpoint}: the network does not fit this engine's observation (network, engine): {stale}")
        self.ai = Evaluator(net, device, amp=amp)
        if look_worlds is None:
            look_worlds = look_worlds_on(self.ai.device)
        self.looker = Looker(self.ai, look_worlds, look_check, look_seconds) if look_worlds else None
        self.about = {"checkpoint": checkpoint.name, "step": meta.get("step"),
                      "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16],
                      "look": {"worlds": look_worlds, "check": look_check, "seconds": look_seconds} if look_worlds else None}
        self.record = Path(record) if record else None
        if self.record:
            self.record.mkdir(parents=True, exist_ok=True)
        self.pace = pace
        self.lock = threading.Lock()  # the table of games
        self.thinking = threading.Lock()  # one network: its moves, in whatever game, one at a time
        self.games: dict[str, dict] = {}
        board = json.loads(engine.map_json())
        self.actions, self.claims, self.routes = board["actions"], board["claims"], board["routes"]
        board["recording"] = self.record is not None
        self.board = json.dumps(board)

    def describe(self) -> list[str]:
        """What the server says of itself at its start: the network, and how it chooses a move."""
        about, cpu = self.about, self.ai.device.type == "cpu"
        lines = [f"network {about['checkpoint']}, step {about['step']}, sha256 {about['sha256']}, on {self.ai.device}"]
        if self.looker is None:
            lines.append(f"no look-ahead on a CPU: the network plays its first choice (--look-worlds {WORLDS} turns it on)" if cpu
                         else "no look-ahead: the network plays its first choice")
        else:
            look = about["look"]
            lines.append(f"look-ahead: {look['worlds']} worlds a candidate, {look['check']} to check, at most {look['seconds']:g} s a move"
                         + ("; a CPU seldom ends a look in that time, and the move is then the first choice" if cpu else ""))
        if self.record:
            lines.append(f"finished games are recorded in {self.record / 'games.jsonl'}, with the player's consent")
        return lines

    def new(self, name: str, consent: bool, seat=None, players: int = 2) -> str:
        """A new game of `players` seats; the human takes `seat` (None: a seat drawn at random)."""
        if self.record and not consent:
            raise ValueError("a recorded game needs the player's consent")
        if not isinstance(players, int) or isinstance(players, bool) or not 2 <= players <= 5:
            raise ValueError("a table has two to five players")
        if seat is not None and (not isinstance(seat, int) or isinstance(seat, bool) or not 0 <= seat < players):
            raise ValueError(f"the seat is one of 0 to {players - 1}, or none for a seat at random")
        name = " ".join(str(name).split())[:60] or "anonymous"
        seat = random.randrange(players) if seat is None else seat
        gid = uuid.uuid4().hex[:16]
        seed = random.getrandbits(62)
        meta = {"id": gid, "name": name, "consent": bool(consent), "players": players, "human_seat": seat, "seed": seed,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ai": self.about,
                "looks": {"decisions": 0, "changed": 0, "timeouts": 0}}
        rec = {"game": engine.Game(players=players, seed=seed), "meta": meta, "lock": threading.Lock(), "worker": None,
               "shown": (time.perf_counter(), SHOW["start"] * self.pace)}  # the last move: when, and how long the page shows it
        with self.lock:
            self.games[gid] = rec
        self._play_on(rec)
        return gid

    def names(self, meta: dict) -> list[str]:
        """Every seat's name: the human's, and the network's seats as AI 1, AI 2, ... in seat order."""
        out, k = [], 0
        for q in range(meta["players"]):
            if q == meta["human_seat"]:
                out.append(meta["name"])
            else:
                k += 1
                out.append(f"AI {k}")
        return out

    def _ai_move(self, rec: dict) -> int:
        """The network's move in a game, by the look-ahead when there is one; the game's record
        counts what its looks did."""
        with self.thinking:
            if self.looker is None:
                return self.ai.move(rec["game"])
            move = self.looker.move(rec["game"])
            seen, looks = self.looker.last, rec["meta"]["looks"]
            if seen is not None:
                looks["decisions"] += len(seen["options"]) > 1
                looks["changed"] += seen["changed"]
                looks["timeouts"] += bool(seen.get("time"))
            return move

    def show_time(self, action: int) -> float:
        """How long the page plays a move out (`SHOW`), times the pace."""
        a = self.actions
        if action < 0:
            s = SHOW["pass"]
        elif action < a["n_claim"]:
            s = SHOW["claim"] + SHOW["train"] * self.routes[self.claims[action][0]]["length"]
        elif action < a["tickets"]:  # a face-up card or the deck
            s = SHOW["card"]
        else:
            s = SHOW["tickets"] if action == a["tickets"] else SHOW["keep"]
        return s * self.pace

    def _shown(self, rec: dict, action: int) -> None:
        rec["shown"] = (time.perf_counter(), self.show_time(action))

    def _play_on(self, rec: dict) -> None:
        """Let the network play its seats of a game, in the background."""
        if rec["worker"] is None or not rec["worker"].is_alive():
            rec["worker"] = threading.Thread(target=self._advance, args=(rec,), daemon=True)
            rec["worker"].start()

    def wait(self, gid: str, timeout: float | None = None) -> None:
        """Until the network has played its moves of a game: the human's turn, or the end."""
        worker = self._game(gid)["worker"]
        if worker is not None:
            worker.join(timeout)

    def _advance(self, rec: dict) -> None:
        """Play the network's moves (and the human's forced passes) until the human has a choice.
        The game is read without its lock while the network thinks (the human cannot move then) and
        changed only under it. A move is made once the page has played out the one before: the
        network thinks meanwhile."""
        g, human = rec["game"], rec["meta"]["human_seat"]
        while True:
            with rec["lock"]:
                if g.is_over():
                    break
                if g.current == human:
                    if g.legal():
                        return
                    g.step(-1)  # nothing is legal for the human: the rules pass the turn
                    self._shown(rec, -1)
                    continue
            action = self._ai_move(rec)
            when, hold = rec["shown"]
            time.sleep(max(0.0, when + hold - time.perf_counter()))
            with rec["lock"]:
                g.step(action)
                self._shown(rec, action)
        if "finished" not in rec["meta"]:
            self._finish(rec)

    def _finish(self, rec: dict) -> None:
        g, meta = rec["game"], rec["meta"]
        gid = meta["id"]
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if self.looker is not None:
            looks = meta["looks"]
            print(f"{time.strftime('%H:%M:%S')} game {gid} is over: {looks['decisions']} looks ahead, {looks['changed']} changed the move, "
                  f"{looks['timeouts']} ran out of time", flush=True)
        if not self.record:
            return
        h = meta["human_seat"]
        f = json.loads(g.view(h))["final"]
        best_ai = min(r for q, r in enumerate(f["ranks"]) if q != h)
        line = {**meta, "names": self.names(meta), "moves": [list(m) for m in g.moves()],
                "final": {k: f[k] for k in ("total", "route", "tickets", "longest", "completed", "ranks")},
                "human_score": f["total"][h], "ai_scores": [t for q, t in enumerate(f["total"]) if q != h],
                "result": "human" if f["ranks"][h] < best_ai else ("ai" if best_ai < f["ranks"][h] else "tie")}
        with (self.record / "games.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(line) + "\n")

    def _game(self, gid: str) -> dict:
        with self.lock:
            rec = self.games.get(gid)
        if rec is None:
            raise KeyError(gid)
        return rec

    def move(self, gid: str, action: int) -> None:
        """The human's move; the network's answer follows in the background (`wait`)."""
        rec = self._game(gid)
        with rec["lock"]:
            g = rec["game"]
            busy = rec["worker"] is not None and rec["worker"].is_alive()
            if g.is_over() or g.current != rec["meta"]["human_seat"] or busy:
                raise ValueError("not your turn")
            g.step(int(action))  # raises ValueError on an illegal move
            self._shown(rec, int(action))
        self._play_on(rec)

    def state(self, gid: str) -> dict:
        rec = self._game(gid)
        meta = rec["meta"]
        with rec["lock"]:
            view = json.loads(rec["game"].view(meta["human_seat"]))
        thinking = view["phase"] != "over" and view["current"] != meta["human_seat"]
        return {"id": gid, "name": meta["name"], "human_seat": meta["human_seat"], "players": meta["players"],
                "names": self.names(meta), "thinking": thinking, "view": view, "ai": {"step": self.about["step"]}}


def handler(srv: Server):
    class H(BaseHTTPRequestHandler):
        server_version = "deepttr-play/1"

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode())

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n > MAX_BODY:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("a JSON object is expected")
            return data

        def do_GET(self) -> None:  # noqa: N802 - the http.server API
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (WEB / "index.html").read_bytes(), STATIC["index.html"])
                if path.startswith("/web/") and path[5:] in STATIC:
                    return self._send(200, (WEB / path[5:]).read_bytes(), STATIC[path[5:]])
                if path == "/api/map":
                    return self._send(200, srv.board.encode())
                if path.startswith("/api/game/"):
                    return self._json(200, srv.state(path[len("/api/game/"):]))
                return self._json(404, {"error": "not found"})
            except KeyError:
                return self._json(404, {"error": "no such game"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._body()
                if path == "/api/new":
                    gid = srv.new(body.get("name", ""), body.get("consent") is True, body.get("seat"), body.get("players", 2))
                    return self._json(200, srv.state(gid))
                if path.startswith("/api/game/") and path.endswith("/move"):
                    gid = path[len("/api/game/"):-len("/move")]
                    action = body.get("action")
                    if not isinstance(action, int) or isinstance(action, bool):
                        raise ValueError("action must be an integer")
                    srv.move(gid, action)
                    return self._json(200, srv.state(gid))
                return self._json(404, {"error": "not found"})
            except KeyError:
                return self._json(404, {"error": "no such game"})
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - answer, log, keep serving
                import traceback

                traceback.print_exc()
                return self._json(500, {"error": f"server error: {exc}"})

        def log_message(self, fmt: str, *args) -> None:
            print(f"{time.strftime('%H:%M:%S')} {self.address_string()} {fmt % args}", flush=True)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="Play against a network in a browser.")
    ap.add_argument("checkpoint", nargs="?", default=None, help="a checkpoint file (default: the published network)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="auto", help="auto (CUDA, else Apple MPS, else CPU), cpu, cuda, cuda:1, mps")
    ap.add_argument("--no-amp", action="store_true", help="float32 even where the device has a native reduced precision")
    ap.add_argument("--record", default=None, metavar="DIR", help="append every finished game to DIR/games.jsonl (asks for consent)")
    ap.add_argument("--look-worlds", type=int, default=None,
                    help=f"the look-ahead's first look: copies of the game per candidate move (default: {WORLDS} on a GPU, 0 on a CPU; 0: no look-ahead)")
    ap.add_argument("--look-check", type=int, default=CHECK, help="the look-ahead's second look: copies per move")
    ap.add_argument("--look-seconds", type=float, default=SECONDS, help="the time a move may take; past it the first choice is played")
    ap.add_argument("--pace", type=float, default=PACE,
                    help="how slowly the network's moves are played out: 1 as the page shows them, 2 at half the speed, 0 no waiting")
    a = ap.parse_args()
    srv = Server(a.checkpoint, a.device, a.record, amp=not a.no_amp, look_worlds=a.look_worlds, look_check=a.look_check,
                 look_seconds=a.look_seconds, pace=a.pace)
    httpd = ThreadingHTTPServer((a.host, a.port), handler(srv))
    for line in srv.describe() + [f"play at http://{a.host}:{httpd.server_address[1]}/"]:
        print(line, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
