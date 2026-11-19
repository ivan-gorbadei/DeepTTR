"""The chart page of a training run, served on this machine.

usage: python -m deepttr.charts RUN_ROOT [--port 8090] [--host 127.0.0.1]

- GET /                  the page (deepttr/charts/page.html): it draws every chart itself
- GET /data?view=V       the phases' list and one view's charts (deepttr.charts.data): V is
                         "curriculum" or a phase's number
- GET /svg?view=V&chart=ID[&x=n|s|h][&smooth=0.6][&from=X&to=X][&hide=0,2][&log=1][&theme=light|dark]
                         the chart as a picture (deepttr.charts.export.render), as the page shows it

The files are read again whenever one of them has changed, so the page, which asks every minute,
follows a run under way. The server only reads. It listens on 127.0.0.1, this machine alone;
`--host 0.0.0.0` opens it to the other machines of the network, to follow a run on another
machine.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from deepttr.charts import data, export

PAGE = Path(__file__).resolve().parent / "page.html"


class Run:
    """A run root's views, read again when a file of it changed."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.lock = threading.Lock()
        self.key, self.views = None, None

    def get(self) -> dict:
        with self.lock:
            key = data.version(self.root)
            if key != self.key:
                self.views, self.key = data.load(self.root), key
            return self.views


def picture(views: dict, q: dict) -> str:
    """The SVG of /svg's query; KeyError for a view or chart that is not there, ValueError for a bad value."""
    one = lambda k, d=None: q.get(k, [d])[0]  # noqa: E731
    view = views["views"][one("view", "curriculum")]
    chart = next((c for c in view["charts"] if c["id"] == one("chart")), None)
    if chart is None:
        raise KeyError(one("chart"))
    x = one("x", "n")
    if x not in ("n", "s", "h"):
        raise ValueError("x is n, s or h")
    smooth = float(one("smooth", export.SMOOTH))
    if not 0 <= smooth < 1:
        raise ValueError("smooth is in [0, 1)")
    lo, hi = one("from"), one("to")
    span = (float(lo), float(hi)) if lo is not None and hi is not None else None
    hide = [int(i) for i in one("hide", "").split(",") if i.strip()]
    theme = one("theme", "auto")
    if theme not in ("auto", "light", "dark"):
        raise ValueError("theme is auto, light or dark")
    return export.render(chart, view, x=x, smooth=smooth, x_range=span, hidden=hide, log=one("log") == "1", theme=theme)


def handler(run: Run):
    class H(BaseHTTPRequestHandler):
        server_version = "deepttr-charts/1"

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, separators=(",", ":")).encode())

        def do_GET(self) -> None:  # noqa: N802 - the http.server API
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if url.path == "/data":
                    views = run.get()
                    view = q.get("view", ["curriculum"])[0]
                    if view not in views["views"]:
                        return self._json(404, {"error": f"no view {view!r}"})
                    return self._json(200, {"version": views["version"], "root": views["root"], "phases": views["phases"],
                                            "view": views["views"][view]})
                if url.path == "/svg":
                    return self._send(200, picture(run.get(), q).encode(), "image/svg+xml; charset=utf-8")
                return self._json(404, {"error": "not found"})
            except KeyError as exc:
                return self._json(404, {"error": f"not there: {exc}"})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})

        def log_message(self, fmt: str, *args) -> None:
            print(f"{time.strftime('%H:%M:%S')} {fmt % args}", flush=True)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="The chart page of a training run, in a browser.")
    ap.add_argument("run_root", help="the run's folder: phase01 .. phase11 and elo.json")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="the address to listen on; 0.0.0.0: every address of the machine")
    a = ap.parse_args()
    root = Path(a.run_root)
    if not root.is_dir():
        raise SystemExit(f"{root}: no such folder")
    run = Run(root)
    found = len(run.get()["phases"])
    httpd = ThreadingHTTPServer((a.host, a.port), handler(run))
    print(f"{found} phase{'' if found == 1 else 's'} in {root}; the charts at http://{a.host}:{httpd.server_address[1]}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
