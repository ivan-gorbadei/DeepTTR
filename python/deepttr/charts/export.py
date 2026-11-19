"""Static charts of a training run, for an article.

usage: python -m deepttr.charts.export RUN_ROOT OUTDIR

For every phase it writes the charts that show its skill being learned (`PICKS`), and two of the
whole curriculum: OUTDIR/phaseNN-<chart>.svg and OUTDIR/curriculum-<chart>.svg. A file needs no
font, script or style sheet from elsewhere, and follows the reader's colour scheme, light or dark,
by a media query of its own, so it shows as it should on a GitHub page. The x axis is the training
games; a training line is smoothed as the page shows it by default, the raw line faint behind.

`render(chart, view)` draws one chart of `deepttr.charts.data`; the chart page asks the server for
its pictures through it too, with its own x axis, smoothing, zoom, hidden series and scheme.
"""
from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path

from deepttr.charts import data

SMOOTH = 0.6  # the page's default smoothing
PICKS = {
    "curriculum": ("elo", "bots"),
    1: ("built",),  # the starting tickets completed
    2: ("fixed", "tickets"),  # ticket points on fixed deals; more tickets taken and completed
    3: ("tables", "fixed"),  # ticket points at every table size; sampled play catching up
    4: ("score2",),  # the routes' points come back while the tickets' hold
    5: ("goal", "vs-card-hoarder"),
    6: ("goal", "vs-phase-5"),
    7: ("goal", "vs-phase-6"),
    8: ("score2", "goal"),  # its own points up, and what that cost in first places
    9: ("goal",),
    10: ("teacher", "agree"),  # fewer finds a game as the network plays the lessons itself
    11: ("goal", "forecast"),  # the move forecast's hints confirmed more often
}
CAPTION = {"n": "training games", "s": "training steps", "h": "hours of training"}
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
LIGHT = {"surface": "#fcfcfb", "frame": "#e1e0d9", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "band": "#f2f1ec",
         "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")}
DARK = {"surface": "#1a1a19", "frame": "#2c2c2a", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "band": "#222221",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767")}


def style(theme: str) -> str:
    def rules(t: dict) -> str:
        out = [f".bg {{ fill: {t['surface']}; stroke: {t['frame']}; }}", f".grid {{ stroke: {t['grid']}; }}",
               f".axis {{ stroke: {t['axis']}; }}", f".pb {{ fill: {t['band']}; }}", f".ref {{ stroke: {t['muted']}; }}",
               f".t {{ fill: {t['ink']}; }}", f".s {{ fill: {t['ink2']}; }}", f".m {{ fill: {t['muted']}; }}",
               f".ring {{ stroke: {t['surface']}; }}", f".kink {{ stroke: {t['ink']}; }}", f".fink {{ fill: {t['ink']}; }}"]
        for i, c in enumerate(t["series"]):
            out.append(f".k{i} {{ stroke: {c}; }} .f{i} {{ fill: {c}; }}")
        return "\n".join(out)
    base = (f"text {{ font-family: {FONT}; }}\n.grid, .axis {{ stroke-width: 1; }}\n"
            ".ref { stroke-width: 1; stroke-dasharray: 4 3; }\n.line { fill: none; stroke-linejoin: round; stroke-linecap: round; }\n"
            ".raw { fill: none; stroke-width: 1.2; opacity: 0.28; }\n.band { opacity: 0.16; }\n.ring { stroke-width: 1.5; }\n")
    if theme == "dark":
        return base + rules(DARK)
    if theme == "light":
        return base + rules(LIGHT)
    return base + rules(LIGHT) + "\n@media (prefers-color-scheme: dark) {\n" + rules(DARK) + "\n}"


def width_of(text: str, size: float) -> float:
    """About how wide a line of the system sans is: no font is loaded to measure it."""
    narrow = sum(text.count(c) for c in " .,:;'!|il()-1")
    return (len(text) - narrow) * 0.56 * size + narrow * 0.3 * size


def wrap(text: str, width: float, size: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        nxt = f"{cur} {word}".strip()
        if cur and width_of(nxt, size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = nxt
    return lines + [cur] if cur else lines


def trim(s: str) -> str:
    return s.rstrip("0").rstrip(".") if "." in s else s


def compact(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return trim(f"{v / 1e6:.{1 if a >= 1e7 else 2}f}") + "M"
    if a >= 1e3:
        return trim(f"{v / 1e3:.{0 if a >= 1e4 else 1}f}") + "k"
    return trim(f"{v:.2f}")


def value(v: float, unit: str) -> str:
    if unit == "Elo":
        return f"{v:+,.0f}" if round(v) else "0"
    a = abs(v)
    d = 0 if a >= 1000 else 1 if a >= 10 else 2 if a >= 1 else 3
    return f"{v:,.{d}f}" + (" %" if unit == "%" else "")


def tick(v: float, unit: str) -> str:
    if unit == "Elo":
        return f"{v:+,.0f}" if round(v) else "0"
    s = trim(f"{v:,.4f}")
    return "0" if s == "-0" else s


def nice(lo: float, hi: float, count: int) -> tuple[float, float, float, list[float]]:
    if not hi > lo:
        d = abs(hi) * 0.1 or 1.0
        lo, hi = lo - d, hi + d
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * 0.999)
    a, b = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    ticks, k = [], 0
    while a + k * step <= b + step * 1e-6:
        ticks.append(round(a + k * step, 12))
        k += 1
    return a, b, step, ticks


def log_ticks(lo: float, hi: float) -> list[float]:
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    out = [m * 10 ** k for k in range(a, b + 1) for m in ((1,) if b - a > 3 else (1, 2, 5)) if lo / 1.0001 <= m * 10 ** k <= hi * 1.0001]
    return out if len(out) >= 2 else [lo, hi]


def ema(ys: list, w: float) -> list:
    out, last, k = [], 0.0, 0
    for y in ys:
        if y is None:
            out.append(None)
            last, k = 0.0, 0
            continue
        last = last * w + (1 - w) * y
        k += 1
        out.append(last / (1 - w ** k))
    return out


def columns(s: dict, kind: str, x: str) -> tuple[list, list, list, list]:
    """A series' x, y, low and high, None where a point is missing (a break)."""
    at = (0, 1, 2, 3) if kind == "phases" else ({"n": 0, "s": 1, "h": 2}[x], 3, 4, 5)
    return tuple([q[i] if q is not None and len(q) > i else None for q in s["points"]] for i in at)


def colour(c) -> str:
    return "ink" if c == data.INK else str(int(c) % 8)


def render(chart: dict, view: dict, x: str = "n", smooth: float = SMOOTH, x_range: tuple[float, float] | None = None,
           hidden=(), log: bool = False, theme: str = "auto", width: int = 760) -> str:
    """One chart as a self-contained SVG. `hidden`: indices of series left out."""
    kind, unit = chart["kind"], chart["unit"]
    ser = [s for i, s in enumerate(chart["series"]) if i not in set(hidden)]
    cols = []
    for s in ser:
        xs, ys, lo, hi = columns(s, kind, x)
        shown = ema(ys, smooth) if s.get("smooth") and smooth > 0 else ys
        cols.append((s, xs, ys, shown, lo, hi))
    if kind == "phases":
        every = [v for _, xs, *_ in cols for v in xs if v is not None] or [1]
        x0, x1 = min(every) - 0.5, max(every) + 0.5
    else:
        x0, x1 = x_range or (0.0, float((view.get("extent") or {}).get(x) or 0.0) or 1.0)
    log = log and all(v is None or v > 0 for c in cols for v in c[2]) and all(r["y"] > 0 for r in chart["refs"])
    vals = [v for _, xs, ys, shown, lo, hi in cols for i, xv in enumerate(xs) if xv is not None and x0 <= xv <= x1
            for v in (ys[i], shown[i], lo[i], hi[i]) if v is not None and (not log or v > 0)]
    vals += [r["y"] for r in chart["refs"] if not log or r["y"] > 0]
    lo_v, hi_v = (min(vals), max(vals)) if vals else ((1.0, 10.0) if log else (0.0, 1.0))

    pad, out = 24, []
    title_y = 34
    about = wrap(chart["about"], width - 2 * pad, 12)[:4]
    y = title_y + 22 + 16 * (len(about) - 1)
    legend = []
    if len(ser) > 1:  # the legend, in rows
        y += 24
        lx = pad
        for s, *_ in cols:
            w = 22 + width_of(s["name"], 12) + 18
            if lx + w > width - pad and lx > pad:
                lx, y = pad, y + 18
            legend.append((lx, y, s))
            lx += w
    direct = kind == "time" and 0 < len(ser) <= 4
    label_w = max([width_of(f"{value(1234.5, unit)} {s['name'] if len(ser) > 1 else ''}", 11.5) for s in ser] + [0]) if direct else 0
    T = y + (32 if view.get("bands") and kind == "time" else 20)
    B = T + 230
    height = B + 48
    if log:
        a_v, b_v = lo_v / 1.15, hi_v * 1.15
        ticks, ly = log_ticks(a_v, b_v), math.log10
    else:
        span = (hi_v - lo_v) * 0.05
        a_v, b_v, _, ticks = nice(lo_v - span, hi_v + span, 6)
        if lo_v >= 0:  # no axis below zero for values that cannot go there, none above 100 % for shares
            a_v = max(0.0, a_v)
        if unit == "%" and hi_v <= 100:
            b_v = min(100.0, b_v)
        ticks = [t for t in ticks if a_v <= t <= b_v]
        ly = float
    labels = [tick(t, unit) for t in ticks]
    L = pad + max(width_of(t, 11.5) for t in labels) + 10
    R = width - pad - (min(230.0, label_w + 14) if direct else 0)

    def sx(v: float) -> float:
        return L + (v - x0) / ((x1 - x0) or 1) * (R - L)

    def sy(v: float) -> float:
        return B - (ly(v) - ly(a_v)) / ((ly(b_v) - ly(a_v)) or 1) * (B - T)

    add = out.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        f'aria-label="{escape(chart["title"])}">')
    add(f"<style>\n{style(theme)}\n</style>")
    add(f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" class="bg"/>')
    add(f'<text x="{pad}" y="{title_y}" class="t" font-size="15" font-weight="600">{escape(chart["title"])}</text>')
    for k, line in enumerate(about):
        add(f'<text x="{pad}" y="{title_y + 22 + 16 * k}" class="s" font-size="12">{escape(line)}</text>')
    for lx, ly_, s in legend:
        c = colour(s["color"])
        if s["style"] == "dots":
            add(f'<circle cx="{lx + 8}" cy="{ly_ - 4}" r="4" class="f{c}"/>')
        else:
            add(f'<line x1="{lx}" y1="{ly_ - 4}" x2="{lx + 16}" y2="{ly_ - 4}" class="k{c}" stroke-width="2.5"/>')
        add(f'<text x="{lx + 22}" y="{ly_}" class="s" font-size="12">{escape(s["name"])}</text>')

    def fmt(v: float) -> str:
        return f"{v:.1f}"

    # the phases' bands, the grid, the axes
    if view.get("bands") and kind == "time":
        for i, band in enumerate(view["bands"]):
            a, b = max(L, sx(band["from"][x])), min(R, sx(band["to"][x]))
            if b <= a:
                continue
            if i % 2 == 0:
                add(f'<rect x="{fmt(a)}" y="{T}" width="{fmt(b - a)}" height="{B - T}" class="pb"/>')
            if b - a > 12:
                add(f'<text x="{fmt((a + b) / 2)}" y="{T - 8}" class="m" font-size="11" text-anchor="middle">{band["phase"]}</text>')
        add(f'<text x="{fmt(L - 8)}" y="{T - 8}" class="m" font-size="11" text-anchor="end">phase</text>')
    for t, lab in zip(ticks, labels):
        yy = sy(t)
        add(f'<line x1="{fmt(L)}" y1="{fmt(yy)}" x2="{fmt(R)}" y2="{fmt(yy)}" class="grid"/>')
        add(f'<text x="{fmt(L - 8)}" y="{fmt(yy + 4)}" class="m" font-size="11.5" text-anchor="end">{escape(lab)}</text>')
    add(f'<line x1="{fmt(L)}" y1="{B}" x2="{fmt(R)}" y2="{B}" class="axis"/>')
    if kind == "phases":
        for k in range(math.ceil(x0), math.floor(x1) + 1):
            add(f'<text x="{fmt(sx(k))}" y="{B + 18}" class="m" font-size="11.5" text-anchor="middle">{k}</text>')
        caption = "phase"
    else:
        _, _, step, xt = nice(x0, x1, max(2, int((R - L) / 100)))
        for t in xt:
            if x0 <= t <= x1:
                lab = (f"{t:g} h" if step < 1 else f"{round(t)} h") if x == "h" else compact(t)
                add(f'<text x="{fmt(sx(t))}" y="{B + 18}" class="m" font-size="11.5" text-anchor="middle">{lab}</text>')
        caption = CAPTION[x]
    add(f'<text x="{fmt((L + R) / 2)}" y="{B + 38}" class="m" font-size="11.5" text-anchor="middle">{caption}</text>')

    add(f'<clipPath id="plot"><rect x="{fmt(L)}" y="{T - 2}" width="{fmt(R - L)}" height="{B - T + 4}"/></clipPath>')
    add('<g clip-path="url(#plot)">')
    if view.get("bands") and kind == "time":
        for band in view["bands"][:-1]:
            xx = sx(band["to"][x])
            add(f'<line x1="{fmt(xx)}" y1="{T}" x2="{fmt(xx)}" y2="{B}" class="axis"/>')
    if view.get("end") and kind == "time":
        xx = sx(view["end"][x])
        add(f'<line x1="{fmt(xx)}" y1="{T}" x2="{fmt(xx)}" y2="{B}" class="ref"/>')
    for r in chart["refs"]:
        add(f'<line x1="{fmt(L)}" y1="{fmt(sy(r["y"]))}" x2="{fmt(R)}" y2="{fmt(sy(r["y"]))}" class="ref"/>')
    ends = []
    for s, xs, ys, shown, lo, hi in cols:
        c = colour(s["color"])
        ok = [xv is not None and v is not None and (not log or v > 0) for xv, v in zip(xs, shown)]
        if any(v is not None for v in lo):
            if kind == "phases":
                for i, xv in enumerate(xs):
                    if xv is not None and lo[i] is not None and hi[i] is not None:
                        add(f'<line x1="{fmt(sx(xv))}" y1="{fmt(sy(lo[i]))}" x2="{fmt(sx(xv))}" y2="{fmt(sy(hi[i]))}" class="k{c}" stroke-width="1.5"/>')
            else:
                run = []
                for i in range(len(xs) + 1):
                    if i < len(xs) and xs[i] is not None and lo[i] is not None and hi[i] is not None and (not log or lo[i] > 0):
                        run.append(i)
                        continue
                    if len(run) > 1:
                        top = " L".join(f"{fmt(sx(xs[j]))} {fmt(sy(hi[j]))}" for j in run)
                        bot = " L".join(f"{fmt(sx(xs[j]))} {fmt(sy(lo[j]))}" for j in reversed(run))
                        add(f'<path d="M{top} L{bot} Z" class="band f{c}"/>')
                    run = []

        def path(vals: list, step: bool) -> str:
            d, pen = [], False
            for i, xv in enumerate(xs):
                v = vals[i]
                if xv is None or v is None or (log and v <= 0):
                    pen = False
                    continue
                X, Y = sx(xv), sy(v)
                d.append(f"M{fmt(X)} {fmt(Y)}" if not pen else f"H{fmt(X)} V{fmt(Y)}" if step else f"L{fmt(X)} {fmt(Y)}")
                pen = True
            return " ".join(d)
        if s.get("smooth") and smooth > 0:
            add(f'<path d="{path(ys, False)}" class="raw k{c}"/>')
        dots = s["style"] == "dots"
        faint = ' opacity="0.65"' if dots else ""
        add(f'<path d="{path(shown, s["style"] == "step")}" class="line k{c}" stroke-width="{1.5 if dots else s.get("width", 2)}"{faint}/>')
        if dots:
            for i, xv in enumerate(xs):
                if ok[i]:
                    big = kind == "phases" and chart.get("mark") == xv
                    add(f'<circle cx="{fmt(sx(xv))}" cy="{fmt(sy(shown[i]))}" r="{6 if big else 4}" class="f{c} ring"/>')
        last = max((i for i, good in enumerate(ok) if good and x0 <= xs[i] <= x1), default=None)
        if last is not None:
            ends.append((sy(shown[last]), s, shown[last]))
    add("</g>")
    for r in chart["refs"]:
        yy = sy(r["y"])
        if T <= yy <= B:  # the label above its line, or below it at the top of the plot
            add(f'<text x="{fmt(R - 4)}" y="{fmt(yy - 5 if yy - T > 14 else yy + 13)}" class="s" font-size="11" text-anchor="end">'
                f'{escape(r["label"])}</text>')
    if view.get("end") and kind == "time" and L <= sx(view["end"][x]) <= R:  # above the plot, clear of the lines
        xx = sx(view["end"][x])
        add(f'<text x="{fmt(xx)}" y="{T - 5}" class="m" font-size="11" text-anchor="{"end" if xx > R - 12 else "middle"}">end</text>')
    if direct:  # the newest value of each line at its end, nudged apart
        ends.sort(key=lambda e: e[0])
        placed: list[float] = []
        for yy, s, v in ends:
            yy = max(T + 4, min(B - 2, yy))
            if placed and yy - placed[-1] < 15:
                yy = placed[-1] + 15
            placed.append(yy)
        shift = max(0.0, placed[-1] - (B + 6)) if placed else 0.0
        for (_, s, v), yy in zip(ends, placed):
            yy -= shift
            c = colour(s["color"])
            add(f'<line x1="{fmt(R + 6)}" y1="{fmt(yy)}" x2="{fmt(R + 16)}" y2="{fmt(yy)}" class="k{c}" stroke-width="2.5"/>')
            name = f" {s['name']}" if len(ser) > 1 else ""
            add(f'<text x="{fmt(R + 20)}" y="{fmt(yy + 4)}" class="s" font-size="11.5"><tspan class="t" font-weight="600">'
                f'{escape(value(v, unit))}</tspan>{escape(name)}</text>')
    if kind == "phases":  # the values of the first, the last and the best phase end
        for s, xs, ys, shown, lo, hi in cols:
            idx = [i for i, xv in enumerate(xs) if xv is not None and shown[i] is not None]
            if not idx:
                continue
            label = {idx[0], idx[-1], max(idx, key=lambda i: shown[i])} | {i for i in idx if xs[i] == chart.get("mark")}
            for i in sorted(label):
                top = sy(hi[i]) if hi[i] is not None else sy(shown[i])
                add(f'<text x="{fmt(sx(xs[i]))}" y="{fmt(top - 8)}" class="t" font-size="12" font-weight="600" text-anchor="middle">'
                    f'{escape(value(shown[i], unit))}</text>')
    add("</svg>")
    return "\n".join(out) + "\n"


def write_all(root: str | Path, outdir: str | Path) -> list[Path]:
    """The picked charts of a run root as SVG files in `outdir`; a chart without data is passed over."""
    views = data.load(root)["views"]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key, ids in PICKS.items():
        view = views.get(str(key))
        for cid in ids if view else ():
            c = next((c for c in view["charts"] if c["id"] == cid), None)
            if c is not None:
                path = out / (f"curriculum-{cid}.svg" if key == "curriculum" else f"phase{key:02d}-{cid}.svg")
                path.write_text(render(c, view), encoding="utf-8", newline="\n")
                written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Static SVG charts of a training run: a few a phase, two of the curriculum.")
    ap.add_argument("run_root", help="the run's folder: phase01 .. phase11 and elo.json")
    ap.add_argument("outdir", help="where the SVG files go")
    a = ap.parse_args()
    for path in write_all(a.run_root, a.outdir):
        print(path)


if __name__ == "__main__":
    main()
