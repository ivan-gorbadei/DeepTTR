"use strict";
// The play page. The board is drawn once, from /api/map and /web/board.json: the picture, the cities
// at their places, a track of spaces for every route. The game comes from /api/game/ID and a move is a
// POST; while the network plays its seats the page asks for the game again. Every new move is played
// out before the table after it is shown: a card taken flies to its seat, drawn tickets fly one after
// another, a claim shows the cards paid and sets the trains down one by one, each with a sound; the
// server waits as long before the next move. The game id stays in the address (#id), so a reload goes
// on with the same game while the server runs.

const NS = "http://www.w3.org/2000/svg";
const CARD = ["black", "blue", "green", "orange", "pink", "red", "white", "yellow", "locomotive"];
const FILL = {black: "#3a3b42", blue: "#2f6fbd", green: "#3d9a4b", orange: "#ea7d22", pink: "#e06fa6", red: "#d1393a",
              white: "#f2efe6", yellow: "#efc236", grey: "#b6babf"};
const LIGHT = new Set(["white", "yellow", "grey"]);  // a dark drawing on these, a white one on the rest
const SEAT = ["#12b886", "#e03131", "#3b5bdb", "#f59f00", "#9c36b5"];
const SEAT_DARK = ["#087f5b", "#9b1c1c", "#22358f", "#9a5f00", "#5f1f73"];
const SEAS = [["PACIFIC", -131, 31.5], ["OCEAN", -131, 30.0], ["ATLANTIC", -76.0, 30.2], ["OCEAN", -76.0, 28.9], ["GULF OF MEXICO", -82.6, 24.4]];
// where a city's name goes: [dx, dy, anchor]; the default is above and to the right
const LABEL = {
  "Vancouver": [-13, 5, "end"], "Seattle": [-13, 5, "end"], "Portland": [-13, 5, "end"], "San Francisco": [-13, 5, "end"],
  "Los Angeles": [-13, 14, "end"], "Boston": [0, -15, "middle"], "New York": [13, 5, "start"], "Washington": [13, 8, "start"],
  "Raleigh": [13, 9, "start"], "Charleston": [13, 9, "start"], "Miami": [13, 9, "start"], "Montreal": [0, -15, "middle"],
  "Toronto": [0, -15, "middle"], "Sault Ste. Marie": [0, -15, "middle"], "Winnipeg": [0, -15, "middle"], "Calgary": [0, -15, "middle"],
  "Duluth": [-13, -9, "end"], "Helena": [13, -9, "start"], "Salt Lake City": [13, 16, "start"], "Las Vegas": [-13, 16, "end"],
  "Phoenix": [-13, 16, "end"], "El Paso": [0, 24, "middle"], "Houston": [0, 24, "middle"], "New Orleans": [13, 18, "start"],
  "Dallas": [13, 16, "start"], "Little Rock": [13, 16, "start"], "Atlanta": [-13, 18, "end"], "Nashville": [13, -9, "start"],
  "Pittsburgh": [-13, -10, "end"], "Chicago": [13, -9, "start"], "Saint Louis": [13, 16, "start"], "Kansas City": [-13, 16, "end"],
  "Omaha": [0, -15, "middle"], "Denver": [13, -9, "start"], "Santa Fe": [13, 16, "start"], "Oklahoma City": [13, 16, "start"],
};
// the routes that bend so as not to lie on the routes beside them, by their cities' names in
// alphabetical order: the middle of the track moves by [dx, dy] board units
const BEND = {"El Paso|Los Angeles": [10, 34], "Denver|Omaha": [0, -14]};
const POLL_MS = 300;
const TRACK = 11, GAP = 2.6, CITY_CUT = 13;  // a space's width, the gap between two spaces, the room a city takes at an end
const LIFT = 4.4;  // how high a train stands on the board: its side shows below its roof
const CALM = matchMedia("(prefers-reduced-motion: reduce)").matches;
let MAP, BOARD, PROJECT = null, POS = [], MARKET = [];
// GAME is the game as the page shows it, LATEST as the server last sent it; SHOWING while moves are played out
let GAME = null, LATEST = null, BUSY = false, POLLING = false, SHOWING = false;
let PICK = null, SELECTED = null, HAND_NEW = new Set(), MY_TURN = false;
const ROUTES = [], CITIES = [], PANELS = {};
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const unit = () => Math.min(innerHeight * 0.01, innerWidth * 0.0062);  // the stylesheet's --u in pixels

async function api(path, body) {
  const opts = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function svg(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function h(tag, attrs, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "style") e.style.cssText = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}

// ---- the projection of board.json (Albers equal-area conic), lon and lat to the board's units
function projector(b) {
  const r = Math.PI / 180, {lat0, lon0, lat1, lat2} = b.albers;
  const n = (Math.sin(lat1 * r) + Math.sin(lat2 * r)) / 2;
  const C = Math.cos(lat1 * r) ** 2 + 2 * n * Math.sin(lat1 * r);
  const rho0 = Math.sqrt(C - 2 * n * Math.sin(lat0 * r)) / n;
  return (lon, lat) => {
    const rho = Math.sqrt(C - 2 * n * Math.sin(lat * r)) / n, th = n * (lon - lon0) * r;
    const x = rho * Math.sin(th), y = rho0 - rho * Math.cos(th);
    return [(x - b.x_min) * b.scale + b.left, (b.y_max - y) * b.scale + b.top];
  };
}

// ---- the pictures of the cards, drawn here
function wagon(ink) {
  return `<g fill="${ink}"><rect x="6" y="9" width="88" height="6" rx="2"/><rect x="9" y="15" width="82" height="28" rx="3"/>
    <rect x="11" y="43" width="78" height="4"/><rect x="2" y="40" width="6" height="3"/><rect x="92" y="40" width="6" height="3"/>
    <circle cx="22" cy="51" r="5.5"/><circle cx="35" cy="51" r="5.5"/><circle cx="65" cy="51" r="5.5"/><circle cx="78" cy="51" r="5.5"/></g>
    <g fill="none" stroke="${ink === "#fff" ? "rgba(0,0,0,.3)" : "rgba(255,255,255,.5)"}" stroke-width="1.6"><rect x="39" y="18" width="22" height="23" rx="1"/>
    <line x1="50" y1="18" x2="50" y2="41"/><line x1="13" y1="25" x2="35" y2="25"/><line x1="13" y1="34" x2="35" y2="34"/><line x1="65" y1="25" x2="87" y2="25"/><line x1="65" y1="34" x2="87" y2="34"/></g>`;
}
function locomotive() {
  return `<g fill="#fff"><rect x="26" y="6" width="12" height="4" rx="1"/><rect x="28" y="9" width="8" height="14"/>
    <rect x="20" y="22" width="48" height="18" rx="8"/><circle cx="50" cy="22" r="5"/><rect x="64" y="7" width="28" height="5" rx="2"/>
    <rect x="66" y="11" width="24" height="31" rx="2"/><rect x="10" y="40" width="82" height="5"/><polygon points="3,47 14,36 14,47"/>
    <circle cx="32" cy="49" r="8"/><circle cx="52" cy="49" r="8"/><circle cx="80" cy="51" r="5.5"/><circle cx="16" cy="51" r="4"/></g>
    <rect x="71" y="15" width="12" height="9" rx="1" fill="#2b2a33"/>`;
}
function cardSvg(type, landscape) {
  const name = CARD[type], loco = name === "locomotive";
  const [W, H] = landscape ? [190, 114] : [100, 155];
  const id = "g" + Math.random().toString(36).slice(2, 8);
  const ink = LIGHT.has(name) ? "#3a3b42" : "#fff";
  let art = "";
  const s = landscape ? 1.35 : 0.78, ax = (W - 100 * s) / 2, ay = landscape ? 18 : (H - 60 * s) / 2 - 6;
  if (loco) {
    const bands = CARD.slice(0, 8).map((c, i) => `<rect x="0" y="${(landscape ? 30 : 48) + i * (landscape ? 6.5 : 7.5)}" width="${W}" height="${landscape ? 6.5 : 7.5}" fill="${FILL[c]}"/>`).join("");
    art = `${bands}<g transform="translate(${ax} ${ay}) scale(${s})">${locomotive()}</g>`;
  } else {
    art = `<g transform="translate(${ax} ${ay}) scale(${s})">${wagon(ink)}</g>`;
  }
  const label = `<text x="${W / 2}" y="${H - (landscape ? 10 : 12)}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="${landscape ? 13 : 11}" font-weight="800" letter-spacing="1.5" fill="${loco ? "#f2c14e" : ink}">${name.toUpperCase()}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" xmlns="${NS}"><defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#fff" stop-opacity=".28"/><stop offset=".55" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".18"/></linearGradient></defs>
    <rect width="${W}" height="${H}" rx="9" fill="${loco ? "#2b2a33" : FILL[name]}"/>${art}<rect width="${W}" height="${H}" rx="9" fill="url(#${id})"/>
    <rect x="5" y="5" width="${W - 10}" height="${H - 10}" rx="6" fill="none" stroke="${LIGHT.has(name) ? "rgba(0,0,0,.25)" : "rgba(255,255,255,.55)"}" stroke-width="2"/>${label}</svg>`;
}
function emptySvg() {
  return `<svg viewBox="0 0 190 114" xmlns="${NS}"><rect x="3" y="3" width="184" height="108" rx="9" fill="rgba(255,255,255,.06)"
    stroke="rgba(255,255,255,.45)" stroke-width="3" stroke-dasharray="10 8"/></svg>`;
}
function deckSvg() {
  const sleepers = Array.from({length: 16}, (_, i) => `<rect x="${-20 + i * 14}" y="44" width="5" height="26" fill="#c9a36a" opacity=".55"/>`).join("");
  return `<svg viewBox="0 0 190 125" xmlns="${NS}"><rect width="190" height="125" rx="9" fill="#1f3555"/>
    <g transform="rotate(-12 95 62)">${sleepers}<rect x="-20" y="48" width="230" height="4" fill="#dfe6ee"/><rect x="-20" y="62" width="230" height="4" fill="#dfe6ee"/></g>
    <rect x="5" y="5" width="180" height="115" rx="6" fill="none" stroke="rgba(255,255,255,.5)" stroke-width="2"/>
    <text x="95" y="104" text-anchor="middle" font-family="Georgia,serif" font-size="17" font-weight="700" fill="#f2c14e" letter-spacing="2">DeepTTR</text></svg>`;
}
function ticketBackSvg() {
  return `<svg viewBox="0 0 190 125" xmlns="${NS}"><rect width="190" height="125" rx="9" fill="#f4ecd8"/>
    <rect x="6" y="6" width="178" height="113" rx="6" fill="none" stroke="#b9a06f" stroke-width="2"/>
    <path d="M38,82 C70,30 120,96 152,42" fill="none" stroke="#7a2e22" stroke-width="4" stroke-dasharray="9 6"/>
    <circle cx="38" cy="82" r="8" fill="#b8432f" stroke="#fff" stroke-width="2"/><circle cx="152" cy="42" r="8" fill="#b8432f" stroke="#fff" stroke-width="2"/>
    <text x="95" y="112" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" font-weight="800" fill="#7a2e22" letter-spacing="2">TICKETS</text></svg>`;
}
function ticketSvg(t) {
  // a small map of the board with the two cities, their names, and the ticket's points
  const [ax, ay] = POS[t.a], [bx, by] = POS[t.b];
  const k = 150 / BOARD.width, mh = BOARD.height * k, id = "c" + Math.random().toString(36).slice(2, 8);
  const name = (c) => MAP.cities[c].toUpperCase();
  // a long name is pressed into the room left of the points
  const fit = (c) => (name(c).length > 11 ? ` textLength="108" lengthAdjust="spacingAndGlyphs"` : "");
  return `<svg viewBox="0 0 320 ${Math.max(112, mh + 16)}" xmlns="${NS}"><rect width="320" height="${Math.max(112, mh + 16)}" rx="10" fill="#f4ecd8"/>
    <rect x="5" y="5" width="310" height="${Math.max(112, mh + 16) - 10}" rx="7" fill="none" stroke="#b9a06f" stroke-width="2"/>
    <defs><clipPath id="${id}"><rect width="${BOARD.width}" height="${BOARD.height}"/></clipPath></defs>
    <g transform="translate(10 8) scale(${k})"><rect width="${BOARD.width}" height="${BOARD.height}" fill="#a9c6d8"/>
      <g clip-path="url(#${id})"><path d="${BOARD.land}" fill="#e1d6b8" stroke="#b9a06f" stroke-width="5"/><path d="${BOARD.lakes}" fill="#a9c6d8"/></g>
      <line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="#7a2e22" stroke-width="14" stroke-dasharray="40 26"/>
      <circle cx="${ax}" cy="${ay}" r="34" fill="#b8432f" stroke="#fff" stroke-width="10"/><circle cx="${bx}" cy="${by}" r="34" fill="#b8432f" stroke="#fff" stroke-width="10"/></g>
    <text x="172" y="40" font-family="system-ui,sans-serif" font-size="15" font-weight="800" fill="#2b2420"${fit(t.a)}>${name(t.a)}</text>
    <text x="172" y="62" font-family="system-ui,sans-serif" font-size="13" fill="#7d7064">to</text>
    <text x="172" y="84" font-family="system-ui,sans-serif" font-size="15" font-weight="800" fill="#2b2420"${fit(t.b)}>${name(t.b)}</text>
    <circle cx="291" cy="${Math.max(112, mh + 16) - 26}" r="18" fill="#7a2e22"/>
    <text x="291" y="${Math.max(112, mh + 16) - 20}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="17" font-weight="800" fill="#fff">${t.points}</text></svg>`;
}
const fromHtml = (s) => { const t = document.createElement("template"); t.innerHTML = s.trim(); return t.content.firstChild; };

// ---- the board
function track(r) {
  // a route's centre line from city a to city b, straight or bent (BEND): the point at a length along
  // it and the direction there
  const [ax, ay] = POS[r.a], [bx, by] = POS[r.b];
  const [ox, oy] = BEND[[MAP.cities[r.a], MAP.cities[r.b]].sort().join("|")] || [0, 0];
  const cx = (ax + bx) / 2 + 2 * ox, cy = (ay + by) / 2 + 2 * oy, N = 48, pts = [], len = [0];
  for (let k = 0; k <= N; k++) {
    const t = k / N, u = 1 - t;
    pts.push([u * u * ax + 2 * u * t * cx + t * t * bx, u * u * ay + 2 * u * t * cy + t * t * by]);
    if (k) len.push(len[k - 1] + Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]));
  }
  const at = (s) => {
    let k = 1;
    while (k < N && len[k] < s) k++;
    const [x0, y0] = pts[k - 1], [x1, y1] = pts[k], d = Math.hypot(x1 - x0, y1 - y0) || 1, f = (s - len[k - 1]) / d;
    return [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, (x1 - x0) / d, (y1 - y0) / d];
  };
  return {length: len[N], at};
}

function corners([x, y, ux, uy], car) {
  const a = car / 2, b = TRACK / 2;
  return [[-a, -b], [a, -b], [a, b], [-a, b]].map(([p, q]) => [x + ux * p - uy * q, y + uy * p + ux * q]);
}
function apart(P, Q) {
  // the gap between two convex outlines, negative where they overlap (separating axes)
  let best = -Infinity;
  for (const poly of [P, Q]) {
    for (let i = 0; i < poly.length; i++) {
      const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length], L = Math.hypot(x2 - x1, y2 - y1) || 1;
      const ax = (y1 - y2) / L, ay = (x2 - x1) / L;
      const p = P.map(([x, y]) => x * ax + y * ay), q = Q.map(([x, y]) => x * ax + y * ay);
      best = Math.max(best, Math.min(...q) - Math.max(...p), Math.min(...p) - Math.max(...q));
    }
  }
  return best;
}
function layout() {
  // every route's spaces, [x, y, ux, uy] each, and the length of a space. The spaces start CITY_CUT
  // from a city's centre, or further where two routes leave the city so close together that their
  // spaces by the city would touch
  const R = MAP.routes, pairs = {};
  R.forEach((r, i) => (pairs[r.pair] = pairs[r.pair] || []).push(i));
  const T = R.map(track), cut = R.map(() => [CITY_CUT, CITY_CUT]);  // at city a, at city b
  const off = R.map((r, i) => (pairs[r.pair].length > 1 ? (pairs[r.pair].indexOf(i) - (pairs[r.pair].length - 1) / 2) * (TRACK + 3.2) : 0));
  const lay = (i) => {
    const r = R[i], [ca, cb] = cut[i], car = (T[i].length - ca - cb - GAP * (r.length - 1)) / r.length, places = [];
    for (let j = 0; j < r.length; j++) {
      const [x, y, ux, uy] = T[i].at(ca + j * (car + GAP) + car / 2);
      places.push([x - uy * off[i], y + ux * off[i], ux, uy]);  // a double route's track runs beside the line
    }
    const ends = [ca, T[i].length - cb].map((s) => { const [x, y, ux, uy] = T[i].at(s); return [x - uy * off[i], y + ux * off[i]]; });
    return {places, car, ends};
  };
  let L = R.map((_, i) => lay(i));
  const at = MAP.cities.map(() => []);  // the routes at each city: [route, 0 at its city a or 1 at its city b]
  R.forEach((r, i) => { at[r.a].push([i, 0]); at[r.b].push([i, 1]); });
  const near = (i, e) => { const P = L[i].places; return (e ? [P.length - 1, P.length - 2] : [0, 1]).filter((j) => j >= 0 && j < P.length).map((j) => corners(P[j], L[i].car)); };
  for (let round = 0; round < 12; round++) {
    let moved = false;
    for (const list of at) {
      for (let x = 0; x < list.length; x++) {
        for (let y = x + 1; y < list.length; y++) {
          const [i, ei] = list[x], [j, ej] = list[y];
          if (R[i].pair === R[j].pair) continue;
          const A = near(i, ei), B = near(j, ej);
          if (Math.min(...A.flatMap((P) => B.map((Q) => apart(P, Q)))) >= 1.5) continue;
          for (const [k, e] of [[i, ei], [j, ej]]) if (L[k].car > 11 && cut[k][e] < 34) { cut[k][e] += 2; moved = true; }
        }
      }
    }
    if (!moved) break;
    L = R.map((_, i) => lay(i));
  }
  return L;
}

function drawBoard() {
  const b = $("board");
  b.setAttribute("viewBox", `0 0 ${BOARD.width} ${BOARD.height}`);
  svg("image", {href: "/web/" + BOARD.image, x: 0, y: 0, width: BOARD.width, height: BOARD.height, preserveAspectRatio: "none"}, b);
  for (const [name, lon, lat] of SEAS) {
    const [x, y] = PROJECT(lon, lat);
    if (x > 40 && x < BOARD.width - 40 && y > 30 && y < BOARD.height - 20) svg("text", {class: "sea-label", x, y, "text-anchor": "middle"}, b).textContent = name;
  }
  // the layers, from the bottom: the routes, a ticket's line, the cities, their names, the pins of the human's tickets
  const routes = svg("g", {id: "routes"}, b);
  svg("g", {id: "marks"}, b);
  const cities = svg("g", {id: "cities"}, b);
  const labels = svg("g", {id: "labels"}, b);
  svg("g", {id: "pins"}, b);
  const tracks = layout();
  MAP.routes.forEach((r, i) => {
    const {places, car, ends} = tracks[i];
    const g = svg("g", {class: "route", "data-route": i}, routes);
    svg("title", {}, g).textContent = `${MAP.cities[r.a]} – ${MAP.cities[r.b]}: ${r.length} ${r.color || "grey"}, ${r.points} points`;
    for (const p of places) {
      svg("rect", {class: "space", x: (-car / 2).toFixed(2), y: -TRACK / 2, width: car.toFixed(2), height: TRACK, rx: 2.2, fill: FILL[r.color || "grey"],
                   transform: `translate(${p[0].toFixed(1)} ${p[1].toFixed(1)}) rotate(${(Math.atan2(p[3], p[2]) * 180 / Math.PI).toFixed(1)})`}, g);
    }
    // the area that takes a click: the whole track, a little wider than its spaces
    const line = [ends[0], ...places, ends[1]].map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    svg("polyline", {points: line, fill: "none", stroke: "transparent", "stroke-width": TRACK + 8, "stroke-linejoin": "round"}, g);
    const cars = svg("g", {class: "cars"}, g);
    g.addEventListener("click", (ev) => onRoute(i, ev));
    ROUTES.push({g, cars, places, car, owner: null});
  });
  MAP.cities.forEach((name, c) => {
    const [x, y] = POS[c], [dx, dy, anchor] = LABEL[name] || [12, -10, "start"];
    const g = svg("g", {class: "city", "data-city": c}, cities);
    svg("circle", {class: "dot", cx: x, cy: y, r: 8.5}, g);
    svg("circle", {class: "core", cx: x, cy: y, r: 3.4}, g);
    const label = svg("text", {class: "city-label", x: x + dx, y: y + dy, "text-anchor": anchor}, labels);
    label.textContent = name;
    CITIES.push([g, label]);
  });
}

// ---- the trains on a claimed route: boxes that stand on the board, seen from the south and above,
// with a locomotive at one end
function hull(points) {
  // the convex hull (Andrew's monotone chain)
  const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list) => {
    const out = [];
    for (const q of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    return out.slice(0, -1);
  };
  return half(p).concat(half(p.slice().reverse()));
}
const pts = (list) => list.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
function box([cx, cy, ux, uy], p0, p1, q0, q1, z0, z1) {
  // a box on the board: from p0 to p1 along the track, from q0 to q1 across it, from height z0 to z1;
  // its roof, and its outline with the sides seen
  const at = (p, q, z) => [cx + ux * p - uy * q, cy + uy * p + ux * q - z];
  const corners = [[p0, q0], [p1, q0], [p1, q1], [p0, q1]];
  const roof = corners.map(([p, q]) => at(p, q, z1));
  return {roof, outline: hull(corners.map(([p, q]) => at(p, q, z0)).concat(roof))};
}
function drawCar(parent, place, len, owner, front) {
  // front: 0 for a wagon; for the locomotive +1 or -1, the end of the car that leads
  const g = svg("g", {class: "car"}, parent);
  const a = len / 2 - 1, b = TRACK / 2 - 0.5, [cx, cy, ux, uy] = place;
  const body = box(place, -a, a, -b, b, 0, LIFT);
  svg("polygon", {class: "car-side", points: pts(body.outline), fill: SEAT_DARK[owner]}, g);
  svg("polygon", {class: "car-top", points: pts(body.roof), fill: SEAT[owner]}, g);
  if (!front) {
    const p = (s) => [cx + ux * s, cy + uy * s - LIFT];
    const [x0, y0] = p(-a + 2.4), [x1, y1] = p(a - 2.4);
    svg("line", {class: "car-shine", x1: x0.toFixed(1), y1: y0.toFixed(1), x2: x1.toFixed(1), y2: y1.toFixed(1)}, g);
    return g;
  }
  const back = [-front * a * 0.92, -front * a * 0.05];
  const cab = box(place, Math.min(...back), Math.max(...back), -b + 1.2, b - 1.2, LIFT, LIFT + 3.4);
  svg("polygon", {class: "car-side", points: pts(cab.outline), fill: SEAT_DARK[owner]}, g);
  svg("polygon", {class: "car-top", points: pts(cab.roof), fill: SEAT[owner]}, g);
  const hx = cx + ux * front * a * 0.55, hy = cy + uy * front * a * 0.55 - LIFT;
  svg("rect", {class: "chimney", x: (hx - 2.1).toFixed(1), y: (hy - 6.4).toFixed(1), width: 4.2, height: 6.4}, g);
  svg("ellipse", {class: "chimney-top", cx: hx.toFixed(1), cy: (hy - 6.4).toFixed(1), rx: 2.1, ry: 1}, g);
  svg("circle", {class: "puff", cx: (hx + 0.9).toFixed(1), cy: (hy - 10.6).toFixed(1), r: 2.6}, g);
  svg("circle", {class: "puff", cx: (hx - 1.9).toFixed(1), cy: (hy - 14.6).toFixed(1), r: 3.4}, g);
  return g;
}
function showCars(i, owner, pop = false) {
  const R = ROUTES[i];
  if (R.owner === owner) return;
  R.owner = owner;
  R.g.classList.toggle("owned", owner !== null);
  R.cars.replaceChildren();
  if (owner === null) return;
  const P = R.places, n = P.length;
  // the locomotive at the end lower on the page (on a level route, the east end), leading outwards
  const head = P[n - 1][1] - P[0][1] + 0.4 * (P[n - 1][0] - P[0][0]) > 0 ? n - 1 : 0;
  // the cars further up the page first, so that the nearer ones stand in front of them
  const order = P.map((_, j) => j).sort((x, y) => P[x][1] - P[y][1]);
  for (const j of order) {
    const g = drawCar(R.cars, P[j], R.car, owner, j !== head ? 0 : head === 0 ? -1 : 1);
    if (pop) {
      g.classList.add("pop");
      g.style.animationDelay = `${120 + Math.abs(j - head) * 110}ms`;
    }
  }
}

// ---- what is legal now, by kind
function legalNow(v) {
  const A = MAP.actions, out = {claims: {}, faceup: {}, blind: false, tickets: false, keep: new Set()};
  for (const a of v.legal) {
    if (a < A.n_claim) (out.claims[MAP.claims[a][0]] = out.claims[MAP.claims[a][0]] || []).push(a);
    else if (a < A.blind) out.faceup[a - A.draw0] = a;
    else if (a === A.blind) out.blind = true;
    else if (a === A.tickets) out.tickets = true;
    else out.keep.add(a);
  }
  return out;
}
const myTurnIn = (st) => st.view.phase !== "over" && st.view.current === st.human_seat && !st.thinking;
// what the human may do this moment: nothing while a move is sent or played out
const legalHere = () => (GAME && myTurnIn(GAME) && !BUSY && !SHOWING ? legalNow(GAME.view) : null);

const seatName = (q) => (q === GAME.human_seat ? "You" : GAME.names[q]);
function eventText(e) {
  const [actor, kind, a, b] = e, who = seatName(actor), route = (r) => `${MAP.cities[MAP.routes[r].a]} – ${MAP.cities[MAP.routes[r].b]}`;
  switch (kind) {
    case 1: return `${who} took a ${CARD[a]} card face up`;
    case 2: return `${who} drew a card from the deck`;
    case 3: { const locos = b >> 4; return `${who} claimed ${route(a)} (${MAP.routes[a].length}) with ${CARD[b & 15]}` + (locos ? ` and ${locos} locomotive${locos > 1 ? "s" : ""}` : ""); }
    case 4: return `${who} drew ${a} destination ticket${a === 1 ? "" : "s"}`;
    case 5: return `${who} kept ${a} of ${b} tickets`;
    case 6: return `${who} kept ${a} of ${b} starting tickets`;
    case 7: return "Three locomotives face up: a new row was laid";
    case 8: return "The discard pile was shuffled into the deck";
    case 9: return `${who} passed`;
    default: return `event ${kind}`;
  }
}

// ---- drawing the table
function render() {
  if (!GAME) return;
  const st = GAME, v = st.view;
  const myTurn = myTurnIn(st) && !BUSY && !SHOWING;
  const L = legalNow(myTurn ? v : {legal: []});
  renderBoard(v, L);
  renderSeats(st, v);
  renderTable(v, L);
  renderHand(v);
  renderStatus(st, v, myTurn);
  renderFeed(v);
  // the tickets ticked in a choice: all at first, and kept while the choice is sent (their pins with them)
  const choosing = v.phase === "keep" && v.current === st.human_seat && !st.thinking;
  if (!choosing) SELECTED = null;
  else if (SELECTED === null || SELECTED.length !== v.offer.length) SELECTED = v.offer.map(() => true);
  renderPins(v, choosing);
  renderOffer(v, L, myTurn);
  renderMyTickets(v);
  renderLog(v);
  if (v.phase === "over") renderResult(st, v); else $("result").classList.add("hidden");
  if (PICK !== null && !L.claims[PICK]) closeClaim();
}

function renderBoard(v, L) {
  v.routes.forEach((r, i) => {
    showCars(i, r.owner);
    const g = ROUTES[i].g;
    g.classList.toggle("closed", r.owner === null && !r.open);
    g.classList.toggle("can", !!L.claims[i]);
    g.classList.toggle("picked", PICK === i);
  });
}

// the panels stay while a game lasts (a panel's light when a card reaches it must not be cut short)
function panel(q) {
  return (PANELS[q] = PANELS[q] || h("div", {class: "seat panel", "data-seat": q}));
}
function fillPanel(st, v, q, mine) {
  const el = panel(q), s = v.seats[q], name = mine ? st.name : st.names[q];
  const thinking = st.thinking && v.current === q;
  el.style.setProperty("--seat", SEAT[q]);
  el.classList.toggle("turn", v.current === q && v.phase !== "over");
  el.replaceChildren(
    h("div", {class: "badge"}, mine ? (name.trim()[0] || "?").toUpperCase() : "AI"),
    h("div", {},
      // the spinner is drawn anew with the panel: its turn goes on from where the clock is
      h("div", {class: "who"}, name, h("small", {}, mine ? "you" : "network"),
        thinking ? h("span", {class: "thinking", title: "thinking", style: `animation-delay: -${Math.round(performance.now() % 800)}ms`}) : null),
      h("div", {class: "counts"},
        h("span", {title: "train cards in hand"}, "cards ", h("b", {}, s.cards)),
        h("span", {title: "destination tickets"}, "tickets ", h("b", {}, s.tickets)),
        h("span", {title: "trains left"}, "trains ", h("b", {}, s.trains)))),
    h("div", {class: "score", title: "points for routes so far"}, s.route_points));
}

function renderSeats(st, v) {
  const me = st.human_seat, n = v.players, box = $("players");
  const others = Array.from({length: n - 1}, (_, i) => (me + 1 + i) % n);
  if (box.dataset.order !== others.join(",")) {
    box.replaceChildren(...others.map(panel));
    box.dataset.order = others.join(",");
  }
  others.forEach((q) => fillPanel(st, v, q, false));
  const mine = $("mine");
  if (mine.firstChild !== panel(me)) mine.replaceChildren(panel(me), h("div", {class: "mine-actions"}));
  fillPanel(st, v, me, true);
  const open = !$("mytickets").classList.contains("hidden");
  mine.querySelector(".mine-actions").replaceChildren(
    h("button", {class: "wide-button", onclick: () => { $("mytickets").classList.toggle("hidden"); render(); }},
      open ? "Hide tickets" : "Your tickets", h("span", {class: "count"}, v.tickets.length)));
}

function renderTable(v, L) {
  v.face_up.forEach((c, slot) => {
    const b = MARKET[slot], card = String(c), taken = b.classList.contains("gone");
    b.classList.remove("gone");
    if (b.dataset.card !== card || taken) {
      const before = b.dataset.card;
      b.dataset.card = card;
      b.replaceChildren(fromHtml(c === null ? emptySvg() : cardSvg(c, true)));
      b.classList.toggle("empty", c === null);
      if (before !== undefined && c !== null) {  // a card laid in the row: it slides in
        b.classList.remove("dealt");
        void b.offsetWidth;
        b.classList.add("dealt");
      }
    }
    b.title = c === null ? "" : `take the ${CARD[c]} card`;
    b.disabled = L.faceup[slot] === undefined;
  });
  const deck = $("deck"), td = $("tdeck");
  deck.querySelector(".count").textContent = v.deck;
  deck.querySelector(".sub").textContent = `discard ${v.discard.reduce((s, x) => s + x, 0)}`;
  deck.disabled = !L.blind;
  td.querySelector(".count").textContent = v.ticket_deck;
  td.disabled = !L.tickets;
}

function renderHand(v) {
  const kinds = v.hand.map((n, t) => [t, n]).filter(([, n]) => n > 0);
  const box = $("hand"), n = kinds.length;
  if (!n) { box.replaceChildren(h("div", {class: "empty-hand"}, "no cards in hand")); return; }
  box.replaceChildren(...kinds.map(([t, k], i) => {
    const mid = (n - 1) / 2, rot = (i - mid) * 3.2, lift = Math.abs(i - mid) * 0.55;
    return h("div", {class: "stack" + (HAND_NEW.has(t) ? " new" : ""), "data-card": t, style: `transform: translateY(${lift}vh) rotate(${rot}deg)`,
                     title: `${k} ${CARD[t]}`}, fromHtml(cardSvg(t, false)), h("span", {class: "n"}, k));
  }));
}

function renderStatus(st, v, myTurn) {
  const s = $("status");
  s.classList.remove("hidden");
  let text;
  if (v.phase === "over") text = "The game is over";
  else if (BUSY) text = "…";
  else if (st.thinking) text = `${st.names[v.current]} is thinking…`;
  else if (!myTurn) text = "…";
  else if (v.phase === "keep") text = `Choose your tickets: keep at least ${v.keep_min}`;
  else if (v.phase === "draw2") text = "Take a second card (not a face-up locomotive)";
  else text = "Your turn: take two cards, claim a route, or draw tickets";
  s.textContent = text;
  s.classList.toggle("mine", myTurn);
  const banner = $("banner");
  if (v.last_round_left >= 0 && v.phase !== "over") {
    banner.textContent = `Last round: ${v.last_round_left} turn${v.last_round_left === 1 ? "" : "s"} left`;
    banner.classList.remove("hidden");
  } else banner.classList.add("hidden");
}

function say(e, k) {
  // the line at the top: who did what; it changes, and fades in, only for a new move (k counts the moves)
  const feed = $("feed");
  if (feed.dataset.event === String(k)) return;
  feed.dataset.event = String(k);
  feed.replaceChildren(h("li", {style: `--seat:${e[0] < GAME.view.players ? SEAT[e[0]] : "#868e96"}`}, eventText(e)));
}
function renderFeed(v) {
  if (v.events.length) say(v.events[v.events.length - 1], v.events_total - 1);
  else { $("feed").replaceChildren(); delete $("feed").dataset.event; }
}

function renderOffer(v, L, myTurn) {
  const box = $("offer");
  if (!myTurn || SELECTED === null) { box.classList.add("hidden"); return; }
  const mask = SELECTED.reduce((m, on, j) => m | (on ? 1 << j : 0), 0);
  const action = MAP.actions.keep1 + mask - 1, ok = mask > 0 && L.keep.has(action);
  const kept = SELECTED.filter(Boolean).length;
  box.replaceChildren(
    h("h3", {}, v.opening ? "Your starting tickets" : "New tickets"),
    h("p", {class: "hint"}, `Keep at least ${v.keep_min}. A completed ticket adds its points at the end, a failed one subtracts them.`),
    ...v.offer.map((t, j) => h("button", {class: "ticket" + (SELECTED[j] ? "" : " off"), onclick: () => { SELECTED[j] = !SELECTED[j]; render(); },
                                         onmouseenter: () => mark(MAP.tickets[t]), onmouseleave: () => mark(null)},
      fromHtml(ticketSvg(MAP.tickets[t])), h("span", {class: "tick"}, SELECTED[j] ? "✓" : "–"), pinMark(MAP.tickets[t]))),
    h("button", {class: "primary", ...(ok ? {} : {disabled: ""}), onclick: () => { mark(null); play(action); }},
      ok ? `Keep ${kept} ticket${kept === 1 ? "" : "s"}` : `Keep at least ${v.keep_min}`));
  box.classList.remove("hidden");
}

function renderMyTickets(v) {
  const box = $("mytickets");
  if (box.classList.contains("hidden")) return;
  box.replaceChildren(h("h3", {}, "Your tickets"),
    ...(v.tickets.length ? v.tickets.map((t) => h("div", {class: "ticket" + (t.done ? "" : " open"), onmouseenter: () => mark(t), onmouseleave: () => mark(null)},
      fromHtml(ticketSvg(t)), t.done ? h("span", {class: "tick"}, "✓") : null, pinMark(t))) : [h("p", {class: "hint"}, "none yet")]));
}

function renderLog(v) {
  const box = $("log");
  if (box.classList.contains("hidden")) return;
  box.querySelector("ol").replaceChildren(...v.events.map((e) => h("li", {}, h("span", {style: `--seat:${e[0] < v.players ? SEAT[e[0]] : "#868e96"}`}), eventText(e))));
}

function renderResult(st, v) {
  const f = v.final, n = v.players, order = Array.from({length: n}, (_, q) => q).sort((a, b) => f.ranks[a] - f.ranks[b] || a - b);
  const name = (q) => (q === st.human_seat ? `${st.name} (you)` : st.names[q]);
  // ranks count from 0, shared by a tie; the bonus of the longest trail is what the total holds beyond routes and tickets
  const rows = order.map((q) => {
    const bonus = f.total[q] - f.route[q] - f.tickets[q];
    return h("tr", {class: q === st.human_seat ? "me" : ""},
      h("td", {}, String(Math.floor(f.ranks[q]) + 1)), h("td", {}, h("span", {class: "dot", style: `background:${SEAT[q]}`}), name(q)),
      h("td", {}, f.route[q]), h("td", {}, (f.tickets[q] > 0 ? "+" : "") + f.tickets[q]),
      h("td", {}, `${f.longest[q]}`, bonus ? h("b", {}, ` +${bonus}`) : null), h("td", {}, h("b", {}, f.total[q])));
  });
  const won = f.ranks[st.human_seat] === Math.min(...f.ranks);
  const sheet = $("result").querySelector(".sheet");
  sheet.replaceChildren(
    h("h2", {}, won ? "You win" : `${name(order[0])} wins`),
    h("table", {class: "standings"}, h("tr", {}, ...["", "player", "routes", "tickets", "longest trail", "total"].map((x) => h("th", {}, x))), ...rows),
    h("div", {class: "hands"}, ...order.map((q) => h("div", {}, h("h5", {}, name(q)),
      h("ul", {}, ...f.hands[q].map((t) => h("li", {class: t.done ? "done" : "fail"}, `${t.done ? "✓" : "✗"} ${MAP.cities[t.a]} – ${MAP.cities[t.b]} (${t.points})`)))))),
    h("div", {class: "row", style: "display:flex;gap:12px;margin-top:10px"},
      h("button", {class: "primary", onclick: newGame}, "A new game"),
      h("button", {class: "primary", style: "background:#5a4d44", onclick: () => $("result").classList.add("hidden")}, "Look at the board")));
  if (!$("result").dataset.shown) { $("result").classList.remove("hidden"); $("result").dataset.shown = "1"; }
}

function mark(t) {
  // a ticket under the pointer: its cities lit, a line between them, its pins larger and the others faint
  CITIES.flat().forEach((e) => e.classList.remove("mark"));
  $("marks").replaceChildren();
  const pins = $("pins");
  pins.classList.toggle("focusing", !!t);
  for (const g of pins.children) g.classList.toggle("focus", !!t && g.dataset.key === ticketKey(t));
  if (!t) return;
  for (const e of [...CITIES[t.a], ...CITIES[t.b]]) e.classList.add("mark");
  const [ax, ay] = POS[t.a], [bx, by] = POS[t.b];
  svg("line", {class: "ticket-line", x1: ax, y1: ay, x2: bx, y2: by}, $("marks"));
}

// ---- the pins of the human's tickets: one on both cities of every ticket held and not completed, and of
// every ticket ticked in a choice. A ticket keeps its colour while it is held; its card shows the same pin.
const PIN = ["#ff6b6b", "#4dabf7", "#ffd43b", "#b197fc", "#69db7c", "#ff922b", "#3bc9db", "#f783ac", "#a9e34b", "#748ffc"];
const PINS = new Map();  // a ticket's key -> the index of its colour
const ticketKey = (t) => `${Math.min(t.a, t.b)}-${Math.max(t.a, t.b)}`;
const pinColour = (t) => (PINS.has(ticketKey(t)) ? PIN[PINS.get(ticketKey(t)) % PIN.length] : null);
function drawPin(c, colour) {
  const [x, y] = POS[c], g = svg("g", {class: "pin"}), body = svg("g", {class: "pin-body", style: `transform-origin: ${x.toFixed(1)}px ${y.toFixed(1)}px`}, g);
  svg("ellipse", {class: "pin-shadow", cx: x + 4.4, cy: y + 1.5, rx: 5.8, ry: 2.2}, body);
  svg("line", {class: "pin-needle", x1: x, y1: y, x2: x, y2: y - 19}, body);
  svg("circle", {class: "pin-head", cx: x, cy: y - 26, r: 8.8, fill: colour}, body);
  svg("circle", {class: "pin-shine", cx: x - 2.8, cy: y - 29.2, r: 2.7}, body);
  return g;
}
function renderPins(v, choosing) {
  const want = v.phase === "over" ? [] : v.tickets.filter((t) => !t.done);
  if (choosing && SELECTED) v.offer.forEach((t, j) => { if (SELECTED[j]) want.push(MAP.tickets[t]); });
  const keys = new Set(want.map(ticketKey));
  for (const k of [...PINS.keys()]) if (!keys.has(k)) PINS.delete(k);
  for (const t of want) {
    const k = ticketKey(t), used = new Set(PINS.values());
    if (PINS.has(k)) continue;
    let c = 0;
    while (used.has(c) && c < PIN.length) c++;
    PINS.set(k, c < PIN.length ? c : PINS.size);
  }
  // the pins already standing stay (a new one drops in); several in one city fan out, their heads apart
  const layer = $("pins"), have = new Map([...layer.children].map((g) => [`${g.dataset.key}|${g.dataset.city}`, g]));
  const count = {}, slots = {}, keep = new Set();
  for (const t of want) for (const c of [t.a, t.b]) count[c] = (count[c] || 0) + 1;
  for (const t of want) {
    for (const c of [t.a, t.b]) {
      const id = `${ticketKey(t)}|${c}`, slot = (slots[c] = (slots[c] || 0) + 1) - 1, n = count[c];
      let g = have.get(id);
      if (!g) {
        g = drawPin(c, pinColour(t));
        g.dataset.key = ticketKey(t);
        g.dataset.city = c;
        g.classList.add("fresh");
        setTimeout(() => g.classList.remove("fresh"), 600);
        layer.append(g);
      }
      const [x, y] = POS[c], lean = (slot - (n - 1) / 2) * Math.min(n === 2 ? 44 : 40, 150 / Math.max(1, n - 1));
      g.setAttribute("transform", `rotate(${lean} ${x.toFixed(1)} ${y.toFixed(1)})`);
      keep.add(id);
    }
  }
  for (const [id, g] of have) if (!keep.has(id)) g.remove();
}
const pinMark = (t) => { const c = pinColour(t); return c ? h("span", {class: "pin-mark", style: `--pin:${c}`, title: "its pin on the map"}) : null; };

// ---- playing a move out: what flies where, the trains set down, the sounds
const centre = (r) => [r.left + r.width / 2, r.top + r.height / 2];
const pose = (x, y, s, deg = 0) => `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) rotate(${deg}deg) scale(${s})`;
function piece(html, w, hgt) {
  const el = h("div", {class: "fly"});
  el.append(fromHtml(html));
  el.style.width = `${w}px`;
  el.style.height = `${hgt}px`;
  $("fx").append(el);
  return el;
}
function fly(html, w, hgt, [x0, y0], [x1, y1], o = {}) {
  // a card of w x hgt from one centre to another: it flies, stays a moment, and fades
  const {ms = 520, s0 = 1, s1 = 0.5, hold = 0, fade = 160, delay = 0, onLand = null} = o;
  const el = piece(html, w, hgt), total = ms + hold + fade;
  const a = [x0 - w / 2, y0 - hgt / 2], b = [x1 - w / 2, y1 - hgt / 2];
  const anim = el.animate([
    {transform: pose(a[0], a[1], s0), opacity: 1, offset: 0, easing: "cubic-bezier(.4, 0, .2, 1)"},
    {transform: pose(b[0], b[1], s1), opacity: 1, offset: ms / total},
    {transform: pose(b[0], b[1], s1), opacity: 1, offset: (ms + hold) / total},
    {transform: pose(b[0], b[1], s1 * 0.92), opacity: 0, offset: 1}], {duration: total, delay, fill: "both"});
  if (onLand) setTimeout(onLand, delay + ms);
  return anim.finished.then(() => el.remove(), () => el.remove());
}
function seatPoint(q) {
  // where a card lands by a seat's panel: its right part
  const el = PANELS[q], r = el && el.getBoundingClientRect();
  return r && r.width ? [r.right - 6 * unit(), r.top + r.height / 2] : null;
}
function pulse(q) {
  const el = PANELS[q];
  if (!el) return;
  el.classList.remove("got");
  void el.offsetWidth;
  el.classList.add("got");
  setTimeout(() => el.classList.remove("got"), 700);
}
function lightUp(el, cls, ms) {
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), ms);
}
function toSeat(html, from, q, mine, card) {
  // a drawn card: to the seat's panel, or into the human's hand (on its stack of the colour, if any)
  const r = from.getBoundingClientRect();
  if (mine) {
    const stack = card === null ? null : document.querySelector(`#hand .stack[data-card="${card}"]`);
    return fly(html, r.width, r.height, centre(r), centre((stack || $("hand")).getBoundingClientRect()), {ms: 480, s1: 0.65, hold: 60});
  }
  const to = seatPoint(q);
  return to ? fly(html, r.width, r.height, centre(r), to, {ms: 520, s1: 0.5, hold: 300, onLand: () => pulse(q)}) : sleep(300);
}
function fan(q, cards) {
  // the cards an opponent pays for a claim: laid out by its panel one by one, then off to the discard
  const p = seatPoint(q);
  if (!p) return sleep(300);
  const u = unit(), w = 5.6 * u, hgt = 8.7 * u, n = cards.length, step = Math.min(3.6 * u, (24 * u) / n);
  const [dx, dy] = centre($("deck").getBoundingClientRect());
  const shown = 70 * (n - 1) + 200 + 350;
  return Promise.all(cards.map((t, k) => {
    const el = piece(cardSvg(t, false), w, hgt), x = p[0] + 5 * u + k * step - w / 2, y = p[1] - hgt / 2, deg = (k - (n - 1) / 2) * 5;
    const total = shown + 40 * k + 380;
    return el.animate([
      {transform: pose(x, y, 0.3, deg), opacity: 0, offset: 0},
      {transform: pose(x, y, 0.3, deg), opacity: 0, offset: (70 * k) / total},
      {transform: pose(x, y, 1, deg), opacity: 1, offset: (70 * k + 200) / total},
      {transform: pose(x, y, 1, deg), opacity: 1, offset: (shown + 40 * k) / total, easing: "ease-in"},
      {transform: pose(dx - w / 2, dy - hgt / 2, 0.6), opacity: 0, offset: 1}], {duration: total, fill: "both"})
      .finished.then(() => el.remove(), () => el.remove());
  }));
}
async function setTrains(i, q) {
  const n = MAP.routes[i].length;
  sfx("whistle");
  showCars(i, q, true);
  for (let k = 0; k < n; k++) sfx("tok", 0.12 + k * 0.11);
  await sleep(120 + n * 110 + 300);
  lightUp(ROUTES[i].g, "flash", 2300);
}
function ticketsOut(q, n, mine) {
  // drawn tickets, one after another: to the panel, or to where the human's choice opens
  const r = $("tdeck").getBoundingClientRect(), u = unit(), runs = [];
  for (let k = 0; k < n; k++) {
    const to = mine ? [18 * u, (15 + 13 * k) * u] : seatPoint(q);
    if (!to) break;
    runs.push(fly(ticketBackSvg(), r.width, r.height, centre(r), to, {delay: 100 + 120 * k, ms: 420, s1: mine ? 1.2 : 0.45, hold: mine ? 0 : 200}));
    sfx("ticket", 0.1 + 0.12 * k);
  }
  return Promise.all(runs).then(() => { if (!mine) pulse(q); });
}
function ticketsBack(q, n) {
  // the tickets an opponent gives back go under the ticket pile
  const p = seatPoint(q), r = $("tdeck").getBoundingClientRect(), runs = [];
  for (let k = 0; p && k < n; k++) runs.push(fly(ticketBackSvg(), r.width, r.height, p, centre(r), {delay: 120 * k, ms: 420, s0: 0.45, s1: 1, fade: 160}));
  return Promise.all(runs);
}
function takenSlot(card, used) {
  // the slot of the row a card was taken from: one that showed it and has not given a card already
  const row = GAME.view.face_up;
  for (let s = 0; s < row.length; s++) if (row[s] === card && !used.has(s)) { used.add(s); return s; }
  return null;
}
async function playOut(e, st, used, k) {
  const [q, kind, a, b] = e, mine = q === st.human_seat;
  say(e, k);
  if (CALM || document.hidden) {
    if (kind !== 6) sfx(kind === 3 ? "tok" : kind === 4 || kind === 5 ? "ticket" : "card");
    await sleep(kind === 3 ? 700 : 400);
    return;
  }
  if (kind === 1) {
    sfx("card");
    const s = takenSlot(a, used);
    if (s !== null) MARKET[s].classList.add("gone");
    await toSeat(cardSvg(a, true), s === null ? $("deck") : MARKET[s], q, mine, a);
  } else if (kind === 2) {
    sfx("card");
    await toSeat(deckSvg(), $("deck"), q, mine, null);
  } else if (kind === 3) {
    const n = MAP.routes[a].length, locos = b >> 4;
    if (!mine) await fan(q, [...Array(n - locos).fill(b & 15), ...Array(locos).fill(8)]);
    await setTrains(a, q);
  } else if (kind === 4) {
    await ticketsOut(q, a, mine);
  } else if (kind === 5 || kind === 6) {
    if (!mine && b > a) await ticketsBack(q, b - a);
    else await sleep(250);
  } else if (kind === 7) {
    sfx("card");
    lightUp($("market"), "relaid", 600);
    await sleep(500);
  } else if (kind === 8) {
    lightUp($("deck"), "shuffled", 600);
    await sleep(500);
  } else {
    await sleep(900);
  }
}

// the moves that came after the game as shown
function freshEvents(before, st) {
  const n = st.view.events_total - before.view.events_total;
  return n > 0 ? st.view.events.slice(-Math.min(n, st.view.events.length)) : [];
}
function receive(st) {
  LATEST = st;
  present();
}
async function present() {
  // play out the moves of the newest game in turn, then show it; once more if a newer one came meanwhile
  if (SHOWING) return;
  SHOWING = true;
  try {
    while (LATEST && LATEST !== GAME) {
      const st = LATEST, before = GAME;
      const fresh = before && before.id === st.id ? freshEvents(before, st) : [];
      if (fresh.length) {
        render();  // the controls off while the moves are played out
        const used = new Set(), first = st.view.events_total - fresh.length;
        if (fresh.length <= 8) for (const [k, e] of fresh.entries()) await playOut(e, st, used, first + k);
      }
      if (!LATEST || LATEST.id !== st.id) break;  // a new game was started meanwhile
      HAND_NEW = new Set(before && before.id === st.id ? st.view.hand.flatMap((k, t) => (k > before.view.hand[t] ? [t] : [])) : []);
      GAME = st;
      if (LATEST !== st) render();
    }
  } catch (e) {
    console.warn(e);
    GAME = LATEST || GAME;
  } finally {
    SHOWING = false;
    render();
    chime();
  }
}
function chime() {
  // a bell when the human comes to move
  const now = !!GAME && myTurnIn(GAME);
  if (now && !MY_TURN) sfx("turn");
  MY_TURN = now;
}

// ---- sounds, made here with the Web Audio API: a card, a ticket, a train set down, a whistle, a bell
const SOUND = {on: true, ctx: null, out: null, noise: null};
try { SOUND.on = localStorage.getItem("deepttr.sound") !== "off"; } catch (e) { /* no storage */ }
function wakeSound() {
  // a browser lets a page make sound only after a click or a key
  if (!SOUND.on) return;
  try {
    if (!SOUND.ctx) {
      const C = window.AudioContext || window.webkitAudioContext;
      if (!C) return;
      const c = new C(), n = c.sampleRate;
      SOUND.out = c.createGain();
      SOUND.out.gain.value = 0.8;
      SOUND.out.connect(c.destination);
      SOUND.noise = c.createBuffer(1, n, n);
      const d = SOUND.noise.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
      SOUND.ctx = c;
    }
    if (SOUND.ctx.state === "suspended") SOUND.ctx.resume();
  } catch (e) { SOUND.ctx = null; }
}
function tone(t, f0, f1, dur, type, vol, attack = 0.008) {
  const c = SOUND.ctx, o = c.createOscillator(), g = c.createGain();
  o.type = type;
  o.frequency.setValueAtTime(f0, t);
  if (f1 !== f0) o.frequency.exponentialRampToValueAtTime(f1, t + dur);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(vol, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g);
  g.connect(SOUND.out);
  o.start(t);
  o.stop(t + dur + 0.02);
}
function hiss(t, dur, freq, q, vol) {
  const c = SOUND.ctx, s = c.createBufferSource(), f = c.createBiquadFilter(), g = c.createGain();
  s.buffer = SOUND.noise;
  f.type = "bandpass";
  f.frequency.value = freq;
  f.Q.value = q;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(vol, t + 0.012);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  s.connect(f);
  f.connect(g);
  g.connect(SOUND.out);
  s.start(t, Math.random() * 0.5);
  s.stop(t + dur + 0.02);
}
function sfx(name, after = 0) {
  const c = SOUND.ctx;
  if (!SOUND.on || !c || c.state !== "running") return;
  const t = c.currentTime + 0.01 + after;
  if (name === "card") { hiss(t, 0.13, 3000, 0.8, 0.35); tone(t, 240, 150, 0.07, "sine", 0.12); }
  else if (name === "ticket") hiss(t, 0.1, 1900, 1.1, 0.3);
  else if (name === "tok") { tone(t, 700, 420, 0.075, "triangle", 0.3); hiss(t, 0.03, 1400, 2.5, 0.12); }
  else if (name === "whistle") { tone(t, 620, 600, 0.36, "sine", 0.1, 0.05); tone(t, 780, 760, 0.36, "sine", 0.08, 0.05); }
  else if (name === "turn") {
    tone(t, 659.3, 659.3, 0.55, "sine", 0.28); tone(t, 1318.5, 1318.5, 0.3, "sine", 0.05);
    tone(t + 0.16, 987.8, 987.8, 0.8, "sine", 0.26); tone(t + 0.16, 1975.5, 1975.5, 0.35, "sine", 0.04);
  }
}
function soundButton() {
  const b = $("btn-sound");
  b.classList.toggle("off", !SOUND.on);
  b.title = SOUND.on ? "Sound on" : "Sound off";
  b.setAttribute("aria-pressed", String(SOUND.on));
}

// ---- claiming a route
function onRoute(i, ev) {
  const L = legalHere(), options = L && L.claims[i];
  if (!options) return;
  const v = GAME.view;
  PICK = i;
  let k = 0;
  const r = MAP.routes[i], box = $("claim");
  const draw = () => {
    const a = options[k], color = MAP.claims[a][1], have = v.hand[color], colored = Math.min(have, r.length), locos = r.length - colored;
    const pay = [...Array(colored).fill(color), ...Array(locos).fill(8)];
    box.replaceChildren(
      h("h4", {}, `${MAP.cities[r.a]} – ${MAP.cities[r.b]}`),
      h("div", {class: "sub"}, `${r.length} ${r.color || "grey"} space${r.length > 1 ? "s" : ""} · ${r.points} point${r.points > 1 ? "s" : ""}`),
      h("div", {class: "pay"}, ...pay.map((c) => fromHtml(cardSvg(c, false))),
        options.length > 1 ? h("button", {class: "ghost arrow", title: "pay with another colour", onclick: () => { k = (k + 1) % options.length; draw(); }}, "⇄") : null),
      h("div", {class: "row"},
        h("button", {class: "primary", onclick: () => { closeClaim(); play(a); }}, "Claim"),
        h("button", {class: "ghost", onclick: closeClaim}, "Cancel")));
  };
  draw();
  box.classList.add("panel");
  box.classList.remove("hidden");
  const w = box.offsetWidth, hgt = box.offsetHeight;
  box.style.left = `${Math.max(8, Math.min(ev.clientX + 14, innerWidth - w - 8))}px`;
  box.style.top = `${Math.max(8, Math.min(ev.clientY + 14, innerHeight - hgt - 8))}px`;
  renderBoard(v, legalNow(v));
}
function closeClaim() {
  PICK = null;
  $("claim").classList.add("hidden");
  document.querySelectorAll(".route.picked").forEach((g) => g.classList.remove("picked"));
}

// ---- moves and the network's answers
async function play(action) {
  if (BUSY || SHOWING || !GAME) return;
  BUSY = true;
  render();
  try {
    receive(await api(`/api/game/${GAME.id}/move`, {action}));
  } catch (e) {
    $("status").textContent = `Not accepted: ${e.message}`;
  } finally {
    BUSY = false;
    if (!SHOWING) render();
    follow();
  }
}

async function follow() {
  if (POLLING) return;
  POLLING = true;
  try {
    while (LATEST && LATEST.thinking) {
      await sleep(POLL_MS);
      const id = LATEST && LATEST.id;
      if (!id) break;
      const st = await api(`/api/game/${id}`);
      if (LATEST && LATEST.id === id) receive(st);
    }
  } catch (e) {
    $("status").textContent = `Lost the server: ${e.message}`;
  } finally {
    POLLING = false;
  }
}

// ---- the lobby
let PLAYERS = 2;
function lobby() {
  location.hash = "";
  GAME = LATEST = null;
  $("status").classList.add("hidden");
  $("lobby").classList.remove("hidden");
  $("result").classList.add("hidden");
  delete $("result").dataset.shown;
  try { $("name").value = localStorage.getItem("deepttr.name") || ""; } catch (e) { /* no storage */ }
  const seats = () => {
    document.querySelectorAll("#seats button").forEach((b) => b.classList.toggle("on", +b.dataset.players === PLAYERS));
    $("seats-note").textContent = `You and ${PLAYERS - 1} AI opponent${PLAYERS > 2 ? "s" : ""}, all played by the network.`;
    const sel = $("seat"), keep = sel.value;
    sel.replaceChildren(h("option", {value: ""}, "a seat at random"),
      ...Array.from({length: PLAYERS}, (_, q) => h("option", {value: q}, q === 0 ? "first to move" : q === PLAYERS - 1 ? "last to move" : `seat ${q + 1}`)));
    sel.value = keep !== "" && +keep < PLAYERS ? keep : "";
  };
  document.querySelectorAll("#seats button").forEach((b) => { b.onclick = () => { PLAYERS = +b.dataset.players; seats(); }; });
  seats();
  const consent = $("consent");
  $("consentrow").classList.toggle("hidden", !MAP.recording);
  $("go").disabled = MAP.recording && !consent.checked;
  consent.onchange = () => { $("go").disabled = MAP.recording && !consent.checked; };
  $("go").onclick = async () => {
    $("go").disabled = true;
    const name = $("name").value.trim() || "Guest", seat = $("seat").value;
    try { localStorage.setItem("deepttr.name", name); } catch (e) { /* no storage */ }
    try {
      const st = await api("/api/new", {name, consent: consent.checked, players: PLAYERS, seat: seat === "" ? null : +seat});
      location.hash = st.id;
      start(st);
    } catch (e) {
      $("seats-note").textContent = `Could not start: ${e.message}`;
      $("go").disabled = false;
    }
  };
}
function newGame() { closeClaim(); $("offer").classList.add("hidden"); lobby(); }

function start(st) {
  $("lobby").classList.add("hidden");
  ROUTES.forEach((_, i) => showCars(i, null));
  for (const q of Object.keys(PANELS)) delete PANELS[q];
  $("players").dataset.order = "";
  $("mine").replaceChildren();
  HAND_NEW = new Set();
  MY_TURN = false;
  delete $("feed").dataset.event;
  $("pins").replaceChildren();
  PINS.clear();
  GAME = LATEST = st;
  render();
  chime();
  follow();
}

async function init() {
  [MAP, BOARD] = await Promise.all([api("/api/map"), fetch("/web/board.json").then((r) => r.json())]);
  PROJECT = projector(BOARD);
  POS = MAP.layout.map((c) => PROJECT(c.lon, c.lat));
  drawBoard();
  MARKET = Array.from({length: 5}, (_, slot) => h("button", {class: "tcard land", disabled: "", onclick: () => {
    const L = legalHere();
    if (L && L.faceup[slot] !== undefined) play(L.faceup[slot]);
  }}));
  $("market").replaceChildren(...MARKET);
  $("deck").replaceChildren(fromHtml(deckSvg()), h("span", {class: "count"}), h("span", {class: "sub"}));
  $("deck").title = "draw a card from the deck";
  $("deck").onclick = () => { const L = legalHere(); if (L && L.blind) play(MAP.actions.blind); };
  $("tdeck").replaceChildren(fromHtml(ticketBackSvg()), h("span", {class: "count"}));
  $("tdeck").title = "draw destination tickets";
  $("tdeck").onclick = () => { const L = legalHere(); if (L && L.tickets) play(MAP.actions.tickets); };
  $("btn-new").onclick = newGame;
  $("btn-log").onclick = () => { $("log").classList.toggle("hidden"); render(); };
  $("btn-sound").onclick = () => {
    SOUND.on = !SOUND.on;
    try { localStorage.setItem("deepttr.sound", SOUND.on ? "on" : "off"); } catch (e) { /* no storage */ }
    wakeSound();
    sfx("card");
    soundButton();
  };
  soundButton();
  addEventListener("pointerdown", wakeSound, true);
  addEventListener("keydown", wakeSound, true);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeClaim(); });
  const id = location.hash.slice(1);
  if (id) {
    try { start(await api(`/api/game/${id}`)); return; } catch (e) { /* the game is gone: a new one */ }
  }
  lobby();
}

init().catch((e) => { $("status").classList.remove("hidden"); $("status").textContent = `Error: ${e.message}`; });
