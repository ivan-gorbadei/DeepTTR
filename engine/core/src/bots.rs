//! Baseline players without a network: the fixed opponents of every strength measurement.
//!
//! - `Bot::Heuristic`, a ticket builder: keep the tightest set of tickets on offer (most pairs
//!   sharing a city, then points, then fewer tickets); claim a route on the shortest paths of its
//!   open tickets, or one that completes a ticket; else draw the colours those paths need; else
//!   claim the best route it can pay for (a block of another player's network counts). A closed
//!   double is no path.
//! - `Bot::Hoard`, a card hoarder: keep the fewest tickets with the fewest points, never draw
//!   tickets, draw cards (the colour the longest open routes want, else blind) until one of the
//!   longest routes still open is affordable, and claim it; once no route of at least 4 is left to
//!   take, claim whatever fits - which ends the game early, before the others' tickets are done.
//!   The strategy is that of HoardCards (Õunmaa, "Analyzing AI players on different Ticket to Ride
//!   variants", MSc thesis, Leiden University, 2022). That player claims nothing for its first
//!   twenty rounds; this one claims as soon as it can pay.
//! - `Bot::Random`: uniformly random legal moves.

use crate::map::{Map, LOCO, MAX_CITIES, MAX_ROUTES, N_COLORS};
use crate::rng::{GameRng, Xoshiro};
use crate::state::{Phase, Rules, State, FACE_UP, NONE};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bot {
    Heuristic,
    Hoard,
    /// uniformly random legal moves
    Random,
}

impl Bot {
    pub fn parse(name: &str) -> Option<Bot> {
        match name {
            "heuristic" => Some(Bot::Heuristic),
            "hoard" => Some(Bot::Hoard),
            "random" => Some(Bot::Random),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Bot::Heuristic => "heuristic",
            Bot::Hoard => "hoard",
            Bot::Random => "random",
        }
    }

    /// The move of the player to act in `s`; `legal` = its legal moves (not empty).
    pub fn choose(&self, map: &Map, s: &State, legal: &[u16], rng: &mut Xoshiro) -> u16 {
        debug_assert!(!legal.is_empty());
        if legal.len() == 1 {
            return legal[0];
        }
        match self {
            Bot::Heuristic => heuristic(map, s, legal, rng),
            Bot::Hoard => hoard(map, s, legal, rng),
            Bot::Random => legal[rng.below(legal.len() as u32) as usize],
        }
    }
}

fn random_of(legal: &[u16], rng: &mut Xoshiro) -> u16 {
    legal[rng.below(legal.len() as u32) as usize]
}

/// The cities on the routes of player `p`.
fn in_network(map: &Map, s: &State, p: usize) -> [bool; MAX_CITIES] {
    let mut inn = [false; MAX_CITIES];
    for (r, route) in map.routes.iter().enumerate() {
        if s.route_owner[r] == p as u8 {
            inn[route.a as usize] = true;
            inn[route.b as usize] = true;
        }
    }
    inn
}

/// The routes on a shortest path (in trains still to lay) of each open ticket of `p`, the tickets
/// on offer included when `p` is to act: Dijkstra over the open routes and `p`'s own, `p`'s own
/// costing nothing; cities at an equal distance are settled in the order of their ids, and of two
/// equal paths the one found first is kept.
fn ticket_paths(map: &Map, s: &State, p: usize) -> [bool; MAX_ROUTES] {
    let mut mark = [false; MAX_ROUTES];
    let mut tickets: Vec<u8> = (0..map.n_tickets() as u8)
        .filter(|&t| s.tickets_held[p] >> t & 1 == 1)
        .collect();
    if p == s.current as usize {
        tickets.extend_from_slice(s.pending_tickets());
    }
    if tickets.is_empty() {
        return mark;
    }
    let nc = map.n_cities();
    let mut adj: Vec<Vec<(u8, u32, u16)>> = vec![Vec::new(); nc];
    for (i, route) in map.routes.iter().enumerate() {
        let owner = s.route_owner[i];
        if owner != NONE && owner != p as u8 {
            continue;
        }
        if owner == NONE && !s.route_open(map, i) {
            continue;
        }
        let w = if owner == p as u8 { 0 } else { route.length as u32 };
        adj[route.a as usize].push((route.b, w, i as u16));
        adj[route.b as usize].push((route.a, w, i as u16));
    }
    let comp = s.components(map, p);
    let mut seen = 0u64;
    for &t in &tickets {
        if seen >> t & 1 == 1 {
            continue;
        }
        seen |= 1u64 << t;
        let tk = &map.tickets[t as usize];
        if comp[tk.a as usize] == comp[tk.b as usize] {
            continue;
        }
        let mut dist = vec![u32::MAX; nc];
        let mut prev: Vec<Option<(u8, u16)>> = vec![None; nc];
        let mut heap = BinaryHeap::new();
        dist[tk.a as usize] = 0;
        heap.push(Reverse((0u32, tk.a)));
        while let Some(Reverse((d, u))) = heap.pop() {
            if d != dist[u as usize] {
                continue;
            }
            if u == tk.b {
                break;
            }
            for &(v, w, ri) in &adj[u as usize] {
                let nd = d + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    prev[v as usize] = Some((u, ri));
                    heap.push(Reverse((nd, v)));
                }
            }
        }
        if dist[tk.b as usize] == u32::MAX {
            continue;
        }
        let mut node = tk.b;
        while let Some((u, ri)) = prev[node as usize] {
            mark[ri as usize] = true;
            node = u;
        }
    }
    mark
}

/// Points of `p`'s open tickets that claiming route `r` would complete.
fn points_closed(map: &Map, s: &State, p: usize, comp: &[u8; MAX_CITIES], r: usize) -> i32 {
    let route = &map.routes[r];
    let (ra, rb) = (comp[route.a as usize], comp[route.b as usize]);
    if ra == rb {
        return 0;
    }
    let mut pts = 0;
    for (t, tk) in map.tickets.iter().enumerate() {
        if s.tickets_held[p] >> t & 1 == 0 {
            continue;
        }
        let (ca, cb) = (comp[tk.a as usize], comp[tk.b as usize]);
        if ca != cb && ((ca == ra && cb == rb) || (ca == rb && cb == ra)) {
            pts += tk.points as i32;
        }
    }
    pts
}

/// The keep mask of `s`'s offer maximising `key(tickets kept)`, among those keeping at least
/// `keep_min`; the first best in mask order.
fn best_keep<K: PartialOrd>(s: &State, key: impl Fn(&[u8]) -> K) -> u8 {
    let pend = s.pending_tickets();
    let mut best: Option<(K, u8)> = None;
    for bits in 1u8..(1u8 << pend.len()) {
        if (bits.count_ones() as u8) < s.keep_min {
            continue;
        }
        let kept: Vec<u8> = (0..pend.len())
            .filter(|&i| bits >> i & 1 == 1)
            .map(|i| pend[i])
            .collect();
        let k = key(&kept);
        if best.as_ref().is_none_or(|(bk, _)| k > *bk) {
            best = Some((k, bits));
        }
    }
    best.map_or(1, |(_, b)| b)
}

fn heuristic(map: &Map, s: &State, legal: &[u16], rng: &mut Xoshiro) -> u16 {
    let p = s.current as usize;
    let n = s.n();
    match s.phase {
        Phase::Keep => {
            let bits = best_keep(s, |kept| {
                let mut shared = 0;
                for i in 0..kept.len() {
                    for j in i + 1..kept.len() {
                        let (x, y) = (&map.tickets[kept[i] as usize], &map.tickets[kept[j] as usize]);
                        if x.a == y.a || x.a == y.b || x.b == y.a || x.b == y.b {
                            shared += 1;
                        }
                    }
                }
                let pts: i32 = kept.iter().map(|&t| map.tickets[t as usize].points as i32).sum();
                (shared, pts, -(kept.len() as i32))
            });
            let aid = map.keep_action(bits) as u16;
            if legal.contains(&aid) {
                aid
            } else {
                legal[0]
            }
        }
        Phase::Turn => {
            let on_path = ticket_paths(map, s, p);
            let comps: Vec<[u8; MAX_CITIES]> = (0..n).map(|q| s.components(map, q)).collect();
            let innet: Vec<[bool; MAX_CITIES]> = (0..n).map(|q| in_network(map, s, q)).collect();
            let mut best_claim: Option<u16> = None;
            let mut best_key = (-1.0f32, -99i32, 0i32);
            for &ai in legal {
                if ai as usize >= map.n_claim {
                    continue;
                }
                let claim = map.claims[ai as usize];
                let r = claim.route as usize;
                let route = &map.routes[r];
                let len = route.length as f32;
                let mut score = len;
                if on_path[r] {
                    score += 20.0 + len;
                } else {
                    let (a, b) = (route.a as usize, route.b as usize);
                    for opp in (0..n).filter(|&q| q != p) {
                        let (lab, inn) = (&comps[opp], &innet[opp]);
                        if inn[a] && inn[b] && lab[a] != lab[b] {
                            score = score.max(12.0 + len);
                        } else if (inn[a] || inn[b]) && route.length >= 5 {
                            score = score.max(8.0 + len);
                        }
                    }
                }
                let closed = points_closed(map, s, p, &comps[p], r);
                if closed > 0 {
                    score += 40.0 + closed as f32;
                }
                let have = s.hand[p][claim.color as usize] as i32;
                let locos = (route.length as i32 - have).max(0);
                let key = (score, -locos, have);
                if key > best_key {
                    best_key = key;
                    best_claim = Some(ai);
                }
            }
            if let Some(c) = best_claim {
                if best_key.0 >= 20.0 {
                    return c;
                }
            }
            let mut needed = [0.0f32; N_COLORS];
            for (i, route) in map.routes.iter().enumerate() {
                if !on_path[i] || s.route_owner[i] != NONE {
                    continue;
                }
                match route.color {
                    None => needed.iter_mut().for_each(|x| *x += 0.25),
                    Some(c) => needed[c as usize] += route.length as f32,
                }
            }
            let mut best_draw: Option<u16> = None;
            let mut best_d = -1.0f32;
            for slot in 0..FACE_UP {
                let aid = map.draw_action(slot) as u16;
                if !legal.contains(&aid) {
                    continue;
                }
                let c = s.face_up[slot];
                let val = if c == LOCO {
                    8.0
                } else if c != NONE {
                    needed[c as usize]
                } else {
                    0.0
                };
                if val > best_d {
                    best_d = val;
                    best_draw = Some(aid);
                }
            }
            let deck = map.blind_action() as u16;
            if legal.contains(&deck) && best_d < 1.0 {
                return deck;
            }
            if let Some(d) = best_draw {
                if best_d > 0.0 {
                    return d;
                }
            }
            if let Some(c) = best_claim {
                return c;
            }
            random_of(legal, rng)
        }
        Phase::Draw2 => {
            let on_path = ticket_paths(map, s, p);
            let mut needed = [0.0f32; N_COLORS];
            for (i, route) in map.routes.iter().enumerate() {
                if let (true, Some(c)) = (on_path[i], route.color) {
                    needed[c as usize] += 1.0;
                }
            }
            let mut best: Option<u16> = None;
            let mut best_v = -1.0f32;
            for slot in 0..FACE_UP {
                let aid = map.draw_action(slot) as u16;
                if !legal.contains(&aid) {
                    continue;
                }
                let c = s.face_up[slot];
                let v = if c == LOCO {
                    5.0
                } else if c != NONE {
                    needed[c as usize]
                } else {
                    0.0
                };
                if v > best_v {
                    best_v = v;
                    best = Some(aid);
                }
            }
            if let Some(b) = best {
                if best_v > 0.0 {
                    return b;
                }
            }
            let deck = map.blind_action() as u16;
            if legal.contains(&deck) {
                return deck;
            }
            random_of(legal, rng)
        }
        Phase::Over => legal[0],
    }
}

/// The hoarder's targets: the routes of the greatest length still open to `p` that its trains can
/// take (the six-routes while any is left; there are nine, one of each colour and a grey one).
fn hoard_targets(map: &Map, s: &State, p: usize) -> (u8, [bool; MAX_ROUTES]) {
    let mut t = [false; MAX_ROUTES];
    let open = |r: usize| {
        s.route_open(map, r) && s.trains[p] >= map.routes[r].length && {
            let sib = &map.pairs[map.routes[r].pair as usize];
            sib.iter().all(|&j| s.route_owner[j as usize] != p as u8)
        }
    };
    let best = (0..map.n_routes())
        .filter(|&r| open(r))
        .map(|r| map.routes[r].length)
        .max()
        .unwrap_or(0);
    for r in (0..map.n_routes()).filter(|&r| open(r) && map.routes[r].length == best) {
        t[r] = true;
    }
    (best, t)
}

fn hoard(map: &Map, s: &State, legal: &[u16], rng: &mut Xoshiro) -> u16 {
    let p = s.current as usize;
    let (target_len, targets) = hoard_targets(map, s, p);
    // the colours the targets still want: a target's colour (grey: the colour held most) while the
    // hand holds fewer than its length
    let want = |c: u8| -> f32 {
        let most = (0..N_COLORS).max_by_key(|&x| s.hand[p][x]).unwrap_or(0) as u8;
        let mut v = 0.0;
        for (r, route) in map.routes.iter().enumerate() {
            if targets[r] && route.color.unwrap_or(most) == c && s.hand[p][c as usize] < route.length {
                v += route.length as f32;
            }
        }
        v
    };
    let mut draw = |first: bool| -> u16 {
        let mut best: Option<u16> = None;
        let mut best_v = 0.0f32;
        for slot in 0..FACE_UP {
            let aid = map.draw_action(slot) as u16;
            let c = s.face_up[slot];
            if !legal.contains(&aid) || c == NONE || c == LOCO {
                continue; // a face-up locomotive is one card for a turn; the deck gives two
            }
            let v = want(c);
            if v > best_v {
                best_v = v;
                best = Some(aid);
            }
        }
        let deck = map.blind_action() as u16;
        match (best, legal.contains(&deck)) {
            (Some(b), _) => b,
            (None, true) => deck,
            (None, false) => {
                // only face-up locomotives (first draw) or tickets are left
                let up: Vec<u16> = legal
                    .iter()
                    .cloned()
                    .filter(|&a| a as usize >= map.n_claim && (a as usize) < map.blind_action())
                    .collect();
                if first && !up.is_empty() {
                    up[0]
                } else {
                    random_of(legal, rng)
                }
            }
        }
    };
    match s.phase {
        Phase::Keep => {
            // the fewest tickets allowed, worth the fewest points: the smallest penalty
            let bits = best_keep(s, |kept| {
                let pts: i32 = kept.iter().map(|&t| map.tickets[t as usize].points as i32).sum();
                (-(kept.len() as i32), -pts)
            });
            let aid = map.keep_action(bits) as u16;
            if legal.contains(&aid) {
                aid
            } else {
                legal[0]
            }
        }
        Phase::Turn => {
            // a target as soon as it is affordable; with no target left that the trains can take
            // (the end), the longest affordable route: the game ends sooner
            let mut best: Option<((u8, i32, i32), u16)> = None;
            for &ai in legal {
                if ai as usize >= map.n_claim {
                    continue;
                }
                let claim = map.claims[ai as usize];
                let r = claim.route as usize;
                let len = map.routes[r].length;
                if target_len >= HOARD_MIN_LENGTH && !targets[r] {
                    continue;
                }
                let have = s.hand[p][claim.color as usize] as i32;
                let key = (len, -(len as i32 - have).max(0), have);
                if best.is_none_or(|(k, _)| key > k) {
                    best = Some((key, ai));
                }
            }
            if let Some((_, ai)) = best {
                return ai;
            }
            draw(true)
        }
        Phase::Draw2 => draw(false),
        Phase::Over => legal[0],
    }
}

/// The hoarder claims only its targets while they are at least this long; below, anything that fits.
const HOARD_MIN_LENGTH: u8 = 4;

/// What a seat did in one game.
#[derive(Clone, Copy, Debug, Default)]
pub struct SeatStats {
    pub score: i16,
    pub rank: f32,
    pub turns: u32,
    pub draw_turns: u32,
    pub ticket_turns: u32,
    pub claims: u32,
    pub six_routes: u32,
    pub tickets_kept: u32,
    pub tickets_completed: u32,
    pub trains_left: u32,
}

/// One game of bots under `rules`, `seats[q]` playing seat `q`: the deal from `deal`, everything
/// after it from `rng`; returns each seat's stats.
pub fn play_bots(map: &Map, seats: &[Bot], rules: Rules, deal: &mut Xoshiro, rng: &mut Xoshiro) -> Vec<SeatStats> {
    let n = seats.len();
    let mut s = State::with_rules(map, n, rules, deal);
    let mut st = vec![SeatStats::default(); n];
    let mut moves = Vec::with_capacity(256);
    let mut steps = 0;
    while !s.is_over() && steps < 5000 {
        steps += 1;
        s.legal_actions(map, &mut moves);
        if moves.is_empty() {
            s.pass();
            continue;
        }
        let p = s.current as usize;
        let a = seats[p].choose(map, &s, &moves, rng) as usize;
        if s.phase == Phase::Turn {
            st[p].turns += 1;
            if a < map.n_claim {
                st[p].claims += 1;
                st[p].six_routes += (map.routes[map.claims[a].route as usize].length == 6) as u32;
            } else if a == map.tickets_action() {
                st[p].ticket_turns += 1;
            } else {
                st[p].draw_turns += 1;
            }
        }
        s.step(map, a, rng);
    }
    let f = s.finalize(map);
    for q in 0..n {
        st[q].score = f.total[q];
        st[q].rank = f.ranks[q];
        st[q].tickets_kept = s.tickets_held[q].count_ones();
        st[q].tickets_completed = f.completed[q] as u32;
        st[q].trains_left = s.trains[q] as u32;
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bots_play_legal_games_to_the_end() {
        let map = Map::usa();
        let mut rng = Xoshiro::new(7);
        for n in 2..=5 {
            for kinds in [
                [Bot::Heuristic; 5],
                [Bot::Hoard; 5],
                [Bot::Heuristic, Bot::Hoard, Bot::Random, Bot::Heuristic, Bot::Hoard],
            ] {
                for g in 0..20 {
                    let mut deal = Xoshiro::new(1000 * n as u64 + g);
                    let st = play_bots(map, &kinds[..n], Rules::OFFICIAL, &mut deal, &mut rng);
                    assert_eq!(st.len(), n);
                    assert!(st.iter().all(|x| x.turns > 0));
                }
            }
        }
    }

    #[test]
    fn bots_play_by_the_rules_given_and_a_game_replays_from_its_seeds() {
        let map = Map::usa();
        let no_routes = Rules {
            route_points: false,
            trains: 20,
            ..Rules::OFFICIAL
        };
        let (mut same, mut longer) = (0, 0);
        for g in 0..20 {
            let seats = [Bot::Heuristic, Bot::Hoard];
            let a = play_bots(
                map,
                &seats,
                Rules::OFFICIAL,
                &mut Xoshiro::new(g),
                &mut Xoshiro::new(100 + g),
            );
            let b = play_bots(
                map,
                &seats,
                Rules::OFFICIAL,
                &mut Xoshiro::new(g),
                &mut Xoshiro::new(100 + g),
            );
            same += (a.iter().zip(&b).all(|(x, y)| x.score == y.score && x.turns == y.turns)) as u32;
            let c = play_bots(map, &seats, no_routes, &mut Xoshiro::new(g), &mut Xoshiro::new(100 + g));
            assert!(c.iter().all(|x| x.trains_left <= 20), "20 trains at most");
            longer += (a[0].turns > c[0].turns) as u32;
        }
        assert_eq!(same, 20, "the same seeds, the same games");
        assert!(longer >= 15, "45 trains last longer than 20: {longer} of 20");
    }

    #[test]
    fn the_heuristic_completes_tickets_and_hoard_does_not_draw_them() {
        let map = Map::usa();
        let mut rng = Xoshiro::new(11);
        let (mut done, mut kept, mut hoard_ticket_turns) = (0, 0, 0);
        for g in 0..50 {
            let mut deal = Xoshiro::new(g);
            let st = play_bots(map, &[Bot::Heuristic, Bot::Hoard], Rules::OFFICIAL, &mut deal, &mut rng);
            done += st[0].tickets_completed;
            kept += st[0].tickets_kept;
            hoard_ticket_turns += st[1].ticket_turns;
        }
        assert!(
            done * 2 > kept,
            "the heuristic completes most of its tickets: {done} of {kept}"
        );
        assert_eq!(hoard_ticket_turns, 0);
    }
}
