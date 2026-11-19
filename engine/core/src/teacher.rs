//! The scripted teacher of the first training phases: a target for every position, in the interest of
//! building the tickets the player to act holds, whoever wins the game.
//!
//! - A claim on a shortest way still open for one of its tickets not yet completed, on the board as it
//!   is: its own routes free, the open ones at their length, every other seat's closed, so a way an
//!   opponent cut gives way to the best detour left. The claim is worth the ticket's points if it
//!   completes the ticket, else the points times the route's share of the ticket's length, summed over
//!   every ticket the route serves.
//! - No such claim it can pay for: a face-up card that pays for an open route of those ways (its
//!   colour; for a grey route the colour the hand holds most of; a locomotive always) is worth
//!   `DRAW_USEFUL`; with no such card in sight, the deck is worth `DRAW_BLIND`.
//! - A choice of tickets: every set of as few tickets as allowed is right alike, so the network learns
//!   to keep any such set.
//! - Anything else, drawing tickets or a route off those ways: nothing.
//!
//! A position where no move is worth anything (every ticket completed, or no way left to build) has
//! no target.
//!
//! The joint plan (`joint`): with several tickets open, each ticket's ways are found on the board with
//! the routes already planned for the tickets before it free, in the order of the tickets that needs
//! the fewest trains, so a route two tickets can share serves, and pays for, both. On this map a
//! ticket's points are about the trains of its shortest path, so only shared routes score more than
//! the trains spent.

use crate::map::{shortest, Map, LOCO, MAX_CITIES, N_CARD_TYPES};
use crate::state::{Phase, State, NONE};

/// The share of the target spread over every legal move; the best moves share the rest.
pub const SPREAD: f32 = 0.1;
/// A face-up card that pays for a route of the ways.
pub const DRAW_USEFUL: f32 = 0.5;
/// The deck, when no such card is in sight.
pub const DRAW_BLIND: f32 = 0.25;
/// A set of as few tickets as allowed.
const KEEP_FEW: f32 = 1.0;

/// The open routes on a shortest way still open between each ticket's cities for player `p`: its own
/// routes free, the routes open to it at their length, every other seat's closed. None for a ticket
/// whose cities can no longer be joined.
fn open_ways(map: &Map, s: &State, p: usize, tickets: &[usize]) -> Vec<Option<Vec<usize>>> {
    const FAR: u16 = u16::MAX;
    let nc = map.n_cities();
    let mut w = [[FAR; MAX_CITIES]; MAX_CITIES];
    let cost = |r: usize| -> Option<u16> {
        if s.route_owner[r] == p as u8 {
            Some(0)
        } else if s.route_open(map, r) && s.double_allows(map, p, r) {
            Some(map.routes[r].length as u16)
        } else {
            None
        }
    };
    for (r, route) in map.routes.iter().enumerate() {
        if let Some(c) = cost(r) {
            let (a, b) = (route.a as usize, route.b as usize);
            w[a][b] = w[a][b].min(c);
            w[b][a] = w[a][b];
        }
    }
    tickets
        .iter()
        .map(|&t| {
            let tk = &map.tickets[t];
            let (da, db) = (shortest(nc, &w, tk.a as usize), shortest(nc, &w, tk.b as usize));
            let whole = da[tk.b as usize];
            if whole == FAR {
                return None;
            }
            let on = |r: usize| {
                let (u, v) = (map.routes[r].a as usize, map.routes[r].b as usize);
                let len = map.routes[r].length as u32;
                let (whole, du, dv, eu, ev) = (whole as u32, da[u] as u32, da[v] as u32, db[u] as u32, db[v] as u32);
                du + len + ev == whole || dv + len + eu == whole
            };
            Some(
                (0..map.routes.len())
                    .filter(|&r| s.route_owner[r] == NONE && cost(r).is_some() && on(r))
                    .collect(),
            )
        })
        .collect()
}

/// Dijkstra from city `src` over the routes `cost` allows: each city's distance, and the route it is
/// reached by (the first one in route order among equally short).
fn tree(map: &Map, src: usize, cost: &dyn Fn(usize) -> Option<u16>) -> ([u32; MAX_CITIES], [usize; MAX_CITIES]) {
    let nc = map.n_cities();
    let mut dist = [u32::MAX; MAX_CITIES];
    let mut via = [usize::MAX; MAX_CITIES];
    let mut done = [false; MAX_CITIES];
    dist[src] = 0;
    while let Some(u) = (0..nc)
        .filter(|&c| !done[c] && dist[c] < u32::MAX)
        .min_by_key(|&c| dist[c])
    {
        done[u] = true;
        for (r, route) in map.routes.iter().enumerate() {
            let (a, b) = (route.a as usize, route.b as usize);
            let v = if a == u {
                b
            } else if b == u {
                a
            } else {
                continue;
            };
            if let Some(c) = cost(r) {
                let d = dist[u] + c as u32;
                if d < dist[v] {
                    dist[v] = d;
                    via[v] = r;
                }
            }
        }
    }
    (dist, via)
}

/// The joint plan of player `p`'s open tickets `open`, the tickets joined in `order`: per ticket, the
/// open routes on any shortest way of it with its own routes and those planned before it free (None:
/// its cities can no longer be joined), and the plan's trains. One of each ticket's ways goes into the
/// plan for the tickets after it: from its first city, the routes the search from its second city
/// reached the cities by.
fn plan(map: &Map, s: &State, p: usize, open: &[usize], order: &[usize]) -> (u32, Vec<Option<Vec<usize>>>) {
    let mut planned = vec![false; map.routes.len()];
    let mut ways: Vec<Option<Vec<usize>>> = vec![None; open.len()];
    let mut total = 0u32;
    for &k in order {
        let tk = &map.tickets[open[k]];
        let (a, b) = (tk.a as usize, tk.b as usize);
        let cost = |r: usize| -> Option<u16> {
            if s.route_owner[r] == p as u8 || planned[r] {
                Some(0)
            } else if s.route_open(map, r) && s.double_allows(map, p, r) {
                Some(map.routes[r].length as u16)
            } else {
                None
            }
        };
        let ((da, _), (db, via)) = (tree(map, a, &cost), tree(map, b, &cost));
        let whole = da[b];
        if whole == u32::MAX {
            continue;
        }
        total += whole;
        let on = |r: usize| {
            let (u, v) = (map.routes[r].a as usize, map.routes[r].b as usize);
            cost(r).is_some_and(|c| {
                let c = c as u32;
                (da[u] != u32::MAX && db[v] != u32::MAX && da[u] + c + db[v] == whole)
                    || (da[v] != u32::MAX && db[u] != u32::MAX && da[v] + c + db[u] == whole)
            })
        };
        ways[k] = Some(
            (0..map.routes.len())
                .filter(|&r| s.route_owner[r] == NONE && on(r))
                .collect(),
        );
        let mut at = a;
        while at != b && via[at] != usize::MAX {
            let r = via[at];
            planned[r] |= s.route_owner[r] == NONE;
            at = if map.routes[r].a as usize == at {
                map.routes[r].b as usize
            } else {
                map.routes[r].a as usize
            };
        }
    }
    (total, ways)
}

/// The orders `joint_ways` tries for the tickets `open`: every order of up to 4; for more, the tickets
/// by points (highest first), by length, and each ticket first with the rest by points.
fn orders(map: &Map, open: &[usize]) -> Vec<Vec<usize>> {
    let n = open.len();
    if n <= 4 {
        let mut out: Vec<Vec<usize>> = vec![Vec::new()];
        for _ in 0..n {
            let mut next = Vec::new();
            for o in &out {
                for i in (0..n).filter(|i| !o.contains(i)) {
                    next.push([o.as_slice(), &[i]].concat());
                }
            }
            out = next;
        }
        return out;
    }
    let by = |key: &dyn Fn(usize) -> i32| {
        let mut o: Vec<usize> = (0..n).collect();
        o.sort_by_key(|&i| (-key(open[i]), i));
        o
    };
    let by_points = by(&|t| map.tickets[t].points as i32);
    let mut out = vec![by_points.clone(), by(&|t| map.ticket_length[t] as i32)];
    for first in 0..n {
        out.push([vec![first], by_points.iter().cloned().filter(|&i| i != first).collect()].concat());
    }
    out
}

/// Each open ticket's ways in the joint plan that needs the fewest trains; with one ticket,
/// `open_ways`.
fn joint_ways(map: &Map, s: &State, p: usize, open: &[usize]) -> Vec<Option<Vec<usize>>> {
    if open.len() <= 1 {
        return open_ways(map, s, p, open);
    }
    let mut best: Option<(u32, Vec<Option<Vec<usize>>>)> = None;
    for order in orders(map, open) {
        let (cost, ways) = plan(map, s, p, open, &order);
        if best.as_ref().is_none_or(|b| cost < b.0) {
            best = Some((cost, ways));
        }
    }
    best.expect("an order").1
}

/// What each of the legal moves `legal` of the player to act is worth, with the joint plan's ways
/// (`joint`) or each ticket's own.
fn rewards(map: &Map, s: &State, legal: &[u16], joint: bool) -> Vec<f32> {
    let p = s.current as usize;
    let mut out = vec![0.0f32; legal.len()];
    if s.phase == Phase::Keep {
        for (k, &a) in legal.iter().enumerate() {
            if map.keep_mask_of(a as usize).count_ones() as u8 == s.keep_min {
                out[k] = KEEP_FEW;
            }
        }
        return out;
    }
    let comp = s.components(map, p);
    let mut open: Vec<usize> = Vec::new();
    let mut held = s.tickets_held[p];
    while held != 0 {
        let t = held.trailing_zeros() as usize;
        held &= held - 1;
        let tk = &map.tickets[t];
        if comp[tk.a as usize] != comp[tk.b as usize] {
            open.push(t);
        }
    }
    if open.is_empty() {
        return out;
    }
    let ways = if joint {
        joint_ways(map, s, p, &open)
    } else {
        open_ways(map, s, p, &open)
    };
    // claims on the tickets' shortest ways still open
    let mut best_claim = 0.0f32;
    for (k, &a) in legal.iter().enumerate() {
        if a as usize >= map.n_claim {
            continue;
        }
        let r = map.claims[a as usize].route as usize;
        let route = &map.routes[r];
        let (ca, cb) = (comp[route.a as usize], comp[route.b as usize]);
        let mut v = 0.0f32;
        for (&t, way) in open.iter().zip(&ways) {
            if !way.as_ref().is_some_and(|w| w.contains(&r)) {
                continue;
            }
            let tk = &map.tickets[t];
            let (ta, tb) = (comp[tk.a as usize], comp[tk.b as usize]);
            let completes = (ca == ta && cb == tb) || (ca == tb && cb == ta);
            v += if completes {
                tk.points as f32
            } else {
                tk.points as f32 * route.length as f32 / map.ticket_length[t] as f32
            };
        }
        out[k] = v;
        best_claim = best_claim.max(v);
    }
    if best_claim > 0.0 {
        return out; // it can build on a way now: that comes first
    }
    // the cards that pay for an open route of those ways
    let hand = &s.hand[p];
    let most = (0..N_CARD_TYPES)
        .filter(|&c| c != LOCO as usize)
        .max_by_key(|&c| (hand[c], usize::MAX - c))
        .unwrap_or(0);
    let mut useful = [false; N_CARD_TYPES];
    let mut any = false;
    for way in ways.iter().flatten() {
        for &r in way {
            any = true;
            match map.routes[r].color {
                Some(c) => useful[c as usize] = true,
                None if hand[most] > 0 => useful[most] = true,
                None => useful.iter_mut().for_each(|u| *u = true),
            }
        }
    }
    if !any {
        return out; // its ways are cut: nothing to build on them
    }
    useful[LOCO as usize] = true;
    let mut seen = false;
    for (k, &a) in legal.iter().enumerate() {
        let a = a as usize;
        if a >= map.n_claim && a < map.blind_action() {
            let c = s.face_up[a - map.n_claim];
            if c != NONE && useful[c as usize] {
                out[k] = DRAW_USEFUL;
                seen = true;
            }
        }
    }
    if !seen {
        for (k, &a) in legal.iter().enumerate() {
            if a as usize == map.blind_action() {
                out[k] = DRAW_BLIND;
            }
        }
    }
    out
}

/// The teacher's target at `s` over every action id: `1 - SPREAD` shared by the best moves of
/// `legal`, `SPREAD` spread over all of them, so the target is never certain. None when no move is
/// worth anything. `joint`: the ways of the joint plan.
pub fn teacher_policy(map: &Map, s: &State, legal: &[u16], joint: bool) -> Option<Vec<f32>> {
    let r = rewards(map, s, legal, joint);
    let best = r.iter().cloned().fold(0.0f32, f32::max);
    if best <= 0.0 || legal.is_empty() {
        return None;
    }
    let n_best = r.iter().filter(|&&v| v >= best - 1e-6).count() as f32;
    let mut pol = vec![0.0f32; map.n_actions];
    for (k, &a) in legal.iter().enumerate() {
        pol[a as usize] = SPREAD / legal.len() as f32
            + if r[k] >= best - 1e-6 {
                (1.0 - SPREAD) / n_best
            } else {
                0.0
            };
    }
    Some(pol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro;
    use crate::state::{Rules, MAX_PLAYERS};

    /// The routes of one shortest path of ticket `t`, from its first city.
    fn one_path(map: &Map, t: usize) -> Vec<usize> {
        let tk = &map.tickets[t];
        let (mut at, mut path) = (tk.a, Vec::new());
        while at != tk.b {
            let e = map.ticket_paths[t]
                .iter()
                .find(|e| e.0 == at)
                .expect("the path goes on");
            path.push(e.2 as usize);
            at = e.1;
        }
        path
    }

    /// The shortest ticket of the map.
    fn short_ticket(map: &Map) -> usize {
        (0..map.n_tickets()).min_by_key(|&t| map.ticket_length[t]).unwrap()
    }

    fn tickets_only() -> Rules {
        Rules {
            route_points: false,
            longest_bonus: false,
            partial_tickets: true,
            ..Rules::OFFICIAL
        }
    }

    /// A game of two past the opening, seat 0 to act holding only ticket `t`, its hand `hand`.
    fn holding(t: usize, hand: [u8; N_CARD_TYPES]) -> State {
        let map = Map::usa();
        let mut rng = Xoshiro::new(5);
        let mut s = State::with_rules(map, 2, tickets_only(), &mut rng);
        let mut moves = Vec::new();
        while s.phase == Phase::Keep {
            s.legal_actions(map, &mut moves);
            s.step(map, moves[0] as usize, &mut rng);
        }
        s.current = 0;
        s.tickets_held = [0; MAX_PLAYERS];
        s.tickets_held[0] = 1u64 << t;
        s.hand[0] = hand;
        s
    }

    fn legal(s: &State) -> Vec<u16> {
        let mut v = Vec::new();
        s.legal_actions(Map::usa(), &mut v);
        v
    }

    #[test]
    fn a_claim_that_completes_a_ticket_comes_first_and_pays_the_ticket() {
        let map = Map::usa();
        // a shortest path built but for its last route: claiming that one completes the ticket
        let t = short_ticket(map);
        let path = one_path(map, t);
        let r = *path.last().unwrap();
        let mut hand = [0u8; N_CARD_TYPES];
        hand[LOCO as usize] = 6;
        let mut s = holding(t, hand);
        for &q in &path[..path.len() - 1] {
            s.route_owner[q] = 0;
        }
        let legal = legal(&s);
        let rew = rewards(map, &s, &legal, false);
        let claim = legal
            .iter()
            .position(|&a| (a as usize) < map.n_claim && map.claims[a as usize].route as usize == r)
            .unwrap();
        assert_eq!(rew[claim], map.tickets[t].points as f32);
        assert_eq!(rew.iter().cloned().fold(0.0, f32::max), rew[claim]);
        assert!(
            legal
                .iter()
                .zip(&rew)
                .all(|(&a, &v)| (a as usize) < map.n_claim || v == 0.0),
            "no card while it can build"
        );
        let pol = teacher_policy(map, &s, &legal, false).unwrap();
        assert!((pol.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(
            pol[legal[claim] as usize] > (1.0 - SPREAD) / 8.0,
            "one of the claims of that route"
        );
    }

    #[test]
    fn a_way_an_opponent_cut_gives_way_to_the_best_detour_left() {
        let map = Map::usa();
        let t = short_ticket(map);
        let mut hand = [0u8; N_CARD_TYPES];
        hand[LOCO as usize] = 6;
        let mut s = holding(t, hand);
        for e in &map.ticket_paths[t] {
            s.route_owner[e.2 as usize] = 1;
        }
        let legal = legal(&s);
        let rew = rewards(map, &s, &legal, false);
        let on_shortest: Vec<u16> = map.ticket_paths[t].iter().map(|e| e.2).collect();
        let (&a, _) = legal
            .iter()
            .zip(&rew)
            .find(|(&a, &v)| (a as usize) < map.n_claim && v > 0.0)
            .expect("a claim on a detour pays");
        assert!(!on_shortest.contains(&map.claims[a as usize].route));
    }

    #[test]
    fn without_the_cards_it_takes_a_useful_card_and_else_the_deck() {
        let map = Map::usa();
        let t = (0..map.n_tickets()).max_by_key(|&t| map.ticket_length[t]).unwrap();
        let mut s = holding(t, [0; N_CARD_TYPES]);
        assert!(
            legal(&s).iter().all(|&a| a as usize >= map.n_claim),
            "no claim it can pay for"
        );
        s.face_up = [LOCO, NONE, NONE, NONE, NONE];
        let moves = legal(&s);
        let rew = rewards(map, &s, &moves, false);
        let at = |a: usize| moves.iter().position(|&x| x as usize == a).map(|k| rew[k]);
        assert_eq!(at(map.draw_action(0)), Some(DRAW_USEFUL));
        assert_eq!(
            at(map.blind_action()),
            Some(0.0),
            "a useful card in sight: not the deck"
        );
        assert_eq!(at(map.tickets_action()), Some(0.0), "drawing tickets is worth nothing");
        s.face_up = [NONE; 5];
        let moves = legal(&s);
        let rew = rewards(map, &s, &moves, false);
        let k = moves.iter().position(|&x| x as usize == map.blind_action()).unwrap();
        assert_eq!(rew[k], DRAW_BLIND);
    }

    #[test]
    fn with_one_ticket_open_the_joint_plan_is_the_tickets_own_ways() {
        let map = Map::usa();
        let mut hand = [0u8; N_CARD_TYPES];
        hand[LOCO as usize] = 6;
        for t in 0..map.n_tickets() {
            let s = holding(t, hand);
            let legal = legal(&s);
            assert_eq!(
                rewards(map, &s, &legal, true),
                rewards(map, &s, &legal, false),
                "ticket {t}"
            );
        }
    }

    #[test]
    fn a_route_two_tickets_share_only_in_the_joint_plan_pays_for_both() {
        let map = Map::usa();
        let mut hand = [0u8; N_CARD_TYPES];
        hand[LOCO as usize] = 6;
        let mut found = None;
        'pairs: for a in 0..map.n_tickets() {
            for b in a + 1..map.n_tickets() {
                let mut s = holding(a, hand);
                s.tickets_held[0] |= 1u64 << b;
                let open = [a, b];
                let best = orders(map, &open)
                    .iter()
                    .map(|o| plan(map, &s, 0, &open, o).0)
                    .min()
                    .unwrap();
                assert!(
                    best <= plan(map, &s, 0, &[a], &[0]).0 + plan(map, &s, 0, &[b], &[0]).0,
                    "tickets {a}, {b}: the plan needs no more trains than the two apart"
                );
                let (joint, own) = (joint_ways(map, &s, 0, &open), open_ways(map, &s, 0, &open));
                let (Some(ja), Some(jb), Some(oa), Some(ob)) = (&joint[0], &joint[1], &own[0], &own[1]) else {
                    continue;
                };
                let Some(&r) = ja
                    .iter()
                    .find(|r| jb.contains(r) && !(oa.contains(r) && ob.contains(r)))
                else {
                    continue;
                };
                let legal = legal(&s);
                let Some(k) = legal
                    .iter()
                    .position(|&x| (x as usize) < map.n_claim && map.claims[x as usize].route as usize == r)
                else {
                    continue;
                };
                let (j, o) = (rewards(map, &s, &legal, true), rewards(map, &s, &legal, false));
                assert!(
                    j[k] > o[k] + 1e-4,
                    "tickets {a}, {b}, route {r}: {} against {}",
                    j[k],
                    o[k]
                );
                found = Some((a, b, r));
                break 'pairs;
            }
        }
        assert!(
            found.is_some(),
            "two tickets of the map share a route only in the joint plan"
        );
    }

    #[test]
    fn the_opening_takes_any_two_tickets_and_a_hand_built_whole_teaches_nothing() {
        let map = Map::usa();
        let s = State::with_rules(map, 2, tickets_only(), &mut Xoshiro::new(9));
        let keeps = legal(&s);
        let size = |a: u16| map.keep_mask_of(a as usize).count_ones() as u8;
        let pol = teacher_policy(map, &s, &keeps, false).expect("a target");
        let pairs: Vec<u16> = keeps.iter().cloned().filter(|&a| size(a) == s.keep_min).collect();
        assert_eq!(pairs.len(), 3, "three pairs of three tickets");
        let top = pairs.iter().map(|&a| pol[a as usize]).fold(0.0f32, f32::max);
        assert!(
            pairs.iter().all(|&a| (pol[a as usize] - top).abs() < 1e-6),
            "every pair alike"
        );
        assert!(keeps
            .iter()
            .filter(|&&a| size(a) != s.keep_min)
            .all(|&a| pol[a as usize] < top));
        let t = short_ticket(map);
        let mut done = holding(t, [0; N_CARD_TYPES]);
        for r in one_path(map, t) {
            done.route_owner[r] = 0;
        }
        assert!(teacher_policy(map, &done, &legal(&done), false).is_none());
    }
}
