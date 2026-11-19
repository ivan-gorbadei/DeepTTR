//! What one player sees, as numbers for the network.
//!
//! The rules the game is played with (`state::Rules`: a variant changes what scores), and exactly
//! what a human at the table sees, nothing more: the board and who holds which route, with every
//! claim's turn and what paid it; the face-up row; the deck's size and the discard pile (paid cards
//! are shown when they are played); every player's trains, route points, number of cards and
//! tickets, the cards they are known to hold (taken face up, not yet seen spent), the cards of each
//! colour they took face up and paid over the whole game; the own hand and tickets; the tickets on
//! offer to oneself; and the last `EVENTS` public events one by one. Nothing public is dropped: an
//! event older than the last `EVENTS` survives in those per-route and per-player records (only the
//! order of the draws between claims is summarised, as counts). In the opening nobody sees what the
//! others kept until all have chosen (the engine deals and chooses in seat order; the table sees it
//! as simultaneous).
//!
//! The test `sample::tests::sampled_worlds_look_the_same` is the guarantee that nothing hidden
//! leaks: a world re-sampled for a player (`sample::sample_world`) gives that player the very same
//! observation.
//!
//! Seats are relative: 0 is the observer, 1 the next player to act after them, and so on.

use crate::map::{Map, LOCO, N_CARD_TYPES, N_COLORS, ROUTE_SCORE};
use crate::state::{event, Phase, State, EVENTS, MAX_PLAYERS, NONE, TICKETS_START_DEAL};

/// Per route: owner (none, seat 0..4) 6 | closed double 1 | barred to me (I hold its twin) 1 |
/// affordable by me 1 | my cards short 1 | length one-hot 6 | colour one-hot (8 + grey) 9 | points 1 |
/// the turn it was claimed in 1 | the colour that paid it one-hot 8 | locomotives that paid it 1
pub const ROUTE_F: usize = 36;
/// Per city: routes there per seat 5 | ends of my tickets 1 | of tickets offered to me 1 |
/// of my completed tickets 1
pub const CITY_F: usize = 8;
/// Per ticket: mine 1 | offered to me 1 | not mine 1 | completed by me 1 | points 1
pub const TICKET_F: usize = 5;
/// Per seat: present 1 | to act 1 | trains 1 | route points 1 | cards 1 | tickets 1 |
/// known cards 9 | blind draws 1 | face-up takes 1 | ticket draws 1 | tickets kept 1 |
/// opening keep 1 | longest trail 1 | holds the longest 1 | face-up takes by card type 9 |
/// cards paid by card type 9
pub const PLAYER_F: usize = 40;
/// The table: the game's phase one-hot 4 | I act 1 | opening 1 | last round 1 | its turns left 1 |
/// deck 1 | discard 9 | my hand 9 | face-up counts 9 | on offer 1 | keep at least 1 | turn 1 |
/// players 2..5 one-hot 4 | ticket deck 1 | deck reshuffles 1 | my opening choice still to make 1 |
/// the rules (`RULES_AT`): routes score, at their price 1 | longest-trail bonus 1 | reserved 1 |
/// starting trains 1 | partial tickets 1 | reserved 7. A reserved input is always 0: a new rule
/// variant can take one without changing the input size of trained networks
pub const GLOBAL_F: usize = 58;
/// Where the rules start in the global features.
pub const RULES_AT: usize = 46;
/// Per event (ints, for embeddings): kind (0 = none) | seat (5 = none) | card or colour (9 = none) |
/// route (n_routes = none) | a count (tickets offered or kept, locomotives paid)
pub const EVENT_I: usize = 5;

/// Where one observation goes. The slices are rows of the network's input arrays.
pub struct ObsOut<'a> {
    pub routes: &'a mut [f32],
    pub cities: &'a mut [f32],
    pub tickets: &'a mut [f32],
    pub players: &'a mut [f32],
    pub global: &'a mut [f32],
    /// the five face-up slots: card type, 9 = empty
    pub slots: &'a mut [i32],
    /// the tickets on offer to the observer, in the order the keep masks' bits use (n_tickets = none)
    pub offer: &'a mut [i32],
    pub events: &'a mut [i32],
    /// 1 for every legal action when the observer is the one to act
    pub legal: &'a mut [u8],
}

/// Sizes of one observation for `map`.
pub struct ObsDims {
    pub routes: usize,
    pub cities: usize,
    pub tickets: usize,
    pub players: usize,
    pub global: usize,
    pub slots: usize,
    pub offer: usize,
    pub events: usize,
    pub legal: usize,
}

pub fn dims(map: &Map) -> ObsDims {
    ObsDims {
        routes: map.n_routes() * ROUTE_F,
        cities: map.n_cities() * CITY_F,
        tickets: map.n_tickets() * TICKET_F,
        players: MAX_PLAYERS * PLAYER_F,
        global: GLOBAL_F,
        slots: 5,
        offer: 3,
        events: EVENTS * EVENT_I,
        legal: map.n_actions,
    }
}

#[inline]
fn rel(s: &State, me: usize, q: usize) -> usize {
    (q + s.n() - me) % s.n()
}

/// Has `q` made their opening choice already (seat order from the first player)?
pub fn opening_chosen(s: &State, q: usize) -> bool {
    if !s.in_opening() {
        return true;
    }
    let n = s.n();
    let order = (q + n - s.opening_first as usize) % n;
    order < n - s.opening_remaining as usize
}

/// The tickets on offer to `me` now (an opening deal not yet chosen, or a draw being chosen).
pub fn offered_to(s: &State, me: usize) -> ([u8; 3], usize) {
    if s.in_opening() && !opening_chosen(s, me) {
        return (s.opening_dealt[me], TICKETS_START_DEAL);
    }
    if s.phase == Phase::Keep && s.current as usize == me {
        return (s.pending, s.pending_len as usize);
    }
    ([NONE; 3], 0)
}

/// How many tickets `q` holds, as `me` sees it (in the opening the others' counts are hidden).
pub fn visible_ticket_count(s: &State, me: usize, q: usize) -> u32 {
    if q != me && s.in_opening() {
        0
    } else {
        s.tickets_held[q].count_ones()
    }
}

/// The ticket deck's size as `me` sees it: in the opening, as dealt (the returns are not yet told).
pub fn visible_ticket_deck(map: &Map, s: &State) -> usize {
    if s.in_opening() {
        map.n_tickets() - s.n() * TICKETS_START_DEAL
    } else {
        s.ticket_deck.len()
    }
}

pub fn encode(map: &Map, s: &State, me: usize, out: &mut ObsOut) {
    let n = s.n();
    let nr = map.n_routes();
    out.routes.fill(0.0);
    out.cities.fill(0.0);
    out.tickets.fill(0.0);
    out.players.fill(0.0);
    out.global.fill(0.0);
    out.legal.fill(0);

    let my_hand = &s.hand[me];
    let my_locos = my_hand[LOCO as usize] as i32;
    let comps: Vec<[u8; crate::map::MAX_CITIES]> = (0..n).map(|q| s.components(map, q)).collect();

    // routes
    for (r, route) in map.routes.iter().enumerate() {
        let f = &mut out.routes[r * ROUTE_F..(r + 1) * ROUTE_F];
        let owner = s.route_owner[r];
        if owner == NONE {
            f[0] = 1.0;
            if !s.route_open(map, r) {
                f[6] = 1.0;
            } else {
                let twin_mine = map.pairs[route.pair as usize]
                    .iter()
                    .any(|&j| j as usize != r && s.route_owner[j as usize] == me as u8);
                if twin_mine {
                    f[7] = 1.0;
                } else {
                    let best = match route.color {
                        Some(c) => my_hand[c as usize] as i32 + my_locos,
                        None => (0..N_COLORS).map(|c| my_hand[c] as i32).max().unwrap_or(0) + my_locos,
                    };
                    let short = (route.length as i32 - best).max(0);
                    if short == 0 && s.trains[me] >= route.length {
                        f[8] = 1.0;
                    }
                    f[9] = short as f32 / 6.0;
                }
            }
        } else {
            f[1 + rel(s, me, owner as usize)] = 1.0;
        }
        f[10 + (route.length as usize - 1)] = 1.0;
        f[16 + route.color.map_or(8, |c| c as usize)] = 1.0;
        f[25] = ROUTE_SCORE[route.length as usize] as f32 / 15.0;
        if owner != NONE {
            let paid = s.route_paid[r];
            f[26] = s.route_turn[r] as f32 / 100.0;
            f[27 + (paid & 0x0f) as usize] = 1.0;
            f[35] = (paid >> 4) as f32 / 6.0;
        }
    }

    // cities
    for (r, route) in map.routes.iter().enumerate() {
        let owner = s.route_owner[r];
        if owner != NONE {
            let k = rel(s, me, owner as usize);
            out.cities[route.a as usize * CITY_F + k] += 0.25;
            out.cities[route.b as usize * CITY_F + k] += 0.25;
        }
    }
    let (offer, n_offer) = offered_to(s, me);
    let mine = s.tickets_held[me];
    for (t, tk) in map.tickets.iter().enumerate() {
        let f = &mut out.tickets[t * TICKET_F..(t + 1) * TICKET_F];
        let held = mine & (1u64 << t) != 0;
        let offered = offer[..n_offer].contains(&(t as u8));
        if held {
            f[0] = 1.0;
            let done = comps[me][tk.a as usize] == comps[me][tk.b as usize];
            if done {
                f[3] = 1.0;
                out.cities[tk.a as usize * CITY_F + 7] += 1.0;
                out.cities[tk.b as usize * CITY_F + 7] += 1.0;
            }
            out.cities[tk.a as usize * CITY_F + 5] += 1.0;
            out.cities[tk.b as usize * CITY_F + 5] += 1.0;
        } else if offered {
            f[1] = 1.0;
            out.cities[tk.a as usize * CITY_F + 6] += 1.0;
            out.cities[tk.b as usize * CITY_F + 6] += 1.0;
        } else {
            f[2] = 1.0;
        }
        f[4] = tk.points as f32 / 22.0;
    }

    // seats
    let longest: Vec<i16> = (0..n).map(|q| s.longest_trail(map, q)).collect();
    let best_trail = longest.iter().cloned().max().unwrap_or(0);
    for q in 0..n {
        let k = rel(s, me, q);
        let f = &mut out.players[k * PLAYER_F..(k + 1) * PLAYER_F];
        f[0] = 1.0;
        f[1] = (s.current as usize == q && s.phase != Phase::Over) as u8 as f32;
        f[2] = s.trains[q] as f32 / 45.0;
        f[3] = s.route_points[q] as f32 / 100.0;
        f[4] = s.hand_size(q) as f32 / 20.0;
        f[5] = visible_ticket_count(s, me, q) as f32 / 8.0;
        for c in 0..N_CARD_TYPES {
            f[6 + c] = s.known[q][c] as f32 / 5.0;
        }
        f[15] = s.blind_draws[q] as f32 / 20.0;
        f[16] = s.face_up_taken[q].iter().map(|&x| x as f32).sum::<f32>() / 20.0;
        f[17] = s.ticket_draws[q] as f32 / 4.0;
        f[18] = s.tickets_kept[q] as f32 / 6.0;
        if !s.in_opening() || q == me {
            f[19] = s.opening_kept[q] as f32 / 3.0;
        }
        f[20] = longest[q] as f32 / 30.0;
        f[21] = (best_trail > 0 && longest[q] == best_trail) as u8 as f32;
        for c in 0..N_CARD_TYPES {
            f[22 + c] = s.face_up_taken[q][c] as f32 / 10.0;
            f[31 + c] = s.paid_cards[q][c] as f32 / 20.0;
        }
    }

    // the table
    let g = &mut *out.global;
    g[s.phase as usize] = 1.0;
    g[4] = (s.current as usize == me && s.phase != Phase::Over) as u8 as f32;
    g[5] = s.in_opening() as u8 as f32;
    g[6] = (s.last_round_left >= 0) as u8 as f32;
    g[7] = if s.last_round_left >= 0 {
        s.last_round_left as f32 / n as f32
    } else {
        0.0
    };
    g[8] = s.deck.len() as f32 / 110.0;
    let discard = s.discard.counts();
    for c in 0..N_CARD_TYPES {
        g[9 + c] = discard[c] as f32 / 14.0;
        g[18 + c] = my_hand[c] as f32 / 12.0;
    }
    for &c in s.face_up.iter().filter(|&&c| c != NONE) {
        g[27 + c as usize] += 0.2;
    }
    let on_offer = if s.in_opening() {
        if opening_chosen(s, me) {
            0
        } else {
            TICKETS_START_DEAL
        }
    } else if s.phase == Phase::Keep {
        s.pending_len as usize
    } else {
        0
    };
    g[36] = on_offer as f32 / 3.0;
    g[37] = if s.phase == Phase::Keep || s.in_opening() {
        s.keep_min as f32 / 2.0
    } else {
        0.0
    };
    g[38] = s.turns as f32 / 100.0;
    g[39 + (n - 2)] = 1.0;
    g[43] = visible_ticket_deck(map, s) as f32 / 30.0;
    g[44] = s.deck_shuffles as f32 / 5.0;
    g[45] = (s.in_opening() && !opening_chosen(s, me)) as u8 as f32;
    // the route price, 0..1: 1 for the official rules, 0 as with no route points
    g[RULES_AT] = if s.rules.route_points {
        s.rules.route_price as f32 / 100.0
    } else {
        0.0
    };
    g[RULES_AT + 1] = s.rules.longest_bonus as u8 as f32;
    // RULES_AT + 2 is reserved
    g[RULES_AT + 3] = s.rules.trains as f32 / 45.0;
    g[RULES_AT + 4] = s.rules.partial_tickets as u8 as f32;

    for (i, &c) in s.face_up.iter().enumerate() {
        out.slots[i] = if c == NONE { 9 } else { c as i32 };
    }
    for i in 0..3 {
        out.offer[i] = if i < n_offer {
            offer[i] as i32
        } else {
            map.n_tickets() as i32
        };
    }

    // events, oldest first, padded at the front
    out.events.fill(0);
    let recent: Vec<_> = s.recent_events().collect();
    let pad = EVENTS - recent.len();
    for i in 0..pad {
        let e = &mut out.events[i * EVENT_I..(i + 1) * EVENT_I];
        e.copy_from_slice(&[0, 5, 9, nr as i32, 0]);
    }
    for (i, ev) in recent.iter().enumerate() {
        let e = &mut out.events[(pad + i) * EVENT_I..(pad + i + 1) * EVENT_I];
        let seat = if (ev.actor as usize) < n {
            rel(s, me, ev.actor as usize) as i32
        } else {
            5
        };
        let (card, route, count) = match ev.kind {
            event::TAKE_FACE_UP => (ev.a as i32, nr as i32, 0),
            event::CLAIM => ((ev.b & 0x0f) as i32, ev.a as i32, (ev.b >> 4) as i32),
            event::DRAW_TICKETS => (9, nr as i32, ev.a as i32),
            event::KEEP_TICKETS | event::OPENING_KEEP => (9, nr as i32, ev.a as i32),
            _ => (9, nr as i32, 0),
        };
        e.copy_from_slice(&[ev.kind as i32, seat, card, route, count]);
    }

    if s.current as usize == me && s.phase != Phase::Over {
        let mut moves = Vec::with_capacity(256);
        s.legal_actions(map, &mut moves);
        for m in moves {
            out.legal[m as usize] = 1;
        }
    }
}

/// One observation in owned buffers.
#[derive(Clone, PartialEq, Debug)]
pub struct Obs {
    pub routes: Vec<f32>,
    pub cities: Vec<f32>,
    pub tickets: Vec<f32>,
    pub players: Vec<f32>,
    pub global: Vec<f32>,
    pub slots: Vec<i32>,
    pub offer: Vec<i32>,
    pub events: Vec<i32>,
    pub legal: Vec<u8>,
}

impl Obs {
    pub fn new(map: &Map) -> Obs {
        let d = dims(map);
        Obs {
            routes: vec![0.0; d.routes],
            cities: vec![0.0; d.cities],
            tickets: vec![0.0; d.tickets],
            players: vec![0.0; d.players],
            global: vec![0.0; d.global],
            slots: vec![0; d.slots],
            offer: vec![0; d.offer],
            events: vec![0; d.events],
            legal: vec![0; d.legal],
        }
    }

    /// What `me` sees in `s`.
    pub fn of(map: &Map, s: &State, me: usize) -> Obs {
        let mut o = Obs::new(map);
        o.fill(map, s, me);
        o
    }

    /// Overwrite this observation with what `me` sees in `s`.
    pub fn fill(&mut self, map: &Map, s: &State, me: usize) {
        let mut out = ObsOut {
            routes: &mut self.routes,
            cities: &mut self.cities,
            tickets: &mut self.tickets,
            players: &mut self.players,
            global: &mut self.global,
            slots: &mut self.slots,
            offer: &mut self.offer,
            events: &mut self.events,
            legal: &mut self.legal,
        };
        encode(map, s, me, &mut out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{GameRng, Xoshiro};
    use crate::state::Rules;

    #[test]
    fn an_observation_has_the_sizes_the_network_is_built_for() {
        let map = Map::usa();
        let d = dims(map);
        assert_eq!(
            (d.routes, d.cities, d.tickets, d.players, d.global),
            (100 * 36, 36 * 8, 30 * 5, 5 * 40, 58)
        );
        assert_eq!((d.slots, d.offer, d.events, d.legal), (5, 3, 32 * 5, 422));
        assert_eq!(
            RULES_AT + 5 + 7,
            GLOBAL_F,
            "the rules and the reserved inputs end the global features"
        );
    }

    #[test]
    fn the_rules_are_in_the_observation_and_the_reserved_inputs_stay_zero() {
        let map = Map::usa();
        let official = Obs::of(map, &State::new(map, 2, &mut Xoshiro::new(1)), 0);
        assert_eq!(official.global[RULES_AT..RULES_AT + 5], [1.0, 1.0, 0.0, 1.0, 0.0]);
        let rules = Rules {
            route_points: true,
            longest_bonus: false,
            trains: 30,
            partial_tickets: true,
            route_price: 40,
        };
        let variant = Obs::of(map, &State::with_rules(map, 2, rules, &mut Xoshiro::new(1)), 0);
        assert_eq!(
            variant.global[RULES_AT..RULES_AT + 5],
            [0.4, 0.0, 0.0, 30.0 / 45.0, 1.0]
        );
        assert!(variant.global[RULES_AT + 5..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn legal_moves_are_marked_only_for_the_player_to_act() {
        let map = Map::usa();
        let mut rng = Xoshiro::new(5);
        let mut s = State::new(map, 3, &mut rng);
        let mut moves = Vec::new();
        for _ in 0..60 {
            if s.is_over() {
                break;
            }
            s.legal_actions(map, &mut moves);
            for me in 0..3 {
                let o = Obs::of(map, &s, me);
                let marked: Vec<u16> = (0..map.n_actions as u16)
                    .filter(|&a| o.legal[a as usize] == 1)
                    .collect();
                if me == s.current as usize {
                    assert_eq!(marked, moves);
                } else {
                    assert!(marked.is_empty());
                }
            }
            if moves.is_empty() {
                s.pass();
            } else {
                let a = moves[rng.below(moves.len() as u32) as usize] as usize;
                s.step(map, a, &mut rng);
            }
        }
    }
}
