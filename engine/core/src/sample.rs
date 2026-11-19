//! A hidden world consistent with what one player sees (a determinization): what a look-ahead from
//! that player's seat plays its lines in, and the check that an observation leaks nothing hidden.
//!
//! `sample_world(map, s, me, rng, weights)` keeps everything `me` sees (the board, the face-up row,
//! the discard pile, every public count, the cards each player is known to hold, `me`'s own hand
//! and tickets) and draws everything else afresh:
//!
//! - the other players' unknown cards and the deck's order: the cards nobody has seen are dealt at
//!   random, each opponent getting as many as their hand holds beyond its known cards;
//! - the other players' tickets, the tickets on offer to an opponent, and the ticket deck's order:
//!   from the tickets `me` has not seen, opponents' held tickets weighted by `weights` (the
//!   network's belief head gives them; uniform without them);
//! - in the opening, what the opponents who have chosen kept (2 or 3 of 3, uniform): the table is
//!   not told until everyone has chosen.
//!
//! What it does not use (the observation does not either): where in the ticket deck the tickets
//! `me` returned after a draw are. A player who returned tickets knows they sit at the bottom.

use crate::map::{Map, LOCO, MAX_TICKETS, N_CARD_TYPES, N_COLORS};
use crate::obs::{offered_to, opening_chosen};
use crate::rng::{GameRng, Xoshiro};
use crate::state::{Phase, State, CARDS_PER_COLOR, MAX_PLAYERS, NONE, N_LOCOS, N_TRAIN_CARDS, TICKETS_START_DEAL};

/// Per opponent, a weight for each ticket (the belief that they hold it); None = uniform.
pub type TicketWeights = [[f32; MAX_TICKETS]; MAX_PLAYERS];

pub fn sample_world(map: &Map, s: &State, me: usize, rng: &mut Xoshiro, weights: Option<&TicketWeights>) -> State {
    let mut w = *s;
    let n = s.n();

    // ---- train cards
    let mut pool = [0u8; N_CARD_TYPES];
    for c in 0..N_COLORS {
        pool[c] = CARDS_PER_COLOR;
    }
    pool[LOCO as usize] = N_LOCOS;
    let take = |pool: &mut [u8; N_CARD_TYPES], c: usize, k: u8| {
        debug_assert!(pool[c] >= k, "the cards in sight exceed the deck");
        pool[c] -= k;
    };
    for c in 0..N_CARD_TYPES {
        take(&mut pool, c, s.hand[me][c]);
    }
    for &c in s.face_up.iter().filter(|&&c| c != NONE) {
        take(&mut pool, c as usize, 1);
    }
    for &c in s.discard.as_slice() {
        take(&mut pool, c as usize, 1);
    }
    for q in (0..n).filter(|&q| q != me) {
        for c in 0..N_CARD_TYPES {
            take(&mut pool, c, s.known[q][c]);
        }
    }
    let mut unseen = [0u8; N_TRAIN_CARDS];
    let mut k = 0;
    for c in 0..N_CARD_TYPES {
        for _ in 0..pool[c] {
            unseen[k] = c as u8;
            k += 1;
        }
    }
    let unseen = &mut unseen[..k];
    rng.shuffle(unseen);
    let mut at = 0;
    for q in (0..n).filter(|&q| q != me) {
        let hidden = s.hand_size(q) as usize - s.known[q].iter().map(|&x| x as usize).sum::<usize>();
        w.hand[q] = s.known[q];
        for &c in &unseen[at..at + hidden] {
            w.hand[q][c as usize] += 1;
        }
        at += hidden;
    }
    debug_assert_eq!(unseen.len() - at, s.deck.len(), "the deck is what nobody has seen");
    w.deck.set(&unseen[at..]);

    // ---- tickets
    let (offer, n_offer) = offered_to(s, me);
    let mut seen = s.tickets_held[me];
    for &t in &offer[..n_offer] {
        seen |= 1u64 << t;
    }
    // in the opening, the tickets `me` already returned sit at the bottom of the deck, and `me` knows it
    let mine_back: Vec<u8> = if s.in_opening() && opening_chosen(s, me) {
        s.opening_dealt[me]
            .iter()
            .cloned()
            .filter(|&t| s.tickets_held[me] & (1u64 << t) == 0)
            .collect()
    } else {
        Vec::new()
    };
    for &t in &mine_back {
        seen |= 1u64 << t;
    }
    let mut hidden: Vec<u8> = (0..map.n_tickets() as u8)
        .filter(|&t| seen & (1u64 << t) == 0)
        .collect();
    rng.shuffle(&mut hidden);

    // who gets how many: in the opening, everyone else is dealt 3 anew and those who have chosen
    // keep 2 or 3; after it, each opponent's count is public
    let mut back: Vec<u8> = Vec::new(); // returned to the bottom of the ticket deck
    for q in (0..n).filter(|&q| q != me) {
        if s.in_opening() {
            let mut dealt = [NONE; TICKETS_START_DEAL];
            for d in dealt.iter_mut() {
                *d = hidden.pop().expect("tickets for the opening");
            }
            w.opening_dealt[q] = dealt;
            if opening_chosen(s, q) {
                let keep = 2 + rng.below(2) as usize;
                let skip = rng.below(3) as usize; // which one goes back when 2 are kept
                w.tickets_held[q] = 0;
                let mut kept = 0u8;
                for (i, &t) in dealt.iter().enumerate() {
                    if keep == 3 || i != skip {
                        w.tickets_held[q] |= 1u64 << t;
                        kept += 1;
                    } else {
                        back.push(t);
                    }
                }
                w.opening_kept[q] = kept;
            } else {
                w.tickets_held[q] = 0;
                w.opening_kept[q] = 0;
            }
            if s.phase == Phase::Keep && s.current as usize == q {
                w.pending = dealt;
                w.pending_len = TICKETS_START_DEAL as u8;
            }
        } else {
            let count = s.tickets_held[q].count_ones() as usize;
            w.tickets_held[q] = 0;
            for _ in 0..count {
                let i = match weights {
                    Some(wt) => weighted_pick(&hidden, &wt[q], rng),
                    None => rng.below(hidden.len() as u32) as usize,
                };
                let t = hidden.swap_remove(i);
                w.tickets_held[q] |= 1u64 << t;
            }
            if s.phase == Phase::Keep && s.current as usize == q {
                let m = s.pending_len as usize;
                let mut p = [NONE; 3];
                for slot in p.iter_mut().take(m) {
                    *slot = hidden.pop().expect("tickets on offer");
                }
                w.pending = p;
                w.pending_len = m as u8;
            }
        }
    }
    // the ticket deck: what is left, in a random order, then what the opening returned
    hidden.extend_from_slice(&back);
    hidden.extend_from_slice(&mine_back);
    w.ticket_deck.set(&hidden);
    w
}

/// Index into `items` drawn with probability proportional to `weight[item]` (uniform if all zero).
fn weighted_pick(items: &[u8], weight: &[f32; MAX_TICKETS], rng: &mut Xoshiro) -> usize {
    let total: f64 = items.iter().map(|&t| weight[t as usize].max(0.0) as f64).sum();
    if total <= 0.0 || !total.is_finite() {
        return rng.below(items.len() as u32) as usize;
    }
    let mut x = rng.unit() * total;
    for (i, &t) in items.iter().enumerate() {
        x -= weight[t as usize].max(0.0) as f64;
        if x < 0.0 {
            return i;
        }
    }
    items.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::Obs;
    use crate::state::N_TRAINS;

    /// Random positions of random games, every phase included.
    fn positions(n: usize, seed: u64, every: usize) -> Vec<State> {
        let map = Map::usa();
        let mut rng = Xoshiro::new(seed);
        let mut s = State::new(map, n, &mut rng);
        let mut moves = Vec::new();
        let mut out = Vec::new();
        let mut step = 0;
        while !s.is_over() && step < 3000 {
            if step % every == 0 {
                out.push(s);
            }
            s.legal_actions(map, &mut moves);
            if moves.is_empty() {
                s.pass();
            } else {
                let claims: Vec<u16> = moves.iter().cloned().filter(|&a| (a as usize) < map.n_claim).collect();
                let pick = if !claims.is_empty() && rng.below(10) < 6 {
                    &claims
                } else {
                    &moves
                };
                let a = pick[rng.below(pick.len() as u32) as usize] as usize;
                s.step(map, a, &mut rng);
            }
            step += 1;
        }
        out.push(s);
        out
    }

    fn cards_total(s: &State) -> [u32; N_CARD_TYPES] {
        let mut t = [0u32; N_CARD_TYPES];
        for p in 0..s.n() {
            for c in 0..N_CARD_TYPES {
                t[c] += s.hand[p][c] as u32;
            }
        }
        for &c in s.face_up.iter().filter(|&&c| c != NONE) {
            t[c as usize] += 1;
        }
        for &c in s.deck.as_slice().iter().chain(s.discard.as_slice()) {
            t[c as usize] += 1;
        }
        t
    }

    #[test]
    fn sampled_worlds_look_the_same() {
        let map = Map::usa();
        let mut rng = Xoshiro::new(99);
        let mut checked = 0;
        for n in 2..=5 {
            for seed in 0..12 {
                for s in positions(n, seed * 7 + n as u64, 3) {
                    for me in 0..n {
                        let w = sample_world(map, &s, me, &mut rng, None);
                        assert!(
                            Obs::of(map, &s, me) == Obs::of(map, &w, me),
                            "{n} players, seed {seed}, seat {me}, move {}",
                            s.moves
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 1000, "{checked}");
    }

    #[test]
    fn sampled_worlds_are_games() {
        let map = Map::usa();
        let mut rng = Xoshiro::new(5);
        for n in 2..=5 {
            for seed in 0..10 {
                for s in positions(n, seed + 100 * n as u64, 5) {
                    for me in 0..n {
                        let w = sample_world(map, &s, me, &mut rng, None);
                        let t = cards_total(&w);
                        for c in 0..N_COLORS {
                            assert_eq!(t[c], 12);
                        }
                        assert_eq!(t[LOCO as usize], 14);
                        for q in 0..n {
                            assert_eq!(w.hand_size(q), s.hand_size(q), "hand sizes are public");
                            for c in 0..N_CARD_TYPES {
                                assert!(w.hand[q][c] >= s.known[q][c], "known cards stay in the hand");
                            }
                            assert_eq!(w.trains[q], s.trains[q]);
                            assert!(w.trains[q] <= N_TRAINS);
                        }
                        assert_eq!(w.hand[me], s.hand[me]);
                        assert_eq!(w.tickets_held[me], s.tickets_held[me]);
                        // every ticket is somewhere exactly once
                        let mut seen = 0u64;
                        let mut count = 0;
                        let mut put = |t: u8| {
                            assert_eq!(seen & (1u64 << t), 0, "ticket {t} twice");
                            seen |= 1u64 << t;
                            count += 1;
                        };
                        for q in 0..n {
                            let mut h = w.tickets_held[q];
                            while h != 0 {
                                put(h.trailing_zeros() as u8);
                                h &= h - 1;
                            }
                        }
                        for &t in w.ticket_deck.as_slice() {
                            put(t);
                        }
                        if w.in_opening() {
                            for q in 0..n {
                                if !opening_chosen(&w, q) {
                                    for &t in &w.opening_dealt[q] {
                                        put(t);
                                    }
                                }
                            }
                        } else {
                            for &t in w.pending_tickets() {
                                put(t);
                            }
                        }
                        assert_eq!(count, map.n_tickets(), "{n} players, seat {me}, move {}", s.moves);
                        // and the world plays on
                        let mut moves = Vec::new();
                        let mut g = w;
                        let mut r2 = rng.fork();
                        for _ in 0..40 {
                            if g.is_over() {
                                break;
                            }
                            g.legal_actions(map, &mut moves);
                            if moves.is_empty() {
                                g.pass();
                            } else {
                                let a = moves[r2.below(moves.len() as u32) as usize] as usize;
                                g.step(map, a, &mut r2);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn weights_steer_the_tickets() {
        let map = Map::usa();
        let s = positions(2, 3, 1000).pop().unwrap();
        let mut rng = Xoshiro::new(1);
        let mut weights = [[0.0f32; MAX_TICKETS]; MAX_PLAYERS];
        // make ticket 0 almost sure for seat 1 when it is not mine
        if s.tickets_held[0] & 1 == 0 && s.tickets_held[1].count_ones() > 0 {
            for t in 0..map.n_tickets() {
                weights[1][t] = 1e-6;
            }
            weights[1][0] = 1.0;
            let hits = (0..200)
                .filter(|_| sample_world(map, &s, 0, &mut rng, Some(&weights)).tickets_held[1] & 1 != 0)
                .count();
            assert!(hits > 190, "{hits}");
        }
    }
}
