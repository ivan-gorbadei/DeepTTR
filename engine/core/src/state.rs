//! The state of a game and the rules that move it.
//!
//! The official rules of the base game on the USA map, for 2 to 5 players:
//!
//! - Turn: claim a route (pay its length in cards of one colour, locomotives wild; grey routes take
//!   any one colour), take two train cards (a face-up locomotive taken first is the whole turn, and
//!   cannot be the second card), or draw three tickets and keep at least one.
//! - Three face-up locomotives: the five go to the discard pile and five new are laid (at most six
//!   times in a row: a deck that holds little but locomotives must not loop).
//! - The deck runs out: the discard pile is shuffled into a new deck.
//! - A double route: one player never takes both; with fewer players than the map says
//!   (`both_doubles_min_players`, 4 on the USA map), once one is taken the other is closed.
//! - Opening: three tickets each, keep at least two, in seat order from the first player; nobody's
//!   choice is public before all have chosen (the observation's job, not the state's).
//! - A claim that leaves a player two trains or fewer starts the last round: every player, that
//!   one included, has one more turn.
//! - A claim is always paid with the fewest locomotives (colour cards first): a locomotive pays for
//!   everything a colour card pays for, so keeping it is never worse on this map.

use crate::map::{Map, LOCO, MAX_ROUTES, MAX_TICKETS, N_CARD_TYPES, N_COLORS, ROUTE_SCORE};
use crate::rng::GameRng;

pub const MAX_PLAYERS: usize = 5;
pub const N_TRAINS: u8 = 45;
pub const HAND_START: usize = 4;
pub const FACE_UP: usize = 5;
pub const CARDS_PER_COLOR: u8 = 12;
pub const N_LOCOS: u8 = 14;
pub const N_TRAIN_CARDS: usize = N_COLORS * CARDS_PER_COLOR as usize + N_LOCOS as usize;
pub const TICKETS_START_DEAL: usize = 3;
pub const KEEP_MIN_START: u8 = 2;
pub const KEEP_MIN_DRAW: u8 = 1;
pub const LAST_ROUND_TRAINS: u8 = 2;
pub const LONGEST_BONUS: i16 = 10;
pub const NONE: u8 = u8::MAX;

/// The rules a game is played and scored with: the official ones, or a training variant that changes
/// what scores, so that a curriculum can teach one skill at a time. The observation carries them
/// (`obs`), so one network plays every variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// claimed routes score their points (official: yes)
    pub route_points: bool,
    /// the longest continuous trail scores `LONGEST_BONUS` (official: yes)
    pub longest_bonus: bool,
    /// the trains each player starts with (official: 45)
    pub trains: u8,
    /// not official: a ticket not completed scores in proportion to the share of its shortest path
    /// built - only routes on a shortest path of the empty board count (`State::built_on_shortest`),
    /// from minus its points (nothing built, the official penalty) through 0 (half) to plus its
    /// points (completed) - every claim on its shortest path pays, a detour nothing
    pub partial_tickets: bool,
    /// with `route_points`, the share of their points claimed routes score, in percent (official: 100):
    /// a curriculum that starts with tickets alone raises it from 0 step by step
    pub route_price: u8,
}

impl Rules {
    pub const OFFICIAL: Rules = Rules {
        route_points: true,
        longest_bonus: true,
        trains: N_TRAINS,
        partial_tickets: false,
        route_price: 100,
    };

    /// A player's route points as they score at the route price, rounded.
    pub fn priced(&self, points: i16) -> i16 {
        ((points as i32 * self.route_price as i32 + 50) / 100) as i16
    }
}

impl Default for Rules {
    fn default() -> Self {
        Rules::OFFICIAL
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Phase {
    /// a player's turn begins: claim, draw a card, or draw tickets
    Turn = 0,
    /// the second train card of a turn
    Draw2 = 1,
    /// choose which of the tickets on offer to keep
    Keep = 2,
    Over = 3,
}

/// A pile of train cards; the top is the last.
#[derive(Clone, Copy)]
pub struct Cards {
    buf: [u8; N_TRAIN_CARDS],
    len: u8,
}

impl Cards {
    pub const fn new() -> Self {
        Cards {
            buf: [0; N_TRAIN_CARDS],
            len: 0,
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf[..self.len as usize]
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf[..self.len as usize]
    }
    #[inline]
    pub fn push(&mut self, c: u8) {
        self.buf[self.len as usize] = c;
        self.len += 1;
    }
    #[inline]
    pub fn pop(&mut self) -> Option<u8> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(self.buf[self.len as usize])
        }
    }
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }
    pub fn set(&mut self, cards: &[u8]) {
        self.buf[..cards.len()].copy_from_slice(cards);
        self.len = cards.len() as u8;
    }
    pub fn counts(&self) -> [u8; N_CARD_TYPES] {
        let mut n = [0u8; N_CARD_TYPES];
        for &c in self.as_slice() {
            n[c as usize] += 1;
        }
        n
    }
}

impl Default for Cards {
    fn default() -> Self {
        Cards::new()
    }
}

/// The destination ticket deck: drawn from the front, returned to the bottom.
#[derive(Clone, Copy)]
pub struct TicketDeck {
    buf: [u8; MAX_TICKETS],
    len: u8,
}

impl TicketDeck {
    pub const fn new() -> Self {
        TicketDeck {
            buf: [0; MAX_TICKETS],
            len: 0,
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf[..self.len as usize]
    }
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf[..self.len as usize]
    }
    pub fn set(&mut self, ids: &[u8]) {
        self.buf[..ids.len()].copy_from_slice(ids);
        self.len = ids.len() as u8;
    }
    /// Take up to `n` from the front.
    pub fn take_front(&mut self, n: usize, out: &mut [u8; 3]) -> usize {
        let k = n.min(self.len as usize);
        out[..k].copy_from_slice(&self.buf[..k]);
        self.buf.copy_within(k..self.len as usize, 0);
        self.len -= k as u8;
        k
    }
    #[inline]
    pub fn push_back(&mut self, t: u8) {
        self.buf[self.len as usize] = t;
        self.len += 1;
    }
}

impl Default for TicketDeck {
    fn default() -> Self {
        TicketDeck::new()
    }
}

/// How many public events the state remembers (the observation's event tokens).
pub const EVENTS: usize = 32;

/// Kinds of public event: what every player at the table sees happen.
pub mod event {
    pub const TAKE_FACE_UP: u8 = 1; // a = the card type
    pub const BLIND_DRAW: u8 = 2;
    pub const CLAIM: u8 = 3; // a = route, b = colour paid | locomotives used << 4
    pub const DRAW_TICKETS: u8 = 4; // a = tickets on offer
    pub const KEEP_TICKETS: u8 = 5; // a = kept, b = on offer
    pub const OPENING_KEEP: u8 = 6; // a = kept of the 3 dealt; told for all players when the opening ends
    pub const FACE_UP_WIPE: u8 = 7; // three face-up locomotives: the row was replaced
    pub const DECK_SHUFFLE: u8 = 8; // the discard pile became the deck
    pub const PASS: u8 = 9;
    pub const KINDS: usize = 10;
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Event {
    pub actor: u8,
    pub kind: u8,
    pub a: u8,
    pub b: u8,
}

/// Everything about a game. `Copy` (about 1.5 KB): a look-ahead copies it for every line it plays.
#[derive(Clone, Copy)]
pub struct State {
    pub n_players: u8,
    pub rules: Rules,
    pub phase: Phase,
    pub current: u8,
    /// turns left in the last round; -1 = it has not started
    pub last_round_left: i8,
    pub route_owner: [u8; MAX_ROUTES],
    pub trains: [u8; MAX_PLAYERS],
    pub route_points: [i16; MAX_PLAYERS],
    pub hand: [[u8; N_CARD_TYPES]; MAX_PLAYERS],
    pub face_up: [u8; FACE_UP],
    pub deck: Cards,
    pub discard: Cards,
    /// bit t = holds ticket t (kept; tickets on offer are in `pending`)
    pub tickets_held: [u64; MAX_PLAYERS],
    pub ticket_deck: TicketDeck,
    pub pending: [u8; 3],
    pub pending_len: u8,
    pub keep_min: u8,
    pub opening_remaining: u8,
    pub opening_first: u8,
    pub opening_dealt: [[u8; TICKETS_START_DEAL]; MAX_PLAYERS],
    // public history
    pub face_up_taken: [[u16; N_CARD_TYPES]; MAX_PLAYERS],
    pub paid_cards: [[u16; N_CARD_TYPES]; MAX_PLAYERS],
    pub blind_draws: [u16; MAX_PLAYERS],
    /// turns completed (every player's)
    pub turns: u16,
    /// moves made (every decision, the opening's too)
    pub moves: u16,
    // public knowledge
    /// cards each player is publicly known to hold: taken face up and not yet seen spent
    pub known: [[u8; N_CARD_TYPES]; MAX_PLAYERS],
    /// ticket draws made after the opening, and tickets kept from them
    pub ticket_draws: [u8; MAX_PLAYERS],
    pub tickets_kept: [u8; MAX_PLAYERS],
    /// tickets kept in the opening (told to the table only when everyone has chosen)
    pub opening_kept: [u8; MAX_PLAYERS],
    /// which they were, a bit per ticket (each player's own; for statistics of the starting tickets)
    pub opening_tickets: [u64; MAX_PLAYERS],
    /// per route, the board's memory of its claim: the turn it was claimed in (`turns` + 1 then;
    /// 0 = unclaimed) and what paid it (colour | locomotives << 4)
    pub route_turn: [u16; MAX_ROUTES],
    pub route_paid: [u8; MAX_ROUTES],
    /// the last EVENTS public events, oldest first once full (a ring: see `recent_events`)
    pub events: [Event; EVENTS],
    pub events_total: u16,
    pub deck_shuffles: u8,
}

#[derive(Clone, Debug, Default)]
pub struct FinalScore {
    pub n_players: usize,
    pub total: [i16; MAX_PLAYERS],
    pub route: [i16; MAX_PLAYERS],
    pub tickets: [i16; MAX_PLAYERS],
    pub longest: [i16; MAX_PLAYERS],
    pub completed: [i16; MAX_PLAYERS],
    /// 0 = first, by total, then tickets completed, then the longest trail (the rules' tie-breaks);
    /// places still tied share the average
    pub ranks: [f32; MAX_PLAYERS],
}

impl FinalScore {
    /// The players sharing first place.
    pub fn winners(&self) -> impl Iterator<Item = usize> + '_ {
        let best = self.ranks[..self.n_players]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        (0..self.n_players).filter(move |&p| self.ranks[p] == best)
    }

    /// The game's utility per seat: +1 for first place, -1 for last, linear between (tied places
    /// share); over the seats it sums to zero.
    pub fn utility(&self) -> [f32; MAX_PLAYERS] {
        let mut u = [0.0; MAX_PLAYERS];
        let n = self.n_players;
        for p in 0..n {
            u[p] = 1.0 - 2.0 * self.ranks[p] / (n as f32 - 1.0);
        }
        u
    }
}

impl State {
    /// A new game with the official rules: shuffle, deal, and start the opening ticket choice.
    pub fn new(map: &Map, n_players: usize, rng: &mut impl GameRng) -> State {
        State::with_rules(map, n_players, Rules::OFFICIAL, rng)
    }

    /// A new game with `rules` (the deal takes the same randomness whatever the rules).
    pub fn with_rules(map: &Map, n_players: usize, rules: Rules, rng: &mut impl GameRng) -> State {
        assert!((2..=MAX_PLAYERS).contains(&n_players), "2 to {MAX_PLAYERS} players");
        assert!(map.n_tickets() >= n_players * TICKETS_START_DEAL);
        assert!(
            rules.trains > LAST_ROUND_TRAINS,
            "more trains than the last round's trigger"
        );
        let mut s = State::blank(n_players, rules);
        s.current = rng.below(n_players as u32) as u8;

        let mut bag = [0u8; N_TRAIN_CARDS];
        let mut k = 0;
        for c in 0..N_COLORS as u8 {
            for _ in 0..CARDS_PER_COLOR {
                bag[k] = c;
                k += 1;
            }
        }
        for _ in 0..N_LOCOS {
            bag[k] = LOCO;
            k += 1;
        }
        rng.shuffle(&mut bag);
        for p in 0..n_players {
            for &c in &bag[p * HAND_START..(p + 1) * HAND_START] {
                s.hand[p][c as usize] += 1;
            }
        }
        let rest = &bag[n_players * HAND_START..];
        s.face_up.copy_from_slice(&rest[..FACE_UP]);
        s.deck.set(&rest[FACE_UP..]);
        s.refresh_face_up(rng);

        let nt = map.n_tickets();
        let mut tix = [0u8; MAX_TICKETS];
        for (i, t) in tix[..nt].iter_mut().enumerate() {
            *t = i as u8;
        }
        rng.shuffle(&mut tix[..nt]);
        let dealt = n_players * TICKETS_START_DEAL;
        s.ticket_deck.set(&tix[dealt..nt]);
        for p in 0..n_players {
            s.opening_dealt[p].copy_from_slice(&tix[p * TICKETS_START_DEAL..(p + 1) * TICKETS_START_DEAL]);
        }
        s.opening_first = s.current;
        s.opening_remaining = n_players as u8;
        s.start_opening_keep();
        s
    }

    /// A table before the deal: nothing dealt, nothing played.
    fn blank(n_players: usize, rules: Rules) -> State {
        State {
            n_players: n_players as u8,
            rules,
            phase: Phase::Turn,
            current: 0,
            last_round_left: -1,
            route_owner: [NONE; MAX_ROUTES],
            trains: [rules.trains; MAX_PLAYERS],
            route_points: [0; MAX_PLAYERS],
            hand: [[0; N_CARD_TYPES]; MAX_PLAYERS],
            face_up: [NONE; FACE_UP],
            deck: Cards::new(),
            discard: Cards::new(),
            tickets_held: [0; MAX_PLAYERS],
            ticket_deck: TicketDeck::new(),
            pending: [NONE; 3],
            pending_len: 0,
            keep_min: KEEP_MIN_DRAW,
            opening_remaining: 0,
            opening_first: 0,
            opening_dealt: [[NONE; TICKETS_START_DEAL]; MAX_PLAYERS],
            face_up_taken: [[0; N_CARD_TYPES]; MAX_PLAYERS],
            paid_cards: [[0; N_CARD_TYPES]; MAX_PLAYERS],
            blind_draws: [0; MAX_PLAYERS],
            turns: 0,
            moves: 0,
            known: [[0; N_CARD_TYPES]; MAX_PLAYERS],
            ticket_draws: [0; MAX_PLAYERS],
            tickets_kept: [0; MAX_PLAYERS],
            opening_kept: [0; MAX_PLAYERS],
            opening_tickets: [0; MAX_PLAYERS],
            route_turn: [0; MAX_ROUTES],
            route_paid: [0; MAX_ROUTES],
            events: [Event::default(); EVENTS],
            events_total: 0,
            deck_shuffles: 0,
        }
    }

    fn start_opening_keep(&mut self) {
        let p = self.current as usize;
        self.pending = self.opening_dealt[p];
        self.pending_len = TICKETS_START_DEAL as u8;
        self.keep_min = KEEP_MIN_START;
        self.phase = Phase::Keep;
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n_players as usize
    }
    #[inline]
    pub fn is_over(&self) -> bool {
        self.phase == Phase::Over
    }
    #[inline]
    pub fn in_opening(&self) -> bool {
        self.opening_remaining > 0
    }
    #[inline]
    pub fn hand_size(&self, p: usize) -> u32 {
        self.hand[p].iter().map(|&x| x as u32).sum()
    }
    #[inline]
    pub fn pending_tickets(&self) -> &[u8] {
        &self.pending[..self.pending_len as usize]
    }

    fn note(&mut self, actor: u8, kind: u8, a: u8, b: u8) {
        self.events[self.events_total as usize % EVENTS] = Event { actor, kind, a, b };
        self.events_total = self.events_total.wrapping_add(1);
    }

    /// The remembered public events, oldest first.
    pub fn recent_events(&self) -> impl Iterator<Item = Event> + '_ {
        let total = self.events_total as usize;
        let k = total.min(EVENTS);
        (total - k..total).map(move |i| self.events[i % EVENTS])
    }

    fn draw_from_deck(&mut self, rng: &mut impl GameRng) -> Option<u8> {
        if self.deck.is_empty() {
            if self.discard.is_empty() {
                return None;
            }
            self.deck = self.discard;
            rng.shuffle(self.deck.as_mut_slice());
            self.discard.clear();
            self.deck_shuffles = self.deck_shuffles.saturating_add(1);
            self.note(self.current, event::DECK_SHUFFLE, 0, 0);
        }
        self.deck.pop()
    }

    fn refill_slot(&mut self, slot: usize, rng: &mut impl GameRng) {
        self.face_up[slot] = self.draw_from_deck(rng).unwrap_or(NONE);
    }

    fn refresh_face_up(&mut self, rng: &mut impl GameRng) {
        for _ in 0..6 {
            let locos = self.face_up.iter().filter(|&&c| c == LOCO).count();
            if locos < 3 {
                return;
            }
            self.note(self.current, event::FACE_UP_WIPE, 0, 0);
            for i in 0..FACE_UP {
                let c = self.face_up[i];
                if c != NONE {
                    self.discard.push(c);
                }
            }
            self.face_up = [NONE; FACE_UP];
            for i in 0..FACE_UP {
                self.refill_slot(i, rng);
            }
        }
    }

    #[inline]
    fn can_pay(&self, p: usize, color: u8, length: u8) -> bool {
        self.hand[p][color as usize] as u32 + self.hand[p][LOCO as usize] as u32 >= length as u32
    }

    /// Pay a claim (colour cards first); returns the locomotives used.
    fn pay(&mut self, p: usize, color: u8, length: u8) -> u8 {
        let use_color = self.hand[p][color as usize].min(length);
        let locos = length - use_color;
        self.hand[p][color as usize] -= use_color;
        self.hand[p][LOCO as usize] -= locos;
        self.paid_cards[p][color as usize] += use_color as u16;
        self.paid_cards[p][LOCO as usize] += locos as u16;
        // the table saw these cards leave the hand: known ones first
        let k = &mut self.known[p];
        k[color as usize] -= k[color as usize].min(use_color);
        k[LOCO as usize] -= k[LOCO as usize].min(locos);
        for _ in 0..use_color {
            self.discard.push(color);
        }
        for _ in 0..locos {
            self.discard.push(LOCO);
        }
        locos
    }

    /// May `p` take route `r` as far as the double-route rules go?
    #[inline]
    pub(crate) fn double_allows(&self, map: &Map, p: usize, r: usize) -> bool {
        let siblings = &map.pairs[map.routes[r].pair as usize];
        if siblings.len() < 2 {
            return true;
        }
        let both_open = self.n_players >= map.both_doubles_min_players;
        for &j in siblings {
            let j = j as usize;
            if j == r {
                continue;
            }
            let owner = self.route_owner[j];
            if owner == p as u8 || (!both_open && owner != NONE) {
                return false;
            }
        }
        true
    }

    #[inline]
    fn can_draw_any(&self) -> bool {
        self.face_up.iter().any(|&c| c != NONE) || !self.deck.is_empty() || !self.discard.is_empty()
    }

    /// Is route `r` still open to anybody (unowned, and no closed double)?
    pub fn route_open(&self, map: &Map, r: usize) -> bool {
        if self.route_owner[r] != NONE {
            return false;
        }
        if self.n_players >= map.both_doubles_min_players {
            return true;
        }
        map.pairs[map.routes[r].pair as usize]
            .iter()
            .all(|&j| self.route_owner[j as usize] == NONE)
    }

    /// The legal actions, in action-id order.
    pub fn legal_actions(&self, map: &Map, out: &mut Vec<u16>) {
        out.clear();
        let p = self.current as usize;
        match self.phase {
            Phase::Over => {}
            Phase::Keep => {
                let n = self.pending_len;
                for bits in 1u8..(1u8 << n) {
                    if bits.count_ones() as u8 >= self.keep_min {
                        out.push(map.keep_action(bits) as u16);
                    }
                }
            }
            Phase::Draw2 => {
                for slot in 0..FACE_UP {
                    let c = self.face_up[slot];
                    if c != NONE && c != LOCO {
                        out.push(map.draw_action(slot) as u16);
                    }
                }
                if !self.deck.is_empty() || !self.discard.is_empty() {
                    out.push(map.blind_action() as u16);
                }
            }
            Phase::Turn => {
                let trains = self.trains[p];
                let mut last_route = usize::MAX;
                let mut route_ok = false;
                for (ai, claim) in map.claims.iter().enumerate() {
                    let r = claim.route as usize;
                    if r != last_route {
                        last_route = r;
                        let route = &map.routes[r];
                        route_ok =
                            self.route_owner[r] == NONE && trains >= route.length && self.double_allows(map, p, r);
                    }
                    if route_ok && self.can_pay(p, claim.color, map.routes[r].length) {
                        out.push(ai as u16);
                    }
                }
                if self.can_draw_any() {
                    for slot in 0..FACE_UP {
                        if self.face_up[slot] != NONE {
                            out.push(map.draw_action(slot) as u16);
                        }
                    }
                    if !self.deck.is_empty() || !self.discard.is_empty() {
                        out.push(map.blind_action() as u16);
                    }
                }
                if !self.ticket_deck.is_empty() {
                    out.push(map.tickets_action() as u16);
                }
            }
        }
    }

    pub fn is_legal(&self, map: &Map, action: usize) -> bool {
        let mut v = Vec::with_capacity(64);
        self.legal_actions(map, &mut v);
        v.contains(&(action as u16))
    }

    /// Player `p` takes a train card into the hand: a face-up slot (0-4) or the deck (5).
    /// Returns (a card was taken, it was a face-up locomotive).
    fn take_train_card(&mut self, p: usize, slot: usize, rng: &mut impl GameRng) -> (bool, bool) {
        if slot == 5 {
            return match self.draw_from_deck(rng) {
                Some(c) => {
                    self.hand[p][c as usize] += 1;
                    self.blind_draws[p] += 1;
                    self.note(p as u8, event::BLIND_DRAW, 0, 0);
                    (true, false)
                }
                None => (false, false),
            };
        }
        let card = self.face_up[slot];
        if card == NONE {
            return (false, false);
        }
        self.hand[p][card as usize] += 1;
        self.face_up_taken[p][card as usize] += 1;
        self.known[p][card as usize] += 1;
        self.note(p as u8, event::TAKE_FACE_UP, card, slot as u8);
        self.refill_slot(slot, rng);
        self.refresh_face_up(rng);
        (true, card == LOCO)
    }

    /// End the current player's turn; `triggered` = this turn's claim started the last round.
    pub fn end_turn(&mut self, triggered: bool) {
        self.turns += 1;
        if self.last_round_left >= 0 {
            self.last_round_left -= 1;
            if self.last_round_left <= 0 {
                self.phase = Phase::Over;
                return;
            }
        } else if triggered {
            self.last_round_left = self.n_players as i8;
        }
        self.current = (self.current + 1) % self.n_players;
        self.phase = Phase::Turn;
    }

    /// Play a legal action (debug builds check that it is legal).
    pub fn step(&mut self, map: &Map, action: usize, rng: &mut impl GameRng) {
        debug_assert!(
            self.is_legal(map, action),
            "illegal action {action}: {}",
            map.describe(action)
        );
        if self.phase == Phase::Over {
            return;
        }
        self.moves += 1;
        let p = self.current as usize;
        match self.phase {
            Phase::Over => {}
            Phase::Keep => {
                let mut bits = map.keep_mask_of(action);
                let n = self.pending_len as usize;
                if bits & ((1u8 << n) - 1) == 0 {
                    bits = 1; // a legal action never gets here; keep the first rather than nothing
                }
                let mut kept = 0u8;
                for i in 0..n {
                    let t = self.pending[i];
                    if bits & (1 << i) != 0 {
                        self.tickets_held[p] |= 1u64 << t;
                        kept += 1;
                    } else {
                        self.ticket_deck.push_back(t);
                    }
                }
                self.pending = [NONE; 3];
                self.pending_len = 0;
                if self.opening_remaining > 0 {
                    self.opening_kept[p] = kept;
                    self.opening_tickets[p] = self.tickets_held[p]; // nothing was held before the opening
                    self.opening_remaining -= 1;
                    self.current = (self.current + 1) % self.n_players;
                    if self.opening_remaining > 0 {
                        self.start_opening_keep();
                    } else {
                        // everyone has chosen: the table learns how many each kept
                        for i in 0..self.n_players {
                            let q = (self.opening_first + i) % self.n_players;
                            self.note(
                                q,
                                event::OPENING_KEEP,
                                self.opening_kept[q as usize],
                                TICKETS_START_DEAL as u8,
                            );
                        }
                        self.current = self.opening_first;
                        self.keep_min = KEEP_MIN_DRAW;
                        self.phase = Phase::Turn;
                    }
                    return;
                }
                self.tickets_kept[p] = self.tickets_kept[p].saturating_add(kept);
                self.note(p as u8, event::KEEP_TICKETS, kept, n as u8);
                self.end_turn(false);
            }
            Phase::Draw2 => {
                let slot = action - map.n_claim;
                self.take_train_card(p, slot, rng);
                self.end_turn(false);
            }
            Phase::Turn => {
                if action < map.n_claim {
                    let claim = map.claims[action];
                    let r = claim.route as usize;
                    let length = map.routes[r].length;
                    let locos = self.pay(p, claim.color, length);
                    self.route_owner[r] = p as u8;
                    self.route_turn[r] = self.turns + 1;
                    self.route_paid[r] = claim.color | (locos << 4);
                    self.trains[p] -= length;
                    self.route_points[p] += ROUTE_SCORE[length as usize];
                    self.note(p as u8, event::CLAIM, r as u8, claim.color | (locos << 4));
                    let triggered = self.trains[p] <= LAST_ROUND_TRAINS && self.last_round_left < 0;
                    self.end_turn(triggered);
                } else if action == map.tickets_action() {
                    let mut got = [NONE; 3];
                    let k = self.ticket_deck.take_front(3, &mut got);
                    self.pending = got;
                    self.pending_len = k as u8;
                    self.keep_min = KEEP_MIN_DRAW;
                    self.phase = Phase::Keep;
                    self.ticket_draws[p] = self.ticket_draws[p].saturating_add(1);
                    self.note(p as u8, event::DRAW_TICKETS, k as u8, 0);
                    if k == 0 {
                        self.end_turn(false);
                    }
                } else {
                    let slot = action - map.n_claim;
                    let (_, face_loco) = self.take_train_card(p, slot, rng);
                    if face_loco || !self.can_draw_any() {
                        self.end_turn(false);
                    } else {
                        self.phase = Phase::Draw2;
                    }
                }
            }
        }
    }

    /// A player with no legal action passes (rare: no route payable, no card anywhere, no ticket
    /// left).
    pub fn pass(&mut self) {
        self.moves += 1;
        self.note(self.current, event::PASS, 0, 0);
        self.end_turn(false);
    }

    /// The cities each player's routes connect: a union-find root per city, per player.
    pub fn components(&self, map: &Map, p: usize) -> [u8; crate::map::MAX_CITIES] {
        let mut parent = [0u8; crate::map::MAX_CITIES];
        for (i, x) in parent.iter_mut().enumerate() {
            *x = i as u8;
        }
        fn find(parent: &mut [u8; crate::map::MAX_CITIES], mut x: u8) -> u8 {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }
        for (r, route) in map.routes.iter().enumerate() {
            if self.route_owner[r] == p as u8 {
                let (ra, rb) = (find(&mut parent, route.a), find(&mut parent, route.b));
                if ra != rb {
                    parent[rb as usize] = ra;
                }
            }
        }
        for i in 0..map.n_cities() {
            parent[i] = find(&mut parent, i as u8);
        }
        parent
    }

    pub fn ticket_completed(&self, map: &Map, p: usize, t: usize) -> bool {
        let comp = self.components(map, p);
        let tk = &map.tickets[t];
        comp[tk.a as usize] == comp[tk.b as usize]
    }

    /// The longest trail (each route once) through a player's routes.
    pub fn longest_trail(&self, map: &Map, p: usize) -> i16 {
        let mut edges: Vec<(u8, u8, u8)> = Vec::with_capacity(48);
        for (r, route) in map.routes.iter().enumerate() {
            if self.route_owner[r] == p as u8 {
                edges.push((route.a, route.b, route.length));
            }
        }
        longest_trail(&edges, map.n_cities())
    }

    /// The fewest trains seat `p` still needs to join each ticket's two cities: its own routes cost
    /// nothing, a route still open costs its length, any other route is closed; `NONE` = the cities
    /// can no longer be joined. How far each ticket is, not which way to build it (there are often
    /// several): a training target of the network, never one of its inputs.
    pub fn ticket_costs(&self, map: &Map, p: usize) -> [u8; MAX_TICKETS] {
        use crate::map::MAX_CITIES;
        const FAR: u16 = u16::MAX;
        let nc = map.n_cities();
        let mut w = [[FAR; MAX_CITIES]; MAX_CITIES];
        for (r, route) in map.routes.iter().enumerate() {
            let cost = if self.route_owner[r] == p as u8 {
                0
            } else if self.route_open(map, r) {
                route.length as u16
            } else {
                continue;
            };
            let (a, b) = (route.a as usize, route.b as usize);
            w[a][b] = w[a][b].min(cost);
            w[b][a] = w[a][b];
        }
        let mut from: [Option<[u16; MAX_CITIES]>; MAX_CITIES] = [None; MAX_CITIES];
        let mut out = [NONE; MAX_TICKETS];
        for (t, tk) in map.tickets.iter().enumerate() {
            let src = tk.a as usize;
            let dist = *from[src].get_or_insert_with(|| crate::map::shortest(nc, &w, src));
            let d = dist[tk.b as usize];
            out[t] = if d == FAR { NONE } else { d.min(254) as u8 };
        }
        out
    }

    /// The trains seat `p` owns on the best of ticket `t`'s shortest paths of the empty board: the
    /// most of its own trains along any one such path (`Map::ticket_paths`, in the order of distance
    /// from the ticket's first city). Only the shortest way counts - a detour does not, nor half of
    /// each of two different shortest paths.
    pub fn built_on_shortest(&self, map: &Map, p: usize, t: usize) -> u8 {
        let tk = &map.tickets[t];
        let mut best = [i16::MIN; crate::map::MAX_CITIES];
        best[tk.a as usize] = 0;
        for &(from, to, r) in &map.ticket_paths[t] {
            let b = best[from as usize];
            if b == i16::MIN {
                continue;
            }
            let own = if self.route_owner[r as usize] == p as u8 {
                map.routes[r as usize].length as i16
            } else {
                0
            };
            best[to as usize] = best[to as usize].max(b + own);
        }
        best[tk.b as usize].max(0) as u8
    }

    /// The final score with the game's rules; call when the game is over (or to score a position as
    /// if it were).
    pub fn finalize(&self, map: &Map) -> FinalScore {
        let n = self.n();
        let mut f = FinalScore {
            n_players: n,
            ..Default::default()
        };
        (f.tickets, f.completed) = self.ticket_points(map);
        for p in 0..n {
            f.route[p] = self.route_points[p];
            f.longest[p] = self.longest_trail(map, p);
        }
        for p in 0..n {
            let route = if self.rules.route_points {
                self.rules.priced(f.route[p])
            } else {
                0
            };
            f.total[p] = route + f.tickets[p];
        }
        let best = f.longest[..n].iter().cloned().max().unwrap_or(0);
        if self.rules.longest_bonus && best > 0 {
            for p in 0..n {
                if f.longest[p] == best {
                    f.total[p] += LONGEST_BONUS;
                }
            }
        }
        let key = |p: usize| (-(f.total[p] as i32), -(f.completed[p] as i32), -(f.longest[p] as i32));
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&p| (key(p), p));
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && key(order[j]) == key(order[i]) {
                j += 1;
            }
            let avg = (i + j - 1) as f32 / 2.0;
            for &q in &order[i..j] {
                f.ranks[q] = avg;
            }
            i = j;
        }
        f
    }

    /// Each player's ticket points if the game ended now, by the game's rules (the partial-ticket rule
    /// included).
    pub fn tickets_score(&self, map: &Map) -> [i16; MAX_PLAYERS] {
        self.ticket_points(map).0
    }

    /// Each player's ticket points by the game's rules, and the tickets it has completed.
    fn ticket_points(&self, map: &Map) -> ([i16; MAX_PLAYERS], [i16; MAX_PLAYERS]) {
        let n = self.n();
        let mut points = [0i16; MAX_PLAYERS];
        let mut completed = [0i16; MAX_PLAYERS];
        let mut failed = [0u64; MAX_PLAYERS];
        for p in 0..n {
            let comp = self.components(map, p);
            let mut held = self.tickets_held[p];
            while held != 0 {
                let t = held.trailing_zeros() as usize;
                held &= held - 1;
                let tk = &map.tickets[t];
                if comp[tk.a as usize] == comp[tk.b as usize] {
                    points[p] += tk.points as i16;
                    completed[p] += 1;
                } else {
                    points[p] -= tk.points as i16;
                    failed[p] |= 1u64 << t;
                }
            }
        }
        if self.rules.partial_tickets {
            for p in (0..n).filter(|&p| failed[p] != 0) {
                let mut bits = failed[p];
                while bits != 0 {
                    let t = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let pts = map.tickets[t].points as i16;
                    let whole = map.ticket_length[t];
                    let left = whole.saturating_sub(self.built_on_shortest(map, p, t));
                    points[p] += pts + partial_score(pts, left, whole);
                }
            }
        }
        (points, completed)
    }
}

/// A ticket not completed, under the partial-ticket rule: its points x (2 x share built - 1), the
/// share built being 1 - `left` / the trains of its whole shortest path on an empty board (`whole`),
/// `left` the trains of that path not yet the player's (`State::built_on_shortest`); nothing built,
/// or `left` NONE or longer than the whole path: minus its points, as the official rules.
pub fn partial_score(points: i16, left: u8, whole: u8) -> i16 {
    if left == NONE || whole == 0 {
        return -points;
    }
    let built = 1.0 - (left as f32 / whole as f32).min(1.0);
    (points as f32 * (2.0 * built - 1.0)).round() as i16
}

/// The longest trail through `edges` (a, b, length), each edge used at most once.
pub fn longest_trail(edges: &[(u8, u8, u8)], n_cities: usize) -> i16 {
    if edges.is_empty() {
        return 0;
    }
    debug_assert!(edges.len() <= 64, "a player owns at most 45 routes");
    let mut adj: Vec<Vec<(u8, u8, u8)>> = vec![Vec::new(); n_cities]; // (next city, edge, length)
    for (i, &(a, b, w)) in edges.iter().enumerate() {
        adj[a as usize].push((b, i as u8, w));
        adj[b as usize].push((a, i as u8, w));
    }
    fn dfs(adj: &[Vec<(u8, u8, u8)>], node: u8, used: u64, length: i16, best: &mut i16) {
        if length > *best {
            *best = length;
        }
        for &(next, e, w) in &adj[node as usize] {
            let bit = 1u64 << e;
            if used & bit == 0 {
                dfs(adj, next, used | bit, length + w as i16, best);
            }
        }
    }
    let mut best = 0i16;
    let mut seen = [false; crate::map::MAX_CITIES];
    for &(a, b, _) in edges {
        for c in [a, b] {
            if !seen[c as usize] {
                seen[c as usize] = true;
                dfs(&adj, c, 0, 0, &mut best);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;
    use crate::rng::Xoshiro;

    fn random_game(n: usize, seed: u64) -> (State, FinalScore) {
        let map = Map::usa();
        let mut rng = Xoshiro::new(seed);
        let mut s = State::new(map, n, &mut rng);
        let mut moves = Vec::new();
        let mut steps = 0;
        while !s.is_over() && steps < 3000 {
            s.legal_actions(map, &mut moves);
            if moves.is_empty() {
                s.pass();
            } else {
                let a = moves[rng.below(moves.len() as u32) as usize] as usize;
                s.step(map, a, &mut rng);
            }
            steps += 1;
        }
        let f = s.finalize(map);
        (s, f)
    }

    #[test]
    fn the_board_remembers_every_claim() {
        // each claimed route: its turn within the game, its colour (a coloured route's own), and
        // colour + locomotives paid adding up to the paid-card totals of its owner
        let map = Map::usa();
        for n in 2..=5 {
            for seed in 0..20 {
                let (s, _) = random_game(n, 700 + seed);
                let mut paid = [[0u16; N_CARD_TYPES]; MAX_PLAYERS];
                for (r, route) in map.routes.iter().enumerate() {
                    let owner = s.route_owner[r];
                    if owner == NONE {
                        assert_eq!(s.route_turn[r], 0);
                        continue;
                    }
                    assert!(
                        s.route_turn[r] >= 1 && s.route_turn[r] <= s.turns + 1,
                        "turn {}",
                        s.route_turn[r]
                    );
                    let (color, locos) = (s.route_paid[r] & 0x0f, s.route_paid[r] >> 4);
                    if let Some(c) = route.color {
                        assert_eq!(color, c);
                    }
                    assert!(locos <= route.length);
                    paid[owner as usize][color as usize] += (route.length - locos) as u16;
                    paid[owner as usize][LOCO as usize] += locos as u16;
                }
                for q in 0..n {
                    assert_eq!(paid[q], s.paid_cards[q], "{n} players, seed {seed}, seat {q}");
                }
            }
        }
    }

    /// Cards are neither made nor lost: hands + face-up + deck + discard = the 110 cards.
    fn card_total(s: &State) -> [u32; N_CARD_TYPES] {
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
    fn random_games_keep_every_card_and_ticket() {
        let map = Map::usa();
        for n in 2..=5 {
            for seed in 0..40 {
                let (s, f) = random_game(n, seed * 31 + n as u64);
                let t = card_total(&s);
                for c in 0..N_COLORS {
                    assert_eq!(t[c], 12, "colour {c}, {n} players, seed {seed}");
                }
                assert_eq!(t[LOCO as usize], 14);
                for p in 0..n {
                    for c in 0..N_CARD_TYPES {
                        assert!(s.known[p][c] <= s.hand[p][c], "a known card is in the hand");
                    }
                }
                let held: u32 = (0..n).map(|p| s.tickets_held[p].count_ones()).sum();
                assert_eq!(
                    held as usize + s.ticket_deck.len() + s.pending_len as usize,
                    map.n_tickets()
                );
                for p in 0..n {
                    let used: u32 = (0..map.n_routes())
                        .filter(|&r| s.route_owner[r] == p as u8)
                        .map(|r| map.routes[r].length as u32)
                        .sum();
                    assert_eq!(used + s.trains[p] as u32, N_TRAINS as u32);
                    assert!(
                        f.total[p]
                            == f.route[p]
                                + f.tickets[p]
                                + if f.longest[p] == *f.longest[..n].iter().max().unwrap() && f.longest[p] > 0 {
                                    LONGEST_BONUS
                                } else {
                                    0
                                }
                    );
                }
            }
        }
    }

    #[test]
    fn variants_score_what_their_rules_say() {
        let map = Map::usa();
        for (k, rules) in [
            Rules {
                route_points: false,
                ..Rules::OFFICIAL
            },
            Rules {
                longest_bonus: false,
                ..Rules::OFFICIAL
            },
            Rules {
                route_points: false,
                longest_bonus: false,
                trains: 20,
                partial_tickets: true,
                route_price: 100,
            },
            Rules {
                trains: 30,
                ..Rules::OFFICIAL
            },
            Rules {
                route_price: 30,
                ..Rules::OFFICIAL
            },
            Rules {
                route_price: 0,
                ..Rules::OFFICIAL
            },
        ]
        .into_iter()
        .enumerate()
        {
            for n in 2..=5 {
                for seed in 0..10 {
                    let mut rng = Xoshiro::new(seed * 7 + n as u64 + 100 * k as u64);
                    let mut s = State::with_rules(map, n, rules, &mut rng);
                    assert_eq!(s.trains[..n], vec![rules.trains; n][..]);
                    let mut moves = Vec::new();
                    let mut steps = 0;
                    while !s.is_over() && steps < 5000 {
                        s.legal_actions(map, &mut moves);
                        if moves.is_empty() {
                            s.pass();
                        } else {
                            let a = moves[rng.below(moves.len() as u32) as usize] as usize;
                            s.step(map, a, &mut rng);
                        }
                        steps += 1;
                    }
                    let f = s.finalize(map);
                    let best = *f.longest[..n].iter().max().unwrap();
                    for p in 0..n {
                        let used: u32 = (0..map.n_routes())
                            .filter(|&r| s.route_owner[r] == p as u8)
                            .map(|r| map.routes[r].length as u32)
                            .sum();
                        assert_eq!(used + s.trains[p] as u32, rules.trains as u32);
                        let bonus = if rules.longest_bonus && best > 0 && f.longest[p] == best {
                            LONGEST_BONUS
                        } else {
                            0
                        };
                        // routes at their price, rounded: 30 % of 15 points is 5 (4.5 up)
                        let route = if rules.route_points {
                            ((f.route[p] as i32 * rules.route_price as i32 + 50) / 100) as i16
                        } else {
                            0
                        };
                        assert_eq!(f.total[p], route + f.tickets[p] + bonus, "rules {rules:?}, {n} players");
                    }
                }
            }
        }
    }

    /// The routes of a cheapest path (in trains) between a ticket's cities on the empty map.
    fn cheapest_path(map: &Map, t: usize) -> Vec<usize> {
        let nc = map.n_cities();
        let (a, b) = (map.tickets[t].a as usize, map.tickets[t].b as usize);
        let (mut dist, mut via, mut done) = (vec![u16::MAX; nc], vec![usize::MAX; nc], vec![false; nc]);
        dist[a] = 0;
        while let Some(v) = (0..nc)
            .filter(|&v| !done[v] && dist[v] < u16::MAX)
            .min_by_key(|&v| dist[v])
        {
            done[v] = true;
            for (r, route) in map.routes.iter().enumerate() {
                let (x, y) = (route.a as usize, route.b as usize);
                let w = if x == v {
                    y
                } else if y == v {
                    x
                } else {
                    continue;
                };
                if dist[v] + (route.length as u16) < dist[w] {
                    dist[w] = dist[v] + route.length as u16;
                    via[w] = r;
                }
            }
        }
        let (mut path, mut c) = (vec![], b);
        while c != a {
            let route = &map.routes[via[c]];
            path.push(via[c]);
            c = if route.a as usize == c {
                route.b as usize
            } else {
                route.a as usize
            };
        }
        path.reverse();
        path
    }

    #[test]
    fn a_partial_ticket_scores_by_the_share_of_its_shortest_path_built() {
        assert_eq!(partial_score(10, 0, 8), 10);
        assert_eq!(partial_score(10, 4, 8), 0);
        assert_eq!(partial_score(10, 8, 8), -10);
        assert_eq!(partial_score(10, 2, 8), 5);
        assert_eq!(partial_score(10, 12, 8), -10, "the way left longer than the whole path");
        assert_eq!(partial_score(10, NONE, 8), -10, "no longer joinable");
        let map = Map::usa();
        let t = (0..map.n_tickets()).max_by_key(|&t| map.ticket_length[t]).unwrap();
        let (tk, whole, pts) = (&map.tickets[t], map.ticket_length[t], map.tickets[t].points as i16);
        let mut s = State::with_rules(
            map,
            2,
            Rules {
                partial_tickets: true,
                ..Rules::OFFICIAL
            },
            &mut Xoshiro::new(3),
        );
        s.tickets_held = [0; MAX_PLAYERS];
        s.tickets_held[0] = 1u64 << t;
        // seat 0 builds a shortest path but its last route: that share of the path counts
        let path = cheapest_path(map, t);
        for &r in &path[..path.len() - 1] {
            s.route_owner[r] = 0;
        }
        let last = map.routes[path[path.len() - 1]].length;
        assert_eq!(s.built_on_shortest(map, 0, t), whole - last);
        let f = s.finalize(map);
        assert_eq!(f.tickets[0], partial_score(pts, last, whole));
        assert!(f.tickets[0] > -pts, "most of the path built: {}", f.tickets[0]);
        assert_eq!(f.completed[0], 0);
        s.rules.partial_tickets = false;
        assert_eq!(
            s.finalize(map).tickets[0],
            -pts,
            "the official penalty without the rule"
        );
        // a route at the first city on no shortest path - a detour - counts nothing
        let on: Vec<u16> = map.ticket_paths[t].iter().map(|e| e.2).collect();
        let detour = (0..map.routes.len())
            .find(|&r| (map.routes[r].a == tk.a || map.routes[r].b == tk.a) && !on.contains(&(r as u16)))
            .expect("a route off the shortest paths");
        let mut s2 = State::with_rules(
            map,
            2,
            Rules {
                partial_tickets: true,
                ..Rules::OFFICIAL
            },
            &mut Xoshiro::new(3),
        );
        s2.tickets_held = s.tickets_held;
        s2.route_owner[detour] = 0;
        assert_eq!(s2.built_on_shortest(map, 0, t), 0);
        assert_eq!(s2.finalize(map).tickets[0], -pts);
        // every route of the board: a whole shortest path built
        s2.route_owner = [0; MAX_ROUTES];
        assert_eq!(s2.built_on_shortest(map, 0, t), whole);
    }

    #[test]
    fn ticket_costs_count_the_trains_still_needed() {
        let map = Map::usa();
        let s = State::new(map, 2, &mut Xoshiro::new(5));
        let fresh = s.ticket_costs(map, 0);
        for t in 0..map.n_tickets() {
            assert!(fresh[t] > 0 && fresh[t] != NONE, "ticket {t}: {}", fresh[t]);
        }
        assert_eq!(
            fresh,
            s.ticket_costs(map, 1),
            "nobody owns anything: the same for both seats"
        );
        // seat 0 owns every route: every ticket is done; seat 1 can join nothing
        let mut all = s;
        all.route_owner[..map.n_routes()].fill(0);
        assert!(all.ticket_costs(map, 0)[..map.n_tickets()].iter().all(|&c| c == 0));
        assert!(all.ticket_costs(map, 1)[..map.n_tickets()].iter().all(|&c| c == NONE));
        // a claimed route on the way makes a ticket cheaper for its owner, dearer or the same for the other
        let mut one = s;
        let r = (0..map.n_routes()).find(|&r| map.routes[r].length >= 3).unwrap();
        one.route_owner[r] = 0;
        let (mine, theirs) = (one.ticket_costs(map, 0), one.ticket_costs(map, 1));
        for t in 0..map.n_tickets() {
            assert!(mine[t] <= fresh[t] && theirs[t] >= fresh[t], "ticket {t}");
        }
    }

    #[test]
    fn utility_is_plus_one_for_first_minus_one_for_last_and_sums_to_zero() {
        for n in 2..=5 {
            for seed in 0..20 {
                let (_, f) = random_game(n, 900 + seed);
                let u = f.utility();
                assert!(u[..n].iter().sum::<f32>().abs() < 1e-5, "{n} players: {u:?}");
                assert!(u[n..].iter().all(|&x| x == 0.0));
                for p in 0..n {
                    assert!((-1.0..=1.0).contains(&u[p]));
                    assert_eq!(u[p] == 1.0, f.ranks[p] == 0.0, "only a sole winner has +1");
                }
                assert_eq!(
                    f.winners().count(),
                    f.ranks[..n]
                        .iter()
                        .filter(|&&r| r == f.ranks[..n].iter().cloned().fold(f32::INFINITY, f32::min))
                        .count()
                );
            }
        }
    }

    #[test]
    fn two_players_never_share_a_double() {
        let map = Map::usa();
        for seed in 0..200 {
            let (s, _) = random_game(2, seed);
            for pair in map.pairs.iter().filter(|p| p.len() == 2) {
                let owned = pair.iter().filter(|&&r| s.route_owner[r as usize] != NONE).count();
                assert!(owned <= 1, "seed {seed}: both routes of a double taken with 2 players");
            }
        }
    }

    #[test]
    fn four_players_may_share_a_double_but_not_alone() {
        let map = Map::usa();
        let mut shared = 0;
        for seed in 0..200 {
            let (s, _) = random_game(4, seed);
            for pair in map.pairs.iter().filter(|p| p.len() == 2) {
                let (a, b) = (s.route_owner[pair[0] as usize], s.route_owner[pair[1] as usize]);
                if a != NONE && b != NONE {
                    assert_ne!(a, b, "seed {seed}: one player holds both routes of a double");
                    shared += 1;
                }
            }
        }
        assert!(shared > 0, "some double is shared in 200 random 4-player games");
    }

    #[test]
    fn longest_trail_uses_each_route_once() {
        // a triangle with a tail: 0-1 (2), 1-2 (3), 2-0 (4), 2-3 (1). The longest trail takes every
        // edge once: 3-2 (1), 2-0 (4), 0-1 (2), 1-2 (3) = 10
        let e = [(0u8, 1u8, 2u8), (1, 2, 3), (2, 0, 4), (2, 3, 1)];
        assert_eq!(longest_trail(&e, 4), 10);
        // a star: the trail can use only two of the three arms
        let e = [(0u8, 1u8, 5u8), (0, 2, 4), (0, 3, 3)];
        assert_eq!(longest_trail(&e, 4), 9);
    }

    #[test]
    fn the_last_round_gives_everyone_one_turn() {
        let map = Map::usa();
        for seed in 0..100 {
            let mut rng = Xoshiro::new(seed);
            let mut s = State::new(map, 3, &mut rng);
            let mut moves = Vec::new();
            let mut trigger_turn = None;
            let mut steps = 0;
            while !s.is_over() && steps < 5000 {
                if s.last_round_left >= 0 && trigger_turn.is_none() {
                    trigger_turn = Some(s.turns);
                }
                s.legal_actions(map, &mut moves);
                if moves.is_empty() {
                    s.pass();
                } else {
                    // prefer claims, so games end
                    let claims: Vec<u16> = moves.iter().cloned().filter(|&a| (a as usize) < map.n_claim).collect();
                    let pick = if !claims.is_empty() { &claims } else { &moves };
                    let a = pick[rng.below(pick.len() as u32) as usize] as usize;
                    s.step(map, a, &mut rng);
                }
                steps += 1;
            }
            if let Some(t) = trigger_turn {
                assert_eq!(
                    s.turns - t,
                    3,
                    "seed {seed}: after the trigger, one turn each for 3 players"
                );
            }
        }
    }
}
