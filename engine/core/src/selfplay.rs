//! Self-play for training: many games at once, each decision of the network being trained a record
//! to learn from.
//!
//! The networks are outside, as in `arena`. `prepare` plays every game on until a network has a
//! choice to make; the waiting rows go to the networks in one batch (`row_obs`, and `row_net` says
//! whose network answers each); `feed` takes their policy logits. The network being trained moves by
//! a sample of its policy, a frozen network by its first choice (`root_move`), a bot by its script;
//! bots' moves, forced moves and passes are played at once. When a game ends, its records get its
//! outcome (`take_records`).
//!
//! A record is a position of the network being trained and a target:
//!
//! - the scripted teacher's policy (`teacher`; `crate::teacher`, with its joint plan by
//!   `joint_paths`); a position the teacher has nothing to say about is not recorded;
//! - else the move played, as a one-hot policy with the probability the network gave it: the trainer
//!   imitates the better games' moves, or weighs each move by its reward (`Record::advantage`);
//! - a lesson from copies of the game, a record of its own (`Record::lesson` 1):
//!   - ticket playouts (`ticket_playouts`, with the teacher): a choice of tickets stops its game, and
//!     each of its options is played in `TICKET_COPIES` copies, the hidden cards dealt afresh and the
//!     rest of the game played by the network's first choice; the options whose tickets score most on
//!     average are the lesson;
//!   - the look-ahead (`lookahead`): at a share of the decisions and at every choice of tickets, the
//!     first choice and up to three other moves are played in `LOOK_WORLDS` sampled worlds each, every
//!     seat by its own player; the best other move, if it ended ahead, goes again with the first choice
//!     into fresh worlds (`LOOK_CHECK`, `look_check_keep` for a choice of tickets), and the move is a
//!     lesson only when the check proves it the better (`look::look_better`). With the move forecast
//!     (`look_forecast`), a move the network's forecast puts far above the first choice goes to the
//!     check at once, and the decision's own record keeps every copy's margin (`Record::look`).
//!
//! The opponents (`Opponents`): none (self-play: every seat the network being trained), a bot or one
//! frozen network at every other seat, or a mixed table, where every other seat gets its own opponent
//! for each game, drawn by weight, and at a table of three or more the card hoarder sits at one seat
//! in half of the games. A pool with opponents that does not look ahead plays duplicate deals: the
//! `players` games of a unit share a deal and everything random after it, the network being trained
//! in seat k of the unit's game k, and every move of the deal gets the mean of the unit's rewards.
//!
//! At most `playout_slots` copies are under way. A choice that finds no room waits in a queue of
//! `QUEUE` (first come, first served); past the queue it goes on without a lesson.

use crate::arena::a_seat;
use crate::bots::Bot;
use crate::look::{look_better, margin, options_by};
use crate::map::{Map, N_CARD_TYPES};
use crate::obs::Obs;
use crate::rng::{splitmix64, Xoshiro};
use crate::sample::sample_world;
use crate::state::{Phase, Rules, State, MAX_PLAYERS, NONE};
use crate::teacher::{teacher_policy, SPREAD};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Ticket playouts: the copies of each option of a choice of tickets.
pub const TICKET_COPIES: usize = 32;
/// Ticket playouts: drawing tickets while something is left to build is tried in copies, against the
/// teacher's best building move, at most this many times a seat and game - at its first turn with
/// something to build, then at about one turn in `EARLY_EVERY` while it holds `EARLY_TRAINS` trains.
const EARLY_DRAWS: u8 = 3;
const EARLY_EVERY: f64 = 10.0;
const EARLY_TRAINS: u8 = 20;
/// Ticket playouts: drawing tickets with nothing left to build is tried in copies at most this many
/// times a seat and game (a seat can face that choice for many turns in a row).
const DRAW_CHECKS: u8 = 3;
/// A choice with no room for its copies waits in a queue of at most this many, first come first
/// served, so that one needing many copies is not passed over by the many that need few.
const QUEUE: usize = 64;
/// The look-ahead: the share of the decisions looked at, at a table of two; at a table of n, this
/// times 2 / n, so that every table gets as many a game.
pub const LOOK_SHARE: f32 = 0.1;
/// The look-ahead: the sampled worlds of each move a look tries.
pub const LOOK_WORLDS: usize = 8;
/// The look-ahead: the fresh worlds of a check of any decision but a choice of tickets.
pub const LOOK_CHECK: usize = 24;
/// The move forecast: a move it puts this many points or more above the first choice is checked.
pub const FORECAST_POINTS: f32 = 10.0;
/// A record keeps the copies of at most this many moves (`Record::look`).
pub const LOOK_MOVES: usize = 5;
/// A proven move's part of what it and the first choice hold together: `PROVEN_BASE`, and
/// `PROVEN_PER_POINT` more a point of its mean lead in the check, at most `PROVEN_MAX`.
const PROVEN_BASE: f32 = 0.5;
const PROVEN_PER_POINT: f32 = 0.01;
const PROVEN_MAX: f32 = 0.8;
/// A mixed table of three or more: the card hoarder sits at one seat in this share of the games.
pub const HOARDER_SHARE: f32 = 0.5;
/// A copy that has not ended after this many moves is scored as it stands.
const PLAYOUT_MOVES: u16 = 2000;
/// The decision rule (`root_move`): the moves it chooses among, and a move's score besides its logit.
const CONSIDERED: usize = 16;
const SIGMA_Q: f32 = 2.5;

/// The streams of a pool's games, from its seed: self-play's deals, a duplicate unit's deals and the
/// randomness after them, a duplicate unit's opponents at a mixed table.
const SELF_DEALS: u64 = 0x2545_F491_4F6C_DD1D;
const DUPLICATE_DEALS: u64 = 0x9FB2_1C65_1E98_DF25;
const MIXED_SEATS: u64 = 0x7F4A_7C15_9E37_79B9;

/// An opponent at a seat of a mixed table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Opponent {
    Bot(Bot),
    /// the frozen network of this index in the caller's list
    Net(u8),
}

/// Who plays the seats of a pool's games that are not the trained network's.
#[derive(Clone, Debug, PartialEq)]
pub enum Opponents {
    /// self-play: the network being trained at every seat
    None,
    /// this bot at every other seat
    Bot(Bot),
    /// one frozen network at every other seat
    Net,
    /// every other seat its own opponent, drawn by these weights for each game (with duplicate deals,
    /// once a deal: the same opponents in the same order after the trained seat in each of its games);
    /// at a table of three or more, the card hoarder at one seat in `HOARDER_SHARE` of the games
    Mixed(Vec<(Opponent, f32)>),
}

#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub games: usize,
    /// the seats at a table, 2 to 5 (with `tables`, unused)
    pub players: usize,
    /// self-play: the table of each game drawn from these, each as likely (empty: `players`)
    pub tables: Vec<usize>,
    /// the rules of every game (`set_route_price` changes the price of the games started after it)
    pub rules: Rules,
    pub seed: u64,
    pub opponents: Opponents,
    /// every record's target is the scripted teacher's
    pub teacher: bool,
    /// the teacher plans one set of ways for all the open tickets
    pub joint_paths: bool,
    /// the teacher's choices of tickets are taught by copies of the game
    pub ticket_playouts: bool,
    /// the look-ahead teaches the moves it proves
    pub lookahead: bool,
    /// the fresh worlds of the look-ahead's check of a choice of tickets
    pub look_check_keep: usize,
    /// the move forecast sends moves to the look-ahead's check, and the records keep the copies
    pub look_forecast: bool,
    /// copies under way at most
    pub playout_slots: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            games: 256,
            players: 2,
            tables: Vec::new(),
            rules: Rules::OFFICIAL,
            seed: 1,
            opponents: Opponents::None,
            teacher: false,
            joint_paths: false,
            ticket_playouts: false,
            lookahead: false,
            look_check_keep: LOOK_CHECK,
            look_forecast: false,
            playout_slots: 0,
        }
    }
}

impl PoolConfig {
    /// Whether the settings go together; the reason when they do not.
    pub fn check(&self) -> Result<(), String> {
        let seats = 2..=MAX_PLAYERS;
        if self.games == 0 || !seats.contains(&self.players) || self.tables.iter().any(|n| !seats.contains(n)) {
            return Err(format!("some games, tables of 2 to {MAX_PLAYERS}"));
        }
        let plain = self.opponents == Opponents::None;
        if !self.tables.is_empty() && !plain {
            return Err("a pool with opponents plays at one table (`players`)".into());
        }
        if self.duplicate() && !self.games.is_multiple_of(self.players) {
            return Err("duplicate deals: the games are whole units of `players`".into());
        }
        if self.teacher && (!plain || self.lookahead) {
            return Err("the teacher's games are self-play, without the look-ahead".into());
        }
        if (self.joint_paths || self.ticket_playouts) && !self.teacher {
            return Err("the joint plan and ticket playouts are the teacher's".into());
        }
        if self.ticket_playouts && self.playout_slots == 0 {
            return Err("ticket playouts need room for copies (`playout_slots`)".into());
        }
        if self.lookahead {
            if matches!(self.opponents, Opponents::Bot(_) | Opponents::Net) {
                return Err("the look-ahead plays self-play or a mixed table".into());
            }
            let most = (4 * LOOK_WORLDS).max(2 * LOOK_CHECK).max(2 * self.look_check_keep);
            if self.look_check_keep < 2 || self.playout_slots < most {
                return Err(format!(
                    "the look-ahead: a check of 2 worlds or more, room for {most} copies"
                ));
            }
        }
        if self.look_forecast && !self.lookahead {
            return Err("the move forecast is the look-ahead's".into());
        }
        if let Opponents::Mixed(list) = &self.opponents {
            if list.iter().any(|(_, w)| !w.is_finite() || *w < 0.0) || !list.iter().any(|(_, w)| *w > 0.0) {
                return Err("a mixed table: weights of 0 or more, some above 0".into());
            }
            if self.players >= 3 && list.iter().any(|(o, _)| *o == Opponent::Bot(Bot::Hoard)) {
                return Err("a mixed table of 3 or more seats the card hoarder itself".into());
            }
        }
        Ok(())
    }

    /// A pool with opponents plays duplicate deals, unless it looks ahead.
    fn duplicate(&self) -> bool {
        self.opponents != Opponents::None && !self.lookahead
    }

    /// Frozen networks decide at the other seats (all of them, or a mixed table's networks' seats):
    /// they play their first choice and are not recorded.
    fn frozen(&self) -> bool {
        matches!(self.opponents, Opponents::Net | Opponents::Mixed(_))
    }

    /// A self-play game's table.
    fn table(&self, rng: &mut Xoshiro) -> usize {
        let mut x = rng.unit() as f32 * self.tables.len() as f32;
        for &n in &self.tables {
            if x < 1.0 {
                return n;
            }
            x -= 1.0;
        }
        self.tables.last().copied().unwrap_or(self.players)
    }
}

/// The stream of game `round` of `unit` (a slot, or a duplicate unit of slots) from `seed`.
fn deal_rng(seed: u64, unit: u64, round: u64) -> Xoshiro {
    let mut x = seed ^ unit.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ round.wrapping_mul(0xD1B5_4A32_D192_ED03);
    Xoshiro::new(splitmix64(&mut x))
}

/// Everything random in a duplicate unit's game after the deal: the same for each of its games.
fn play_rng(seed: u64, unit: u64, round: u64) -> Xoshiro {
    deal_rng(seed ^ 0xA076_1D64_78BD_642F, unit, round)
}

/// The draw of a self-play game's table.
fn variant_rng(seed: u64, unit: u64, round: u64) -> Xoshiro {
    deal_rng(seed ^ 0x5851_F42D_4C95_7F2D, unit, round)
}

/// Game `round` of slot `index`. With duplicate deals the slot's randomness after the deal (`rng`) is
/// its unit's; else `rng` goes on as it is.
fn new_game(map: &Map, cfg: &PoolConfig, index: u64, round: u64, rng: &mut Xoshiro) -> State {
    if cfg.duplicate() {
        let unit = index / cfg.players as u64;
        let seed = cfg.seed ^ DUPLICATE_DEALS;
        *rng = play_rng(seed, unit, round);
        State::with_rules(map, cfg.players, cfg.rules, &mut deal_rng(seed, unit, round))
    } else {
        let seed = cfg.seed ^ SELF_DEALS;
        let players = if cfg.tables.is_empty() {
            cfg.players
        } else {
            cfg.table(&mut variant_rng(seed, index, round))
        };
        State::with_rules(map, players, cfg.rules, &mut deal_rng(seed, index, round))
    }
}

/// The opponents of game `round` of slot `index` at a mixed table: None at the trained seat, and
/// everywhere in a pool that is not mixed. With duplicate deals they are the unit's draws (seat a + k
/// gets the k-th, a the trained seat), else the slot's.
fn seat_opponents(cfg: &PoolConfig, index: u64, round: u64, rng: &mut Xoshiro) -> [Option<Opponent>; MAX_PLAYERS] {
    let mut out = [None; MAX_PLAYERS];
    let Opponents::Mixed(list) = &cfg.opponents else {
        return out;
    };
    let n = cfg.players;
    let a = a_seat(index as usize, n) as usize;
    let mut unit = cfg
        .duplicate()
        .then(|| deal_rng(cfg.seed ^ MIXED_SEATS, index / n as u64, round));
    let draws = match unit.as_mut() {
        Some(u) => u,
        None => rng,
    };
    let total: f32 = list.iter().map(|(_, w)| w.max(0.0)).sum();
    // the card hoarder's place after the trained seat, 1 to n - 1, drawn first
    let hoarder = if n >= 3 {
        let here = draws.unit() < HOARDER_SHARE as f64;
        let place = (1 + (draws.unit() * (n - 1) as f64) as usize).min(n - 1);
        here.then_some(place)
    } else {
        None
    };
    for k in 1..n {
        if hoarder == Some(k) {
            out[(a + k) % n] = Some(Opponent::Bot(Bot::Hoard));
            continue;
        }
        let mut x = draws.unit() as f32 * total;
        let mut pick = list.last().expect("opponents").0;
        for &(o, w) in list {
            let w = w.max(0.0);
            if x < w {
                pick = o;
                break;
            }
            x -= w;
        }
        out[(a + k) % n] = Some(pick);
    }
    out
}

/// A position of the network being trained, for training. Seats are relative to the seat that
/// decided (0).
#[derive(Clone, Debug)]
pub struct Record {
    pub obs: Obs,
    /// the target, over every action id
    pub policy: Vec<f32>,
    pub n_players: u8,
    /// the outcome per seat: +1 first place .. -1 last
    pub value: [f32; MAX_PLAYERS],
    /// what the network's forecast said for the deciding seat when it decided (NaN: none given)
    pub root_forecast: f32,
    /// the network's probability of the move played, among the legal moves
    pub root_prob: f32,
    /// the final score per seat / 100
    pub score: [f32; MAX_PLAYERS],
    /// the final owner of every route (-1: nobody)
    pub route_owner: Vec<i8>,
    /// the tickets each seat held at the decision, a bit per ticket
    pub tickets: [u64; MAX_PLAYERS],
    /// the deciding seat's tickets at the end: 1 completed, -1 failed, 0 not held
    pub ticket_outcome: Vec<i8>,
    /// 1 for the deciding seat's tickets kept at the opening
    pub ticket_start: Vec<u8>,
    /// every seat's longest trail at the end / 30
    pub trail: [f32; MAX_PLAYERS],
    /// 1 for every seat with the longest trail at the end (all 0 while nobody has a trail)
    pub trail_best: [u8; MAX_PLAYERS],
    /// every seat's hand at the decision
    pub hands: [[u8; N_CARD_TYPES]; MAX_PLAYERS],
    /// per seat and ticket, the trains the seat still needed to join the ticket's cities
    /// (`State::ticket_costs`), MAX_PLAYERS x tickets
    pub path_cost: Vec<u8>,
    /// the game: slot i's k-th game is i + k x games
    pub game: u64,
    /// the decision's move number in its game
    pub moves: u16,
    /// 0: a move of the game; 1: a lesson from copies
    pub lesson: u8,
    /// the move's reward, in hundreds of points: its seat's margin over the strongest other seat,
    /// less the table's mean margin (zero summed over a game's seats); with duplicate deals, the mean
    /// of the trained seat's over the unit's games
    pub advantage: f32,
    /// with the move forecast, on a decision that was looked at or checked: the moves tried and,
    /// move by move, each copy's margin - a look's worlds, then its check's (NaN where a move was not
    /// tried)
    pub look: Vec<u16>,
    pub look_table: Vec<f32>,
    /// the deciding seat
    pub actor: u8,
}

/// A decision waiting for its network: its legal moves, whether its move is a sample of the policy,
/// the seed of that sample's noise, and once fed, the logits and the forecast.
struct Ask {
    legal: Vec<u16>,
    noise: bool,
    seed: u64,
    logits: Option<Vec<f32>>,
    forecast: f32,
}

/// A decision made: the move, its probability, the forecast.
struct Decision {
    action: u16,
    prob: f32,
    forecast: f32,
}

/// What a lesson from copies is about.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// ticket playouts: the opening's keep, a later keep, drawing with nothing left to build,
    /// drawing while something is
    Opening = 0,
    Keep = 1,
    Draw = 2,
    Early = 3,
    /// the look-ahead
    Look = 4,
}

/// A decision waiting for its copies.
struct Pending {
    /// the lesson's record, its target made when the copies are back
    record: Record,
    legal: Vec<u16>,
    /// the moves tried
    options: Vec<u16>,
    /// per option, what its copies ended with, summed, and the copies back
    sum: Vec<f32>,
    count: Vec<u32>,
    /// copies still under way or not started
    left: usize,
    /// drawing tickets with nothing left to build: what the tickets score as they are
    base: Option<f32>,
    /// drawing tickets while something is left to build: the teacher's target, the lesson when
    /// building ends no worse
    building: Option<Vec<f32>>,
    /// the move played once the lesson is made
    action: u16,
    kind: Kind,
    /// the look-ahead: the network's logits at the decision, and per option and world its copy's
    /// margin (a check's, and a look's with the move forecast; NaN until the copy is back)
    logits: Vec<f32>,
    table: Vec<f32>,
    /// the copies of each option
    worlds: usize,
    /// the look-ahead: 0 a look, 1 its check
    stage: u8,
    /// a check the move forecast asked for
    flagged: bool,
    /// the decision's own record (`Slot::records`), which keeps the copies with the move forecast
    game_rec: Option<usize>,
}

/// A copy of a game at a decision with one of its options played, played to the end.
struct Playout {
    /// its game's opponents
    opp: [Option<Opponent>; MAX_PLAYERS],
    game: State,
    rng: Xoshiro,
    asked: Option<Ask>,
    leaf: Obs,
    /// the slot whose decision it tries, the option, and the world
    parent: usize,
    option: usize,
    world: usize,
    decider: u8,
    passes: u32,
    row: usize,
    /// at its end: the decider's ticket points (ticket playouts) or margin (the look-ahead)
    end: Option<f32>,
}

struct Slot {
    opp: [Option<Opponent>; MAX_PLAYERS],
    /// duplicate deals: the finished games whose unit has not finished that deal yet - the round,
    /// the trained seat's reward, its records
    dup_wait: Vec<(u64, f32, Vec<Record>)>,
    game: State,
    rng: Xoshiro,
    asked: Option<Ask>,
    records: Vec<Record>,
    done: Vec<Record>,
    game_id: u64,
    leaf: Obs,
    row: Option<usize>,
    passes: u32,
    pending: Option<Pending>,
    /// the copies started in the last `advance`, for the pool to take
    spawn: Vec<Playout>,
    /// the decision joined the queue in the last `advance`
    fresh_queue: bool,
    /// ticket playouts: per seat, the draws of tickets tried in copies in the game under way, with
    /// nothing left to build and while something is
    draw_checks: [u8; MAX_PLAYERS],
    early_checks: [u8; MAX_PLAYERS],
    /// the move forecast for the decision under way (`feed_forecast`)
    forecast: Option<Vec<f32>>,
    finished_games: u64,
    decisions: u64,
    evals: u64,
    moves_in_finished: u64,
    ticket_lessons: u64,
    ticket_skipped: u64,
    early_lessons: u64,
    lessons_by_kind: [u64; 4],
    look_tried: u64,
    look_checked: u64,
    look_proven: u64,
    forecast_flagged: u64,
    forecast_proven: u64,
    forecast_keeps: u64,
    keep_checked: u64,
    keep_proven: u64,
}

/// A row of this step: a game's decision, or a copy's.
#[derive(Clone, Copy)]
enum Row {
    Slot(usize),
    Playout(usize),
}

pub struct Pool {
    cfg: PoolConfig,
    map: &'static Map,
    slots: Vec<Slot>,
    /// the copies under way
    playouts: Vec<Playout>,
    rows: Vec<Row>,
    playouts_done: u64,
    playout_evals: u64,
    /// the slots whose decision waits for room for its copies
    queue: VecDeque<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoolStats {
    pub games_finished: u64,
    /// the networks' decisions, the frozen ones' too
    pub decisions: u64,
    /// the rows of the games' decisions answered
    pub evals: u64,
    pub moves_per_game: f64,
    /// copies played to the end, and their rows answered
    pub playouts: u64,
    pub playout_evals: u64,
    /// lessons from copies; decisions that found no room for their copies
    pub ticket_lessons: u64,
    pub ticket_skipped: u64,
    /// of the lessons, those on drawing tickets while something is left to build
    pub early_lessons: u64,
    /// ticket playouts' lessons by kind: the opening's keep, a later keep, drawing with nothing left
    /// to build, drawing while something is
    pub lessons_by_kind: [u64; 4],
    /// the look-ahead: decisions looked at, their checks, the moves proven (each a lesson)
    pub look_tried: u64,
    pub look_checked: u64,
    pub look_proven: u64,
    /// the checks the move forecast asked for, and of them those that proved their move
    pub forecast_flagged: u64,
    pub forecast_proven: u64,
    /// choices of tickets whose look took the forecast's move as one more option
    pub forecast_keeps: u64,
    /// the checks of a choice of tickets, and those that proved their set
    pub keep_checked: u64,
    pub keep_proven: u64,
}

impl Pool {
    pub fn new(map: &'static Map, cfg: PoolConfig) -> Pool {
        if let Err(e) = cfg.check() {
            panic!("a self-play pool: {e}");
        }
        let mut seeder = Xoshiro::new(cfg.seed);
        let mut slots = Vec::with_capacity(cfg.games);
        for i in 0..cfg.games {
            let mut rng = seeder.fork();
            let game = new_game(map, &cfg, i as u64, 0, &mut rng);
            let opp = seat_opponents(&cfg, i as u64, 0, &mut rng);
            slots.push(Slot {
                opp,
                dup_wait: Vec::new(),
                game,
                rng,
                asked: None,
                records: Vec::new(),
                done: Vec::new(),
                game_id: i as u64,
                leaf: Obs::new(map),
                row: None,
                passes: 0,
                pending: None,
                spawn: Vec::new(),
                fresh_queue: false,
                draw_checks: [0; MAX_PLAYERS],
                early_checks: [0; MAX_PLAYERS],
                forecast: None,
                finished_games: 0,
                decisions: 0,
                evals: 0,
                moves_in_finished: 0,
                ticket_lessons: 0,
                ticket_skipped: 0,
                early_lessons: 0,
                lessons_by_kind: [0; 4],
                look_tried: 0,
                look_checked: 0,
                look_proven: 0,
                forecast_flagged: 0,
                forecast_proven: 0,
                forecast_keeps: 0,
                keep_checked: 0,
                keep_proven: 0,
            });
        }
        Pool {
            cfg,
            map,
            slots,
            playouts: Vec::new(),
            rows: Vec::new(),
            playouts_done: 0,
            playout_evals: 0,
            queue: VecDeque::new(),
        }
    }

    pub fn config(&self) -> &PoolConfig {
        &self.cfg
    }

    pub fn games(&self) -> usize {
        self.slots.len()
    }

    /// Play every game on until a network must choose or the game waits for its copies, and every
    /// copy until its network must choose; returns the rows that wait for `feed`: the games' decisions
    /// in game order, then the copies'.
    pub fn prepare(&mut self) -> usize {
        let map = self.map;
        let Pool {
            cfg,
            slots,
            playouts,
            rows,
            playouts_done,
            queue,
            ..
        } = self;
        let cfg = &*cfg;
        let n_slots = slots.len() as u64;
        // the queue first, in its order: a decision starts its copies once there is room for all of them
        let mut free = cfg.playout_slots.saturating_sub(playouts.len());
        while let Some(&i) = queue.front() {
            let slot = &mut slots[i];
            let wait = slot.pending.as_mut().expect("a queued decision");
            if wait.left > free {
                break;
            }
            free -= wait.left;
            for mut p in copies(map, &slot.game, &wait.options, wait.worlds, &mut slot.rng, &slot.opp) {
                p.parent = i;
                playouts.push(p);
            }
            queue.pop_front();
        }
        // nobody passes the queue: while it waits, a new decision joins it, or past it has no lesson
        let room = AtomicUsize::new(if queue.is_empty() { free } else { 0 });
        let line = AtomicUsize::new(QUEUE.saturating_sub(queue.len()));
        slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| advance(map, cfg, slot, n_slots, i as u64, &room, &line));
        for (i, slot) in slots.iter_mut().enumerate() {
            for mut p in slot.spawn.drain(..) {
                p.parent = i;
                playouts.push(p);
            }
            if std::mem::take(&mut slot.fresh_queue) {
                queue.push_back(i);
            }
        }
        if !playouts.is_empty() {
            playouts.par_iter_mut().for_each(|p| advance_playout(map, cfg, p));
            // the copies that ended report to their decisions
            let mut k = 0;
            while k < playouts.len() {
                match playouts[k].end {
                    Some(v) => {
                        let p = playouts.swap_remove(k);
                        let wait = slots[p.parent]
                            .pending
                            .as_mut()
                            .expect("a decision waits for its copies");
                        wait.sum[p.option] += v;
                        wait.count[p.option] += 1;
                        if !wait.table.is_empty() {
                            let worlds = wait.table.len() / wait.options.len();
                            wait.table[p.option * worlds + p.world] = v;
                        }
                        wait.left -= 1;
                        *playouts_done += 1;
                    }
                    None => k += 1,
                }
            }
        }
        rows.clear();
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.row = slot.asked.is_some().then(|| {
                rows.push(Row::Slot(i));
                rows.len() - 1
            });
        }
        for (j, p) in playouts.iter_mut().enumerate() {
            p.row = rows.len();
            rows.push(Row::Playout(j));
        }
        slots.par_iter_mut().filter(|s| s.asked.is_some()).for_each(|s| {
            s.leaf.fill(map, &s.game, s.game.current as usize);
        });
        playouts
            .par_iter_mut()
            .for_each(|p| p.leaf.fill(map, &p.game, p.game.current as usize));
        rows.len()
    }

    /// The rows that wait for `feed` (after `prepare`).
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Row `i`'s position, as its player to act sees it.
    pub fn row_obs(&self, i: usize) -> &Obs {
        match self.rows[i] {
            Row::Slot(k) => &self.slots[k].leaf,
            Row::Playout(k) => &self.playouts[k].leaf,
        }
    }

    /// The seat to act at row `i`.
    pub fn decider_seat(&self, i: usize) -> u8 {
        match self.rows.get(i) {
            Some(&Row::Slot(k)) => self.slots[k].game.current,
            Some(&Row::Playout(k)) => self.playouts[k].game.current,
            None => NONE,
        }
    }

    /// The network that answers row `i`: 0 the one being trained, k + 1 frozen network k of a mixed
    /// table, 1 the frozen network of a pool against one.
    pub fn row_net(&self, i: usize) -> u8 {
        let (game, opp, slot) = match self.rows[i] {
            Row::Slot(k) => (&self.slots[k].game, &self.slots[k].opp, Some(k)),
            Row::Playout(k) => (&self.playouts[k].game, &self.playouts[k].opp, None),
        };
        match opp[game.current as usize] {
            Some(Opponent::Net(j)) => j + 1,
            _ => match (slot, &self.cfg.opponents) {
                (Some(k), Opponents::Net) => (game.current != a_seat(k, self.cfg.players)) as u8,
                _ => 0,
            },
        }
    }

    /// With the move forecast: the rows of the trained network's decisions in the games (after
    /// `prepare`), whose forecast of every move `feed_forecast` takes before `feed`.
    pub fn forecast_rows(&self) -> Vec<usize> {
        if !self.cfg.look_forecast {
            return Vec::new();
        }
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match *r {
                Row::Slot(k) => {
                    let s = &self.slots[k];
                    s.opp[s.game.current as usize].is_none().then_some(i)
                }
                Row::Playout(_) => None,
            })
            .collect()
    }

    /// The move forecast of `rows` (rows of `forecast_rows`, in any order): rows x actions, the
    /// margin each move would end the game with, in hundreds of points.
    pub fn feed_forecast(&mut self, rows: &[usize], moves: &[f32]) {
        let na = self.map.n_actions;
        assert_eq!(moves.len(), rows.len() * na, "the move forecast: rows x actions");
        for (j, &i) in rows.iter().enumerate() {
            if let Some(&Row::Slot(k)) = self.rows.get(i) {
                self.slots[k].forecast = Some(moves[j * na..(j + 1) * na].to_vec());
            }
        }
    }

    /// The networks' answers for this step's rows: policy logits, rows x actions, and optionally the
    /// forecast for each row's player to act, which the records of the games' decisions keep
    /// (`Record::root_forecast`).
    pub fn feed(&mut self, logits: &[f32], forecast: Option<&[f32]>) {
        let na = self.map.n_actions;
        let n = self.rows.len();
        assert_eq!(logits.len(), n * na, "logits: rows x actions");
        assert!(forecast.is_none_or(|f| f.len() == n), "the forecast: one a row");
        self.slots.par_iter_mut().for_each(|slot| {
            let Some(i) = slot.row else { return };
            let ask = slot.asked.as_mut().expect("a decision waits");
            ask.logits = Some(logits[i * na..(i + 1) * na].to_vec());
            ask.forecast = forecast.map_or(f32::NAN, |f| f[i]);
            slot.evals += 1;
        });
        self.playouts.par_iter_mut().for_each(|p| {
            let ask = p.asked.as_mut().expect("a copy waits");
            ask.logits = Some(logits[p.row * na..(p.row + 1) * na].to_vec());
        });
        self.playout_evals += self.playouts.len() as u64;
    }

    /// The records of the games that ended since the last call; with duplicate deals, of the deals
    /// every game of the unit has finished.
    pub fn take_records(&mut self) -> Vec<Record> {
        let mut out = Vec::new();
        for slot in self.slots.iter_mut() {
            out.append(&mut slot.done);
        }
        if self.cfg.duplicate() {
            let n = self.cfg.players;
            for unit in self.slots.chunks_mut(n) {
                while unit.iter().all(|s| !s.dup_wait.is_empty()) {
                    let adv = unit.iter().map(|s| s.dup_wait[0].1).sum::<f32>() / n as f32;
                    for s in unit.iter_mut() {
                        let (_, _, recs) = s.dup_wait.remove(0);
                        out.extend(recs.into_iter().map(|mut r| {
                            r.advantage = adv;
                            r
                        }));
                    }
                }
            }
        }
        out
    }

    /// The route price, 0 to 1, of the games started from now on; the games under way keep theirs.
    pub fn set_route_price(&mut self, price: f32) {
        self.cfg.rules.route_price = (price.clamp(0.0, 1.0) * 100.0).round() as u8;
    }

    pub fn stats(&self) -> PoolStats {
        let mut st = PoolStats::default();
        let mut moves = 0;
        for s in &self.slots {
            st.games_finished += s.finished_games;
            st.decisions += s.decisions;
            st.evals += s.evals;
            moves += s.moves_in_finished;
            st.ticket_lessons += s.ticket_lessons;
            st.ticket_skipped += s.ticket_skipped;
            st.early_lessons += s.early_lessons;
            for k in 0..4 {
                st.lessons_by_kind[k] += s.lessons_by_kind[k];
            }
            st.look_tried += s.look_tried;
            st.look_checked += s.look_checked;
            st.look_proven += s.look_proven;
            st.forecast_flagged += s.forecast_flagged;
            st.forecast_proven += s.forecast_proven;
            st.forecast_keeps += s.forecast_keeps;
            st.keep_checked += s.keep_checked;
            st.keep_proven += s.keep_proven;
        }
        st.moves_per_game = moves as f64 / st.games_finished.max(1) as f64;
        st.playouts = self.playouts_done;
        st.playout_evals = self.playout_evals;
        st
    }
}

/// Play a game's moves until a network must choose, or its decision waits for its copies.
fn advance(
    map: &Map,
    cfg: &PoolConfig,
    slot: &mut Slot,
    n_slots: u64,
    index: u64,
    room: &AtomicUsize,
    line: &AtomicUsize,
) {
    let mut moves = Vec::with_capacity(256);
    let me_seat = a_seat(index as usize, cfg.players);
    loop {
        // once all the copies of its decision are back, the game has its lesson and goes on
        if let Some(wait) = slot.pending.as_ref() {
            if wait.left > 0 {
                return;
            }
            let wait = slot.pending.take().expect("a decision waits");
            let action = wait.action;
            if wait.kind == Kind::Look {
                if let Some(i) = wait.game_rec {
                    forecast_record(&mut slot.records[i], &wait.options, &wait.table, wait.worlds);
                }
                if wait.stage == 0 {
                    let mean = |k: usize| wait.sum[k] / wait.count[k].max(1) as f32;
                    let best = (1..wait.options.len())
                        .max_by(|&a, &b| mean(a).total_cmp(&mean(b)))
                        .filter(|&b| mean(b) > mean(0));
                    if let Some(best) = best {
                        if look_check_start(map, cfg, slot, wait, best, room, line) {
                            return; // the check's copies are on their way
                        }
                    }
                } else if look_better(&wait.table, wait.worlds) {
                    let mut r = wait.record;
                    let lead = look_lead(&wait.table, wait.worlds);
                    r.policy = proven_lesson(map, &wait.legal, &wait.logits, wait.options[0], wait.options[1], lead);
                    slot.records.push(r);
                    slot.ticket_lessons += 1;
                    slot.look_proven += 1;
                    slot.forecast_proven += wait.flagged as u64;
                    slot.keep_proven += (slot.game.phase == Phase::Keep) as u64;
                }
            } else {
                let mut r = wait.record;
                r.policy = match &wait.building {
                    Some(build) => {
                        slot.early_lessons += 1;
                        early_lesson(map, &wait.legal, &wait.sum, &wait.count, build)
                    }
                    None => lesson_policy(map, &wait.legal, &wait.options, &wait.sum, &wait.count, wait.base),
                };
                slot.records.push(r);
                slot.ticket_lessons += 1;
                slot.lessons_by_kind[wait.kind as usize] += 1;
            }
            slot.game.step(map, action as usize, &mut slot.rng);
            continue;
        }
        if let Some(ask) = slot.asked.take() {
            let Some(lg) = ask.logits else {
                slot.asked = Some(ask); // the network has not answered yet
                return;
            };
            let action = root_move(&ask.legal, &lg, ask.noise, ask.seed);
            let d = Decision {
                action,
                prob: root_prob(&ask.legal, &lg, action),
                forecast: ask.forecast,
            };
            let kept = !(cfg.frozen() && slot.game.current != me_seat);
            if cfg.ticket_playouts {
                if ticket_playouts(map, cfg, slot, &d, ask.legal, kept, room, line) {
                    return; // its copies are on their way
                }
                slot.game.step(map, action as usize, &mut slot.rng);
                continue;
            }
            let target = if cfg.teacher {
                kept.then(|| teacher_policy(map, &slot.game, &ask.legal, cfg.joint_paths))
                    .flatten()
            } else {
                Some(one_hot(map, action))
            };
            let keep = slot.game.phase == Phase::Keep;
            // the move forecast: a move it puts far enough above the first choice goes to the check at once,
            // but a choice of tickets is looked at whatever it says, the move it puts there one more option
            let hint = match (slot.forecast.take(), kept && cfg.look_forecast) {
                (Some(f), true) => forecast_flag(map, &ask.legal, &lg, &f, FORECAST_POINTS),
                _ => None,
            };
            let flag = if keep { None } else { hint };
            let share = (LOOK_SHARE * 2.0 / slot.game.n() as f32) as f64;
            let look = if flag.is_none() && cfg.lookahead && kept && (keep || slot.rng.unit() < share) {
                let mut options = options_by(map, &ask.legal, &lg, true);
                if let (true, Some((_, best))) = (keep, hint) {
                    if !options.contains(&best) && options.len() < LOOK_MOVES {
                        options.push(best);
                        slot.forecast_keeps += 1;
                    }
                }
                Some(options)
            } else {
                None
            };
            let mut game_rec = None;
            if let (true, Some(policy)) = (kept, target) {
                let r = record_of(map, slot, &d, policy, 0);
                slot.records.push(r);
                game_rec = cfg.look_forecast.then(|| slot.records.len() - 1);
            }
            if let Some(pair) = flag {
                if forecast_check_start(map, slot, &d, pair, ask.legal, lg, game_rec, room, line) {
                    return; // its copies are on their way
                }
            } else if let Some(options) = look {
                if look_start(map, slot, &d, options, ask.legal, lg, game_rec, room, line) {
                    return; // its copies are on their way
                }
            }
            slot.game.step(map, action as usize, &mut slot.rng);
            continue;
        }
        // the next decision of this game
        if slot.game.is_over() {
            finish(map, slot, cfg.duplicate().then_some(me_seat as usize));
            slot.finished_games += 1;
            slot.game_id = index + n_slots * slot.finished_games;
            slot.game = new_game(map, cfg, index, slot.finished_games, &mut slot.rng);
            slot.opp = seat_opponents(cfg, index, slot.finished_games, &mut slot.rng);
            slot.passes = 0;
            slot.draw_checks = [0; MAX_PLAYERS];
            slot.early_checks = [0; MAX_PLAYERS];
            continue;
        }
        slot.game.legal_actions(map, &mut moves);
        if moves.is_empty() {
            slot.game.pass();
            slot.passes += 1;
            if slot.passes >= slot.game.n() as u32 {
                // nobody can move - no card to draw, no ticket, no route anyone can pay, and no claim
                // left to start the last round (few trains make it happen): the game ends as it stands
                slot.game.phase = Phase::Over;
            }
            continue;
        }
        slot.passes = 0;
        if moves.len() == 1 {
            slot.game.step(map, moves[0] as usize, &mut slot.rng);
            continue;
        }
        let seat = slot.game.current;
        let bot = match (slot.opp[seat as usize], &cfg.opponents) {
            (Some(Opponent::Bot(b)), _) => Some(b),
            (_, &Opponents::Bot(b)) if seat != me_seat => Some(b),
            _ => None,
        };
        if let Some(bot) = bot {
            let a = bot.choose(map, &slot.game, &moves, &mut slot.rng);
            slot.game.step(map, a as usize, &mut slot.rng);
            continue;
        }
        slot.decisions += 1;
        slot.asked = Some(Ask {
            legal: moves,
            noise: !(cfg.frozen() && seat != me_seat),
            seed: slot.rng.next_u64(),
            logits: None,
            forecast: f32::NAN,
        });
        return;
    }
}

/// The move of a decision made without search: the root of Gumbel AlphaZero (Danihelka et al. 2022)
/// with no simulation, the rule the published network was trained with. With `noise` the logits get
/// Gumbel noise from `seed` and the move is a sample of the policy (the Gumbel-max trick); without,
/// the move is the first choice. Of the `CONSIDERED` best moves by noisy logit, the move is the best
/// by noisy logit plus the root's sigma(q), which with no simulation is `SIGMA_Q` for every move: the
/// same move, but for two moves the addition's rounding makes equal, where the later one in that
/// order is played.
fn root_move(legal: &[u16], logits: &[f32], noise: bool, seed: u64) -> u16 {
    let mut g = vec![0.0f32; logits.len()];
    if noise {
        let mut rng = Xoshiro::new(seed);
        for &a in legal {
            let u = rng.unit().clamp(1e-12, 1.0 - 1e-12);
            g[a as usize] = -(-u.ln()).ln() as f32;
        }
    }
    let noisy = |a: u16| g[a as usize] + logits[a as usize];
    let mut order = legal.to_vec();
    order.sort_by(|&a, &b| noisy(b).partial_cmp(&noisy(a)).unwrap_or(std::cmp::Ordering::Equal));
    order.truncate(CONSIDERED.min(order.len()).max(1));
    *order
        .iter()
        .max_by(|&&a, &&b| {
            (noisy(a) + SIGMA_Q)
                .partial_cmp(&(noisy(b) + SIGMA_Q))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("a legal move")
}

/// The policy's probability of `a`: the softmax of `logits` over `legal`.
fn root_prob(legal: &[u16], logits: &[f32], a: u16) -> f32 {
    let top = legal
        .iter()
        .map(|&b| logits[b as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = legal.iter().map(|&b| (logits[b as usize] - top).exp()).sum();
    (logits[a as usize] - top).exp() / sum
}

/// The move played as a target, all on it.
fn one_hot(map: &Map, action: u16) -> Vec<f32> {
    let mut p = vec![0.0f32; map.n_actions];
    p[action as usize] = 1.0;
    p
}

/// The game is over: give its records their outcome.
fn finish(map: &Map, slot: &mut Slot, duplicate: Option<usize>) {
    let s = &slot.game;
    let n = s.n();
    let f = s.finalize(map);
    let u = f.utility();
    let margin: Vec<f32> = (0..n)
        .map(|q| {
            (f.total[q]
                - (0..n)
                    .filter(|&o| o != q)
                    .map(|o| f.total[o])
                    .max()
                    .unwrap_or(f.total[q])) as f32
        })
        .collect();
    let mean = margin.iter().sum::<f32>() / n as f32;
    let centred: Vec<f32> = margin.iter().map(|m| (m - mean) / 100.0).collect();
    let comps: Vec<_> = (0..n).map(|q| s.components(map, q)).collect();
    let trails: Vec<i16> = (0..n).map(|q| s.longest_trail(map, q)).collect();
    let best_trail = trails.iter().copied().max().unwrap_or(0);
    slot.moves_in_finished += s.moves as u64;
    let mut finished = Vec::with_capacity(slot.records.len());
    for mut r in slot.records.drain(..) {
        let a = r.actor as usize;
        for k in 0..n {
            let q = (a + k) % n;
            r.value[k] = u[q];
            r.score[k] = f.total[q] as f32 / 100.0;
            r.trail[k] = trails[q] as f32 / 30.0;
            r.trail_best[k] = (best_trail > 0 && trails[q] == best_trail) as u8;
        }
        r.route_owner = (0..map.n_routes())
            .map(|i| {
                let o = s.route_owner[i];
                if o == NONE {
                    -1
                } else {
                    ((o as usize + n - a) % n) as i8
                }
            })
            .collect();
        r.ticket_outcome = (0..map.n_tickets())
            .map(|t| {
                if s.tickets_held[a] & (1u64 << t) == 0 {
                    0
                } else {
                    let tk = &map.tickets[t];
                    if comps[a][tk.a as usize] == comps[a][tk.b as usize] {
                        1
                    } else {
                        -1
                    }
                }
            })
            .collect();
        r.ticket_start = (0..map.n_tickets())
            .map(|t| ((s.opening_tickets[a] >> t) & 1) as u8)
            .collect();
        r.advantage = centred[a];
        finished.push(r);
    }
    match duplicate {
        Some(seat) => slot.dup_wait.push((slot.finished_games, centred[seat], finished)),
        None => slot.done.append(&mut finished),
    }
}

/// The record of the game's position now, its target `policy` (its outcome comes at the game's end).
fn record_of(map: &Map, slot: &Slot, d: &Decision, policy: Vec<f32>, lesson: u8) -> Record {
    let me = slot.game.current as usize;
    let n = slot.game.n();
    let nt = map.n_tickets();
    let mut tickets = [0u64; MAX_PLAYERS];
    let mut hands = [[0u8; N_CARD_TYPES]; MAX_PLAYERS];
    let mut path_cost = vec![NONE; MAX_PLAYERS * nt];
    for k in 0..n {
        let q = (me + k) % n;
        tickets[k] = slot.game.tickets_held[q];
        hands[k] = slot.game.hand[q];
        path_cost[k * nt..(k + 1) * nt].copy_from_slice(&slot.game.ticket_costs(map, q)[..nt]);
    }
    Record {
        obs: Obs::of(map, &slot.game, me),
        policy,
        n_players: n as u8,
        value: [0.0; MAX_PLAYERS],
        root_forecast: d.forecast,
        root_prob: d.prob,
        score: [0.0; MAX_PLAYERS],
        route_owner: Vec::new(),
        tickets,
        ticket_outcome: Vec::new(),
        ticket_start: Vec::new(),
        trail: [0.0; MAX_PLAYERS],
        trail_best: [0; MAX_PLAYERS],
        hands,
        path_cost,
        game: slot.game_id,
        moves: slot.game.moves,
        lesson,
        advantage: 0.0,
        look: Vec::new(),
        look_table: Vec::new(),
        actor: me as u8,
    }
}

/// Take `n` places for copies out of `room`, if that many are left.
fn reserve(room: &AtomicUsize, n: usize) -> bool {
    room.try_update(Ordering::Relaxed, Ordering::Relaxed, |r| r.checked_sub(n))
        .is_ok()
}

/// Ticket playouts at the decision `d` among `legal` (`kept`: the position may be recorded). A
/// choice of tickets - a keep, drawing tickets with nothing left to build, and now and then drawing
/// them while something is - gets its copies when there is room for them, and the game waits
/// (true). Any other position gets the teacher's target.
#[allow(clippy::too_many_arguments)]
fn ticket_playouts(
    map: &Map,
    cfg: &PoolConfig,
    slot: &mut Slot,
    d: &Decision,
    legal: Vec<u16>,
    kept: bool,
    room: &AtomicUsize,
    line: &AtomicUsize,
) -> bool {
    let me = slot.game.current as usize;
    let draw = map.tickets_action() as u16;
    let (options, base, building) = if slot.game.phase == Phase::Keep {
        (legal.clone(), None, None)
    } else {
        let others: Vec<u16> = legal.iter().cloned().filter(|&a| a != draw).collect();
        let build = teacher_policy(map, &slot.game, &others, cfg.joint_paths);
        // where something is left to build, the teacher's target over every legal move, drawing among them
        let lesson = build
            .as_ref()
            .map(|policy| teacher_policy(map, &slot.game, &legal, cfg.joint_paths).unwrap_or_else(|| policy.clone()));
        let early = build.is_some()
            && others.len() < legal.len()
            && slot.early_checks[me] < EARLY_DRAWS
            && slot.game.trains[me] >= EARLY_TRAINS
            && (slot.early_checks[me] == 0 || slot.rng.unit() < 1.0 / EARLY_EVERY);
        if early && kept {
            // drawing now against the teacher's best building move, both played out
            let policy = build.as_ref().expect("something to build");
            let best = others
                .iter()
                .cloned()
                .max_by(|&a, &b| policy[a as usize].total_cmp(&policy[b as usize]).then(b.cmp(&a)))
                .expect("a move");
            (vec![draw, best], None, lesson)
        } else if build.is_some() || others.len() == legal.len() || slot.draw_checks[me] >= DRAW_CHECKS {
            if let (true, Some(policy)) = (kept, lesson) {
                let r = record_of(map, slot, d, policy, 0);
                slot.records.push(r);
            }
            return false;
        } else {
            (vec![draw], Some(slot.game.tickets_score(map)[me] as f32), None)
        }
    };
    let need = options.len() * TICKET_COPIES;
    // room for all its copies now, else a place in the queue, else no lesson
    let now = kept && need <= cfg.playout_slots && reserve(room, need);
    let queued = !now && kept && need <= cfg.playout_slots && reserve(line, 1);
    if !now && !queued {
        slot.ticket_skipped += 1;
        if let (true, Some(policy)) = (kept, building) {
            // no room for the copies of an early draw: the teacher's target, as at any building move
            let r = record_of(map, slot, d, policy, 0);
            slot.records.push(r);
        }
        return false;
    }
    if base.is_some() {
        slot.draw_checks[me] += 1;
    }
    if building.is_some() {
        slot.early_checks[me] += 1;
    }
    let kind = match (slot.game.phase == Phase::Keep, slot.game.in_opening(), base.is_some()) {
        (true, true, _) => Kind::Opening,
        (true, false, _) => Kind::Keep,
        (false, _, true) => Kind::Draw,
        (false, _, false) => Kind::Early,
    };
    let record = record_of(map, slot, d, Vec::new(), 1);
    if now {
        slot.spawn = copies(map, &slot.game, &options, TICKET_COPIES, &mut slot.rng, &slot.opp);
    }
    slot.fresh_queue = queued;
    let k = options.len();
    slot.pending = Some(Pending {
        record,
        legal,
        options,
        sum: vec![0.0; k],
        count: vec![0; k],
        left: need,
        base,
        building,
        action: d.action,
        kind,
        logits: Vec::new(),
        table: Vec::new(),
        worlds: TICKET_COPIES,
        stage: 0,
        flagged: false,
        game_rec: None,
    });
    true
}

/// The look-ahead's look at the decision `d`: its options' copies start when there is room for
/// them, or it joins the queue (true: the game waits); past the queue the decision goes without.
#[allow(clippy::too_many_arguments)]
fn look_start(
    map: &Map,
    slot: &mut Slot,
    d: &Decision,
    options: Vec<u16>,
    legal: Vec<u16>,
    logits: Vec<f32>,
    game_rec: Option<usize>,
    room: &AtomicUsize,
    line: &AtomicUsize,
) -> bool {
    if options.len() < 2 {
        return false;
    }
    let need = options.len() * LOOK_WORLDS;
    let now = reserve(room, need);
    let queued = !now && reserve(line, 1);
    if !now && !queued {
        slot.ticket_skipped += 1;
        return false;
    }
    let record = record_of(map, slot, d, Vec::new(), 1);
    if now {
        slot.spawn = copies(map, &slot.game, &options, LOOK_WORLDS, &mut slot.rng, &slot.opp);
    }
    slot.fresh_queue = queued;
    slot.look_tried += 1;
    let k = options.len();
    let table = if game_rec.is_some() {
        vec![f32::NAN; k * LOOK_WORLDS]
    } else {
        Vec::new()
    };
    slot.pending = Some(Pending {
        record,
        legal,
        options,
        sum: vec![0.0; k],
        count: vec![0; k],
        left: need,
        base: None,
        building: None,
        action: d.action,
        kind: Kind::Look,
        logits,
        table,
        worlds: LOOK_WORLDS,
        stage: 0,
        flagged: false,
        game_rec,
    });
    true
}

/// The look's best other option `best` goes again with the first choice into fresh worlds; the game
/// waits for them (or joins the queue: true), past the queue it goes on without a lesson.
#[allow(clippy::too_many_arguments)]
fn look_check_start(
    map: &Map,
    cfg: &PoolConfig,
    slot: &mut Slot,
    wait: Pending,
    best: usize,
    room: &AtomicUsize,
    line: &AtomicUsize,
) -> bool {
    let options = vec![wait.options[0], wait.options[best]];
    let keep = slot.game.phase == Phase::Keep;
    let worlds = if keep { cfg.look_check_keep } else { LOOK_CHECK };
    let need = options.len() * worlds;
    let now = reserve(room, need);
    let queued = !now && reserve(line, 1);
    if !now && !queued {
        slot.ticket_skipped += 1;
        return false;
    }
    if now {
        slot.spawn = copies(map, &slot.game, &options, worlds, &mut slot.rng, &slot.opp);
    }
    slot.fresh_queue = queued;
    slot.look_checked += 1;
    slot.keep_checked += keep as u64;
    slot.pending = Some(Pending {
        options,
        sum: vec![0.0; 2],
        count: vec![0; 2],
        left: need,
        table: vec![f32::NAN; need],
        worlds,
        stage: 1,
        ..wait
    });
    true
}

/// The first choice of `logits` among `legal` (the look's order) and the move the move forecast `f`
/// (hundreds of points) puts highest of the others - when it puts that one `points` or more above the
/// first choice. None else, or when the forecast has nothing to say (not a number).
fn forecast_flag(map: &Map, legal: &[u16], logits: &[f32], f: &[f32], points: f32) -> Option<(u16, u16)> {
    let first = *options_by(map, legal, logits, true).first()?;
    let best = legal
        .iter()
        .copied()
        .filter(|&a| a != first && f[a as usize].is_finite())
        .max_by(|&a, &b| f[a as usize].total_cmp(&f[b as usize]).then(b.cmp(&a)))?;
    let lead = (f[best as usize] - f[first as usize]) * 100.0;
    (lead.is_finite() && lead >= points).then_some((first, best))
}

/// A move the move forecast put above the first choice (`pair`, the first choice first) goes to the
/// check at once, as a look's best other move does, and is taught by the same rule; the game waits
/// for its copies (or joins the queue: true), past the queue it goes on without a lesson.
#[allow(clippy::too_many_arguments)]
fn forecast_check_start(
    map: &Map,
    slot: &mut Slot,
    d: &Decision,
    pair: (u16, u16),
    legal: Vec<u16>,
    logits: Vec<f32>,
    game_rec: Option<usize>,
    room: &AtomicUsize,
    line: &AtomicUsize,
) -> bool {
    let options = vec![pair.0, pair.1];
    let need = options.len() * LOOK_CHECK;
    let now = reserve(room, need);
    let queued = !now && reserve(line, 1);
    if !now && !queued {
        slot.ticket_skipped += 1;
        return false;
    }
    let record = record_of(map, slot, d, Vec::new(), 1);
    if now {
        slot.spawn = copies(map, &slot.game, &options, LOOK_CHECK, &mut slot.rng, &slot.opp);
    }
    slot.fresh_queue = queued;
    slot.forecast_flagged += 1;
    slot.pending = Some(Pending {
        record,
        legal,
        options,
        sum: vec![0.0; 2],
        count: vec![0; 2],
        left: need,
        base: None,
        building: None,
        action: d.action,
        kind: Kind::Look,
        logits,
        table: vec![f32::NAN; need],
        worlds: LOOK_CHECK,
        stage: 1,
        flagged: true,
        game_rec,
    });
    true
}

/// The copies of a look or of its check (`options`, the table move by move, `worlds` a move) onto the
/// record of their decision: a check's after its look's, a world each (NaN where a move was not tried).
fn forecast_record(r: &mut Record, options: &[u16], table: &[f32], worlds: usize) {
    if table.len() != options.len() * worlds {
        return;
    }
    if r.look.is_empty() {
        r.look = options.to_vec();
        r.look_table = table.to_vec();
        return;
    }
    let w0 = r.look_table.len() / r.look.len();
    let mut look = r.look.clone();
    for &a in options {
        if !look.contains(&a) && look.len() < LOOK_MOVES {
            look.push(a);
        }
    }
    let w = w0 + worlds;
    let mut t = vec![f32::NAN; look.len() * w];
    for (o, &a) in look.iter().enumerate() {
        if let Some(p) = r.look.iter().position(|&x| x == a) {
            t[o * w..o * w + w0].copy_from_slice(&r.look_table[p * w0..(p + 1) * w0]);
        }
        if let Some(p) = options.iter().position(|&x| x == a) {
            t[o * w + w0..(o + 1) * w].copy_from_slice(&table[p * worlds..(p + 1) * worlds]);
        }
    }
    r.look = look;
    r.look_table = t;
}

/// The network's policy over `legal` from the logits `lg` (every other action 0).
fn policy_of(map: &Map, legal: &[u16], lg: &[f32]) -> Vec<f32> {
    let top = legal.iter().map(|&a| lg[a as usize]).fold(f32::NEG_INFINITY, f32::max);
    let mut pol = vec![0.0f32; map.n_actions];
    let mut total = 0.0;
    for &a in legal {
        let e = (lg[a as usize] - top).exp();
        pol[a as usize] = e;
        total += e;
    }
    for &a in legal {
        pol[a as usize] /= total;
    }
    pol
}

/// The lesson of a move the check proved better than the first choice, `lead` its mean lead there
/// (points): the network's policy with what the first choice and the proven move hold together
/// shared anew, every other move as it stands.
fn proven_lesson(map: &Map, legal: &[u16], logits: &[f32], first: u16, proven: u16, lead: f32) -> Vec<f32> {
    let mut pol = policy_of(map, legal, logits);
    let pair = pol[first as usize] + pol[proven as usize];
    let part = (PROVEN_BASE + PROVEN_PER_POINT * lead).clamp(PROVEN_BASE, PROVEN_MAX);
    pol[proven as usize] = pair * part;
    pol[first as usize] = pair * (1.0 - part);
    pol
}

/// The candidate's mean lead over the first choice in a check's table (the first choice's margins,
/// then the candidate's in the same worlds), points.
fn look_lead(table: &[f32], worlds: usize) -> f32 {
    (0..worlds).map(|j| table[worlds + j] - table[j]).sum::<f32>() / worlds.max(1) as f32
}

/// Ticket playouts' lesson on drawing tickets while something is left to build (options: drawing, the
/// teacher's best building move): drawing, if its copies' tickets scored more on average, else the
/// teacher's target `building`.
fn early_lesson(map: &Map, legal: &[u16], sum: &[f32], count: &[u32], building: &[f32]) -> Vec<f32> {
    let mean = |k: usize| sum[k] / count[k].max(1) as f32;
    if mean(0) <= mean(1) {
        return building.to_vec();
    }
    let mut pol = vec![0.0f32; map.n_actions];
    for &a in legal {
        pol[a as usize] = SPREAD / legal.len() as f32;
    }
    pol[map.tickets_action()] += 1.0 - SPREAD;
    pol
}

/// Ticket playouts' lesson at a choice of tickets: `1 - SPREAD` shared by the best moves, `SPREAD`
/// spread over every legal move. A keep: the sets whose copies' tickets scored most on average.
/// Drawing with nothing left to build (`base`: the tickets as they score now): drawing, if its copies
/// scored more than that on average, else every other move.
fn lesson_policy(map: &Map, legal: &[u16], options: &[u16], sum: &[f32], count: &[u32], base: Option<f32>) -> Vec<f32> {
    let mean: Vec<f32> = sum.iter().zip(count).map(|(&s, &c)| s / c.max(1) as f32).collect();
    let best: Vec<u16> = match base {
        Some(b) if mean[0] <= b => legal.iter().cloned().filter(|&a| a != options[0]).collect(),
        Some(_) => vec![options[0]],
        None => {
            let top = mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            options
                .iter()
                .zip(&mean)
                .filter(|&(_, &m)| m >= top - 1e-6)
                .map(|(&a, _)| a)
                .collect()
        }
    };
    let best = if best.is_empty() { options.to_vec() } else { best };
    let mut pol = vec![0.0f32; map.n_actions];
    for &a in legal {
        pol[a as usize] = SPREAD / legal.len() as f32;
    }
    for &a in &best {
        pol[a as usize] += (1.0 - SPREAD) / best.len() as f32;
    }
    pol
}

/// `k` copies of game `s` for each move of `options`, the hidden cards dealt afresh for the player to
/// act (`sample::sample_world`). The j-th copy of every option shares its world and its randomness,
/// so the options are compared on the same cards.
fn copies(
    map: &Map,
    s: &State,
    options: &[u16],
    k: usize,
    rng: &mut Xoshiro,
    opp: &[Option<Opponent>; MAX_PLAYERS],
) -> Vec<Playout> {
    let me = s.current;
    let mut out = Vec::with_capacity(options.len() * k);
    for j in 0..k {
        let world = sample_world(map, s, me as usize, rng, None);
        let seed = rng.next_u64();
        for (option, &a) in options.iter().enumerate() {
            let mut game = world;
            let mut prng = Xoshiro::new(seed);
            game.step(map, a as usize, &mut prng);
            out.push(Playout {
                opp: *opp,
                game,
                rng: prng,
                asked: None,
                leaf: Obs::new(map),
                parent: 0,
                option,
                world: j,
                decider: me,
                passes: 0,
                row: 0,
                end: None,
            });
        }
    }
    out
}

/// Play a copy's moves until a network must choose; at its end, the decider's ticket points (ticket
/// playouts) or margin over the best other seat (the look-ahead). The trained network's seats play
/// its first choice in ticket playouts and samples of its policy in the look-ahead; a frozen
/// network's seat, its first choice; a bot's seat, the bot.
fn advance_playout(map: &Map, cfg: &PoolConfig, p: &mut Playout) {
    let mut moves = Vec::with_capacity(256);
    loop {
        if let Some(ask) = p.asked.take() {
            let Some(lg) = ask.logits else {
                p.asked = Some(ask);
                return;
            };
            let a = root_move(&ask.legal, &lg, ask.noise, ask.seed);
            p.game.step(map, a as usize, &mut p.rng);
            continue;
        }
        if p.game.is_over() || p.game.moves >= PLAYOUT_MOVES {
            p.end = Some(if cfg.lookahead {
                margin(map, &p.game, p.decider as usize)
            } else {
                p.game.tickets_score(map)[p.decider as usize] as f32
            });
            return;
        }
        p.game.legal_actions(map, &mut moves);
        if moves.is_empty() {
            p.game.pass();
            p.passes += 1;
            if p.passes >= p.game.n() as u32 {
                p.game.phase = Phase::Over; // nobody can move: the game ends as it stands
            }
            continue;
        }
        p.passes = 0;
        if moves.len() == 1 {
            p.game.step(map, moves[0] as usize, &mut p.rng);
            continue;
        }
        let seat = p.opp[p.game.current as usize];
        if let Some(Opponent::Bot(b)) = seat {
            let a = b.choose(map, &p.game, &moves, &mut p.rng);
            p.game.step(map, a as usize, &mut p.rng);
            continue;
        }
        p.asked = Some(Ask {
            legal: moves,
            noise: !cfg.ticket_playouts && seat.is_none(),
            seed: p.rng.next_u64(),
            logits: None,
            forecast: f32::NAN,
        });
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::first_choice;
    use std::collections::{HashMap, HashSet};

    /// A stand-in network: logits that depend on the position alone; `phase` makes two of them play
    /// differently.
    fn stub(map: &Map, obs: &Obs, phase: f32) -> Vec<f32> {
        let h: f32 = obs.global.iter().sum::<f32>() + obs.players.iter().sum::<f32>() + obs.tickets.iter().sum::<f32>();
        (0..map.n_actions)
            .map(|a| ((h * 0.37 + phase) * (a % 7 + 1) as f32 + a as f32).sin())
            .collect()
    }

    /// Play `pool` until `games` games have ended, every row answered by the stub of its network
    /// (`row_net`); `each` sees the pool after every `prepare`. The records.
    fn play(pool: &mut Pool, games: u64, mut each: impl FnMut(&Pool)) -> Vec<Record> {
        let map = Map::usa();
        let mut out = Vec::new();
        for _ in 0..400_000 {
            let n = pool.prepare();
            each(pool);
            let mut logits = Vec::with_capacity(n * map.n_actions);
            for i in 0..n {
                logits.extend(stub(map, pool.row_obs(i), pool.row_net(i) as f32));
            }
            pool.feed(&logits, None);
            out.extend(pool.take_records());
            if pool.stats().games_finished >= games {
                return out;
            }
        }
        panic!("the games do not end: {:?}", pool.stats());
    }

    fn tickets_only(partial: bool) -> Rules {
        Rules {
            route_points: false,
            longest_bonus: false,
            partial_tickets: partial,
            ..Rules::OFFICIAL
        }
    }

    fn legal_of(r: &Record) -> Vec<u16> {
        (0..r.obs.legal.len() as u16)
            .filter(|&a| r.obs.legal[a as usize] == 1)
            .collect()
    }

    fn assert_a_policy(r: &Record) {
        assert!((r.policy.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{:?}", r.policy);
        for (a, &p) in r.policy.iter().enumerate() {
            assert!(p == 0.0 || r.obs.legal[a] == 1, "mass on an illegal move");
        }
    }

    /// The share of the best moves in a target that spreads `SPREAD` over the legal moves.
    fn assert_the_best_share(r: &Record) {
        let n_legal = legal_of(r).len() as f32;
        let top = r.policy.iter().cloned().fold(0.0f32, f32::max);
        let best = r.policy.iter().filter(|&&p| (p - top).abs() < 1e-6).count() as f32;
        assert!((top - (SPREAD / n_legal + (1.0 - SPREAD) / best)).abs() < 1e-4, "{top}");
    }

    #[test]
    fn the_teachers_games_record_the_teachers_targets() {
        let map = Map::usa();
        let cfg = PoolConfig {
            games: 8,
            seed: 11,
            tables: vec![2],
            rules: tickets_only(true),
            teacher: true,
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        let records = play(&mut pool, 4, |_| {});
        let st = pool.stats();
        assert!(
            st.evals <= st.decisions && st.decisions <= st.evals + 8,
            "one row a decision: {st:?}"
        );
        assert!(!records.is_empty());
        for r in &records {
            assert_a_policy(r);
            assert_eq!(r.lesson, 0);
            assert_the_best_share(r);
        }
    }

    #[test]
    fn ticket_playouts_teach_a_choice_of_tickets_by_its_copies() {
        let map = Map::usa();
        let cfg = PoolConfig {
            games: 4,
            seed: 7,
            tables: vec![2],
            rules: tickets_only(false),
            teacher: true,
            joint_paths: true,
            ticket_playouts: true,
            playout_slots: 512,
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        let mut copies_seen = false;
        let records = play(&mut pool, 4, |p| {
            copies_seen |= p.rows() > p.games();
            assert!(p.rows() <= p.games() + 512, "at most playout_slots copies");
            assert!(p.queue.len() <= QUEUE);
        });
        let st = pool.stats();
        assert!(copies_seen && st.playouts > 0 && st.ticket_lessons > 0, "{st:?}");
        assert_eq!(st.lessons_by_kind.iter().sum::<u64>(), st.ticket_lessons, "{st:?}");
        assert!(st.lessons_by_kind[Kind::Opening as usize] > 0, "{st:?}");
        let lessons: Vec<&Record> = records.iter().filter(|r| r.lesson == 1).collect();
        assert!(!lessons.is_empty());
        for r in &records {
            assert_a_policy(r);
        }
        for r in &lessons {
            assert_the_best_share(r);
        }
        assert!(
            lessons.iter().any(|r| r.obs.legal[map.keep_action(7)] == 1),
            "an opening's keep among the lessons"
        );
    }

    #[test]
    fn imitation_records_every_move_played_with_its_probability_and_forecast() {
        let map = Map::usa();
        let na = map.n_actions;
        let cfg = PoolConfig {
            games: 8,
            seed: 13,
            tables: vec![2, 3, 4, 5],
            rules: Rules {
                route_price: 50,
                ..Rules::OFFICIAL
            },
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        let mut records = Vec::new();
        while pool.stats().games_finished < 12 {
            let n = pool.prepare();
            pool.feed(&vec![0.0; n * na], Some(&vec![0.25; n]));
            records.extend(pool.take_records());
        }
        let tables: HashSet<u8> = records.iter().map(|r| r.n_players).collect();
        assert_eq!(tables.len(), 4, "every table: {tables:?}");
        for r in &records {
            let a = r.policy.iter().position(|&x| x == 1.0).expect("the move played");
            assert!(r.policy.iter().filter(|&&x| x != 0.0).count() == 1 && r.obs.legal[a] == 1);
            assert_eq!(r.root_forecast, 0.25);
            let legal = legal_of(r).len() as f32;
            assert!((r.root_prob - 1.0 / legal).abs() < 1e-5, "level logits: one of {legal}");
        }
    }

    #[test]
    fn a_pool_against_a_bot_plays_duplicate_deals_and_records_the_trained_seat() {
        let map = Map::usa();
        for players in [2usize, 3, 5] {
            let cfg = PoolConfig {
                games: 2 * players,
                players,
                seed: 31,
                opponents: Opponents::Bot(Bot::Heuristic),
                ..Default::default()
            };
            let pool = Pool::new(map, cfg.clone());
            for unit in pool.slots.chunks(players) {
                for s in unit {
                    assert_eq!(s.game.hand, unit[0].game.hand, "one deal: every seat's cards");
                    assert_eq!(s.game.opening_dealt, unit[0].game.opening_dealt, "and its tickets");
                }
            }
            let mut pool = Pool::new(map, cfg);
            let records = play(&mut pool, 2 * players as u64, |p| {
                for i in 0..p.rows() {
                    assert_eq!(p.row_net(i), 0, "a bot needs no network");
                }
            });
            assert!(!records.is_empty());
            let mut by_deal: HashMap<(u64, u64), Vec<f32>> = HashMap::new();
            for r in &records {
                let (slot, round) = (r.game % (2 * players) as u64, r.game / (2 * players) as u64);
                assert_eq!(r.actor as u64, slot % players as u64, "only the trained seat's moves");
                by_deal
                    .entry((slot / players as u64, round))
                    .or_default()
                    .push(r.advantage);
            }
            for advs in by_deal.values() {
                assert!(advs.iter().all(|&a| a == advs[0]), "one reward a deal: {advs:?}");
            }
        }
    }

    #[test]
    fn a_frozen_network_answers_its_own_seats_and_is_not_recorded() {
        let map = Map::usa();
        for players in [2usize, 4] {
            let cfg = PoolConfig {
                games: 2 * players,
                players,
                seed: 19,
                opponents: Opponents::Net,
                ..Default::default()
            };
            let mut pool = Pool::new(map, cfg);
            let mut asked = [0usize; 2];
            let records = play(&mut pool, 2 * players as u64, |p| {
                for i in 0..p.rows() {
                    let Row::Slot(k) = p.rows[i] else { panic!("no copies") };
                    let frozen = p.decider_seat(i) != a_seat(k, players);
                    assert_eq!(p.row_net(i), frozen as u8);
                    asked[frozen as usize] += 1;
                }
            });
            assert!(asked[0] > 0 && asked[1] > 0, "{asked:?}");
            for r in &records {
                assert_eq!(r.actor as u64, (r.game % (2 * players) as u64) % players as u64);
                assert_eq!(r.n_players as usize, players);
            }
        }
    }

    #[test]
    fn a_games_rewards_are_its_seats_centred_margins_summing_to_zero() {
        let map = Map::usa();
        let cfg = PoolConfig {
            games: 6,
            players: 3,
            seed: 32,
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        let records = play(&mut pool, 6, |_| {});
        let mut by_game: HashMap<u64, [Option<f32>; MAX_PLAYERS]> = HashMap::new();
        for r in &records {
            let e = by_game.entry(r.game).or_insert([None; MAX_PLAYERS]);
            assert!(
                e[r.actor as usize].is_none_or(|a| a == r.advantage),
                "one reward a seat of a game"
            );
            e[r.actor as usize] = Some(r.advantage);
            let m: Vec<f32> = (0..3)
                .map(|q| r.score[q] - (0..3).filter(|&o| o != q).map(|o| r.score[o]).fold(f32::MIN, f32::max))
                .collect();
            let mean = m.iter().sum::<f32>() / 3.0;
            assert!(
                (r.advantage - (m[0] - mean)).abs() < 1e-3,
                "{} against {}",
                r.advantage,
                m[0] - mean
            );
        }
        for seats in by_game.values().filter(|s| s[..3].iter().all(|x| x.is_some())) {
            assert!(
                seats[..3].iter().map(|x| x.unwrap()).sum::<f32>().abs() < 1e-3,
                "{seats:?}"
            );
        }
    }

    #[test]
    fn the_route_price_of_new_games_follows_the_pool() {
        let map = Map::usa();
        let cfg = PoolConfig {
            games: 4,
            seed: 3,
            rules: Rules {
                route_price: 0,
                ..Rules::OFFICIAL
            },
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        assert!(pool.slots.iter().all(|s| s.game.rules.route_price == 0));
        pool.set_route_price(0.4);
        assert!(
            pool.slots.iter().all(|s| s.game.rules.route_price == 0),
            "the games under way keep theirs"
        );
        let fresh = new_game(map, &pool.cfg, 0, 1, &mut Xoshiro::new(1));
        assert_eq!(
            fresh.rules,
            Rules {
                route_price: 40,
                ..Rules::OFFICIAL
            }
        );
        let seen = |s: &State| Obs::of(map, s, 0).global[crate::obs::RULES_AT];
        assert!((seen(&fresh) - 0.4).abs() < 1e-6 && seen(&pool.slots[0].game) == 0.0);
    }

    #[test]
    fn a_mixed_tables_seats_are_drawn_by_weight_and_the_hoarder_sits_at_one_seat_in_half_the_games() {
        let mixed = vec![
            (Opponent::Bot(Bot::Heuristic), 1.0),
            (Opponent::Net(0), 1.0),
            (Opponent::Net(1), 2.0),
        ];
        let mut rng = Xoshiro::new(1);
        // a table of two: one opponent a game, by weight; with duplicate deals the same in both games of a unit
        let cfg = PoolConfig {
            games: 4,
            players: 2,
            seed: 3,
            opponents: Opponents::Mixed(mixed.clone()),
            ..Default::default()
        };
        let mut count = [0usize; 3];
        for round in 0..4000u64 {
            let (a, b) = (
                seat_opponents(&cfg, 0, round, &mut rng),
                seat_opponents(&cfg, 1, round, &mut rng),
            );
            assert_eq!((a[0], b[1]), (None, None), "the trained seat");
            assert_eq!(a[1], b[0], "the same opponent after the trained seat");
            count[mixed.iter().position(|(m, _)| Some(*m) == a[1]).unwrap()] += 1;
        }
        for (k, want) in [0.25f32, 0.25, 0.5].into_iter().enumerate() {
            assert!((count[k] as f32 / 4000.0 - want).abs() < 0.03, "by weight: {count:?}");
        }
        // tables of 3 to 5: the card hoarder at one seat in half the games
        let hoard = Some(Opponent::Bot(Bot::Hoard));
        for players in [3usize, 4, 5] {
            let c = PoolConfig {
                games: players,
                players,
                ..cfg.clone()
            };
            let (mut with, mut place) = (0, vec![0usize; players]);
            for round in 0..2000u64 {
                let s = seat_opponents(&c, 0, round, &mut rng);
                let n = s.iter().filter(|o| **o == hoard).count();
                assert!(
                    n <= 1 && s[0].is_none() && s[1..players].iter().all(|o| o.is_some()),
                    "{s:?}"
                );
                with += n;
                if let Some(q) = s.iter().position(|o| *o == hoard) {
                    place[q] += 1;
                }
            }
            assert!((with as f32 / 2000.0 - HOARDER_SHARE).abs() < 0.04, "{players}: {with}");
            assert!(place[1..].iter().all(|&c| c > 0), "its seat varies: {place:?}");
        }
        let hoarder_listed = PoolConfig {
            players: 3,
            games: 3,
            opponents: Opponents::Mixed(vec![(Opponent::Bot(Bot::Hoard), 1.0)]),
            ..Default::default()
        };
        assert!(
            hoarder_listed.check().is_err(),
            "the hoarder's seat is fixed at a table of 3"
        );
    }

    #[test]
    fn a_mixed_table_plays_its_bots_at_once_and_asks_each_network_for_its_seats() {
        let map = Map::usa();
        let mixed = vec![
            (Opponent::Bot(Bot::Heuristic), 1.0),
            (Opponent::Net(0), 1.0),
            (Opponent::Net(1), 1.0),
        ];
        let cfg = PoolConfig {
            games: 8,
            players: 4,
            seed: 5,
            opponents: Opponents::Mixed(mixed),
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        let mut asked = [0usize; 3];
        let records = play(&mut pool, 8, |p| {
            for i in 0..p.rows() {
                let Row::Slot(k) = p.rows[i] else { panic!("no copies") };
                let (net, seat) = (p.row_net(i), p.decider_seat(i));
                let opp = p.slots[k].opp[seat as usize];
                assert!(!matches!(opp, Some(Opponent::Bot(_))), "a bot's seat moves at once");
                assert_eq!(net == 0, seat == a_seat(k, 4), "the trained network's rows are 0");
                if net > 0 {
                    assert_eq!(opp, Some(Opponent::Net(net - 1)));
                }
                asked[net as usize] += 1;
            }
        });
        assert!(asked.iter().all(|&c| c > 0), "{asked:?}");
        assert!(
            records.iter().all(|r| r.actor as u64 == (r.game % 8) % 4),
            "only the trained seat"
        );
    }

    #[test]
    fn the_look_ahead_teaches_only_the_moves_its_check_proves() {
        let map = Map::usa();
        let cfg = PoolConfig {
            games: 4,
            seed: 9,
            tables: vec![2],
            lookahead: true,
            playout_slots: 512,
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        let mut copy_rows = 0;
        let records = play(&mut pool, 6, |p| {
            copy_rows += p.rows() - p.slots.iter().filter(|s| s.row.is_some()).count();
        });
        let st = pool.stats();
        assert!(copy_rows > 0 && st.look_tried > 0 && st.look_checked > 0, "{st:?}");
        assert!(
            st.look_checked <= st.look_tried && st.look_proven <= st.look_checked,
            "{st:?}"
        );
        assert_eq!(
            st.ticket_lessons, st.look_proven,
            "a lesson for each move proven: {st:?}"
        );
        assert!(st.keep_checked <= st.look_checked && st.forecast_flagged == 0, "{st:?}");
        for r in &records {
            assert_a_policy(r);
            assert!(
                r.look.is_empty() && r.look_table.is_empty(),
                "no move forecast: no copies kept"
            );
            let top = r.policy.iter().cloned().fold(0.0f32, f32::max);
            if r.lesson == 0 {
                assert_eq!(top, 1.0, "a game's own move");
            } else {
                assert!(top < 1.0, "a lesson is a policy");
            }
        }
    }

    #[test]
    fn the_move_forecast_sends_moves_to_the_check_and_the_records_keep_the_copies() {
        let map = Map::usa();
        let na = map.n_actions;
        let cfg = PoolConfig {
            games: 6,
            players: 3,
            seed: 9,
            opponents: Opponents::Mixed(vec![(Opponent::Bot(Bot::Heuristic), 1.0), (Opponent::Net(0), 1.0)]),
            lookahead: true,
            look_check_keep: 40,
            look_forecast: true,
            playout_slots: 1024,
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        let mut records = Vec::new();
        let mut asked = 0;
        while pool.stats().games_finished < 6 {
            let n = pool.prepare();
            let rows = pool.forecast_rows();
            // every legal move but the lowest (the first choice of level logits) 50 points above it,
            // the highest one 90 points
            let mut f = vec![0.0f32; rows.len() * na];
            for (j, &i) in rows.iter().enumerate() {
                assert_eq!(pool.row_net(i), 0, "the trained network's decisions");
                let legal: Vec<usize> = (0..na).filter(|&a| pool.row_obs(i).legal[a] == 1).collect();
                for &a in legal.iter().skip(1) {
                    f[j * na + a] = 0.5;
                }
                f[j * na + legal[legal.len() - 1]] = 0.9;
            }
            asked += rows.len();
            pool.feed_forecast(&rows, &f);
            pool.feed(&vec![0.0f32; n * na], None);
            records.extend(pool.take_records());
        }
        let st = pool.stats();
        assert!(asked > 0 && st.forecast_flagged > 0 && st.forecast_keeps > 0, "{st:?}");
        assert!(
            st.forecast_proven <= st.forecast_flagged && st.forecast_proven <= st.look_proven,
            "{st:?}"
        );
        let looked: Vec<&Record> = records.iter().filter(|r| !r.look.is_empty()).collect();
        assert!(!looked.is_empty());
        for r in &looked {
            assert_eq!(r.lesson, 0, "the decision's own record keeps the copies");
            let w = r.look_table.len() / r.look.len();
            assert!(
                [LOOK_WORLDS, LOOK_CHECK, LOOK_WORLDS + LOOK_CHECK, LOOK_WORLDS + 40].contains(&w),
                "{w}"
            );
            assert!(r.look.len() <= LOOK_MOVES && r.look_table.iter().any(|v| v.is_finite()));
        }
        assert!(
            looked
                .iter()
                .any(|r| r.look.len() == 4 && r.look.iter().all(|&a| a as usize > map.tickets_action())),
            "a later keep's three sets and the hint"
        );
    }

    #[test]
    fn the_move_forecast_flags_a_move_far_enough_above_the_first_choice() {
        let map = Map::usa();
        let legal = vec![1u16, 2, 3];
        let mut lg = vec![-9.0f32; map.n_actions];
        lg[1] = 2.0;
        lg[2] = 1.0;
        lg[3] = 0.0;
        let mut f = vec![f32::NAN; map.n_actions];
        f[1] = 0.10;
        f[2] = 0.20;
        f[3] = 0.12;
        assert_eq!(forecast_flag(map, &legal, &lg, &f, 5.0), Some((1, 2)));
        assert_eq!(forecast_flag(map, &legal, &lg, &f, 15.0), None, "10 points: not 15");
        f[2] = f32::NAN;
        assert_eq!(forecast_flag(map, &legal, &lg, &f, 5.0), None);
        assert_eq!(forecast_flag(map, &legal, &lg, &f, 1.0), Some((1, 3)));
        f[1] = f32::NAN;
        assert_eq!(
            forecast_flag(map, &legal, &lg, &f, 1.0),
            None,
            "no forecast of the first choice"
        );
    }

    #[test]
    fn a_checks_copies_go_after_its_looks_on_the_decisions_record() {
        let map = Map::usa();
        let pool = Pool::new(
            map,
            PoolConfig {
                games: 2,
                ..Default::default()
            },
        );
        let d = Decision {
            action: 0,
            prob: 1.0,
            forecast: 0.0,
        };
        let mut r = record_of(map, &pool.slots[0], &d, Vec::new(), 0);
        forecast_record(&mut r, &[7, 8, 9], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        assert_eq!(r.look, vec![7, 8, 9]);
        assert_eq!(r.look_table, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        forecast_record(&mut r, &[7, 9], &[10.0, 11.0, 12.0, 20.0, 21.0, 22.0], 3);
        assert_eq!(r.look, vec![7, 8, 9]);
        let t = &r.look_table;
        assert_eq!(t.len(), 15);
        assert_eq!(&t[0..5], &[1.0, 2.0, 10.0, 11.0, 12.0]);
        assert_eq!(&t[5..7], &[3.0, 4.0]);
        assert!(t[7..10].iter().all(|v| v.is_nan()), "move 8 was not in the check");
        assert_eq!(&t[10..15], &[5.0, 6.0, 20.0, 21.0, 22.0]);
    }

    #[test]
    fn a_proven_lesson_shares_the_first_choices_weight_by_the_lead() {
        let map = Map::usa();
        let (first, proven, other) = (1u16, 2u16, 3u16);
        let mut lg = vec![-9.0f32; map.n_actions];
        for (a, v) in [(first, 5.0), (proven, -4.0), (other, 1.0)] {
            lg[a as usize] = v;
        }
        let legal = vec![first, proven, other];
        let before = policy_of(map, &legal, &lg);
        let pair = before[first as usize] + before[proven as usize];
        for (lead, part) in [(3.0f32, 0.53f32), (10.0, 0.60), (30.0, 0.80), (55.0, 0.80)] {
            let pol = proven_lesson(map, &legal, &lg, first, proven, lead);
            assert!((pol[proven as usize] - pair * part).abs() < 1e-5, "lead {lead}");
            assert!((pol[first as usize] - pair * (1.0 - part)).abs() < 1e-5, "lead {lead}");
            assert_eq!(pol[other as usize], before[other as usize]);
        }
        assert!((look_lead(&[0.0, 10.0, -5.0, 3.0, 5.0, 15.0, 0.0, 8.0], 4) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn ticket_playouts_lessons_compare_the_copies_with_the_tickets_as_they_are() {
        let map = Map::usa();
        let draw = map.tickets_action() as u16;
        let legal = vec![0u16, 5, draw];
        let pol = lesson_policy(map, &legal, &[draw], &[30.0], &[3], Some(8.0));
        assert!(
            (pol[draw as usize] - (SPREAD / 3.0 + 1.0 - SPREAD)).abs() < 1e-6,
            "10 against 8: draw"
        );
        let pol = lesson_policy(map, &legal, &[draw], &[18.0], &[3], Some(8.0));
        assert!(
            (pol[draw as usize] - SPREAD / 3.0).abs() < 1e-6,
            "6 against 8: any other move"
        );
        assert!((pol[0] - (SPREAD / 3.0 + (1.0 - SPREAD) / 2.0)).abs() < 1e-6 && (pol[5] - pol[0]).abs() < 1e-6);
        let keeps: Vec<u16> = [3u8, 5, 7].iter().map(|&m| map.keep_action(m) as u16).collect();
        let pol = lesson_policy(map, &keeps, &keeps, &[20.0, 20.0, 12.0], &[2, 2, 2], None);
        assert!(
            (pol[keeps[0] as usize] - pol[keeps[1] as usize]).abs() < 1e-6,
            "the best average, ties shared"
        );
        assert!((pol[keeps[2] as usize] - SPREAD / 3.0).abs() < 1e-6);
        let build: Vec<f32> = (0..map.n_actions).map(|a| if a == 3 { 0.9 } else { 0.0 }).collect();
        let legal = vec![3u16, 7, draw];
        assert_eq!(
            early_lesson(map, &legal, &[20.0, 30.0], &[2, 2], &build),
            build,
            "building scored more"
        );
        let pol = early_lesson(map, &legal, &[40.0, 30.0], &[2, 2], &build);
        assert!(
            (pol[draw as usize] - (SPREAD / 3.0 + 1.0 - SPREAD)).abs() < 1e-6,
            "drawing scored more"
        );
    }

    #[test]
    fn the_decision_rule_samples_the_policy_and_without_noise_plays_the_first_choice() {
        let map = Map::usa();
        let legal: Vec<u16> = vec![2, 9, 40, 300, map.blind_action() as u16];
        let mut lg = vec![-30.0f32; map.n_actions];
        for (k, &a) in legal.iter().enumerate() {
            lg[a as usize] = [0.5f32, -1.0, 1.2, 0.0, 0.3][k];
        }
        assert_eq!(root_move(&legal, &lg, false, 1), first_choice(&lg, &legal));
        let mut seen = [0usize; 5];
        for seed in 0..40_000u64 {
            let a = root_move(&legal, &lg, true, seed);
            seen[legal.iter().position(|&x| x == a).unwrap()] += 1;
        }
        for (k, &a) in legal.iter().enumerate() {
            let p = root_prob(&legal, &lg, a);
            assert!(
                (seen[k] as f32 / 40_000.0 - p).abs() < 0.01,
                "move {a}: {} against {p}",
                seen[k]
            );
        }
    }

    #[test]
    fn games_with_few_trains_always_end() {
        let map = Map::usa();
        let cfg = PoolConfig {
            games: 10,
            players: 5,
            seed: 7,
            rules: Rules {
                route_points: false,
                trains: 10,
                ..Rules::OFFICIAL
            },
            ..Default::default()
        };
        let mut pool = Pool::new(map, cfg);
        play(&mut pool, 20, |_| {});
    }
}
