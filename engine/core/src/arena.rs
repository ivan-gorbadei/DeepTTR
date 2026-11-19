//! Matches: many games at once, each seat played by a network or by a baseline bot.
//!
//! A match sets contestant A against contestant B on a sequence of deals. Each deal is played once
//! per seat - a unit of `players` games: A takes each seat in turn and B holds every other seat, so
//! the luck of the deal cancels (two players: the same deal twice with the seats swapped). The games
//! of a unit also share everything random after the deal, so two identical contestants play the very
//! same game in every seating: a match measures only how the contestants differ.
//!
//! A network lives outside the engine. `prepare` plays every game on until the player to act is a
//! network with a real choice; `row_obs` gives those positions as the network's input; `feed` takes
//! the policy logits and plays each position's first choice (`first_choice`). Bots' moves, forced
//! moves and passes are played at once, so a match of two bots needs no call but `prepare`.
//!
//! A contestant that looks ahead is the caller's too: `look` gives the look-ahead of a waiting
//! position (`look::Lookahead`) and `play` takes the moves the caller decided on.

use crate::bots::Bot;
use crate::look::Lookahead;
use crate::map::Map;
use crate::obs::Obs;
use crate::rng::{splitmix64, Xoshiro};
use crate::state::{FinalScore, Phase, Rules, State, MAX_PLAYERS};
use rayon::prelude::*;

/// Who plays a contestant's seats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    /// a network: the caller evaluates its positions
    Net,
    Bot(Bot),
}

impl Player {
    /// `"net"` or a bot's name (`Bot::parse`).
    pub fn parse(name: &str) -> Option<Player> {
        if name == "net" {
            Some(Player::Net)
        } else {
            Bot::parse(name).map(Player::Bot)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ArenaConfig {
    /// seats at every table, 2 to 5
    pub players: usize,
    /// deals to play; each is played `players` times
    pub deals: usize,
    /// the first deal's place in the seed's sequence: a later part of a match starts where the
    /// part before it ended
    pub first_deal: u64,
    pub seed: u64,
    pub a: Player,
    pub b: Player,
    pub rules: Rules,
}

/// The deal `deal` of the sequence of `seed`.
pub fn deal_rng(seed: u64, deal: u64) -> Xoshiro {
    let mut x = seed ^ deal.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    Xoshiro::new(splitmix64(&mut x))
}

/// Everything random after the deal (the reshuffles of the discard pile, a bot's coin flips): one
/// stream per deal, the same for each of its seatings.
pub fn play_rng(seed: u64, deal: u64) -> Xoshiro {
    deal_rng(seed ^ 0xA076_1D64_78BD_642F, deal)
}

/// Contestant A's seat in game `index` of a match at tables of `n`: the games of a unit take the
/// seats in turn.
pub fn a_seat(index: usize, n: usize) -> u8 {
    (index % n) as u8
}

/// The first choice of a policy: the legal move with the highest logit; of equal ones, the one with
/// the highest action id.
pub fn first_choice(logits: &[f32], legal: &[u16]) -> u16 {
    *legal
        .iter()
        .max_by(|&&a, &&b| {
            logits[a as usize]
                .partial_cmp(&logits[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("a legal move")
}

/// A finished game. The per-seat arrays are by absolute seat.
#[derive(Clone, Debug)]
pub struct GameResult {
    /// the deal's place in the seed's sequence
    pub deal: u64,
    pub a_seat: u8,
    /// turns played, all the seats together
    pub turns: u16,
    pub score: FinalScore,
    /// +1 for first place, -1 for last, linear between (`FinalScore::utility`)
    pub utility: [f32; MAX_PLAYERS],
    /// tickets held at the end
    pub tickets_held: [u8; MAX_PLAYERS],
    /// ticket draws made after the opening
    pub ticket_draws: [u8; MAX_PLAYERS],
    pub trains_left: [u8; MAX_PLAYERS],
    /// train cards taken from the deck and from the face-up row over the game
    pub blind_draws: [u16; MAX_PLAYERS],
    pub face_up_draws: [u16; MAX_PLAYERS],
    /// per route, the seat that claimed it (`state::NONE`: nobody)
    pub route_owner: Vec<u8>,
}

/// One game under way, in a match or in a look-ahead: its state, the stream of everything random
/// after the deal, and the position a network is asked about while the game waits for its move.
pub(crate) struct GameRun {
    pub(crate) game: State,
    pub(crate) rng: Xoshiro,
    pub(crate) obs: Obs,
    pub(crate) waiting: bool,
    /// the number the waiting decision took from the stream: the seed of a look-ahead at it
    pub(crate) seed: u64,
    /// passes in a row
    passes: u32,
}

impl GameRun {
    pub(crate) fn new(map: &Map, game: State, rng: Xoshiro) -> GameRun {
        GameRun {
            game,
            rng,
            obs: Obs::new(map),
            waiting: false,
            seed: 0,
            passes: 0,
        }
    }

    /// Play the game on until it ends, has `max_moves` moves, or a network must choose: it then
    /// waits, `obs` the position as the seat to act sees it. `bot` names the seats a bot plays.
    pub(crate) fn advance(&mut self, map: &Map, max_moves: u16, bot: impl Fn(u8) -> Option<Bot>) {
        if self.waiting {
            return;
        }
        let mut moves = Vec::with_capacity(256);
        while !self.game.is_over() && self.game.moves < max_moves {
            self.game.legal_actions(map, &mut moves);
            if moves.is_empty() {
                self.game.pass();
                self.passes += 1;
                if self.passes >= self.game.n() as u32 {
                    // nobody can move - no card to draw, no ticket, no route anyone can pay, and no claim
                    // left to start the last round (a variant with few trains can get there): the game
                    // ends as it stands
                    self.game.phase = Phase::Over;
                }
                continue;
            }
            self.passes = 0;
            if moves.len() == 1 {
                self.game.step(map, moves[0] as usize, &mut self.rng);
                continue;
            }
            let seat = self.game.current;
            if let Some(bot) = bot(seat) {
                let action = bot.choose(map, &self.game, &moves, &mut self.rng);
                self.game.step(map, action as usize, &mut self.rng);
                continue;
            }
            // every decision of a network takes one number from the game's stream: the seed of a
            // look-ahead at that decision. It is taken also when no look-ahead is made, so the cards
            // a game deals after a move do not depend on how the move was chosen
            self.seed = self.rng.next_u64();
            self.obs.fill(map, &self.game, seat as usize);
            self.waiting = true;
            return;
        }
    }

    /// The legal moves of the waiting decision.
    pub(crate) fn legal(&self, map: &Map) -> Vec<u16> {
        let mut moves = Vec::with_capacity(256);
        self.game.legal_actions(map, &mut moves);
        moves
    }

    /// The waiting network's move.
    pub(crate) fn play(&mut self, map: &Map, action: u16) {
        self.game.step(map, action as usize, &mut self.rng);
        self.waiting = false;
    }
}

struct Slot {
    run: GameRun,
    result: Option<GameResult>,
}

pub struct Arena {
    map: &'static Map,
    cfg: ArenaConfig,
    slots: Vec<Slot>,
    /// the games waiting for a network's answer, in game order
    rows: Vec<usize>,
}

impl Arena {
    pub fn new(map: &'static Map, cfg: ArenaConfig) -> Arena {
        let n = cfg.players;
        assert!((2..=MAX_PLAYERS).contains(&n), "2 to {MAX_PLAYERS} players");
        let mut slots = Vec::with_capacity(cfg.deals * n);
        for index in 0..cfg.deals * n {
            let deal = cfg.first_deal + (index / n) as u64;
            let game = State::with_rules(map, n, cfg.rules, &mut deal_rng(cfg.seed, deal));
            slots.push(Slot {
                run: GameRun::new(map, game, play_rng(cfg.seed, deal)),
                result: None,
            });
        }
        Arena {
            map,
            cfg,
            slots,
            rows: Vec::new(),
        }
    }

    /// The games of the match: deals x players.
    pub fn games(&self) -> usize {
        self.slots.len()
    }

    /// Play every game on until it ends or a network must choose; returns the number of rows that
    /// wait for `feed` or `play` (0: every game is over).
    pub fn prepare(&mut self) -> usize {
        let (map, cfg) = (self.map, &self.cfg);
        self.slots.par_iter_mut().enumerate().for_each(|(index, slot)| {
            if slot.result.is_some() {
                return;
            }
            let n = cfg.players;
            slot.run.advance(map, u16::MAX, |seat| {
                match if seat == a_seat(index, n) { cfg.a } else { cfg.b } {
                    Player::Bot(bot) => Some(bot),
                    Player::Net => None,
                }
            });
            if !slot.run.waiting {
                slot.result = Some(result_of(map, cfg, index, &slot.run.game));
            }
        });
        self.rows = (0..self.slots.len()).filter(|&i| self.slots[i].run.waiting).collect();
        self.rows.len()
    }

    /// The rows that wait for the networks (after `prepare`).
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Row `i`'s position, as its player to act sees it.
    pub fn row_obs(&self, i: usize) -> &Obs {
        &self.slots[self.rows[i]].run.obs
    }

    /// Whose network answers row `i`: 0 contestant A's, 1 contestant B's.
    pub fn row_side(&self, i: usize) -> u8 {
        let index = self.rows[i];
        (self.slots[index].run.game.current != a_seat(index, self.cfg.players)) as u8
    }

    /// The look-ahead of row `i`'s decision over `options` (`look::Lookahead`). Its seed is the
    /// number the decision took from the game's stream plus `stage`, so a match with look-aheads
    /// replays from its seed, and a second look at the same decision (`stage` 1) deals other worlds.
    pub fn look(&self, i: usize, options: &[u16], worlds: usize, stage: u64) -> Lookahead {
        let run = &self.slots[self.rows[i]].run;
        Lookahead::new(self.map, &run.game, options, worlds, run.seed.wrapping_add(stage))
    }

    /// The networks' answers, `rows` x `n_actions` policy logits: every waiting game plays its
    /// first choice.
    pub fn feed(&mut self, logits: &[f32]) {
        let map = self.map;
        let na = map.n_actions;
        assert_eq!(logits.len(), self.rows.len() * na, "logits: rows x actions");
        for (j, &index) in self.rows.iter().enumerate() {
            let run = &mut self.slots[index].run;
            let action = first_choice(&logits[j * na..(j + 1) * na], &run.legal(map));
            run.play(map, action);
        }
        self.rows.clear();
    }

    /// The moves the caller chose for the waiting rows, one each. Nothing is played if any of them
    /// is not legal.
    pub fn play(&mut self, actions: &[u16]) -> Result<(), String> {
        let map = self.map;
        if actions.len() != self.rows.len() {
            return Err(format!("{} moves for {} waiting rows", actions.len(), self.rows.len()));
        }
        for (j, &index) in self.rows.iter().enumerate() {
            if !self.slots[index].run.legal(map).contains(&actions[j]) {
                return Err(format!("row {j}: move {} is not legal", actions[j]));
            }
        }
        for (j, &index) in self.rows.iter().enumerate() {
            self.slots[index].run.play(map, actions[j]);
        }
        self.rows.clear();
        Ok(())
    }

    pub fn is_done(&self) -> bool {
        self.slots.iter().all(|s| s.result.is_some())
    }

    /// The finished games, in game order (all of them once `is_done`).
    pub fn results(&self) -> Vec<GameResult> {
        self.slots.iter().filter_map(|s| s.result.clone()).collect()
    }
}

fn result_of(map: &Map, cfg: &ArenaConfig, index: usize, s: &State) -> GameResult {
    let n = cfg.players;
    let score = s.finalize(map);
    let mut r = GameResult {
        deal: cfg.first_deal + (index / n) as u64,
        a_seat: a_seat(index, n),
        turns: s.turns,
        utility: score.utility(),
        score,
        tickets_held: [0; MAX_PLAYERS],
        ticket_draws: s.ticket_draws,
        trains_left: s.trains,
        blind_draws: s.blind_draws,
        face_up_draws: [0; MAX_PLAYERS],
        route_owner: s.route_owner[..map.n_routes()].to_vec(),
    };
    for q in 0..n {
        r.tickets_held[q] = s.tickets_held[q].count_ones() as u8;
        r.face_up_draws[q] = s.face_up_taken[q].iter().sum();
    }
    r.trains_left[n..].fill(0);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::NONE;

    /// A stand-in network: logits that depend on the position alone, so the same position always
    /// gets the same move; `phase` makes two of them play differently.
    fn stub(map: &Map, obs: &Obs, phase: f32) -> Vec<f32> {
        let h: f32 = obs.global.iter().sum::<f32>() + obs.players.iter().sum::<f32>() + obs.tickets.iter().sum::<f32>();
        (0..map.n_actions)
            .map(|a| ((h * 0.37 + phase) * (a % 7 + 1) as f32).sin())
            .collect()
    }

    fn config(players: usize, deals: usize, a: Player, b: Player) -> ArenaConfig {
        ArenaConfig {
            players,
            deals,
            first_deal: 0,
            seed: 2024,
            a,
            b,
            rules: Rules::OFFICIAL,
        }
    }

    /// Play a match to the end, A's network `phase_a` and B's `phase_b`.
    fn play(arena: &mut Arena, phase_a: f32, phase_b: f32) -> Vec<GameResult> {
        let map = Map::usa();
        let mut steps = 0;
        while arena.prepare() > 0 {
            let mut logits = Vec::with_capacity(arena.rows() * map.n_actions);
            for i in 0..arena.rows() {
                let phase = if arena.row_side(i) == 0 { phase_a } else { phase_b };
                logits.extend(stub(map, arena.row_obs(i), phase));
            }
            arena.feed(&logits);
            steps += 1;
            assert!(steps < 5000, "a game does not end");
        }
        assert!(arena.is_done());
        arena.results()
    }

    #[test]
    fn the_first_choice_is_the_highest_legal_logit() {
        let logits = [0.5, 3.0, -1.0, 3.0, 9.0];
        assert_eq!(first_choice(&logits, &[0, 2]), 0);
        assert_eq!(first_choice(&logits, &[0, 1, 2]), 1);
        assert_eq!(
            first_choice(&logits, &[0, 1, 2, 3]),
            3,
            "of equal logits the higher action id"
        );
        assert_eq!(first_choice(&logits, &[2]), 2);
    }

    #[test]
    fn identical_contestants_play_every_seating_of_a_deal_the_same() {
        let map = Map::usa();
        for n in 2..=5 {
            let mut arena = Arena::new(map, config(n, 3, Player::Net, Player::Net));
            let res = play(&mut arena, 0.3, 0.3);
            assert_eq!(res.len(), 3 * n);
            for unit in res.chunks(n) {
                let mut a_total = 0.0;
                for (k, g) in unit.iter().enumerate() {
                    assert_eq!((g.deal, g.a_seat as usize), (unit[0].deal, k));
                    assert_eq!(
                        g.score.total, unit[0].score.total,
                        "{n} players: the same game in every seating"
                    );
                    assert_eq!(g.route_owner, unit[0].route_owner);
                    a_total += g.utility[g.a_seat as usize];
                }
                assert!(
                    a_total.abs() < 1e-5,
                    "{n} players: A's seats of a deal sum to {a_total}"
                );
            }
        }
    }

    #[test]
    fn different_contestants_play_the_same_deals_differently() {
        let map = Map::usa();
        let mut arena = Arena::new(map, config(2, 6, Player::Net, Player::Net));
        let res = play(&mut arena, 0.3, 1.7);
        assert!(
            res.chunks(2).any(|u| u[0].score.total != u[1].score.total),
            "the seatings of some deal differ"
        );
    }

    #[test]
    fn the_games_of_a_unit_share_the_deal_and_the_seats_rotate() {
        let map = Map::usa();
        let arena = Arena::new(map, config(3, 2, Player::Net, Player::Net));
        assert_eq!(arena.games(), 6);
        for unit in arena.slots.chunks(3) {
            for s in unit {
                assert_eq!(s.run.game.hand, unit[0].run.game.hand);
                assert_eq!(s.run.game.face_up, unit[0].run.game.face_up);
                assert_eq!(s.run.game.opening_dealt, unit[0].run.game.opening_dealt);
                assert_eq!(s.run.game.current, unit[0].run.game.current);
            }
        }
        assert_ne!(
            arena.slots[0].run.game.hand, arena.slots[3].run.game.hand,
            "another deal"
        );
        assert_eq!((0..6).map(|i| a_seat(i, 3)).collect::<Vec<_>>(), [0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn a_later_part_of_a_match_plays_the_deals_that_follow() {
        let map = Map::usa();
        let whole = play(
            &mut Arena::new(map, config(2, 4, Player::Net, Player::Bot(Bot::Hoard))),
            0.3,
            0.0,
        );
        let mut cfg = config(2, 2, Player::Net, Player::Bot(Bot::Hoard));
        cfg.first_deal = 2;
        let part = play(&mut Arena::new(map, cfg), 0.3, 0.0);
        assert_eq!(part.len(), 4);
        for (x, y) in part.iter().zip(&whole[4..]) {
            assert_eq!(
                (x.deal, x.a_seat, x.score.total, x.turns),
                (y.deal, y.a_seat, y.score.total, y.turns)
            );
        }
    }

    #[test]
    fn a_waiting_row_is_the_position_of_its_player_to_act() {
        let map = Map::usa();
        let mut arena = Arena::new(map, config(3, 2, Player::Net, Player::Bot(Bot::Heuristic)));
        let mut seen = 0;
        while arena.prepare() > 0 && seen < 200 {
            let mut logits = Vec::new();
            for i in 0..arena.rows() {
                let index = arena.rows[i];
                let game = &arena.slots[index].run.game;
                assert_eq!(game.current, a_seat(index, 3), "only A is a network here");
                assert_eq!(arena.row_side(i), 0);
                assert!(arena.row_obs(i) == &Obs::of(map, game, game.current as usize));
                assert!(
                    arena.row_obs(i).legal.iter().filter(|&&x| x == 1).count() > 1,
                    "a real choice"
                );
                logits.extend(stub(map, arena.row_obs(i), 0.9));
                seen += 1;
            }
            arena.feed(&logits);
        }
        assert!(seen > 0);
    }

    #[test]
    fn bots_play_a_match_without_a_network() {
        let map = Map::usa();
        let mut arena = Arena::new(
            map,
            config(2, 40, Player::Bot(Bot::Heuristic), Player::Bot(Bot::Random)),
        );
        assert_eq!(arena.prepare(), 0);
        assert!(arena.is_done());
        let res = arena.results();
        assert_eq!(res.len(), 80);
        let wins = res.iter().filter(|g| g.utility[g.a_seat as usize] > 0.0).count();
        assert!(wins > 70, "the ticket builder beats random play: {wins} of 80");
        for g in &res {
            let n = 2;
            assert!(g.turns > 0);
            for q in 0..n {
                let built: u32 = (0..map.n_routes())
                    .filter(|&r| g.route_owner[r] == q as u8)
                    .map(|r| map.routes[r].length as u32)
                    .sum();
                assert_eq!(built + g.trains_left[q] as u32, 45);
                assert!(g.tickets_held[q] >= 2, "two tickets or more are kept at the start");
                assert!(g.score.completed[q] as u8 <= g.tickets_held[q]);
            }
            assert!(g.route_owner.iter().all(|&o| o == NONE || (o as usize) < n));
            assert_eq!(g.trains_left[n..], [0; MAX_PLAYERS - 2]);
        }
    }

    #[test]
    fn a_match_of_bots_is_the_same_on_every_platform() {
        // pinned: the deals, the rules and the bots are deterministic, with no floating-point operation whose result
        // depends on the platform. A change here means the engine deals or plays differently than before
        let map = Map::usa();
        let mut arena = Arena::new(
            map,
            ArenaConfig {
                seed: 1,
                ..config(2, 50, Player::Bot(Bot::Heuristic), Player::Bot(Bot::Hoard))
            },
        );
        arena.prepare();
        let res = arena.results();
        let points: i32 = res
            .iter()
            .map(|g| g.score.total.iter().map(|&x| x as i32).sum::<i32>())
            .sum();
        let turns: u32 = res.iter().map(|g| g.turns as u32).sum();
        let completed: i32 = res
            .iter()
            .map(|g| g.score.completed.iter().map(|&x| x as i32).sum::<i32>())
            .sum();
        assert_eq!((points, turns, completed), (14409, 6576, 223));
    }

    #[test]
    fn games_with_few_trains_always_end() {
        let map = Map::usa();
        for trains in [3u8, 5, 8] {
            let mut cfg = config(4, 6, Player::Net, Player::Bot(Bot::Random));
            cfg.rules = Rules {
                trains,
                ..Rules::OFFICIAL
            };
            let res = play(&mut Arena::new(map, cfg), 0.5, 0.0);
            assert_eq!(res.len(), 24);
        }
    }
}
