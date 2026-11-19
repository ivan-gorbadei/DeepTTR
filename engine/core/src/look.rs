//! The look-ahead of one decision: candidate moves played out to the end of the game.
//!
//! A policy's first choice is one forward pass; whether another move would have won more it cannot
//! say. The look-ahead measures it. The position is copied into `worlds` hidden worlds, each dealt
//! afresh from what the seat to act sees (`sample::sample_world`). In every world each candidate
//! move is played, and the rest of the game is played by the network's first choice at every seat.
//! The result is a table, candidates x worlds, of the deciding seat's final margin over the best
//! other seat.
//!
//! The copies of a world differ only in the candidate: they start from the same hidden cards and
//! take the same randomness afterwards. The candidates are therefore compared in pairs, world by
//! world, and `look_better` asks of a candidate what a paired test asks: a lead over the first
//! choice of at least `LOOK_POINTS` points on average, and of `LOOK_Z` standard errors.
//!
//! This is not a tree search. No tree is built and no position is visited twice: every line is one
//! playout by the policy, as deep as the game is long, and the only branching is at the decision
//! itself. The network stays outside, as in `arena`: `prepare`, the rows' observations, `feed`.

use crate::arena::{first_choice, GameRun};
use crate::map::Map;
use crate::obs::Obs;
use crate::rng::Xoshiro;
use crate::sample::sample_world;
use crate::state::State;

/// A candidate is the better move when its copies end at least this many points of margin above
/// the first choice's on average, and this many standard errors of the paired differences.
pub const LOOK_POINTS: f32 = 3.0;
pub const LOOK_Z: f32 = 2.0;
/// The candidates of a decision that is not a choice of tickets.
const MAX_OPTIONS: usize = 4;
/// A copy that has not ended after this many moves is scored as it stands.
const MAX_MOVES: u16 = 2000;

/// The moves a look-ahead tries at a decision among `legal`, the first choice first. A choice of
/// tickets: every set when there are four or fewer (the opening's any two of three, or all three),
/// else the policy's best three. Any other decision: the first choice, the best move of each other
/// kind (a claim, a card, drawing tickets), then the next by the policy, four at most.
pub fn look_options(map: &Map, legal: &[u16], logits: &[f32]) -> Vec<u16> {
    // the order of `first_choice`: the higher logit, then the higher action id
    options_by(map, legal, logits, false)
}

/// `look_options` with equal logits in the order of their action ids, the lower first when
/// `lower_first` (the look-ahead of training orders them so).
pub(crate) fn options_by(map: &Map, legal: &[u16], logits: &[f32], lower_first: bool) -> Vec<u16> {
    let mut order = legal.to_vec();
    order.sort_by(|&a, &b| {
        let ids = if lower_first { a.cmp(&b) } else { b.cmp(&a) };
        logits[b as usize].total_cmp(&logits[a as usize]).then(ids)
    });
    if order.iter().all(|&a| a as usize > map.tickets_action()) {
        if order.len() > MAX_OPTIONS {
            order.truncate(3);
        }
        return order;
    }
    let kind = |a: u16| match a as usize {
        x if x < map.n_claim => 0,
        x if x < map.tickets_action() => 1,
        _ => 2,
    };
    let mut out = vec![order[0]];
    for k in (0..3).filter(|&k| k != kind(order[0])) {
        out.extend(order.iter().find(|&&a| kind(a) == k));
    }
    for &a in &order {
        if out.len() >= MAX_OPTIONS {
            break;
        }
        if !out.contains(&a) {
            out.push(a);
        }
    }
    out
}

/// The verdict on a candidate from a table of two rows, the first choice's margins in `worlds`
/// worlds and then the candidate's in the same worlds: is the candidate the better move?
pub fn look_better(table: &[f32], worlds: usize) -> bool {
    let d: Vec<f32> = (0..worlds).map(|j| table[worlds + j] - table[j]).collect();
    let n = worlds as f32;
    let mean = d.iter().sum::<f32>() / n;
    let var = d.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (n - 1.0).max(1.0);
    mean >= LOOK_POINTS && mean * mean * n > LOOK_Z * LOOK_Z * var
}

/// One candidate in one world, played to the end.
struct Playout {
    run: GameRun,
    option: usize,
    world: usize,
    done: bool,
}

pub struct Lookahead {
    map: &'static Map,
    /// the seat that decides
    seat: usize,
    worlds: usize,
    copies: Vec<Playout>,
    /// options x worlds: the deciding seat's margin at the end of each copy
    table: Vec<f32>,
    /// the copies waiting for the network's answer
    rows: Vec<usize>,
}

impl Lookahead {
    /// The look-ahead of the decision `s` stands at, over `options` (legal moves of its player to
    /// act), `worlds` copies each, everything random from `seed`.
    pub fn new(map: &'static Map, s: &State, options: &[u16], worlds: usize, seed: u64) -> Lookahead {
        let seat = s.current as usize;
        let mut rng = Xoshiro::new(seed);
        let mut copies = Vec::with_capacity(options.len() * worlds);
        for world in 0..worlds {
            let hidden = sample_world(map, s, seat, &mut rng, None);
            let stream = rng.next_u64();
            for (option, &action) in options.iter().enumerate() {
                let mut run = GameRun::new(map, hidden, Xoshiro::new(stream));
                run.game.step(map, action as usize, &mut run.rng);
                copies.push(Playout {
                    run,
                    option,
                    world,
                    done: false,
                });
            }
        }
        Lookahead {
            map,
            seat,
            worlds,
            copies,
            table: vec![0.0; options.len() * worlds],
            rows: Vec::new(),
        }
    }

    /// Play every copy on until it ends or needs the network; returns the number of rows that wait
    /// for `feed` (0: the table is complete).
    pub fn prepare(&mut self) -> usize {
        let (map, seat, worlds) = (self.map, self.seat, self.worlds);
        self.rows.clear();
        for (i, copy) in self.copies.iter_mut().enumerate() {
            if copy.done {
                continue;
            }
            copy.run.advance(map, MAX_MOVES, |_| None);
            if copy.run.waiting {
                self.rows.push(i);
            } else {
                self.table[copy.option * worlds + copy.world] = margin(map, &copy.run.game, seat);
                copy.done = true;
            }
        }
        self.rows.len()
    }

    /// The rows that wait for the network (after `prepare`).
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Row `i`'s position, as its player to act sees it.
    pub fn row_obs(&self, i: usize) -> &Obs {
        &self.copies[self.rows[i]].run.obs
    }

    /// The network's answer, `rows` x `n_actions` policy logits: every waiting copy plays its first
    /// choice.
    pub fn feed(&mut self, logits: &[f32]) {
        let map = self.map;
        let na = map.n_actions;
        assert_eq!(logits.len(), self.rows.len() * na, "logits: rows x actions");
        for (j, &i) in self.rows.iter().enumerate() {
            let run = &mut self.copies[i].run;
            let action = first_choice(&logits[j * na..(j + 1) * na], &run.legal(map));
            run.play(map, action);
        }
        self.rows.clear();
    }

    /// (options, worlds): the shape of `table`.
    pub fn shape(&self) -> (usize, usize) {
        (self.table.len() / self.worlds.max(1), self.worlds)
    }

    /// Options x worlds, complete once `prepare` returns 0: the deciding seat's final score less the
    /// best other seat's at the end of each copy.
    pub fn table(&self) -> &[f32] {
        &self.table
    }
}

/// A seat's final score over the best other seat's.
pub(crate) fn margin(map: &Map, s: &State, seat: usize) -> f32 {
    let f = s.finalize(map);
    let best = (0..s.n()).filter(|&q| q != seat).map(|q| f.total[q]).max().unwrap_or(0);
    (f.total[seat] - best) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::GameRng;

    /// A stand-in network: logits that depend on the position alone, so the same position always
    /// gets the same move.
    fn stub(map: &Map, obs: &Obs, phase: f32) -> Vec<f32> {
        let h: f32 = obs.global.iter().sum::<f32>() + obs.players.iter().sum::<f32>() + obs.tickets.iter().sum::<f32>();
        (0..map.n_actions)
            .map(|a| ((h * 0.37 + phase) * (a % 7 + 1) as f32 + a as f32).sin())
            .collect()
    }

    /// A position `moves` random moves into a game.
    fn position(players: usize, seed: u64, moves: usize) -> State {
        let map = Map::usa();
        let mut rng = Xoshiro::new(seed);
        let mut s = State::new(map, players, &mut rng);
        let mut legal = Vec::new();
        for _ in 0..moves {
            s.legal_actions(map, &mut legal);
            if s.is_over() || legal.is_empty() {
                break;
            }
            let a = legal[rng.below(legal.len() as u32) as usize] as usize;
            s.step(map, a, &mut rng);
        }
        s
    }

    /// Run a look-ahead to the end with the stub network; returns its table and the calls made.
    fn run(look: &mut Lookahead, phase: f32) -> (Vec<f32>, usize) {
        let map = Map::usa();
        let mut steps = 0;
        while look.prepare() > 0 {
            let mut logits = Vec::with_capacity(look.rows() * map.n_actions);
            for i in 0..look.rows() {
                logits.extend(stub(map, look.row_obs(i), phase));
            }
            look.feed(&logits);
            steps += 1;
            assert!(steps < 5000, "a copy does not end");
        }
        (look.table().to_vec(), steps)
    }

    #[test]
    fn the_options_are_the_first_choice_and_the_best_of_each_other_kind() {
        let map = Map::usa();
        let tickets = map.tickets_action() as u16;
        let (claim_a, claim_b, face, blind) = (3u16, 7u16, map.draw_action(1) as u16, map.blind_action() as u16);
        let mut lg = vec![-9.0f32; map.n_actions];
        for (a, v) in [
            (claim_a, 1.0),
            (claim_b, 5.0),
            (face, 2.0),
            (blind, 3.0),
            (tickets, 0.5),
        ] {
            lg[a as usize] = v;
        }
        let legal = vec![claim_a, claim_b, face, blind, tickets];
        assert_eq!(look_options(map, &legal, &lg), vec![claim_b, blind, tickets, face]);
        assert_eq!(look_options(map, &[face, blind], &lg), vec![blind, face]);
        assert_eq!(look_options(map, &[face], &lg), vec![face]);
        // six claims and nothing else: the policy's best four
        let claims: Vec<u16> = (10..16).collect();
        for (i, &a) in claims.iter().enumerate() {
            lg[a as usize] = i as f32;
        }
        assert_eq!(look_options(map, &claims, &lg), vec![15, 14, 13, 12]);
    }

    #[test]
    fn a_choice_of_tickets_tries_every_set_of_the_opening_and_the_best_three_of_a_draw() {
        let map = Map::usa();
        let mut lg = vec![0.0f32; map.n_actions];
        let keeps: Vec<u16> = (1..=7u8).map(|m| map.keep_action(m) as u16).collect();
        for (i, &a) in keeps.iter().enumerate() {
            lg[a as usize] = i as f32;
        }
        assert_eq!(
            look_options(map, &keeps, &lg),
            vec![keeps[6], keeps[5], keeps[4]],
            "a draw: seven sets, the best three"
        );
        // the opening keeps two or three of three: masks 3, 5, 6 and 7
        let opening: Vec<u16> = [3u8, 5, 6, 7].iter().map(|&m| map.keep_action(m) as u16).collect();
        assert_eq!(
            look_options(map, &opening, &lg),
            vec![opening[3], opening[2], opening[1], opening[0]]
        );
    }

    #[test]
    fn the_first_option_is_the_first_choice_also_among_equal_logits() {
        let map = Map::usa();
        let lg = vec![1.0f32; map.n_actions];
        let legal: Vec<u16> = vec![2, 9, 40, map.blind_action() as u16, map.tickets_action() as u16];
        assert_eq!(look_options(map, &legal, &lg)[0], first_choice(&lg, &legal));
        let keeps: Vec<u16> = (1..=7u8).map(|m| map.keep_action(m) as u16).collect();
        assert_eq!(look_options(map, &keeps, &lg)[0], first_choice(&lg, &keeps));
    }

    #[test]
    fn a_candidate_is_better_by_points_and_by_its_noise() {
        // the first choice's margins in 4 worlds, then the candidate's in the same worlds
        assert!(
            look_better(&[0.0, 10.0, -5.0, 3.0, 5.0, 15.0, 0.0, 8.0], 4),
            "5 points in every world"
        );
        assert!(
            !look_better(&[0.0, 10.0, -5.0, 3.0, 2.0, 12.0, -3.0, 5.0], 4),
            "2 points in every world: too few"
        );
        assert!(
            !look_better(&[0.0, 10.0, -5.0, 3.0, 40.0, -20.0, 30.0, -20.0], 4),
            "6 points on average, the noise far above"
        );
        assert!(!look_better(&[5.0, 15.0, 0.0, 8.0, 0.0, 10.0, -5.0, 3.0], 4), "worse");
        assert!(look_better(&[1.0, 4.0], 1), "one world: the points alone decide");
    }

    #[test]
    fn a_look_fills_its_table_with_the_margins_of_whole_games() {
        let map = Map::usa();
        for players in 2..=5 {
            let s = position(players, 30 + players as u64, 25);
            let legal = {
                let mut v = Vec::new();
                s.legal_actions(map, &mut v);
                v
            };
            let options: Vec<u16> = legal.iter().cloned().take(3).collect();
            let mut look = Lookahead::new(map, &s, &options, 4, 7);
            assert_eq!(look.shape(), (options.len(), 4));
            let (table, steps) = run(&mut look, 0.2);
            assert_eq!(table.len(), options.len() * 4);
            assert!(
                steps > 10,
                "{players} players: the copies are whole games, {steps} calls"
            );
            assert!(
                table.iter().all(|v| v.abs() < 400.0) && table.iter().any(|&v| v != 0.0),
                "{table:?}"
            );
            assert_eq!(look.prepare(), 0, "nothing waits once the table is complete");
        }
    }

    #[test]
    fn a_look_replays_from_its_seed_and_another_seed_deals_other_worlds() {
        let map = Map::usa();
        let s = position(3, 5, 40);
        let mut legal = Vec::new();
        s.legal_actions(map, &mut legal);
        let options = &legal[..2.min(legal.len())];
        let a = run(&mut Lookahead::new(map, &s, options, 6, 11), 0.4).0;
        let b = run(&mut Lookahead::new(map, &s, options, 6, 11), 0.4).0;
        let c = run(&mut Lookahead::new(map, &s, options, 6, 12), 0.4).0;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn the_copies_of_a_world_share_what_is_hidden_and_differ_in_the_move() {
        let map = Map::usa();
        let s = position(4, 8, 30);
        let me = s.current as usize;
        let mut legal = Vec::new();
        s.legal_actions(map, &mut legal);
        assert!(legal.len() >= 2);
        let options = [legal[0], legal[legal.len() - 1]];
        let look = Lookahead::new(map, &s, &options, 5, 3);
        let mut other_worlds = 0;
        for world in look.copies.chunks(2) {
            let (x, y) = (&world[0].run.game, &world[1].run.game);
            assert_eq!(
                (world[0].world, world[0].option, world[1].option),
                (world[1].world, 0, 1)
            );
            for q in (0..4).filter(|&q| q != me) {
                assert_eq!(
                    x.tickets_held[q], y.tickets_held[q],
                    "the same hidden tickets under both moves"
                );
                assert_eq!(x.hand_size(q), s.hand_size(q), "hand sizes are public");
            }
            assert_eq!(x.moves, s.moves + 1, "the candidate is played");
            other_worlds += (x.tickets_held != look.copies[0].run.game.tickets_held) as usize;
        }
        assert!(other_worlds > 0, "the worlds deal the others' tickets differently");
    }

    #[test]
    fn the_same_candidate_twice_gives_the_same_row() {
        // two copies of one world differ only in the candidate: the same candidate, the same game
        let map = Map::usa();
        let s = position(2, 21, 35);
        let mut legal = Vec::new();
        s.legal_actions(map, &mut legal);
        let (table, _) = run(&mut Lookahead::new(map, &s, &[legal[0], legal[0]], 5, 9), 0.7);
        assert_eq!(table[..5], table[5..]);
        assert!(!look_better(&table, 5));
    }

    /// The last turn of a game - whatever is played, the game ends with it - whose player to act can
    /// claim a route or draw a card; and the claim.
    fn last_turn_with_a_claim(skip: usize) -> (State, u16) {
        let map = Map::usa();
        // short games: the seat that ends one still holds cards, and short routes are still open
        let few_trains = crate::state::Rules {
            trains: 12,
            ..crate::state::Rules::OFFICIAL
        };
        let mut found = 0;
        for seed in 0..2000 {
            let mut rng = Xoshiro::new(seed);
            let mut s = State::with_rules(map, 2, few_trains, &mut rng);
            let mut legal = Vec::new();
            while !s.is_over() {
                s.legal_actions(map, &mut legal);
                if legal.is_empty() {
                    s.pass();
                    continue;
                }
                let claims: Vec<u16> = legal.iter().cloned().filter(|&a| (a as usize) < map.n_claim).collect();
                let last = s.last_round_left == 1 && s.phase == crate::state::Phase::Turn;
                if last && !claims.is_empty() && legal.contains(&(map.blind_action() as u16)) {
                    found += 1;
                    if found > skip {
                        return (s, claims[0]);
                    }
                }
                // claims more often than not, so that the games reach their last round
                let from = if !claims.is_empty() && rng.below(10) < 6 {
                    &claims
                } else {
                    &legal
                };
                let a = from[rng.below(from.len() as u32) as usize] as usize;
                s.step(map, a, &mut rng);
            }
        }
        panic!(
            "{found} games ended on a turn with a claim to make, {} wanted",
            skip + 1
        );
    }

    #[test]
    fn on_the_last_turn_the_table_holds_exactly_what_the_move_changes() {
        // the game ends with this turn, so a claim's copy and a draw's copy of one world differ by the
        // claim alone: by the same points in every world, whatever the others hold there
        let map = Map::usa();
        let worlds = 6;
        for k in 0..8 {
            let (s, claim) = last_turn_with_a_claim(k);
            let me = s.current as usize;
            let mut after = s;
            after.step(map, claim as usize, &mut Xoshiro::new(0));
            let change = margin(map, &after, me) - margin(map, &s, me);
            let options = [map.blind_action() as u16, claim];
            let (table, _) = run(&mut Lookahead::new(map, &s, &options, worlds, 1 + k as u64), 0.3);
            for j in 0..worlds {
                assert_eq!(
                    table[worlds + j] - table[j],
                    change,
                    "position {k}, world {j}: {table:?}"
                );
            }
            assert_eq!(
                look_better(&table, worlds),
                change >= LOOK_POINTS,
                "position {k}: a change of {change}"
            );
        }
    }
}
