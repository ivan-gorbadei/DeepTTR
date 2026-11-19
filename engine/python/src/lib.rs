//! `deepttr._engine`: the rules engine for Python, numpy arrays in and out.
//!
//! ```python
//! from deepttr import _engine as engine
//!
//! game = engine.Game(players=2, seed=7)        # one game: step it, look at it from a seat
//! obs = game.observation(game.current)         # the network's input, one row
//! game.step(engine.first_choice(logits, game.legal()))
//!
//! arena = engine.Arena("net", "hoard", deals=200)   # a match: 200 deals x 2 seatings
//! while arena.prepare():                       # every game on to its next network decision
//!     arena.feed(net(arena.observations()))    # policy logits, one row per waiting game
//! results = arena.results()
//!
//! look = engine.Lookahead(game, options, worlds=8, seed=3)   # candidate moves played to the end
//! while look.prepare():
//!     look.feed(net(look.observations()))
//! margins = look.table()                       # [options, worlds]
//!
//! pool = engine.SelfPlay(games=1024, tables=[2, 3, 4, 5])   # self-play for training
//! while training:
//!     pool.prepare()                           # every game on to its next decision
//!     pool.feed(net(pool.observations()))      # policy logits, one row per waiting decision
//!     records = pool.take_records()            # None, or the ended games' records
//! ```

use deepttr_core::arena::{first_choice as first_of, Arena as Match, ArenaConfig, Player};
use deepttr_core::bots::Bot;
use deepttr_core::look::{look_better as better_of, look_options as options_of, Lookahead as Look};
use deepttr_core::map::N_CARD_TYPES;
use deepttr_core::obs::{dims as obs_dims, Obs, CITY_F, EVENT_I, GLOBAL_F, PLAYER_F, ROUTE_F, RULES_AT, TICKET_F};
use deepttr_core::selfplay::{Opponent, Opponents, Pool, PoolConfig, Record, LOOK_CHECK, LOOK_MOVES, LOOK_WORLDS};
use deepttr_core::state::{EVENTS, MAX_PLAYERS, NONE};
use deepttr_core::{Map, Rules, State, Xoshiro};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn shaped<'py, T: numpy::Element>(py: Python<'py>, v: Vec<T>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let a = PyArray1::from_vec(py, v);
    Ok(a.reshape(shape.to_vec())?.into_any())
}

/// The observations `rows` as a dict of arrays, one row each.
fn obs_dict<'py>(py: Python<'py>, rows: &[&Obs]) -> PyResult<Bound<'py, PyDict>> {
    let map = Map::usa();
    let n = rows.len();
    let floats = |f: &dyn Fn(&Obs) -> &Vec<f32>| -> Vec<f32> {
        let mut v = Vec::new();
        for o in rows {
            v.extend_from_slice(f(o));
        }
        v
    };
    let ints = |f: &dyn Fn(&Obs) -> &Vec<i32>| -> Vec<i32> {
        let mut v = Vec::new();
        for o in rows {
            v.extend_from_slice(f(o));
        }
        v
    };
    let mut legal = Vec::with_capacity(n * map.n_actions);
    for o in rows {
        legal.extend_from_slice(&o.legal);
    }
    let d = PyDict::new(py);
    d.set_item(
        "routes",
        shaped(py, floats(&|o| &o.routes), &[n, map.n_routes(), ROUTE_F])?,
    )?;
    d.set_item(
        "cities",
        shaped(py, floats(&|o| &o.cities), &[n, map.n_cities(), CITY_F])?,
    )?;
    d.set_item(
        "tickets",
        shaped(py, floats(&|o| &o.tickets), &[n, map.n_tickets(), TICKET_F])?,
    )?;
    d.set_item(
        "players",
        shaped(py, floats(&|o| &o.players), &[n, MAX_PLAYERS, PLAYER_F])?,
    )?;
    d.set_item("global", shaped(py, floats(&|o| &o.global), &[n, GLOBAL_F])?)?;
    d.set_item("slots", shaped(py, ints(&|o| &o.slots), &[n, 5])?)?;
    d.set_item("offer", shaped(py, ints(&|o| &o.offer), &[n, 3])?)?;
    d.set_item("events", shaped(py, ints(&|o| &o.events), &[n, EVENTS, EVENT_I])?)?;
    d.set_item("legal", shaped(py, legal, &[n, map.n_actions])?)?;
    Ok(d)
}

/// The rules of a game from a dict of the settings that differ from the official ones:
/// `route_points`, `longest_bonus`, `partial_tickets` (bool), `trains` (3 to 60), `route_price` (0..1).
fn rules_of(rules: Option<&Bound<'_, PyDict>>) -> PyResult<Rules> {
    let mut r = Rules::OFFICIAL;
    let Some(d) = rules else { return Ok(r) };
    for (key, value) in d.iter() {
        match key.extract::<String>()?.as_str() {
            "route_points" => r.route_points = value.extract()?,
            "longest_bonus" => r.longest_bonus = value.extract()?,
            "partial_tickets" => r.partial_tickets = value.extract()?,
            "trains" => {
                r.trains = value.extract()?;
                if !(3..=60).contains(&r.trains) {
                    return Err(PyValueError::new_err("rules: trains 3 to 60"));
                }
            }
            "route_price" => {
                let price: f64 = value.extract()?;
                if !(0.0..=1.0).contains(&price) {
                    return Err(PyValueError::new_err("rules: route_price 0 to 1"));
                }
                r.route_price = (price * 100.0).round() as u8;
            }
            other => return Err(PyValueError::new_err(format!("rules: unknown setting {other:?}"))),
        }
    }
    Ok(r)
}

fn rules_dict<'py>(py: Python<'py>, r: &Rules) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("route_points", r.route_points)?;
    d.set_item("longest_bonus", r.longest_bonus)?;
    d.set_item("partial_tickets", r.partial_tickets)?;
    d.set_item("trains", r.trains)?;
    d.set_item("route_price", r.route_price as f64 / 100.0)?;
    Ok(d)
}

fn check_players(players: usize) -> PyResult<()> {
    if (2..=MAX_PLAYERS).contains(&players) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("players must be 2..={MAX_PLAYERS}")))
    }
}

/// One game: the deal from `seed`, moves by `step`, and what each seat sees. Everything random
/// after the deal comes from the same seed, so (seed, moves) replays the game.
#[pyclass]
struct Game {
    state: State,
    rng: Xoshiro,
    seed: u64,
    moves: Vec<(u8, i32)>,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (players=2, seed=1, rules=None))]
    fn new(players: usize, seed: u64, rules: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        check_players(players)?;
        let mut rng = Xoshiro::new(seed);
        let state = State::with_rules(Map::usa(), players, rules_of(rules)?, &mut rng);
        Ok(Game {
            state,
            rng,
            seed,
            moves: Vec::new(),
        })
    }

    /// The rules the game is played by.
    #[getter]
    fn rules<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        rules_dict(py, &self.state.rules)
    }

    /// The network's input for `seat` (one row; its legal moves are marked only when it is to act).
    fn observation<'py>(&self, py: Python<'py>, seat: usize) -> PyResult<Bound<'py, PyDict>> {
        self.check_seat(seat)?;
        obs_dict(py, &[&Obs::of(Map::usa(), &self.state, seat)])
    }

    /// The table as `seat` sees it (JSON).
    fn view(&self, seat: usize) -> PyResult<String> {
        self.check_seat(seat)?;
        Ok(deepttr_core::view::view_json(Map::usa(), &self.state, seat).to_string())
    }

    /// The legal moves of the player to act (empty: that player must pass, or the game is over).
    fn legal(&self) -> Vec<u16> {
        let mut v = Vec::new();
        self.state.legal_actions(Map::usa(), &mut v);
        v
    }

    #[getter]
    fn current(&self) -> u8 {
        self.state.current
    }

    #[getter]
    fn players(&self) -> usize {
        self.state.n()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    fn is_over(&self) -> bool {
        self.state.is_over()
    }

    /// Play a legal move of the player to act; `-1` passes (only when nothing is legal).
    fn step(&mut self, action: i32) -> PyResult<()> {
        if self.state.is_over() {
            return Err(PyValueError::new_err("the game is over"));
        }
        let legal = self.legal();
        let seat = self.state.current;
        if action < 0 {
            if !legal.is_empty() {
                return Err(PyValueError::new_err("a pass while a move is legal"));
            }
            self.state.pass();
        } else {
            if action > u16::MAX as i32 || !legal.contains(&(action as u16)) {
                return Err(PyValueError::new_err(format!("illegal move {action}")));
            }
            self.state.step(Map::usa(), action as usize, &mut self.rng);
        }
        self.moves.push((seat, action));
        Ok(())
    }

    /// Every move so far: (seat, action), -1 = a pass.
    fn moves(&self) -> Vec<(u8, i32)> {
        self.moves.clone()
    }

    /// A move in words.
    fn describe(&self, action: usize) -> String {
        Map::usa().describe(action)
    }

    /// The move a baseline bot (heuristic, hoard, random) would play now for the player to act
    /// (-1: nothing is legal). The bots read the full state.
    #[pyo3(signature = (kind, seed=1))]
    fn bot_move(&self, kind: &str, seed: u64) -> PyResult<i32> {
        let bot = Bot::parse(kind).ok_or_else(|| PyValueError::new_err(format!("unknown bot {kind}")))?;
        let legal = self.legal();
        if legal.is_empty() || self.state.is_over() {
            return Ok(-1);
        }
        Ok(bot.choose(Map::usa(), &self.state, &legal, &mut Xoshiro::new(seed)) as i32)
    }

    /// A copy of the game as it stands.
    fn copy(&self) -> Game {
        Game {
            state: self.state,
            rng: self.rng.clone(),
            seed: self.seed,
            moves: self.moves.clone(),
        }
    }
}

impl Game {
    fn check_seat(&self, seat: usize) -> PyResult<()> {
        if seat < self.state.n() {
            Ok(())
        } else {
            Err(PyValueError::new_err("no such seat"))
        }
    }
}

/// A match of contestant `a` against contestant `b` (`deepttr_core::arena`): each of `deals` deals
/// is played `players` times, `a` in each seat in turn and `b` in all the others. A contestant is
/// `"net"` (a network, evaluated by the caller) or a bot's name.
#[pyclass]
struct Arena {
    arena: Match,
    threads: rayon::ThreadPool,
}

#[pymethods]
impl Arena {
    #[new]
    #[pyo3(signature = (a, b, deals, players=2, seed=1, first_deal=0, threads=0, rules=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        a: &str,
        b: &str,
        deals: usize,
        players: usize,
        seed: u64,
        first_deal: u64,
        threads: usize,
        rules: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        check_players(players)?;
        if deals == 0 {
            return Err(PyValueError::new_err("deals must be > 0"));
        }
        let side = |name: &str| {
            Player::parse(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "a contestant is \"net\" or a bot (heuristic, hoard, random), not {name:?}"
                ))
            })
        };
        let cfg = ArenaConfig {
            players,
            deals,
            first_deal,
            seed,
            a: side(a)?,
            b: side(b)?,
            rules: rules_of(rules)?,
        };
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("threads: {e}")))?;
        Ok(Arena {
            arena: Match::new(Map::usa(), cfg),
            threads,
        })
    }

    /// The games of the match: deals x players.
    #[getter]
    fn games(&self) -> usize {
        self.arena.games()
    }

    /// Play every game on until it ends or a network must choose (the GIL is released meanwhile);
    /// returns the number of rows waiting for `feed` - 0 when every game is over.
    fn prepare(&mut self, py: Python<'_>) -> usize {
        let Arena { arena, threads } = self;
        py.detach(|| threads.install(|| arena.prepare()))
    }

    /// The waiting positions as the network's input: a dict of arrays, one row each.
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows: Vec<&Obs> = (0..self.arena.rows()).map(|i| self.arena.row_obs(i)).collect();
        obs_dict(py, &rows)
    }

    /// Per waiting row, whose network answers it: 0 contestant a's, 1 contestant b's.
    fn sides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v: Vec<u8> = (0..self.arena.rows()).map(|i| self.arena.row_side(i)).collect();
        shaped(py, v, &[self.arena.rows()])
    }

    /// The policy logits of the waiting rows, [rows, actions] float32: every waiting game plays
    /// its first choice.
    fn feed(&mut self, logits: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let (rows, na) = (self.arena.rows(), Map::usa().n_actions);
        if logits.shape() != [rows, na] {
            return Err(PyValueError::new_err(format!(
                "logits must be [{rows}, {na}], got {:?}",
                logits.shape()
            )));
        }
        let l = logits
            .as_slice()
            .map_err(|_| PyValueError::new_err("logits must be C-contiguous"))?;
        self.arena.feed(l);
        Ok(())
    }

    /// The moves the caller chose for the waiting rows, one each, in place of `feed`: for a
    /// contestant that decides by more than its first choice.
    fn play(&mut self, moves: Vec<u16>) -> PyResult<()> {
        self.arena.play(&moves).map_err(PyValueError::new_err)
    }

    /// The look-ahead of waiting row `row`'s decision over `options`, `worlds` copies each. Its
    /// seed comes from the game's own stream, so a match with look-aheads replays from its seed;
    /// `stage` 1 gives a second look at the same decision other worlds.
    #[pyo3(signature = (row, options, worlds, stage=0))]
    fn look(&self, row: usize, options: Vec<u16>, worlds: usize, stage: u64) -> PyResult<Lookahead> {
        if row >= self.arena.rows() {
            return Err(PyValueError::new_err(format!("no waiting row {row}")));
        }
        let legal: Vec<u16> = (0..Map::usa().n_actions as u16)
            .filter(|&a| self.arena.row_obs(row).legal[a as usize] == 1)
            .collect();
        check_look(&legal, &options, worlds)?;
        Ok(Lookahead {
            look: self.arena.look(row, &options, worlds, stage),
        })
    }

    #[getter]
    fn done(&self) -> bool {
        self.arena.is_done()
    }

    /// The finished games as a dict of arrays, a row a game, in game order (a deal's seatings
    /// together). Per game: `deal`, `a_seat`, `turns`; per game and absolute seat (seats past the
    /// table's stay 0): `utility`, `total`, `route` (route points before the route price),
    /// `tickets`, `trail` (the longest trail's length), `completed`, `rank` (0 = first), `held`
    /// (tickets at the end), `ticket_draws`, `trains`, `blind_draws`, `face_up_draws`; and
    /// `route_owner` [games, routes], the seat that claimed each route (-1: nobody).
    fn results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let res = self.arena.results();
        let n = res.len();
        let nr = Map::usa().n_routes();
        let d = PyDict::new(py);
        d.set_item(
            "deal",
            shaped(py, res.iter().map(|g| g.deal).collect::<Vec<u64>>(), &[n])?,
        )?;
        d.set_item(
            "a_seat",
            shaped(py, res.iter().map(|g| g.a_seat).collect::<Vec<u8>>(), &[n])?,
        )?;
        d.set_item(
            "turns",
            shaped(py, res.iter().map(|g| g.turns as i32).collect::<Vec<i32>>(), &[n])?,
        )?;
        let seats_f = |f: &dyn Fn(&deepttr_core::arena::GameResult) -> [f32; MAX_PLAYERS]| -> Vec<f32> {
            res.iter().flat_map(f).collect()
        };
        let seats_i = |f: &dyn Fn(&deepttr_core::arena::GameResult, usize) -> i32| -> Vec<i32> {
            res.iter()
                .flat_map(|g| (0..MAX_PLAYERS).map(move |q| f(g, q)))
                .collect()
        };
        let per_seat = [n, MAX_PLAYERS];
        d.set_item("utility", shaped(py, seats_f(&|g| g.utility), &per_seat)?)?;
        d.set_item("rank", shaped(py, seats_f(&|g| g.score.ranks), &per_seat)?)?;
        d.set_item(
            "total",
            shaped(py, seats_i(&|g, q| g.score.total[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "route",
            shaped(py, seats_i(&|g, q| g.score.route[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "tickets",
            shaped(py, seats_i(&|g, q| g.score.tickets[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "trail",
            shaped(py, seats_i(&|g, q| g.score.longest[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "completed",
            shaped(py, seats_i(&|g, q| g.score.completed[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "held",
            shaped(py, seats_i(&|g, q| g.tickets_held[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "ticket_draws",
            shaped(py, seats_i(&|g, q| g.ticket_draws[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "trains",
            shaped(py, seats_i(&|g, q| g.trains_left[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "blind_draws",
            shaped(py, seats_i(&|g, q| g.blind_draws[q] as i32), &per_seat)?,
        )?;
        d.set_item(
            "face_up_draws",
            shaped(py, seats_i(&|g, q| g.face_up_draws[q] as i32), &per_seat)?,
        )?;
        let owner: Vec<i8> = res
            .iter()
            .flat_map(|g| g.route_owner.iter().map(|&o| if o == NONE { -1 } else { o as i8 }))
            .collect();
        d.set_item("route_owner", shaped(py, owner, &[n, nr])?)?;
        Ok(d)
    }
}

fn check_look(legal: &[u16], options: &[u16], worlds: usize) -> PyResult<()> {
    if options.is_empty() || worlds == 0 || options.iter().any(|a| !legal.contains(a)) {
        return Err(PyValueError::new_err(
            "a look-ahead needs legal moves of the player to act and at least one world",
        ));
    }
    Ok(())
}

/// The look-ahead of the decision `game` stands at (`deepttr_core::look`): each move of `options`
/// is played in `worlds` copies of the game, the hidden cards and tickets dealt afresh from what
/// the seat to act sees, and every copy is played to the end by the network's first choice at
/// every seat. `table()` is then [options, worlds]: the deciding seat's final score less the best
/// other seat's. The game itself is not touched.
#[pyclass]
struct Lookahead {
    look: Look,
}

#[pymethods]
impl Lookahead {
    #[new]
    #[pyo3(signature = (game, options, worlds, seed=1))]
    fn new(game: &Game, options: Vec<u16>, worlds: usize, seed: u64) -> PyResult<Self> {
        check_look(&game.legal(), &options, worlds)?;
        Ok(Lookahead {
            look: Look::new(Map::usa(), &game.state, &options, worlds, seed),
        })
    }

    /// Play every copy on until it ends or needs the network (the GIL is released meanwhile);
    /// returns the number of rows waiting for `feed` - 0 when the table is complete.
    fn prepare(&mut self, py: Python<'_>) -> usize {
        let look = &mut self.look;
        py.detach(|| look.prepare())
    }

    /// The waiting positions as the network's input: a dict of arrays, one row each.
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows: Vec<&Obs> = (0..self.look.rows()).map(|i| self.look.row_obs(i)).collect();
        obs_dict(py, &rows)
    }

    /// The policy logits of the waiting rows, [rows, actions] float32: every waiting copy plays
    /// its first choice.
    fn feed(&mut self, logits: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let (rows, na) = (self.look.rows(), Map::usa().n_actions);
        if logits.shape() != [rows, na] {
            return Err(PyValueError::new_err(format!(
                "logits must be [{rows}, {na}], got {:?}",
                logits.shape()
            )));
        }
        let l = logits
            .as_slice()
            .map_err(|_| PyValueError::new_err("logits must be C-contiguous"))?;
        self.look.feed(l);
        Ok(())
    }

    /// [options, worlds] float32, complete once `prepare` returns 0.
    fn table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (options, worlds) = self.look.shape();
        shaped(py, self.look.table().to_vec(), &[options, worlds])
    }
}

/// Self-play for training (`deepttr_core::selfplay`): many games at once, each decision of the
/// network being trained a record with its target - the scripted teacher's (`teacher`), the move
/// played, or a lesson from copies of the game (`ticket_playouts`, `lookahead`).
///
/// The opponents: none (self-play), a bot at every other seat (`bot`), one frozen network
/// (`frozen=True`), or a mixed table (`mixed`: [(name, weight)], a bot's name or "net:K" for the
/// caller's frozen network K), each other seat drawn by weight for every game, the card hoarder at
/// one seat of a table of three or more in half the games. A pool with opponents plays duplicate
/// deals, unless it looks ahead. Self-play draws each game's table from `tables` (else `players`);
/// `rules` as `Game` takes them. `joint_paths`: the teacher plans one set of ways for all the open
/// tickets. `look_check_keep`: the fresh worlds of the look-ahead's check of a choice of tickets
/// (24, as any other check's, unless given). `look_forecast`: the move forecast (`forecast_rows`,
/// `feed_forecast`) sends moves to the check, and the records keep the copies. `playout_slots`: the
/// copies under way at most.
#[pyclass]
struct SelfPlay {
    pool: Pool,
    threads: rayon::ThreadPool,
}

#[pymethods]
impl SelfPlay {
    #[new]
    #[pyo3(signature = (games, players=2, tables=None, rules=None, seed=1, threads=0, bot=None, frozen=false, mixed=None,
        teacher=false, joint_paths=false, ticket_playouts=false, lookahead=false, look_check_keep=LOOK_CHECK,
        look_forecast=false, playout_slots=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        games: usize,
        players: usize,
        tables: Option<Vec<usize>>,
        rules: Option<&Bound<'_, PyDict>>,
        seed: u64,
        threads: usize,
        bot: Option<&str>,
        frozen: bool,
        mixed: Option<Vec<(String, f32)>>,
        teacher: bool,
        joint_paths: bool,
        ticket_playouts: bool,
        lookahead: bool,
        look_check_keep: usize,
        look_forecast: bool,
        playout_slots: usize,
    ) -> PyResult<Self> {
        let bot_of =
            |name: &str| Bot::parse(name).ok_or_else(|| PyValueError::new_err(format!("unknown bot {name:?}")));
        let opponents = match (bot, frozen, mixed) {
            (Some(_), true, _) | (Some(_), _, Some(_)) => {
                return Err(PyValueError::new_err(
                    "the other seats: a bot, a frozen network or a mixed table",
                ))
            }
            (Some(name), false, None) => Opponents::Bot(bot_of(name)?),
            (None, _, Some(list)) => {
                let mut out = Vec::with_capacity(list.len());
                for (name, w) in list {
                    let o = match name.strip_prefix("net:") {
                        Some(k) => Opponent::Net(k.parse().map_err(|_| {
                            PyValueError::new_err(format!("mixed: a frozen network is net:K, not {name:?}"))
                        })?),
                        None => Opponent::Bot(bot_of(&name)?),
                    };
                    out.push((o, w));
                }
                Opponents::Mixed(out)
            }
            (None, true, None) => Opponents::Net,
            (None, false, None) => Opponents::None,
        };
        let cfg = PoolConfig {
            games,
            players,
            tables: tables.unwrap_or_default(),
            rules: rules_of(rules)?,
            seed,
            opponents,
            teacher,
            joint_paths,
            ticket_playouts,
            lookahead,
            look_check_keep,
            look_forecast,
            playout_slots,
        };
        cfg.check().map_err(PyValueError::new_err)?;
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("threads: {e}")))?;
        Ok(SelfPlay {
            pool: Pool::new(Map::usa(), cfg),
            threads,
        })
    }

    #[getter]
    fn games(&self) -> usize {
        self.pool.games()
    }

    /// Play every game on until the network must choose or the game waits for its copies, and
    /// every copy until its network must choose (the GIL is released meanwhile); returns the
    /// number of rows waiting for `feed`: the games' decisions in game order, then the copies'.
    fn prepare(&mut self, py: Python<'_>) -> usize {
        let SelfPlay { pool, threads } = self;
        py.detach(|| threads.install(|| pool.prepare()))
    }

    /// The waiting positions as the network's input: a dict of arrays, one row each.
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows: Vec<&Obs> = (0..self.pool.rows()).map(|i| self.pool.row_obs(i)).collect();
        obs_dict(py, &rows)
    }

    /// The seat to act at each waiting row.
    fn decider_seats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v: Vec<u8> = (0..self.pool.rows()).map(|i| self.pool.decider_seat(i)).collect();
        shaped(py, v, &[self.pool.rows()])
    }

    /// The network that answers each waiting row: 0 the one being trained, k + 1 frozen network k
    /// of a mixed table, 1 the frozen network of a pool against one.
    fn row_nets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v: Vec<u8> = (0..self.pool.rows()).map(|i| self.pool.row_net(i)).collect();
        shaped(py, v, &[self.pool.rows()])
    }

    /// With `look_forecast`: the rows of the trained network's decisions in the games, whose
    /// forecast of every move `feed_forecast` takes before `feed`.
    fn forecast_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v: Vec<u32> = self.pool.forecast_rows().into_iter().map(|i| i as u32).collect();
        let n = v.len();
        shaped(py, v, &[n])
    }

    /// The move forecast of `rows` (rows of `forecast_rows`), [len(rows), actions] float32: the
    /// margin each move would end the game with, in hundreds of points.
    fn feed_forecast(&mut self, rows: Vec<usize>, moves: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let na = Map::usa().n_actions;
        if moves.shape() != [rows.len(), na] {
            return Err(PyValueError::new_err(format!(
                "the move forecast must be [{}, {na}], got {:?}",
                rows.len(),
                moves.shape()
            )));
        }
        if rows.iter().any(|&i| i >= self.pool.rows()) {
            return Err(PyValueError::new_err("a row past this step's"));
        }
        let m = moves
            .as_slice()
            .map_err(|_| PyValueError::new_err("the move forecast must be C-contiguous"))?;
        self.pool.feed_forecast(&rows, m);
        Ok(())
    }

    /// The networks' answers for the waiting rows: policy logits [rows, actions] float32, and
    /// optionally the forecast for each row's player to act [rows] float32, which the records keep
    /// (`root_forecast`; without it NaN).
    #[pyo3(signature = (logits, forecast=None))]
    fn feed(
        &mut self,
        py: Python<'_>,
        logits: PyReadonlyArray2<'_, f32>,
        forecast: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<()> {
        let (rows, na) = (self.pool.rows(), Map::usa().n_actions);
        if logits.shape() != [rows, na] {
            return Err(PyValueError::new_err(format!(
                "logits must be [{rows}, {na}], got {:?}",
                logits.shape()
            )));
        }
        let l = logits
            .as_slice()
            .map_err(|_| PyValueError::new_err("logits must be C-contiguous"))?;
        let f = match &forecast {
            Some(f) if f.shape() != [rows] => {
                return Err(PyValueError::new_err(format!(
                    "the forecast must be [{rows}], got {:?}",
                    f.shape()
                )))
            }
            Some(f) => Some(
                f.as_slice()
                    .map_err(|_| PyValueError::new_err("the forecast must be C-contiguous"))?,
            ),
            None => None,
        };
        let SelfPlay { pool, threads } = self;
        py.detach(|| threads.install(|| pool.feed(l, f)));
        Ok(())
    }

    /// The records of the games that ended since the last call, a row a record, or None: the
    /// position (the keys of `observations`), `policy` (the target), `n_players`, `game`, `moves`,
    /// `lesson` (1: a lesson from copies), `root_prob`, `root_forecast`, `advantage` (the move's
    /// reward, hundreds of points), and the outcome per seat relative to the deciding one: `value`,
    /// `score` (/ 100), `route_owner` (-1 nobody), `opp_tickets`, `ticket_outcome`, `ticket_start`,
    /// `trail`, `trail_best`, `hands`, `path_cost`. With `look_forecast`, `look` [rows, 5] (-1 past
    /// the moves tried) and `look_table` [rows, 5, 32] (NaN where no copy).
    fn take_records<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let recs = self.pool.take_records();
        if recs.is_empty() {
            return Ok(None);
        }
        let looks = self.pool.config().look_forecast.then_some(LOOK_WORLDS + LOOK_CHECK);
        Ok(Some(records_dict(py, &recs, looks)?))
    }

    /// The route price, 0 to 1, of the games started from now on.
    fn set_route_price(&mut self, price: f32) -> PyResult<()> {
        if !(0.0..=1.0).contains(&price) {
            return Err(PyValueError::new_err("the route price is 0 to 1"));
        }
        self.pool.set_route_price(price);
        Ok(())
    }

    /// The pool's counters since it was made: games, decisions and rows, the copies played and
    /// their rows, the lessons, and the look-ahead's looks, checks and proven moves.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let st = self.pool.stats();
        let d = PyDict::new(py);
        d.set_item("games_finished", st.games_finished)?;
        d.set_item("decisions", st.decisions)?;
        d.set_item("evals", st.evals)?;
        d.set_item("moves_per_game", st.moves_per_game)?;
        d.set_item("playouts", st.playouts)?;
        d.set_item("playout_evals", st.playout_evals)?;
        d.set_item("ticket_lessons", st.ticket_lessons)?;
        d.set_item("ticket_skipped", st.ticket_skipped)?;
        d.set_item("early_lessons", st.early_lessons)?;
        for (k, name) in ["lessons_opening", "lessons_keep", "lessons_draw", "lessons_early"]
            .iter()
            .enumerate()
        {
            d.set_item(*name, st.lessons_by_kind[k])?;
        }
        d.set_item("look_tried", st.look_tried)?;
        d.set_item("look_checked", st.look_checked)?;
        d.set_item("look_proven", st.look_proven)?;
        d.set_item("forecast_flagged", st.forecast_flagged)?;
        d.set_item("forecast_proven", st.forecast_proven)?;
        d.set_item("forecast_keeps", st.forecast_keeps)?;
        d.set_item("keep_checked", st.keep_checked)?;
        d.set_item("keep_proven", st.keep_proven)?;
        Ok(d)
    }
}

/// Records as a dict of arrays, a row a record; `looks`: the worlds a move of `look_table`.
fn records_dict<'py>(py: Python<'py>, recs: &[Record], looks: Option<usize>) -> PyResult<Bound<'py, PyDict>> {
    let map = Map::usa();
    let (n, na, nt, nr) = (recs.len(), map.n_actions, map.n_tickets(), map.n_routes());
    let rows: Vec<&Obs> = recs.iter().map(|r| &r.obs).collect();
    let d = obs_dict(py, &rows)?;
    let floats =
        |f: &dyn Fn(&Record) -> &[f32]| -> Vec<f32> { recs.iter().flat_map(|r| f(r).iter().copied()).collect() };
    let seats = [n, MAX_PLAYERS];
    d.set_item("policy", shaped(py, floats(&|r| &r.policy), &[n, na])?)?;
    d.set_item("value", shaped(py, floats(&|r| &r.value), &seats)?)?;
    d.set_item("score", shaped(py, floats(&|r| &r.score), &seats)?)?;
    d.set_item("trail", shaped(py, floats(&|r| &r.trail), &seats)?)?;
    let one = |f: &dyn Fn(&Record) -> f32| -> Vec<f32> { recs.iter().map(f).collect() };
    d.set_item("root_forecast", shaped(py, one(&|r| r.root_forecast), &[n])?)?;
    d.set_item("root_prob", shaped(py, one(&|r| r.root_prob), &[n])?)?;
    d.set_item("advantage", shaped(py, one(&|r| r.advantage), &[n])?)?;
    let owner: Vec<i8> = recs.iter().flat_map(|r| r.route_owner.iter().copied()).collect();
    d.set_item("route_owner", shaped(py, owner, &[n, nr])?)?;
    let tickets: Vec<u8> = recs
        .iter()
        .flat_map(|r| {
            r.tickets
                .iter()
                .flat_map(|&held| (0..nt).map(move |t| ((held >> t) & 1) as u8))
        })
        .collect();
    d.set_item("opp_tickets", shaped(py, tickets, &[n, MAX_PLAYERS, nt])?)?;
    let outcome: Vec<i8> = recs.iter().flat_map(|r| r.ticket_outcome.iter().copied()).collect();
    d.set_item("ticket_outcome", shaped(py, outcome, &[n, nt])?)?;
    let start: Vec<u8> = recs.iter().flat_map(|r| r.ticket_start.iter().copied()).collect();
    d.set_item("ticket_start", shaped(py, start, &[n, nt])?)?;
    let best: Vec<u8> = recs.iter().flat_map(|r| r.trail_best.iter().copied()).collect();
    d.set_item("trail_best", shaped(py, best, &seats)?)?;
    let hands: Vec<u8> = recs.iter().flat_map(|r| r.hands.iter().flatten().copied()).collect();
    d.set_item("hands", shaped(py, hands, &[n, MAX_PLAYERS, N_CARD_TYPES])?)?;
    let cost: Vec<u8> = recs.iter().flat_map(|r| r.path_cost.iter().copied()).collect();
    d.set_item("path_cost", shaped(py, cost, &[n, MAX_PLAYERS, nt])?)?;
    d.set_item(
        "n_players",
        shaped(py, recs.iter().map(|r| r.n_players).collect::<Vec<u8>>(), &[n])?,
    )?;
    d.set_item(
        "game",
        shaped(py, recs.iter().map(|r| r.game).collect::<Vec<u64>>(), &[n])?,
    )?;
    d.set_item(
        "moves",
        shaped(py, recs.iter().map(|r| r.moves).collect::<Vec<u16>>(), &[n])?,
    )?;
    d.set_item(
        "lesson",
        shaped(py, recs.iter().map(|r| r.lesson).collect::<Vec<u8>>(), &[n])?,
    )?;
    if let Some(w) = looks {
        let k = LOOK_MOVES;
        let mut look = vec![-1i16; n * k];
        let mut table = vec![f32::NAN; n * k * w];
        for (i, r) in recs.iter().enumerate().filter(|(_, r)| !r.look.is_empty()) {
            let full = r.look_table.len() / r.look.len();
            let rw = full.min(w);
            for (o, &a) in r.look.iter().enumerate().take(k) {
                look[i * k + o] = a as i16;
                let at = (i * k + o) * w;
                table[at..at + rw].copy_from_slice(&r.look_table[o * full..o * full + rw]);
            }
        }
        d.set_item("look", shaped(py, look, &[n, k])?)?;
        d.set_item("look_table", shaped(py, table, &[n, k, w])?)?;
    }
    Ok(d)
}

/// The moves a look-ahead tries at a decision among `legal`, the first choice first
/// (`deepttr_core::look::look_options`). `logits` covers every action id.
#[pyfunction]
fn look_options(logits: PyReadonlyArray1<'_, f32>, legal: Vec<u16>) -> PyResult<Vec<u16>> {
    let l = logits
        .as_slice()
        .map_err(|_| PyValueError::new_err("logits must be C-contiguous"))?;
    let na = Map::usa().n_actions;
    if l.len() != na || legal.is_empty() || legal.iter().any(|&a| a as usize >= na) {
        return Err(PyValueError::new_err(format!(
            "look_options: {na} logits and at least one legal move"
        )));
    }
    Ok(options_of(Map::usa(), &legal, l))
}

/// The verdict of a look-ahead's second look: `table` is [2, worlds], the first choice's margins
/// and the candidate's in the same worlds. True when the candidate is better by at least 3 points
/// on average and by 2 standard errors of the paired differences.
#[pyfunction]
fn look_better(table: PyReadonlyArray2<'_, f32>) -> PyResult<bool> {
    if table.shape()[0] != 2 || table.shape()[1] == 0 {
        return Err(PyValueError::new_err("look_better: a table of 2 x worlds"));
    }
    let worlds = table.shape()[1];
    let t = table
        .as_slice()
        .map_err(|_| PyValueError::new_err("the table must be C-contiguous"))?;
    Ok(better_of(t, worlds))
}

/// The first choice of a policy among `legal`: the move with the highest logit (of equal ones, the
/// highest action id). `logits` covers every action id.
#[pyfunction]
fn first_choice(logits: PyReadonlyArray1<'_, f32>, legal: Vec<u16>) -> PyResult<u16> {
    let l = logits
        .as_slice()
        .map_err(|_| PyValueError::new_err("logits must be C-contiguous"))?;
    let na = Map::usa().n_actions;
    if l.len() != na || legal.is_empty() || legal.iter().any(|&a| a as usize >= na) {
        return Err(PyValueError::new_err(format!(
            "first_choice: {na} logits and at least one legal move"
        )));
    }
    Ok(first_of(l, &legal))
}

/// The static board (JSON): cities and where to draw them, routes, tickets, the claim table, the
/// action ids.
#[pyfunction]
fn map_json() -> String {
    let mut board = deepttr_core::view::map_json(Map::usa());
    board["layout"] = deepttr_core::view::usa_layout();
    board.to_string()
}

/// The sizes of one observation and of the action space, and the static facts of the map a network
/// is built from: which cities each route joins, each ticket's ends, the route and the colour of
/// every claim action.
#[pyfunction]
fn dims<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let map = Map::usa();
    let d = PyDict::new(py);
    d.set_item("n_actions", map.n_actions)?;
    d.set_item("n_claim", map.n_claim)?;
    d.set_item("n_routes", map.n_routes())?;
    d.set_item("n_cities", map.n_cities())?;
    d.set_item("n_tickets", map.n_tickets())?;
    d.set_item("max_players", MAX_PLAYERS)?;
    d.set_item("events", EVENTS)?;
    d.set_item("route_f", ROUTE_F)?;
    d.set_item("city_f", CITY_F)?;
    d.set_item("ticket_f", TICKET_F)?;
    d.set_item("player_f", PLAYER_F)?;
    d.set_item("global_f", GLOBAL_F)?;
    d.set_item("event_i", EVENT_I)?;
    d.set_item("legal", obs_dims(map).legal)?;
    d.set_item("card_types", N_CARD_TYPES)?;
    d.set_item("rules_at", RULES_AT)?;
    d.set_item("route_a", map.routes.iter().map(|r| r.a as i64).collect::<Vec<i64>>())?;
    d.set_item("route_b", map.routes.iter().map(|r| r.b as i64).collect::<Vec<i64>>())?;
    d.set_item(
        "route_pair",
        map.routes.iter().map(|r| r.pair as i64).collect::<Vec<i64>>(),
    )?;
    d.set_item("ticket_a", map.tickets.iter().map(|t| t.a as i64).collect::<Vec<i64>>())?;
    d.set_item("ticket_b", map.tickets.iter().map(|t| t.b as i64).collect::<Vec<i64>>())?;
    d.set_item(
        "claim_route",
        map.claims.iter().map(|c| c.route as i64).collect::<Vec<i64>>(),
    )?;
    d.set_item(
        "claim_color",
        map.claims.iter().map(|c| c.color as i64).collect::<Vec<i64>>(),
    )?;
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<Arena>()?;
    m.add_class::<Lookahead>()?;
    m.add_class::<SelfPlay>()?;
    m.add_function(wrap_pyfunction!(first_choice, m)?)?;
    m.add_function(wrap_pyfunction!(look_options, m)?)?;
    m.add_function(wrap_pyfunction!(look_better, m)?)?;
    m.add_function(wrap_pyfunction!(map_json, m)?)?;
    m.add_function(wrap_pyfunction!(dims, m)?)?;
    Ok(())
}
