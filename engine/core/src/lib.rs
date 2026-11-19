//! Ticket to Ride (base game, USA map) rules engine.
//!
//! - `map`: cities, routes, tickets, and the action table built from them
//! - `state`: the game state and the rules that move it (2 to 5 players), scoring
//! - `rng`: the randomness the rules ask for
//! - `obs`: what one player sees, as the network's input
//! - `sample`: a hidden world consistent with what one player sees
//! - `bots`: baseline players without a network (a ticket builder, a card hoarder, random)
//! - `arena`: matches - many games at once, each seat a network or a bot
//! - `look`: the look-ahead of one decision - candidate moves played out in sampled hidden worlds
//! - `teacher`: the scripted teacher of the first training phases - a target for every position
//! - `selfplay`: self-play for training - many games at once, the trained network's decisions recorded
//! - `view`: what one player sees, as JSON for people (the play server)

// seats, card types, face-up slots and tickets are indexed as the rules number them: the index is
// the meaning, and loops over it read better than iterator chains
#![allow(clippy::needless_range_loop)]

pub mod arena;
pub mod bots;
pub mod look;
pub mod map;
pub mod obs;
pub mod rng;
pub mod sample;
pub mod selfplay;
pub mod state;
pub mod teacher;
pub mod view;

pub use map::Map;
pub use rng::{GameRng, Xoshiro};
pub use state::{FinalScore, Phase, Rules, State};
