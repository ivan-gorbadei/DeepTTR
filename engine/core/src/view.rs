//! What one player sees, for people: the facts of `obs` (nothing hidden from `me` is shown), as
//! JSON for the play server's page; at the end of the game every hand of tickets is shown, as at the
//! table. `map_json` is the static board: cities, routes, tickets, the claim table, action ids;
//! `usa_layout` is where the page draws the USA map's cities.

use crate::map::{Map, COLORS, N_CARD_TYPES, ROUTE_SCORE};
use crate::obs::{offered_to, opening_chosen, visible_ticket_count, visible_ticket_deck};
use crate::state::{Phase, State, NONE};
use serde_json::{json, Value};

/// The latitude and longitude of every city of the USA map, in the map's city order: the play page
/// projects them to draw the board (public geography; no artwork of the published game).
pub fn usa_layout() -> Value {
    let file: Value = serde_json::from_str(include_str!("../../../data/maps/usa/layout.json"))
        .expect("the embedded layout is valid JSON");
    Value::Array(
        Map::usa()
            .cities
            .iter()
            .map(|c| file["cities"][c.as_str()].clone())
            .collect(),
    )
}

pub const CARD_NAMES: [&str; N_CARD_TYPES] = [
    "black",
    "blue",
    "green",
    "orange",
    "pink",
    "red",
    "white",
    "yellow",
    "locomotive",
];

pub fn map_json(map: &Map) -> Value {
    json!({
        "id": map.id,
        "cities": map.cities,
        "routes": map.routes.iter().map(|r| json!({
            "a": r.a, "b": r.b, "color": r.color.map(|c| COLORS[c as usize]), "length": r.length,
            "pair": r.pair, "points": ROUTE_SCORE[r.length as usize],
        })).collect::<Vec<_>>(),
        "tickets": map.tickets.iter().map(|t| json!({"a": t.a, "b": t.b, "points": t.points})).collect::<Vec<_>>(),
        "claims": map.claims.iter().map(|c| json!([c.route, c.color])).collect::<Vec<_>>(),
        "cards": CARD_NAMES,
        "actions": {
            "n_claim": map.n_claim, "draw0": map.draw_action(0), "blind": map.blind_action(),
            "tickets": map.tickets_action(), "keep1": map.keep_action(1), "n": map.n_actions,
        },
        "both_doubles_min_players": map.both_doubles_min_players,
    })
}

fn tickets_of(map: &Map, s: &State, q: usize) -> Vec<Value> {
    let comp = s.components(map, q);
    (0..map.n_tickets())
        .filter(|&t| s.tickets_held[q] >> t & 1 == 1)
        .map(|t| {
            let tk = &map.tickets[t];
            json!({"id": t, "a": tk.a, "b": tk.b, "points": tk.points, "done": comp[tk.a as usize] == comp[tk.b as usize]})
        })
        .collect()
}

/// The table as player `me` sees it.
pub fn view_json(map: &Map, s: &State, me: usize) -> Value {
    let n = s.n();
    let phase = match s.phase {
        Phase::Turn => "turn",
        Phase::Draw2 => "draw2",
        Phase::Keep => "keep",
        Phase::Over => "over",
    };
    let (offer, n_offer) = offered_to(s, me);
    let seats: Vec<Value> = (0..n)
        .map(|q| {
            json!({
                "trains": s.trains[q], "route_points": s.route_points[q], "cards": s.hand_size(q),
                "tickets": visible_ticket_count(s, me, q), "known": s.known[q],
                "longest": s.longest_trail(map, q), "face_up_taken": s.face_up_taken[q],
                "paid": s.paid_cards[q], "blind_draws": s.blind_draws[q],
                "ticket_draws": s.ticket_draws[q], "tickets_kept": s.tickets_kept[q],
                "opening_kept": if !s.in_opening() || q == me { json!(s.opening_kept[q]) } else { Value::Null },
                "chose_opening": opening_chosen(s, q),
            })
        })
        .collect();
    let routes: Vec<Value> = (0..map.n_routes())
        .map(|r| {
            let o = s.route_owner[r];
            json!({
                "owner": if o == NONE { Value::Null } else { json!(o) }, "open": s.route_open(map, r),
                "turn": s.route_turn[r], "paid": s.route_paid[r] & 0x0f, "locos": s.route_paid[r] >> 4,
            })
        })
        .collect();
    let mut legal = Vec::new();
    if s.current as usize == me && s.phase != Phase::Over {
        s.legal_actions(map, &mut legal);
    }
    let events: Vec<Value> = s.recent_events().map(|e| json!([e.actor, e.kind, e.a, e.b])).collect();
    let mut v = json!({
        "players": n, "me": me, "current": s.current, "phase": phase, "opening": s.in_opening(),
        "last_round_left": s.last_round_left, "turn": s.turns, "moves": s.moves,
        "face_up": s.face_up.iter().map(|&c| if c == NONE { Value::Null } else { json!(c) }).collect::<Vec<_>>(),
        "deck": s.deck.len(), "discard": s.discard.counts(), "ticket_deck": visible_ticket_deck(map, s),
        "deck_shuffles": s.deck_shuffles,
        "routes": routes, "seats": seats,
        "hand": s.hand[me], "tickets": tickets_of(map, s, me),
        "offer": offer[..n_offer].to_vec(), "keep_min": s.keep_min,
        "legal": legal, "events": events, "events_total": s.events_total,
    });
    if s.is_over() {
        let f = s.finalize(map);
        v["final"] = json!({
            "total": f.total[..n], "route": f.route[..n], "tickets": f.tickets[..n],
            "longest": f.longest[..n], "completed": f.completed[..n], "ranks": f.ranks[..n],
            "hands": (0..n).map(|q| tickets_of(map, s, q)).collect::<Vec<_>>(),
        });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{GameRng, Xoshiro};
    use crate::sample::sample_world;

    #[test]
    fn every_city_of_the_usa_map_has_a_place() {
        let layout = usa_layout();
        let places = layout.as_array().unwrap();
        assert_eq!(places.len(), Map::usa().n_cities());
        for p in places {
            let (lat, lon) = (p["lat"].as_f64().unwrap(), p["lon"].as_f64().unwrap());
            assert!(
                (24.0..=52.0).contains(&lat) && (-125.0..=-65.0).contains(&lon),
                "{lat}, {lon}"
            );
        }
    }

    #[test]
    fn the_view_hides_what_the_player_does_not_see() {
        // like obs: a world re-sampled for `me` gives `me` the very same view (until the end, when
        // every ticket is shown)
        let map = Map::usa();
        let mut rng = Xoshiro::new(3);
        for n in 2..=5 {
            for seed in 0..6 {
                let mut g = Xoshiro::new(seed * 11 + n as u64);
                let mut s = State::new(map, n, &mut g);
                let mut moves = Vec::new();
                let mut step = 0;
                while !s.is_over() && step < 2000 {
                    if step % 7 == 0 {
                        for me in 0..n {
                            let w = sample_world(map, &s, me, &mut rng, None);
                            assert_eq!(
                                view_json(map, &s, me),
                                view_json(map, &w, me),
                                "{n} players, step {step}, seat {me}"
                            );
                        }
                    }
                    s.legal_actions(map, &mut moves);
                    if moves.is_empty() {
                        s.pass();
                    } else {
                        let a = moves[g.below(moves.len() as u32) as usize] as usize;
                        s.step(map, a, &mut g);
                    }
                    step += 1;
                }
                let v = view_json(map, &s, 0);
                assert_eq!(v["final"]["hands"].as_array().unwrap().len(), n);
            }
        }
    }
}
