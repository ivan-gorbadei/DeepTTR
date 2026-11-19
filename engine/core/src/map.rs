//! The board: cities, routes, destination tickets, and the action table built from them.
//!
//! The USA map is embedded from `data/maps/usa/*.json`. Every move of the game has a fixed action
//! id, in this order: the claims in route order (one for a coloured route; eight for a grey route,
//! one per colour that can pay it), the five face-up slots, the blind draw, the ticket draw, and the
//! seven keep masks of a ticket choice. The order of the routes and tickets in the files therefore
//! fixes the layout of a network's policy output: reordering them invalidates trained weights.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

pub const COLORS: [&str; 8] = ["black", "blue", "green", "orange", "pink", "red", "white", "yellow"];
pub const N_COLORS: usize = 8;
pub const LOCO: u8 = 8;
pub const N_CARD_TYPES: usize = 9;
pub const ROUTE_SCORE: [i16; 7] = [0, 1, 2, 4, 7, 10, 15];
pub const MAX_ROUTES: usize = 128;
pub const MAX_TICKETS: usize = 64;
pub const MAX_CITIES: usize = 64;

#[derive(Debug, Clone)]
pub struct Route {
    pub a: u8,
    pub b: u8,
    /// None = grey (any one colour pays it)
    pub color: Option<u8>,
    pub length: u8,
    /// index into `Map::pairs`: the routes between the same two cities
    pub pair: u16,
}

#[derive(Debug, Clone)]
pub struct Ticket {
    pub a: u8,
    pub b: u8,
    pub points: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct Claim {
    pub route: u16,
    /// the colour the cards are of: the route's own for a coloured route, the chosen one for grey
    pub color: u8,
}

#[derive(Debug)]
pub struct Map {
    pub id: String,
    pub cities: Vec<String>,
    pub routes: Vec<Route>,
    pub tickets: Vec<Ticket>,
    pub claims: Vec<Claim>,
    /// the routes of each city pair, in route order (a double has two)
    pub pairs: Vec<Vec<u16>>,
    /// with fewer players than this, a double pair is closed once either of its routes is taken
    pub both_doubles_min_players: u8,
    pub n_claim: usize,
    pub n_actions: usize,
    /// the fewest trains that join each ticket's cities on an empty board: its whole path, for the
    /// partial-ticket rule (`state::partial_score`)
    pub ticket_length: Vec<u8>,
    /// each ticket's routes that lie on a shortest path of the empty board, as (from, to, route), `from`
    /// the end nearer the ticket's first city, in the order of that distance: the partial-ticket rule
    /// counts only what a player built on one such path (`State::built_on_shortest`)
    pub ticket_paths: Vec<Vec<(u8, u8, u16)>>,
}

/// Dijkstra over a dense graph of `n` cities: `w[a][b]` is the cost of the cheapest usable route
/// between a and b (`u16::MAX`: none); returns the cost from `src` to every city (`u16::MAX`: no way).
pub fn shortest(n: usize, w: &[[u16; MAX_CITIES]; MAX_CITIES], src: usize) -> [u16; MAX_CITIES] {
    const FAR: u16 = u16::MAX;
    let mut d = [FAR; MAX_CITIES];
    let mut done = [false; MAX_CITIES];
    d[src] = 0;
    for _ in 0..n {
        let Some(u) = (0..n).filter(|&c| !done[c] && d[c] < FAR).min_by_key(|&c| d[c]) else {
            break;
        };
        done[u] = true;
        for v in 0..n {
            if w[u][v] < FAR && d[u] + w[u][v] < d[v] {
                d[v] = d[u] + w[u][v];
            }
        }
    }
    d
}

#[derive(Deserialize)]
struct RoutesFile {
    map: String,
    players_for_both_doubles: u8,
    cities: Vec<String>,
    routes: Vec<RouteRow>,
}

#[derive(Deserialize)]
struct RouteRow {
    a: String,
    b: String,
    color: String,
    length: u8,
}

#[derive(Deserialize)]
struct TicketsFile {
    tickets: Vec<TicketRow>,
}

#[derive(Deserialize)]
struct TicketRow {
    a: String,
    b: String,
    points: u8,
}

impl Map {
    pub fn from_json(routes_json: &str, tickets_json: &str) -> Result<Map, String> {
        let rf: RoutesFile = serde_json::from_str(routes_json).map_err(|e| format!("routes: {e}"))?;
        let tf: TicketsFile = serde_json::from_str(tickets_json).map_err(|e| format!("tickets: {e}"))?;
        if rf.cities.len() > MAX_CITIES || rf.routes.len() > MAX_ROUTES || tf.tickets.len() > MAX_TICKETS {
            return Err(format!(
                "map too large: {} cities (max {MAX_CITIES}), {} routes (max {MAX_ROUTES}), {} tickets (max {MAX_TICKETS})",
                rf.cities.len(),
                rf.routes.len(),
                tf.tickets.len()
            ));
        }
        let city_id: HashMap<&str, u8> = rf
            .cities
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i as u8))
            .collect();
        let city = |name: &str| {
            city_id
                .get(name)
                .copied()
                .ok_or_else(|| format!("unknown city {name:?}"))
        };
        let mut pair_of: HashMap<(u8, u8), u16> = HashMap::new();
        let mut pairs: Vec<Vec<u16>> = Vec::new();
        let mut routes = Vec::with_capacity(rf.routes.len());
        let mut claims = Vec::new();
        for (i, row) in rf.routes.iter().enumerate() {
            let (a, b) = (city(&row.a)?, city(&row.b)?);
            let color = if row.color == "grey" {
                None
            } else {
                Some(
                    COLORS
                        .iter()
                        .position(|c| *c == row.color)
                        .ok_or_else(|| format!("unknown colour {:?}", row.color))? as u8,
                )
            };
            if row.length == 0 || row.length as usize >= ROUTE_SCORE.len() {
                return Err(format!("route {i}: length {} has no score", row.length));
            }
            let key = (a.min(b), a.max(b));
            let pair = *pair_of.entry(key).or_insert_with(|| {
                pairs.push(Vec::new());
                (pairs.len() - 1) as u16
            });
            pairs[pair as usize].push(i as u16);
            routes.push(Route {
                a,
                b,
                color,
                length: row.length,
                pair,
            });
            match color {
                Some(c) => claims.push(Claim {
                    route: i as u16,
                    color: c,
                }),
                None => {
                    for c in 0..N_COLORS as u8 {
                        claims.push(Claim {
                            route: i as u16,
                            color: c,
                        });
                    }
                }
            }
        }
        let mut tickets = Vec::with_capacity(tf.tickets.len());
        for row in &tf.tickets {
            tickets.push(Ticket {
                a: city(&row.a)?,
                b: city(&row.b)?,
                points: row.points,
            });
        }
        let n_claim = claims.len();
        let mut w = [[u16::MAX; MAX_CITIES]; MAX_CITIES];
        for r in &routes {
            let (a, b) = (r.a as usize, r.b as usize);
            w[a][b] = w[a][b].min(r.length as u16);
            w[b][a] = w[a][b];
        }
        let ticket_length = tickets
            .iter()
            .map(|t| shortest(rf.cities.len(), &w, t.a as usize)[t.b as usize].min(254) as u8)
            .collect();
        let ticket_paths = tickets
            .iter()
            .map(|t| {
                let (da, db) = (
                    shortest(rf.cities.len(), &w, t.a as usize),
                    shortest(rf.cities.len(), &w, t.b as usize),
                );
                let whole = da[t.b as usize] as u32;
                let mut on: Vec<(u32, u8, u8, u16)> = Vec::new();
                for (i, r) in routes.iter().enumerate() {
                    let (a, b, len) = (r.a as usize, r.b as usize, r.length as u32);
                    if da[a] as u32 + len + db[b] as u32 == whole {
                        on.push((da[a] as u32, r.a, r.b, i as u16));
                    } else if da[b] as u32 + len + db[a] as u32 == whole {
                        on.push((da[b] as u32, r.b, r.a, i as u16));
                    }
                }
                on.sort_by_key(|e| (e.0, e.3));
                on.into_iter().map(|(_, from, to, r)| (from, to, r)).collect()
            })
            .collect();
        Ok(Map {
            id: rf.map,
            cities: rf.cities,
            routes,
            tickets,
            claims,
            pairs,
            both_doubles_min_players: rf.players_for_both_doubles,
            n_claim,
            // claims + 6 draws (5 face-up slots, the deck) + the ticket draw + 7 keep masks
            n_actions: n_claim + 6 + 1 + 7,
            ticket_length,
            ticket_paths,
        })
    }

    /// The base game's USA map.
    pub fn usa() -> &'static Map {
        static USA: OnceLock<Map> = OnceLock::new();
        USA.get_or_init(|| {
            Map::from_json(
                include_str!("../../../data/maps/usa/routes.json"),
                include_str!("../../../data/maps/usa/tickets.json"),
            )
            .expect("the embedded USA map is valid")
        })
    }

    pub fn n_routes(&self) -> usize {
        self.routes.len()
    }
    pub fn n_tickets(&self) -> usize {
        self.tickets.len()
    }
    pub fn n_cities(&self) -> usize {
        self.cities.len()
    }

    #[inline]
    pub fn draw_action(&self, slot: usize) -> usize {
        self.n_claim + slot
    }
    #[inline]
    pub fn blind_action(&self) -> usize {
        self.n_claim + 5
    }
    #[inline]
    pub fn tickets_action(&self) -> usize {
        self.n_claim + 6
    }
    /// `mask` 1..=7: bit i keeps the i-th ticket on offer
    #[inline]
    pub fn keep_action(&self, mask: u8) -> usize {
        self.n_claim + 7 + (mask as usize - 1)
    }
    #[inline]
    pub fn keep_mask_of(&self, action: usize) -> u8 {
        (action - self.n_claim - 7 + 1) as u8
    }

    pub fn describe(&self, action: usize) -> String {
        if action < self.n_claim {
            let c = self.claims[action];
            let r = &self.routes[c.route as usize];
            format!(
                "claim {}-{} ({} {}) with {}",
                self.cities[r.a as usize],
                self.cities[r.b as usize],
                r.length,
                r.color.map_or("grey", |x| COLORS[x as usize]),
                COLORS[c.color as usize]
            )
        } else if action < self.n_claim + 5 {
            format!("take face-up slot {}", action - self.n_claim)
        } else if action == self.blind_action() {
            "draw from the deck".into()
        } else if action == self.tickets_action() {
            "draw tickets".into()
        } else if action < self.n_actions {
            format!("keep tickets {:03b}", self.keep_mask_of(action))
        } else {
            format!("action {action}?")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usa_map_has_the_known_shape() {
        let m = Map::usa();
        assert_eq!(m.n_cities(), 36);
        assert_eq!(m.n_routes(), 100);
        assert_eq!(m.n_tickets(), 30);
        assert_eq!(m.pairs.len(), 78);
        assert_eq!(m.pairs.iter().filter(|p| p.len() == 2).count(), 22);
        // 56 coloured routes + 44 grey x 8 colours
        assert_eq!(m.n_claim, 56 + 44 * 8);
        assert_eq!(m.n_actions, 422);
        assert_eq!(m.both_doubles_min_players, 4);
        // a property of the published board: the coloured routes of each colour take 27 trains
        for c in 0..N_COLORS as u8 {
            let trains: u32 = m
                .routes
                .iter()
                .filter(|r| r.color == Some(c))
                .map(|r| r.length as u32)
                .sum();
            assert_eq!(trains, 27, "colour {}", COLORS[c as usize]);
        }
    }

    #[test]
    fn action_ids_round_trip() {
        let m = Map::usa();
        for mask in 1..=7u8 {
            assert_eq!(m.keep_mask_of(m.keep_action(mask)), mask);
        }
        assert_eq!(m.tickets_action(), m.n_claim + 6);
        assert_eq!(m.keep_action(7), m.n_actions - 1);
    }

    #[test]
    fn every_ticket_has_a_shortest_path_from_its_first_city_to_its_second() {
        let m = Map::usa();
        assert!(m.ticket_length.iter().all(|&l| l > 0 && l < 40));
        for (t, edges) in m.ticket_paths.iter().enumerate() {
            assert!(!edges.is_empty() && edges[0].0 == m.tickets[t].a, "ticket {t}");
            assert!(edges.iter().any(|e| e.1 == m.tickets[t].b), "ticket {t}");
        }
    }
}
