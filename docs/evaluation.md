# Evaluation

How strength is measured here, and what has been measured.

## The duplicate match

Ticket to Ride has a lot of luck in the deal: the starting tickets and cards decide many games
between players of similar strength. A duplicate match removes most of it.

A match sets contestant A against contestant B on a sequence of deals. A deal fixes the shuffled
train cards, the starting hands, the tickets dealt and who moves first. Each deal is played once
per seat: A takes each seat in turn and B holds all the others. At a table of two that is the same
deal twice with the seats swapped. At a table of four, A plays one seat against three copies of B,
four times.

The games of a deal also share everything random after the deal (the reshuffles of the discard
pile), so two identical contestants play the very same game in every seating.

A's utility of a game is `1 - 2 * rank / (players - 1)`: +1 for first place, -1 for last. Tied
places share the average rank, after the rules' tie-breaks. Averaged over the seatings of a deal the
utility is exactly 0 for equal contestants, whatever the deal.

The match reports A's mean utility over the deals with a 95 % interval (the deals are the
independent samples), the games A won (first place alone) and lost (last place alone), and the mean
scores.

A network plays its first choice, the legal move with the highest policy logit, with no search
(`--look a` gives side A the look-ahead of the play server instead). The
match is `deepttr.play.duplicate_match`, on the command line `python -m deepttr.match`. The engine
side is `engine/core/src/arena.rs`.

## The baselines

Three bots are built into the engine (`engine/core/src/bots.rs`). They read the full state, need no
network, and are the fixed opponents of every measurement.

`hoard` is a card hoarder. It keeps the fewest and cheapest tickets, never draws more, collects
cards until one of the longest routes still open can be paid for, claims it, and ends the game
early. The strategy is that of HoardCards in Õunmaa (2022), which won 85 of 100 four-player games
on the USA map against an MCTS player and two other scripted players. The bot is not that player:
HoardCards claims nothing for its first twenty rounds and then buys the longest routes left, and
this bot claims as soon as it can pay. Which of the two is stronger has not been measured.

`heuristic` is a ticket builder, this project's own baseline. It keeps the tickets that fit
together, claims routes on the shortest paths of its open tickets, and draws the colours those
paths need.

`random` plays uniformly random legal moves.

At a table of two the card hoarder beats the ticket builder 388 to 12 over 200 deals, with mean
scores of 95.9 and 46.1. Giving up tickets and racing for six-train routes is a strong plan
against a player who only builds.

## Results

The network against the baselines: 200 deals per row, seed 1, first choice, bfloat16 on a CUDA
card. A win is a game the network finished first alone. At tables of three and more it holds one
seat and the bot all the others, so an even result would be 33 %, 25 % and 20 %. Intervals are cut
at 1, the highest utility there is.

| Opponent | Table | Games | Won by the network | Mean duplicate utility (95 % interval) | Mean score, network : opponent |
|---|---|---|---|---|---|
| Card hoarder | 2 | 400 | 356 (89.0 %) | 0.780 (0.722, 0.838) | 132.9 : 88.0 |
| Card hoarder | 3 | 600 | 572 (95.3 %) | 0.928 (0.901, 0.955) | 135.3 : 76.8 |
| Card hoarder | 4 | 800 | 762 (95.3 %) | 0.937 (0.916, 0.957) | 121.3 : 70.6 |
| Card hoarder | 5 | 1000 | 931 (93.1 %) | 0.908 (0.886, 0.929) | 114.4 : 64.2 |
| Ticket builder | 2 | 400 | 399 (99.8 %) | 0.995 (0.985, 1.000) | 177.3 : 67.8 |
| Ticket builder | 3 | 600 | 577 (96.2 %) | 0.945 (0.921, 0.969) | 157.1 : 68.7 |
| Ticket builder | 4 | 800 | 775 (96.9 %) | 0.964 (0.950, 0.979) | 153.2 : 70.2 |
| Ticket builder | 5 | 1000 | 875 (87.5 %) | 0.855 (0.829, 0.880) | 133.2 : 68.4 |
| Random | 2 | 400 | 396 (99.0 %) | 0.980 (0.961, 0.999) | 146.2 : -45.7 |

The first row repeated in float32 on a CPU gives 354 wins of 400 (88.5 %) and mean scores of 132.6
and 88.0. The difference is the precision, not the code: the two pick a different move in about one
position in 400.

How the two sides play, from the same 400 games against the card hoarder at a table of two
(`python -m deepttr.versus`), as means per game:

| | Network | Card hoarder |
|---|---|---|
| Final score | 132.9 | 88.0 |
| Route points | 68.5 | 106.7 |
| Ticket points | 54.8 | -19.2 |
| Tickets held at the end | 6.35 | 2.00 |
| Tickets completed | 5.12 | 0.00 |
| Ticket draws after the start | 3.09 | 0 |
| Holds the longest trail | 96 % of games | 5 % |
| Routes claimed | 11.6 | 7.8 |
| of them six trains long | 1.5 | 6.8 |

The card hoarder scores more for its routes and loses on tickets: it completes none and pays for
the two it must keep. The network draws tickets three times a game and completes five of the six
it holds.

## The rating scale

Matches between many players are put on one scale by a Bradley-Terry model (`deepttr.elo`):
`P(a beats b) = 1 / (1 + 10^((R_b - R_a) / 400))`, with a game neither side won counted as half a
win. One player is the anchor at 0. Every other player has two virtual draws against the anchor, so
that a one-sided record still gives a finite estimate. The 95 % intervals come from the curvature
of the likelihood at its maximum.

```sh
python -m deepttr.match hoard heuristic --append games.jsonl
python -m deepttr.match net hoard --append games.jsonl
python -m deepttr.elo games.jsonl --anchor heuristic
```

A scale mixes only matches at one table size (`--players`, two by default).

## Reproducing

```sh
python -m deepttr.match net hoard --deals 200 --players 2 --seed 1
python -m deepttr.versus net hoard --deals 200 --players 2 --seed 1
```

The deals depend only on `--seed` and `--first-deal`. The bots are deterministic given the deal,
and the engine uses no floating-point operation whose result depends on the platform, so a match
between bots gives the same games everywhere; the tests pin a few of them.

A network adds the device's arithmetic. On one device a match is exactly repeatable. Between
float32 and bfloat16 the move differed in 0.26 % of 3,044 recorded positions, which shifts a result
by a fraction of its interval. `--no-amp` runs float32 on any device.

## The rules as implemented

The engine plays the base game on the USA map (36 cities, 100 routes, 30 destination tickets) for
two to five players.

- 45 trains each; 110 train cards (12 of each of eight colours, 14 locomotives); four cards dealt;
  five face up.
- A turn is one of: claim a route; take two train cards; draw three tickets and keep at least one.
  A face-up locomotive is a whole turn and cannot be the second card. Three locomotives face up
  replace the whole row.
- At the start each player is dealt three tickets and keeps at least two. Tickets not kept go to
  the bottom of the ticket deck.
- A grey route takes cards of any one colour; locomotives are wild. Both tracks of a double route
  are open at tables of four and five, one only at tables of two and three, and no player takes
  both.
- When a player is left with two trains or fewer, every player, that one included, has one more
  turn.
- Routes score 1, 2, 4, 7, 10, 15 points for lengths 1 to 6. A completed ticket adds its points, a
  failed one subtracts them. The longest continuous trail scores 10, to each player who has it.
- Ties are broken by completed tickets, then by the longest trail.

Published editions and digital versions of the game differ in details, the number of starting
tickets for one. Scores from two engines can be compared only after their rules have been.

The engine can also play variants used as a training curriculum: routes that score nothing or a
share of their points, no longest-trail bonus, a failed ticket scored by the share of its shortest
path that was built, fewer trains. A variant has to be asked for (`rules=` in the Python API).
Every number in this document is by the rules above.
