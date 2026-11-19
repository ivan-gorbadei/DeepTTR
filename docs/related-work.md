# Related work

Earlier computer players of Ticket to Ride, as of November 2026. Each work was read at the source
given, except where noted. Strengths are as their authors report them. No two of these works share
an engine, a rule set or a benchmark, so the numbers are not comparable with each other or with
ours.

## Where the field stands

No published agent claims human-level play at the full game. The learning agents below were
measured on reduced maps; the agents for the full map are scripted or search-based.

No benchmark or rating scale is shared between projects. The general game-playing frameworks we
checked (OpenSpiel, RLCard, PettingZoo, TAG, Ludii, boardgame.io) do not ship the game.

The strongest scripted play reported for the USA map goes for the long routes and nearly ignores
the tickets. Witter and Lyford (2020) find that the scoring overvalues long routes. In Õunmaa
(2022) a player built on that, HoardCards, wins 85 of 100 four-player games on the base map and at
least 75 % in every other setting tested. A learned agent has to beat that strategy first.

## The closest earlier works

These are the works DeepTTR is to be compared with in strength. The comparison has not been made,
and nothing below is a claim about it.

### HoardCards (Õunmaa 2022)

Õunmaa, *Analyzing AI players on different Ticket to Ride variants*, MSc thesis, Leiden University,
2022. [PDF](https://theses.liacs.nl/pdf/2021-2022-OunmaaM.pdf)

Four players written inside the TAG framework: its MCTS and three scripted players, HoardCards,
RuleBased and Evaluation. They play the USA map with the base tickets and with the three ticket
sets of the 1910 expansion, and the Europe and Germany boards without their special rules. On the
base USA game, over 100 four-player games with one player of each kind, HoardCards won 85,
Evaluation 14 and MCTS 1. Eight people who had not played on the map before each played one
four-player game against the three scripted players: HoardCards won six of the eight games and the
people two.

HoardCards keeps the two cheapest of its three starting tickets. It then collects cards for twenty
rounds without claiming anything, and from then on buys the longest routes still free until the
game ends, before the other players have completed their tickets.

The `hoard` bot of this repository plays the same strategy and is not this player. It keeps the
same tickets, but it claims one of the longest free routes as soon as it can pay for it, where
HoardCards waits out its twenty rounds. DeepTTR has not been measured against HoardCards itself.

### Reinforcement learning on reduced maps

Strömberg and Lind, *Board Game AI Using Reinforcement Learning*, BSc thesis, Örebro University,
2022. [PDF](https://www.diva-portal.org/smash/get/diva2:1680520/FULLTEXT01.pdf)
Double DQN with self-play and action masking on a small board (18 cities, 39 routes, 20 trains, two
players). Up to 98.8 % of 250 games against a random agent.

Xinhe Zhou, *Investigating Reinforcement Learning and Evolutionary Computation for Games with
Stochasticity and Incomplete Information*, MEng thesis, MIT, 2022.
[Record](https://dspace.mit.edu/handle/1721.1/147529)
REINFORCE and evolution strategies on a game redesigned from Ticket to Ride: First Journey (7
cities, 10 trains), for one player and for two. In the two-player game with open information the
policy-gradient players share the top tier with a scripted greedy player; with hidden information
the greedy player beats every trained one.

Yang, Barlow, Townsend, Liu, Samarasinghe, Lakshika, Moy, Lynar and Turnbull, *Reinforcement
Learning Agents Playing Ticket to Ride–A Complex Imperfect Information Board Game With Delayed
Rewards*, IEEE Access 11, 2023.
[doi:10.1109/ACCESS.2023.3287100](https://doi.org/10.1109/ACCESS.2023.3287100)
PPO on a scaled-down map (8 cities, 16 routes, 4 colours, 10 trains, two players) under eight
training schemes: six against one heuristic player each, one against five of them in turn, one by
self-play. Averaged over the six heuristic players the agents win 57 % to 79 % of games, the
self-play agent 75 %, and the self-play agent beats each of the other seven. The authors leave the
full maps as future work.

### AlphaZero for Ticket to Ride (Petrenko 2025)

Petrenko, *An AlphaZero agent for Ticket to Ride*, MS thesis, Quinnipiac University, 2025.
[PDF](https://antonpetrenko.com/data/alphattr/alphattr.pdf), code:
[github.com/Anton-Petrenko/autoTTR](https://github.com/Anton-Petrenko/autoTTR). The author's
project page calls it AlphaTTR.

The network of AlphaGo Zero with Monte Carlo tree search, trained by self-play. The thesis reports
that training on the full USA map showed no improvement, and evaluates a model trained on the
8-city map of Yang et al. Over ten two-player games against each opponent it won most games
against the four scripted agents of de Mesentier Silva et al., five against a random player, one
against Huchler's Monte Carlo tree search player and none against flat Monte Carlo.

## Other work

| Work | Method | Scope | Reported result |
|---|---|---|---|
| Huchler, *An MCTS Agent for Ticket to Ride*, MSc thesis, Maastricht University, 2015. [PDF](https://project.dke.maastrichtuniversity.nl/games/files/msc/Huchler_thesis.pdf) | Monte Carlo tree search over determinizations; progressive unpruning and bias | USA map, two players | Measured against its own variants: the best wins 92.5 % of 200 games against plain flat Monte Carlo, and 52 % to 58 % against flat Monte Carlo with the same enhancements |
| de Mesentier Silva, Lee, Togelius and Nealen, *AI as Evaluator: Search Driven Playtesting of Modern Board Games*, AAAI workshop, 2017 ([PDF](https://cdn.aaai.org/ocs/ws/ws0367/15171-68466-1-PB.pdf)); *AI-based Playtesting of Contemporary Board Games*, FDG 2017 ([doi:10.1145/3102071.3102105](https://doi.org/10.1145/3102071.3102105)); *Evolving Maps and Decks for Ticket to Ride*, FDG 2018 ([PDF](https://www.pcgworkshop.com/archive/silva2018evolving.pdf)) | Four scripted agents (Destination Hungry, Route Focused, One Step, Long Route), used to playtest maps and variants | Eleven published variants of the game with two to four players (FDG 2017); the workshop paper has two agents on seven variants with two players | No agent dominates: the best style depends on the map and the number of players. An engine by the first author: [Ticket-to-Ride-Engine](https://github.com/fernandomsilva/Ticket-to-Ride-Engine) |
| Dinjian and Nguyen, *The Difficulty of Learning Ticket to Ride*, course report, Tufts University, 2018. [PDF](https://www.eecs.tufts.edu/~jsinapov/teaching/comp150_RL/reports/Nguyen_Dinjian_report.pdf) | A design: hand-made features, a linear value function, a curriculum | USA rules | No results reported |
| Witter and Lyford, *Applications of Graph Theory and Probability in the Board Game Ticket to Ride*, FDG 2020. [doi:10.1145/3402942.3402963](https://doi.org/10.1145/3402942.3402963) | Analysis with the four agents of de Mesentier Silva et al. | USA map | Long routes are overvalued, which gives a simple winning strategy |
| Smith and Anton, *Artificial Intelligence Approaches to Build Ticket to Ride Maps*, EAAI symposium at AAAI 2022. [doi:10.1609/aaai.v36i11.21564](https://doi.org/10.1609/aaai.v36i11.21564) | A map generator balanced by four strategy agents | Generated maps | none for play |
| Schaefer and Schaefer, *Maximizing the Score in "Ticket to Ride"*, 2025. [arXiv:2511.08441](https://arxiv.org/abs/2511.08441) | Mixed-integer programming; not an agent | USA map, one player alone on the board | 285 points from routes and tickets is the most one player can score, which bounds a real game's score at 295 with the longest-trail bonus |

Witter and Lyford's paper was read in the authors' draft and its published abstract. Forums and
unpublished projects were not searched.
