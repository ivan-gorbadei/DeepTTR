"""DeepTTR: a neural network that plays Ticket to Ride (the base game, USA map, 2 to 5 players).

- `deepttr._engine`: the rules engine (compiled from `engine/`): games, observations, matches
- `deepttr.net`: the network; `net.load()` gives the published one, `deepttr/weights/deepttr.pt`
- `deepttr.play`: running a network for the engine; the duplicate match
- `deepttr.look`: the look-ahead, a network that tests its first choice by playing moves out
- `deepttr.match`, `deepttr.versus`: a match and its statistics from the command line
- `deepttr.elo`: one rating scale from many matches
- `deepttr.serve`: play against the network in a browser
"""

__version__ = "0.1.0"
