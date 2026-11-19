"""Training the network: a trainer and one self-play worker a device, which talk through the files
of a run directory.

- `deepttr.train.config`: `TrainConfig`, what a phase of training sets, and `Opponent`
- `deepttr.train.losses`: the losses, the reward of a game and its TD(lambda) targets, PPO's pieces
- `deepttr.train.replay`: the shards of self-play and the window of positions
- `deepttr.train.trainer`: the trainer, `python -m deepttr.train.trainer RUN_DIR`
- `deepttr.train.worker`: a self-play worker
- `deepttr.train.process`: the devices and the run directory's files they share
"""
