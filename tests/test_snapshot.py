"""The published network: its checkpoint gives, on recorded positions, the outputs recorded for them,
and through the engine it beats the baseline it was measured against.

`data/final_network_positions.npz` holds 32 positions of training games - eight at each table size, two
of them choices of tickets - as the replay buffer stored them (float16 features), and every output
of the network on them, computed in float32 on a CPU by the code the network was trained with.
The checkpoint is the one that comes with the package (`net.PUBLISHED`).
"""
import hashlib

import numpy as np
import pytest
import torch

from conftest import DATA
from deepttr import net as netmod

OUTPUTS = ("logits", "value", "score", "belief", "owner", "ticket", "hand", "path", "trail", "adv", "move")


@pytest.fixture(scope="module")
def recorded():
    return np.load(DATA / "final_network_positions.npz")


def test_the_fixture_is_whole(recorded):
    assert recorded["table"].tolist() == [2] * 8 + [3] * 8 + [4] * 8 + [5] * 8
    assert all(recorded["expect_" + k].shape[0] == 32 for k in OUTPUTS)
    assert (recorded["legal"].sum(axis=1) > 1).all(), "every position is a real choice"
    assert ((recorded["offer"] < 30).any(axis=1).reshape(4, 8).sum(axis=1) == 2).all(), "two choices of tickets a table"
    present = recorded["players"].astype(np.float32).reshape(32, 5, 40)[:, :, 0].sum(axis=1)
    assert present.tolist() == recorded["table"].tolist(), "the seats present are the table's"


def test_the_package_holds_the_network_the_positions_were_recorded_for(recorded):
    assert hashlib.sha256(netmod.PUBLISHED.read_bytes()).hexdigest() == str(recorded["snapshot_sha256"])
    net, meta = netmod.load()
    assert meta["step"] == int(recorded["step"]) and net.n_params() == 7_711_871


def test_the_checkpoint_gives_the_recorded_outputs(snapshot, recorded):
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != str(recorded["snapshot_sha256"]):
        pytest.skip("DEEPTTR_SNAPSHOT is another network than the one the positions were recorded for")
    net, meta = netmod.load(snapshot)
    assert meta["step"] == int(recorded["step"]) and net.cfg == netmod.NetConfig() and net.n_params() == 7_711_871
    obs = {k: recorded[k].astype(np.float32 if recorded[k].dtype == np.float16 else recorded[k].dtype) for k in netmod.OBS_KEYS}
    with torch.inference_mode():
        out = net(netmod.to_device(obs, torch.device("cpu")))
    assert set(out) == set(OUTPUTS)
    legal = recorded["legal"] > 0
    for k in OUTPUTS:
        got, want = out[k].numpy(), recorded["expect_" + k]
        if k in ("logits", "move"):  # an illegal move's logit is the mask, its forecast means nothing
            got, want = got[legal], want[legal]
        # float32 rounds differently from one CPU and math library to the next, and six layers
        # carry it: up to a few 1e-4 here, where a wrong reading of the checkpoint is off by 0.1 or more
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=2e-3, err_msg=k)
    # the first choice is the same wherever the two best moves are not within rounding of each other
    masked = np.where(legal, recorded["expect_logits"], -np.inf)
    top = np.sort(masked, axis=1)
    clear = top[:, -1] - top[:, -2] > 1e-2
    assert clear.sum() >= 24
    assert (np.where(legal, out["logits"].numpy(), -np.inf).argmax(axis=1)[clear] == masked.argmax(axis=1)[clear]).all()


def test_the_network_beats_the_card_hoarder_through_the_engine(engine, snapshot):
    """The network and the engine together: by its first choice the published network wins about
    nine games in ten against the card hoarder at a table of two. Four deals here, eight games, to
    stay quick on a CPU: enough to tell the network from one that reads the engine wrongly, not a
    measurement of its strength."""
    from deepttr.play import Bot, Evaluator, duplicate_match

    net, _ = netmod.load(snapshot)
    m = duplicate_match(Evaluator(net, "cpu"), Bot("hoard"), deals=4, players=2, seed=1)
    assert m["a_wins"] >= 5 and m["a_score"] > m["b_score"] + 20, m
