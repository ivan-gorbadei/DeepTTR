"""The shards of self-play and the window: the shard format, the window's ring of positions, lessons
mixed into batches, and what a shard's name says. `synthetic` builds records of whole games with the
keys and dtypes of the engine's, for these tests and the other tests of training."""
import numpy as np
import pytest

from deepttr.train import replay as R

SHAPES = {"routes": (100, 36), "cities": (36, 8), "tickets": (30, 5), "players": (5, 40), "global": (58,)}


def synthetic(rng, games=3, tables=(2, 3, 4, 5), turns=16, lesson_share=0.0, looks=False, game0=0) -> dict:
    """Records of whole games: the seats move in turn, a turn of one or two records; every record
    sees the game's final scores from its seat, the mover's hand in `hands` row 0, and the seat's
    starting tickets in `ticket_start`."""
    seats = []
    for g in range(games):
        n = tables[g % len(tables)]
        final = rng.integers(30, 170, size=n) / 100.0
        starts = rng.permutation(30)[:2 * n].reshape(n, 2)
        move = 0
        for t in range(turns):
            for _ in range(int(rng.integers(1, 3))):
                seats.append((game0 + g, n, t % n, move, final, starts))
                move += 1
    N = len(seats)
    r = {k: rng.normal(0, 0.5, (N,) + s).astype(np.float16) for k, s in SHAPES.items()}
    r["slots"] = rng.integers(0, 10, (N, 5)).astype(np.int8)
    r["offer"] = np.where(rng.random((N, 3)) < 0.2, rng.integers(0, 30, (N, 3)), 30).astype(np.int8)
    ev = np.stack([rng.integers(0, 10, (N, 32)), rng.integers(0, 6, (N, 32)), rng.integers(0, 10, (N, 32)),
                   rng.integers(0, 101, (N, 32)), rng.integers(0, 8, (N, 32))], -1)
    r["events"] = ev.astype(np.int16)
    legal = rng.random((N, 422)) < 0.2
    legal[:, :2] = True
    r["legal"] = legal.astype(np.uint8)
    played = np.array([rng.choice(np.flatnonzero(row)) for row in legal])
    r["policy"] = np.eye(422, dtype=np.float16)[played]
    score = np.zeros((N, 5), np.float32)
    start = np.zeros((N, 30), np.uint8)
    for i, (g, n, s, mv, final, starts) in enumerate(seats):
        score[i, :n] = np.roll(final, -s)
        start[i, starts[s]] = 1
    r["score"] = score.astype(np.float16)
    r["n_players"] = np.array([x[1] for x in seats], np.uint8)
    r["game"] = np.array([x[0] for x in seats], np.uint64)
    r["moves"] = np.array([x[3] for x in seats], np.uint16)
    r["ticket_start"] = start
    r["value"] = np.where(score[:, :1] >= score.max(1, keepdims=True), 1.0, -1.0).repeat(5, 1).astype(np.float32)
    r["route_owner"] = rng.integers(-1, 5, (N, 100)).astype(np.int8)
    r["opp_tickets"] = (rng.random((N, 5, 30)) < 0.1).astype(np.uint8)
    r["ticket_outcome"] = np.where(start > 0, rng.choice([-1, 1], (N, 30)), 0).astype(np.int8)
    r["hands"] = rng.integers(0, 4, (N, 5, 9)).astype(np.uint8)
    r["path_cost"] = np.where(rng.random((N, 5, 30)) < 0.1, 255, rng.integers(0, 40, (N, 5, 30))).astype(np.uint8)
    r["trail"] = rng.random((N, 5)).astype(np.float16)
    r["trail_best"] = (rng.random((N, 5)) < 0.3).astype(np.uint8)
    r["root_prob"] = rng.uniform(0.05, 1.0, N).astype(np.float32)
    r["root_forecast"] = rng.normal(0, 0.1, N).astype(np.float32)
    r["lesson"] = (rng.random(N) < lesson_share).astype(np.uint8)
    if looks:
        look = np.full((N, R.LOOK_K), -1, np.int16)
        table = np.full((N, R.LOOK_K, R.LOOK_W), np.nan, np.float16)
        for i in np.flatnonzero(rng.random(N) < 0.3):
            tried = np.flatnonzero(legal[i])[:3]
            look[i, :len(tried)] = tried
            table[i, :len(tried), :8] = rng.normal(0, 10, (len(tried), 8)) + 6 * np.arange(len(tried))[:, None]
        r["look"], r["look_table"] = look, table
    return r


def test_a_shard_round_trips_in_the_windows_dtypes(tmp_path):
    recs = synthetic(np.random.default_rng(1), looks=True)
    recs["score"] = recs["score"].astype(np.float32)  # a key in another dtype: stored in the window's
    half = len(recs["policy"]) // 2
    R.write_shard([{k: v[:half] for k, v in recs.items()}, {k: v[half:] for k, v in recs.items()}], tmp_path / "self-w0-0000001.npz")
    assert not list(tmp_path.glob("*.tmp"))
    back = R.read_shard(tmp_path / "self-w0-0000001.npz")
    assert set(back) == set(recs)
    for k, v in back.items():
        assert v.dtype == R.stored_dtype(k, recs[k].dtype), k
        np.testing.assert_array_equal(v, recs[k].astype(v.dtype), err_msg=k)
    assert back["score"].dtype == np.float16 and back["game"].dtype == np.uint64 and back["look_table"].dtype == np.float16


def test_the_window_keeps_the_newest_records():
    rng = np.random.default_rng(2)
    a, b = synthetic(rng, games=2), synthetic(rng, games=2, game0=10)
    w = R.Replay(len(a["policy"]) + 10)
    w.add(a)
    assert w.size == len(a["policy"])
    assert (w.arrays["root_prob"][:w.size] == a["root_prob"]).all()
    assert np.isnan(w.arrays["adv_td"][:w.size]).all() and (w.arrays["opponent"][:w.size] == 0).all(), "defaults"
    w.add({**b, "adv_td": np.ones(len(b["policy"]), np.float32)})
    assert w.size == w.capacity
    newest = (w.head - 1) % w.capacity
    assert w.arrays["score"][newest].tolist() == b["score"][-1].tolist()
    assert w.arrays["adv_td"][newest] == 1.0
    batch = w.sample(64, rng)
    assert set(batch) == set(R.STORE) | set(R.DEFAULTS) and len(batch["policy"]) == 64
    assert "look" not in batch, "a window without looks keeps none"


def test_a_window_with_looks_fits_them_to_its_shape():
    recs = synthetic(np.random.default_rng(3), looks=True)
    recs["look_table"] = recs["look_table"][:, :, :8]  # fewer playouts than the window keeps
    w = R.Replay(1000, looks=True)
    w.add(recs)
    t = w.arrays["look_table"][:w.size]
    assert t.shape[1:] == (R.LOOK_K, R.LOOK_W) and np.isnan(t[:, :, 8:].astype(np.float32)).all()
    np.testing.assert_array_equal(t[:, :, :8], recs["look_table"])
    plain = R.Replay(1000, looks=True)
    plain.add(synthetic(np.random.default_rng(4)))
    assert (plain.arrays["look"][:plain.size] == -1).all(), "records without looks: none"


def test_lessons_are_split_off_and_take_the_last_rows_of_a_batch():
    rng = np.random.default_rng(5)
    recs = synthetic(rng, games=4, lesson_share=0.2)
    games, lessons = R.split_lessons(recs)
    assert (games["lesson"] == 0).all() and (lessons["lesson"] == 1).all()
    assert len(games["policy"]) + len(lessons["policy"]) == len(recs["policy"])
    assert R.split_lessons(games)[1] is None
    window, store = R.Replay(10_000), R.Replay(1000)
    window.add(games)
    store.add({**lessons, "root_prob": np.full(len(lessons["policy"]), 0.5, np.float32)})
    batch = R.mixed_batch(window, store, 10, 64, rng)
    assert len(batch["policy"]) == 64 and (batch["root_prob"][-10:] == 0.5).all()
    assert len(R.mixed_batch(window, store, 0, 64, rng)["policy"]) == 64


@pytest.mark.parametrize("opponent,table,look,name", [
    ("self", 0, False, "self-w0-0000003.npz"),
    ("self", 0, True, "kself-w1-0000010.npz"),
    ("card hoarder", 2, False, "L-card-hoarder-n2-w0-0000007.npz"),
    ("phase 5", 3, False, "L-phase-5-n3-w12-0000001.npz"),
    ("old-n2-w1", 5, False, "L-old-n2-w1-n5-w3-0000002.npz"),
    ("mixed", 4, False, "L-mixed-n4-w0-0000001.npz"),
    ("mixed", 2, True, "kL-mixed-n2-w1-0000020.npz"),
])
def test_a_shards_name_says_its_pool(opponent, table, look, name):
    worker, seq = int(name.split("-w")[-1].split("-")[0]), int(name[-11:-4])
    prefix = R.pool_prefix(opponent, table, look)
    assert R.shard_name(prefix, worker, seq) == name and " " not in name and "@" not in name
    s = R.shard_pool(name)
    assert s == {"look": look, "prefix": prefix, "opponent": R.spelling(opponent), "table": table, "worker": worker, "seq": seq}


def test_which_shards_are_held_out():
    assert R.held_out("self-w0-0000000.npz") and R.held_out("self-w1-0000025.npz") and not R.held_out("self-w0-0000010.npz")
    assert R.held_out("kself-w0-0000010.npz") and not R.held_out("kself-w0-0000025.npz")
    assert R.held_out("kL-mixed-n3-w0-0000020.npz"), "the look-ahead's games against the league too"
    assert not R.held_out("L-card-hoarder-n2-w0-0000000.npz") and not R.held_out("L-mixed-n4-w0-0000050.npz")
    assert R.shard_pool("0-0000001.npz") is None and not R.held_out("notes.npz")
