"""The commands: a match, its statistics, and the rating table, as a user runs them."""
import json
import subprocess
import sys


def run(*args: str) -> str:
    done = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_match_then_elo(engine, tmp_path):
    games = tmp_path / "games.jsonl"
    m = json.loads(run("deepttr.match", "hoard", "heuristic", "--deals", "20", "--seed", "3", "--append", str(games)))
    assert (m["a"], m["b"], m["players"], m["deals"], m["games"], m["seed"]) == ("hoard", "heuristic", 2, 20, 40, 3)
    assert m["a_wins"] + m["a_losses"] <= 40 and m["a_win_rate"] > 0.7
    run("deepttr.match", "random", "heuristic", "--deals", "20", "--name-a", "dice", "--append", str(games))
    lines = [json.loads(x) for x in games.read_text(encoding="utf-8").splitlines()]
    assert [(x["a"], x["b"]) for x in lines] == [("hoard", "heuristic"), ("dice", "heuristic")]
    table = [row.split()[0] for row in run("deepttr.elo", str(games)).splitlines()]
    assert table == ["hoard", "heuristic", "dice"]


def test_a_match_needs_real_contestants(engine):
    done = subprocess.run([sys.executable, "-m", "deepttr.match", "no-such-file.pt", "hoard"], capture_output=True, text=True, timeout=600)
    assert done.returncode != 0 and "not a checkpoint file" in done.stderr


def test_net_names_the_published_network(engine):
    from deepttr import match
    from deepttr import net as netmod

    assert match.checkpoint("net") == netmod.PUBLISHED and match.checkpoint("some/dir/other.pt").name == "other.pt"
    assert (match.name_of("net"), match.name_of("some/dir/other.pt"), match.name_of("hoard")) == ("deepttr", "other", "hoard")
    assert match.player("net", "cpu").net.n_params() == 7_711_871
    m = json.loads(run("deepttr.match", "net", "random", "--deals", "1", "--device", "cpu"))
    assert (m["a"], m["b"], m["games"], m["a_wins"], m["device"]) == ("deepttr", "random", 2, 2, "cpu")


def test_only_a_network_can_look_ahead(engine):
    done = subprocess.run([sys.executable, "-m", "deepttr.match", "hoard", "heuristic", "--look", "a"], capture_output=True, text=True,
                          timeout=600)
    assert done.returncode != 0 and "no policy to look ahead with" in done.stderr


def test_versus_tells_how_the_sides_played(engine):
    v = json.loads(run("deepttr.versus", "hoard", "heuristic", "--deals", "10", "--players", "3"))
    assert (v["a"], v["b"], v["players"], v["games"]) == ("hoard", "heuristic", 3, 30)
    a, b = v["a_stats"], v["b_stats"]
    assert a["seats"] == 30 and b["seats"] == 60
    assert a["ticket_draws"] == 0 and a["held"] == 2, "the card hoarder keeps two tickets and draws no more"
    assert b["completed"] > a["completed"], "the ticket builder completes more tickets"
    assert abs(sum(a["claims_by_length"].values()) - a["claims"]) < 0.05
    assert set(a) == set(b) >= {"total", "route", "tickets", "trail", "longest_share", "trains", "blind_draws", "face_up_draws"}
