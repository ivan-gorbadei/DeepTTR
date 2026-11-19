"""The strength measures of the training's networks."""


def test_a_table_match_counts_the_games_and_places_of_every_seating(engine, stub):
    from deepttr import measure
    from deepttr.play import Bot

    for players in (2, 3):
        line = measure.table_match(stub(), Bot("hoard"), players, deals=6, threads=2)
        assert line["players"] == players and line["games"] == 6 * players and line["move"] == "first"
        assert 0 <= line["first_places"] <= line["games"]
        assert line["opponent_points"] > -1000 and line["completed"] >= 0
    again = measure.table_match(stub(), Bot("hoard"), 2, deals=6, threads=2)
    assert again["first_places"] == measure.table_match(stub(), Bot("hoard"), 2, deals=6, threads=2)["first_places"]


def test_a_sampled_match_plays_other_moves_than_the_first_choice_and_repeats_with_its_seed(engine, stub):
    from deepttr import measure

    first = measure.table_match(stub(), stub(0.5), 2, deals=8, threads=2)
    sampled = [measure.table_match(stub(), stub(0.5), 2, deals=8, threads=2, sample=True) for _ in range(2)]
    assert sampled[0]["move"] == "sample" and sampled[0] == {**sampled[1], "seconds": sampled[0]["seconds"]}
    assert (sampled[0]["points"], sampled[0]["first_places"]) != (first["points"], first["first_places"])


def test_the_ticket_measure_scores_tickets_alone(engine, stub):
    from deepttr import measure

    line = measure.ticket_measure(stub(), deals=8, threads=2)
    assert line["deals"] == 8 and line["held_per_seat"] >= 2
    assert -200 < line["ticket_points_per_seat"] < 200


def test_the_goal_is_the_mean_share_won_over_the_cells_of_its_move():
    from deepttr.measure import goal

    lines = [{"move": "sample", "games": 400, "first_places": 200}, {"move": "sample", "games": 400, "first_places": 100},
             {"move": "first", "games": 400, "first_places": 400}]
    assert goal(lines) == 0.375 and goal(lines, "first") == 1.0 and goal([]) is None
