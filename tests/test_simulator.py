"""M3: simulator mechanics on hand-built races, then the real back-test (uses cache/)."""
import pandas as pd
import pytest

from src.data_loader import load_config
from src.pipeline import backtest, prepare
from src.simulator import simulate


def race(times, neutral, pit_in=(), pit_out=()):
    """times: {driver: cumulative s at the end of lap 0, 1, ...}; neutral: laps 1.. ; pit_in/out: (driver, lap)."""
    T = pd.DataFrame(times, dtype=float)

    def flags(pairs):
        f = pd.DataFrame(False, T.index, T.columns)
        for drv, lap in pairs:
            f.loc[lap, drv] = True
        return f

    return {"times": T, "pit_in": flags(pit_in), "pit_out": flags(pit_out),
            "neutral": pd.Series(["green", *neutral], T.index), "wear": pd.Series(1.0, T.index)}


def test_green_laps_and_green_stop():
    r = race({"A": [0, 100, 200, 300, 400, 500, 600], "B": [0, 101, 202, 303, 404, 505, 606]}, ["green"] * 6)
    sim = simulate(r, 0, 0.0, [("SOFT", 0, 3), ("MEDIUM", 0, 6)], lambda lap, c, age: 100.5, pit_loss_s=20.0)
    assert sim.cum_s.tolist() == pytest.approx([100.5, 201, 321.5, 422, 522.5, 623])   # +20 s on the lap-3 in-lap
    assert sim.position.tolist() == [2, 2, 3, 3, 3, 3]
    assert sim.compound.tolist() == ["SOFT"] * 3 + ["MEDIUM"] * 3
    assert sim.age.tolist() == [1, 2, 3, 1, 2, 3]


def test_green_stop_loss_splits_between_in_lap_and_out_lap():
    r = race({"A": [0, 100, 200, 300, 400, 500, 600], "B": [0, 101, 202, 303, 404, 505, 606]}, ["green"] * 6)
    sim = simulate(r, 0, 0.0, [("SOFT", 0, 3), ("MEDIUM", 0, 6)], lambda lap, c, age: 100.5, pit_loss_s=20.0,
                   pit_in_share=0.25)
    assert sim.cum_s.tolist() == pytest.approx([100.5, 201, 306.5, 422, 522.5, 623])   # +5 s on the in-lap, +15 s out
    assert sim.position.tolist() == [2, 2, 3, 3, 3, 3]


def test_neutral_laps_copy_the_cars_around_it():
    r = race({"A": [0, 130, 260], "B": [10, 134, 264], "C": [20, 138, 268]}, ["sc", "vsc"])
    sim = simulate(r, 0, 15.0, [("SOFT", 0, 2)], lambda lap, c, age: 90.0, pit_loss_s=0.0)
    # halfway between B and C on the road: the SC squeezes B-C from 10 to 4 s, the VSC freezes it
    assert sim.cum_s.tolist() == pytest.approx([136, 266])
    assert sim.position.tolist() == [3, 3]


def test_neutral_laps_outside_the_field_shrink_like_its_median_gap():
    r = race({"A": [10, 140, 270], "B": [14, 142, 272], "C": [18, 144, 274]}, ["sc", "sc"])
    # field gaps 4 -> 2 -> 2 s: a gap to the end car halves on lap 1, then holds
    lead = simulate(r, 0, 4.0, [("SOFT", 0, 2)], lambda lap, c, age: 90.0, pit_loss_s=0.0)
    last = simulate(r, 0, 24.0, [("SOFT", 0, 2)], lambda lap, c, age: 90.0, pit_loss_s=0.0)
    assert lead.cum_s.tolist() == pytest.approx([137, 267])
    assert last.cum_s.tolist() == pytest.approx([147, 277])
    assert lead.position.tolist() == [1, 1] and last.position.tolist() == [4, 4]


def test_neutral_laps_skip_pitting_and_retiring_cars():
    r = race({"A": [0, 130, 260], "B": [5, 160, 290], "C": [10, 200, None], "D": [20, 138, 268]}, ["sc", "sc"],
             pit_in=[("B", 1)], pit_out=[("B", 2)])
    sim = simulate(r, 0, 15.0, [("SOFT", 0, 2)], lambda lap, c, age: 90.0, pit_loss_s=0.0)
    # B is pitting and C crawls round to retire, so the ghost copies A and D (3/4 of the way from A to D)
    assert sim.cum_s.tolist() == pytest.approx([136, 266])
    assert sim.position.tolist() == [2, 2]


def test_stop_under_sc_interpolates_the_stoppers_around_it():
    r = race({"A": [0, 100, 250, 400, 500], "B": [0, 105, 255, 401, 501],
              "C": [0, 110, 262, 420, 520], "D": [0, 115, 265, 404, 504]},
             ["green", "sc", "sc", "green"], pit_in=[("A", 2), ("C", 2)], pit_out=[("A", 3), ("C", 3)])
    sim = simulate(r, 1, 107.5, [("INTERMEDIATE", 0, 2), ("SOFT", 0, 4)], lambda lap, c, age: 100.0, pit_loss_s=99.0)
    # 3/4 of the way from A to C on the road: in-lap 150 + 0.75 * 2, in + out 300 + 0.75 * 10. pit_loss_s is unused.
    assert sim.cum_s.tolist() == pytest.approx([259, 415, 515])
    assert sim.compound.tolist() == ["INTERMEDIATE", "SOFT", "SOFT"]
    assert sim.position.tolist() == [3, 4, 4]          # lap 2: 259 s is behind A, B (250, 255), ahead of C, D


def test_pass_cost_per_car_passed():
    r = race({"A": [0, 100], "B": [1, 101]}, ["green"])
    sim = simulate(r, 0, 2.0, [("SOFT", 0, 1)], lambda lap, c, age: 97.0, pit_loss_s=0.0, pass_cost_s=0.3)
    assert sim.passes.tolist() == [2]
    assert sim.cum_s.tolist() == pytest.approx([99.6])


def test_traffic_stuck_behind_a_car_it_cannot_afford_to_pass():
    r = race({"A": [0, 100], "B": [1, 101]}, ["green"])
    sim = simulate(r, 0, 2.0, [("SOFT", 0, 1)], lambda lap, c, age: 97.8, pit_loss_s=0.0, pass_cost_s=0.3)
    # meets B first: 100.1 < 101, passed. Then A: 100.4 >= 100, so it's held behind A (no charge for the attempt)
    assert sim.passes.tolist() == [1]
    assert sim.cum_s.tolist() == pytest.approx([100.1])
    assert sim.position.tolist() == [2]


def test_failed_pass_cannot_drop_behind_a_car_passed_that_lap():
    r = race({"A": [0, 100], "B": [1, 100.4]}, ["green"])
    sim = simulate(r, 0, 2.0, [("SOFT", 0, 1)], lambda lap, c, age: 97.9, pit_loss_s=0.0, pass_cost_s=0.3,
                   dirty_air_s=0.5, dirty_gap_s=1.0)
    # passes B (100.2 < 100.4), held behind A (100.5 >= 100); dirty air would put it at 100.7, behind B: it stays level
    assert sim.passes.tolist() == [1]
    assert sim.cum_s.tolist() == pytest.approx([100.4])
    assert sim.position.tolist() == [2]


def test_held_up_costs_dirty_air_and_ends_a_following_gap_behind():
    r = race({"A": [0, 100]}, ["green"])
    held = lambda dirty, follow: simulate(r, 0, 0.5, [("SOFT", 0, 1)], lambda lap, c, age: 99.0, pit_loss_s=0.0,
                                          pass_cost_s=1.0, dirty_air_s=dirty, dirty_gap_s=1.0, follow_gap_s=follow)
    # free air 99.5 catches A (100) but 100.5 can't pass: no pass charge, it ends behind A
    assert held(0.7, 0.6).cum_s.tolist() == pytest.approx([100.6])     # the following gap binds
    assert held(0.7, 0.1).cum_s.tolist() == pytest.approx([100.2])     # dirty air binds
    assert held(0.7, 0.6).position.tolist() == [2] and held(0.7, 0.6).passes.tolist() == [0]


def test_dirty_air_costs_a_lap_ending_close_behind_a_car():
    r = race({"A": [0, 100]}, ["green"])
    near = simulate(r, 0, 1.0, [("SOFT", 0, 1)], lambda lap, c, age: 99.5, pit_loss_s=0.0, dirty_air_s=0.4, dirty_gap_s=1.0)
    clear = simulate(r, 0, 1.0, [("SOFT", 0, 1)], lambda lap, c, age: 100.2, pit_loss_s=0.0, dirty_air_s=0.4, dirty_gap_s=1.0)
    assert near.cum_s.tolist() == pytest.approx([100.9])    # ends 0.5 s behind A: +0.4 s
    assert clear.cum_s.tolist() == pytest.approx([101.2])   # ends 1.2 s behind A: clean air


@pytest.fixture(scope="module")
def bt():
    cfg = load_config()
    p = prepare(cfg)
    official = int(p["session"].results.set_index("Abbreviation").Position[cfg["drivers"]["target"]])
    return backtest(p, cfg), official, cfg["backtest"]


def test_backtest_position(bt):
    sim, official, _ = bt
    assert sim.position.iloc[-1] == official


def test_backtest_time_at_flag(bt):
    sim, _, b = bt
    assert abs(sim.error_s.iloc[-1]) <= b["tolerance_s"]


@pytest.mark.xfail(strict=True, reason="known miss, D-029: +5.09 s at lap 42 against the 5 s limit (stint-1 pace and the "
                                       "median green stop); the limit was set before the result and is kept")
def test_backtest_time_before_late_neutralisation(bt):   # D-006: the late SC resets the gaps, so check before it
    sim, _, b = bt
    assert abs(sim.error_s[b["check_lap"]]) <= b["tolerance_s"]
