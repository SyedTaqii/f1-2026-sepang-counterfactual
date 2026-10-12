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


def test_sc_closes_up_and_vsc_freezes():
    r = race({"A": [0, 100, 250, 380, 480], "B": [0, 101, 252, 382.5, 483], "C": [0, 102, 254, 385, 486]},
             ["green", "sc", "vsc", "green"])
    sim = simulate(r, 0, 0.0, [("SOFT", 0, 4)], lambda lap, c, age: 110.0, pit_loss_s=0.0)
    # 8 s behind C -> SC: 2 s (the field's median gap at the line that lap) -> VSC: frozen at 2 s
    assert sim.cum_s.tolist() == pytest.approx([110, 256, 387, 497])
    assert sim.position.tolist() == [4, 4, 4, 4]


def test_sc_leader_has_the_car_behind_close_up():
    r = race({"A": [0, 100, 250], "B": [0, 102, 253]}, ["green", "sc"])
    sim = simulate(r, 0, 0.0, [("SOFT", 0, 2)], lambda lap, c, age: 90.0, pit_loss_s=0.0)
    assert sim.cum_s.tolist() == pytest.approx([90, 247])   # 10 s lead -> 3 s
    assert sim.position.tolist() == [1, 1]


def test_stop_under_sc_interpolates_the_stoppers_around_it():
    r = race({"A": [0, 100, 250, 400, 500], "B": [0, 105, 255, 401, 501],
              "C": [0, 110, 262, 420, 520], "D": [0, 115, 265, 404, 504]},
             ["green", "sc", "sc", "green"], pit_in=[("A", 2), ("C", 2)], pit_out=[("A", 3), ("C", 3)])
    sim = simulate(r, 1, 107.5, [("INTERMEDIATE", 0, 2), ("SOFT", 0, 4)], lambda lap, c, age: 100.0, pit_loss_s=99.0)
    # 3/4 of the way from A to C on the road: in-lap 150 + 0.75 * 2, in + out 300 + 0.75 * 10. pit_loss_s is unused.
    assert sim.cum_s.tolist() == pytest.approx([259, 415, 515])
    assert sim.compound.tolist() == ["INTERMEDIATE", "SOFT", "SOFT"]
    assert sim.position.tolist() == [3, 4, 4]          # lap 2: 259 s is behind A, B (250, 255), ahead of C, D


def test_traffic_penalty_per_car_passed():
    r = race({"A": [0, 100], "B": [1, 101]}, ["green"])
    sim = simulate(r, 0, 2.0, [("SOFT", 0, 1)], lambda lap, c, age: 97.0, pit_loss_s=0.0, traffic_penalty_s=0.3)
    assert sim.passes.tolist() == [2]
    assert sim.cum_s.tolist() == pytest.approx([99.6])


def test_traffic_stuck_behind_a_car_it_cannot_afford_to_pass():
    r = race({"A": [0, 100], "B": [1, 101]}, ["green"])
    sim = simulate(r, 0, 2.0, [("SOFT", 0, 1)], lambda lap, c, age: 97.8, pit_loss_s=0.0, traffic_penalty_s=0.3)
    # meets B first: 100.1 < 101, passed. Then A: 100.4 >= 100, stuck behind it (and charged for the attempt)
    assert sim.passes.tolist() == [1]
    assert sim.cum_s.tolist() == pytest.approx([100.4])
    assert sim.position.tolist() == [2]


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


def test_backtest_time_before_late_neutralisation(bt):   # D-006: the late SC resets the gaps, so check before it
    sim, _, b = bt
    assert abs(sim.error_s[b["check_lap"]]) <= b["tolerance_s"]
