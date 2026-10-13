"""M2: add_lap_features on tiny hand-built laps (no download)."""
import pandas as pd
import pytest

from src.pace_model import add_lap_features

CFG = {"race": {"first_racing_lap": 3}, "pace": {"traffic_gap_s": 1.0, "wet_sc_wear": 0.0}}


def make_laps(rows):
    base = dict(Driver="A", Stint=1, PitInTime=pd.NaT, PitOutTime=pd.NaT, TrackStatus="1", IsAccurate=True, Deleted=False)
    out = []
    for r in rows:
        r = {**base, **r}
        r.setdefault("LapTime", pd.Timedelta(seconds=100))
        r.setdefault("Time", pd.Timedelta(seconds=100 * r["LapNumber"]))
        r.setdefault("TyreLife", float(r["LapNumber"]))
        out.append(r)
    return pd.DataFrame(out)


def phases(n=6, **over):
    ph = pd.DataFrame({"condition": "dry", "neutral": "green"}, index=pd.Index(range(1, n + 1), name="LapNumber"))
    for lap, (c, ne) in over.items():
        ph.loc[int(lap[1:])] = [c, ne]
    return ph


def clean5(change, ph=None):
    """`clean` of lap 5 for driver A, with `change` applied to it (lap 4 gives lap 5 a car ahead, 100 s away)."""
    laps = make_laps([{"LapNumber": 4}, {"LapNumber": 5, **change}])
    return bool(add_lap_features(laps, phases() if ph is None else ph, CFG).clean.iloc[1])


def test_normal_lap_is_clean():
    assert clean5({})


@pytest.mark.parametrize("change", [
    {"PitInTime": pd.Timedelta(seconds=490)},
    {"PitOutTime": pd.Timedelta(seconds=410)},
    {"TrackStatus": "16"},
    {"IsAccurate": False},
    {"Deleted": True},
])
def test_unclean_laps(change):
    assert not clean5(change)


def test_first_racing_lap_and_sc_phase():
    laps = make_laps([{"LapNumber": 2}, {"LapNumber": 3}])
    assert not add_lap_features(laps, phases(), CFG).clean.iloc[1]
    assert not clean5({}, phases(L5=("dry", "sc")))


def test_traffic():
    laps = make_laps([{"LapNumber": 4}, {"LapNumber": 5, "Driver": "A"},
                      {"LapNumber": 5, "Driver": "B", "Time": pd.Timedelta(seconds=500.5)}])
    assert add_lap_features(laps, phases(), CFG).clean.tolist() == [False, True, False]


@pytest.mark.parametrize("wear, first_life, expected", [
    (0.0, 1, [0, 0, 0, 1, 2, 3]),
    (1.0, 1, [1, 2, 3, 4, 5, 6]),
    (1.0, 4, [4, 5, 6, 7, 8, 9]),
])
def test_eff_age(wear, first_life, expected):
    laps = make_laps([{"LapNumber": n, "TyreLife": float(n + first_life - 1)} for n in range(1, 7)])
    ph = phases(L1=("wet", "green"), L2=("wet", "green"), L3=("dry", "sc"))
    cfg = {**CFG, "pace": {**CFG["pace"], "wet_sc_wear": wear}}
    assert add_lap_features(laps, ph, cfg).eff_age.tolist() == expected


def test_stint_pace_recovers_the_targets_faster_second_stint():
    from src.pace_model import dry_lap_time, fit_pace
    rows = []
    for drv, offset, gap in (("A", 0.0, 0), ("B", 0.4, 6)):            # B is 0.4 s slower and on older tyres
        for lap in range(17, 41):
            stint = 1 if lap <= 30 else 2
            age = lap - (16 if stint == 1 else 30) + gap
            faster = -0.7 if drv == "A" and stint == 2 else 0.0           # the target's second stint is 0.7 s quicker
            rows.append(dict(Driver=drv, LapNumber=lap, Stint=stint, Compound="SOFT", eff_age=float(age), clean=True,
                             condition="dry", lap_s=100 + 0.15 * age + 0.05 * (40 - lap) + offset + faster))
    df = pd.DataFrame(rows)
    cfg = {"drivers": {"target": "A"}, "pace": {"dry_fit_start_lap": 17, "stint_pace": True}}
    coef = fit_pace(df, cfg)["coef"]
    assert coef["stint_2"] == pytest.approx(-0.7)
    assert coef["d_B"] == pytest.approx(0.4)
    lap_time = dry_lap_time(coef, {}, 40, stint_of={35: 2})
    assert lap_time(35, "SOFT", 5) == pytest.approx(100 + 0.75 + 0.25 - 0.7)
    assert "stint_2" not in fit_pace(df, {**cfg, "pace": {**cfg["pace"], "stint_pace": False}})["coef"]
