"""M1: the loader returns the race we think it does. First run downloads into cache/."""
import pytest

from src.data_loader import fix_timing, lap_ranges, load_config, load_session, neutralised_laps, timing_vs_official


def test_lap_ranges():
    assert lap_ranges([45, 9, 10, 11, 12, 46, 50]) == "9-12, 45-46, 50"
    assert lap_ranges([]) == "none"


@pytest.fixture(scope="module")
def session():
    return load_session(load_config())


def test_race_facts(session):
    res = session.results.set_index("Abbreviation")
    assert session.laps.LapNumber.max() == 55
    assert res.Position[["VER", "ANT", "HAM", "LEC"]].tolist() == [1, 2, 3, 4]
    assert res.GridPosition["HAM"] == 2
    flags = neutralised_laps(session.laps)
    assert flags.index[flags.sc].tolist() == [9, 10, 11, 12, 45, 46, 47, 48, 49, 50, 51]


def test_only_lec_timing_is_off(session):
    tv = timing_vs_official(session.laps, session.results)
    assert tv.index[tv.error.abs() > 1.0].tolist() == ["LEC"]


def test_timing_fix_matches_official(session):
    tv = timing_vs_official(fix_timing(session, load_config()), session.results)
    assert tv.error.abs().max() < 0.2      # LEC now as close as everyone else (within 0.1 s)
