"""FastF1 load + cache, and the M1 fact check (`python -m src.data_loader`)."""
from pathlib import Path

import fastf1
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SLICKS = ["SOFT", "MEDIUM", "HARD"]


def load_config(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text())


def load_session(cfg):
    """Load the race by year + round (the event name is ambiguous: 'Bahrain GP' ran at Sepang)."""
    cache = ROOT / cfg["cache_dir"]
    cache.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    r = cfg["race"]
    session = fastf1.get_session(r["year"], r["round"], r["session"])
    location = session.event["Location"]
    assert location == r["expected_location"], f"round {r['round']} is at {location}, expected {r['expected_location']}"
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    # If the classification API is unreachable, FastF1 silently rebuilds results from timing data
    # (no grid, no status). That put LEC P17 instead of P4 here, so refuse to continue on it.
    if session.results["Status"].isna().any():
        raise RuntimeError("Official classification not loaded (FastF1 fell back to timing data). Retry later.")
    return session


def stint_table(laps, drivers):
    """One row per stint: compound and first/last lap."""
    return (laps.pick_drivers(drivers)
            .groupby(["Driver", "Stint"])
            .agg(compound=("Compound", "first"), first_lap=("LapNumber", "min"), last_lap=("LapNumber", "max"))
            .replace({"compound": {"None": "UNKNOWN"}})  # FastF1 stores unknown compounds as the *string* 'None'
            .reset_index()
            .astype({"Stint": int, "first_lap": int, "last_lap": int}))


def neutralised_laps(laps):
    """Per lap, whether most cars saw SC (4), VSC (6/7) or red (5) in their TrackStatus string.
    Majority, not "any": a back-marker's lap 8 can overlap the leaders' lap 9 SC."""
    codes = {"sc": "4", "vsc": "[67]", "red": "5"}
    status = laps.dropna(subset=["TrackStatus"]).groupby("LapNumber").TrackStatus
    flags = pd.DataFrame({k: status.apply(lambda s, c=c: s.str.contains(c).mean() > 0.5) for k, c in codes.items()})
    flags.index = flags.index.astype(int)
    return flags


def lap_ranges(laps):
    """[9, 10, 11, 45, 46] -> '9-11, 45-46'."""
    laps = sorted(laps)
    if not laps:
        return "none"
    runs, start = [], laps[0]
    for prev, cur in zip(laps, laps[1:] + [None]):
        if cur != prev + 1:
            runs.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = cur
    return ", ".join(runs)


def crossover_laps(laps, stints, lap, top_n):
    """Pit-in lap (= last lap on inters) of the top_n cars that were on inters at `lap`."""
    on_inters = laps[(laps.LapNumber == lap) & (laps.Compound == "INTERMEDIATE")].nsmallest(top_n, "Position").Driver
    inter = stints[stints.Driver.isin(on_inters) & (stints.compound == "INTERMEDIATE")]
    return inter.groupby("Driver").last_lap.max().reindex(on_inters)


def official_gaps(results):
    """Official gap to the winner in seconds, per finisher (the winner's Time is the total race time)."""
    res = results.set_index("Abbreviation")
    finished = res[res.Status == "Finished"]
    return finished.Time.dt.total_seconds().mask(finished.Position == 1, 0.0)


def timing_vs_official(session):
    """Finish gap to the winner from FastF1 lap timing vs the official classification, per finisher."""
    official = official_gaps(session.results)
    last = session.laps.sort_values("LapNumber").groupby("Driver").tail(1).set_index("Driver").Time
    timing = (last - last[official.index[0]]).dt.total_seconds().reindex(official.index)
    return pd.DataFrame({"official_gap": official, "timing_gap": timing, "error": timing - official})


def fact_check(session, cfg):
    """Print the M1 fact table: everything the PRD's race story claims, checked against the data."""
    laps, res, fc = session.laps, session.results, cfg["fact_check"]
    target, mate = cfg["drivers"]["target"], cfg["drivers"]["teammate"]
    first = cfg["race"]["first_racing_lap"]
    section = lambda title: print(f"\n=== {title} ===")

    section("Event")
    print(f"{session.event['EventName']} | {session.event['Location']} | {session.event['EventDate']:%Y-%m-%d} | "
          f"{int(laps.LapNumber.max())} laps")

    section("Official classification (top 6 + Ferraris)")
    keep = res[(res.Position <= 6) | res.Abbreviation.isin([target, mate])].set_index("Abbreviation")
    keep = keep.assign(gap_s=official_gaps(res))[["TeamName", "GridPosition", "Position", "Status", "gap_s"]]
    print(keep.to_string())

    section("How the race started (race control)")
    rcm = session.race_control_messages
    start_msgs = rcm[rcm.Message.str.contains("START|FORMATION|SUSPENDED|SAFETY CAR LIGHTS")]
    print(start_msgs[["Lap", "Message"]].to_string(index=False))
    print(f"-> laps 1-{first - 1} behind the SC, first racing lap = {first} (config race.first_racing_lap)")

    section("Neutralisations (majority of cars' TrackStatus)")
    flags = neutralised_laps(laps)
    for kind in flags.columns:
        print(f"{kind.upper():>4}: laps {lap_ranges(flags.index[flags[kind]].tolist())}")
    print(f"note: the SC at the start (laps 1-{first - 1}) is NOT in TrackStatus, only in race control")

    section("Rain")
    w = session.weather_data
    wet = w[w.Rainfall]
    print(f"rainfall from {wet.Time.min()} to {wet.Time.max()} (session time); race started {session.session_start_time}")
    print(f"-> rain during the race: {bool((wet.Time >= session.session_start_time).any())}")

    section(f"Stints: {target}, {mate} + top {fc['top_n']} finishers")
    drivers = [target, mate] + res.nsmallest(fc["top_n"], "Position").Abbreviation.tolist()
    stints = stint_table(laps, list(dict.fromkeys(drivers)))
    print(stints.to_string(index=False))

    section(f"Crossover: when the top {fc['top_n']} on inters at lap {first} pitted for slicks")
    cross = crossover_laps(laps, stint_table(laps, laps.Driver.unique().tolist()), first, fc["top_n"])
    print(cross.to_string())
    print(f"-> crossover lap (median pit-in) = {int(cross.median())}; SC on that lap: {bool(flags.sc.get(int(cross.median())))}")

    section(f"{target} running position")
    pos = laps.pick_drivers(target).set_index("LapNumber").Position
    grid = int(res.set_index("Abbreviation").GridPosition[target])
    print(f"grid P{grid} | " + " | ".join(f"lap {n}: P{int(pos[n])}" for n in fc["position_laps"]))
    print(f"ever on inters: {(laps.pick_drivers(target).Compound == 'INTERMEDIATE').any()}")

    section(f"Data quality: FastF1 timing gap vs official (|error| > {fc['timing_tolerance_s']} s)")
    tv = timing_vs_official(session)
    bad = tv[tv.error.abs() > fc["timing_tolerance_s"]]
    print(bad.round(3).to_string() if len(bad) else "all finishers within tolerance")
    print(f"max |error| among the rest: {tv.drop(bad.index).error.abs().max():.3f} s")


if __name__ == "__main__":
    cfg = load_config()
    fact_check(load_session(cfg), cfg)
