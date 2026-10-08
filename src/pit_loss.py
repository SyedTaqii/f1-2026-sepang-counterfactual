"""Pit-stop time loss (in-lap + out-lap vs a reference pace), split by green / SC / VSC."""
import fastf1
import pandas as pd

from src.data_loader import load_config, load_session
from src.pace_model import add_lap_features
from src.segmentation import phase_table


def pit_stops(df, cfg):
    """One row per stop: loss_s = (in_s - ref_in_s) + (out_s - ref_out_s).

    Green stops use the driver's own clean-lap median within ref_window laps; SC/VSC stops use the median
    lap of the cars that did not pit on that lap. Lap time falls back to the Time difference when lap_s is NaN."""
    first, window = cfg["race"]["first_racing_lap"], cfg["pit_loss"]["ref_window"]
    df = df.sort_values(["Driver", "LapNumber"]).copy()
    prev_time = df.groupby("Driver").Time.shift()
    df["t_s"] = df.lap_s.fillna((df.Time - prev_time).dt.total_seconds())
    by_lap = df.set_index(["Driver", "LapNumber"])
    free = df[df.PitInTime.isna() & df.PitOutTime.isna()].groupby("LapNumber").t_s.median()

    rows = []
    for (drv, n), a in by_lap[by_lap.PitInTime.notna()].iterrows():
        if n <= first or (drv, n + 1) not in by_lap.index or pd.isna(by_lap.loc[(drv, n + 1), "PitOutTime"]):
            continue
        b = by_lap.loc[(drv, n + 1)]
        kinds = {a.neutral, b.neutral}
        neutral = "sc" if "sc" in kinds else "vsc" if "vsc" in kinds else "green"
        if neutral == "green":
            mine = df[(df.Driver == drv) & df.clean & df.LapNumber.between(n - window, n + window)]
            if mine.empty:
                continue
            ref_in = ref_out = mine.lap_s.median()
        else:
            ref_in, ref_out = free.get(n), free.get(n + 1)
        rows.append(dict(Driver=drv, in_lap=n, neutral=neutral, in_s=a.t_s, out_s=b.t_s,
                         ref_in_s=ref_in, ref_out_s=ref_out,
                         loss_s=(a.t_s - ref_in) + (b.t_s - ref_out)))
    return pd.DataFrame(rows).sort_values(["in_lap", "Driver"]).reset_index(drop=True)


def pit_loss_summary(stops):
    """Count, median, min and max loss per neutralisation type."""
    s = stops.groupby("neutral").loss_s.agg(n="count", median_s="median", min_s="min", max_s="max")
    return s.round(2)


if __name__ == "__main__":
    fastf1.set_log_level("ERROR")
    cfg = load_config()
    session = load_session(cfg)
    stops = pit_stops(add_lap_features(session.laps, phase_table(session, cfg), cfg), cfg)
    print(stops.round(2).to_string())
    print(pit_loss_summary(stops).to_string())
