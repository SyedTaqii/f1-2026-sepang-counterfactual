"""Per-lap phase table: track condition (wet/dry), neutralisation (green/sc/vsc) and the evidence behind them."""
import numpy as np
import pandas as pd

from src.data_loader import SLICKS, crossover_laps, neutralised_laps, stint_table


def phase_table(session, cfg):
    """One row per lap.

    condition: 'wet' up to and including the crossover lap (median lap the leaders pitted off inters), 'dry' after.
    neutral:   'sc' / 'vsc' by majority TrackStatus, plus the start laps behind the SC (TrackStatus misses those).
    Evidence:  slick_minus_inter_s (median slick lap - median inter lap; crosses 0 near the crossover) and
               field_median_s (shows the track still drying after the crossover).
    No rain column: M1 showed no rainfall during the race, so the boundary comes from the leaders' stops.
    """
    laps, first = session.laps, cfg["race"]["first_racing_lap"]
    stints = stint_table(laps, laps.Driver.unique().tolist())
    cross = int(crossover_laps(laps, stints, first, cfg["race"]["leaders_n"]).median())

    t = pd.DataFrame(index=pd.RangeIndex(1, int(laps.LapNumber.max()) + 1, name="LapNumber"))
    flags = neutralised_laps(laps).reindex(t.index, fill_value=False)
    t["condition"] = np.where(t.index <= cross, "wet", "dry")
    t["neutral"] = np.select([t.index < first, flags.sc, flags.vsc], ["sc", "sc", "vsc"], "green")
    t["crossover"] = t.index == cross

    racing = laps[laps.PitInTime.isna() & laps.PitOutTime.isna() & laps.LapTime.notna()
                  & laps.Compound.isin(SLICKS + ["INTERMEDIATE"])]
    racing = racing.assign(slick=racing.Compound.isin(SLICKS), lap_s=racing.LapTime.dt.total_seconds(),
                           LapNumber=racing.LapNumber.astype(int))
    med = racing.groupby(["LapNumber", "slick"]).lap_s.median().unstack()
    green = t.neutral == "green"
    t["slick_minus_inter_s"] = (med[True] - med[False]).where(green & (t.condition == "wet")).round(2)
    t["field_median_s"] = racing.groupby("LapNumber").lap_s.median().where(green).round(2)
    return t
