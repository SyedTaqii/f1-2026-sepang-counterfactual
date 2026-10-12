"""Clean-lap filter, dry pace fit, inter-pace estimate and standing-start gaps."""
import numpy as np
import pandas as pd

from src.data_loader import inter_leaders


def lap_wear(phases, cfg):
    """Tyre-age increment per lap: 1 on a green dry lap, pace.wet_sc_wear on a wet or neutralised one."""
    green_dry = (phases.condition == "dry") & (phases.neutral == "green")
    return pd.Series(np.where(green_dry, 1.0, cfg["pace"]["wet_sc_wear"]), phases.index)


def add_lap_features(laps, phases, cfg):
    """Copy of `laps` with lap_s (lap time, s), gap_ahead (s), condition/neutral (from the phase table),
    eff_age (effective tyre age) and clean (usable for pace fits)."""
    p, first = cfg["pace"], cfg["race"]["first_racing_lap"]
    df = laps.copy()
    df["LapNumber"] = df.LapNumber.astype(int)
    df["lap_s"] = df.LapTime.dt.total_seconds()
    # Gap to whichever car crossed the line just before, lapped or not: that's the car physically ahead.
    df["gap_ahead"] = df.Time.sort_values().diff().dt.total_seconds()
    df["condition"] = phases.condition.reindex(df.LapNumber).to_numpy()
    df["neutral"] = phases.neutral.reindex(df.LapNumber).to_numpy()

    # Effective tyre age: laps the set did before this stint count fully; in this stint, a wet or
    # neutralised lap counts wet_sc_wear of a green dry lap. With wet_sc_wear = 1 this equals TyreLife.
    wear = lap_wear(phases, cfg).reindex(df.LapNumber).to_numpy()
    stint = df.groupby(["Driver", "Stint"])
    df["eff_age"] = stint.TyreLife.transform("first") - 1 + pd.Series(wear, df.index).groupby([df.Driver, df.Stint]).cumsum()

    own_neutral = df.TrackStatus.fillna("").str.contains("[4567]")  # this car's lap overlapped an SC/VSC
    df["clean"] = (
        df.lap_s.notna()
        & df.PitInTime.isna() & df.PitOutTime.isna()      # not an in- or out-lap
        & (df.LapNumber > first)                           # not behind the start SC, not the standing-start lap
        & (df.neutral == "green") & ~own_neutral
        & (df.gap_ahead >= p["traffic_gap_s"])              # clean air
        & df.IsAccurate.fillna(False).astype(bool)
        & ~df.Deleted.fillna(False).astype(bool)           # track-limits laps
    )
    return df


def design(d, n_laps, target):
    """The dry fit's regressors for the laps in d (see fit_pace)."""
    comp = pd.get_dummies(d.Compound.replace("None", "UNKNOWN"), dtype=float)
    return pd.concat([comp, comp.mul(d.eff_age, axis=0).add_prefix("deg_"),
                      (n_laps - d.LapNumber).rename("remaining").astype(float),
                      pd.get_dummies(d.Driver, dtype=float).drop(columns=target, errors="ignore").add_prefix("d_")], axis=1)


def fit_pace(df, cfg):
    """Field-wide OLS on every driver's clean dry laps from dry_fit_start_lap:

        lap_s = <compound> + deg_<compound> * eff_age + remaining * laps_remaining + d_<driver>

    The target has d = 0, so <compound> is its pace on fresh tyres on the last lap, and d_<drv> is everyone
    else's offset to it (positive = slower). `remaining` is fuel burn-off and the track rubbering in, together.
    Only the whole field separates it from tyre deg: on any one lap, 20 cars run very different tyre ages.
    Laps with unknown tyre age are dropped. The coefficient covariance and the target's residual sd are the
    uncertainty the Monte Carlo samples."""
    target = cfg["drivers"]["target"]
    d = df[df.clean & (df.condition == "dry") & (df.LapNumber >= cfg["pace"]["dry_fit_start_lap"])]
    d = d.dropna(subset=["eff_age"])
    X = design(d, df.LapNumber.max(), target)
    coef, *_ = np.linalg.lstsq(X, d.lap_s, rcond=None)
    resid = d.lap_s - X @ coef
    s2 = resid @ resid / (len(d) - X.shape[1])
    return {"coef": pd.Series(coef, X.columns), "cov": pd.DataFrame(s2 * np.linalg.inv(X.T @ X), X.columns, X.columns),
            "resid_sd": float(resid[d.Driver == target].std()), "n": len(d), "n_target": int((d.Driver == target).sum())}


def residuals(d, fit, n_laps, target):
    """Lap time minus the fit's prediction, for the laps in d whose driver and compound are in the fit."""
    X = design(d, n_laps, target)
    keep = X[X.columns.difference(fit["coef"].index)].eq(0).all(axis=1)
    return d.lap_s[keep] - X[keep].reindex(columns=fit["coef"].index, fill_value=0.0) @ fit["coef"]


def lap_effects(df, fit, cfg):
    """Median residual of the field's clean dry laps, per lap: what every car felt that the fit can't see
    (the track still drying on laps 13-16, yellow flags)."""
    d = df[df.clean & (df.condition == "dry")].dropna(subset=["eff_age"])
    r = residuals(d, fit, df.LapNumber.max(), cfg["drivers"]["target"])
    return r.groupby(d.LapNumber[r.index]).median()


def traffic_costs(df, fit, effects, passes, incident_laps, cfg):
    """What traffic costs a car, from the other drivers: OLS of (lap time - fit - lap effect) on
        pass  = cars it passed that lap (simulator.passes_per_lap: lap x driver counts)
        dirty = 1 if it ended the lap within traffic_gap_s of the car ahead (dirty air)
    over every green dry lap without a stop. The two are fitted together: a car stuck behind another loses time on
    laps with no pass. Laps on which a car's timing stops (incident_laps: a crawl back to retire, yellow flags) are
    left out: cars there lose 10-30 s to the incident, not to traffic. The target is left out too, so the
    back-test checks these numbers instead of being fitted by them.
    Returns coef and cov (pass, dirty) for sampling, and the counts behind them."""
    target = cfg["drivers"]["target"]
    d = df[(df.condition == "dry") & (df.neutral == "green") & df.PitInTime.isna() & df.PitOutTime.isna()
           & (df.Driver != target) & ~df.LapNumber.isin(incident_laps)].dropna(subset=["lap_s", "eff_age"])
    r = residuals(d, fit, df.LapNumber.max(), target)
    d = d.loc[r.index]
    y = (r - d.LapNumber.map(effects).fillna(0.0)).to_numpy()
    x = pd.DataFrame({"pass": passes.stack().reindex(pd.MultiIndex.from_arrays([d.LapNumber, d.Driver])).to_numpy(float),
                      "dirty": (d.gap_ahead < cfg["pace"]["traffic_gap_s"]).to_numpy(float)})
    A = np.c_[np.ones(len(x)), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ coef
    cov = e @ e / (len(y) - A.shape[1]) * np.linalg.inv(A.T @ A)
    return {"coef": pd.Series(coef[1:], x.columns), "cov": pd.DataFrame(cov[1:, 1:], x.columns, x.columns),
            "n_passes": int(x["pass"].sum()), "n_dirty": int(x.dirty.sum()), "n": len(x)}


def dry_lap_time(coef, effects, n_laps):
    """The target's green dry lap time, lap_time(lap, compound, age), for the simulator. coef can be a sampled set."""
    return lambda lap, comp, age: (coef[comp] + coef[f"deg_{comp}"] * age + coef["remaining"] * (n_laps - lap)
                                   + effects.get(lap, 0.0))


def se(fit, name):
    return float(np.sqrt(fit["cov"].loc[name, name]))


def inter_pace(df, cfg):
    """The target's inter pace as an offset (s/lap) to `ref`, the leaders' median clean inter lap on each wet lap.
    `ref` moves lap by lap with the drying track; the offset is assumed constant.

    proxy: the teammate's clean inter laps minus ref, plus the target-teammate gap on clean dry laps.
    ratio: the target's dry gap to the leaders, scaled up by the wet/dry lap-length ratio.
    The two disagree for real reasons, so sigma = sqrt(mean within-method variance + (half the disagreement)^2)."""
    target, mate = cfg["drivers"]["target"], cfg["drivers"]["teammate"]
    leaders = inter_leaders(df, cfg["race"]["first_racing_lap"], cfg["race"]["leaders_n"])
    wet = df[df.clean & (df.condition == "wet") & (df.Compound == "INTERMEDIATE")]
    ref = wet[wet.Driver.isin(leaders)].groupby("LapNumber").lap_s.median()
    field = fit_pace(df, cfg)

    diffs = (wet[wet.Driver == mate].set_index("LapNumber").lap_s - ref).dropna()
    proxy = diffs.mean() - field["coef"][f"d_{mate}"]
    proxy_se = np.sqrt(diffs.var() / len(diffs) + se(field, f"d_{mate}") ** 2)

    names = [f"d_{d}" for d in leaders]
    dry_gap = -field["coef"][names].mean()                       # target minus leaders' average, dry
    dry_gap_se = np.sqrt(field["cov"].loc[names, names].to_numpy().sum()) / len(names)
    dry_lap = df[df.clean & (df.condition == "dry") & df.Driver.isin(leaders)
                 & (df.LapNumber >= cfg["pace"]["dry_fit_start_lap"])].lap_s.median()
    scale = ref.mean() / dry_lap
    ratio, ratio_se = dry_gap * scale, dry_gap_se * scale

    w = cfg["pace"]["proxy_weight"]
    mu = w * proxy + (1 - w) * ratio
    sigma = np.sqrt((proxy_se ** 2 + ratio_se ** 2) / 2 + ((proxy - ratio) / 2) ** 2)
    return {"ref": ref, "leaders": leaders, "proxy": proxy, "proxy_se": proxy_se, "proxy_laps": diffs,
            "ratio": ratio, "ratio_se": ratio_se, "mu": mu, "sigma": sigma, "lap_sd": float(diffs.std())}


def start_gaps(df, results, cfg):
    """Gap to the leader at the end of the first racing lap (the standing start) for cars that started on inters
    from grid P1..comparable_grid_max. The ghost's start is sampled from these."""
    lap = df[df.LapNumber == cfg["race"]["first_racing_lap"]].set_index("Driver")
    grid = results.set_index("Abbreviation").GridPosition.reindex(lap.index)
    gap = (lap.Time - lap.Time.min()).dt.total_seconds()
    started_on_inters = (lap.Compound == "INTERMEDIATE") & (lap.Stint == 1)  # not those who pitted for them behind the SC
    keep = started_on_inters & (grid <= cfg["start"]["comparable_grid_max"])
    return pd.DataFrame({"grid": grid, "pos": lap.Position, "gap_s": gap})[keep].sort_values("grid")


if __name__ == "__main__":
    import fastf1
    from src.data_loader import fix_timing, load_config, load_session
    from src.segmentation import phase_table

    fastf1.set_log_level("ERROR")
    cfg = load_config()
    session = load_session(cfg)
    df = add_lap_features(fix_timing(session, cfg), phase_table(session, cfg), cfg)
    target, mate = cfg["drivers"]["target"], cfg["drivers"]["teammate"]

    print(f"clean laps: {int(df.clean.sum())} of {len(df)}")
    fit = fit_pace(df, cfg)
    ip = inter_pace(df, cfg)
    print(f"\n=== Dry pace fit (field-wide, {fit['n']} laps, {fit['n_target']} of them {target}; "
          f"{target} residual sd {fit['resid_sd']:.3f} s) ===")
    table = pd.DataFrame({"coef": fit["coef"], "se": np.sqrt(np.diag(fit["cov"]))}).round(3)
    shown = [f"d_{d}" for d in [mate] + ip["leaders"]]
    print(table[~table.index.str.startswith("d_") | table.index.isin(shown)].to_string())

    print(f"\n=== Inter pace: {target} offset to the leaders' median inter lap ({', '.join(ip['leaders'])}) ===")
    print("reference (s):", ip["ref"].round(2).to_dict())
    print(f"{mate} minus reference per lap: {ip['proxy_laps'].round(2).to_dict()}")
    print(f"proxy: {ip['proxy']:+.2f} +/- {ip['proxy_se']:.2f}   ratio: {ip['ratio']:+.2f} +/- {ip['ratio_se']:.2f}"
          f"   ->  Normal(mu={ip['mu']:+.2f}, sigma={ip['sigma']:.2f}) s/lap, lap noise sd {ip['lap_sd']:.2f}")

    print("\n=== Standing-start gaps (inter starters, front of grid) ===")
    print(start_gaps(df, session.results, cfg).round(2).to_string())
