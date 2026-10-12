"""Wires the race data and the M2 models into the simulator: the shared inputs and the back-test (M3).
`python -m src.pipeline` prints the back-test lap by lap."""
from src.data_loader import fix_timing, load_session
from src.pace_model import add_lap_features, dry_lap_time, fit_pace, lap_effects, lap_wear, pass_cost
from src.pit_loss import pit_stops
from src.segmentation import phase_table
from src.simulator import passes_per_lap, race_from_laps, simulate


def prepare(cfg):
    """Everything the simulator needs, built once. The real field excludes the target: the ghost takes his place."""
    target = cfg["drivers"]["target"]
    session = load_session(cfg)
    phases = phase_table(session, cfg)
    df = add_lap_features(fix_timing(session, cfg), phases, cfg)
    fit = fit_pace(df, cfg)
    effects = lap_effects(df, fit, cfg)
    stops = pit_stops(df, cfg)
    wear = lap_wear(phases, cfg)
    return {"session": session, "phases": phases, "df": df, "fit": fit, "effects": effects,
            "n_laps": int(df.LapNumber.max()),
            "green_loss": stops[(stops.neutral == "green") & (stops.Driver != target)].loss_s,
            "pass_cost": pass_cost(df, fit, effects, passes_per_lap(race_from_laps(df, phases.neutral, wear)), cfg),
            "race": race_from_laps(df[df.Driver != target], phases.neutral, wear)}


def traffic_penalty(p, cfg):
    """sim.traffic_penalty_s if set, else the field-measured cost per pass."""
    v = cfg["sim"]["traffic_penalty_s"]
    return p["pass_cost"]["mean"] if v is None else v


def real_stints(me, start):
    """A driver's real stints from the end of lap `start`: [(compound, age at the stint's start, last lap)]."""
    out = []
    for _, s in me.loc[start:].groupby("Stint"):
        age = s.eff_age.iloc[0] if s.index[0] == start else s.TyreLife.iloc[0] - 1
        out.append((s.Compound.iloc[0], float(age), int(s.index[-1])))
    return out


def backtest(p, cfg):
    """Replay the target's real strategy from backtest.start_lap through the simulator (mean pace, median green
    pit loss, no noise) and line it up with his real race."""
    start = cfg["backtest"]["start_lap"]
    df = p["df"]
    me = df[df.Driver == cfg["drivers"]["target"]].set_index("LapNumber")
    sim = simulate(p["race"], start, me.Time[start].total_seconds(), real_stints(me, start),
                   dry_lap_time(p["fit"]["coef"], p["effects"], p["n_laps"]), p["green_loss"].median(),
                   traffic_penalty(p, cfg))
    real = me.loc[sim.index]
    return sim.assign(real_cum_s=real.Time.dt.total_seconds(), real_lap_s=real.lap_s, real_position=real.Position,
                      error_s=lambda s: s.cum_s - s.real_cum_s)


if __name__ == "__main__":
    import fastf1
    import pandas as pd
    from src.data_loader import load_config

    fastf1.set_log_level("ERROR")
    pd.set_option("display.width", 200)
    cfg = load_config()
    sim = backtest(prepare(cfg), cfg)
    print(sim.round(2).to_string())
