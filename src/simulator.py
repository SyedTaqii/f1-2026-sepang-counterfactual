"""Generic ghost-car race engine: one car (any driver, any race, any tyre plan) raced lap by lap against the
real field's recorded times. The real cars don't react to it. Nothing race-specific lives here."""
import numpy as np
import pandas as pd


def race_from_laps(laps, neutral, wear):
    """The real field in simulator form, from FastF1-style laps (Driver, LapNumber, Time, PitInTime, PitOutTime).
    neutral: per lap 'green' | 'sc' | 'vsc'. wear: per-lap tyre-age increment."""
    wide = lambda values: laps.assign(v=values).pivot(index="LapNumber", columns="Driver", values="v")
    return {"times": wide(laps.Time.dt.total_seconds()), "pit_in": wide(laps.PitInTime.notna()).eq(True),
            "pit_out": wide(laps.PitOutTime.notna()).eq(True), "neutral": neutral, "wear": wear}


def simulate(race, start_lap, start_s, stints, lap_time, pit_loss_s, traffic_penalty_s=0.0):
    """Run the ghost from the end of `start_lap` (cumulative time `start_s`, same clock as race['times']) to the flag.

    race: times (lap x driver cumulative s, NaN = not running), pit_in / pit_out (lap x driver, in-/out-lap),
          neutral and wear (per lap); see race_from_laps.
    stints: [(compound, age at the end of start_lap or at the stop, last lap)]; the ghost stops after each
          stint's last lap except the final stint's.
    lap_time(lap, compound, age): the ghost's green-lap time.
    Green lap: lap_time, + pit_loss_s on an in-lap, + traffic (see _traffic).
    Neutralised lap: see _neutral_lap; a stop on one copies the real stoppers that lap (_stop_under_neutral).
    Returns one row per lap: cum_s, lap_s, position, compound, age, passes."""
    T, neutral = race["times"], race["neutral"]
    busy = race["pit_in"] | race["pit_out"]         # this lap's time includes a stop: not a reference, not a pass
    k, (comp, age, last) = 0, stints[0]
    t, out_lap, rows = start_s, None, []
    for n in range(start_lap + 1, int(T.index.max()) + 1):
        age += race["wear"][n]
        prev, now, t_prev, passes = T.loc[n - 1], T.loc[n], t, 0
        stop = n == last and k < len(stints) - 1
        if out_lap is not None:                       # second half of a stop under SC/VSC, already timed
            t, out_lap = out_lap, None
        elif stop and neutral[n] != "green":
            t, out_lap = _stop_under_neutral(T, race["pit_in"], n, t)
        elif neutral[n] == "green":
            t += lap_time(n, comp, age) + (pit_loss_s if stop else 0.0)
            t, passes = _traffic(prev, now, busy.loc[n], t_prev, t, traffic_penalty_s)
        else:
            t = _neutral_lap(prev, now, busy.loc[n], t, neutral[n])
        rows.append(dict(LapNumber=n, cum_s=t, lap_s=t - t_prev, position=1 + int((now < t).sum()),
                         compound=comp, age=age, passes=passes))
        if stop:
            k += 1
            comp, age, last = stints[k]
    return pd.DataFrame(rows).set_index("LapNumber")


def passes_per_lap(race):
    """Real on-track passes per lap (index) and driver (columns), counted the same way as the ghost's."""
    T, busy = race["times"], race["pit_in"] | race["pit_out"]
    out = pd.DataFrame(0, T.index, T.columns)
    for n in T.index[1:]:
        for d in T.columns:
            # ahead at the end of the previous lap, behind at the end of this one, not on an in- or out-lap
            out.loc[n, d] = int(((T.loc[n - 1] < T.at[n - 1, d]) & (T.loc[n] > T.at[n, d]) & ~busy.loc[n]).sum())
    return out


def _traffic(prev, now, busy, t_prev, t, penalty):
    """Each car the ghost catches this lap (ahead at the end of the last one, behind its free-air time now, not
    stopping), met in road order, costs `penalty`. Still ahead after paying: a pass. Not: it's stuck behind that
    car and can't reach the ones further up. Returns the ghost's time and its number of passes."""
    caught = (prev < t_prev) & (now > t) & ~busy
    passes = 0
    for drv in prev[caught].sort_values(ascending=False).index:
        t += penalty
        if t >= now[drv]:
            break
        passes += 1
    return t, passes


def _neutral_lap(prev, now, busy, t_prev, kind):
    """Follow the nearest car ahead on the road (the nearest behind, when leading), skipping cars on an in- or out-lap.
    SC: the gap shrinks to at most the field's median gap at the line that lap (the queue forming). VSC: frozen."""
    ok = prev.notna() & now.notna() & ~busy
    ahead = prev[ok & (prev < t_prev)]
    ref = ahead.idxmax() if len(ahead) else prev[ok].idxmin()
    gap = t_prev - prev[ref]                          # > 0 behind ref, < 0 ahead of it
    if kind == "sc":   # ponytail: a car far back closes up in one lap, so mid-SC positions are rough; the restart order is what the back-test checks
        gap = np.sign(gap) * min(abs(gap), np.median(np.diff(np.sort(now.dropna()))))
    return now[ref] + gap


def _stop_under_neutral(T, pit_in, n, t_prev):
    """In-lap and out-lap end times of a stop on neutralised lap n. What a stop costs then depends on where you
    were on the road when the SC/VSC came out, so interpolate (by time at the end of lap n-1) between the real
    cars that stopped that lap; clamp to the nearest one outside their range."""
    d = pit_in.columns[pit_in.loc[n]]
    d = d[T.loc[[n - 1, n, n + 1], d].notna().all().to_numpy()]
    if d.empty:
        raise ValueError(f"no real car stopped on neutralised lap {n}: nothing to copy")
    x = T.loc[n - 1, d].sort_values()
    return tuple(t_prev + np.interp(t_prev, x, T.loc[m, x.index] - x) for m in (n, n + 1))
