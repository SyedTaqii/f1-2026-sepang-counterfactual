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


def simulate(race, start_lap, start_s, stints, lap_time, pit_loss_s, pass_cost_s=0.0, dirty_air_s=0.0, dirty_gap_s=0.0):
    """Run the ghost from the end of `start_lap` (cumulative time `start_s`, same clock as race['times']) to the flag.

    race: times (lap x driver cumulative s, NaN = not running), pit_in / pit_out (lap x driver, in-/out-lap),
          neutral and wear (per lap); see race_from_laps.
    stints: [(compound, age at the end of start_lap or at the stop, last lap)]; the ghost stops after each
          stint's last lap except the final stint's.
    lap_time(lap, compound, age): the ghost's green-lap time.
    Green lap: lap_time, + pit_loss_s on an in-lap, + traffic: pass_cost_s per car caught, dirty_air_s if it ends
    the lap within dirty_gap_s of a car ahead (see _traffic). The defaults are free passing and no dirty air.
    Neutralised lap: the ghost copies the real cars around it on the road (_copy): the ones on a normal lap or,
    when it stops, the ones that stopped that lap (over the in-lap and the out-lap).
    Returns one row per lap: cum_s, lap_s, position, compound, age, passes."""
    T, neutral = race["times"], race["neutral"]
    busy = _busy(race)
    k, (comp, age, last) = 0, stints[0]
    t, out_lap, rows = start_s, None, []
    for n in range(start_lap + 1, int(T.index.max()) + 1):
        age += race["wear"][n]
        prev, now, t_prev, passes = T.loc[n - 1], T.loc[n], t, 0
        stop = n == last and k < len(stints) - 1
        if out_lap is not None:                       # second half of a stop under SC/VSC, already timed
            t, out_lap = out_lap, None
        elif stop and neutral[n] != "green":
            stoppers = race["pit_in"].loc[n] & T.loc[n - 1:n + 1].notna().all()
            t, out_lap = [_copy(T, stoppers, n, t, m) for m in (n, n + 1)]
        elif neutral[n] == "green":
            t += lap_time(n, comp, age) + (pit_loss_s if stop else 0.0)
            t, passes = _traffic(prev, now, busy.loc[n], t_prev, t, pass_cost_s, dirty_air_s, dirty_gap_s)
        else:
            t = _copy(T, prev.notna() & now.notna() & ~busy.loc[n], n, t, n)
        rows.append(dict(LapNumber=n, cum_s=t, lap_s=t - t_prev, position=1 + int((now < t).sum()),
                         compound=comp, age=age, passes=passes))
        if stop:
            k += 1
            comp, age, last = stints[k]
    return pd.DataFrame(rows).set_index("LapNumber")


def passes_per_lap(race):
    """Real on-track passes per lap (index) and driver (columns), counted the same way as the ghost's."""
    T, busy = race["times"], _busy(race)
    out = pd.DataFrame(0, T.index, T.columns)
    for n in T.index[1:]:
        for d in T.columns:
            # ahead at the end of the previous lap, behind at the end of this one, on a normal lap (see _busy)
            out.loc[n, d] = int(((T.loc[n - 1] < T.at[n - 1, d]) & (T.loc[n] > T.at[n, d]) & ~busy.loc[n]).sum())
    return out


def retiring(times):
    """Lap x driver: the car's last lap before its times stop (a retirement, often a crawl back, or a lapped car
    taking the flag)."""
    r = times.notna() & times.shift(-1).isna()
    r.iloc[-1] = False
    return r


def _busy(race):
    """Cars whose lap isn't a normal racing lap: an in- or out-lap, or their last lap. Not a reference on
    neutralised laps, not a pass."""
    return race["pit_in"] | race["pit_out"] | retiring(race["times"])


def _traffic(prev, now, busy, t_prev, t, pass_s, dirty_s, gap_s):
    """Each car the ghost catches this lap (ahead at the end of the last one, behind its free-air time now, on a
    normal lap), met in road order, costs pass_s. Still ahead after paying: a pass. Not: it's stuck behind that car
    and can't reach the ones further up. Then, ending the lap within gap_s of a car ahead costs dirty_s. Neither
    cost can drop it behind a car it passed this lap (it would re-pass the same car next lap); it stays level.
    Returns the ghost's time and its number of passes."""
    caught = (prev < t_prev) & (now > t) & ~busy
    passes, floor = 0, np.inf
    for drv in prev[caught].sort_values(ascending=False).index:
        t += pass_s
        if t >= now[drv]:
            break
        passes += 1
        floor = min(floor, now[drv])
    ahead = now[(now <= t) & ~busy]
    if len(ahead) and t - ahead.max() < gap_s:
        t += dirty_s
    return min(t, floor), passes


def _copy(T, cars, n, t_prev, m):
    """The ghost's time at the end of lap m if it does what `cars` (a driver mask) did from the end of lap n-1.
    Between two of them on the road: interpolate their times by its time at the end of lap n-1, so the gaps around
    it shrink as theirs do under the SC, hold under the VSC, and it never jumps one of them. Ahead of all of them
    or behind all of them: its gap to the end car shrinks like their median gap (the SC queue forming)."""
    x = T.loc[n - 1][cars].sort_values()
    if x.empty:
        raise ValueError(f"no real car to copy on neutralised lap {n}")
    y = T.loc[m, x.index]
    if x.iloc[0] <= t_prev <= x.iloc[-1]:
        return float(np.interp(t_prev, x, y))
    end = 0 if t_prev < x.iloc[0] else -1
    shrink = min(1.0, np.median(np.diff(np.sort(y))) / np.median(np.diff(x))) if len(x) > 1 else 1.0
    return float(y.iloc[end] + shrink * (t_prev - x.iloc[end]))
