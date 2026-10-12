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
    laps = race["times"].index                       # consecutive lap numbers; numpy inside for speed (M4: 10k runs)
    T, pit_in, busy = race["times"].to_numpy(float), race["pit_in"].to_numpy(), _busy(race).to_numpy()
    neutral, wear = race["neutral"].reindex(laps).to_numpy(), race["wear"].reindex(laps).to_numpy(float)
    k, (comp, age, last) = 0, stints[0]
    t, out_lap, rows = start_s, None, []
    for i in range(laps.get_loc(start_lap) + 1, len(laps)):
        n = int(laps[i])
        age += wear[i]
        prev, now, t_prev, passes = T[i - 1], T[i], t, 0
        stop = n == last and k < len(stints) - 1
        if out_lap is not None:                       # second half of a stop under SC/VSC, already timed
            t, out_lap = out_lap, None
        elif stop and neutral[i] != "green":
            stoppers = pit_in[i] & ~np.isnan(T[i - 1:i + 2]).any(axis=0)
            if not stoppers.any():
                raise ValueError(f"no real car stopped on neutralised lap {n}: nothing to copy")
            t, out_lap = [_copy(T, stoppers, i, t, m) for m in (i, i + 1)]
        elif neutral[i] == "green":
            t += lap_time(n, comp, age) + (pit_loss_s if stop else 0.0)
            t, passes = _traffic(prev, now, busy[i], t_prev, t, pass_cost_s, dirty_air_s, dirty_gap_s)
        else:
            t = _copy(T, ~np.isnan(prev) & ~np.isnan(now) & ~busy[i], i, t, i)
        rows.append((n, t, t - t_prev, 1 + int(np.sum(now < t)), comp, age, passes))
        if stop:
            k += 1
            comp, age, last = stints[k]
    return pd.DataFrame(rows, columns=["LapNumber", "cum_s", "lap_s", "position", "compound", "age", "passes"]
                        ).set_index("LapNumber")


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
    Arrays are one lap of the field. Returns the ghost's time and its number of passes."""
    caught = np.flatnonzero((prev < t_prev) & (now > t) & ~busy)
    passes, floor = 0, np.inf
    for j in caught[np.argsort(-prev[caught], kind="stable")]:
        t += pass_s
        if t >= now[j]:
            break
        passes += 1
        floor = min(floor, now[j])
    ahead = now[(now <= t) & ~busy]
    if ahead.size and t - ahead.max() < gap_s:
        t += dirty_s
    return min(t, floor), passes


def _copy(T, cars, i, t_prev, m):
    """The ghost's time at the end of row m if it does what `cars` (a driver mask) did from the end of row i-1.
    Between two of them on the road: interpolate their times by its time at the end of row i-1, so the gaps around
    it shrink as theirs do under the SC, hold under the VSC, and it never jumps one of them. Ahead of all of them
    or behind all of them: its gap to the end car shrinks like their median gap (the SC queue forming)."""
    order = np.argsort(T[i - 1][cars], kind="stable")
    x, y = T[i - 1][cars][order], T[m][cars][order]
    if x[0] <= t_prev <= x[-1]:
        return float(np.interp(t_prev, x, y))
    end = 0 if t_prev < x[0] else -1
    shrink = min(1.0, np.median(np.diff(np.sort(y))) / np.median(np.diff(x))) if x.size > 1 else 1.0
    return float(y[end] + shrink * (t_prev - x[end]))
