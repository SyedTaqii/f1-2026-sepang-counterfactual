"""The race replay (D-028): every real car on its FastF1 GPS track, plus the central inters run as a ghost.
The ghost has lap times only, so each of its laps borrows the path of a real car that ran the same kind of lap
(normal, in-lap, out-lap) and ended it nearest to the ghost on the road, stretched to the ghost's lap time.
Laps up to the standing start borrow the target's own path, so the ghost starts from his real grid slot.
`python -m src.replay [bare_page_path]` writes the page and the GIF named in config `replay`."""
import base64
import json

import numpy as np

from src.data_loader import ROOT, load_session
from src.monte_carlo import run, scenario
from src.pipeline import prepare


def _seconds(s):
    return s.dt.total_seconds()


def _pits(laps):
    """[(pit-in, pit-out)] session seconds. FastF1 puts PitInTime on lap n and PitOutTime on lap n + 1."""
    pin, pout = _seconds(laps.PitInTime), _seconds(laps.PitOutTime)
    return [(float(pin[n]), float(pout[n + 1])) for n in pin.dropna().index if n + 1 in pout.index and pout[n + 1] == pout[n + 1]]


def _gps(session):
    """{driver code: (session seconds, X, Y)} from FastF1's position data (units of 0.1 m)."""
    code = session.results.set_index("DriverNumber").Abbreviation.to_dict()
    return {code[k]: (_seconds(d.SessionTime).to_numpy(), d.X.to_numpy(float), d.Y.to_numpy(float))
            for k, d in session.pos_data.items() if k in code}


def ghost_path(df, gps, target, ends, stops, first_lap, n_laps, grid):
    """The ghost's X, Y on `grid` (session seconds). ends: {lap: ghost lap-end time} from lap 0 (race start);
    stops: laps the ghost pits after. Returns x, y and {lap: (donor, donor window, ghost window)}."""
    me = {d: g.set_index("LapNumber") for d, g in df.groupby("Driver")}
    kind = lambda laps, n: (n in laps.index and laps.PitInTime.notna().get(n, False),
                            n in laps.index and laps.PitOutTime.notna().get(n, False))
    x, y = np.full(grid.size, np.nan), np.full(grid.size, np.nan)
    donors = {}
    for n in range(1, n_laps + 1):
        if n <= first_lap:
            donor = target                                   # the real car's grid slot, SC laps and start
        else:
            want = (n in stops, n - 1 in stops)               # normal lap, in-lap or out-lap
            cands = [d for d, laps in me.items() if {n - 1, n} <= set(laps.index)
                     and (n + 1 in laps.index or n == n_laps)  # not a car's crawl back to retire
                     and tuple(map(bool, kind(laps, n))) == want]
            donor = min(cands, key=lambda d: abs(me[d].Time[n].total_seconds() - ends[n]))
        a0, a1 = me[donor].Time[n - 1].total_seconds() if n > 1 else ends[0], me[donor].Time[n].total_seconds()
        b0, b1 = ends[n - 1], ends[n]
        m = (grid >= b0) & (grid < b1)
        t_donor = a0 + (grid[m] - b0) / (b1 - b0) * (a1 - a0)
        tt, xx, yy = gps[donor]
        x[m], y[m] = np.interp(t_donor, tt, xx), np.interp(t_donor, tt, yy)
        donors[n] = (donor, a0, a1, b0, b1)
    return x, y, donors


def _pack(a):
    """Int16 little-endian, base64, for the page's typed arrays. NaN (no data) repeats the nearest valid value."""
    a = a.copy()
    ok = ~np.isnan(a)
    a[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), a[ok])
    return base64.b64encode(np.round(a).astype("<i2").tobytes()).decode()


def _rel(v, t0):
    return None if v is None or v != v else round(float(v) - t0, 2)


def build(cfg):
    """Everything the page and the GIF draw (times relative to the race start). Keys starting with _ stay in Python."""
    target, first = cfg["drivers"]["target"], cfg["race"]["first_racing_lap"]
    session = load_session(cfg, telemetry=True)
    p = prepare(cfg)
    df, phases, n_laps = p["df"], p["phases"], p["n_laps"]
    s = scenario(p, cfg)
    res = run(cfg, p, n=1)
    trace = res["traces"]["inters"]
    hz = cfg["replay"]["hz"]

    t0 = float(df[df.LapNumber == 1].LapStartTime.min().total_seconds())
    grid = t0 + np.arange(int((df.Time.max().total_seconds() + 5 - t0) * hz) + 1) / hz
    gps = _gps(session)
    me = df[df.Driver == target].set_index("LapNumber")

    # the ghost: the target's real laps up to the standing start, then the central inters run (as in monte_carlo.run)
    ends = {0: t0, **{n: float(me.Time[n].total_seconds()) for n in range(1, first)},
            first: s["inters"]["start_s"] + float(np.median(s["start_gaps"])), **trace.cum_s.astype(float).to_dict()}
    stops = {int(last) for _, _, last in s["inters"]["stints"][:-1]}
    gx, gy, donors = ghost_path(df, gps, target, ends, stops, first, n_laps, grid)
    ghost_pits = []
    for n in sorted(donors):                                  # the out-lap donor's stop, mapped onto the ghost
        donor, a0, a1, b0, b1 = donors[n]
        warp = lambda t: b0 + (t - a0) / (a1 - a0) * (b1 - b0)
        if n - 1 in stops:
            ghost_pits += [(warp(i), warp(o)) for i, o in _pits(df[df.Driver == donor].set_index("LapNumber")) if a0 <= i and o <= a1]

    results = session.results.set_index("Abbreviation")
    xy = {}
    cars = []
    for drv in results.Position.sort_values().index:
        laps = df[df.Driver == drv].set_index("LapNumber").reindex(range(1, n_laps + 1))
        tt, xx, yy = gps[drv]
        xy[drv] = (np.interp(grid, tt, xx), np.interp(grid, tt, yy))
        cars.append({"code": drv, "kind": "real" if drv == target else "field", "team": str(results.TeamName[drv]),
                     "x": _pack(xy[drv][0]), "y": _pack(xy[drv][1]),
                     "ends": [_rel(v, t0) for v in _seconds(laps.Time)],
                     "compound": [{"None": "UNKNOWN"}.get(c, c) for c in laps.Compound.fillna("UNKNOWN")],
                     "pits": [[_rel(i, t0), _rel(o, t0)] for i, o in _pits(df[df.Driver == drv].set_index("LapNumber"))],
                     "out": results.Status[drv] == "Retired", "classified": int(results.Position[drv])})
    comp = trace.compound.to_dict()
    xy["ghost"] = (gx, gy)
    cars.append({"code": target, "kind": "ghost", "team": "Model: started on inters",
                 "x": _pack(gx), "y": _pack(gy), "ends": [_rel(ends.get(n), t0) for n in range(1, n_laps + 1)],
                 "compound": [comp.get(n, "INTERMEDIATE") for n in range(1, n_laps + 1)],
                 "pits": [[_rel(i, t0), _rel(o, t0)] for i, o in ghost_pits],
                 "out": False, "classified": int(res["central_run"]["finish"])})

    winner = results.Position.idxmin()
    fast = session.laps.pick_drivers(winner).pick_fastest()
    pos = fast.get_pos_data()
    tt, xx, yy = gps[target]
    pin, pout = _pits(me)[0]
    pit = (tt >= pin) & (tt <= pout)
    full = json.loads((ROOT / cfg["monte_carlo"]["results_path"]).read_text(encoding="utf-8"))
    ls = fast.LapStartTime.total_seconds()
    return {"hz": hz, "n": int(grid.size), "laps": n_laps, "target": target, "first_lap": first,
            "track": [[int(a), int(b)] for a, b in zip(pos.X, pos.Y)],
            "pit": [[int(a), int(b)] for a, b in zip(xx[pit], yy[pit])],
            "sf": [round(float(np.interp(ls, gps[winner][0], gps[winner][1]))), round(float(np.interp(ls, gps[winner][0], gps[winner][2])))],
            "neutral": [phases.neutral.get(n, "green") for n in range(1, n_laps + 1)],
            "wet": [bool(phases.condition.get(n) == "wet") for n in range(1, n_laps + 1)],
            "cars": cars,
            "summary": {"p3": full["finish"]["distribution"]["3"], "expected": full["finish"]["expected"],
                        "actual_expected": full["actual_strategy"]["expected"], "n": full["n"],
                        "same": full["paired"]["p_same"], "better": full["paired"]["p_better"]},
            "_t0": t0, "_grid": grid, "_xy": xy, "_ends": ends, "_donors": donors, "_stops": stops}


def positions(data, t):
    """{code or 'ghost': running position} at t (seconds from the start). Real cars rank among the real field;
    the ghost ranks among the real field without the target, as it does in the model."""
    def progress(c):
        e = [v for v in c["ends"] if v is not None]
        done = sum(v <= t for v in e)
        if done == len(e):                         # finished (or out): the earlier the last lap, the higher
            return done + 0.5 / (1.0 + e[-1] / 1e4)
        prev = e[done - 1] if done else 0.0
        return done + (t - prev) / (e[done] - prev)
    prog = {("ghost" if c["kind"] == "ghost" else c["code"]): progress(c) for c in data["cars"]}
    real = sorted((k for k in prog if k != "ghost"), key=lambda k: -prog[k])
    out = {k: i + 1 for i, k in enumerate(real)}
    out["ghost"] = 1 + sum(prog[k] > prog["ghost"] for k in real if k != data["target"])
    return out


def write_html(data, cfg, bare_path=None):
    """The self-contained page: src/replay.html with the data inlined. bare_path also gets the page without a
    doctype or head, the form an Artifact publish wants."""
    page = (ROOT / "src" / "replay.html").read_text(encoding="utf-8")
    page = page.replace("/*__REPLAY_DATA__*/null", json.dumps({k: v for k, v in data.items() if not k.startswith("_")},
                                                              separators=(",", ":")))
    out = ROOT / cfg["replay"]["html"]
    out.parent.mkdir(exist_ok=True)
    out.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   f"</head>\n<body>\n{page}\n</body>\n</html>\n", encoding="utf-8")
    if bare_path:
        open(bare_path, "w", encoding="utf-8").write(page)
    return out


def write_gif(data, cfg):
    """The README clip: config replay.gif_segments, each from the leader's start of from_lap to his end of to_lap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from src.plots import BLUE, INK, INK_2, MUTED, ORANGE, SURFACE

    r = cfg["replay"]
    grid, xy, hz, t0 = data["_grid"], data["_xy"], data["hz"], data["_t0"]
    lead = [min(c["ends"][n] for c in data["cars"] if c["kind"] != "ghost" and c["ends"][n] is not None) for n in range(data["laps"])]
    frames = []
    for seg in r["gif_segments"]:
        a = lead[seg["from_lap"] - 2] if seg["from_lap"] > 1 else 0.0
        frames += [(seg, t) for t in np.arange(a, lead[seg["to_lap"] - 1], seg["speed"] / r["gif_fps"])]
    track, pit = np.array(data["track"]), np.array(data["pit"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=90)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.86)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(*np.r_[track, track[:1]].T, color="#d9dcd6", lw=7, solid_capstyle="round", zorder=1)
    ax.plot(*pit.T, color="#d9dcd6", lw=2, ls=(0, (2, 2)), zorder=1)
    field = ax.scatter([], [], s=22, color=MUTED, zorder=3)
    real = ax.scatter([], [], s=70, color=ORANGE, edgecolor=SURFACE, lw=1.2, zorder=4)
    ghost = ax.scatter([], [], s=70, color=BLUE, edgecolor=SURFACE, lw=1.2, zorder=5)
    title = fig.text(0.04, 0.95, "", fontsize=12, fontweight="bold", color=INK, va="top")
    sub = fig.text(0.04, 0.895, "", fontsize=9.5, color=INK_2, va="top")
    tags = [ax.text(0, 0, "", fontsize=8.5, fontweight="bold", color=c, zorder=6) for c in (ORANGE, BLUE)]
    span = track.max(0) - track.min(0)
    offs = [span * np.array([0.02, -0.06]), span * np.array([0.02, 0.025])]     # real label below, ghost label above
    finished = {c["code"] if c["kind"] != "ghost" else "ghost": max(v for v in c["ends"] if v is not None) for c in data["cars"]}

    def draw(frame):
        seg, t = frame
        i = min(int(round(t * hz)), grid.size - 1)
        live = lambda k: t <= finished[k]
        pts = np.array([(xy[k][0][i], xy[k][1][i]) for k in xy if k not in ("ghost", data["target"]) and live(k)])
        field.set_offsets(pts if len(pts) else np.empty((0, 2)))
        pos = positions(data, t)
        lap = min(sum(v <= t for v in lead) + 1, data["laps"])
        for sc, tag, off, k, name in ((real, tags[0], offs[0], data["target"], "real"), (ghost, tags[1], offs[1], "ghost", "ghost")):
            p = np.array([[xy[k][0][i], xy[k][1][i]]])
            sc.set_offsets(p if live(k) else np.empty((0, 2)))
            tag.set_position(p[0] + off)
            tag.set_text(f"HAM {name} P{pos[k]}" if live(k) else "")
        flag = {"sc": "  ·  SAFETY CAR", "vsc": "  ·  VIRTUAL SAFETY CAR"}.get(data["neutral"][lap - 1], "")
        title.set_text(seg["label"])
        sub.set_text(f"Lap {lap}/{data['laps']}{flag}   ·   blue: ghost (the model, started on inters)   orange: real")
        return [field, real, ghost, title, sub, *tags]

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out = ROOT / r["gif"]
    anim.save(out, writer=PillowWriter(fps=r["gif_fps"]))
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys

    import fastf1
    from src.data_loader import load_config

    fastf1.set_log_level("ERROR")
    cfg = load_config()
    data = build(cfg)
    print("page:", write_html(data, cfg, sys.argv[1] if len(sys.argv) > 1 else None))
    print("gif:", write_gif(data, cfg))
