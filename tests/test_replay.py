"""The replay ghost is the model's run: on the line at its lap-end times, in the model's positions (uses cache/)."""
import numpy as np

from src.data_loader import load_config
from src.monte_carlo import run
from src.pipeline import prepare
from src.replay import build, positions


def test_ghost_crosses_the_line_at_its_model_lap_times_in_the_model_positions():
    cfg = load_config()
    d = build(cfg)
    grid, t0, sf = d["_grid"], d["_t0"], np.array(d["sf"])
    first, stops, n_laps = cfg["race"]["first_racing_lap"], d["_stops"], d["laps"]
    # green laps only (under SC/VSC the timing lags the GPS by ~8 s for every car), no pit lane, not the last lap
    laps = [n for n in range(first + 1, n_laps) if d["neutral"][n - 1] == "green" and n not in stops and n - 1 not in stops]
    off = lambda x, y, t: np.hypot(np.interp(t, grid, x) - sf[0], np.interp(t, grid, y) - sf[1])
    ghost = [off(*d["_xy"]["ghost"], d["_ends"][n]) for n in laps]
    real = [off(*d["_xy"][c["code"]], c["ends"][n - 1] + t0) for c in d["cars"] if c["kind"] != "ghost"
            for n in laps if c["ends"][n - 1] is not None]
    assert max(ghost) <= np.percentile(real, 95)      # as close to the line as the real cars are at theirs

    model = run(cfg, prepare(cfg), n=1)["traces"]["inters"].position
    ends = next(c for c in d["cars"] if c["kind"] == "ghost")["ends"]
    for lap in [*cfg["monte_carlo"]["report_laps"].values(), n_laps]:
        assert positions(d, ends[lap - 1] + 0.01)["ghost"] == model[lap]
