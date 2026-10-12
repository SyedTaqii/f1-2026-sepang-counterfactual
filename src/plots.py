"""M5: the 4 PRD charts -> outputs/*.png. Numbers come from outputs/results.json; the two race traces are recomputed
from the data (the central run is deterministic). `python -m src.plots`"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import ROOT

# reference palette (dataviz skill), light mode: slot 1 blue, slot 2 orange; validated as a pair
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SC_BAND, VSC_BAND = "#e1e0d9", "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _neutral_bands(ax, phases):
    """Shade SC and VSC laps (each lap n spans n-1..n on a 'lap' axis); they get legend entries."""
    for lap, kind in phases.neutral.items():
        if kind in ("sc", "vsc"):
            ax.axvspan(lap - 1, lap, color=SC_BAND if kind == "sc" else VSC_BAND, lw=0, zorder=0,
                       label={"sc": "Safety car", "vsc": "Virtual safety car"}[kind])


def _legend(ax, **kw):
    """One legend entry per label (the bands repeat theirs)."""
    h, l = ax.get_legend_handles_labels()
    keep = dict(zip(l, h))
    ax.legend(keep.values(), keep.keys(), **kw)


def race_trace(p, cfg, path):
    """Chart 1: the real race. Running position per lap of the target and his teammate (from the fixed timing)."""
    df = p["df"]
    pos = df.pivot(index="LapNumber", columns="Driver", values="Time").rank(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _neutral_bands(ax, p["phases"])
    for drv, color in ((cfg["drivers"]["target"], BLUE), (cfg["drivers"]["teammate"], ORANGE)):
        s = pos[drv].dropna()
        ax.plot(s.index, s.values, color=color, lw=2, label=drv, zorder=3)
        ax.text(s.index[-1] + 0.6, s.values[-1], f"{drv} P{int(s.values[-1])}", color=INK, va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set(xlabel="Lap", ylabel="Running position", xlim=(0, p["n_laps"] + 4))
    ax.set_title("The real race: both Ferraris started on slicks on a wet track", pad=30)
    _legend(ax, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ghost_vs_actual(p, cfg, trace, path):
    """Chart 2: gap to the leader per lap, the target's real race vs the central inters ghost."""
    T = p["race"]["times"]                                   # the field without the target
    me = p["df"][p["df"].Driver == cfg["drivers"]["target"]].set_index("LapNumber").Time.dt.total_seconds()
    real = (me - np.minimum(T.min(axis=1), me)).dropna()     # leader = the real field including him
    real = real[real.index >= cfg["race"]["first_racing_lap"]]  # laps 1-2: behind the SC, aborted start (no racing times)
    ghost = (trace.cum_s - np.minimum(T.min(axis=1).reindex(trace.index), trace.cum_s))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _neutral_bands(ax, p["phases"])
    ax.plot(real.index, real.values, color=ORANGE, lw=2, label=f"{cfg['drivers']['target']}, real race (slicks)", zorder=3)
    ax.plot(ghost.index, ghost.values, color=BLUE, lw=2, label="Ghost that started on inters (central run)", zorder=3)
    ax.set(xlabel="Lap", ylabel="Gap to the leader (s)", xlim=(0, p["n_laps"] + 1), ylim=(0, None))
    ax.set_title("On inters he'd have run at the front, but two safety cars squeezed the gaps away", pad=30)
    _legend(ax, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def finish_histogram(res, path):
    """Chart 3: finishing-position probabilities, inters vs the actual strategy, same draws."""
    inters, actual = res["finish"]["distribution"], res["actual_strategy"]["distribution"]
    places = range(1, max(int(k) for k in {**inters, **actual}) + 1)
    x = np.arange(len(places))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for off, dist, color, label in ((-w / 2 - 0.01, inters, BLUE, "Started on inters"),
                                    (w / 2 + 0.01, actual, ORANGE, "Actual strategy (slicks), same draws")):
        vals = [100 * dist.get(str(k), 0.0) for k in places]
        ax.bar(x + off, vals, width=w, color=color, label=label, zorder=3)
        k3 = list(places).index(3)
        ax.text(x[k3] + off, vals[k3] + 1.5, f"{vals[k3]:.0f}%", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [f"P{k}" for k in places])
    ax.set(ylabel="Share of 10,000 draws (%)", ylim=(0, 100))
    ax.set_title(f"Where {res['driver']} finishes: P3 either way (real result: P{res['actual_position']})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tornado_chart(res, path):
    """Chart 4: one assumption at a time at its low and high value -> expected finishing position."""
    t = res["sensitivity"]
    bars = t["bars"][::-1]                                   # biggest swing on top
    base = t["base_expected"]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(bars) + 1.6))
    y = np.arange(len(bars))
    for i, b in enumerate(bars):
        for side, color in (("low", BLUE), ("high", ORANGE)):
            v = b[f"expected_{side}"]
            ax.barh(i + (0.2 if side == "low" else -0.2), v - base, left=base, height=0.36, color=color, zorder=3)
            ax.text(v + (0.01 if v >= base else -0.01), i + (0.2 if side == "low" else -0.2), f"{b[side]:g}",
                    ha="left" if v >= base else "right", va="center", color=INK_2, fontsize=8)
    ax.axvline(base, color=INK_2, lw=1, zorder=4)
    ax.set_yticks(y, [b["label"] for b in bars])
    ax.tick_params(axis="y", colors=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Expected finishing position on inters (base P{base:.2f}; left = better)")
    ax.set_title("What moves the answer: one assumption at a time")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Low setting", "High setting"], loc="lower right")
    lo = min(min(b["expected_low"], b["expected_high"]) for b in bars)
    hi = max(max(b["expected_low"], b["expected_high"]) for b in bars)
    pad = 0.15 * max(hi - lo, 0.2)
    ax.set_xlim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(res, p, cfg, trace):
    """Write the 4 charts next to results.json. trace: the central inters run (monte_carlo.run(...)['traces'])."""
    out = (ROOT / cfg["monte_carlo"]["results_path"]).parent
    out.mkdir(exist_ok=True)
    race_trace(p, cfg, out / "1_race_trace.png")
    ghost_vs_actual(p, cfg, trace, out / "2_ghost_vs_actual.png")
    finish_histogram(res, out / "3_finish_distribution.png")
    tornado_chart(res, out / "4_tornado.png")
    return sorted(str(f.relative_to(ROOT)) for f in out.glob("*.png"))


if __name__ == "__main__":
    import fastf1
    from src.data_loader import load_config
    from src.monte_carlo import run
    from src.pipeline import prepare

    fastf1.set_log_level("ERROR")
    cfg = load_config()
    res = json.loads((ROOT / cfg["monte_carlo"]["results_path"]).read_text(encoding="utf-8"))
    p = prepare(cfg)
    print(make_all(res, p, cfg, run(cfg, p, n=1)["traces"]["inters"]))
