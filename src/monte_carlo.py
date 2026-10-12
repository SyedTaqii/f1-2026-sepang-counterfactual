"""M4: the counterfactual. The target starts on inters from his grid slot, pits for slicks with the leaders under the
lap-9 SC, then runs his own real dry plan: only the start tyre changes. A Monte Carlo over the model's uncertainty
gives his finishing-position distribution. Each draw also replays his actual strategy with the same sampled inputs,
so the two can be compared like for like. `python -m src.monte_carlo` writes outputs/results.json."""
import json

import numpy as np
import pandas as pd

from src.data_loader import ROOT
from src.pace_model import dry_lap_time, inter_pace, start_gaps
from src.pipeline import prepare, real_stints, traffic
from src.simulator import simulate


def scenario(p, cfg):
    """What stays fixed across draws: both plans and their starts, the inter reference curve, the noise per lap."""
    target, first, stop = cfg["drivers"]["target"], cfg["race"]["first_racing_lap"], cfg["counterfactual"]["crossover_lap"]
    real_from = cfg["backtest"]["start_lap"]
    me = p["df"][p["df"].Driver == target].set_index("LapNumber")
    ip = inter_pace(p["df"], cfg)
    wet = p["phases"].condition == "wet"
    return {"inters": {"start_lap": first, "start_s": float(p["race"]["times"].loc[first].min()),   # + a start gap
                       "stints": [("INTERMEDIATE", 0.0, stop), *real_stints(me, stop)]},
            "actual": {"start_lap": real_from, "start_s": me.Time[real_from].total_seconds(),        # as the back-test
                       "stints": real_stints(me, real_from)},
            "start_gaps": start_gaps(p["df"], p["session"].results, cfg).gap_s.to_numpy(),
            "dirty_gap_s": cfg["pace"]["traffic_gap_s"],
            "inter": ip, "ref": ip["ref"].to_dict(), "effects": p["effects"].to_dict(),
            # lap-to-lap noise: the teammate's inter laps around the reference when wet, the target's dry residuals when dry
            "noise_sd": wet.map({True: ip["lap_sd"], False: p["fit"]["resid_sd"]}).to_dict()}


def ghost(p, s, plan, start_gap_s, offset, coef, pass_s, dirty_s, pit_loss_s, noise):
    """One race of `plan` (s["inters"] or s["actual"]). offset: inter pace vs the reference (s/lap); coef: dry-fit
    coefficients (dict); noise: {lap: s} added to every green lap."""
    dry = dry_lap_time(coef, s["effects"], p["n_laps"])
    ref = s["ref"]
    lap_time = lambda lap, comp, age: (ref[lap] + offset if comp == "INTERMEDIATE" else dry(lap, comp, age)) + noise.get(lap, 0.0)
    return simulate(p["race"], plan["start_lap"], plan["start_s"] + start_gap_s, plan["stints"], lap_time, pit_loss_s,
                    pass_cost_s=pass_s, dirty_air_s=dirty_s, dirty_gap_s=s["dirty_gap_s"])


def run(cfg, p=None, n=None):
    """The central run (every input at its central value) and n paired Monte Carlo draws -> the results dict."""
    p = p or prepare(cfg)
    s = scenario(p, cfg)
    mc = cfg["monte_carlo"]
    n = n or mc["n"]
    rng = np.random.default_rng(mc["seed"])
    fit, tr, ip = p["fit"], p["traffic"], s["inter"]
    names = list(fit["coef"].index)
    central_tr = traffic(p, cfg)

    offsets = rng.normal(ip["mu"], ip["sigma"], n)                                    # inter pace (D-014)
    coefs = rng.multivariate_normal(fit["coef"], fit["cov"], n)                         # dry pace + deg + fuel
    costs = np.clip(rng.multivariate_normal(tr["coef"], tr["cov"], n), 0.0, None)      # pass, dirty air (D-022)
    for j, key in enumerate(("pass_cost_s", "dirty_air_s")):                           # a config number fixes it
        if cfg["sim"][key] is not None:
            costs[:, j] = cfg["sim"][key]
    losses = rng.choice(p["green_loss"].to_numpy(), n)                                  # green pit loss, bootstrap
    gaps = rng.choice(s["start_gaps"], n)                                               # standing start, bootstrap
    laps = list(s["noise_sd"])
    noise = rng.normal(0.0, 1.0, (n, len(laps))) * np.array([s["noise_sd"][lap] for lap in laps])

    winner_s = float(p["race"]["times"].iloc[-1].min())
    report_laps = mc["report_laps"]
    report = lambda sim: {**{k: int(sim.position[lap]) for k, lap in report_laps.items()},
                          "finish": int(sim.position.iloc[-1]), "gap_to_winner_s": float(sim.cum_s.iloc[-1] - winner_s)}
    central_args = (ip["mu"], fit["coef"].to_dict(), central_tr["pass_cost_s"], central_tr["dirty_air_s"],
                    float(p["green_loss"].median()))
    central = report(ghost(p, s, s["inters"], float(np.median(s["start_gaps"])), *central_args, {}))

    # ponytail: plain loop, ~10 ms a draw for both plans (about 2 min for 10k); vectorise across draws if N grows a lot
    rows = []
    for k in range(n):
        args = (offsets[k], dict(zip(names, coefs[k])), *costs[k], losses[k], dict(zip(laps, noise[k])))
        real = ghost(p, s, s["actual"], 0.0, *args)
        rows.append({**report(ghost(p, s, s["inters"], gaps[k], *args)),
                     "actual_finish": int(real.position.iloc[-1]), "actual_gap_to_winner_s": float(real.cum_s.iloc[-1] - winner_s)})
    d = pd.DataFrame(rows).assign(inter_offset=offsets, start_gap=gaps, pass_cost=costs[:, 0], dirty_air=costs[:, 1],
                                  pit_loss=losses)

    actual = int(p["session"].results.set_index("Abbreviation").Position[cfg["drivers"]["target"]])
    gained = d.actual_finish - d.finish                                                 # > 0: inters finish higher
    pct = lambda x: {q: round(float(np.percentile(x, v)), 2) for q, v in (("p10", 10), ("median", 50), ("p90", 90))}
    return {
        "question": f"Where would {cfg['drivers']['target']} have finished had he started on intermediates?",
        "n": n, "seed": mc["seed"],
        "actual_position": actual,
        "finish": {"distribution": _shares(d.finish), "expected": round(float(d.finish.mean()), 3),
                   "median": float(d.finish.median()), "p_win": round(float((d.finish == 1).mean()), 4),
                   "p_podium": round(float((d.finish <= 3).mean()), 4),
                   "expected_gain_vs_actual": round(actual - float(d.finish.mean()), 3),
                   "gap_to_winner_s": pct(d.gap_to_winner_s)},
        "actual_strategy": {"distribution": _shares(d.actual_finish), "expected": round(float(d.actual_finish.mean()), 3),
                            "gap_to_winner_s": pct(d.actual_gap_to_winner_s)},
        "paired": {"p_better": round(float((gained > 0).mean()), 4), "p_same": round(float((gained == 0).mean()), 4),
                   "p_worse": round(float((gained < 0).mean()), 4), "expected_places_gained": round(float(gained.mean()), 3),
                   "time_gained_s": pct(d.actual_gap_to_winner_s - d.gap_to_winner_s)},
        "positions_by_lap": {k: {"lap": lap, "distribution": _shares(d[k])} for k, lap in report_laps.items()},
        "central_run": central,
        "inputs": {"inter_offset_s_per_lap": {"mu": round(float(ip["mu"]), 3), "sigma": round(float(ip["sigma"]), 3)},
                   "start_gaps_s": [round(float(g), 2) for g in s["start_gaps"]],
                   "pass_cost_s": central_tr["pass_cost_s"], "dirty_air_s": central_tr["dirty_air_s"],
                   "green_pit_loss_median_s": round(float(p["green_loss"].median()), 2),
                   "lap_noise_sd_s": {"wet": round(float(ip["lap_sd"]), 3), "dry": round(float(fit["resid_sd"]), 3)},
                   "inters_plan": [[c, float(a), int(last)] for c, a, last in s["inters"]["stints"]],
                   "actual_plan": [[c, float(a), int(last)] for c, a, last in s["actual"]["stints"]]},
        "config": cfg,
        "draws": d,                                                                      # per draw; not saved
    }


def save(res, path):
    """Write the results (without the per-draw table) to `path`, relative to the repo root."""
    path = ROOT / path
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in res.items() if k != "draws"}, indent=2, default=float), encoding="utf-8")


def _shares(x):
    v, c = np.unique(x, return_counts=True)
    return {str(int(k)): round(float(m) / len(x), 4) for k, m in zip(v, c)}


if __name__ == "__main__":
    import time

    import fastf1
    from src.data_loader import load_config

    fastf1.set_log_level("ERROR")
    cfg = load_config()
    t0 = time.perf_counter()
    res = run(cfg)
    save(res, cfg["monte_carlo"]["results_path"])
    f, a, pr = res["finish"], res["actual_strategy"], res["paired"]
    print(f"{res['n']} paired draws in {time.perf_counter() - t0:.0f} s -> {cfg['monte_carlo']['results_path']}")
    print(f"inters: expected P{f['expected']:.2f}, P(win) {f['p_win']:.1%}, P(podium) {f['p_podium']:.1%}, "
          f"gap to winner {f['gap_to_winner_s']}")
    print(f"actual strategy (same draws): expected P{a['expected']:.2f}, gap to winner {a['gap_to_winner_s']}")
    print(f"paired: inters better {pr['p_better']:.1%}, same {pr['p_same']:.1%}, worse {pr['p_worse']:.1%}; "
          f"places gained {pr['expected_places_gained']:+.2f}; time gained {pr['time_gained_s']}")
    print("inters finish:", f["distribution"])
    print("actual finish:", a["distribution"])
    for k, v in res["positions_by_lap"].items():
        print(f"{k} (lap {v['lap']}):", v["distribution"])
    print("central run:", res["central_run"])
