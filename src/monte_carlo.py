"""M4-M5: the counterfactual. The target starts on inters from his grid slot, pits for slicks with the leaders under
the lap-9 SC, then runs his own real dry plan: only the start tyre changes. A Monte Carlo over the model's uncertainty
gives his finishing-position distribution. Each draw also replays his actual strategy with the same sampled inputs,
so the two can be compared like for like. M5 adds the tornado and the teammate repeat.
`python -m src.monte_carlo` writes everything to outputs/results.json."""
import copy
import json

import numpy as np
import pandas as pd

from src.data_loader import ROOT
from src.pace_model import dry_lap_time, inter_pace, start_gaps
from src.pipeline import backtest, prepare, real_stints, traffic
from src.simulator import simulate

PREPARE_READS = ("race", "drivers", "data_fixes", "pace", "pit_loss")   # config sections prepare() depends on


def scenario(p, cfg):
    """What stays fixed across draws: both plans and their starts, the inter reference curve, the noise per lap."""
    target, first, stop = cfg["drivers"]["target"], cfg["race"]["first_racing_lap"], cfg["counterfactual"]["crossover_lap"]
    real_from = cfg["backtest"]["start_lap"]
    me = p["df"][p["df"].Driver == target].set_index("LapNumber")
    ip = inter_pace(p["df"], cfg)
    wet = p["phases"].condition == "wet"
    return {"inters": {"start_lap": first, "start_s": float(p["race"]["times"].loc[first].min()),   # + a start gap
                       "stints": [("INTERMEDIATE", 0.0, stop), *[st for st in real_stints(me, stop) if st[2] > stop]]},
            "actual": {"start_lap": real_from, "start_s": me.Time[real_from].total_seconds(),        # as the back-test
                       "stints": real_stints(me, real_from)},
            "start_gaps": start_gaps(p["df"], p["session"].results, cfg).gap_s.to_numpy(),
            "dirty_gap_s": cfg["pace"]["traffic_gap_s"], "dry_offset_s": cfg["monte_carlo"]["dry_pace_offset_s"],
            "inter": ip, "ref": ip["ref"].to_dict(), "effects": p["effects"].to_dict(),
            # lap-to-lap noise: inter laps around the reference when wet, the target's dry residuals when dry
            "noise_sd": wet.map({True: ip["lap_sd"], False: p["fit"]["resid_sd"]}).to_dict()}


def ghost(p, s, plan, start_gap_s, offset, coef, pass_s, dirty_s, pit_loss_s, noise):
    """One race of `plan` (s["inters"] or s["actual"]). offset: inter pace vs the reference (s/lap); coef: dry-fit
    coefficients (dict); noise: {lap: s} added to every green lap."""
    dry = dry_lap_time(coef, s["effects"], p["n_laps"])
    ref, dry_offset = s["ref"], s["dry_offset_s"]
    lap_time = lambda lap, comp, age: noise.get(lap, 0.0) + (
        ref[lap] + offset if comp == "INTERMEDIATE" else dry(lap, comp, age) + dry_offset)
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
    losses = rng.choice(p["green_loss"].to_numpy(), n)                                  # green pit loss, bootstrap
    gaps = rng.choice(s["start_gaps"], n)                                               # standing start, bootstrap
    laps = list(s["noise_sd"])
    noise = rng.normal(0.0, 1.0, (n, len(laps))) * np.array([s["noise_sd"][lap] for lap in laps])
    # a number in the config pins that input in every draw (drawn anyway, so the other inputs' draws don't move)
    for j, key in enumerate(("pass_cost_s", "dirty_air_s")):
        if cfg["sim"][key] is not None:
            costs[:, j] = cfg["sim"][key]
    fixed = mc["fixed"]
    for key, arr in (("inter_offset_s", offsets), ("start_gap_s", gaps), ("green_pit_loss_s", losses)):
        if fixed[key] is not None:
            arr[:] = fixed[key]

    winner_s = float(p["race"]["times"].iloc[-1].min())
    report_laps = mc["report_laps"]
    report = lambda sim: {**{k: int(sim.position[lap]) for k, lap in report_laps.items()},
                          "finish": int(sim.position.iloc[-1]), "gap_to_winner_s": float(sim.cum_s.iloc[-1] - winner_s)}
    central_args = (ip["mu"] if fixed["inter_offset_s"] is None else fixed["inter_offset_s"], fit["coef"].to_dict(),
                    central_tr["pass_cost_s"], central_tr["dirty_air_s"],
                    float(p["green_loss"].median()) if fixed["green_pit_loss_s"] is None else fixed["green_pit_loss_s"])
    central_gap = float(np.median(s["start_gaps"])) if fixed["start_gap_s"] is None else fixed["start_gap_s"]
    traces = {"inters": ghost(p, s, s["inters"], central_gap, *central_args, {}),
              "actual": ghost(p, s, s["actual"], 0.0, *central_args, {})}

    # ponytail: plain loop, ~7 ms a draw for both plans (about 70 s for 10k); vectorise across draws if N grows a lot
    rows = []
    for k in range(n):
        args = (offsets[k], dict(zip(names, coefs[k])), *costs[k], losses[k], dict(zip(laps, noise[k])))
        real = ghost(p, s, s["actual"], 0.0, *args)
        rows.append({**report(ghost(p, s, s["inters"], gaps[k], *args)),
                     "actual_finish": int(real.position.iloc[-1]), "actual_gap_to_winner_s": float(real.cum_s.iloc[-1] - winner_s)})
    d = pd.DataFrame(rows).assign(inter_offset=offsets, start_gap=gaps, pass_cost=costs[:, 0], dirty_air=costs[:, 1],
                                  pit_loss=losses)

    results = p["session"].results.set_index("Abbreviation")
    actual = int(results.Position[cfg["drivers"]["target"]])
    ahead = [drv for drv in results.Position.sort_values().index[:actual + 1] if f"d_{drv}" in fit["coef"]]
    gained = d.actual_finish - d.finish                                                 # > 0: inters finish higher
    pct = lambda x: {q: round(float(np.percentile(x, v)), 2) for q, v in (("p10", 10), ("median", 50), ("p90", 90))}
    return {
        "question": f"Where would {cfg['drivers']['target']} have finished had he started on intermediates?",
        "driver": cfg["drivers"]["target"], "n": n, "seed": mc["seed"],
        "actual_position": actual,
        "finish": {"distribution": _shares(d.finish), "expected": round(float(d.finish.mean()), 3),
                   "median": float(d.finish.median()), "p_win": round(float((d.finish == 1).mean()), 4),
                   "p_podium": round(float((d.finish <= 3).mean()), 4),
                   "expected_gain_vs_actual": round(actual - float(d.finish.mean()), 3),
                   "gap_to_winner_s": pct(d.gap_to_winner_s)},
        "actual_strategy": {"distribution": _shares(d.actual_finish), "expected": round(float(d.actual_finish.mean()), 3),
                            "p_podium": round(float((d.actual_finish <= 3).mean()), 4),
                            "gap_to_winner_s": pct(d.actual_gap_to_winner_s)},
        "paired": {"p_better": round(float((gained > 0).mean()), 4), "p_same": round(float((gained == 0).mean()), 4),
                   "p_worse": round(float((gained < 0).mean()), 4), "expected_places_gained": round(float(gained.mean()), 3),
                   "time_gained_s": pct(d.actual_gap_to_winner_s - d.gap_to_winner_s)},
        "positions_by_lap": {k: {"lap": lap, "distribution": _shares(d[k])} for k, lap in report_laps.items()},
        "central_run": report(traces["inters"]),
        "inputs": {"inter_offset_s_per_lap": {"mu": round(float(ip["mu"]), 3), "sigma": round(float(ip["sigma"]), 3),
                                              "source": "own inter laps" if "own_laps" in ip else "LEC proxy + dry ratio"},
                   "start_gaps_s": [round(float(g), 2) for g in s["start_gaps"]],
                   "pass_cost_s": round(float(central_tr["pass_cost_s"]), 3), "dirty_air_s": round(float(central_tr["dirty_air_s"]), 3),
                   "green_pit_loss_median_s": round(float(p["green_loss"].median()), 2),
                   "lap_noise_sd_s": {"wet": round(float(ip["lap_sd"]), 3), "dry": round(float(fit["resid_sd"]), 3)},
                   "inters_plan": [[c, float(a), int(last)] for c, a, last in s["inters"]["stints"]],
                   "actual_plan": [[c, float(a), int(last)] for c, a, last in s["actual"]["stints"]]},
        "context": {"dry_pace_vs_target_s_per_lap": {drv: round(float(fit["coef"][f"d_{drv}"]), 2) for drv in ahead},
                    "crossover_lap_stoppers": int(p["race"]["pit_in"].loc[cfg["counterfactual"]["crossover_lap"]].sum()),
                    "other_cars_running_at_crossover": int(p["race"]["times"].loc[cfg["counterfactual"]["crossover_lap"]].notna().sum())},
        "draws": d, "traces": traces,                                                    # not saved
    }


def backtest_summary(p, cfg):
    """The back-test's checks (M3) in numbers: position at the flag, time error at backtest.check_lap and the flag."""
    b, check = backtest(p, cfg), cfg["backtest"]["check_lap"]
    return {"driver": cfg["drivers"]["target"], "position": int(b.position.iloc[-1]),
            "real_position": int(p["session"].results.set_index("Abbreviation").Position[cfg["drivers"]["target"]]),
            "check_lap": check, "error_at_check_lap_s": round(float(b.error_s[check]), 2),
            "error_at_flag_s": round(float(b.error_s.iloc[-1]), 2), "tolerance_s": cfg["backtest"]["tolerance_s"]}


def tornado(cfg, p):
    """Each sensitivity.tornado assumption at its low and high value, one at a time, sensitivity.n draws, same seed.
    Bars: the expected finishing position on inters (and the paired gain) at each end."""
    n = cfg["sensitivity"]["n"]
    base = run(cfg, p, n)
    bars = []
    for t in cfg["sensitivity"]["tornado"]:
        bar = {"label": t["label"], "key": t["key"], "low": t["low"], "high": t["high"]}
        for side in ("low", "high"):
            c = copy.deepcopy(cfg)
            *path, leaf = t["key"].split(".")
            sub = c
            for part in path:
                sub = sub[part]
            sub[leaf] = t[side]
            r = run(c, prepare(c) if path[0] in PREPARE_READS else p, n)
            bar[f"expected_{side}"] = r["finish"]["expected"]
            bar[f"gain_{side}"] = r["paired"]["expected_places_gained"]
        bar["swing"] = round(abs(bar["expected_high"] - bar["expected_low"]), 3)
        bars.append(bar)
    return {"n": n, "base_expected": base["finish"]["expected"], "base_gain": base["paired"]["expected_places_gained"],
            "bars": sorted(bars, key=lambda b: -b["swing"])}


def for_driver(cfg, driver):
    """The same config with `driver` as the target and the old target as his teammate."""
    c = copy.deepcopy(cfg)
    c["drivers"] = {"target": driver, "teammate": cfg["drivers"]["target"]}
    return c


def report(cfg):
    """Everything outputs/results.json holds: the counterfactual, its back-test, the tornado, the teammate repeat."""
    p = prepare(cfg)
    res = run(cfg, p)
    res["backtest"] = backtest_summary(p, cfg)
    res["sensitivity"] = tornado(cfg, p)
    mate = for_driver(cfg, cfg["repeat"]["driver"])
    pm = prepare(mate)
    res["repeat"] = {**run(mate, pm), "backtest": backtest_summary(pm, mate)}
    res["config"] = cfg
    return res


def save(res, path):
    """Write the results (without the per-draw tables and traces) to `path`, relative to the repo root."""
    drop = lambda r: {k: v for k, v in r.items() if k not in ("draws", "traces")}
    out = {**drop(res), "repeat": drop(res["repeat"])} if "repeat" in res else drop(res)
    (ROOT / path).parent.mkdir(exist_ok=True)
    (ROOT / path).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")


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
    res = report(cfg)
    save(res, cfg["monte_carlo"]["results_path"])
    print(f"done in {time.perf_counter() - t0:.0f} s -> {cfg['monte_carlo']['results_path']}")
    for r in (res, res["repeat"]):
        f, a, pr, b = r["finish"], r["actual_strategy"], r["paired"], r["backtest"]
        print(f"\n=== {r['driver']} (real P{r['actual_position']}; back-test P{b['position']}, "
              f"{b['error_at_check_lap_s']:+.2f} s at lap {b['check_lap']}, {b['error_at_flag_s']:+.2f} s at the flag) ===")
        print(f"inters: expected P{f['expected']:.2f}, P(win) {f['p_win']:.1%}, P(podium) {f['p_podium']:.1%}  |  "
              f"actual strategy: expected P{a['expected']:.2f}")
        print(f"paired: better {pr['p_better']:.1%}, same {pr['p_same']:.1%}, worse {pr['p_worse']:.1%}; "
              f"{pr['expected_places_gained']:+.2f} places")
        print("inters finish:", f["distribution"])
    t = res["sensitivity"]
    print(f"\n=== tornado (n={t['n']}, base expected P{t['base_expected']:.2f}) ===")
    for b in t["bars"]:
        print(f"{b['label']:34s} {b['low']!s:>6} -> P{b['expected_low']:.2f}   {b['high']!s:>6} -> P{b['expected_high']:.2f}   swing {b['swing']:.2f}")
