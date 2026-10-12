"""PRD: every model number in README.md comes from outputs/results.json."""
import json

from src.data_loader import ROOT


def test_readme_numbers_come_from_results():
    r = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8").replace("−", "-")
    f, a, pr, b, m, i, c = (r[k] for k in ("finish", "actual_strategy", "paired", "backtest", "repeat", "inputs", "context"))
    bar = {x["key"]: x for x in r["sensitivity"]["bars"]}
    wet = r["positions_by_lap"]["end_of_wet_phase"]["distribution"]
    pct = lambda x: f"{100 * x:.1f}%"
    expected = [
        f"{r['n']:,}", f"seed {r['seed']}",
        f"P{f['expected']:.2f}", f"P{a['expected']:.2f}", pct(f["distribution"]["3"]), pct(f["p_podium"]),
        f"{100 * f['p_win']:.2f}%", pct(a["distribution"]["3"]), pct(a["p_podium"]),
        f"{f['gap_to_winner_s']['median']:.2f} s", f"{a['gap_to_winner_s']['median']:.2f} s",
        pct(pr["p_same"]), pct(pr["p_better"]), pct(pr["p_worse"]), f"{pr['expected_places_gained']:+.2f} places",
        f"median of {pr['time_gained_s']['median']:.2f} s", f"{f['expected_gain_vs_actual']:+.2f} places",
        pct(wet["1"]), pct(sum(wet.get(str(k), 0.0) for k in range(1, 5))),
        f"{c['crossover_lap_stoppers']} of the {c['other_cars_running_at_crossover']} other cars",
        f"VER was {-c['dry_pace_vs_target_s_per_lap']['VER']:.2f} s/lap", f"ANT {-c['dry_pace_vs_target_s_per_lap']['ANT']:.2f} s/lap",
        f"P{bar['monte_carlo.dry_pace_offset_s']['expected_low']:.2f} to P{bar['monte_carlo.dry_pace_offset_s']['expected_high']:.2f}",
        f"swing of {bar['monte_carlo.dry_pace_offset_s']['swing']:.2f}", f"swing {bar['pace.wet_sc_wear']['swing']:.2f}",
        f"swing {bar['counterfactual.crossover_lap']['swing']:.2f}", f"swing {bar['monte_carlo.fixed.inter_offset_s']['swing']:.2f}",
        f"{i['inter_offset_s_per_lap']['mu']:.2f} ± {i['inter_offset_s_per_lap']['sigma']:.2f} s/lap",
        f"{i['pass_cost_s']:.2f} s per car", f"{i['dirty_air_s']:.2f} s per lap", f"{i['green_pit_loss_median_s']:.1f} s",
        f"0–{max(i['start_gaps_s']):.2f} s",
        f"**P{b['position']}**", f"{b['error_at_check_lap_s']:.2f} s off at lap {b['check_lap']}",
        f"{b['error_at_flag_s']:+.2f} s at the flag", f"±{b['tolerance_s']:g} s",
        f"**P{m['backtest']['position']}**", f"{m['backtest']['error_at_check_lap_s']:+.2f} s", f"{m['backtest']['error_at_flag_s']:+.2f} s",
        f"{m['inputs']['inter_offset_s_per_lap']['mu']:.2f} ± {m['inputs']['inter_offset_s_per_lap']['sigma']:.2f} s/lap",
        f"P{m['finish']['expected']:.2f}", f"P{m['actual_strategy']['expected']:.2f}", pct(m["paired"]["p_better"]),
        f"{m['paired']['expected_places_gained']:+.2f} places",
    ]
    assert not [s for s in expected if s not in readme]
