# What if Hamilton had started on inters?

**A data-driven counterfactual of the 2026 Bahrain Grand Prix (Round 16, held at Sepang).** Ferrari started both cars on slicks on a wet track. Hamilton dropped from P2 on the grid to P17 by lap 4, and was 83 s behind the leader after lap 8. He still finished P3.

The question: **where would he most likely have finished had he started on intermediates?**

## The answer: most likely the same P3

| 10,000 Monte Carlo draws, seed 16 | Expected finish | P3 | Podium | Win | Gap to winner (median) |
|---|---|---|---|---|---|
| **Started on inters** | **P3.20** | 88.5% | 88.8% | 0.01% | 5.12 s |
| Actual strategy (slicks), same draws | P3.41 | 79.3% | 79.4% | 0% | 5.40 s |

Each draw races both strategies with the same sampled pace, traffic, pit-stop and noise inputs, so the comparison is like for like. Draw by draw, inters give:
- **the same position in 85.4%** of draws;
- a better one in 14.1%;
- a worse one in 0.6%.

That's **+0.21 places** on average, and a median of 0.21 s at the flag.

Compared with his single real result (P3, with a normal pit stop), inters look −0.20 places worse. That gap is pit-stop luck, not the tyre choice: the simulation includes the chance of a slow stop, and the paired comparison takes that luck out.

![Finishing positions](outputs/3_finish_distribution.png)

### Why it's P3 either way

1. **On inters he'd have run at the front.** By lap 8 the ghost car is P1 in 41.6% of draws and P1–P4 in 98.6%.
2. **The lap-9 safety car freezes that order.** The leaders all switched to slicks under it: 18 of the 20 other cars still running pitted that lap, so the stop gains or costs nobody anything. The same SC also cut the real Hamilton's 83 s deficit to 6.5 s, which is why the real strategy error cost so little.
3. **The dry race decides.** Against Hamilton's dry pace, VER was 0.61 s/lap faster and ANT 0.26 s/lap. On inters he settles into P3–P4 behind them.
4. **The late VSC and SC (laps 43–51) bunch the field again**, and almost any position from P1 to P4 at lap 42 becomes P3 at the flag.

![Ghost vs actual](outputs/2_ghost_vs_actual.png)

### What moves the answer

![Tornado](outputs/4_tornado.png)

One assumption at a time, set to a low and a high value (1,000 draws each, same seed). The bar is the change in his expected finishing position on inters.
- **Dry pace dominates.** ±0.3 s/lap moves the expected finish from P2.96 to P3.87: a swing of 0.92 places.
- **Next come two safety-car and tyre questions:**
  - whether wet/SC laps wear the tyres (swing 0.27);
  - which lap he pits for slicks (swing 0.23). Pitting a lap early, on lap 8, means a green stop while everyone else waits for the SC.
- **The inter-pace inputs barely register** (swing 0.03), and they are what the question is about. The lap-9 SC wipes out whatever he gains or loses on inters.

## How the model works

1. **Data:** FastF1 (`2026, round 16, 'R'`). Every fact is checked against the official classification first. FastF1 has Leclerc's timing 19.5 s late from lap 48, so it is shifted to match the official result.
2. **Race phases:** wet laps 3–8, slicks from the lap-9 SC, then the dry race, then the late VSC/SC.
3. **Dry pace:** one regression over every car's clean dry laps: tyre compound + wear × tyre age + fuel/track evolution + a driver offset. Lap effects capture what the whole field felt on each lap (the track still drying on laps 13–16).
4. **Inter pace:** Hamilton never ran inters. His pace on them is Leclerc's inter laps adjusted by their dry-pace gap, blended with a cross-check (his dry gap to the leaders, scaled to wet lap times): **−0.94 ± 0.78 s/lap** against the leaders' median inter lap.
5. **Ghost-car simulator** (`src/simulator.py`, generic): races one car lap by lap against the other cars' *recorded* times.
   - **Traffic** has two costs, both measured on the other drivers: **0.36 s per car passed** and **0.49 s per lap in dirty air** (within 1 s of a car ahead).
   - **Under the SC or VSC** the ghost copies the real cars around it on the road.
   - **A stop under the SC/VSC** copies the cars that really stopped that lap.
   - **A green stop** costs the median of the others' stops, 26.6 s.
6. **The counterfactual** changes only the start tyre:
   - a standing-start gap sampled from the inter starters near the front (0–5.14 s);
   - laps 4–8 at the leaders' inter pace plus his offset;
   - a pit stop for slicks under the lap-9 SC;
   - then his own real dry strategy.
7. **Monte Carlo:** 10,000 paired draws. Each samples the inter pace, the dry-fit coefficients, the traffic costs, the pit loss, the start gap and lap-to-lap noise.

## Validation

- **Back-test (Hamilton):** replaying his *actual* strategy from lap 8 through the same engine gives **P3** (real: P3). The time is **−2.97 s off at lap 42**, before the late SC resets the gaps, and **+0.21 s at the flag**; the tolerance is ±5 s. Lap 42 is checked because the late SC hides errors at the flag: an early version was within 0.7 s at the flag but 11.9 s off at lap 42.
- **Honesty notes:**
  - The check is in-sample for his pace (he is the pace fit's reference driver).
  - Two errors that cancelled each other were found and fixed along the way.
  - The dirty-air term was added after an honest back-test failed. It is measured on the other drivers, and was fixed before rerunning.
- **Second check (Leclerc), which fails.** Replaying Leclerc's actual race gives **P9** against his real P4, **+8.37 s** off at lap 42 and **+7.90 s** at the flag. His pace drifted during the race (slower than his average early, about 0.9 s/lap faster in the final sprint), and his climb through the field cost him far less than the field-average traffic costs. The model gives each driver one pace offset and average traffic costs. That works for Hamilton, whose race sits near those averages, but not for Leclerc.

## The Leclerc repeat

Leclerc ran inters himself (laps 4–9, after a lap-3 collision and pit stop), so his own laps replace the proxy: **−0.57 ± 0.33 s/lap** against the leaders. Starting on inters, the model has him at expected **P6.42**, against **P8.88** for his actual strategy in the same draws. Inters finish higher in **88.4%** of paired draws, **+2.46 places** on average. Because his back-test fails, **read only the direction**: starting on inters, and so skipping the lap-3 stop, would very likely have helped him. The sizes aren't trustworthy.

## Data limits

- No tyre temperatures, fuel loads, energy data or team overtaking models.
- **Hamilton has no inter laps**, so his inter pace is estimated through Leclerc. It barely matters here, because the lap-9 SC resets the wet phase.
- Wet-race timing is messy: an aborted start behind the SC, a suspended start procedure, Leclerc's 19.5 s timing glitch, and tyre compounds recorded as unknown.

## Caveats

- **Static field.** Rivals' races are their real ones; nobody reacts to the ghost. Real cars pass it for free, so it can never defend a position.
- **SC asymmetry.** The late SC helped Hamilton's real recovery. Starting near the front, it would have helped the cars behind him instead, and in the model it pulls almost every result back to P3.
- **Proxy assumption.** Leclerc's inter pace, adjusted by their dry gap, may not reflect Hamilton's.
- **Traffic is simplified** to two field-average costs. Trains of close cars and driver-specific passing aren't captured: Hamilton lost about twice the average per pass, Leclerc almost nothing.
- **One pace offset per driver** for the whole dry race.

## Reproduce

```powershell
py -3.13 -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
python -m src.monte_carlo     # 10k paired draws + tornado + Leclerc repeat -> outputs/results.json (~6 min)
python -m src.plots           # the 4 charts -> outputs/*.png
pytest                        # simulator mechanics, back-test, Monte Carlo, README numbers
```

The notebook `notebooks/sepang_counterfactual.ipynb` runs everything end to end, with the reasoning for each step. Every model number in this README comes from `outputs/results.json`, and a test checks that. The race facts come from `python -m src.data_loader`. Every assumption is in `config.yaml`; design decisions and their history are in `Decisions.md`.
