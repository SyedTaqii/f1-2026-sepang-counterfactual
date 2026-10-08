"""M2: the phase table on the real race (uses cache/)."""
from src.data_loader import load_config, load_session
from src.segmentation import phase_table


def test_phase_table():
    cfg = load_config()
    t = phase_table(load_session(cfg), cfg)
    assert t.index[t.condition == "wet"].tolist() == list(range(1, 10))      # wet through the lap-9 crossover
    assert t.index[t.crossover].tolist() == [9]
    assert t.index[t.neutral == "sc"].tolist() == [1, 2, 9, 10, 11, 12, 45, 46, 47, 48, 49, 50, 51]
    assert t.index[t.neutral == "vsc"].tolist() == [43, 44]
    gap = t.slick_minus_inter_s.dropna()                                     # slicks catch inters lap by lap
    assert gap.is_monotonic_decreasing and gap.iloc[-1] < 1.0
