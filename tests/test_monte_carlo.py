"""M4: the counterfactual Monte Carlo is reproducible and its distribution is complete (uses cache/)."""
import pytest

from src.data_loader import load_config
from src.monte_carlo import run
from src.pipeline import prepare


def test_same_seed_same_answer_and_shares_sum_to_one():
    cfg = load_config()
    p = prepare(cfg)
    a, b = run(cfg, p, n=30), run(cfg, p, n=30)
    assert a["finish"] == b["finish"]
    assert sum(a["finish"]["distribution"].values()) == pytest.approx(1, abs=1e-3)   # shares are stored to 4 decimals
    assert 1 <= a["finish"]["expected"] <= 20
