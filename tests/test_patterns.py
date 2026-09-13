"""Offline unit tests for S3b (pattern analysis) and the shared stats helpers."""

import numpy as np
import yaml

from patterns import (fields_matrix, instruction_deltas, menu_use, rubric_field_names,
                          summarize_deltas, twin_gap_by_tier)
from utils.stats import partial_spearman, standardized_rel_weights


def _rubric():
    with open("config.yaml") as f:
        return yaml.safe_load(f)["rubric"]


# -- shared stats helpers ----------------------------------------------------------------
def test_partial_spearman_kills_pure_control_dependence():
    rng = np.random.default_rng(0)
    control = rng.normal(size=2000)
    x = control + rng.normal(scale=0.3, size=2000)     # x ~ control + independent noise
    y = control + rng.normal(scale=0.3, size=2000)     # y ~ control, no direct x-y link
    assert abs(partial_spearman(x, y, control)) < 0.1
    # raw dependence is strong; the partial is what removes it
    from scipy.stats import spearmanr
    assert spearmanr(x, y).correlation > 0.8


def test_partial_spearman_keeps_direct_dependence():
    rng = np.random.default_rng(1)
    control = rng.normal(size=500)
    x = rng.normal(size=500)
    y = x + 0.5 * control                              # y depends on x beyond control
    assert partial_spearman(x, y, control) > 0.7


def test_standardized_rel_weights_recovers_known_ratio():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=1000), rng.normal(size=1000)
    y = 2.0 * a + 1.0 * b
    betas, rel, r2 = standardized_rel_weights(np.c_[a, b], y)
    assert r2 > 0.99
    assert abs(rel[0] - 2 / 3) < 0.02 and abs(rel[1] - 1 / 3) < 0.02


# -- rubric field matrix -----------------------------------------------------------------
def test_fields_matrix_normalization_and_inversion():
    rubric = _rubric()
    fnames = rubric_field_names(rubric)
    p_low = {"past_drawdown_reaction": "sold_everything", "horizon_years": 1,
             "investing_experience": "none", "income_stability": "precarious",
             "stated_goal": "capital_preservation", "emergency_fund_months": 0,
             "dependents": 4, "age": 75}
    p_high = {"past_drawdown_reaction": "bought_more", "horizon_years": 35,
              "investing_experience": "extensive", "income_stability": "stable",
              "stated_goal": "maximum_growth", "emergency_fund_months": 24,
              "dependents": 0, "age": 23}
    X = fields_matrix([p_low, p_high], rubric, fnames)
    # every field of the all-low profile normalizes to 0, all-high to 1 (incl. inversions:
    # dependents=4 and age=75 SUPPORT low risk -> 0)
    assert np.allclose(X[0], 0.0) and np.allclose(X[1], 1.0)


# -- menu / twin-gap / instruction summaries ---------------------------------------------
def _row(pid, tier, aggr, p_alloc, vtype="implicit"):
    return {"profile_id": pid, "tier": tier, "aggressiveness": aggr, "p_alloc": p_alloc,
            "vignette_type": vtype, "vignette_id": f"v_{pid}_{vtype}", "contradictory": False}


def test_menu_use_argmax_shares():
    rows = [_row("p1", "aggressive", 0.9, [0, 0, 0, 1]),
            _row("p2", "aggressive", 0.7, [0, 0, 1, 0]),
            _row("p3", "aggressive", 0.7, [0, 0, 1, 0])]
    mu = menu_use(rows, ["conservative", "moderate", "aggressive"])
    assert mu["aggressive"]["argmax_share"] == [0.0, 0.0, round(2 / 3, 4), round(1 / 3, 4)]
    assert "conservative" not in mu           # no rows -> no entry


def test_twin_gap_by_tier():
    exp = [_row("p1", "aggressive", 0.8, [0, 0, 0, 1], "explicit"),
           _row("p2", "conservative", 0.3, [1, 0, 0, 0], "explicit")]
    imp = [_row("p1", "aggressive", 0.6, [0, 0, 1, 0]),
           _row("p2", "conservative", 0.28, [1, 0, 0, 0])]
    gap = twin_gap_by_tier(exp, imp, ["conservative", "moderate", "aggressive"])
    assert abs(gap["aggressive"]["mean_gap"] - 0.2) < 1e-9
    assert abs(gap["conservative"]["mean_gap"] - 0.02) < 1e-9
    assert gap["moderate"]["n"] == 0


def test_instruction_deltas_and_summary():
    base, instr = [], []
    for i, (tier, b, d) in enumerate([("conservative", 0.3, -0.01),
                                      ("aggressive", 0.5, +0.03),
                                      ("aggressive", 0.6, +0.01)]):
        r = _row(f"p{i}", tier, b, [1, 0, 0, 0])
        base.append(r)
        r2 = dict(r)
        r2["aggressiveness"] = b + d
        instr.append(r2)
    deltas = instruction_deltas(base, instr)
    assert len(deltas) == 3
    s = summarize_deltas(deltas, ["conservative", "moderate", "aggressive"])
    assert s["by_tier"]["aggressive"]["n"] == 2
    assert abs(s["by_tier"]["aggressive"]["mean_delta"] - 0.02) < 1e-9
    assert s["by_tier"]["conservative"]["share_positive"] == 0.0
