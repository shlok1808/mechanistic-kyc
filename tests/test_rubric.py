import random
from collections import Counter

import pytest

from rubric import (BINS, TIERS, factor_bin, factor_cell, factor_score, factor_scores,
                    is_conflicted, load_rubric, risk_score, risk_tier, sample_profile)
from profiles import generate_cell_balanced_profiles

ALL_CONSERVATIVE = {
    "past_drawdown_reaction": "sold_everything",
    "horizon_years": 1,
    "investing_experience": "none",
    "income_stability": "precarious",
    "stated_goal": "capital_preservation",
    "emergency_fund_months": 0,
    "dependents": 4,
    "age": 75,
}

ALL_AGGRESSIVE = {
    "past_drawdown_reaction": "bought_more",
    "horizon_years": 35,
    "investing_experience": "extensive",
    "income_stability": "stable",
    "stated_goal": "maximum_growth",
    "emergency_fund_months": 24,
    "dependents": 0,
    "age": 23,
}

BORING_MIDDLE = {
    "past_drawdown_reaction": "held",
    "horizon_years": 18,
    "investing_experience": "some",
    "income_stability": "variable",
    "stated_goal": "balanced_growth",
    "emergency_fund_months": 12,
    "dependents": 2,
    "age": 49,
}


@pytest.fixture(scope="module")
def rubric():
    return load_rubric()


def test_weights_sum_to_one(rubric):
    total = sum(spec["weight"] for spec in rubric["fields"].values())
    assert total == pytest.approx(1.0)


def test_all_conservative_profile_scores_zero(rubric):
    score = risk_score(ALL_CONSERVATIVE, rubric)
    assert score == pytest.approx(0.0)
    assert risk_tier(score, rubric) == "conservative"


def test_all_aggressive_profile_scores_hundred(rubric):
    score = risk_score(ALL_AGGRESSIVE, rubric)
    assert score == pytest.approx(100.0)
    assert risk_tier(score, rubric) == "aggressive"


def test_boring_middle_profile_is_moderate(rubric):
    score = risk_score(BORING_MIDDLE, rubric)
    assert score == pytest.approx(56.33, abs=0.01)
    assert risk_tier(score, rubric) == "moderate"


def test_cell_balanced_generation_fills_every_factor_cell(rubric):
    """Sampling balances the willingness x capacity grid, not the blended tier."""
    profiles, drawn = generate_cell_balanced_profiles(rubric, n_total=90, seed=42)
    counts = Counter(p["factor_cell"] for p in profiles)
    assert len(counts) == len(BINS) ** 2          # every cell present
    assert set(counts.values()) == {90 // len(counts)}   # and equally full
    assert drawn >= len(profiles)


def test_cell_balancing_oversamples_conflict_cases(rubric):
    """The point of the grid: conflict cases stop being a 3.5% rounding error."""
    profiles, _ = generate_cell_balanced_profiles(rubric, n_total=900, seed=7)
    share = sum(p["conflicted"] for p in profiles) / len(profiles)
    assert share > 0.15          # pilot (tier-balanced) sat near 0.07


def test_factor_scores_span_unit_interval(rubric):
    """Each factor is renormalised within itself, so extremes hit 0 and 1."""
    assert factor_score(ALL_CONSERVATIVE, rubric, "willingness") == pytest.approx(0.0)
    assert factor_score(ALL_AGGRESSIVE, rubric, "willingness") == pytest.approx(1.0)
    assert factor_score(ALL_CONSERVATIVE, rubric, "capacity") == pytest.approx(0.0)
    assert factor_score(ALL_AGGRESSIVE, rubric, "capacity") == pytest.approx(1.0)


def test_every_rubric_field_belongs_to_exactly_one_factor(rubric):
    """No field may be double-counted or orphaned, or factor scores stop being clean."""
    assigned = [f for names in rubric["factors"].values() for f in names]
    assert sorted(assigned) == sorted(rubric["fields"])
    assert len(assigned) == len(set(assigned))


def test_conflict_requires_opposing_bins(rubric):
    """Conflict is willingness vs capacity, not merely a low or high overall score."""
    assert not is_conflicted(ALL_CONSERVATIVE, rubric)   # both low -> concordant
    assert not is_conflicted(ALL_AGGRESSIVE, rubric)     # both high -> concordant
    mixed = dict(ALL_CONSERVATIVE, past_drawdown_reaction="bought_more",
                 investing_experience="extensive")
    assert factor_cell(mixed, rubric) == ("high", "low")
    assert is_conflicted(mixed, rubric)


def test_naive_distribution_matches_sanity_check(rubric):
    rng = random.Random(0)
    n = 50000
    counts = Counter()
    for i in range(n):
        profile = sample_profile(rubric, rng, profile_id=str(i))
        counts[profile["tier"]] += 1
    fractions = {tier: counts[tier] / n for tier in TIERS}
    assert fractions["conservative"] == pytest.approx(0.26, abs=0.03)
    assert fractions["moderate"] == pytest.approx(0.64, abs=0.03)
    assert fractions["aggressive"] == pytest.approx(0.10, abs=0.03)
