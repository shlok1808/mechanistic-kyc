"""Offline unit tests for case_studies.py pure logic (no plotting, no results files)."""

import numpy as np
import pytest

from case_studies import (bootstrap_mean_ci, field_effects, fit_residuals, pair_shifts,
                          percentile_of, select_case)


def row(vid, pid, risk, aggr, **kw):
    return {"vignette_id": vid, "profile_id": pid, "risk_score": risk, "aggressiveness": aggr, **kw}


def pool():
    # advice = 0.01 * risk exactly, except three deliberate outliers
    rows = [row(f"v{i:02d}", f"p{i:02d}", 10 * i, 0.1 * i, contradictory=False, conflicted=False)
            for i in range(10)]
    rows[3]["aggressiveness"] += 0.4            # over-aggressive outlier
    rows[7]["aggressiveness"] -= 0.3            # over-conservative outlier
    rows[5]["contradictory"] = True
    profiles = {r["profile_id"]: {"stated_goal": "maximum_growth" if i in (2, 3, 4) else "income"}
                for i, r in enumerate(rows)}
    return rows, profiles


def test_fit_residuals_recovers_clean_line():
    rows = [row(f"v{i}", f"p{i}", i, 0.2 + 0.01 * i) for i in range(20)]
    resid, (slope, b) = fit_residuals(rows)
    assert slope == pytest.approx(0.01) and b == pytest.approx(0.2)
    assert np.allclose(resid, 0, atol=1e-9)


def test_select_extremes_and_filters():
    rows, profiles = pool()
    resid, _ = fit_residuals(rows)
    assert rows[select_case(rows, profiles, resid, {"pick": "max_residual"})]["vignette_id"] == "v03"
    assert rows[select_case(rows, profiles, resid, {"pick": "min_residual"})]["vignette_id"] == "v07"
    # profile-level filter (stated_goal) restricts to v02-v04; typical member = closest to group mean
    i = select_case(rows, profiles, resid, {"filter": {"stated_goal": "maximum_growth"},
                                            "pick": "closest_to_group_mean"})
    assert rows[i]["profile_id"] in {"p02", "p03", "p04"}
    # row-level filter + risk band: contradictory row excluded, only |risk-50| >= 25 kept
    i = select_case(rows, profiles, resid, {"filter": {"contradictory": False},
                                            "min_abs_risk_from_50": 25, "pick": "min_abs_residual"})
    assert abs(rows[i]["risk_score"] - 50) >= 25 and rows[i]["vignette_id"] != "v05"


def test_select_is_deterministic_on_ties():
    rows = [row(v, v, 50, 0.5) for v in ("vb", "va", "vc")]
    resid = np.zeros(3)
    for pick in ("min_abs_residual", "median_residual", "max_residual", "min_residual",
                 "closest_to_group_mean"):
        assert rows[select_case(rows, {}, resid, {"pick": pick})]["vignette_id"] == "va"


def test_select_raises_on_empty_and_unknown():
    rows, profiles = pool()
    resid, _ = fit_residuals(rows)
    with pytest.raises(ValueError):
        select_case(rows, profiles, resid, {"filter": {"stated_goal": "nope"}, "pick": "max_residual"})
    with pytest.raises(ValueError):
        select_case(rows, profiles, resid, {"pick": "vibes"})


def test_pair_shifts_matches_lo_hi_and_skips_incomplete():
    pr = [row("v_pair_0000_lo", "a", 40, 0.30, pair_id="pair_0000", pair_field="stated_goal", pair_factor="goals"),
          row("v_pair_0000_hi", "b", 48, 0.75, pair_id="pair_0000", pair_field="stated_goal", pair_factor="goals"),
          row("v_pair_0001_hi", "c", 60, 0.50, pair_id="pair_0001", pair_field="age", pair_factor="capacity")]
    s = pair_shifts(pr)
    assert list(s) == ["pair_0000"]
    assert s["pair_0000"]["shift"] == pytest.approx(0.45)
    assert s["pair_0000"]["d_risk"] == pytest.approx(8)


def test_field_effects_and_wrong_direction_share():
    shifts = {f"p{i}": {"field": "dependents", "factor": "capacity", "shift": v, "d_risk": 4}
              for i, v in enumerate([0.1, -0.05, 0.02, 0.03])}
    e = field_effects(shifts, n_boot=200, seed=0)["dependents"]
    assert e["n"] == 4 and e["share_wrong_direction"] == 0.25
    assert e["ci"][0] <= e["mean_shift"] <= e["ci"][1]


def test_bootstrap_and_percentile():
    m, lo, hi = bootstrap_mean_ci([1.0] * 10)
    assert m == lo == hi == 1.0
    assert percentile_of(5, range(10)) == 60.0
