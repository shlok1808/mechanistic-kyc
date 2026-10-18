"""Offline tests for S7 steering + the side-effect suite (no model, no GPU)."""

import math

import numpy as np
import pytest

from side_effects import chunk_ids, is_refusal, mmlu_message, side_effect_verdict
from steering import (build_units, class_label, diff_in_means, direction_rows, per_vignette_rho,
                      random_directions, steering_verdict, stratified_ids, summarize_direction)

STEER = {"dose_spearman_min": 0.8, "control_shift_ratio_max": 0.25}


# -- directions --------------------------------------------------------------------------
def test_diff_in_means_is_high_minus_low():
    X = np.array([[1.0, 0], [3.0, 0], [0, 2.0], [0, 4.0]])
    v, n_hi, n_lo = diff_in_means(X, ["hi", "hi", "lo", "lo"], "hi", "lo")
    assert np.allclose(v, [2.0, -3.0]) and (n_hi, n_lo) == (2, 2)


def test_diff_in_means_rejects_empty_class():
    with pytest.raises(ValueError):
        diff_in_means(np.zeros((2, 3)), ["hi", "hi"], "hi", "lo")


def test_class_label_bins_factor_scores_and_passes_tier_through():
    assert class_label({"tier": "aggressive"}, "tier") == "aggressive"
    assert class_label({"willingness": 0.9}, "willingness") == "high"
    assert class_label({"capacity": 0.1}, "capacity") == "low"


def test_random_directions_have_matched_norm_and_are_seeded():
    a = random_directions(64, 5, 3.0, seed=1)
    assert a.shape == (5, 64) and np.allclose(np.linalg.norm(a, axis=1), 3.0, atol=1e-5)
    assert np.array_equal(a, random_directions(64, 5, 3.0, seed=1))


def test_stratified_ids_caps_each_stratum_and_is_stable():
    ids = [f"v{i}" for i in range(30)]
    strata = ["a"] * 20 + ["b"] * 10
    out = stratified_ids(ids, strata, 5, seed=0)
    assert len(out) == 10 and out == stratified_ids(ids, strata, 5, seed=0)
    assert sum(i in ids[:20] for i in out) == 5


# -- units -------------------------------------------------------------------------------
def test_dose_zero_is_shared_across_directions():
    dirs = {"names": ["tier", "random_0"], "eval_ids": ["v1", "v2"], "control_ids": ["v1"]}
    vby = {"v1": {"text": "a"}, "v2": {"text": "b"}}
    units = build_units(dirs, [-1, 0, 1], vby, "F", ["x", "y"], [[0, 1], [1, 0]])
    zero = [u for u in units if u["dose"] == 0]
    assert {u["direction"] for u in zero} == {"none"}
    assert len(zero) == 2 * 2                      # v1, v2 x 2 perms, scored once
    assert {u["vid"] for u in units if u["direction"] == "random_0"} == {"v1"}
    assert len(units) == 4 + 2 * 2 * 2 + 1 * 2 * 2  # dose0 + tier(+-1) + random(+-1)


def test_direction_rows_include_shared_baseline():
    rows = [{"direction": "none", "dose": 0, "vignette_id": "v1"},
            {"direction": "tier", "dose": 1, "vignette_id": "v1"},
            {"direction": "goals", "dose": 1, "vignette_id": "v1"},
            {"direction": "tier", "dose": 1, "vignette_id": "v9"}]
    got = direction_rows(rows, "tier", ["v1"])
    assert [(r["direction"], r["dose"]) for r in got] == [("none", 0), ("tier", 1)]


# -- dose-response -----------------------------------------------------------------------
def _rows(vid, doses, aggr):
    return [{"vignette_id": vid, "dose": d, "aggressiveness": a, "hedged": False}
            for d, a in zip(doses, aggr)]


def test_per_vignette_rho_monotone_flat_and_reversed():
    doses = [-2, -1, 0, 1, 2]
    rows = (_rows("up", doses, [0.1, 0.2, 0.3, 0.4, 0.5]) + _rows("flat", doses, [0.3] * 5)
            + _rows("down", doses, [0.5, 0.4, 0.3, 0.2, 0.1]))
    r = per_vignette_rho(rows)
    assert r["up"] == pytest.approx(1.0) and r["down"] == pytest.approx(-1.0)
    assert math.isnan(r["flat"])


def test_summarize_direction_counts_flat_and_measures_shift():
    doses = [-1, 0, 1]
    rows = _rows("a", doses, [0.2, 0.4, 0.6]) + _rows("b", doses, [0.5, 0.5, 0.5])
    s = summarize_direction(rows, n_boot=200, seed=0)
    assert s["rho_mean"] == pytest.approx(1.0) and s["n_flat"] == 1
    assert s["shift_full_range"] == pytest.approx(0.2)
    assert s["per_dose"][0]["n"] == 2


def test_verdict_needs_tier_pass_and_small_random_shifts():
    tier = {"rho_mean": 0.9, "shift_full_range": 0.20}
    ok = steering_verdict({"tier": tier, "random_0": {"rho_mean": 0.9, "shift_full_range": 0.02},
                           "random_1": {"rho_mean": -0.9, "shift_full_range": -0.04}}, STEER)
    assert ok["RQ2b_clean_win"]                  # high |rho| but tiny shift is still flat
    leaky = steering_verdict({"tier": tier, "random_0": {"rho_mean": 0.1, "shift_full_range": 0.08}}, STEER)
    assert leaky["tier_pass"] and not leaky["controls_flat"] and not leaky["RQ2b_clean_win"]
    weak = steering_verdict({"tier": {"rho_mean": 0.6, "shift_full_range": 0.2},
                             "random_0": {"rho_mean": 0.0, "shift_full_range": 0.0}}, STEER)
    assert not weak["RQ2b_clean_win"]


# -- side effects ------------------------------------------------------------------------
def test_refusal_detector_separates_declines_from_disclaimers():
    assert is_refusal("I'm sorry, but I can't provide financial advice.")
    assert is_refusal("I cannot help with that request.")
    assert not is_refusal("I'm not a financial advisor, but an index fund is a basket of stocks.")
    assert not is_refusal("A bond is a loan you make to a government or company.")


def test_mmlu_message_has_four_lettered_choices():
    msg = mmlu_message("2+2?", ["3", "4", "5", "6"])
    assert "A) 3" in msg and "D) 6" in msg and "Answer with only the letter." in msg


def test_chunk_ids_drops_ragged_tail():
    assert chunk_ids(list(range(10)), 4, 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert chunk_ids(list(range(100)), 10, 3) == [list(range(10)), list(range(10, 20)),
                                                  list(range(20, 30))]


BARS = {"mmlu_delta_max": 2.0, "ppl_ratio_max": 1.05, "refusal_delta_max": 0.05}


def test_side_effect_safe_range_stops_at_first_failing_magnitude():
    res = {0: {"mmlu_acc": 0.70, "ppl": 10.0, "refusal_rate": 0.0},
           1: {"mmlu_acc": 0.695, "ppl": 10.2, "refusal_rate": 0.02},
           -1: {"mmlu_acc": 0.70, "ppl": 10.1, "refusal_rate": 0.0},
           2: {"mmlu_acc": 0.60, "ppl": 10.3, "refusal_rate": 0.0},     # MMLU drop 10 pts
           -2: {"mmlu_acc": 0.70, "ppl": 10.1, "refusal_rate": 0.0}}
    v = side_effect_verdict(res, BARS)
    assert v["per_dose"][1]["pass"] and not v["per_dose"][2]["pass"]
    assert v["safe_abs_dose"] == 1


def test_side_effect_nothing_safe_when_smallest_dose_fails():
    res = {0: {"mmlu_acc": 0.7, "ppl": 10.0, "refusal_rate": 0.0},
           1: {"mmlu_acc": 0.7, "ppl": 12.0, "refusal_rate": 0.0}}
    assert side_effect_verdict(res, BARS)["safe_abs_dose"] == 0.0
