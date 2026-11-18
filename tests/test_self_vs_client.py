"""Offline tests for RQ2c self vs. client (no model, no GPU)."""

import numpy as np
import pytest

from self_vs_client import (PREFIXES, VERBAL, VERBAL_PILOT, common_kmax, deal_tracking, leak_verdict,
                            leaks, lottery_message, numeric_items, options_block, paired_shifts,
                            per_item_rho, split_by_cell, verbal_items)


# -- items -------------------------------------------------------------------------------
def test_numeric_items_are_ev_ties_at_ratio_one_and_drop_dominated():
    items = numeric_items([0.5, 0.9], [1.0, 1.15], [1000], drop_dominated=True)
    tie = next(i for i in items if i["p"] == 0.5 and i["ratio"] == 1.0)
    assert tie["S"] == 500 and tie["X"] == 1000
    assert not any(i["S"] >= i["X"] for i in items)               # p .9 x 1.15 dropped
    assert len(items) == 3


def test_split_by_cell_keeps_every_scale_of_a_cell_together():
    items = numeric_items([0.1, 0.5], [0.9, 1.0, 1.1], [100, 200, 300]) + verbal_items(VERBAL)
    fit, ev = split_by_cell(items, 0.5, seed=1)
    assert {i["cell"] for i in fit}.isdisjoint({i["cell"] for i in ev})
    assert len(fit) + len(ev) == len(items)
    assert any(i["kind"] == "verbal" for i in fit) and any(i["kind"] == "verbal" for i in ev)


def test_pilot_and_main_verbal_items_never_overlap():
    assert not {s for s, _ in VERBAL} & {s for s, _ in VERBAL_PILOT}


# -- prompts -----------------------------------------------------------------------------
def test_options_span_is_identical_across_prefixes_and_referents():
    it = numeric_items([0.5], [1.0], [1000])[0]
    spans = set()
    for name, ref in (("S_lot", "self"), ("C_lot", "client"), ("N_lot", "self")):
        for pre in PREFIXES[name]:
            msg, opts, slot = lottery_message(it, True, pre, ref)
            assert opts in msg and pre in msg and slot == 0
            assert msg.index(pre) < msg.index(opts)
            spans.add(opts)
    assert len(spans) == 1
    assert "risk" not in spans.pop().lower()


def test_gamble_slot_follows_order():
    it = numeric_items([0.5], [1.0], [1000])[0]
    assert options_block(it, True).startswith("A) A 50% chance")
    assert lottery_message(it, False)[2] == 1


# -- gates -------------------------------------------------------------------------------
def _rows(fn):
    return [{"p": p, "X": X, "ratio": r, "p_gamble": fn(r)} for p in (0.5,) for X in (100, 200)
            for r in (0.85, 1.0, 1.15)]


def test_deal_tracking_flat_floor_fails_and_counts_flat_as_zero():
    flat = deal_tracking(_rows(lambda r: 0.01))
    assert flat["within_cell_rho"] == 0.0 and flat["slope"] == 0.0


def test_deal_tracking_tiny_monotone_wobble_has_rho_but_no_slope():
    dt = deal_tracking(_rows(lambda r: 0.02 - 0.01 * r))
    assert dt["within_cell_rho"] == pytest.approx(-1.0) and dt["slope"] < 0.01


def test_deal_tracking_real_tracking():
    dt = deal_tracking(_rows(lambda r: 1.0 - r * 0.6))
    assert dt["within_cell_rho"] == pytest.approx(-1.0) and dt["slope"] == pytest.approx(0.18)


# -- shifts, leaks, verdict --------------------------------------------------------------
def test_paired_shifts_and_flat_items_count_as_zero_rho():
    rows = [{"item": "a", "k": k, "y": 0.5 + 0.1 * k} for k in (-2, -1, 0, 1, 2)]
    rows += [{"item": "b", "k": k, "y": 0.3} for k in (-2, -1, 0, 1, 2)]
    sh = paired_shifts(rows, 2)
    assert sh == {"a": pytest.approx(0.4), "b": 0.0}
    assert per_item_rho(rows, 2) == pytest.approx(0.5)            # a: 1.0, b (flat): 0


def _shift(mean, n=200, sd=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return {f"i{i}": v for i, v in enumerate(rng.normal(mean, sd, n))}


def _lk(s_adv, c_adv, s_lot, c_lot):
    return leaks({("S", "advice"): _shift(s_adv, seed=1), ("C", "advice"): _shift(c_adv, seed=2),
                  ("S", "lottery"): _shift(s_lot, seed=3), ("C", "lottery"): _shift(c_lot, seed=4)})


def test_leak_verdicts_cover_the_four_outcomes():
    assert leak_verdict(_lk(0.01, 0.15, 0.40, 0.02)) == "DISTINCT"
    assert leak_verdict(_lk(0.14, 0.15, 0.40, 0.38)) == "CONFLATION"
    assert leak_verdict(_lk(0.14, 0.15, 0.40, 0.02)) == "ASYMMETRIC"
    assert leak_verdict(_lk(0.07, 0.15, 0.40, 0.20)) == "PARTIAL"


def test_leaks_are_column_normalized():
    lk = _lk(0.05, 0.10, 0.40, 0.08)
    assert lk["leak_self"] == pytest.approx(0.5, abs=0.05)
    assert lk["leak_client"] == pytest.approx(0.2, abs=0.05)
    assert lk["product"] == pytest.approx(0.1, abs=0.03)


def test_common_kmax_is_shared_across_directions_and_tasks():
    hedge = {("S", "advice", 1): 0.0, ("S", "advice", -1): 0.0, ("S", "advice", 2): 0.0,
             ("C", "lottery", 2): 0.0, ("C", "lottery", -2): 0.3, ("S", "advice", 4): 0.0}
    assert common_kmax(hedge, [1, 2, 4], 0.10) == 1
    assert common_kmax({k: 0.0 for k in hedge}, [1, 2, 4], 0.10) == 4


# -- v2 (amendment 3) --------------------------------------------------------------------
def test_v2_lottery_scales_are_fresh_and_split_is_clean():
    import yaml
    from self_vs_client import items_v2
    cfg = yaml.safe_load(open("config.yaml"))
    sv = cfg["self_vs_client"]
    old = set(sv["amended"]["lotteries"]["scales"]) | set(sv["amended"]["pilot_scales"])
    assert not set(sv["v2"]["lottery"]["scales"]) & old
    assert not set(sv["v2"]["lottery"]["pilot_scales"]) & (old | set(sv["v2"]["lottery"]["scales"]))
    items, fit, ev = items_v2(cfg)
    assert {i["cell"] for i in fit}.isdisjoint({i["cell"] for i in ev})
    assert len({(i["safe"], i["risky"]) for i in items}) == len(items)


def test_v2_options_span_identical_with_and_without_prefix():
    from self_vs_client import LOT_V2, lottery_message_v2
    it = numeric_items([0.5], [1.0], [1100])[0]
    for variant in LOT_V2:
        m0, o0, s0 = lottery_message_v2(it, False, variant)
        m1, o1, s1 = lottery_message_v2(it, False, variant, PREFIXES["S_lot"][0])
        assert o0 == o1 and s0 == s1 == 1 and "letter" in m0 and m1.startswith("You are")


def test_hedged_v2_absolute_and_relative_rules():
    from self_vs_client import hedged_v2
    assert hedged_v2(0.49, 1.0, "absolute") and not hedged_v2(0.51, 1.0, "absolute")
    assert hedged_v2(0.9, 0.4, "absolute")                       # argmax usually not a letter
    assert not hedged_v2(0.31, 1.0, "relative", base_mass=0.6)    # >= 0.5 x own baseline
    assert hedged_v2(0.29, 1.0, "relative", base_mass=0.6)
    assert hedged_v2(0.24, 1.0, "relative", base_mass=0.4)        # absolute floor 0.25


def test_lever_rule_needs_both_orders_same_way():
    from self_vs_client import lever_ok, order_shift_ci
    both = [{"item": f"i{j}", "k": k, "gamble_first": 0.5 + 0.1 * k + 0.001 * j,
             "gamble_second": 0.3 + 0.05 * k} for j in range(30) for k in (-2, 2)]
    pos_only = [{"item": f"i{j}", "k": k, "gamble_first": 0.5 + 0.1 * k + 0.001 * j,
                 "gamble_second": 0.3 - 0.05 * k} for j in range(30) for k in (-2, 2)]
    assert lever_ok(order_shift_ci(both, 2))
    assert not lever_ok(order_shift_ci(pos_only, 2))             # a letter-position lever
