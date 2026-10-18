"""Offline tests for S6 patching (no model)."""

import math

from patching import flip_summary, is_dev_pair, pair_members, site_positions, transfer_effect


def test_pair_members_drops_incomplete_pairs():
    rows = [{"pair_id": "pair_0001", "pair_side": "lo"}, {"pair_id": "pair_0001", "pair_side": "hi"},
            {"pair_id": "pair_0002", "pair_side": "lo"}]
    assert list(pair_members(rows)) == ["pair_0001"]


def test_site_positions():
    assert site_positions("profile_end", 5, 9) == [5]
    assert site_positions("post_profile", 5, 9) == [6, 7, 8]


def test_transfer_effect_scale_and_zero_gap():
    assert abs(transfer_effect(0.2, 0.6, 0.4) - 0.5) < 1e-9
    assert transfer_effect(0.6, 0.2, 0.2) == 1.0
    assert math.isnan(transfer_effect(0.3, 0.3, 0.3))


def test_dev_report_split_by_parity():
    assert is_dev_pair("pair_0002") and not is_dev_pair("pair_0003")


def test_flip_summary_only_counts_eligible_pairs():
    rows = [{"base": 0.2, "donor": 0.6, "patched": 0.5},     # effect 0.75 flip
            {"base": 0.2, "donor": 0.6, "patched": 0.3},     # effect 0.25 no flip
            {"base": 0.4, "donor": 0.42, "patched": 0.42}]   # ineligible (gap 0.02)
    s = flip_summary(rows, 0.05, 0.5)
    assert s["n_eligible"] == 2 and s["flip_rate"] == 0.5 and s["mean_effect"] == 0.5
