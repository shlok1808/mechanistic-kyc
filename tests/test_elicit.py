"""Offline unit tests for S3c (verbal elicitation baseline; no model / no GPU)."""

from advice import aggregate_allocation_probs, cyclic_permutations, slot_options
from elicit import build_elicit_message, build_units, elicit_score, tier_display

TIERS = ["conservative", "moderate", "aggressive"]


# -- prompt assembly ---------------------------------------------------------------------
def test_build_elicit_message_letters_follow_permutation():
    msg = build_elicit_message("FRAMING", "I retired last year and sleep poorly after losses.",
                               slot_options(TIERS, [2, 0, 1]))
    assert "A) Aggressive" in msg and "B) Conservative" in msg and "C) Moderate" in msg
    assert "sleep poorly after losses" in msg
    assert "Respond with only the letter" in msg
    assert "D)" not in msg                     # exactly three options


def test_tier_display_capitalizes():
    assert tier_display("conservative") == "Conservative"


# -- score math --------------------------------------------------------------------------
def test_elicit_score_endpoints_and_midpoint():
    assert elicit_score([1, 0, 0]) == 0.0
    assert elicit_score([0, 0, 1]) == 1.0
    assert abs(elicit_score([1 / 3, 1 / 3, 1 / 3]) - 0.5) < 1e-9


# -- units + permutation aggregation ------------------------------------------------------
def _vignette(i):
    return {"vignette_id": f"v{i}", "profile_id": f"p{i}", "pair_id": None,
            "vignette_type": "implicit", "tier": "moderate", "risk_score": 50.0,
            "contradictory": False, "text": f"narrative {i}"}


def test_build_units_one_per_vignette_perm():
    perms = cyclic_permutations(3)
    units = build_units([_vignette(0), _vignette(1)], "F", TIERS, perms)
    assert len(units) == 6
    assert sorted({(u["vid"], u["perm_idx"]) for u in units}) == [
        ("v0", 0), ("v0", 1), ("v0", 2), ("v1", 0), ("v1", 1), ("v1", 2)]


def test_three_option_aggregation_cancels_position_bias():
    # A pure slot-A picker must aggregate to uniform over tiers -> elicit_score 0.5.
    perms = cyclic_permutations(3)
    perm_letter_probs = [[1.0, 0.0, 0.0] for _ in perms]
    p_tier = aggregate_allocation_probs(perm_letter_probs, perms)
    assert all(abs(p - 1 / 3) < 1e-9 for p in p_tier)
    assert abs(elicit_score(p_tier) - 0.5) < 1e-9


def test_three_option_aggregation_recovers_true_tier():
    perms = cyclic_permutations(3)
    true_tier = 2                                        # aggressive
    perm_letter_probs = []
    for perm in perms:
        probs = [0.0, 0.0, 0.0]
        probs[perm.index(true_tier)] = 1.0
        perm_letter_probs.append(probs)
    p_tier = aggregate_allocation_probs(perm_letter_probs, perms)
    assert p_tier[true_tier] == 1.0 and elicit_score(p_tier) == 1.0
