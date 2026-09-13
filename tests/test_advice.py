"""Offline unit tests for S3 (no model / no GPU)."""

import yaml

from advice import (
    aggregate_allocation_probs, aggressiveness, build_user_message, cyclic_permutations,
    is_hedged, slot_options,
)
from gate import compute_gate


ALLOCS = ["80% bonds, 20% stocks", "60% bonds, 40% stocks",
          "30% bonds, 70% stocks", "10% bonds, 90% stocks"]
EQUITY = [0.20, 0.40, 0.70, 0.90]


# -- permutations ----------------------------------------------------------------------
def test_cyclic_permutations_full_coverage():
    perms = cyclic_permutations(4)
    assert len(perms) == 4
    # each allocation index appears in each slot exactly once across the 4 perms
    for slot in range(4):
        assert sorted(p[slot] for p in perms) == [0, 1, 2, 3]
    for alloc in range(4):
        assert sorted(p.index(alloc) for p in perms) == [0, 1, 2, 3]


def test_cyclic_permutations_partial_count():
    assert len(cyclic_permutations(4, n_rot=2)) == 2


def test_slot_options_reorders():
    assert slot_options(ALLOCS, [1, 2, 3, 0]) == [ALLOCS[1], ALLOCS[2], ALLOCS[3], ALLOCS[0]]


# -- aggregation cancels position --------------------------------------------------------
def test_aggregate_recovers_true_allocation_under_permutation():
    # Model "truly" prefers allocation index 2, regardless of which slot it sits in.
    perms = cyclic_permutations(4)
    true_pref = 2
    perm_letter_probs = []
    for perm in perms:
        slot_of_pref = perm.index(true_pref)         # where allocation 2 sits this perm
        probs = [0.0, 0.0, 0.0, 0.0]
        probs[slot_of_pref] = 1.0                    # all mass on that slot's letter
        perm_letter_probs.append(probs)
    p_alloc = aggregate_allocation_probs(perm_letter_probs, perms)
    assert p_alloc[true_pref] == 1.0 and sum(p_alloc) == 1.0


def test_aggregate_cancels_constant_position_bias():
    # A model that ALWAYS picks slot A (pure position bias) should yield a uniform
    # allocation distribution after cyclic aggregation -> aggressiveness = mean equity.
    perms = cyclic_permutations(4)
    perm_letter_probs = [[1.0, 0.0, 0.0, 0.0] for _ in perms]
    p_alloc = aggregate_allocation_probs(perm_letter_probs, perms)
    assert all(abs(p - 0.25) < 1e-9 for p in p_alloc)
    assert abs(aggressiveness(p_alloc, EQUITY) - sum(EQUITY) / 4) < 1e-9


# -- score math ------------------------------------------------------------------------
def test_aggressiveness_endpoints():
    assert aggressiveness([1, 0, 0, 0], EQUITY) == 0.20   # all mass on most conservative
    assert aggressiveness([0, 0, 0, 1], EQUITY) == 0.90   # all mass on most aggressive


# -- hedge logic -----------------------------------------------------------------------
def test_is_hedged():
    assert is_hedged(0.3, 1.0, 0.5) is True       # low letter mass
    assert is_hedged(0.9, 0.2, 0.5) is True       # usually argmaxes a non-letter
    assert is_hedged(0.9, 1.0, 0.5) is False      # confident on a letter


# -- prompt assembly -------------------------------------------------------------------
def test_build_user_message_has_options_and_no_leakage():
    msg = build_user_message("FRAMING", "I am a 70-year-old who panics in downturns.",
                             slot_options(ALLOCS, [3, 2, 1, 0]))
    assert "A) " + ALLOCS[3] in msg and "D) " + ALLOCS[0] in msg
    assert "panics in downturns" in msg
    assert "Respond with only the letter" in msg


def test_instruction_condition_prepends_nudge():
    msg = build_user_message("FRAMING", "narrative", slot_options(ALLOCS, [0, 1, 2, 3]),
                             integrate_instruction="INTEGRATE NUDGE")
    assert msg.startswith("INTEGRATE NUDGE")


# -- gate analysis (synthetic) ---------------------------------------------------------
def _cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _row(vid, vtype, tier, risk, aggr, cond="baseline", contra=False, pair_id=None, hedged=False):
    return {"vignette_id": vid, "profile_id": vid, "pair_id": pair_id, "vignette_type": vtype,
            "tier": tier, "risk_score": risk, "contradictory": contra, "condition": cond,
            "aggressiveness": aggr, "p_letters": 0.9, "hedged": hedged}


def test_gate_passes_on_clean_monotonic_signal():
    cfg = _cfg()
    # implicit twins with aggressiveness tracking risk_score -> high rho, no hedging
    rows = [_row(f"v{i}", "implicit", "moderate", risk=float(i), aggr=0.2 + 0.007 * i)
            for i in range(100)]
    rep = compute_gate(rows, cfg)
    assert rep["spearman"]["implicit"]["rho"] > 0.9
    assert rep["gate"]["PASS"] is True


def test_gate_fails_on_high_hedging():
    cfg = _cfg()
    rows = [_row(f"v{i}", "implicit", "moderate", risk=float(i), aggr=0.2 + 0.007 * i,
                 hedged=(i % 2 == 0)) for i in range(100)]   # 50% hedge
    rep = compute_gate(rows, cfg)
    assert rep["hedge_rate"] == 0.5
    assert rep["gate"]["hedge_pass"] is False and rep["gate"]["PASS"] is False


def test_gate_excludes_pairs_from_rho():
    cfg = _cfg()
    twins = [_row(f"t{i}", "implicit", "moderate", risk=float(i), aggr=0.2 + 0.007 * i)
             for i in range(50)]
    # pairs are extreme + anti-correlated; must NOT contaminate rho
    pairs = [_row(f"p{i}", "implicit", "conservative", risk=99.0, aggr=0.2, pair_id=f"pair_{i}")
             for i in range(50)]
    rep = compute_gate(twins + pairs, cfg)
    assert rep["n_twins"] == 50
    assert rep["spearman"]["implicit"]["rho"] > 0.9   # pairs excluded -> clean signal


def test_two_condition_paired_delta():
    cfg = _cfg()
    rows = []
    for i in range(20):
        vid = f"c{i}"
        rows.append(_row(vid, "implicit", "moderate", 50.0, 0.50, cond="baseline", contra=True))
        rows.append(_row(vid, "implicit", "moderate", 50.0, 0.40, cond="instruction", contra=True))
    rep = compute_gate(rows, cfg)
    assert rep["two_condition"]["n_pairs"] == 20
    assert abs(rep["two_condition"]["mean_delta"] - (-0.10)) < 1e-9
