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


# -- checkpoint / resume ---------------------------------------------------------------
def test_unit_key_changes_with_model_and_prompt():
    from advice import unit_key
    assert unit_key("m", "p") == unit_key("m", "p")
    assert unit_key("m", "p") != unit_key("m", "p ")
    assert unit_key("m", "p") != unit_key("m2", "p")


def test_load_checkpoint_skips_torn_last_line(tmp_path):
    import json
    from advice import load_checkpoint
    p = tmp_path / "ckpt.jsonl"
    p.write_text(json.dumps({"k": "a", "r": [[0.25] * 4, 0.9, True]}) + "\n" + '{"k": "b", "r": [[0.2')
    assert load_checkpoint(p) == {"a": ([0.25] * 4, 0.9, True)}
    assert load_checkpoint(tmp_path / "missing.jsonl") == {}


def test_run_resumes_after_crash_without_rescoring(tmp_path, monkeypatch):
    """Crash partway, re-run: only unscored prompts hit the model, output matches a clean run."""
    import json
    from argparse import Namespace
    import advice

    cfg = yaml.safe_load(open("config.yaml"))
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    cfg["paths"]["vignettes_dir"] = str(tmp_path / "vig")
    cfg["s3"]["batch_size"] = 2
    (tmp_path / "vig").mkdir()
    vigs = [{"vignette_id": f"v{i}", "profile_id": f"p{i}", "vignette_type": "implicit",
             "tier": cfg["data"]["tiers"][i % len(cfg["data"]["tiers"])], "risk_score": 0.5,
             "contradictory": i == 0, "text": f"client {i} " * (i + 1)} for i in range(5)]
    (tmp_path / "vig" / "implicit.jsonl").write_text("".join(json.dumps(v) + "\n" for v in vigs))

    def fake_score(tok, model, prompts, letter_ids):
        calls.extend(prompts)
        if crash_after is not None and len(calls) > crash_after:
            raise RuntimeError("simulated crash")
        n = len(letter_ids)
        lp = [[(len(p) % 7 + k + 1) / 10 for k in range(n)] for p in prompts]
        lp = [[x / sum(r) for x in r] for r in lp]
        return lp, [0.95] * len(prompts), [True] * len(prompts)

    monkeypatch.setattr(advice, "load_model", lambda *a, **k: (None, None))
    monkeypatch.setattr(advice, "make_prompt", lambda tok, msg: msg)
    monkeypatch.setattr(advice, "resolve_letter_token_ids", lambda tok, p, n: list(range(n)))
    monkeypatch.setattr(advice, "score_prompts", fake_score)
    args = Namespace(dry_run=False, model="fake/model", device="cpu", batch_size=None, shard=None)
    n_prompts = 6 * cfg["s3"]["n_permutations"]      # 5 baseline + 1 instruction row

    calls, crash_after = [], 8
    try:
        advice.run(cfg, args)
        assert False, "expected simulated crash"
    except RuntimeError:
        pass
    scored_before_crash = 8

    calls, crash_after = [], None
    advice.run(cfg, args)
    assert len(calls) == n_prompts - scored_before_crash
    resumed = open(tmp_path / "results" / "model" / "advice.jsonl").read()

    # clean run in a fresh results dir must give identical rows
    cfg["paths"]["results_dir"] = str(tmp_path / "clean")
    calls = []
    advice.run(cfg, args)
    assert len(calls) == n_prompts
    assert open(tmp_path / "clean" / "model" / "advice.jsonl").read() == resumed


def test_parse_shard():
    from advice import parse_shard
    import pytest
    assert parse_shard(None) == (0, 1)
    assert parse_shard("2/8") == (2, 8)
    for bad in ("8/8", "-1/4", "0/0"):
        with pytest.raises(ValueError):
            parse_shard(bad)


def test_sharded_run_merges_to_same_output(tmp_path, monkeypatch):
    """3 shards then an unsharded merge: every prompt scored exactly once, output identical
    to a single-process run, and dry-run checkpoints are never picked up."""
    import json
    from argparse import Namespace
    import advice

    cfg = yaml.safe_load(open("config.yaml"))
    cfg["paths"]["vignettes_dir"] = str(tmp_path / "vig")
    (tmp_path / "vig").mkdir()
    vigs = [{"vignette_id": f"v{i}", "profile_id": f"p{i}", "vignette_type": "implicit",
             "tier": cfg["data"]["tiers"][i % len(cfg["data"]["tiers"])], "risk_score": 0.5,
             "contradictory": i < 2, "text": f"client {i} " * (i + 1)} for i in range(7)]
    (tmp_path / "vig" / "implicit.jsonl").write_text("".join(json.dumps(v) + "\n" for v in vigs))
    calls = []

    def fake_score(tok, model, prompts, letter_ids):
        calls.extend(prompts)
        lp = [[(len(p) % 5 + k + 1) for k in range(len(letter_ids))] for p in prompts]
        return [[x / sum(r) for x in r] for r in lp], [0.9] * len(prompts), [True] * len(prompts)

    monkeypatch.setattr(advice, "load_model", lambda *a, **k: (None, None))
    monkeypatch.setattr(advice, "make_prompt", lambda tok, msg: msg)
    monkeypatch.setattr(advice, "resolve_letter_token_ids", lambda tok, p, n: list(range(n)))
    monkeypatch.setattr(advice, "score_prompts", fake_score)
    mk = lambda shard: Namespace(dry_run=False, model="fake/model", device="cpu",
                                 batch_size=3, shard=shard)

    cfg["paths"]["results_dir"] = str(tmp_path / "sharded")
    (tmp_path / "sharded" / "model").mkdir(parents=True)
    (tmp_path / "sharded" / "model" / "advice_checkpoint_dryrun.jsonl").write_text("garbage\n")
    for i in range(3):
        advice.run(cfg, mk(f"{i}/3"))
    n_prompts = len(calls)
    assert len(set(calls)) == n_prompts == 9 * cfg["s3"]["n_permutations"]
    calls.clear()
    advice.run(cfg, mk(None))
    assert calls == []                                  # merge scores nothing
    sharded = open(tmp_path / "sharded" / "model" / "advice.jsonl").read()

    cfg["paths"]["results_dir"] = str(tmp_path / "single")
    advice.run(cfg, mk(None))
    assert open(tmp_path / "single" / "model" / "advice.jsonl").read() == sharded
