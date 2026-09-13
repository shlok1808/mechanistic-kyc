"""Offline unit tests for S2 (no API calls)."""

import json
import random
import re
from collections import Counter

import pytest
import yaml

from rubric import field_subscore, is_contradictory, load_rubric, risk_tier, risk_score
from edge_profiles import assign_decoys, build_edge_rubric_fields
from qc import tenure_incoherences
from templates import (
    BAND_PHRASINGS, ENUM_PHRASINGS, NET_WORTH_PHRASINGS, TEMPLATE_IDS,
    build_banned_regex, find_banned, pick_template_id, render,
)
from vignettes import build_pair_jobs, build_twin_jobs, finalize


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def banned(cfg):
    return cfg["vignette"]["banned_lexicon"], build_banned_regex(cfg["vignette"]["banned_lexicon"])


SAMPLE_PROFILE = {
    "profile_id": "p99999", "past_drawdown_reaction": "bought_more", "horizon_years": 30,
    "investing_experience": "extensive", "income_stability": "stable",
    "stated_goal": "maximum_growth", "emergency_fund_months": 20, "dependents": 0, "age": 27,
    "net_worth_band": "over_2m", "occupation": "software engineer", "hobbies": "hiking",
    "city": "Austin", "name": "Wei Zhang", "gender_hint": "neutral",
    "risk_score": 95.0, "tier": "aggressive",
}


# -- banned-lexicon regex --------------------------------------------------------------
def test_regex_catches_inflections(banned):
    _, rx = banned
    for bad in ["risk", "Risky", "riskier", "AGGRESSIVE", "aggressively",
                "conservative", "conservatively", "volatility", "safely"]:
        assert find_banned(f"I am {bad} about it.", rx), bad


def test_regex_ignores_clean_text(banned):
    _, rx = banned
    assert find_banned("I moved my money into cash when the market fell.", rx) == []


def test_moderate_is_not_banned(banned):
    _, rx = banned
    assert find_banned("I earn a moderate income.", rx) == []


# -- exhaustive implicit cleanliness (the core guarantee) ------------------------------
def test_no_implicit_phrasing_contains_banned_term(banned):
    _, rx = banned
    cells = []
    for field, byval in ENUM_PHRASINGS.items():
        for val, modes in byval.items():
            cells += [(f"{field}={val}", s) for s in modes["implicit"]]
    for field, bands in BAND_PHRASINGS.items():
        for lo, hi, modes in bands:
            cells += [(f"{field}[{lo},{hi}]", s) for s in modes["implicit"]]
    for val, phrs in NET_WORTH_PHRASINGS.items():
        cells += [(f"net_worth={val}", s) for s in phrs]
    bad = [(label, find_banned(s, rx)) for label, s in cells if find_banned(s, rx)]
    assert bad == [], f"banned terms in implicit phrasings: {bad}"


def test_rendered_implicit_profile_is_clean(banned):
    _, rx = banned
    for tid in TEMPLATE_IDS[:8]:
        text = render(SAMPLE_PROFILE, tid, "implicit", random.Random(0))
        assert find_banned(text, rx) == []


def test_extensive_experience_phrasings_are_duration_free():
    # Guards the age/experience impossibility: extensive experience must not assert a
    # tenure ("two decades", "for N years") that conflicts with young sampled ages.
    for s in ENUM_PHRASINGS["investing_experience"]["extensive"]["implicit"]:
        assert not re.search(r"\b(decade|year)", s, re.I), s


def test_explicit_render_contains_facts():
    text = render(SAMPLE_PROFILE, TEMPLATE_IDS[0], "explicit", random.Random(0))
    assert "Wei Zhang" in text and "Austin" in text and "27" in text


# -- template assignment ---------------------------------------------------------------
def test_template_id_is_deterministic_and_shared_by_twins():
    a = pick_template_id("p00042", seed=42)
    b = pick_template_id("p00042", seed=42)
    assert a == b and a in TEMPLATE_IDS


# -- twin job construction -------------------------------------------------------------
def test_twins_share_template_differ_in_type(cfg):
    jobs = build_twin_jobs([SAMPLE_PROFILE], seed=42, rubric=cfg["rubric"])
    assert len(jobs) == 2
    assert jobs[0]["template_id"] == jobs[1]["template_id"]
    assert {j["vignette_type"] for j in jobs} == {"explicit", "implicit"}
    assert all(j["tier"] == "aggressive" for j in jobs)


# -- pair construction -----------------------------------------------------------------
def _field_of(job, field, rubric):
    """One rubric field's value for a pair side (jobs carry the raw dict)."""
    return job["rubric_fields"][field]


def test_pairs_differ_in_exactly_one_rubric_field(cfg):
    """The whole point of a counterfactual pair.

    The June pilot drew both sides independently, so a "matched pair" differed on a
    median of 7 of 8 rubric fields and never on fewer than 4. Patching across that
    cannot attribute an effect to any one factor.
    """
    rubric = cfg["rubric"]
    jobs = build_pair_jobs(rubric, n_pairs=40, render_tier="implicit", seed=42)
    by_pair = {}
    for j in jobs:
        by_pair.setdefault(j["pair_id"], []).append(j)
    assert len(by_pair) == 40
    for pid, sides in by_pair.items():
        assert len(sides) == 2
        a, b = sorted(sides, key=lambda j: j["pair_side"])
        assert a["template_id"] == b["template_id"]          # same template
        assert a["name"] == b["name"]                        # same person
        assert a["pair_field"] == b["pair_field"]            # same field under test
        field = a["pair_field"]
        differing = [f for f in rubric["fields"]
                     if _field_of(a, f, rubric) != _field_of(b, f, rubric)]
        assert differing == [field], f"{pid}: expected only {field}, got {differing}"


def test_pair_field_coverage_is_even_across_factors(cfg):
    """Every rubric field gets tested equally, so no factor is under-powered."""
    rubric = cfg["rubric"]
    jobs = build_pair_jobs(rubric, n_pairs=len(rubric["fields"]) * 4,
                           render_tier="implicit", seed=7)
    counts = Counter(j["pair_field"] for j in jobs)
    assert set(counts) == set(rubric["fields"])
    assert len(set(counts.values())) == 1          # perfectly even


def test_pair_sides_straddle_the_field_scale(cfg):
    """lo and hi must sit at opposite ends, or the contrast has no strength."""
    rubric = cfg["rubric"]
    jobs = build_pair_jobs(rubric, n_pairs=24, render_tier="implicit", seed=3)
    by_pair = {}
    for j in jobs:
        by_pair.setdefault(j["pair_id"], []).append(j)
    for sides in by_pair.values():
        lo = next(j for j in sides if j["pair_side"] == "lo")
        hi = next(j for j in sides if j["pair_side"] == "hi")
        assert hi["risk_score"] > lo["risk_score"]
        spec = rubric["fields"][lo["pair_field"]]
        assert field_subscore(spec, _field_of(hi, hi["pair_field"], rubric)) == 1.0
        assert field_subscore(spec, _field_of(lo, lo["pair_field"], rubric)) == 0.0


def test_pair_factor_matches_the_ontology(cfg):
    """pair_factor must name the factor that owns pair_field."""
    rubric = cfg["rubric"]
    for j in build_pair_jobs(rubric, n_pairs=16, render_tier="implicit", seed=11):
        assert j["pair_field"] in rubric["factors"][j["pair_factor"]]


# -- finalize / schema -----------------------------------------------------------------
def test_finalize_schema_and_banned_field(banned, cfg):
    _, rx = banned
    job = build_twin_jobs([SAMPLE_PROFILE], seed=42, rubric=cfg["rubric"])[1]  # implicit
    row = finalize(job, "I moved everything into cash.", "gpt-4o-mini", rx)
    required = {"vignette_id", "profile_id", "pair_id", "tier", "risk_score",
                "vignette_type", "template_id", "contradictory", "paraphrase_model",
                "banned_terms_found", "text",
                # factor targets: a blended tier cannot say which concept was learned
                "willingness", "capacity", "goals", "factor_cell", "conflicted",
                # confound controls, so surface form can be ruled out at probe time
                "name", "n_chars"}
    assert required <= set(row)
    assert "template_text" not in row          # intermediate, must be dropped
    assert "rubric_fields" not in row          # intermediate, must be dropped
    assert row["banned_terms_found"] == []
    assert row["n_chars"] == len(row["text"])


def test_factor_columns_agree_with_the_rubric(cfg):
    """The columns written onto a row must equal a fresh computation from the profile."""
    from rubric import factor_scores, is_conflicted
    job = build_twin_jobs([SAMPLE_PROFILE], seed=42, rubric=cfg["rubric"])[0]
    expected = factor_scores(SAMPLE_PROFILE, cfg["rubric"])
    for name, value in expected.items():
        assert job[name] == pytest.approx(round(value, 4))
    assert job["conflicted"] == is_conflicted(SAMPLE_PROFILE, cfg["rubric"])


def test_finalize_flags_leak(banned, cfg):
    _, rx = banned
    job = build_twin_jobs([SAMPLE_PROFILE], seed=42, rubric=cfg["rubric"])[1]
    row = finalize(job, "I have a high risk tolerance.", "gpt-4o-mini", rx)
    assert "risk" in [h.lower() for h in row["banned_terms_found"]]


# -- determinism -----------------------------------------------------------------------
def test_render_is_deterministic():
    a = render(SAMPLE_PROFILE, TEMPLATE_IDS[3], "implicit", random.Random(123))
    b = render(SAMPLE_PROFILE, TEMPLATE_IDS[3], "implicit", random.Random(123))
    assert a == b


# -- contradiction detection -----------------------------------------------------------
def test_contradiction_flags_opposite_extremes(cfg):
    rub = cfg["rubric"]
    base = dict(SAMPLE_PROFILE)
    base.update({"stated_goal": "maximum_growth", "past_drawdown_reaction": "sold_everything"})
    assert is_contradictory(base, rub) is True
    base.update({"stated_goal": "capital_preservation", "past_drawdown_reaction": "bought_more"})
    assert is_contradictory(base, rub) is True


def test_contradiction_ignores_aligned_and_mild(cfg):
    rub = cfg["rubric"]
    aligned = dict(SAMPLE_PROFILE)
    aligned.update({"stated_goal": "maximum_growth", "past_drawdown_reaction": "bought_more"})
    assert is_contradictory(aligned, rub) is False
    mild = dict(SAMPLE_PROFILE)
    mild.update({"stated_goal": "maximum_growth", "past_drawdown_reaction": "reduced"})
    assert is_contradictory(mild, rub) is False  # reduced (a=0.33) is not extreme enough


def test_contradictory_field_propagates_to_rows(cfg):
    contra = dict(SAMPLE_PROFILE)
    contra.update({"stated_goal": "capital_preservation", "past_drawdown_reaction": "bought_more"})
    jobs = build_twin_jobs([contra], seed=42, rubric=cfg["rubric"])
    assert all(j["contradictory"] is True for j in jobs)


# -- edge generator --------------------------------------------------------------------
def test_edge_generator_covers_buckets_and_decoys(cfg):
    rng = random.Random(cfg["seed"])
    items = build_edge_rubric_fields(cfg["rubric"], rng)
    profiles = assign_decoys(items, rng)
    for p in profiles:
        s = risk_score(p, cfg["rubric"])
        p["risk_score"], p["tier"] = s, risk_tier(s, cfg["rubric"])
    buckets = {p["edge_bucket"] for p in profiles}
    assert {"borderline", "contradiction", "extreme_demo",
            "anchor_conservative", "anchor_aggressive"} <= buckets
    # full decoy coverage (all 10 of each)
    for k in ("occupation", "city", "hobbies"):
        assert len({p[k] for p in profiles}) == 10
    # anchors land at the extremes; contradictions are rubric-flagged
    anchors_c = [p for p in profiles if p["edge_bucket"] == "anchor_conservative"]
    anchors_a = [p for p in profiles if p["edge_bucket"] == "anchor_aggressive"]
    assert all(p["tier"] == "conservative" for p in anchors_c)
    assert all(p["tier"] == "aggressive" for p in anchors_a)
    assert all(is_contradictory(p, cfg["rubric"])
               for p in profiles if p["edge_bucket"] == "contradiction")


# -- QC age/experience coherence -------------------------------------------------------
def test_tenure_incoherence_flags_young_long_tenure():
    profiles = {"young": {"profile_id": "young", "age": 23},
                "old": {"profile_id": "old", "age": 70}}
    rows = [
        {"vignette_id": "a", "profile_id": "young",
         "text": "I'm 23 and I've been actively investing for nearly two decades now."},
        {"vignette_id": "b", "profile_id": "young",
         "text": "I'm 23 and I've been through several market cycles."},   # clean
        {"vignette_id": "c", "profile_id": "old",
         "text": "I'm 70 and I've been investing for two decades."},        # plausible at 70
    ]
    hits = tenure_incoherences(rows, profiles)
    assert [h["vignette_id"] for h in hits] == ["a"]


# -- Batch API (offline; no live OpenAI calls) -----------------------------------------
from paraphrase import ParaphraseClient  # noqa: E402
from templates import build_banned_regex as _bbr  # noqa: E402


def _client(tmp_path, banned):
    words, _ = banned
    return ParaphraseClient(backend="openai", model="gpt-4o-mini", banned_words=words,
                            banned_regex=_bbr(words), cache_dir=tmp_path)


def test_build_batch_skips_cached_and_dedups(tmp_path, banned):
    c = _client(tmp_path, banned)
    # pre-cache one job so it is excluded from the batch
    c._store("explicit", "already done", "cached out", "gpt-4o-mini")
    jobs = [("already done", "explicit"),
            ("fresh one", "implicit"),
            ("fresh one", "implicit")]            # duplicate -> single request
    lines, sidecar = c.build_batch(jobs)
    assert len(lines) == 1
    line = lines[0]
    assert line["custom_id"] == c._cache_id("implicit", "fresh one")
    assert line["url"] == "/v1/chat/completions"
    # implicit prompt carries the banned-word instruction
    assert "do NOT use" in line["body"]["messages"][1]["content"]
    assert sidecar[line["custom_id"]] == {"mode": "implicit", "source": "fresh one"}


class _Fake:
    """Minimal stand-in for the OpenAI client used by collect_batch."""
    class _B:
        status, output_file_id = "completed", "out123"
    def __init__(self, text):
        self._text = text
        self.batches = type("X", (), {"retrieve": lambda s, b: _Fake._B()})()
        self.files = type("X", (), {"content": lambda s, f: type("R", (), {"text": text})()})()


def test_collect_batch_caches_clean_and_falls_back_on_leak(tmp_path, banned):
    c = _client(tmp_path, banned)
    jobs = [("narrative a", "implicit"), ("narrative b", "implicit")]
    _, sidecar = c.build_batch(jobs)
    ids = list(sidecar)
    clean_id = next(i for i in ids if sidecar[i]["source"] == "narrative a")
    leak_id = next(i for i in ids if sidecar[i]["source"] == "narrative b")
    out = "\n".join(json.dumps({
        "custom_id": cid,
        "response": {"status_code": 200, "body": {"choices": [{"message": {"content": content}}]}},
    }) for cid, content in [(clean_id, "I moved everything into cash."),
                            (leak_id, "I have a high risk tolerance.")])  # leaked
    c._client = _Fake(out)  # bypass _ensure_client
    summary = c.collect_batch("batch_x", sidecar)
    assert summary["cached"] == 2 and summary["fallbacks"] == 1
    assert c.paraphrase_one("narrative a", "implicit") == ("I moved everything into cash.", "gpt-4o-mini")
    assert c.paraphrase_one("narrative b", "implicit") == ("narrative b", "template-only")


def test_pair_narratives_differ_in_exactly_one_sentence(cfg):
    """The token-level guarantee activation patching actually depends on.

    Field-level isolation is not enough. templates.render draws phrasing from one rng
    stream, so a field whose wording consumes a different number of draws used to
    shift every sentence after it -- flipping one field silently reworded unrelated
    sentences. build_pair_jobs passes a stable per-field seed to stop that.
    """
    rubric = cfg["rubric"]
    jobs = build_pair_jobs(rubric, n_pairs=48, render_tier="implicit", seed=42)
    by_pair = {}
    for j in jobs:
        by_pair.setdefault(j["pair_id"], []).append(j)
    for pid, sides in by_pair.items():
        lo = next(j for j in sides if j["pair_side"] == "lo")["template_text"]
        hi = next(j for j in sides if j["pair_side"] == "hi")["template_text"]
        a, b = lo.split(". "), hi.split(". ")
        assert len(a) == len(b), f"{pid}: sentence count changed"
        differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert len(differing) == 1, f"{pid}: {len(differing)} sentences differ, expected 1"
