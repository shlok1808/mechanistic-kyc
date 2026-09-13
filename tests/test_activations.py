"""Offline unit tests for S4 (no model / no GPU).

Focus: the position-indexing logic, which the design doc flags as S4's main failure mode.
"""

import pytest

from activations import (
    label_row, model_tag, narrative_end_char, narrative_start_char, parse_layers,
    token_covering_char,
)


# -- token_covering_char ---------------------------------------------------------------
def test_token_covering_char_basic():
    # tokens spanning "abcdef": [ab][cd][ef], plus a leading special token (0,0)
    offsets = [(0, 0), (0, 2), (2, 4), (4, 6)]
    assert token_covering_char(offsets, 0) == 1   # 'a'
    assert token_covering_char(offsets, 3) == 2   # 'd'
    assert token_covering_char(offsets, 5) == 3   # 'f' (last char)


def test_token_covering_char_skips_zero_width():
    offsets = [(0, 0), (0, 3), (0, 0), (3, 5)]    # specials interleaved
    assert token_covering_char(offsets, 4) == 3


def test_token_covering_char_out_of_range_returns_none():
    offsets = [(0, 2), (2, 4)]
    assert token_covering_char(offsets, 9) is None


# -- narrative_end_char ----------------------------------------------------------------
def test_narrative_end_char_points_at_last_char():
    narrative = "I bought more when the market fell."
    prompt = f"<start>System framing.\nClient description:\n{narrative}\nOptions: A) x"
    end = narrative_end_char(prompt, narrative)
    assert prompt[end] == "."                       # last char of the narrative
    assert prompt[end - len(narrative) + 1:end + 1] == narrative


def test_narrative_end_char_takes_last_occurrence():
    narrative = "hello world"
    prompt = f"{narrative} ... and again {narrative}"
    end = narrative_end_char(prompt, narrative)
    assert end == len(prompt) - 1                    # the *second* occurrence


def test_narrative_end_char_missing_raises():
    with pytest.raises(ValueError):
        narrative_end_char("no narrative here", "absent text")


# -- narrative_start_char (profile_mean span start) ------------------------------------
def test_narrative_start_char_points_at_first_char():
    narrative = "When the market fell I bought more."
    prompt = f"<start>framing\nClient description:\n{narrative}\nOptions: A) x"
    start = narrative_start_char(prompt, narrative)
    assert prompt[start] == "W"
    assert prompt[start:start + len(narrative)] == narrative


def test_narrative_start_char_takes_last_occurrence():
    narrative = "hello world"
    prompt = f"{narrative} ... and again {narrative}"
    assert narrative_start_char(prompt, narrative) == prompt.rindex(narrative)


def test_span_start_le_end_and_inside_narrative():
    narrative = "When the market dropped I invested more aggressively over time."
    prompt = f"prefix\nClient description:\n{narrative}\nPortfolio options:\nA) x"
    c0, c1 = narrative_start_char(prompt, narrative), narrative_end_char(prompt, narrative)
    offsets = [(i, min(i + 4, len(prompt))) for i in range(0, len(prompt), 4)]   # fake 4-char tokens
    t0, t1 = token_covering_char(offsets, c0), token_covering_char(offsets, c1)
    assert t0 is not None and t1 is not None and t0 <= t1
    # the span tokens lie within the narrative's char extent (not the framing or options)
    assert offsets[t0][1] > c0 and offsets[t1][0] <= c1


# -- parse_layers ----------------------------------------------------------------------
def test_parse_layers_all():
    assert parse_layers("all", 5) == [0, 1, 2, 3, 4]


def test_parse_layers_list_and_str():
    assert parse_layers([20, 25, 31], 42) == [20, 25, 31]
    assert parse_layers("10,15,20", 42) == [10, 15, 20]


# -- combined: a token covering the narrative end falls inside the narrative span -------
def test_p1_char_lands_inside_narrative():
    narrative = "When the market dropped I invested more."
    prompt = f"prefix\nClient description:\n{narrative}\nPortfolio options:\nA) x"
    char_end = narrative_end_char(prompt, narrative)
    # simulate 4-char tokens over the whole prompt
    offsets = [(i, min(i + 4, len(prompt))) for i in range(0, len(prompt), 4)]
    tok_idx = token_covering_char(offsets, char_end)
    s, e = offsets[tok_idx]
    assert s <= char_end < e
    # the covering token's span overlaps the narrative, not the trailing options
    assert s < char_end + 1


# -- helpers ---------------------------------------------------------------------------
def test_model_tag_sanitizes():
    assert model_tag("google/gemma-2-9b-it") == "gemma-2-9b-it"
    assert model_tag("meta-llama/Llama-3.1-8B-Instruct") == "Llama-3.1-8B-Instruct"


def test_label_row_schema():
    v = {"vignette_id": "v_p1_implicit", "profile_id": "p1", "pair_id": None,
         "tier": "aggressive", "risk_score": 74.1, "vignette_type": "implicit",
         "contradictory": 0, "text": "...", "template_id": "t"}
    row = label_row(v)
    assert row["contradictory"] is False
    # split keys + blended label ...
    assert {"vignette_id", "profile_id", "pair_id", "tier", "risk_score",
            "vignette_type", "contradictory"} <= set(row)
    # ... the factor targets a blended tier cannot separate ...
    assert {"willingness", "capacity", "goals", "factor_cell", "conflicted"} <= set(row)
    # ... which field a counterfactual pair varies, for attributable patching ...
    assert {"pair_field", "pair_factor"} <= set(row)
    # ... and the surface-form controls.
    assert {"template_id", "name", "n_chars"} <= set(row)
    # pair_id is legitimately None on twins, but the string columns must never be:
    # np.savez stores them as object arrays and None round-trips badly.
    assert all(row[k] is not None
               for k in ("pair_field", "pair_factor", "template_id", "name", "factor_cell"))
