"""Offline tests for the roundtrip scorer (no API calls)."""

import pytest
import yaml

from roundtrip import (BAND_FIELDS, ENUM_FIELDS, FIELDS, build_extract_prompt,
                       field_options, parse_reply, score, weakest_levels)


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open("config.yaml"))


@pytest.fixture(scope="module")
def options(cfg):
    return field_options(cfg)[0]


def _rows(truths):
    return [{"_truth": t} for t in truths]


def _perfect(options):
    return {f: options[f][1] for f in FIELDS}


def test_covers_every_prose_rendered_rubric_field(cfg):
    """Only `age` may be missing: it renders as a literal number and is diffed instead."""
    rubric = set(cfg["rubric"]["fields"])
    assert set(FIELDS) | {"age"} == rubric
    assert not set(ENUM_FIELDS) & set(BAND_FIELDS)


def test_options_come_from_the_rendering_tables(cfg, options):
    """Options must match the templates, or the reader is scored on the wrong vocabulary."""
    import templates as T
    for f in ENUM_FIELDS:
        assert options[f] == list(cfg["rubric"]["fields"][f]["levels"])
    for f in BAND_FIELDS:
        assert len(options[f]) == len(T.BAND_PHRASINGS[f])


def test_band_values_map_into_their_band(cfg):
    """A raw field value has to resolve to the label the reader can choose."""
    opts, to_label = field_options(cfg)
    assert to_label["dependents"][0] == opts["dependents"][0]
    assert to_label["horizon_years"][1] == opts["horizon_years"][0]
    assert to_label["horizon_years"][35] == opts["horizon_years"][-1]


def test_perfect_predictions_score_one(options):
    truth = _perfect(options)
    res = score(_rows([truth]), [dict(truth)], options)
    for f in FIELDS:
        assert res[f]["accuracy"] == 1.0
        assert res[f]["flip_rate"] == 0.0


def test_one_step_error_is_not_a_flip(options):
    f = "past_drawdown_reaction"
    truth = _perfect(options)
    pred = dict(truth, **{f: options[f][2]})          # one step along the scale
    res = score(_rows([truth]), [pred], options)
    assert res[f]["accuracy"] == 0.0
    assert res[f]["mean_ordinal_distance"] == 1.0
    assert res[f]["flip_rate"] == 0.0


def test_opposite_end_counts_as_a_flip(options):
    f = "past_drawdown_reaction"
    truth = dict(_perfect(options), **{f: options[f][0]})
    pred = dict(truth, **{f: options[f][-1]})
    res = score(_rows([truth]), [pred], options)
    assert res[f]["flip_rate"] == 1.0
    assert res[f]["mean_ordinal_distance"] == len(options[f]) - 1


def test_per_level_breakdown_exposes_a_level_the_field_average_hides(options):
    """The whole point: one dead level must not be laundered by the other levels.

    This is the precarious/variable collapse in miniature -- the field average looks
    survivable while one level is never recovered at all.
    """
    f = "income_stability"
    low, mid = options[f][0], options[f][1]
    rows, preds = [], []
    for _ in range(30):                       # every `low` misread as `mid`
        rows.append({"_truth": dict(_perfect(options), **{f: low})})
        preds.append(dict(_perfect(options), **{f: mid}))
    for _ in range(70):                       # every other level fine
        rows.append({"_truth": dict(_perfect(options), **{f: mid})})
        preds.append(dict(_perfect(options), **{f: mid}))
    res = score(rows, preds, options)
    assert res[f]["accuracy"] == 0.70         # field average looks tolerable
    assert res[f]["per_level"][low]["accuracy"] == 0.0      # the level is dead
    assert res[f]["per_level"][mid]["accuracy"] == 1.0
    assert res[f]["per_level"][low]["mistaken_for"][0][0] == mid

    flagged = weakest_levels({"splits": {"implicit": res}}, threshold=0.85)
    assert any(x[1] == f and x[2] == low for x in flagged)


def test_weakest_levels_ignores_tiny_samples(options):
    """A level seen twice should not be reported as broken."""
    f = "stated_goal"
    rows = [{"_truth": dict(_perfect(options), **{f: options[f][0]})}]
    preds = [dict(_perfect(options), **{f: options[f][1]})]
    res = score(rows, preds, options)
    assert weakest_levels({"splits": {"implicit": res}}, threshold=0.85) == []


def test_unparseable_and_out_of_vocabulary_never_count_as_correct(options):
    truth = _perfect(options)
    res = score(_rows([truth, truth]),
                [None, dict(truth, investing_experience="expert")], options)
    assert res["past_drawdown_reaction"]["n"] == 1
    assert res["past_drawdown_reaction"]["unreadable"] == 1
    assert res["investing_experience"]["n"] == 0
    assert res["investing_experience"]["unreadable"] == 2


def test_parse_reply_survives_fences_and_prose():
    assert parse_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_reply('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert parse_reply("not json") is None
    assert parse_reply("[1, 2]") is None
    assert parse_reply(None) is None


def test_prompt_lists_every_allowed_option(options):
    p = build_extract_prompt("I sold everything.", options)
    for f in FIELDS:
        assert f in p
        for lv in options[f]:
            assert str(lv) in p
