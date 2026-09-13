"""Ground-truth client rubric: factor scoring, tiering, and profile sampling.

Every rubric field belongs to exactly one FACTOR (see config.yaml `rubric.factors`):

  willingness  how comfortable the client is with volatility and loss
  capacity     whether they can objectively absorb a loss
  goals        what the money is for

Each factor scores independently in [0,1]. The blended `risk_score` / `tier` still
exist as derived convenience labels, but they are no longer the probe target: a
single blended number cannot say WHICH concept the model learned, and suitability
law (FINRA 2111) turns on willingness and capacity being separate things.

A profile is CONFLICTED when willingness and capacity land in opposite bins. Those
are the cases the study is about, so profiles are sampled to fill the willingness x
capacity grid evenly rather than to balance the blended tier.

See config.yaml's `rubric:` block for field weights, factor membership, and bins.
"""

import yaml

TIERS = ["conservative", "moderate", "aggressive"]

DECOY_POOLS = {
    "net_worth_band": ["under_100k", "100k_to_500k", "500k_to_2m", "over_2m"],
    "occupation": [
        "high school teacher",
        "registered nurse",
        "software engineer",
        "small business owner",
        "graphic designer",
        "electrician",
        "accountant",
        "marketing manager",
        "warehouse supervisor",
        "freelance writer",
    ],
    "hobbies": [
        "hiking",
        "cooking",
        "woodworking",
        "gardening",
        "playing guitar",
        "running",
        "painting",
        "board games",
        "photography",
        "fishing",
    ],
    "city": [
        "Columbus",
        "Denver",
        "Austin",
        "Pittsburgh",
        "Sacramento",
        "Tampa",
        "Minneapolis",
        "Charlotte",
        "Albuquerque",
        "Portland",
    ],
}

NAMES = [
    ("Maria Lopez", "female"),
    ("James Carter", "male"),
    ("Wei Zhang", "neutral"),
    ("Aisha Khan", "female"),
    ("David Kim", "male"),
    ("Sofia Rossi", "female"),
    ("Marcus Johnson", "male"),
    ("Priya Patel", "female"),
    ("Liam O'Brien", "male"),
    ("Emma Nguyen", "female"),
    ("Noah Thompson", "male"),
    ("Olivia Brooks", "female"),
]


def load_rubric(config_path="config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["rubric"]


def field_subscore(field_spec, value):
    """Map a raw field value to a_i in [0,1] (0 = supports lower risk, 1 = higher)."""
    if "levels" in field_spec:
        levels = field_spec["levels"]
        return levels.index(value) / (len(levels) - 1)
    lo, hi = field_spec["range"]
    a = (value - lo) / (hi - lo)
    return 1 - a if field_spec.get("invert") else a


def risk_score(profile, rubric):
    """Continuous risk score s in [0,100] = 100 * sum(weight_i * a_i)."""
    return 100 * sum(
        spec["weight"] * field_subscore(spec, profile[field])
        for field, spec in rubric["fields"].items()
    )


def risk_tier(score, rubric):
    cutoffs = rubric["tier_cutoffs"]
    if score < cutoffs["conservative_max"]:
        return "conservative"
    if score <= cutoffs["moderate_max"]:
        return "moderate"
    return "aggressive"


BINS = ("low", "mid", "high")


def factor_fields(rubric, factor):
    """Field names belonging to `factor`, in config order."""
    return list(rubric["factors"][factor])


def factor_score(profile, rubric, factor):
    """Factor score in [0,1]: the factor's fields, reweighted within the factor.

    Weights are renormalised so each factor spans the full [0,1] range regardless of
    how much of the blended score its fields happen to carry.
    """
    fields = factor_fields(rubric, factor)
    total = sum(rubric["fields"][f]["weight"] for f in fields)
    return sum(
        rubric["fields"][f]["weight"] * field_subscore(rubric["fields"][f], profile[f])
        for f in fields
    ) / total


def factor_scores(profile, rubric):
    """{factor: score in [0,1]} for every factor in the ontology."""
    return {f: factor_score(profile, rubric, f) for f in rubric["factors"]}


def factor_bin(value, bins=BINS):
    """Cut a [0,1] factor score into evenly-spaced ordinal bins."""
    idx = min(int(value * len(bins)), len(bins) - 1)
    return bins[idx]


def factor_cell(profile, rubric):
    """(willingness_bin, capacity_bin) -- the cell profiles are balanced over."""
    return (factor_bin(factor_score(profile, rubric, "willingness")),
            factor_bin(factor_score(profile, rubric, "capacity")))


def is_conflicted(profile, rubric):
    """True when willingness and capacity point opposite ways (the FINRA 2111 case).

    Distinct from `is_contradictory`, which compares what the client SAYS they want
    (stated_goal) against how they BEHAVED (past_drawdown_reaction). This one is the
    want-vs-afford conflict; that one is the say-vs-do conflict.
    """
    cell = [list(c) for c in rubric["conflict_bins"]]
    return list(factor_cell(profile, rubric)) in cell


# Stated-vs-revealed risk conflict: what the client SAYS they want (stated_goal) vs how
# they actually BEHAVED in a drawdown (past_drawdown_reaction). A strong mismatch is the
# "contradictory" case we tag for the S3 integration experiment.
CONTRADICTION_FIELDS = ("stated_goal", "past_drawdown_reaction")


def is_contradictory(profile, rubric, fields=CONTRADICTION_FIELDS, hi=0.8, lo=0.2):
    """True if `fields` carry strongly opposing risk signals (one a_i >= hi, one <= lo).

    Catches e.g. maximum_growth + sold_everything, or capital_preservation + bought_more;
    leaves milder mismatches (e.g. maximum_growth + reduced) untagged.
    """
    sigs = [field_subscore(rubric["fields"][f], profile[f]) for f in fields]
    return max(sigs) >= hi and min(sigs) <= lo


def sample_rubric_fields(rubric, rng):
    profile = {}
    for field, spec in rubric["fields"].items():
        if "levels" in spec:
            profile[field] = rng.choice(spec["levels"])
        else:
            lo, hi = spec["range"]
            profile[field] = rng.randint(lo, hi)
    return profile


def sample_decoy_fields(rng):
    name, gender_hint = rng.choice(NAMES)
    return {
        "net_worth_band": rng.choice(DECOY_POOLS["net_worth_band"]),
        "occupation": rng.choice(DECOY_POOLS["occupation"]),
        "hobbies": rng.choice(DECOY_POOLS["hobbies"]),
        "city": rng.choice(DECOY_POOLS["city"]),
        "name": name,
        "gender_hint": gender_hint,
    }


def sample_profile(rubric, rng, profile_id):
    profile = {"profile_id": profile_id}
    profile.update(sample_rubric_fields(rubric, rng))
    profile.update(sample_decoy_fields(rng))
    score = risk_score(profile, rubric)
    profile["risk_score"] = round(score, 2)
    profile["tier"] = risk_tier(score, rubric)          # derived, no longer the target
    for name, value in factor_scores(profile, rubric).items():
        profile[name] = round(value, 4)
        profile[f"{name}_bin"] = factor_bin(value)
    profile["factor_cell"] = "/".join(factor_cell(profile, rubric))
    profile["conflicted"] = is_conflicted(profile, rubric)
    return profile
