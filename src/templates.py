"""S2 template bank + per-field verbalizers + banned-lexicon detection.

A vignette is assembled from a profile as:  opening (name/age/occupation/city) +
a template-ordered sequence of rubric sentences + decoy filler sentences.

Two verbalization modes:
  - "explicit": overt risk language allowed (used to TRAIN the probe).
  - "implicit": same facts via life narrative, ZERO banned vocabulary (held-out eval).

The implicit-mode phrasings here are the first line of defense against lexical leakage;
`tests/test_s2.py` exhaustively asserts no implicit cell emits a banned term, and S2B QC
re-scans the rendered output. The LLM paraphrase pass (src/paraphrase.py) only adds surface
diversity on top of these.
"""

import hashlib
import random
import re

# --- Banned lexicon (implicit tier only). The canonical list lives in config.yaml; this
# default is a fallback for direct imports/tests. Matching is word-boundary + stem so
# inflections are caught (risk -> risky/riskier; aggressive -> aggressively). ---
DEFAULT_BANNED = [
    "risk", "conservative", "aggressive", "volatile", "volatility", "tolerance",
    "cautious", "caution", "safe", "safety", "secure", "gamble", "speculative",
    "speculation", "bold", "daring", "timid", "wary", "prudent", "reckless",
]


def build_banned_regex(words=None):
    """Case-insensitive word-boundary + stem regex over the banned lexicon."""
    words = words if words is not None else DEFAULT_BANNED
    # \b<word>\w*  -> 'risk' catches risk/risky/riskier/risking; 'safe' catches safer/safety.
    alts = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\w*", re.IGNORECASE)


def find_banned(text, regex):
    """Return the list of banned-term hits in `text` (empty == clean)."""
    return regex.findall(text)


# --------------------------------------------------------------------------------------
# Per-field phrasings. Enum fields: dict[value] -> {explicit:[...], implicit:[...]}.
# Each list holds interchangeable phrasings; multiple phrasings give pre-paraphrase
# diversity. Every implicit phrasing must be banned-word-free (enforced by test).
# --------------------------------------------------------------------------------------

ENUM_PHRASINGS = {
    "past_drawdown_reaction": {
        "sold_everything": {
            "explicit": [
                "When the market dropped I sold everything; I'm a cautious, conservative investor who can't stomach big swings.",
                "I bailed out completely in the last downturn because I have very low tolerance for losses.",
            ],
            "implicit": [
                "When my portfolio fell about 20% a couple of years back, I couldn't sleep, so I moved everything into cash.",
                "The last big slump rattled me so badly that I pulled all of my money out near the bottom.",
            ],
        },
        "reduced": {
            "explicit": [
                "When markets fell I trimmed my positions; I lean conservative and like to limit how much I can lose.",
                "I cut back my exposure in the last drop because my appetite for risk is on the low side.",
            ],
            "implicit": [
                "When things dropped sharply, I sold off a chunk of my holdings to settle my nerves.",
                "During the last slide I lightened up, letting go of some of what I held so I could breathe easier.",
            ],
        },
        "held": {
            "explicit": [
                "I held through the last downturn; my appetite for swings is somewhere in the middle.",
                "I rode out the last drop without selling, taking a fairly balanced view of the ups and downs.",
            ],
            "implicit": [
                "When the market slid, I left my accounts alone and simply waited it out.",
                "I didn't touch anything during the last downturn and just let it recover on its own.",
            ],
        },
        "bought_more": {
            "explicit": [
                "When the market crashed I bought more; I'm an aggressive investor who treats dips as opportunities.",
                "I added heavily during the last sell-off because I have a high tolerance for risk.",
            ],
            "implicit": [
                "The last big drop actually had me adding money while prices were low.",
                "When everything fell, I saw a chance and put in more than I usually would.",
            ],
        },
    },
    "investing_experience": {
        "none": {
            "explicit": ["I have no real investing experience and feel cautious about it."],
            "implicit": [
                "I've never really invested before; this is all pretty new to me.",
                "Honestly I've never managed any investments, so I'm starting from scratch.",
            ],
        },
        "some": {
            "explicit": ["I have some investing experience but still play it fairly safe."],
            "implicit": [
                "I've dabbled in investing on and off for a few years.",
                "I've handled a modest account for a while now, nothing too involved.",
            ],
        },
        "extensive": {
            "explicit": ["I'm a very experienced investor and comfortable with aggressive positions."],
            # Duration-free: convey deep experience without asserting a tenure that would
            # contradict young ages (age and experience are sampled independently).
            "implicit": [
                "I've been through several market cycles and know how I react when things get rough.",
                "I'm a seasoned investor and comfortable making my own calls.",
                "I know my way around the markets and manage my own portfolio confidently.",
            ],
        },
    },
    "income_stability": {
        "precarious": {
            "explicit": ["My income is precarious, which makes me want to play it safe."],
            "implicit": [
                "My income is unpredictable; some months are great and others I barely cover the bills.",
                "My earnings swing a lot and I can't count on a steady paycheck.",
            ],
        },
        "variable": {
            "explicit": ["My income is variable, so I keep a fairly cautious stance."],
            "implicit": [
                "My income varies a fair bit from month to month.",
                "My pay isn't fixed; it rises and falls with how much work comes in.",
            ],
        },
        "stable": {
            "explicit": ["My income is stable, which lets me take on more risk."],
            "implicit": [
                "I have a steady, dependable paycheck every month.",
                "My income is reliable and easy to plan around.",
            ],
        },
    },
    "stated_goal": {
        "capital_preservation": {
            "explicit": ["My goal is capital preservation; keeping what I have safe matters most."],
            "implicit": [
                "Mostly I just want to protect the money I've saved and not lose it.",
                "My main aim is to hold onto what I've built up over the years.",
            ],
        },
        "income": {
            "explicit": ["My goal is steady income with a conservative tilt."],
            "implicit": [
                "I'm looking for steady income from my investments to help cover expenses.",
                "I want my savings to throw off regular cash I can live on.",
            ],
        },
        "balanced_growth": {
            "explicit": ["My goal is balanced growth with a moderate appetite for swings."],
            "implicit": [
                "I'd like my money to grow steadily over time without wild swings.",
                "I'm after reasonable, gradual growth more than anything dramatic.",
            ],
        },
        "maximum_growth": {
            "explicit": ["I want maximum growth and I'll take on aggressive risk to get it."],
            "implicit": [
                "I want my money to grow as much as possible, even if the ride is bumpy.",
                "I'm chasing the biggest long-term gains I can get, swings and all.",
            ],
        },
    },
}

# Banded continuous fields: list of (lo, hi, {explicit, implicit}) inclusive ranges.
BAND_PHRASINGS = {
    "horizon_years": [
        (1, 3, {
            "explicit": ["My time horizon is short, so I invest conservatively."],
            "implicit": ["I expect to need this money within the next year or two.",
                         "My timeline is short; I'll want these funds fairly soon."],
        }),
        (4, 9, {
            "explicit": ["I have a medium horizon and a moderate stance."],
            "implicit": ["I'm looking at roughly a five-to-ten year timeline.",
                         "I won't need this money for about half a decade or so."],
        }),
        (10, 20, {
            "explicit": ["I have a long horizon, which lets me take more risk."],
            "implicit": ["I'm investing for the long haul, maybe ten to twenty years out.",
                         "This is money I won't touch for a decade or two."],
        }),
        (21, 35, {
            "explicit": ["I have a very long horizon and can be aggressive."],
            "implicit": ["I'm thinking very long term, thirty years or more down the road.",
                         "I won't need any of this for decades."],
        }),
    ],
    "emergency_fund_months": [
        (0, 2, {
            "explicit": ["I have almost no emergency fund, so I stay cautious."],
            "implicit": ["I don't really have an emergency cushion set aside.",
                         "If something went wrong I'd have little to fall back on."],
        }),
        (3, 6, {
            "explicit": ["I keep a small emergency fund."],
            "implicit": ["I've got a few months of expenses saved as a cushion.",
                         "My rainy-day reserve covers maybe three to six months."],
        }),
        (7, 12, {
            "explicit": ["I keep a solid emergency fund."],
            "implicit": ["I keep close to a year of expenses in reserve.",
                         "I have a good cushion, roughly nine months to a year of costs."],
        }),
        (13, 24, {
            "explicit": ["I keep a very large emergency fund, which makes me feel secure."],
            "implicit": ["My cash reserve covers well over a year of expenses.",
                         "I keep nearly two years of living costs tucked away."],
        }),
    ],
    "dependents": [
        (0, 0, {
            "explicit": ["I have no dependents."],
            "implicit": ["I don't have anyone financially depending on me.",
                         "It's just me, no dependents."],
        }),
        (1, 1, {
            "explicit": ["I have one dependent."],
            "implicit": ["I have one child who depends on me.",
                         "I support one dependent at home."],
        }),
        (2, 2, {
            "explicit": ["I have two dependents."],
            "implicit": ["I have two kids relying on me.",
                         "There are two dependents counting on my income."],
        }),
        (3, 4, {
            "explicit": ["I have several dependents."],
            "implicit": ["I have a big family, with several children depending on me.",
                         "I support three or four dependents at home."],
        }),
    ],
}

NET_WORTH_PHRASINGS = {
    "under_100k": ["My net worth is under $100k.", "I've got less than a hundred thousand to my name."],
    "100k_to_500k": ["My net worth sits somewhere between $100k and $500k."],
    "500k_to_2m": ["I'm worth somewhere in the $500k to $2 million range."],
    "over_2m": ["My net worth is north of $2 million."],
}

OPENINGS = [
    "My name is {name}. I'm {age} and I work as {occupation} in {city}.",
    "{name} here, {age} years old, working as {occupation} over in {city}.",
    "I'm {name}, {age}, {occupation} based in {city}.",
    "Let me introduce myself: I'm {name}, {age} years old, {occupation} in {city}.",
    "I'm {name}. At {age}, I make my living as {occupation} here in {city}.",
    "Hi, I'm {name}, a {age}-year-old {occupation} from {city}.",
]

HOBBY_PHRASINGS = [
    "In my free time I enjoy {hobby}.",
    "Outside of work I'm really into {hobby}.",
    "When I'm not working you'll usually find me {hobby}.",
]

# Order in which the middle blocks (everything after the opening) are laid out. Four
# distinct orderings over the same block keys give structural variety; the paraphraser
# adds the rest. 'age'/'name'/'occupation'/'city' live in the opening, not here.
MIDDLE_KEYS = [
    "stated_goal", "horizon_years", "investing_experience", "income_stability",
    "past_drawdown_reaction", "emergency_fund_months", "dependents", "net_worth", "hobbies",
]
ORDERINGS = [
    MIDDLE_KEYS,
    ["past_drawdown_reaction", "investing_experience", "income_stability", "stated_goal",
     "horizon_years", "emergency_fund_months", "net_worth", "dependents", "hobbies"],
    ["hobbies", "dependents", "income_stability", "net_worth", "horizon_years",
     "stated_goal", "investing_experience", "emergency_fund_months", "past_drawdown_reaction"],
    ["stated_goal", "investing_experience", "past_drawdown_reaction", "horizon_years",
     "income_stability", "net_worth", "emergency_fund_months", "hobbies", "dependents"],
]

# Stable template ids = (opening index, ordering index).
TEMPLATE_IDS = [f"tmpl_{o:02d}_{k}" for o in range(len(OPENINGS)) for k in range(len(ORDERINGS))]


def _stable_int(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def pick_template_id(profile_id, seed):
    """Deterministically assign a template to a profile (twins share it)."""
    return TEMPLATE_IDS[_stable_int(f"{seed}|{profile_id}") % len(TEMPLATE_IDS)]


def _band_lookup(field, value):
    for lo, hi, phr in BAND_PHRASINGS[field]:
        if lo <= value <= hi:
            return phr
    raise ValueError(f"{field}={value} out of banded range")


def verbalize_field(field, value, mode, rng):
    """Render one field as a natural sentence in the given mode."""
    if field in ENUM_PHRASINGS:
        return rng.choice(ENUM_PHRASINGS[field][value][mode])
    if field in BAND_PHRASINGS:
        return rng.choice(_band_lookup(field, value)[mode])
    if field == "net_worth":
        return rng.choice(NET_WORTH_PHRASINGS[value])
    if field == "hobbies":
        return rng.choice(HOBBY_PHRASINGS).format(hobby=value)
    raise KeyError(field)


def render(profile, template_id, mode, rng, stable_seed=None):
    """Assemble the full client narrative for `profile` under `template_id`/`mode`.

    `stable_seed` makes each field's phrasing depend ONLY on that field. Normally the
    fields share one rng stream, so a field whose wording consumes a different number
    of draws shifts every sentence after it: flipping one field then rewords unrelated
    sentences too. That is invisible for twins, but it silently breaks one-factor
    counterfactual pairs, where the whole point is that exactly one sentence moves.
    Pass a per-pair seed there and each field draws from its own independent stream.
    """
    _, oi, ki = template_id.split("_")
    opening = OPENINGS[int(oi)].format(
        name=profile["name"], age=profile["age"],
        occupation=profile["occupation"], city=profile["city"],
    )

    def field_rng(field):
        return random.Random(f"{stable_seed}|{field}") if stable_seed is not None else rng

    block = {}
    for field in ("stated_goal", "horizon_years", "investing_experience",
                  "income_stability", "past_drawdown_reaction", "emergency_fund_months",
                  "dependents"):
        block[field] = verbalize_field(field, profile[field], mode, field_rng(field))
    block["net_worth"] = verbalize_field("net_worth", profile["net_worth_band"], mode,
                                         field_rng("net_worth"))
    block["hobbies"] = verbalize_field("hobbies", profile["hobbies"], mode,
                                       field_rng("hobbies"))

    middle = " ".join(block[k] for k in ORDERINGS[int(ki)])
    return opening + " " + middle
