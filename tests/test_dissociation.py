"""Offline unit tests for S5b (dissociation analysis; no activation cache needed)."""

import numpy as np
import yaml

from dissociation import (continuous_readout, dissociation_verdict, elicitation_verdict,
                              eval_probe_on, implicit_rows_for_splits, probe_predict_proba)


def _cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# -- probe reconstruction matches sklearn -------------------------------------------------
def test_probe_predict_proba_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    y = np.array(["conservative", "moderate", "aggressive"])[
        (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=0).fit(sc.transform(X), y)
    probe = {"classes": list(clf.classes_), "coef": clf.coef_.astype(np.float64),
             "intercept": clf.intercept_.astype(np.float64),
             "scaler_mean": sc.mean_.astype(np.float64),
             "scaler_scale": sc.scale_.astype(np.float64)}
    Xnew = rng.normal(size=(50, 12))
    ours = probe_predict_proba(probe, Xnew)
    theirs = clf.predict_proba(sc.transform(Xnew))
    assert np.allclose(ours, theirs, atol=1e-8)


def test_continuous_readout_sign():
    classes = ["aggressive", "conservative", "moderate"]      # order from the npz, not sorted
    proba = np.array([[0.8, 0.1, 0.1],       # confident aggressive -> positive
                      [0.1, 0.8, 0.1]])      # confident conservative -> negative
    r = continuous_readout(proba, classes)
    assert r[0] > 0.5 and r[1] < -0.5


# -- row selection ------------------------------------------------------------------------
def test_implicit_rows_for_splits_filters_pairs_type_and_split():
    labels = [
        {"profile_id": "p1", "pair_id": None, "vignette_type": "implicit"},    # val -> in
        {"profile_id": "p2", "pair_id": None, "vignette_type": "implicit"},    # test -> in
        {"profile_id": "p3", "pair_id": None, "vignette_type": "implicit"},    # train -> out
        {"profile_id": "p1", "pair_id": None, "vignette_type": "explicit"},    # type -> out
        {"profile_id": "p2", "pair_id": "x", "vignette_type": "implicit"},     # pair -> out
    ]
    split = {"p1": "val", "p2": "test", "p3": "train"}
    idx = implicit_rows_for_splits(labels, split, ("val", "test"))
    assert idx.tolist() == [0, 1]


# -- pre-registered verdict bands ---------------------------------------------------------
def test_dissociation_verdicts_against_config_bands():
    bands = _cfg()["dissociation"]
    assert dissociation_verdict(0.86, 0.90, bands) == "readout_collapse"
    assert dissociation_verdict(0.80, 0.90, bands) == "readout_collapse"   # boundary holds
    assert dissociation_verdict(0.60, 0.90, bands) == "encoding_collapse"
    assert dissociation_verdict(0.72, 0.90, bands) == "partial"
    assert dissociation_verdict(0.82, 0.95, bands) == "partial"            # holds but drops >0.10


def test_elicitation_verdicts_against_config_bands():
    bands = _cfg()["elicitation"]
    assert elicitation_verdict(0.894, 0.80, bands) == "latent_knowledge"
    assert elicitation_verdict(0.894, 0.88, bands) == "promptable_knowledge"
    assert elicitation_verdict(0.894, 0.85, bands) == "partial"


# -- subset evaluation --------------------------------------------------------------------
def test_eval_probe_on_perfect_probe():
    classes = ["aggressive", "conservative", "moderate"]
    tier = np.array(["aggressive", "conservative", "moderate"] * 20)
    proba = np.array([{"aggressive": [0.9, 0.05, 0.05],
                       "conservative": [0.05, 0.9, 0.05],
                       "moderate": [0.05, 0.05, 0.9]}[t] for t in tier])
    rec = eval_probe_on(proba, tier, np.arange(len(tier)), classes, n_boot=50, seed=0)
    assert rec["auroc"] == 1.0 and rec["acc"] == 1.0 and rec["n"] == 60
