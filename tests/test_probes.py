"""Offline unit tests for S5 (numpy/sklearn, no torch / no GPU)."""

import json

import numpy as np

from probes import (
    consolidate, eval_probe, fit_logistic, fractional_depth, load_plane,
    make_splits, selectivity, split_indices,
)
from utils.stats import auroc_ci, macro_ovr_auroc


# -- splits ----------------------------------------------------------------------------
def test_make_splits_deterministic():
    ids = [f"p{i}" for i in range(100)]
    a = make_splits(ids, seed=42)
    b = make_splits(ids, seed=42)
    assert a == b
    assert make_splits(ids, seed=1) != a            # seed actually matters


def test_make_splits_fractions_and_disjoint():
    ids = [f"p{i}" for i in range(1000)]
    s = make_splits(ids, seed=42, fracs=(0.7, 0.1, 0.2))
    counts = {k: sum(v == k for v in s.values()) for k in ("train", "val", "test")}
    assert counts == {"train": 700, "val": 100, "test": 200}
    assert set(s.values()) == {"train", "val", "test"}


def test_split_indices_routes_types_and_excludes_pairs():
    labels = [
        {"profile_id": "p1", "vignette_type": "explicit", "pair_id": None},   # 0 train
        {"profile_id": "p1", "vignette_type": "implicit", "pair_id": None},   # 1 implicit_train
        {"profile_id": "p2", "vignette_type": "explicit", "pair_id": None},   # 2 explicit_test
        {"profile_id": "p2", "vignette_type": "implicit", "pair_id": None},   # 3 implicit_test
        {"profile_id": "p2", "vignette_type": "implicit", "pair_id": "pr1"},  # 4 EXCLUDED (pair)
    ]
    split = {"p1": "train", "p2": "test"}
    idx = split_indices(labels, split)
    assert list(idx["train"]) == [0]
    assert list(idx["implicit_train"]) == [1]
    assert list(idx["explicit_test"]) == [2]
    assert list(idx["implicit_test"]) == [3]          # the pair row (4) is absent everywhere
    assert 4 not in set().union(*[set(v) for v in idx.values()])


# -- helpers ---------------------------------------------------------------------------
def test_fractional_depth():
    assert fractional_depth(0, 42) == 0.0
    assert fractional_depth(41, 42) == 1.0
    assert abs(fractional_depth(27, 42) - 0.659) < 1e-3   # ~65% sweet spot


def test_selectivity():
    assert selectivity(0.9, 0.5) == 0.4


# -- AUROC wrapper ---------------------------------------------------------------------
def test_macro_ovr_auroc_perfect_and_chance():
    y = np.array(["c", "c", "m", "m", "a", "a"])
    perfect = np.array([[1, 0, 0]] * 2 + [[0, 1, 0]] * 2 + [[0, 0, 1]] * 2, float)  # cols a,c,m order?
    # build proba aligned to sorted labels ['a','c','m']
    labels = ["a", "c", "m"]
    proba = np.zeros((6, 3))
    for i, t in enumerate(y):
        proba[i, labels.index(t)] = 1.0
    assert macro_ovr_auroc(y, proba, labels) == 1.0
    rng = np.random.default_rng(0)
    chance = rng.dirichlet(np.ones(3), size=6)
    assert 0.0 <= macro_ovr_auroc(y, chance, labels) <= 1.0


def test_auroc_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = np.array((["a"] * 30) + (["b"] * 30) + (["c"] * 30))
    labels = ["a", "b", "c"]
    # informative-ish scores: true class gets a boost
    proba = rng.dirichlet(np.ones(3), size=90)
    for i, t in enumerate(y):
        proba[i, labels.index(t)] += 0.5
    proba /= proba.sum(1, keepdims=True)
    pt, lo, hi = auroc_ci(y, proba, labels, n_boot=200, seed=1)
    assert lo <= pt <= hi


# -- probe trains on separable signal, control collapses to chance ---------------------
def _separable(n_per=120, d=16, n_sig=2, seed=0):
    """3 tiers separable along the first `n_sig` dims; the rest are pure noise.

    Signal-in-a-few-dims (not every dim) keeps the shuffled-label control at chance -- with
    every dim informative + heavy L2, finite-sample mean fluctuations let even a shuffled
    probe recover the geometry, which the real 3584-dim activations do not.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(3 * n_per, d))
    y = []
    for k, (tier, mu) in enumerate({"conservative": -2.5, "moderate": 0.0, "aggressive": 2.5}.items()):
        X[k * n_per:(k + 1) * n_per, :n_sig] += mu
        y += [tier] * n_per
    return X, np.array(y)


def test_probe_learns_signal_control_is_chance():
    Xtr, ytr = _separable(seed=0)
    Xva, yva = _separable(seed=1)
    Xte, yte = _separable(seed=2)
    probe = fit_logistic(Xtr, ytr, Xva, yva, seed=42)
    auc, acc, _ = eval_probe(probe, Xte, yte)
    assert auc > 0.9 and acc > 0.8
    # shuffled-label control (shuffle BOTH train and val, as run() does): AUROC ~ chance
    rng = np.random.default_rng(7)
    ctrl = fit_logistic(Xtr, ytr[rng.permutation(len(ytr))],
                        Xva, yva[rng.permutation(len(yva))], seed=42)
    cauc, _, _ = eval_probe(ctrl, Xte, yte)
    assert cauc < 0.65                      # not learning the real signal
    assert selectivity(auc, cauc) > 0.3


# -- consolidate stitches shards into a memmap with aligned labels ---------------------
def _write_shard(path, acts, labels):
    """Shard columns come from label_row, so this fixture cannot drift from the
    real schema when new label columns (factors, controls) are added."""
    from activations import label_row
    rows = [label_row({"text": "", "template_id": "t", "name": "n", **row})
            for row in labels]
    cols = {k: np.array([r[k] for r in rows], dtype=object) for k in rows[0]}
    np.savez(path, acts=acts.astype(np.float16), **cols)


def test_consolidate_and_load_plane(tmp_path):
    L, P, d = 2, 3, 8
    meta = {"model": "toy", "layers": [0, 1], "positions": ["profile_end", "profile_mean", "decision"],
            "hidden_size": d, "num_hidden_layers": 4}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    rows = []
    for s in range(2):                          # two shards, 5 rows each
        acts = np.arange(5 * L * P * d, dtype=np.float32).reshape(5, L, P, d) + s * 1000
        labs = [{"vignette_id": f"v{s}_{i}", "profile_id": f"p{s}_{i}", "pair_id": None,
                 "tier": "moderate", "risk_score": float(i), "vignette_type": "explicit",
                 "contradictory": False} for i in range(5)]
        _write_shard(tmp_path / f"shard_{s:04d}.npz", acts, labs)
        rows += labs
    acts, labels, m = consolidate(tmp_path)
    assert acts.shape == (10, L, P, d)
    assert len(labels) == 10
    assert labels[6]["vignette_id"] == "v1_1"
    plane = load_plane(acts, l_idx=1, p_idx=2)
    assert plane.shape == (10, d) and plane.dtype == np.float32
    # idempotent: second call reuses acts_all.npy
    acts2, labels2, _ = consolidate(tmp_path)
    assert acts2.shape == (10, L, P, d) and labels2[6]["vignette_id"] == "v1_1"


def test_consolidate_rebuilds_when_a_shard_is_added(tmp_path):
    """A stale acts_all.npy (from an earlier smaller run) must be rebuilt, not reused."""
    import os
    L, P, d = 2, 3, 8
    (tmp_path / "meta.json").write_text(json.dumps(
        {"model": "toy", "layers": [0, 1], "positions": ["profile_end", "profile_mean", "decision"],
         "hidden_size": d, "num_hidden_layers": 4}))

    def shard(s, n):
        acts = np.zeros((n, L, P, d), np.float32)
        labs = [{"vignette_id": f"v{s}_{i}", "profile_id": f"p{s}_{i}", "pair_id": None,
                 "tier": "moderate", "risk_score": 0.0, "vignette_type": "explicit",
                 "contradictory": False} for i in range(n)]
        _write_shard(tmp_path / f"shard_{s:04d}.npz", acts, labs)

    shard(0, 5)
    acts, labels, _ = consolidate(tmp_path)
    assert acts.shape[0] == 5                      # initial (small) consolidation

    shard(1, 7)                                    # S4 appends a new shard later
    os.utime(tmp_path / "shard_0001.npz", (1 << 31, 1 << 31))   # ensure it's newer than acts_all
    acts2, labels2, _ = consolidate(tmp_path)
    assert acts2.shape[0] == 12 and len(labels2) == 12          # rebuilt, not the stale 5
