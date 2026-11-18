"""RQ1 lexical baseline trained on IMPLICIT stories (CPU only, no activations needed).

The baseline in probes.json (`baselines.tfidf`) is fit on the EXPLICIT train vignettes and
scored on implicit_test. Explicit text is full of risk words that implicit text lacks by
construction, so that baseline is handicapped. The fair lexical opponent is fit on the
implicit TRAIN split (same profile split as the probe, `make_splits` / `split_indices`) and
scored on the same implicit_test rows, same metric (macro one-vs-rest AUROC).

Targets: the blended tier plus the factor targets probes.json reports (willingness,
capacity, goals, factor_cell). Both training regimes are reported, with bootstrap 95% CIs
over implicit_test rows, next to the probe numbers read from probes.json.

Usage:
    PYTHONPATH=src python src/baselines.py [--config config.yaml]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from probes import make_splits, split_indices
from rubric import BINS, factor_bin
from utils.stats import auroc_ci, macro_ovr_auroc

TARGETS = ("tier", "willingness", "capacity", "goals", "factor_cell")
C_GRID = (0.1, 1.0, 10.0, 100.0)


def load_rows(vignettes_dir):
    """explicit + implicit + pairs, in the order the S4 cache used.

    The pair rows are not probe rows, but their profile ids go into make_splits, so they
    must be here for the split to be the probe's split (n_train 4182, n_implicit_test 1202).
    """
    rows = []
    for name in ("explicit", "implicit", "pairs"):
        with open(Path(vignettes_dir) / f"{name}.jsonl") as f:
            rows += [json.loads(line) for line in f]
    return rows


def target_labels(rows, target):
    if target in ("tier", "factor_cell"):
        return np.array([r[target] for r in rows])
    return np.array([factor_bin(float(r[target]), BINS) for r in rows])


def fit_tfidf(train_texts, y, C=1.0, seed=42):
    """Same vectorizer and classifier as probes.text_baseline; returns a scorer."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(vec.fit_transform(train_texts), y)
    return lambda texts: clf.predict_proba(vec.transform(texts)), list(clf.classes_)


def run(cfg):
    rows = load_rows(cfg["paths"]["vignettes_dir"])
    split = make_splits([r["profile_id"] for r in rows], seed=cfg["seed"])
    idx = split_indices(rows, split)
    text = np.array([r["text"] for r in rows], dtype=object)
    probes = json.loads((Path(cfg["paths"]["results_dir"]) / "gemma-2-9b-it" / "probes.json").read_text())
    probe_auc = {"tier": probes["best"]["implicit_auroc"]}
    probe_auc.update({f: v["implicit_auroc"] for f, v in probes["factor_probes"].items()})

    out = {"n": {k: int(len(v)) for k, v in idx.items()},
           "setup": "TF-IDF(20000, 1-2gram)+LogReg, macro OvR AUROC on implicit_test, bootstrap 95% CI over rows",
           "margin_bar": cfg["probe"]["baseline_margin_min"], "targets": {}}
    tg = idx["implicit_test"]
    for target in TARGETS:
        y = target_labels(rows, target)
        rec = {"probe_implicit_auroc": probe_auc.get(target)}
        for name, train_key in (("trained_on_explicit", "train"), ("trained_on_implicit", "implicit_train")):
            score, classes = fit_tfidf(text[idx[train_key]], y[idx[train_key]], 1.0, cfg["seed"])
            auc, lo, hi = auroc_ci(y[tg], score(text[tg]), classes, seed=cfg["seed"])
            rec[name] = {"auroc": round(auc, 4), "ci": [round(lo, 4), round(hi, 4)], "C": 1.0}
        # stronger variant: C picked on implicit_val (dev), never on the test rows
        best = max(C_GRID, key=lambda C: _val_auc(text, y, idx, C, cfg["seed"]))
        score, classes = fit_tfidf(text[idx["implicit_train"]], y[idx["implicit_train"]], best, cfg["seed"])
        auc, lo, hi = auroc_ci(y[tg], score(text[tg]), classes, seed=cfg["seed"])
        rec["trained_on_implicit_C_dev"] = {"auroc": round(auc, 4), "ci": [round(lo, 4), round(hi, 4)], "C": best}
        if rec["probe_implicit_auroc"] is not None:
            strongest = max(rec[k]["auroc"] for k in ("trained_on_explicit", "trained_on_implicit",
                                                      "trained_on_implicit_C_dev"))
            rec["probe_minus_strongest_baseline"] = round(rec["probe_implicit_auroc"] - strongest, 4)
            rec["margin_bar_met"] = bool(rec["probe_minus_strongest_baseline"] >= out["margin_bar"])
        out["targets"][target] = rec
        print(f"{target:12s} probe={rec['probe_implicit_auroc']}  "
              f"explicit-trained={rec['trained_on_explicit']['auroc']}  "
              f"implicit-trained={rec['trained_on_implicit']['auroc']} {rec['trained_on_implicit']['ci']}  "
              f"implicit C={best}: {rec['trained_on_implicit_C_dev']['auroc']}  "
              f"margin={rec.get('probe_minus_strongest_baseline')}")
    out["old_tfidf_in_probes_json"] = probes["baselines"]["tfidf"]["implicit_test"]
    return out


def _val_auc(text, y, idx, C, seed):
    score, classes = fit_tfidf(text[idx["implicit_train"]], y[idx["implicit_train"]], C, seed)
    return macro_ovr_auroc(y[idx["implicit_val"]], score(text[idx["implicit_val"]]), classes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    result = run(cfg)
    dest = Path(cfg["paths"]["results_dir"]) / "gemma-2-9b-it" / "baselines" / "tfidf_implicit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"wrote {dest}")
