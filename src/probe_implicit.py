"""Like-for-like RQ1 probe: trained on IMPLICIT_TRAIN, scored on IMPLICIT_TEST.

The frozen probe (probes.json) is trained on EXPLICIT vignettes and transferred to implicit
text, while the strong TF-IDF baseline (baselines.py) is trained on implicit text. This
script gives the probe the same training distribution: same estimator, standardization and C
grid as probes.fit_logistic (reused), same splits, same metric (macro one-vs-rest AUROC).

  - C is chosen on implicit_val inside fit_logistic; the (layer, position) cell is chosen on
    implicit_val too, per target. implicit_test is reported once at the chosen cell (the test
    score of every other cell is stored only to draw the layer curve, never used to select).
  - Bootstrap 95% CIs (n_boot, seed) over implicit_test rows for the probe, for TF-IDF, and
    for the paired difference probe - tfidf (same resampled rows for both).
  - Controls: shuffled-label probe at the chosen cell; the layer-10 profile_mean probe.

Rows are aligned by vignette_id: the cache row order is not the vignette-file order, but the
profile split is identical (checked), so texts are looked up by id.

Usage:
    PYTHONPATH=src python src/probe_implicit.py [--config config.yaml] [--n-jobs N]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from baselines import C_GRID, TARGETS, fit_tfidf, target_labels
from probes import (eval_probe, fit_logistic, fractional_depth, load_plane, load_texts,
                    make_splits, probe_proba, split_indices)
from utils.stats import macro_ovr_auroc


def paired_bootstrap(y, scores, classes, n_boot=1000, seed=42):
    """Shared-row bootstrap. scores: {name: proba[n, k]}. Returns point + 95% CI per name and
    for every pairwise difference first - other (resamples missing a class are skipped)."""
    y = np.asarray(y)
    names = list(scores)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = {k: [] for k in names}
    for _ in range(n_boot):
        s = rng.choice(n, size=n, replace=True)
        if set(np.unique(y[s])) != set(classes):
            continue
        for k in names:
            boots[k].append(macro_ovr_auroc(y[s], scores[k][s], classes))
    ci = lambda a: [round(float(v), 4) for v in np.percentile(a, [2.5, 97.5])]
    out = {k: {"auroc": round(macro_ovr_auroc(y, scores[k], classes), 4), "ci": ci(boots[k])}
           for k in names}
    base = names[0]
    for k in names[1:]:
        d = np.array(boots[base]) - np.array(boots[k])
        out[f"{base}_minus_{k}"] = {"diff": round(out[base]["auroc"] - out[k]["auroc"], 4),
                                    "ci": ci(d), "frac_boot_positive": round(float((d > 0).mean()), 4)}
    return out


def _cell(acts_path, l_idx, p_idx, layer, position, idx, ys, cell_seed):
    """Fit one probe per target at one (layer, position); score implicit_val and implicit_test."""
    from threadpoolctl import threadpool_limits
    out = {}
    with threadpool_limits(1):
        plane = load_plane(np.load(acts_path, mmap_mode="r"), l_idx, p_idx)
        Xtr, Xv, Xt = (plane[idx[k]] for k in ("implicit_train", "implicit_val", "implicit_test"))
        for t, y in ys.items():
            ytr, yv, yt = y[idx["implicit_train"]], y[idx["implicit_val"]], y[idx["implicit_test"]]
            pr = fit_logistic(Xtr, ytr, Xv, yv, seed=cell_seed)
            v_auc, _, _ = eval_probe(pr, Xv, yv)
            t_auc, _, t_proba = eval_probe(pr, Xt, yt)
            out[t] = {"layer": layer, "position": position, "C": pr["C"], "val": float(v_auc),
                      "test": float(t_auc), "classes": pr["classes"], "test_proba": t_proba}
    return out


def run(cfg, act_dir, n_jobs, n_boot=1000, cache=None):
    from joblib import Parallel, delayed
    act_dir = Path(act_dir)
    meta = json.loads((act_dir / "meta.json").read_text())
    labels = [json.loads(line) for line in open(act_dir / "labels.jsonl")]
    layers, positions = meta["layers"], meta["positions"]
    seed = cfg["seed"]
    split = make_splits([lab["profile_id"] for lab in labels], seed=seed)
    idx = split_indices(labels, split)
    ys = {t: target_labels(labels, t) for t in TARGETS}
    acts_path = str(act_dir / "acts_all.npy")

    cells = [(l_i, p_i, l, p) for p_i, p in enumerate(positions) for l_i, l in enumerate(layers)]
    if cache and Path(cache).exists():
        res = pickle.load(open(cache, "rb"))
    else:
        res = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_cell)(acts_path, l_i, p_i, l, p, idx, ys, seed + 1000 * p_i + l)
            for l_i, p_i, l, p in cells)
        if cache:
            pickle.dump(res, open(cache, "wb"))
    by_cell = {(c[2], c[3]): r for c, r in zip(cells, res)}

    texts = load_texts(cfg)
    # pair rows have no text and are never indexed (excluded by split_indices)
    text = np.array([texts.get(lab["vignette_id"], "") for lab in labels], dtype=object)
    tg = idx["implicit_test"]
    out = {"meta": {"cache_id": meta["cache_id"], "seed": seed, "n_boot": n_boot,
                    "n": {k: int(len(v)) for k, v in idx.items()},
                    "estimator": "probes.fit_logistic: StandardScaler + L2 logistic, C in (0.001,0.01,0.1) "
                                 "picked on implicit_val; trained on implicit_train",
                    "selection": "(layer, position) argmax implicit_val AUROC per target; "
                                 "implicit_test reported at that cell only"},
           "targets": {}}

    for t in TARGETS:
        y = ys[t]
        best = max(by_cell, key=lambda c: by_cell[c][t]["val"])
        b = by_cell[best][t]
        classes = b["classes"]

        # TF-IDF: C=1 (as baselines.py main) and C picked on implicit_val
        tr, va = idx["implicit_train"], idx["implicit_val"]

        def val_auc(C):
            sc, cl = fit_tfidf(text[tr], y[tr], C, seed)
            return macro_ovr_auroc(y[va], sc(text[va]), cl)
        C_dev = max(C_GRID, key=val_auc)
        sc1, cl1 = fit_tfidf(text[tr], y[tr], 1.0, seed)
        scd, cld = fit_tfidf(text[tr], y[tr], C_dev, seed)
        assert cl1 == classes and cld == classes
        scores = {"probe": b["test_proba"], "tfidf_C1": sc1(text[tg]), "tfidf_Cdev": scd(text[tg])}
        boot = paired_bootstrap(y[tg], scores, classes, n_boot, seed)

        # shuffled-label control at the chosen cell (labels permuted in train and val, as in probes.py)
        l_i, p_i = layers.index(best[0]), positions.index(best[1])
        plane = load_plane(np.load(acts_path, mmap_mode="r"), l_i, p_i)
        rng = np.random.default_rng(seed + 1000 * p_i + best[0])
        ytr, yv = y[tr], y[va]
        ctrl = fit_logistic(plane[tr], ytr[rng.permutation(len(ytr))], plane[va],
                            yv[rng.permutation(len(yv))], Cs=(0.1,), seed=seed)
        ctrl_auc = eval_probe(ctrl, plane[tg], y[tg])[0]

        curves = {p: {str(l): {"val": round(by_cell[(l, p)][t]["val"], 4),
                               "test": round(by_cell[(l, p)][t]["test"], 4)} for l in layers}
                  for p in ("profile_mean", "decision", "profile_end")}
        l10 = by_cell[(10, "profile_mean")][t]
        out["targets"][t] = {
            "chosen": {"layer": best[0], "position": best[1], "frac_depth": fractional_depth(best[0], meta["num_hidden_layers"]),
                       "C": b["C"], "implicit_val_auroc": round(b["val"], 4)},
            "implicit_test": boot,
            "tfidf_C": {"tfidf_C1": 1.0, "tfidf_Cdev": C_dev},
            "shuffled_control_auroc": round(float(ctrl_auc), 4),
            "layer10_profile_mean": {"val": round(l10["val"], 4), "test": round(l10["test"], 4)},
            "layer_curves": curves,
            "frozen_explicit_trained_probe_auroc": None,
        }
        p_diff = boot["probe_minus_tfidf_Cdev"]
        print(f"{t:12s} cell=L{best[0]}/{best[1]} probe={boot['probe']['auroc']} {boot['probe']['ci']} "
              f"tfidf(C1)={boot['tfidf_C1']['auroc']} tfidf(Cdev={C_dev})={boot['tfidf_Cdev']['auroc']} "
              f"diff={p_diff['diff']} {p_diff['ci']} shuffled={ctrl_auc:.3f} L10pm={l10['test']:.3f}", flush=True)

    probes_json = Path(cfg["paths"]["results_dir"]) / "gemma-2-9b-it" / "probes.json"
    if probes_json.exists():
        pj = json.loads(probes_json.read_text())
        froz = {"tier": pj["best"]["implicit_auroc"]}
        froz.update({f: v["implicit_auroc"] for f, v in pj["factor_probes"].items()})
        for t in TARGETS:
            out["targets"][t]["frozen_explicit_trained_probe_auroc"] = froz.get(t)
    margin = cfg["probe"]["baseline_margin_min"]
    tier = out["targets"]["tier"]["implicit_test"]
    out["tier_margin_bar"] = {"bar": margin,
                              "probe_minus_tfidf_Cdev": tier["probe_minus_tfidf_Cdev"]["diff"],
                              "met": bool(tier["probe_minus_tfidf_Cdev"]["diff"] >= margin)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--activations", default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--cache", default=None, help="pickle of the layer sweep, to skip refitting on rerun")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    act = args.activations or str(Path(cfg["paths"]["results_dir"]) / "gemma-2-9b-it" / "activations" / "34590aab67")
    result = run(cfg, act, args.n_jobs, cache=args.cache)
    dest = Path(cfg["paths"]["results_dir"]) / "gemma-2-9b-it" / "probes" / "implicit_trained.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(f"wrote {dest}")
