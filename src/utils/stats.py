"""Shared statistics helpers: bootstrap CIs and macro one-vs-rest AUROC.

Used by S3 (gate, Spearman CI) and S5 (probe AUROC CI). Kept dependency-light (numpy +
scipy + sklearn, all CPU) so the probe-evaluation logic is unit-testable without a GPU.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def spearman_ci(x, y, n_boot=2000, seed=42):
    """Spearman rho with a bootstrap 95% percentile CI (resampling paired observations)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    boot = []
    idx = np.arange(len(x))
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if np.std(x[s]) > 0 and np.std(y[s]) > 0:
            boot.append(spearmanr(x[s], y[s]).correlation)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    return float(rho), float(lo), float(hi)


def macro_ovr_auroc(y_true, y_score, labels):
    """Macro-averaged one-vs-rest AUROC.

    `y_true`: int/str class labels, shape [n]. `y_score`: class probabilities, shape
    [n, n_classes] aligned to `labels` order. Falls back to the binary AUROC when there
    are two classes (sklearn wants the positive-class column there).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, float)
    labels = list(labels)
    if len(labels) == 2:
        pos = labels[1]
        return float(roc_auc_score((y_true == pos).astype(int), y_score[:, 1]))
    return float(roc_auc_score(y_true, y_score, multi_class="ovr",
                               average="macro", labels=labels))


def partial_spearman(x, y, control):
    """Spearman partial correlation of x and y controlling for `control`.

    Rank-transform all three, residualize x and y on the control ranks (with intercept),
    then Pearson on the residuals. Used for the "whose side is the advice on" analysis:
    corr(advice, stated_goal | risk_score) vs corr(advice, risk_score | stated_goal).
    """
    from scipy.stats import rankdata
    rx, ry, rc = (rankdata(np.asarray(v, float)) for v in (x, y, control))
    A = np.c_[rc, np.ones(len(rc))]
    res_x = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    denom = np.std(res_x) * np.std(res_y)
    if denom == 0:
        return float("nan")
    return float(np.mean((res_x - res_x.mean()) * (res_y - res_y.mean())) / denom)


def standardized_rel_weights(X, y):
    """Standardized OLS betas of y on X columns, plus each column's share of total |beta|.

    The behavioral/probe "field importance" measure: fit y (advice or probe readout) on the
    rubric-normalized field values, z-scoring everything first so betas are comparable, then
    normalize |beta| to sum to 1 for comparison against the rubric's own weights.
    Returns (betas, rel_weights, r2).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    A = np.c_[Xs, np.ones(len(ys))]
    b = np.linalg.lstsq(A, ys, rcond=None)[0]
    betas = b[:-1]
    resid = A @ b - ys
    r2 = 1.0 - float(resid @ resid) / float(((ys - ys.mean()) ** 2).sum())
    rel = np.abs(betas) / np.abs(betas).sum()
    return betas, rel, r2


def auroc_ci(y_true, y_score, labels, n_boot=2000, seed=42):
    """Macro-OvR AUROC with a bootstrap 95% CI over the evaluation rows.

    Bootstrap resamples that happen to miss a class (so OvR is undefined) are skipped.
    Returns (point_estimate, lo, hi).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, float)
    labels = list(labels)
    point = macro_ovr_auroc(y_true, y_score, labels)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    boot = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if set(np.unique(y_true[s])) != set(labels):     # need every class present for OvR
            continue
        try:
            boot.append(macro_ovr_auroc(y_true[s], y_score[s], labels))
        except ValueError:
            continue
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)
