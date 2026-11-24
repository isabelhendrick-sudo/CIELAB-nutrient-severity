"""
stats_utils.py — statistics used by the analysis (and by Part D of the notebook).

Agreement for the known-severity test:
    from stats_utils import agreement_stats, truth_from_alpha
    true_sev = truth_from_alpha(alpha, leaf_mask)   # summed lesion opacity, not "any coverage"
    agreement_stats(true_list, measured_list)       # r, slope, bias, LoA, Lin's CCC, Cb
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


# ---------------------------------------------------------------- classification metrics
def safe_auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_ap(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return float(average_precision_score(y[ok], s[ok]))


def youden_cutoff(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    fpr, tpr, thr = roc_curve(y[ok], s[ok])
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def binary_metrics(y, s, cutoff: float) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    pred = s >= cutoff
    tp, fn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 1))
    tn, fp = np.sum(~pred & (y == 0)), np.sum(pred & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"auc": safe_auc(y, s), "pr_auc": safe_ap(y, s), "sensitivity": sens,
            "specificity": spec, "balanced_accuracy": np.nanmean([sens, spec]),
            "prevalence": float(np.mean(y)), "n": int(len(y))}


# ---------------------------------------------------------------- group bootstrap
def group_bootstrap(df: pd.DataFrame, group_col: str, stat_fn, n_boot: int = 2000,
                    seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap resampling whole groups (plants). stat_fn(df) -> float or dict."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    idx = {g: np.flatnonzero(df[group_col].values == g) for g in groups}
    point = stat_fn(df)
    is_dict = isinstance(point, dict)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        try:
            draws.append(stat_fn(df.iloc[rows]))
        except Exception:  # noqa: BLE001  (e.g. a resample with one class only)
            continue
    out = {"n_groups": len(groups), "n_boot_ok": len(draws)}
    keys = point.keys() if is_dict else ["value"]
    for k in keys:
        vals = np.array([d[k] if is_dict else d for d in draws], dtype=float)
        vals = vals[~np.isnan(vals)]
        pv = point[k] if is_dict else point
        lo, hi = (np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) > 10 else (np.nan, np.nan))
        out[k] = pv
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    return out


# ---------------------------------------------------------------- agreement (known severity)
def lins_ccc(x, y) -> dict:
    """Lin's concordance correlation coefficient with its bias-correction factor Cb (= CCC / r)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(), y.var()                     # population variances, as in Lin (1989)
    sxy = np.mean((x - mx) * (y - my))
    ccc = 2 * sxy / (sx2 + sy2 + (mx - my) ** 2)
    r = np.corrcoef(x, y)[0, 1]
    return {"ccc": float(ccc), "r": float(r), "cb": float(ccc / r) if r else np.nan}


def agreement_stats(truth, measured) -> dict:
    """Agreement between true and measured severity (%): r, OLS slope/intercept,
    mean bias, 95% limits of agreement (Bland & Altman), Lin's CCC and Cb."""
    t, m = np.asarray(truth, float), np.asarray(measured, float)
    ok = ~(np.isnan(t) | np.isnan(m))
    t, m = t[ok], m[ok]
    reg = stats.linregress(t, m)
    diff = m - t
    sd = diff.std(ddof=1)
    out = {"n": int(len(t)), "slope": reg.slope, "intercept": reg.intercept,
           "mean_bias": diff.mean(), "loa_lo": diff.mean() - 1.96 * sd,
           "loa_hi": diff.mean() + 1.96 * sd}
    out.update(lins_ccc(t, m))
    return out


def truth_from_alpha(alpha: np.ndarray, leaf_mask: np.ndarray) -> float:
    """True severity (%) of a synthetic leaf = summed lesion opacity / leaf area."""
    leaf = leaf_mask.astype(bool)
    return 100.0 * float(np.clip(alpha, 0, 1)[leaf].sum()) / max(1, int(leaf.sum()))


# ---------------------------------------------------------------- progression
def audpc(t, y) -> float:
    t, y = np.asarray(t, float), np.asarray(y, float)
    o = np.argsort(t)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(y[o], t[o])) if len(t) > 1 else np.nan


def relative_audpc(t, y, max_value: float = 100.0) -> float:
    """AUDPC divided by (duration x max severity): comparable across series of different length."""
    t = np.asarray(t, float)
    dur = t.max() - t.min()
    return audpc(t, y) / (dur * max_value) if dur > 0 else np.nan


def per_group_spearman(df: pd.DataFrame, group: str, x: str, y: str, min_n: int = 4) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(group):
        d = d[[x, y]].dropna()
        if len(d) >= min_n and d[x].nunique() > 1 and d[y].nunique() > 1:
            rho = stats.spearmanr(d[x], d[y]).statistic
        else:
            rho = np.nan
        rows.append({group: g, "rho": rho, "n_points": len(d)})
    return pd.DataFrame(rows)


def wilcoxon_greater(values) -> dict:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return {"n": len(v), "median": float(np.median(v)) if len(v) else np.nan, "p": np.nan}
    res = stats.wilcoxon(v, alternative="greater")
    return {"n": len(v), "median": float(np.median(v)), "statistic": float(res.statistic),
            "p": float(res.pvalue)}


def mixed_model(df: pd.DataFrame, formula: str, group: str, re_formula: str = "~day"):
    """Linear mixed model via statsmodels (optional dependency). Returns a summary table or None."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("[stats] statsmodels not installed; skipping mixed model")
        return None
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm(formula, df, groups=df[group], re_formula=re_formula).fit(method="lbfgs")
            structure = f"random intercept + slope ({re_formula})"
        except Exception as e:  # noqa: BLE001
            print(f"[stats] random-slope model failed ({e}); refitting with random intercept only")
            fit = smf.mixedlm(formula, df, groups=df[group]).fit(method="lbfgs")
            structure = "random intercept only"
    n_warn = sum("converge" in str(w.message).lower() or "hessian" in str(w.message).lower() for w in caught)
    if n_warn or not getattr(fit, "converged", True):
        print(f"[stats] mixed model ({structure}) has convergence warnings: treat its p-values with caution")
    tab = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues})
    tab.attrs["structure"] = structure
    tab.attrs["converged"] = bool(getattr(fit, "converged", True)) and n_warn == 0
    ci = fit.conf_int()
    tab["ci_lo"], tab["ci_hi"] = ci[0], ci[1]
    return tab
