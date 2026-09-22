"""
Shared helpers for the variance decomposition and dominance analyses
(decomposition.py, embed_domanalysis.py, bootstrap_embedding_comparison.py).

Marginal R2 follows Nakagawa & Schielzeth (2013) for a random-intercept model:

    sigma2_f = var(X @ beta_hat)        fixed-effect linear predictor only
    sigma2_u = random-intercept variance
    sigma2_e = residual variance
    R2_marginal    = sigma2_f / (sigma2_f + sigma2_u + sigma2_e)
    R2_conditional = (sigma2_f + sigma2_u) / (sigma2_f + sigma2_u + sigma2_e)
    ICC            = sigma2_u / (sigma2_u + sigma2_e)

Note: statsmodels' MixedLMResults.fittedvalues includes the random-intercept predictions, so
var(fittedvalues) is NOT the fixed-effect variance and must not be used for marginal R2.
"""
import itertools
import multiprocessing
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config

warnings.filterwarnings("ignore")

MODELS = ["roberta", "mentalroberta"]
MAX_DOMINANCE_PCS = 10


# --------------------------------------------------------------------------- R2 / ICC
def r2_parts(res):
    """(marginal R2, conditional R2, ICC of this model) from a fitted MixedLM (random intercept)."""
    X = np.asarray(res.model.exog)
    b = np.asarray(res.fe_params)
    vf = float(np.var(X @ b))
    vu = float(res.cov_re.iloc[0, 0])
    ve = float(res.scale)
    tot = vf + vu + ve
    return vf / tot, (vf + vu) / tot, vu / (vu + ve)


def marginal_r2(res):
    if res is None:
        return np.nan
    try:
        return r2_parts(res)[0]
    except Exception:
        return np.nan


# --------------------------------------------------------------------------- model fitting
def fit_reml(formula, data):
    """REML, L-BFGS (used for the SEANCE models)."""
    try:
        return smf.mixedlm(formula, data, groups=data["record_id"]).fit(
            reml=True, method="lbfgs", disp=False)
    except Exception:
        return None


def fit_ml(formula, data):
    """ML with optimiser fallbacks (used for the embedding models)."""
    for method in ["lbfgs", "bfgs", "cg", "nm"]:
        try:
            return smf.mixedlm(formula, data, groups=data["record_id"]).fit(
                reml=False, method=method, maxiter=200, disp=False)
        except Exception:
            continue
    return None


FITTERS = {"reml": fit_reml, "ml": fit_ml}


# --------------------------------------------------------------------------- dominance analysis
def dominance(outcome, preds, base_terms, data, fitter):
    """Average marginal-R2 contribution of each predictor over all subsets of the other predictors.

    avg_flat    : mean over every (subset, predictor) pair (the definition in the Methods).
    avg_budescu : mean of the per-subset-size means (Budescu 1993 weighting), reported for reference.
    Each of the 2^k distinct models is fitted once and cached.
    Returns ({predictor: (avg_flat, avg_budescu)}, r2_base, r2_full).
    """
    base = " + ".join(base_terms) if base_terms else "1"
    cache = {}

    def r2(sub):
        key = frozenset(sub)
        if key not in cache:
            terms = base + (" + " + " + ".join(sorted(key, key=preds.index)) if key else "")
            cache[key] = marginal_r2(fitter(f"{outcome} ~ {terms}", data))
        return cache[key]

    flat = {p: [] for p in preds}
    bysize = {p: {} for p in preds}
    for size in range(len(preds)):
        for subset in itertools.combinations(preds, size):
            rb = r2(subset)
            for p in preds:
                if p not in subset:
                    ra = r2(subset + (p,))
                    if not (np.isnan(rb) or np.isnan(ra)):
                        flat[p].append(ra - rb)
                        bysize[p].setdefault(size, []).append(ra - rb)
    out = {}
    for p in preds:
        f = float(np.mean(flat[p])) if flat[p] else 0.0
        b = float(np.mean([np.mean(v) for v in bysize[p].values()])) if bysize[p] else 0.0
        out[p] = (f, b)
    return out, cache[frozenset()], cache[frozenset(preds)]


def _dominance_job(args):
    outcome, preds, base_terms, data, fitter_name, tag = args
    dom, r2_base, r2_full = dominance(outcome, preds, base_terms, data, FITTERS[fitter_name])
    return tag, dom, r2_base, r2_full, len(data)


def parallel_dominance(jobs, processes=None):
    """Run a list of dominance jobs (outcome, preds, base_terms, data, fitter_name, tag) in parallel.
    Uses the 'spawn' start method: forking a process that has already used the BLAS library crashes on macOS.
    Scripts that call this must be guarded by `if __name__ == "__main__":`."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes or max(1, (multiprocessing.cpu_count() or 2) - 1)) as pool:
        return pool.map(_dominance_job, jobs, chunksize=1)


# --------------------------------------------------------------------------- data loading
def zscore(s):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 0 else s * 0


def load_seance_frame():
    """oura_aggregated + seance_features (inner merge), concern-present rows only."""
    seance = pd.read_csv(config.out_path("seance_features.csv")).drop_duplicates(["record_id", "Week"], keep="first")
    oura = pd.read_csv(config.out_path("oura_aggregated.csv")).drop_duplicates(["record_id", "Week"], keep="first")
    drop = ["concern_present", "concern_text_cleaned", "original_turns", "language", "timestamp", "original_index"]
    df = oura.merge(seance.drop(columns=[c for c in drop if c in seance.columns]),
                    on=["record_id", "Week"], how="inner")
    return df[df["concern_present"] == 1].copy()


def load_embed_frame():
    """Merged embedding file + nwords, concern-present rows, Week and nwords z-scored within that subset."""
    merged = pd.read_csv(config.out_path("embedding_outputs/text_lemurs_embeddings_merged.csv"))
    nw = pd.read_csv(config.out_path("seance_features.csv"))[["record_id", "Week", "nwords"]]
    df = merged.merge(nw, on=["record_id", "Week"], how="left")
    df = df[df["concern_present"] == 1].copy().reset_index(drop=True)
    df["Week_z"] = (df["Week"] - df["Week"].mean()) / df["Week"].std()
    df["nwords_z"] = (df["nwords"] - df["nwords"].mean()) / df["nwords"].std()
    df["nwords_z"] = df["nwords_z"].fillna(0)
    return df


_SIG = None
def _sig_table():
    global _SIG
    if _SIG is None:
        _SIG = pd.read_csv(config.out_path("results/embedding_model_results_significant.csv"))
    return _SIG


def sig_pcs(model_key, outcome):
    """PCs significant (p<.05) in the joint embedding model for this outcome."""
    d = _sig_table()
    return d[(d["model"] == f"{model_key}_pca_only") & (d["outcome"] == outcome) &
             (~d["term"].isin(["Intercept", "Group Var", "Week_z"]))]["term"].tolist()


def top_sig_pcs(model_key, outcome, cap=MAX_DOMINANCE_PCS):
    """Significant PCs, trimmed to the `cap` with the largest |coefficient| (dominance analysis only)."""
    pcs = sig_pcs(model_key, outcome)
    if len(pcs) > cap:
        d = _sig_table()
        t = d[(d["model"] == f"{model_key}_pca_only") & (d["outcome"] == outcome) &
              (~d["term"].isin(["Intercept", "Group Var", "Week_z"]))].copy()
        t["abs_coef"] = t["coef"].abs()
        pcs = t.nlargest(cap, "abs_coef")["term"].tolist()
    return pcs


def all_pcs(df, model_key):
    """Every PCA component of the embedding model (127 RoBERTa / 100 MentalRoBERTa)."""
    return [c for c in df.columns if c.startswith(f"{model_key}_PC")]
