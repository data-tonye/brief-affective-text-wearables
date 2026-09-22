#!/usr/bin/env python3
"""
TEXT LEMURS - Variance Decomposition + Dominance Analysis (Embeddings)
=======================================================================
Block structure mirrors SEANCE decomposition for three-way comparison:

  Block 0: outcome ~ 1 + (1 | record_id)
  Block 1: outcome ~ nwords_z + (1 | record_id)
  Block 2: outcome ~ nwords_z + Week_z + (1 | record_id)
  Block 3: outcome ~ nwords_z + Week_z + all PCs (127 RoBERTa / 100 MentalRoBERTa) + (1 | record_id)

Note: concern_present omitted - constant = 1 on concern-present rows.
Dominance analysis uses the significant PCs only, capped at MAX_DOMINANCE_PCS (10) per outcome for runtime
(2^10 subsets); the variance decomposition uses the full PC set with no significance filtering.
Marginal R2 = variance of the fixed-effect linear predictor only (r2_tools.r2_parts).
SEANCE reference values read from file - not hardcoded.

Inputs (all relative to working directory):
  embedding_outputs/text_lemurs_embeddings_merged.csv
  seance_features.csv
  results/embedding_model_results_significant.csv
  variance_decomposition.csv (written by seance_variance_and_dominance.py) - SEANCE delta R2

Outputs -> variance_dominance_outputs/
  variance_decomposition.csv
  dominance_analysis.csv
  three_way_comparison.csv

Requirements: pip install pandas numpy statsmodels
"""

import os, warnings, itertools, time, datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

_script_start = time.time()

def elapsed():
    """Return elapsed time since script start as a readable string."""
    secs = time.time() - _script_start
    return str(datetime.timedelta(seconds=int(secs)))

def log_time(msg=""):
    print(f"[{elapsed()}] {msg}", flush=True)

# -- CONFIG --------------------------------------------------------------------
import config
import r2_tools
EMBEDDING_DIR   = config.out_path('embedding_outputs')
SEANCE_PATH     = config.out_path('seance_features.csv')
RESULTS_DIR     = config.out_path('results')
SEANCE_DECOMP   = config.out_path('variance_decomposition.csv')
OUTPUT_DIR      = config.out_path('variance_dominance_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOMES = [
    "sleep_total_hrs",
    "sleep_efficiency",
    "sleep_rem_hrs",
    "sleep_deep_hrs",
    "sleep_rmssd",
    "sleep_onset_latency_min",
    "act_steps",
    "act_met_min_medium",
    "act_met_min_high",
]

OUTCOME_LABELS = {
    "sleep_total_hrs":         "Sleep Duration (hrs)",
    "sleep_efficiency":        "Sleep Efficiency (%)",
    "sleep_rem_hrs":           "REM Sleep (hrs)",
    "sleep_deep_hrs":          "Deep Sleep (hrs)",
    "sleep_rmssd":             "RMSSD (ms)",
    "sleep_onset_latency_min": "Sleep Onset Latency (min)",
    "act_steps":               "Steps / day",
    "act_met_min_medium":      "MET-min Medium",
    "act_met_min_high":        "MET-min High",
}

MODELS              = ["roberta", "mentalroberta"]
MAX_DOMINANCE_PCS   = 10

# -- HELPERS -------------------------------------------------------------------
def log(msg=""):
    print(f"[{elapsed()}] {msg}", flush=True)

marginal_r2 = r2_tools.marginal_r2
fit = r2_tools.fit_ml
sig_pcs = r2_tools.sig_pcs

def icc_from_null(result):
    try:
        return r2_tools.r2_parts(result)[2]
    except Exception:
        return np.nan

def conditional_r2(result):
    try:
        return r2_tools.r2_parts(result)[1]
    except Exception:
        return np.nan

# -- LOAD DATA -----------------------------------------------------------------
def load_data():
    log("=" * 55)
    log("Loading data")
    log("=" * 55)

    merged = pd.read_csv(
        os.path.join(EMBEDDING_DIR, "text_lemurs_embeddings_merged.csv"))
    nwords = pd.read_csv(SEANCE_PATH)[["record_id", "Week", "nwords"]]
    df     = merged.merge(nwords, on=["record_id", "Week"], how="left")
    df     = df[df["concern_present"] == 1].copy().reset_index(drop=True)

    df["Week_z"]   = (df["Week"]   - df["Week"].mean())   / df["Week"].std()
    df["nwords_z"] = (df["nwords"] - df["nwords"].mean()) / df["nwords"].std()
    df["nwords_z"] = df["nwords_z"].fillna(0)

    log(f"  Rows (concern-present): {len(df):,} | Subjects: {df['record_id'].nunique()}")
    return df

# -- VARIANCE DECOMPOSITION ----------------------------------------------------
def run_variance_decomposition(df):
    log("\n" + "=" * 55)
    log("VARIANCE DECOMPOSITION")
    log("=" * 55)

    rows = []
    for model_key in MODELS:
        log(f"\n  Model: {model_key.upper()}")
        pcs = r2_tools.all_pcs(df, model_key)          # full PC set, no significance filtering
        for outcome in OUTCOMES:
            if outcome not in df.columns:
                log(f"    {outcome}: column not found - skipping")
                continue

            data = df[[outcome, "record_id", "Week_z", "nwords_z"] + pcs].dropna()
            if len(data) < 50:
                continue

            log(f"    {outcome}: {len(data)} obs, {len(pcs)} PCs")
            t0 = time.time()

            m0 = fit(f"{outcome} ~ 1", data)
            if m0 is None:
                continue

            r2_0 = marginal_r2(m0)
            icc  = icc_from_null(m0)

            m1   = fit(f"{outcome} ~ nwords_z", data)
            r2_1 = marginal_r2(m1) if m1 else np.nan

            m2   = fit(f"{outcome} ~ nwords_z + Week_z", data)
            r2_2 = marginal_r2(m2) if m2 else np.nan

            pc_str = " + ".join(pcs)
            m3     = fit(f"{outcome} ~ nwords_z + Week_z + {pc_str}", data)
            r2_3   = marginal_r2(m3) if m3 else np.nan
            cond   = conditional_r2(m3) if m3 else np.nan

            d1 = r2_1 - r2_0
            d2 = r2_2 - r2_1
            d3 = r2_3 - r2_2
            pct = round(d3 / r2_3 * 100, 2) if (r2_3 and r2_3 > 0
                                                  and not np.isnan(d3)) else np.nan

            rows.append({
                "model":              model_key,
                "outcome":            outcome,
                "outcome_label":      OUTCOME_LABELS.get(outcome, outcome),
                "n_obs":              len(data),
                "n_subjects":         data["record_id"].nunique(),
                "n_pcs":              len(pcs),
                "n_sig_pcs":          len(sig_pcs(model_key, outcome)),
                "ICC":                icc,
                "R2_null":            r2_0,
                "R2_covariates":      r2_1,
                "R2_semester":        r2_2,
                "R2_language":        r2_3,
                "deltaR2_covariates": d1,
                "deltaR2_semester":   d2,
                "deltaR2_language":   d3,
                "pct_language_of_full_R2": pct,
                "conditional_R2":     cond,
            })
            log(f"      done in {time.time()-t0:.1f}s")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "variance_decomposition.csv"), index=False)
    log(f"\n  Saved: variance_decomposition.csv  {out.shape}")
    log(out[["model","outcome","ICC","deltaR2_semester",
             "deltaR2_language","pct_language_of_full_R2"]].to_string(index=False))
    return out

# -- DOMINANCE ANALYSIS -------------------------------------------------------
def run_dominance_analysis(df):
    log("\n" + "=" * 55)
    log(f"DOMINANCE ANALYSIS  (max {MAX_DOMINANCE_PCS} PCs = "
        f"{2**MAX_DOMINANCE_PCS} subsets per outcome)")
    log("=" * 55)

    jobs = []
    for model_key in MODELS:
        for outcome in OUTCOMES:
            if outcome not in df.columns:
                continue
            pcs = r2_tools.top_sig_pcs(model_key, outcome, MAX_DOMINANCE_PCS)
            if not pcs:
                continue
            data = df[[outcome, "record_id", "Week_z", "nwords_z"] + pcs].dropna()
            if len(data) < 50:
                continue
            jobs.append((outcome, pcs, ["nwords_z", "Week_z"], data, "ml", (model_key, outcome)))
    log(f"  {len(jobs)} model x outcome jobs (run in parallel)")

    rows = []
    for (model_key, outcome), dom, r2_base, r2_full, n_obs in r2_tools.parallel_dominance(jobs):
        total = r2_full - r2_base
        for rank, (pc, (avg_dr2, avg_bud)) in enumerate(
                sorted(dom.items(), key=lambda kv: -kv[1][0]), 1):
            rows.append({
                "model":             model_key,
                "outcome":           outcome,
                "outcome_label":     OUTCOME_LABELS.get(outcome, outcome),
                "total_R2_language": total,
                "rank":              rank,
                "predictor":         pc,
                "avg_deltaR2":       avg_dr2,
                "avg_deltaR2_budescu": avg_bud,
                "pct_of_explained":  (avg_dr2 / total * 100) if total > 0 else np.nan,
                "n_obs":             n_obs,
            })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "dominance_analysis.csv"), index=False)
    log(f"\n  Saved: dominance_analysis.csv  {out.shape}")

    if not out.empty:
        log("\n  TOP DOMINANT PC PER OUTCOME:")
        top = out[out["rank"]==1][["model","outcome","predictor",
                                    "avg_deltaR2","pct_of_explained"]]
        log(top.to_string(index=False))
    return out

# -- THREE-WAY COMPARISON ------------------------------------------------------
def build_three_way(vardecomp_df):
    log("\n" + "=" * 55)
    log("THREE-WAY COMPARISON TABLE")
    log("=" * 55)

    # Read SEANCE values from file - not hardcoded
    if not os.path.exists(SEANCE_DECOMP):
        log(f"  WARNING: SEANCE decomposition file not found at {SEANCE_DECOMP}")
        log("  Run decomposition.py first.")
        seance_lookup = {}
    else:
        seance_df = pd.read_csv(SEANCE_DECOMP)
        seance_lookup = {
            row["outcome_var"]: {
                "semester": row["delta_semester"],
                "language": row["delta_language"],
            }
            for _, row in seance_df.iterrows()
        }
        log(f"  SEANCE reference values loaded from: {SEANCE_DECOMP}")
        log(f"  Outcomes in SEANCE file: {list(seance_lookup.keys())}")

    rows = []
    for _, row in vardecomp_df.iterrows():
        label  = row["outcome_label"]
        seance = seance_lookup.get(row["outcome"], {})
        rows.append({
            "outcome":                  label,
            "model":                    row["model"],
            "ICC":                      row["ICC"],
            "n_obs":                    row["n_obs"],
            "deltaR2_semester":         row["deltaR2_semester"],
            "deltaR2_language_embed":   row["deltaR2_language"],
            "deltaR2_language_seance":  seance.get("language", np.nan),
            "pct_language_of_full_R2":  row["pct_language_of_full_R2"],
            "conditional_R2":           row["conditional_R2"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "three_way_comparison.csv"), index=False)
    log(f"\n  Saved: three_way_comparison.csv  {out.shape}")
    log(out.to_string(index=False))
    return out

# -- MAIN ----------------------------------------------------------------------
def main():
    log("=" * 55)
    log("TEXT LEMURS - Embedding Variance Decomposition + Dominance")
    log("=" * 55)

    df           = load_data()
    vardecomp_df = run_variance_decomposition(df)
    dom_df       = run_dominance_analysis(df)
    build_three_way(vardecomp_df)

    log("\n" + "=" * 55)
    log("DONE - outputs in ./" + OUTPUT_DIR + "/")
    log("  variance_decomposition.csv")
    log("  dominance_analysis.csv")
    log("  three_way_comparison.csv")
    log(f"Total runtime: {elapsed()}")
    log("=" * 55)

if __name__ == "__main__":
    main()