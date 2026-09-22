#!/usr/bin/env python3
"""
TEXT LEMURS - Embedding Interpretation Script

For each significant PC, this script:
  1. Correlates it with named SEANCE features     -> semantic label
  2. Correlates it with zero-shot domain scores   -> domain label
  3. Finds top/bottom representative texts        -> human-readable examples

Run AFTER embed.py and embed_mixed2.py.

Requirements:
    pip install pandas numpy scipy openpyxl

Usage:
    python text_lemurs_interpret_embeddings.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

import config
EMBEDDING_DIR  = config.out_path('embedding_outputs')
RESULTS_DIR    = config.out_path('results')
SEANCE_PATH    = config.out_path('seance_features.csv')
OUTPUT_DIR     = config.out_path('interpretation_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_EXAMPLES        = 5      # representative texts to show per PC
CORR_THRESHOLD    = 0.10   # minimum |r| to assign a label
MODELS            = ["roberta", "mentalroberta"]

# All named SEANCE features present in seance_features.csv
SEANCE_FEATURES = [
    "Admiration/Awe_GALC", "Amusement_GALC", "Anger_GALC", "Anxiety_GALC",
    "Beingtouched_GALC", "Boredom_GALC", "Compassion_GALC", "Contempt_GALC",
    "Contentment_GALC", "Desperation_GALC", "Disappointment_GALC", "Disgust_GALC",
    "Dissatisfaction_GALC", "Envy_GALC", "Fear_GALC", "Feelinglove_GALC",
    "Gratitude_GALC", "Guilt_GALC", "Happiness_GALC", "Hatred_GALC",
    "Hope_GALC", "Humility_GALC", "Interest/Enthusiasm_GALC", "Irritation_GALC",
    "Jealousy_GALC", "Joy_GALC", "Longing_GALC", "Lust_GALC",
    "Pleasure/Enjoyment_GALC", "Pride_GALC", "Relaxation/Serenity_GALC",
    "Relief_GALC", "Sadness_GALC", "Shame_GALC", "Surprise_GALC",
    "Tension/Stress_GALC", "Positive_GALC", "Negative_GALC",
    "Anger_EmoLex", "Anticipation_EmoLex", "Disgust_EmoLex", "Fear_EmoLex",
    "Joy_EmoLex", "Negative_EmoLex", "Positive_EmoLex", "Sadness_EmoLex",
    "Surprise_EmoLex", "Trust_EmoLex",
    "Valence", "Arousal", "Dominance",
    "pleasantness", "attention", "sensitivity", "aptitude", "polarity",
    "hu_liu_pos_nwords", "hu_liu_neg_nwords", "hu_liu_prop",
    "Positiv_GI", "Negativ_GI", "Strong_GI", "Power_GI", "Active_GI",
    "Pleasur_GI", "Feel_GI", "Emot_GI", "Virtue_GI", "Academ_GI",
    "Work_GI", "Social_GI", "Think_GI", "Know_GI", "Goal_GI",
    "Means_GI", "Complet_GI", "Quality_GI", "Need_GI",
    "Powtot_Lasswell", "Enltot_Lasswell", "Afftot_Lasswell",
    "Skltot_Lasswell", "Wlbtot_Lasswell",
]


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def safe_corr(x: pd.Series, y: pd.Series) -> tuple:
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return 0.0, 1.0
    try:
        r, p = pearsonr(x[mask].astype(float), y[mask].astype(float))
        return round(float(r), 4), round(float(p), 4)
    except Exception:
        return 0.0, 1.0


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

def load_data() -> tuple:
    log("Loading merged embedding file...")
    merged_path = os.path.join(EMBEDDING_DIR, "text_lemurs_embeddings_merged.csv")
    df_merged   = pd.read_csv(merged_path)
    log(f"  Merged shape : {df_merged.shape}")

    log("Loading SEANCE features...")
    df_seance = pd.read_csv(SEANCE_PATH)
    log(f"  SEANCE shape : {df_seance.shape}")

    log("Loading significant results...")
    sig_path  = os.path.join(RESULTS_DIR, "embedding_model_results_significant.csv")
    df_sig    = pd.read_csv(sig_path)

    # Merge merged + seance on record_id + Week so everything is aligned
    seance_cols = ["record_id", "Week", "concern_text_cleaned"] + \
                  [f for f in SEANCE_FEATURES if f in df_seance.columns]
    df_combined = df_merged.merge(
        df_seance[seance_cols],
        on=["record_id", "Week"],
        how="left",
        suffixes=("", "_seance")
    )
    log(f"  Combined shape: {df_combined.shape}")

    return df_combined, df_sig


# -----------------------------------------------------------------------------
# STEP 1 - Correlate PCs with SEANCE features
# -----------------------------------------------------------------------------

def correlate_with_seance(df: pd.DataFrame, pc_cols: list,
                          model_key: str) -> pd.DataFrame:
    log(f"\n  Correlating {len(pc_cols)} PCs with {len(SEANCE_FEATURES)} SEANCE features...")

    rows = []
    available_seance = [f for f in SEANCE_FEATURES if f in df.columns]

    for pc in pc_cols:
        if pc not in df.columns:
            continue
        for feat in available_seance:
            r, p = safe_corr(df[pc], df[feat])
            rows.append({
                "model":         model_key,
                "pc":            pc,
                "seance_feature": feat,
                "r":             r,
                "p":             p,
                "abs_r":         abs(r)
            })

    corr_df = pd.DataFrame(rows)
    return corr_df


def get_top_seance_label(corr_df: pd.DataFrame, pc: str) -> dict:
    """Get the strongest SEANCE correlation for a given PC."""
    pc_corrs = corr_df[corr_df["pc"] == pc].copy()
    if pc_corrs.empty:
        return {"seance_label": "none", "seance_r": 0.0, "seance_direction": ""}

    best = pc_corrs.loc[pc_corrs["abs_r"].idxmax()]
    direction = "+" if best["r"] > 0 else "-"
    if best["abs_r"] < CORR_THRESHOLD:
        label = "weak/unclear"
    else:
        label = best["seance_feature"]

    return {
        "seance_label":     label,
        "seance_r":         best["r"],
        "seance_direction": direction
    }


# -----------------------------------------------------------------------------
# STEP 2 - Correlate PCs with domain scores
# -----------------------------------------------------------------------------

def correlate_with_domains(df: pd.DataFrame, pc_cols: list,
                           model_key: str) -> pd.DataFrame:
    domain_cols = [c for c in df.columns
                   if c.startswith("domain_")
                   and c not in ("top_domain", "top_score")]

    if not domain_cols:
        log("  No domain columns found - skipping domain correlation")
        return pd.DataFrame()

    log(f"  Correlating {len(pc_cols)} PCs with {len(domain_cols)} domain scores...")

    rows = []
    for pc in pc_cols:
        if pc not in df.columns:
            continue
        for dom in domain_cols:
            r, p = safe_corr(df[pc], df[dom])
            rows.append({
                "model":          model_key,
                "pc":             pc,
                "domain":         dom.replace("domain_", ""),
                "r":              r,
                "p":              p,
                "abs_r":          abs(r)
            })

    return pd.DataFrame(rows)


def get_top_domain_label(domain_corr_df: pd.DataFrame, pc: str) -> dict:
    if domain_corr_df.empty:
        return {"domain_label": "none", "domain_r": 0.0}

    pc_corrs = domain_corr_df[domain_corr_df["pc"] == pc].copy()
    if pc_corrs.empty:
        return {"domain_label": "none", "domain_r": 0.0}

    best = pc_corrs.loc[pc_corrs["abs_r"].idxmax()]
    label = best["domain"] if best["abs_r"] >= CORR_THRESHOLD else "weak/unclear"

    return {
        "domain_label": label,
        "domain_r":     best["r"]
    }


# -----------------------------------------------------------------------------
# STEP 3 - Representative texts
# -----------------------------------------------------------------------------

def get_representative_texts(df: pd.DataFrame, pc: str,
                              n: int = N_EXAMPLES) -> dict:
    col = "concern_text_cleaned"
    if pc not in df.columns or col not in df.columns:
        return {"top_texts": [], "bottom_texts": []}

    valid = df[[pc, col]].dropna()
    if len(valid) < n * 2:
        return {"top_texts": [], "bottom_texts": []}

    valid = valid.sort_values(pc)

    bottom_texts = valid.head(n)[col].tolist()
    top_texts    = valid.tail(n)[col].tolist()

    return {
        "top_texts":    top_texts,
        "bottom_texts": bottom_texts
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    log("=" * 60)
    log("TEXT LEMURS - Embedding Interpretation")
    log("=" * 60)

    df_combined, df_sig = load_data()

    all_interpretation_rows = []
    all_seance_corr_rows    = []
    all_domain_corr_rows    = []

    for model_key in MODELS:
        log(f"\n{'='*60}")
        log(f"  Model: {model_key.upper()}")
        log(f"{'='*60}")

        sig_pcs = df_sig[
            (df_sig["model"] == f"{model_key}_pca_only") &
            (~df_sig["term"].isin(["Intercept", "Group Var"]))
        ]["term"].unique().tolist()

        log(f"  Significant PCs: {len(sig_pcs)}")

        if not sig_pcs:
            log("  No significant PCs found - skipping")
            continue

        seance_corr = correlate_with_seance(df_combined, sig_pcs, model_key)
        all_seance_corr_rows.append(seance_corr)

        domain_corr = correlate_with_domains(df_combined, sig_pcs, model_key)
        all_domain_corr_rows.append(domain_corr)

        log(f"\n  Building interpretation table...")

        for pc in sig_pcs:
            pc_outcomes = df_sig[
                (df_sig["model"] == f"{model_key}_pca_only") &
                (df_sig["term"] == pc)
            ][["outcome", "coef", "p_value"]].to_dict("records")

            outcomes_str = "; ".join([
                f"{r['outcome']} (β={r['coef']:.4f}, p={r['p_value']:.4f})"
                for r in pc_outcomes
            ])

            seance_info = get_top_seance_label(seance_corr, pc)
            domain_info = get_top_domain_label(domain_corr, pc)
            texts       = get_representative_texts(df_combined, pc)

            row = {
                "model":            model_key,
                "pc":               pc,
                "n_sig_outcomes":   len(pc_outcomes),
                "sig_outcomes":     outcomes_str,
                "seance_label":     seance_info["seance_label"],
                "seance_r":         seance_info["seance_r"],
                "domain_label":     domain_info["domain_label"],
                "domain_r":         domain_info["domain_r"],
                "top_texts":        " | ".join(texts["top_texts"]),
                "bottom_texts":     " | ".join(texts["bottom_texts"]),
            }
            all_interpretation_rows.append(row)

    # -- Save outputs ----------------------------------------------------------

    # Main interpretation table
    interp_df = pd.DataFrame(all_interpretation_rows)
    interp_path = os.path.join(OUTPUT_DIR, "pc_interpretation_table.csv")
    interp_df.to_csv(interp_path, index=False)
    log(f"\nInterpretation table saved: {interp_path}  {interp_df.shape}")

    # Full SEANCE correlations
    if all_seance_corr_rows:
        seance_corr_df = pd.concat(all_seance_corr_rows, ignore_index=True)
        seance_path = os.path.join(OUTPUT_DIR, "pc_seance_correlations.csv")
        seance_corr_df.to_csv(seance_path, index=False)
        log(f"SEANCE correlations saved : {seance_path}  {seance_corr_df.shape}")

    # Full domain correlations
    if all_domain_corr_rows:
        domain_corr_df = pd.concat(all_domain_corr_rows, ignore_index=True)
        domain_path = os.path.join(OUTPUT_DIR, "pc_domain_correlations.csv")
        domain_corr_df.to_csv(domain_path, index=False)
        log(f"Domain correlations saved : {domain_path}  {domain_corr_df.shape}")

    # -- Print summary ----------------------------------------------------------
    log(f"\n{'='*60}")
    log("  INTERPRETATION SUMMARY")
    log(f"{'='*60}")

    for model_key in MODELS:
        model_rows = interp_df[interp_df["model"] == model_key]
        if model_rows.empty:
            continue
        log(f"\n  {model_key.upper()}")

        clear = (model_rows["seance_label"] != "weak/unclear") & \
                (model_rows["seance_label"] != "none")
        log(f"  PCs with clear SEANCE label : {clear.sum()} / {len(model_rows)}")

        label_counts = model_rows[clear]["seance_label"].value_counts().head(10)
        log(f"  Top SEANCE labels:")
        for label, count in label_counts.items():
            log(f"    {count:>3}  {label}")

        dom_clear = (model_rows["domain_label"] != "weak/unclear") & \
                    (model_rows["domain_label"] != "none")
        dom_counts = model_rows[dom_clear]["domain_label"].value_counts().head(5)
        if not dom_counts.empty:
            log(f"  Top domain labels:")
            for label, count in dom_counts.items():
                log(f"    {count:>3}  {label}")

    log(f"\nAll outputs saved to: ./{OUTPUT_DIR}/")
    log("Done.")


if __name__ == "__main__":
    main()
