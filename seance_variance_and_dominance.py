#!/usr/bin/env python3
"""
TEXT LEMURS - Variance Decomposition & Dominance Analysis
==========================================================
Answers: "What percentage of outcome variance does each predictor explain?"

Two complementary methods:

1. VARIANCE DECOMPOSITION (Nakagawa & Schielzeth 2013)
   Sequentially adds predictor blocks and computes marginal R² (fixed-effects variance only):
     Block 0: Person (random intercept only)
     Block 1: + Covariates (concern_present, nwords_z)
     Block 2: + Semester (Week_z)
     Block 3: + Linguistic features
   ΔR² for each block = unique variance explained by that block.

2. DOMINANCE ANALYSIS (Budescu 1993, Azen & Budescu 2003)
   Tests all possible predictor subsets to compute each predictor's
   average marginal R² contribution - the fairest way to rank predictors
   when they are correlated.

Design notes:
  1. Restricted to concern-present rows only (n=3,073) for comparability
     with the embedding models.
  2. All 9 outcomes are included in the variance decomposition so the SEANCE
     ΔR² column is complete for the three-way comparison table.

Outputs:
  - variance_decomposition.csv    (R² by block, per outcome)
  - dominance_analysis.csv        (% contribution per predictor, per outcome)
  - variance_summary.txt          (readable report)

Run from the folder containing seance_features.csv and oura_aggregated.csv.
Requires: pandas, numpy, statsmodels
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.formula.api import mixedlm
from itertools import combinations

warnings.filterwarnings('ignore')

# -- CONFIG ------------------------------------------------------------------
import config
import r2_tools
SEANCE_FILE   = config.out_path('seance_features.csv')
OURA_FILE     = config.out_path('oura_aggregated.csv')
DECOMP_OUT    = config.out_path('variance_decomposition.csv')
DOMINANCE_OUT = config.out_path('dominance_analysis.csv')
SUMMARY_OUT   = config.out_path('variance_summary.txt')

# Linguistic predictors for dominance analysis
# Chosen for: coverage >15%, theoretical relevance, raw significance in models
LINGUISTIC_PREDICTORS = [
    'Week_z',           # semester timing - included as a predictor so we can
                        # rank it against language in the dominance table
    'Strong_GI_z',
    'Negative_EmoLex_z',
    'Sadness_EmoLex_z',
    'Fear_EmoLex_z',
    'Valence_z',
    'Academ_GI_z',
    'Work_GI_z',
]

# All 9 outcomes are included
# SEANCE ΔR² will be near-zero for sleep_total_hrs, sleep_deep_hrs, sleep_rem_hrs
# but these are needed for the complete variance decomposition table
OUTCOMES = {
    'sleep_rmssd':             'RMSSD (ms)',
    'sleep_efficiency':        'Sleep Efficiency (%)',
    'sleep_onset_latency_min': 'Sleep Onset Latency (min)',
    'sleep_total_hrs':         'Sleep Duration (hrs)',
    'sleep_deep_hrs':          'Deep Sleep (hrs)',
    'sleep_rem_hrs':           'REM Sleep (hrs)',
    'act_steps':               'Steps/day',
    'act_met_min_medium':      'MET-min Medium',
    'act_met_min_high':        'MET-min High',
}

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

def section(title):
    log()
    log('=' * 65)
    log(f'  {title}')
    log('=' * 65)

# -- LOAD & MERGE -------------------------------------------------------------
section('LOADING & MERGING')

seance = pd.read_csv(SEANCE_FILE)
oura   = pd.read_csv(OURA_FILE)
seance = seance.drop_duplicates(subset=['record_id','Week'], keep='first')
oura   = oura.drop_duplicates(subset=['record_id','Week'], keep='first')

drop_cols = ['concern_present','concern_text_cleaned','original_turns',
             'language','timestamp','original_index']
seance_clean = seance.drop(columns=[c for c in drop_cols if c in seance.columns])
df = oura.merge(seance_clean, on=['record_id','Week'], how='inner')
log(f'  Merged (all rows): {len(df):,} rows | {df["record_id"].nunique()} participants')
df_all = df.copy()   # all waves, used for the all-wave ICC

# Restrict to concern-present rows so all three methods use the same waves
# (embeddings exist only for concern-present rows)
df = df[df['concern_present'] == 1].copy()
log(f'  After restricting to concern-present: {len(df):,} rows | {df["record_id"].nunique()} participants')
log(f'  (concern_present covariate omitted from formulas - constant = 1 in this subset)')

# -- STANDARDIZE --------------------------------------------------------------
section('STANDARDIZING')

def zscore(s):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 0 else s * 0

# Standardize everything we need
std_cols = ['Week', 'Strong_GI', 'Negative_EmoLex', 'Sadness_EmoLex',
            'Fear_EmoLex', 'Valence', 'Academ_GI', 'Work_GI', 'nwords']

for col in std_cols:
    if col in df.columns:
        df[f'{col}_z'] = zscore(df[col])
    else:
        log(f'  Warning: {col} not found in data')

available_predictors = [p for p in LINGUISTIC_PREDICTORS if p in df.columns]
log(f'  Predictors available: {available_predictors}')

# -- CORE FUNCTIONS ------------------------------------------------------------

def nakagawa_r2(formula, data):
    """
    Marginal R2, conditional R2 and ICC of a random-intercept model (Nakagawa & Schielzeth 2013).
    Marginal R2 uses only the variance of the fixed-effect linear predictor (see r2_tools.r2_parts).
    The third value is the ICC of the fitted model (for the null model: the intraclass correlation).
    """
    model = r2_tools.fit_reml(formula, data)
    if model is None:
        return np.nan, np.nan, np.nan
    try:
        return r2_tools.r2_parts(model)
    except Exception:
        return np.nan, np.nan, np.nan


BUDESCU = {}   # size-weighted averages, stored per outcome for reference

def dominance_analysis(outcome_var, predictors, covariates, data):
    """
    Average marginal R2 of each predictor across all subsets of the other predictors.
    Covariates are always included (not competed over).
    Returns {predictor: avg_delta_r2}; size-weighted (Budescu) averages are kept in BUDESCU.
    """
    dom, _, _ = r2_tools.dominance(outcome_var, list(predictors), list(covariates), data, r2_tools.fit_reml)
    BUDESCU[outcome_var] = {p: v[1] for p, v in dom.items()}
    return {p: v[0] for p, v in dom.items()}


# -- VARIANCE DECOMPOSITION ----------------------------------------------------
section('VARIANCE DECOMPOSITION BY BLOCK')
log('  Nakagawa & Schielzeth (2013) marginal R²')
log('  Sample: concern-present rows only (n=3,073)')
log('  Block 0: Person random intercept')
log('  Block 1: + Covariates (nwords_z)')
log('  Block 2: + Semester (Week_z)')
log('  Block 3: + Linguistic features')
log('  Note: concern_present omitted as covariate (constant=1 in this subset)')

decomp_results = []

# Linguistic features for block 3 (exclude Week_z - that is block 2)
ling_block = [p for p in available_predictors if p != 'Week_z']

for outcome_var, outcome_label in OUTCOMES.items():
    if outcome_var not in df.columns:
        log(f'\n  SKIP {outcome_label} - column not found')
        continue

    cols_needed = [outcome_var, 'record_id', 'Week_z', 'nwords_z'] + ling_block
    sub = df[[c for c in cols_needed if c in df.columns]].dropna()

    if len(sub) < 100 or sub['record_id'].nunique() < 30:
        log(f'\n  SKIP {outcome_label} - insufficient data ({len(sub)} rows)')
        continue

    # Block 0: null (person only)
    r2_0m, r2_0c, icc = nakagawa_r2(f'{outcome_var} ~ 1', sub)
    # Block 1: + covariates (nwords_z only - concern_present constant in this subset)
    r2_1m, r2_1c, _   = nakagawa_r2(f'{outcome_var} ~ nwords_z', sub)
    # Block 2: + Week_z
    r2_2m, r2_2c, _   = nakagawa_r2(f'{outcome_var} ~ nwords_z + Week_z', sub)
    # Block 3: + linguistic features
    ling_available = [p for p in ling_block if p in sub.columns]
    ling_str       = ' + '.join(ling_available)
    r2_3m, r2_3c, _   = nakagawa_r2(
        f'{outcome_var} ~ nwords_z + Week_z + {ling_str}', sub)

    delta_covars   = r2_1m - r2_0m
    delta_semester = r2_2m - r2_1m
    delta_language = r2_3m - r2_2m if not np.isnan(r2_3m) else np.nan

    decomp_results.append({
        'outcome':             outcome_label,
        'outcome_var':         outcome_var,
        'icc':                 icc,
        'icc_all_waves':       nakagawa_r2(f'{outcome_var} ~ 1', df_all[[outcome_var, 'record_id']].dropna())[2],
        'r2_null_marginal':    r2_0m,
        'r2_covar_marginal':   r2_1m,
        'r2_week_marginal':    r2_2m,
        'r2_full_marginal':    r2_3m,
        'r2_full_conditional': r2_3c,
        'delta_covariates':    delta_covars,
        'delta_semester':      delta_semester,
        'delta_language':      delta_language,
        'n_obs':               len(sub),
        'n_participants':      sub['record_id'].nunique(),
    })

    log(f'\n  {outcome_label}')
    log(f'    ICC (person variance):    {icc:.3f}')
    log(f'    Block 1 ΔR² (covariates): {delta_covars:.4f}')
    log(f'    Block 2 ΔR² (semester):   {delta_semester:.4f}')
    log(f'    Block 3 ΔR² (language):   {delta_language:.4f}')
    log(f'    Full model marginal R²:   {r2_3m:.4f}')
    log(f'    Full model conditional R²:{r2_3c:.4f}')

decomp_df = pd.DataFrame(decomp_results)

# -- DOMINANCE ANALYSIS --------------------------------------------------------
section('DOMINANCE ANALYSIS')
log('  Azen & Budescu (2003) - average marginal R² across all predictor subsets')
log('  Sample: concern-present rows only (n=3,073)')
log('  Covariates held constant: nwords_z')
log('  Note: concern_present omitted - constant in this subset')
log(f'  Predictors competed: {available_predictors}')
log('  Dominance analysis run for all nine outcomes')
log()

# Dominance analysis is run for all nine outcomes.
DOMINANCE_OUTCOMES = OUTCOMES

dominance_results = []

for outcome_var, outcome_label in DOMINANCE_OUTCOMES.items():
    if outcome_var not in df.columns:
        continue

    cols_needed = [outcome_var, 'record_id', 'nwords_z'] + available_predictors
    sub = df[[c for c in cols_needed if c in df.columns]].dropna()

    if len(sub) < 100 or sub['record_id'].nunique() < 30:
        continue

    preds_available = [p for p in available_predictors if p in sub.columns]
    log(f'  Running: {outcome_label} ({len(sub)} obs)...')

    dom = dominance_analysis(
        outcome_var,
        preds_available,
        covariates=['nwords_z'],   # concern_present omitted - constant in subset
        data=sub
    )

    total = sum(v for v in dom.values() if v > 0)

    row = {
        'outcome':     outcome_label,
        'outcome_var': outcome_var,
        'total_dom_r2': total,
        'n_obs':       len(sub),
    }
    for p, d in dom.items():
        row[f'dom_{p}'] = d
        row[f'pct_{p}'] = (d / total * 100) if total > 0 else 0
        row[f'budescu_{p}'] = BUDESCU[outcome_var][p]

    dominance_results.append(row)

dominance_df = pd.DataFrame(dominance_results)

# -- PRINT DOMINANCE TABLE ----------------------------------------------------
section('DOMINANCE ANALYSIS - PERCENTAGE CONTRIBUTIONS')

for _, row in dominance_df.iterrows():
    log(f'\n  {row["outcome"]} (total R² explained = {row["total_dom_r2"]:.5f})')
    log(f'  {"Predictor":<25} {"Avg ΔR²":>10}  {"% of explained":>14}')
    log(f'  {"-"*25}  {"-"*10}  {"-"*14}')

    pct_cols = [
        (c.replace('pct_', ''), row[c], row[c.replace('pct_', 'dom_')])
        for c in row.index if c.startswith('pct_')
    ]
    pct_cols_sorted = sorted(pct_cols, key=lambda x: -x[2])

    for pred, pct, avg_dr2 in pct_cols_sorted:
        bar = '#' * max(0, int(pct / 2.5))
        log(f'  {pred:<25} {avg_dr2:>10.6f}  {pct:>13.1f}%  {bar}')

# -- SAVE ----------------------------------------------------------------------
section('SAVING')

decomp_df.to_csv(DECOMP_OUT, index=False)
log(f'  {DECOMP_OUT}')

dominance_df.to_csv(DOMINANCE_OUT, index=False)
log(f'  {DOMINANCE_OUT}')

with open(SUMMARY_OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'  {SUMMARY_OUT}')

log()
log('  Done.')