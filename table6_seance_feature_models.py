#!/usr/bin/env python3
"""
TEXT LEMURS - Merge SEANCE Features with Oura Data & Run Mixed Effects Models
==============================================================================
Inputs:
  - seance_features.csv     (SEANCE linguistic features per person-wave)
  - oura_aggregated.csv     (Oura sleep & activity per person-wave)

Outputs:
  - analytic_merged.csv     (merged dataset)
  - model_results.csv       (all model results)
  - model_results_fdr.csv   (FDR-corrected results)
  - model_summary.txt       (readable summary report)

Model formula (with confound covariates):
  outcome ~ feature_z + concern_present + nwords_z + Week_z + (1 | record_id)

  Covariates:
    concern_present  - controls for whether any concern was reported at all
    nwords_z         - controls for response length / verbosity
    Week_z           - controls for semester timing / academic calendar

Run from the folder containing both input files.
Requires: pandas, numpy, statsmodels, scipy
  pip install pandas numpy statsmodels scipy
"""

import re
import pandas as pd
import numpy as np
import warnings
from statsmodels.formula.api import mixedlm
from scipy.stats import false_discovery_control

warnings.filterwarnings('ignore')

# -- CONFIG -----------------------------------------------------------------
import config
SEANCE_FILE  = config.out_path('seance_features.csv')
OURA_FILE    = config.out_path('oura_aggregated.csv')
MERGED_OUT   = config.out_path('analytic_merged.csv')
RESULTS_OUT  = config.out_path('model_results.csv')
FDR_OUT      = config.out_path('model_results_fdr.csv')
SUMMARY_OUT  = config.out_path('model_summary.txt')

# Theoretically pre-specified PRIMARY features (stress-coping theory)
# Confirmatory tests - Bonferroni correction applied
PRIMARY_FEATURES = [
    'Negative_EmoLex',   # negative affect words - primary appraisal (threat)
    'Sadness_EmoLex',    # sadness - primary appraisal
    'Fear_EmoLex',       # fear/anxiety - primary appraisal
    'Valence',           # ANEW valence - emotional tone
    'Arousal',           # ANEW arousal - activation
    'Dominance',         # ANEW dominance - perceived control
]

# Exploratory features - FDR (Benjamini-Hochberg) correction applied
EXPLORATORY_FEATURES = [
    'Anxiety_GALC',
    'Tension/Stress_GALC',
    'Positive_GALC',
    'Negative_GALC',
    'Anger_EmoLex',
    'Anticipation_EmoLex',
    'Joy_EmoLex',
    'Trust_EmoLex',
    'Positive_EmoLex',
    'Negativ_GI',
    'Academ_GI',
    'Work_GI',
    'Active_GI',
    'Strong_GI',
    'polarity',
    'pleasantness',
    'aptitude',
    'attention',
    'sensitivity',
    'Powtot_Lasswell',
    'Enltot_Lasswell',
    'Afftot_Lasswell',
]

# Primary outcomes
PRIMARY_OUTCOMES = {
    'sleep_rmssd':             'RMSSD (ms)',
    'sleep_deep_hrs':          'Deep Sleep (hrs)',
    'sleep_total_hrs':         'Sleep Duration (hrs)',
    'sleep_efficiency':        'Sleep Efficiency (%)',
}

# Secondary outcomes
SECONDARY_OUTCOMES = {
    'sleep_rem_hrs':           'REM Sleep (hrs)',
    'sleep_onset_latency_min': 'Sleep Onset Latency (min)',
    'act_steps':               'Steps/day',
    'act_met_min_medium':      'MET-min Medium',
    'act_met_min_high':        'MET-min High',
}

ALL_OUTCOMES = {**PRIMARY_OUTCOMES, **SECONDARY_OUTCOMES}

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

def section(title):
    log()
    log('=' * 65)
    log(f'  {title}')
    log('=' * 65)

# -- LOAD & MERGE -----------------------------------------------------------
section('LOADING & MERGING')

seance = pd.read_csv(SEANCE_FILE)
oura   = pd.read_csv(OURA_FILE)

log(f'  SEANCE:  {seance.shape[0]:,} rows | {seance["record_id"].nunique()} participants')
log(f'  Oura:    {oura.shape[0]:,} rows  | {oura["record_id"].nunique()} participants')

seance = seance.drop_duplicates(subset=['record_id','Week'], keep='first')
oura   = oura.drop_duplicates(subset=['record_id','Week'], keep='first')

drop_from_seance = ['concern_present','concern_text_cleaned',
                    'original_turns','language','timestamp','original_index']
seance_clean = seance.drop(columns=[c for c in drop_from_seance if c in seance.columns])

df = oura.merge(seance_clean, on=['record_id','Week'], how='inner')
log(f'  Merged:  {len(df):,} rows  | {df["record_id"].nunique()} participants')

df.to_csv(MERGED_OUT, index=False)
log(f'  Saved:   {MERGED_OUT}')

# -- STANDARDIZE PREDICTORS & COVARIATES ------------------------------------
section('STANDARDIZING PREDICTORS & COVARIATES')

all_features = PRIMARY_FEATURES + EXPLORATORY_FEATURES
available    = [f for f in all_features if f in df.columns]
missing      = [f for f in all_features if f not in df.columns]

if missing:
    log(f'  Warning: {len(missing)} features not found: {missing}')

# Column names for standardized features. Non-word characters (e.g. the '/' in Tension/Stress_GALC) are replaced
# so the name can be used inside a model formula.
def zname(feat):
    return re.sub(r'\W', '_', feat) + '_z'

# Standardize SEANCE features
for feat in available:
    mu, sd = df[feat].mean(), df[feat].std()
    df[zname(feat)] = (df[feat] - mu) / sd if sd > 0 else 0

# Standardize continuous covariates
for covar in ['nwords', 'Week']:
    mu, sd = df[covar].mean(), df[covar].std()
    df[f'{covar}_z'] = (df[covar] - mu) / sd if sd > 0 else 0

log(f'  SEANCE features standardized: {len(available)}')
log(f'  Covariates: concern_present (binary), nwords_z, Week_z')
log()
log('  Model formula:')
log('    outcome ~ feature_z + concern_present + nwords_z + Week_z + (1|record_id)')
log()
log('  Covariate rationale:')
log('    concern_present - isolates linguistic content from mere concern presence')
log('    nwords_z        - controls for response length / verbosity')
log('    Week_z          - controls for semester timing / academic calendar')

# -- RUN MODELS -------------------------------------------------------------
section('RUNNING MIXED EFFECTS MODELS')

results = []
failed = []

for feat in available:
    feat_z     = zname(feat)
    is_primary = feat in PRIMARY_FEATURES

    for outcome, outcome_label in ALL_OUTCOMES.items():
        if outcome not in df.columns:
            continue

        sub = df[[feat_z, outcome, 'record_id',
                  'concern_present', 'nwords_z', 'Week_z']].dropna()

        if len(sub) < 100 or sub['record_id'].nunique() < 30:
            continue

        try:
            model = mixedlm(
                f'{outcome} ~ {feat_z} + concern_present + nwords_z + Week_z',
                sub,
                groups=sub['record_id']
            ).fit(reml=True, method='lbfgs', disp=False)

            results.append({
                'feature':              feat,
                'outcome':              outcome_label,
                'outcome_var':          outcome,
                'beta':                 model.fe_params[feat_z],
                'se':                   model.bse[feat_z],
                'p':                    model.pvalues[feat_z],
                'beta_concern_present': model.fe_params.get('concern_present', np.nan),
                'beta_nwords_z':        model.fe_params.get('nwords_z', np.nan),
                'beta_Week_z':          model.fe_params.get('Week_z', np.nan),
                'p_Week_z':             model.pvalues.get('Week_z', np.nan),
                'n_obs':                len(sub),
                'n_participants':       sub['record_id'].nunique(),
                'type':                 'primary' if is_primary else 'exploratory'
            })

        except Exception as e:
            failed.append((feat, outcome, str(e)[:80]))

results_df = pd.DataFrame(results)
log(f'  Ran {len(results_df)} models successfully')
if failed:
    log(f'  WARNING: {len(failed)} models failed to fit:')
    for f_, o_, e_ in failed:
        log(f'    {f_} / {o_}: {e_}')

# -- MULTIPLE COMPARISON CORRECTION -----------------------------------------
section('MULTIPLE COMPARISON CORRECTION')

primary_df     = results_df[results_df['type'] == 'primary'].copy()
exploratory_df = results_df[results_df['type'] == 'exploratory'].copy()

n_primary  = len(primary_df)
bonf_alpha = 0.05 / n_primary if n_primary > 0 else 0.05

primary_df['p_bonferroni'] = (primary_df['p'] * n_primary).clip(upper=1.0)
primary_df['sig_bonferroni'] = primary_df['p_bonferroni'].apply(
    lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('.' if p<0.10 else '')))
)

if len(exploratory_df) > 0:
    exploratory_df['p_fdr'] = false_discovery_control(exploratory_df['p'].values, method='bh')
    exploratory_df['sig_fdr'] = exploratory_df['p_fdr'].apply(
        lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('.' if p<0.10 else '')))
    )

results_df['sig_raw'] = results_df['p'].apply(
    lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('.' if p<0.10 else '')))
)
for frame in [primary_df, exploratory_df]:
    frame['sig_raw'] = frame['p'].apply(
        lambda p: '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('.' if p<0.10 else '')))
    )

log(f'  Primary tests:     {n_primary} | Bonferroni threshold: p < {bonf_alpha:.4f}')
log(f'  Exploratory tests: {len(exploratory_df)} | FDR (Benjamini-Hochberg)')

# -- RESULTS SUMMARY --------------------------------------------------------
section('PRIMARY RESULTS (Bonferroni corrected)')

log(f'\n  {"Feature":<22} {"Outcome":<24} {"Beta":>7}  {"SE":>6}  {"p_raw":>7}  {"p_bonf":>7}  {"Sig":>4}')
log(f'  {"-"*22}  {"-"*24}  {"-"*7}  {"-"*6}  {"-"*7}  {"-"*7}  {"-"*4}')
for _, r in primary_df.sort_values('p').iterrows():
    log(f'  {r["feature"]:<22} {r["outcome"]:<24} {r["beta"]:>7.3f}  {r["se"]:>6.3f}  '
        f'{r["p"]:>7.4f}  {r["p_bonferroni"]:>7.4f}  {r["sig_bonferroni"]:>4}')

sig_primary = primary_df[primary_df['p_bonferroni'] < 0.05]
log(f'\n  Bonferroni-significant: {len(sig_primary)} of {n_primary} primary tests')

section('EXPLORATORY RESULTS (FDR corrected, p_fdr < .05)')

sig_exp = exploratory_df[exploratory_df['p_fdr'] < 0.05].sort_values('p_fdr') if len(exploratory_df) > 0 else pd.DataFrame()
log(f'\n  {"Feature":<22} {"Outcome":<24} {"Beta":>7}  {"SE":>6}  {"p_raw":>7}  {"p_fdr":>7}  {"Sig":>4}')
log(f'  {"-"*22}  {"-"*24}  {"-"*7}  {"-"*6}  {"-"*7}  {"-"*7}  {"-"*4}')
for _, r in sig_exp.iterrows():
    log(f'  {r["feature"]:<22} {r["outcome"]:<24} {r["beta"]:>7.3f}  {r["se"]:>6.3f}  '
        f'{r["p"]:>7.4f}  {r["p_fdr"]:>7.4f}  {r["sig_fdr"]:>4}')

if len(sig_exp) == 0:
    log('  No exploratory results survive FDR correction.')

section('ALL RAW SIGNIFICANT RESULTS (p < .05, uncorrected, for reference)')

all_sig = results_df[results_df['p'] < 0.05].sort_values('p')
log(f'\n  {"Feature":<22} {"Outcome":<24} {"Beta":>7}  {"p":>7}  {"Type":<12}  {"Sig":>4}')
log(f'  {"-"*22}  {"-"*24}  {"-"*7}  {"-"*7}  {"-"*12}  {"-"*4}')
for _, r in all_sig.iterrows():
    log(f'  {r["feature"]:<22} {r["outcome"]:<24} {r["beta"]:>7.3f}  '
        f'{r["p"]:>7.4f}  {r["type"]:<12}  {r["sig_raw"]:>4}')

log(f'\n  Total raw significant: {len(all_sig)} of {len(results_df)} tests')

# -- SAVE -------------------------------------------------------------------
section('SAVING OUTPUTS')

full_results = pd.concat([primary_df, exploratory_df], ignore_index=True)
full_results = full_results.sort_values(['type','p'])
full_results.to_csv(RESULTS_OUT, index=False)
log(f'  {RESULTS_OUT} - all {len(full_results)} models')

fdr_results = full_results[
    ((full_results['type']=='primary')     & (full_results['p_bonferroni'] < 0.05)) |
    ((full_results['type']=='exploratory') & (full_results['p_fdr'] < 0.05))
].copy()
fdr_results.to_csv(FDR_OUT, index=False)
log(f'  {FDR_OUT} - {len(fdr_results)} results surviving correction')

with open(SUMMARY_OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'  {SUMMARY_OUT} - full summary report')

log()
log('  Done.')