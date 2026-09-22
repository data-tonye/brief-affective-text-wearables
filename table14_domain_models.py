#!/usr/bin/env python3
"""
TEXT LEMURS - Standalone Zero-Shot Domain Classification Models
===============================================================
Tests whether zero-shot domain probability scores independently
associate with wearable outcomes, controlling for semester timing.

Formula: outcome ~ domain_scores + Week_z + (1 | record_id)

Note: Models run on concern-present rows only (n=3,073) since
domain classifications were only produced for those rows.
domain_no_concern is dropped as the reference category to avoid
perfect multicollinearity (domain probabilities sum to 1).

Inputs:
    domain_classifications.csv   (from embed.py)
    oura_aggregated.csv

Outputs (saved to ./domain_results/):
    domain_model_results_full.csv       - all 72 tests
    domain_model_results_significant.csv - nominally significant (p < .05)
    domain_model_results_fdr.csv        - FDR-corrected results
    domain_model_summary.txt            - readable report

Requirements:
    pip install pandas numpy statsmodels scipy
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from statsmodels.formula.api import mixedlm
from scipy.stats import false_discovery_control

# -- CONFIG --------------------------------------------------------------------
import config
DOMAIN_FILE = config.out_path('embedding_outputs/domain_classifications.csv')
OURA_FILE   = config.out_path('oura_aggregated.csv')
OUT_DIR     = config.out_path('domain_results')
os.makedirs(OUT_DIR, exist_ok=True)

OUTCOMES = {
    'sleep_total_hrs':         'Sleep Duration (hrs)',
    'sleep_efficiency':        'Sleep Efficiency (%)',
    'sleep_rem_hrs':           'REM Sleep (hrs)',
    'sleep_deep_hrs':          'Deep Sleep (hrs)',
    'sleep_rmssd':             'RMSSD (ms)',
    'sleep_onset_latency_min': 'Sleep Onset Latency (min)',
    'act_steps':               'Steps/day',
    'act_met_min_medium':      'MET-min Medium',
    'act_met_min_high':        'MET-min High',
}

# Domain predictors - domain_no_concern dropped as reference category
DOMAIN_PREDICTORS = [
    'domain_academic_workload_exams',
    'domain_money_finances',
    'domain_physical_health_illness',
    'domain_mental_health_anxiety',
    'domain_relationships_social_life',
    'domain_housing_living_situation',
    'domain_future_plans_career',
    'domain_general_stress',
    # domain_no_concern excluded (reference)
]

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# -- LOAD & MERGE --------------------------------------------------------------
log('=' * 60)
log('TEXT LEMURS - Zero-Shot Domain Classification Models')
log('=' * 60)

domain = pd.read_csv(DOMAIN_FILE)
oura   = pd.read_csv(OURA_FILE)

log(f'Domain file: {len(domain):,} rows | {domain["record_id"].nunique()} participants')
log(f'Oura file:   {len(oura):,} rows  | {oura["record_id"].nunique()} participants')

df = oura.merge(domain, on=['record_id', 'Week'], how='inner')
log(f'Merged:      {len(df):,} rows  | {df["record_id"].nunique()} participants')
log(f'             (concern-present rows only - domain classifications')
log(f'              produced only for waves where concern text was present)')

# Standardize Week
mu, sd = df['Week'].mean(), df['Week'].std()
df['Week_z'] = (df['Week'] - mu) / sd

# Verify domain predictors exist
missing = [c for c in DOMAIN_PREDICTORS if c not in df.columns]
available = [c for c in DOMAIN_PREDICTORS if c in df.columns]
if missing:
    log(f'\nWARNING: {len(missing)} domain columns not found: {missing}')
log(f'\nDomain predictors: {len(available)} (reference: domain_no_concern excluded)')

# -- RUN MODELS ----------------------------------------------------------------
log('\n' + '=' * 60)
log('RUNNING MODELS')
log('Formula: outcome ~ domain_scores + Week_z + (1 | record_id)')
log('=' * 60)

# Build formula with Q() to handle special characters in column names
domain_formula_parts = ' + '.join([f'Q("{c}")' for c in available])
results = []

for outcome_col, outcome_label in OUTCOMES.items():
    if outcome_col not in df.columns:
        log(f'  SKIP {outcome_label} - column not found')
        continue

    needed = [outcome_col, 'record_id', 'Week_z'] + available
    sub = df[needed].dropna()

    if len(sub) < 100 or sub['record_id'].nunique() < 30:
        log(f'  SKIP {outcome_label} - insufficient data ({len(sub)} rows)')
        continue

    formula = f'{outcome_col} ~ {domain_formula_parts} + Week_z'

    try:
        model  = mixedlm(formula, sub, groups=sub['record_id'])
        result = model.fit(reml=True, method='lbfgs', disp=False)

        sig_count = 0
        for dc in available:
            key = f'Q("{dc}")'
            if key in result.params:
                beta = result.params[key]
                se   = result.bse[key]
                p    = result.pvalues[key]
                ci_l = result.conf_int().loc[key, 0]
                ci_u = result.conf_int().loc[key, 1]
                if p < 0.05:
                    sig_count += 1
                results.append({
                    'domain':        dc.replace('domain_', '').replace('_', ' '),
                    'domain_col':    dc,
                    'outcome':       outcome_label,
                    'outcome_col':   outcome_col,
                    'beta':          round(beta, 4),
                    'se':            round(se, 4),
                    'ci_lower':      round(ci_l, 4),
                    'ci_upper':      round(ci_u, 4),
                    'p':             p,
                    'n_obs':         len(sub),
                    'n_participants':sub['record_id'].nunique(),
                })

        log(f'  {outcome_label:<28}: {sig_count} nominally significant '
            f'(n={len(sub):,})')

    except Exception as e:
        log(f'  {outcome_label}: FAILED - {e}')

# -- MULTIPLE COMPARISON CORRECTION -------------------------------------------
log('\n' + '=' * 60)
log('MULTIPLE COMPARISON CORRECTION')
log('=' * 60)

res = pd.DataFrame(results)
n_tests = len(res)

# FDR correction across all domain × outcome tests
res['p_fdr'] = false_discovery_control(res['p'].values, method='bh')

res['sig_raw'] = res['p'].apply(
    lambda p: '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else ''))
)
res['sig_fdr'] = res['p_fdr'] < 0.05

log(f'Total tests:                    {n_tests}  ({len(available)} domains × {len(OUTCOMES)} outcomes)')
log(f'Nominally significant (p<.05):  {(res["p"] < 0.05).sum()}')
log(f'Survive FDR correction:         {res["sig_fdr"].sum()}')

# -- RESULTS DISPLAY -----------------------------------------------------------
log('\n' + '=' * 60)
log('ALL NOMINALLY SIGNIFICANT RESULTS (p < .05, uncorrected)')
log('=' * 60)

sig_raw = res[res['p'] < 0.05].sort_values('p')
if len(sig_raw) > 0:
    log(f'\n  {"Domain":<30} {"Outcome":<28} {"Beta":>7}  {"SE":>6}  '
        f'{"p_raw":>8}  {"p_fdr":>8}  {"Sig_FDR":>8}')
    log(f'  {"-"*30}  {"-"*28}  {"-"*7}  {"-"*6}  {"-"*8}  {"-"*8}  {"-"*8}')
    for _, r in sig_raw.iterrows():
        fdr_marker = 'YES' if r['sig_fdr'] else 'no'
        log(f'  {r["domain"]:<30} {r["outcome"]:<28} {r["beta"]:>7.3f}  '
            f'{r["se"]:>6.3f}  {r["p"]:>8.4f}  {r["p_fdr"]:>8.4f}  {fdr_marker:>8}')
else:
    log('  No nominally significant results.')

log('\n' + '=' * 60)
log('FDR-SIGNIFICANT RESULTS')
log('=' * 60)

sig_fdr = res[res['sig_fdr']].sort_values('p_fdr')
if len(sig_fdr) > 0:
    for _, r in sig_fdr.iterrows():
        log(f'  {r["domain"]} -> {r["outcome"]}')
        log(f'    β = {r["beta"]:.4f}, SE = {r["se"]:.4f}, '
            f'p = {r["p"]:.4f}, p_fdr = {r["p_fdr"]:.4f}')
        log(f'    95% CI: [{r["ci_lower"]:.4f}, {r["ci_upper"]:.4f}]')
else:
    log('  No results survive FDR correction.')

log('\n' + '=' * 60)
log('INTERPRETATION')
log('=' * 60)
log(f'  At alpha = .05 with {n_tests} tests, we expect ~{n_tests * 0.05:.1f} false positives by chance.')
log(f'  Observed {(res["p"] < 0.05).sum()} nominally significant associations.')
n_fdr = res["sig_fdr"].sum()
if n_fdr == 0:
    log('  None survive FDR correction - consistent with chance-level findings.')
    log('  This supports the conclusion that concern topic (domain) does not')
    log('  independently predict wearable outcomes.')
else:
    log(f'  {n_fdr} association(s) survive FDR correction.')
    log('  These should be reported and interpreted in the paper.')

# -- SAVE ---------------------------------------------------------------------
log('\n' + '=' * 60)
log('SAVING OUTPUTS')
log('=' * 60)

# Format p-values for display
res['p_display'] = res['p'].apply(
    lambda p: '< .001' if p < .001 else f'= .{str(round(p, 3)).split(".")[1].zfill(3)}'
)
res['p_fdr_display'] = res['p_fdr'].apply(
    lambda p: '< .001' if p < .001 else f'= .{str(round(p, 3)).split(".")[1].zfill(3)}'
)

full_path = os.path.join(OUT_DIR, 'domain_model_results_full.csv')
res.sort_values(['outcome', 'p']).to_csv(full_path, index=False)
log(f'  Full results:        {full_path}  ({len(res)} rows)')

sig_path = os.path.join(OUT_DIR, 'domain_model_results_significant.csv')
sig_raw.to_csv(sig_path, index=False)
log(f'  Significant (p<.05): {sig_path}  ({len(sig_raw)} rows)')

fdr_path = os.path.join(OUT_DIR, 'domain_model_results_fdr.csv')
sig_fdr.to_csv(fdr_path, index=False)
log(f'  FDR significant:     {fdr_path}  ({len(sig_fdr)} rows)')

summary_path = os.path.join(OUT_DIR, 'domain_model_summary.txt')
with open(summary_path, 'w') as f:
    f.write('\n'.join(lines))
log(f'  Summary report:      {summary_path}')

log('\nDone.')