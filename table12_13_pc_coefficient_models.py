#!/usr/bin/env python3
"""
TEXT LEMURS - Mixed-Effects Models with Embedding Features
=======================================================================
Week_z is included as a covariate in all models, consistent with the
SEANCE mixed-effects models.

Formula (Model A - PCA only):
    outcome ~ PC1 + PC2 + ... + PCn + Week_z + (1 | record_id)

Formula (Model B - PCA + domain scores):
    outcome ~ PC1 + ... + PCn + domain_scores + Week_z + (1 | record_id)

Outputs are written to results/.

Requirements:
    pip install pandas numpy statsmodels

Usage:
    python embed_mixed2.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# -- CONFIG --------------------------------------------------------------------
import config
OUTPUT_DIR  = config.out_path('embedding_outputs')
RESULTS_DIR = config.out_path('results')
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTCOMES = [
    'sleep_total_hrs',
    'sleep_efficiency',
    'sleep_rem_hrs',
    'sleep_deep_hrs',
    'sleep_rmssd',
    'sleep_onset_latency_min',
    'act_steps',
    'act_met_min_medium',
    'act_met_min_high',
]

MODELS = ['roberta', 'mentalroberta']

# -- HELPERS -------------------------------------------------------------------
def log(msg=''):
    print(msg, flush=True)


def load_data():
    path = os.path.join(OUTPUT_DIR, 'text_lemurs_embeddings_merged.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Merged file not found: {path}\n'
            f'Run embed.py first.'
        )
    log(f'Loading: {path}')
    df = pd.read_csv(path)
    log(f'  Rows: {len(df):,}  |  Columns: {len(df.columns)}')

    # Standardize Week_z
    if 'Week' in df.columns:
        mu, sd = df['Week'].mean(), df['Week'].std()
        df['Week_z'] = (df['Week'] - mu) / sd if sd > 0 else 0
        log(f'  Week_z added (mean={mu:.1f}, sd={sd:.1f})')
    else:
        raise ValueError('Week column not found in merged file.')

    return df


def run_model(df, outcome, predictors, model_label):
    """
    Run mixed-effects model with Week_z covariate.
    predictors = list of PC column names (and optionally domain columns)
    Week_z is always added as a covariate.
    """
    # Always include Week_z as covariate
    all_predictors = predictors + ['Week_z']
    needed = [outcome] + all_predictors + ['record_id']
    data   = df[needed].dropna()

    if len(data) == 0:
        log(f'    SKIP {outcome} - no complete cases')
        return None

    n_subjects = data['record_id'].nunique()
    n_obs      = len(data)

    formula = outcome + ' ~ ' + ' + '.join(all_predictors)

    try:
        fit    = smf.mixedlm(formula, data, groups=data['record_id'])
        result = fit.fit(reml=True, method='lbfgs', disp=False)
    except Exception as e:
        log(f'    FAILED {outcome}: {e}')
        return None

    coef_df = pd.DataFrame({
        'term':      result.params.index,
        'coef':      result.params.values,
        'std_err':   result.bse.values,
        'z':         result.tvalues.values,
        'p_value':   result.pvalues.values,
        'ci_lower':  result.conf_int()[0].values,
        'ci_upper':  result.conf_int()[1].values,
    })

    for col in ['coef', 'std_err', 'z', 'p_value', 'ci_lower', 'ci_upper']:
        coef_df[col] = pd.to_numeric(coef_df[col], errors='coerce')

    coef_df['outcome']     = outcome
    coef_df['model']       = model_label
    coef_df['n_subjects']  = n_subjects
    coef_df['n_obs']       = n_obs

    # Count significant PC predictors only (exclude Intercept and Week_z)
    pc_terms = coef_df[
        ~coef_df['term'].isin(['Intercept', 'Week_z', 'Group Var'])
        & ~coef_df['term'].str.startswith('domain_')
    ]
    n_sig = (pc_terms['p_value'] < 0.05).sum()
    log(f'    {outcome:<26} | {n_obs:>5} obs | {n_subjects:>3} subjects '
        f'| {n_sig} significant PCs')

    return coef_df


# -- MAIN ----------------------------------------------------------------------
def main():
    log('=' * 65)
    log('TEXT LEMURS - Embedding Mixed-Effects Models')
    log('=' * 65)

    df = load_data()

    # Identify domain probability columns
    domain_cols = [
        c for c in df.columns
        if c.startswith('domain_')
        and c not in ('top_domain', 'top_score', 'domain_no_concern')
    ]
    log(f'\nDomain columns ({len(domain_cols)}): {domain_cols}')

    all_results = []

    for model_key in MODELS:
        pc_cols = [c for c in df.columns if c.startswith(f'{model_key}_PC')]

        if not pc_cols:
            log(f'\nWARNING: No PC columns for "{model_key}" - skipping')
            continue

        # -- Model A: PCA components + Week_z ---------------------------------
        label_a = f'{model_key}_pca_only'
        log(f'\n{"="*65}')
        log(f'  {label_a}  ({len(pc_cols)} PCs + Week_z)')
        log(f'{"="*65}')

        for outcome in OUTCOMES:
            if outcome not in df.columns:
                log(f'    SKIP {outcome} - not in data')
                continue
            result = run_model(df, outcome, pc_cols, label_a)
            if result is not None:
                all_results.append(result)

        # -- Model B: PCA + domain scores + Week_z ----------------------------
        if domain_cols:
            label_b   = f'{model_key}_pca_plus_domain'
            all_preds = pc_cols + domain_cols
            log(f'\n{"="*65}')
            log(f'  {label_b}  ({len(pc_cols)} PCs + {len(domain_cols)} domains + Week_z)')
            log(f'{"="*65}')

            for outcome in OUTCOMES:
                if outcome not in df.columns:
                    continue
                result = run_model(df, outcome, all_preds, label_b)
                if result is not None:
                    all_results.append(result)

    if not all_results:
        log('\nERROR: No models ran.')
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # -- Save full results -----------------------------------------------------
    full_path = os.path.join(RESULTS_DIR, 'embedding_model_results_full.csv')
    results_df.to_csv(full_path, index=False)
    log(f'\nFull results:    {full_path}  {results_df.shape}')

    # -- Significant fixed effects (PCs and domains only, not Week_z) ----------
    sig_df = results_df[
        (results_df['term'] != 'Intercept') &
        (results_df['term'] != 'Week_z') &
        (results_df['term'] != 'Group Var') &
        (results_df['p_value'] < 0.05)
    ].copy()

    sig_path = os.path.join(RESULTS_DIR, 'embedding_model_results_significant.csv')
    sig_df.to_csv(sig_path, index=False)
    log(f'Significant:     {sig_path}  ({len(sig_df)} rows)')

    # -- Summary ---------------------------------------------------------------
    log(f'\n{"="*65}')
    log('  SUMMARY - Significant PCs per outcome × model')
    log(f'{"="*65}')

    # PC-only significant counts
    pc_sig = sig_df[
        ~sig_df['term'].str.startswith('domain_')
    ].copy()

    summary = (
        pc_sig[pc_sig['model'].str.endswith('pca_only')]
        .groupby(['model', 'outcome'])
        .agg(n_sig=('term', 'count'), min_p=('p_value', 'min'))
        .reset_index()
        .sort_values(['outcome', 'model'])
    )
    log(summary.to_string(index=False))
    summary.to_csv(
        os.path.join(RESULTS_DIR, 'embedding_sig_summary.csv'), index=False
    )

    # Unique PC counts per model
    log(f'\n{"="*65}')
    log('  UNIQUE SIGNIFICANT PCs PER MODEL (pca_only)')
    log(f'{"="*65}')
    for model_key in MODELS:
        label = f'{model_key}_pca_only'
        msig  = pc_sig[
            (pc_sig['model'] == label) &
            (pc_sig['term'].str.startswith(model_key))
        ]
        log(f'  {model_key}: {msig["term"].nunique()} unique PCs, '
            f'{len(msig)} total associations')

    log('\nDone.')
    log(f'Results saved to: ./{RESULTS_DIR}/')


if __name__ == '__main__':
    main()