#!/usr/bin/env python3
"""
TEXT LEMURS - Descriptives & Semester Decline
===============================================
Produces three output files:

1. TEXT_LEMURS_descriptives.csv      - sample, text, and SEANCE coverage stats
2. TEXT_LEMURS_outcome_descriptives.csv - health outcome means/SD + semester decline table
3. TEXT_LEMURS_semester_decline.csv  - clean model-derived semester decline
                                       (baseline model, no linguistic feature)

Run from the folder containing oura_aggregated.csv and seance_features.csv.
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.formula.api import mixedlm

warnings.filterwarnings('ignore')

# -- FILE PATHS ------------------------------------------------------------
import config
OURA_FILE       = config.out_path('oura_aggregated.csv')
SEANCE_FILE     = config.out_path('seance_features.csv')

DESC_OUT        = config.out_path('TEXT_LEMURS_descriptives.csv')
OUTCOME_OUT     = config.out_path('TEXT_LEMURS_outcome_descriptives.csv')
SEMESTER_OUT    = config.out_path('TEXT_LEMURS_semester_decline.csv')

# -- OUTCOMES --------------------------------------------------------------
OUTCOMES = {
    'sleep_rmssd':             'RMSSD (ms)',
    'sleep_efficiency':        'Sleep Efficiency (%)',
    'sleep_total_hrs':         'Sleep Duration (hrs)',
    'sleep_deep_hrs':          'Deep Sleep (hrs)',
    'sleep_rem_hrs':           'REM Sleep (hrs)',
    'sleep_onset_latency_min': 'Sleep Onset Latency (min)',
    'act_steps':               'Steps / day',
    'act_met_min_medium':      'MET-min Medium',
    'act_met_min_high':        'MET-min High',
}

# -- SEANCE FEATURES FOR COVERAGE TABLE -----------------------------------
SEANCE_FEATURES = {
    'Strong_GI':       'Strong / forceful language (GI)',
    'Negative_EmoLex': 'Negative emotion (EmoLex)',
    'Sadness_EmoLex':  'Sadness (EmoLex)',
    'Fear_EmoLex':     'Fear (EmoLex)',
    'Valence':         'Valence (ANEW)',
    'Arousal':         'Arousal (ANEW)',
    'Dominance':       'Dominance (ANEW)',
    'Academ_GI':       'Academic language (GI)',
    'Work_GI':         'Work language (GI)',
    'sensitivity':     'Sensitivity (SenticNet)',
    'pleasantness':    'Pleasantness (SenticNet)',
    'polarity':        'Polarity (SenticNet)',
}

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

def zscore(s):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 0 else s * 0

# -- LOAD & MERGE ----------------------------------------------------------
log('Loading data...')
oura   = pd.read_csv(OURA_FILE)
seance = pd.read_csv(SEANCE_FILE)

oura   = oura.drop_duplicates(subset=['record_id', 'Week'], keep='first')
seance = seance.drop_duplicates(subset=['record_id', 'Week'], keep='first')

drop_cols = ['concern_present', 'concern_text_cleaned', 'original_turns',
             'language', 'timestamp', 'original_index']
seance_clean = seance.drop(columns=[c for c in drop_cols if c in seance.columns])
df = oura.merge(seance_clean, on=['record_id', 'Week'], how='inner')

log(f'  Merged: {len(df):,} rows | {df["record_id"].nunique()} participants')

# -- STANDARDIZE WEEK ------------------------------------------------------
df['Week_z'] = zscore(df['Week'])
df['nwords_z'] = zscore(df['nwords'])

# ===========================================================================
# FILE 1 - DESCRIPTIVES
# ===========================================================================
log('\nBuilding descriptives...')

desc_rows = []

# -- Sample ----------------------------------------------------------------
waves_per = df.groupby('record_id').size()
desc_rows += [
    {'Section': 'Sample',  'Measure': 'Total participants',                   'Value': df['record_id'].nunique()},
    {'Section': 'Sample',  'Measure': 'Total person-waves',                   'Value': f"{len(df):,}"},
    {'Section': 'Sample',  'Measure': 'Waves per person - mean',              'Value': f"{waves_per.mean():.1f}"},
    {'Section': 'Sample',  'Measure': 'Waves per person - median',            'Value': f"{waves_per.median():.0f}"},
    {'Section': 'Sample',  'Measure': 'Waves per person - min',               'Value': waves_per.min()},
    {'Section': 'Sample',  'Measure': 'Waves per person - max',               'Value': waves_per.max()},
    {'Section': 'Sample',  'Measure': 'Semester weeks covered',               'Value': f"Weeks {df['Week'].min()}–{df['Week'].max()}"},
]

# -- Concern text ----------------------------------------------------------
def word_count_rows(nw, scope):
    n = len(nw)
    return [
        {'Section': 'Concern text', 'Measure': f'Responses ({scope})',                'Value': f"{n:,}"},
        {'Section': 'Concern text', 'Measure': f'Word count - mean ({scope})',        'Value': f"{nw.mean():.1f}"},
        {'Section': 'Concern text', 'Measure': f'Word count - median ({scope})',      'Value': f"{nw.median():.1f}"},
        {'Section': 'Concern text', 'Measure': f'Word count - SD ({scope})',          'Value': f"{nw.std():.2f}"},
        {'Section': 'Concern text', 'Measure': f'Word count - min ({scope})',         'Value': f"{nw.min():.0f}"},
        {'Section': 'Concern text', 'Measure': f'Word count - max ({scope})',         'Value': f"{nw.max():.0f}"},
        {'Section': 'Concern text', 'Measure': f'Responses of exactly 1 word ({scope})', 'Value': f"{(nw==1).sum():,}  ({(nw==1).mean()*100:.2f}%)"},
        {'Section': 'Concern text', 'Measure': f'Responses of <= 3 words ({scope})',  'Value': f"{(nw<=3).sum():,}  ({(nw<=3).mean()*100:.2f}%)"},
        {'Section': 'Concern text', 'Measure': f'Responses of >= 10 words ({scope})', 'Value': f"{(nw>=10).sum():,}  ({(nw>=10).mean()*100:.2f}%)"},
    ]

# Word-count statistics are reported for the concern-present responses (the corpus of text responses).
# Concern-absent waves ("nothing", blank) carry nwords = 1 and are shown separately for reference.
desc_rows += [
    {'Section': 'Concern text', 'Measure': 'Waves with concern text present',  'Value': f"{df['concern_present'].sum():,}  ({df['concern_present'].mean()*100:.1f}%)"},
    {'Section': 'Concern text', 'Measure': 'Waves with no concern',            'Value': f"{(df['concern_present']==0).sum():,}  ({(df['concern_present']==0).mean()*100:.1f}%)"},
]
desc_rows += word_count_rows(df.loc[df['concern_present'] == 1, 'nwords'], 'concern-present responses')
desc_rows += word_count_rows(df['nwords'], 'all waves')

# -- SEANCE coverage -------------------------------------------------------
for col, label in SEANCE_FEATURES.items():
    if col in df.columns:
        nonzero = (df[col] > 0).sum()
        cov = nonzero / len(df) * 100
        mean_nz = df[df[col] > 0][col].mean()
        desc_rows.append({
            'Section': 'SEANCE coverage',
            'Measure': label,
            'Value': f"{nonzero:,}  ({cov:.1f}%)",
            'Mean_when_nonzero': f"{mean_nz:.4f}"
        })

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(DESC_OUT, index=False)
log(f'  Saved: {DESC_OUT}')

# ===========================================================================
# FILE 2 - OUTCOME DESCRIPTIVES
# ===========================================================================
log('\nBuilding outcome descriptives...')

outcome_rows = []
for col, label in OUTCOMES.items():
    if col in df.columns:
        s = df[col].dropna()
        outcome_rows.append({
            'Outcome':  label,
            'N':        len(s),
            'Mean':     round(s.mean(), 2),
            'SD':       round(s.std(), 2),
            'Min':      round(s.min(), 2),
            'Max':      round(s.max(), 2),
            'Median':   round(s.median(), 2),
        })

outcome_df = pd.DataFrame(outcome_rows)
outcome_df.to_csv(OUTCOME_OUT, index=False)
log(f'  Saved: {OUTCOME_OUT}')

# -- Week 1 and Week 33 raw means ------------------------------------------
log('\n  Raw means by week (first and last):')
week_means = df.groupby('Week')[[c for c in OUTCOMES if c in df.columns]].mean()
log(f"  First week ({df['Week'].min()}):")
log(f"  {week_means.iloc[0].round(2).to_dict()}")
log(f"  Last week ({df['Week'].max()}):")
log(f"  {week_means.iloc[-1].round(2).to_dict()}")

# ===========================================================================
# FILE 3 - SEMESTER DECLINE (CLEAN BASELINE MODEL)
# ===========================================================================
log('\nRunning baseline semester decline models...')
log('  Formula: outcome ~ concern_present + nwords_z + Week_z + (1 | record_id)')
log('  (No linguistic feature - pure semester effect)')

semester_rows = []

for col, label in OUTCOMES.items():
    if col not in df.columns:
        continue

    sub = df[['record_id', col, 'Week_z', 'Week', 'concern_present', 'nwords_z']].dropna()
    if len(sub) < 100 or sub['record_id'].nunique() < 30:
        continue

    try:
        formula = f'{col} ~ concern_present + nwords_z + Week_z'
        model = mixedlm(formula, sub, groups=sub['record_id']).fit(
            reml=True, method='lbfgs', disp=False)

        # Extract Week_z coefficient
        beta  = model.params.get('Week_z', np.nan)
        se    = model.bse.get('Week_z', np.nan)
        pval  = model.pvalues.get('Week_z', np.nan)

        # Model-implied means at Week 1 and Week 33
        # Using fixed effects: intercept + concern_present*mean + nwords_z*mean + Week_z*week_z_value
        week1_z  = (df['Week'].min()  - df['Week'].mean()) / df['Week'].std()
        week33_z = (df['Week'].max()  - df['Week'].mean()) / df['Week'].std()

        intercept       = model.params.get('Intercept', np.nan)
        beta_concern    = model.params.get('concern_present', np.nan)
        beta_nwords     = model.params.get('nwords_z', np.nan)
        mean_concern    = sub['concern_present'].mean()
        mean_nwords     = sub['nwords_z'].mean()  # 0 by definition (z-scored)

        implied_week1  = intercept + beta_concern * mean_concern + beta_nwords * mean_nwords + beta * week1_z
        implied_week33 = intercept + beta_concern * mean_concern + beta_nwords * mean_nwords + beta * week33_z

        # Format p-value
        if pval < .001:
            p_str = 'p < .001***'
        elif pval < .01:
            p_str = f'p = {pval:.3f}**'
        elif pval < .05:
            p_str = f'p = {pval:.3f}*'
        else:
            p_str = f'p = {pval:.3f}'

        # Raw means for reference
        raw_week1  = df[df['Week'] == df['Week'].min()][col].mean()
        raw_week33 = df[df['Week'] == df['Week'].max()][col].mean()

        semester_rows.append({
            'Outcome':               label,
            'N_obs':                 len(sub),
            'N_participants':        sub['record_id'].nunique(),
            'Raw_mean_Week1':        round(raw_week1, 2),
            'Raw_mean_Week33':       round(raw_week33, 2),
            'Model_implied_Week1':   round(implied_week1, 2),
            'Model_implied_Week33':  round(implied_week33, 2),
            'Beta_Week_z':           round(beta, 3),
            'SE':                    round(se, 3),
            'p_value':               round(pval, 4),
            'p_formatted':           p_str,
            'Direction':             'decline' if beta < 0 else 'increase',
            'Significant':           pval < .05,
        })

        log(f'  {label:<30} beta={beta:>7.3f}  {p_str}  '
            f'Week1={implied_week1:.1f}  Week33={implied_week33:.1f}')

    except Exception as e:
        log(f'  {label}: model failed - {e}')

semester_df = pd.DataFrame(semester_rows)
semester_df.to_csv(SEMESTER_OUT, index=False)
log(f'\n  Saved: {SEMESTER_OUT}')

# -- SUMMARY ---------------------------------------------------------------
log('\n' + '='*60)
log('SUMMARY')
log('='*60)
log(f'  {DESC_OUT}         - {len(desc_df)} descriptive stats')
log(f'  {OUTCOME_OUT}  - {len(outcome_df)} outcomes')
log(f'  {SEMESTER_OUT}   - {len(semester_df)} semester decline models')
sig = semester_df[semester_df['Significant']]['Outcome'].tolist()
log(f'\n  Significant semester effects ({len(sig)}):')
for o in sig:
    log(f'    {o}')