#!/usr/bin/env python3
"""
TEXT LEMURS - Bimonthly Window Aggregation
===========================================
For each concern text response (one per participant per wave), aggregates
daily Oura sleep and activity data from the 14 days preceding the survey date.

This produces one row per participant per wave with:
  - Concern text metadata (record_id, Week, concern_present, concern_text_cleaned)
  - Mean sleep outcomes across the 14-day window
  - Mean activity outcomes across the 14-day window
  - Data quality flags (n_sleep_days, n_activity_days)

The 14-day preceding window is standard in EMA/wearables research and aligns
with the bimonthly survey cadence of the TEXT LEMURS study.

Inputs:  concern_matched.csv, sleep_matched.csv, activity_matched.csv
Outputs: oura_aggregated.csv
         aggregation_log.txt
"""

import pandas as pd
import numpy as np

# -- FILE PATHS - update if running locally ---------------------------------
import config
CONCERN_FILE  = config.out_path('concern_matched.csv')
SLEEP_FILE    = config.out_path('sleep_matched.csv')
ACTIVITY_FILE = config.out_path('activity_matched.csv')
OUTPUT_FILE   = config.out_path('oura_aggregated.csv')
LOG_OUT       = config.out_path('aggregation_log.txt')

WINDOW_DAYS   = 14   # days preceding survey date to aggregate
MIN_DAYS      = 3    # minimum valid days to compute a window aggregate
# ---------------------------------------------------------------------------

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

def section(title):
    log()
    log('=' * 65)
    log(f'  {title}')
    log('=' * 65)

# -- LOAD -------------------------------------------------------------------
section('LOADING FILES')

concern  = pd.read_csv(CONCERN_FILE,  parse_dates=['survey_date'])
sleep    = pd.read_csv(SLEEP_FILE,    parse_dates=['summary_date'])
activity = pd.read_csv(ACTIVITY_FILE, parse_dates=['summary_date'])

concern['survey_date'] = concern['survey_date'].dt.normalize()

log(f'  Concern:  {concern["record_id"].nunique():>5} participants  |  {len(concern):>6,} rows')
log(f'  Sleep:    {sleep["record_id"].nunique():>5} participants  |  {len(sleep):>6,} rows')
log(f'  Activity: {activity["record_id"].nunique():>5} participants  |  {len(activity):>6,} rows')
log(f'\n  Window:   {WINDOW_DAYS} days preceding survey date')
log(f'  Min valid days per window: {MIN_DAYS}')

# -- OUTCOME COLUMNS --------------------------------------------------------
SLEEP_COLS = [
    'total',        # seconds -> will convert to hours
    'efficiency',   # percent
    'rem',          # seconds -> will convert to hours
    'deep',         # seconds -> will convert to hours
    'rmssd',        # milliseconds
    'hr_lowest',    # bpm
    'restless',     # percent
    'onset_latency' # seconds -> will convert to minutes
]

ACTIVITY_COLS = [
    'steps',
    'cal_active',
    'inactive',
    'inactivity_alerts',
    'average_met',
    'met_min_medium',
    'met_min_high'
]

# -- AGGREGATION ------------------------------------------------------------
section(f'AGGREGATING ({WINDOW_DAYS}-DAY PRECEDING WINDOW)')
log('  Processing...')

sleep_rows    = []
activity_rows = []

for i, row in concern.iterrows():
    pid         = row['record_id']
    week        = row['Week']
    survey_dt   = row['survey_date']

    if pd.isna(survey_dt):
        continue

    win_end   = survey_dt
    win_start = survey_dt - pd.Timedelta(days=WINDOW_DAYS - 1)

    # -- Sleep window --
    s = sleep[
        (sleep['record_id']    == pid) &
        (sleep['summary_date'] >= win_start) &
        (sleep['summary_date'] <= win_end)
    ]

    s_row = {'record_id': pid, 'Week': week, 'n_sleep_days': len(s)}
    for col in SLEEP_COLS:
        s_row[f'sleep_{col}'] = s[col].mean() if len(s) >= MIN_DAYS else np.nan
    sleep_rows.append(s_row)

    # -- Activity window --
    a = activity[
        (activity['record_id']    == pid) &
        (activity['summary_date'] >= win_start) &
        (activity['summary_date'] <= win_end)
    ]

    a_row = {'record_id': pid, 'Week': week, 'n_activity_days': len(a)}
    for col in ACTIVITY_COLS:
        a_row[f'act_{col}'] = a[col].mean() if len(a) >= MIN_DAYS else np.nan
    activity_rows.append(a_row)

sleep_agg    = pd.DataFrame(sleep_rows)
activity_agg = pd.DataFrame(activity_rows)

log(f'  Done. {len(sleep_agg)} person-wave rows created')

# -- UNIT CONVERSIONS -------------------------------------------------------
section('UNIT CONVERSIONS')

sleep_agg['sleep_total_hrs']          = sleep_agg['sleep_total']          / 3600
sleep_agg['sleep_rem_hrs']            = sleep_agg['sleep_rem']             / 3600
sleep_agg['sleep_deep_hrs']           = sleep_agg['sleep_deep']            / 3600
sleep_agg['sleep_onset_latency_min']  = sleep_agg['sleep_onset_latency']   / 60

log('  sleep_total       -> sleep_total_hrs          (seconds ÷ 3600)')
log('  sleep_rem         -> sleep_rem_hrs            (seconds ÷ 3600)')
log('  sleep_deep        -> sleep_deep_hrs           (seconds ÷ 3600)')
log('  sleep_onset_latency -> sleep_onset_latency_min (seconds ÷ 60)')
log('  All other columns unchanged (efficiency=%, rmssd=ms, hr_lowest=bpm)')

# -- MERGE CONCERN + SLEEP + ACTIVITY --------------------------------------
section('MERGING INTO ANALYTIC DATASET')

analytic = concern[[
    'record_id', 'Week', 'survey_date',
    'concern_present', 'concern_text_cleaned'
]].merge(
    sleep_agg, on=['record_id', 'Week'], how='left'
).merge(
    activity_agg, on=['record_id', 'Week'], how='left'
)

log(f'  Rows:         {len(analytic):,}')
log(f'  Participants: {analytic["record_id"].nunique()}')
log(f'  Columns:      {analytic.shape[1]}')

# -- COVERAGE REPORT --------------------------------------------------------
section('WINDOW COVERAGE')

log(f'  Sleep days per window:')
log(f'    Mean:                {sleep_agg["n_sleep_days"].mean():.1f}')
log(f'    Median:              {sleep_agg["n_sleep_days"].median():.0f}')
log(f'    Windows with 0 days: {(sleep_agg["n_sleep_days"]==0).sum():>5}  ({(sleep_agg["n_sleep_days"]==0).mean()*100:.1f}%)')
log(f'    Windows < {MIN_DAYS} days:   {(sleep_agg["n_sleep_days"]<MIN_DAYS).sum():>5}  ({(sleep_agg["n_sleep_days"]<MIN_DAYS).mean()*100:.1f}%)  -> set to NaN')
log(f'    Windows >= {MIN_DAYS} days:  {(sleep_agg["n_sleep_days"]>=MIN_DAYS).sum():>5}  ({(sleep_agg["n_sleep_days"]>=MIN_DAYS).mean()*100:.1f}%)  -> valid')

log(f'\n  Activity days per window:')
log(f'    Mean:                {activity_agg["n_activity_days"].mean():.1f}')
log(f'    Median:              {activity_agg["n_activity_days"].median():.0f}')
log(f'    Windows with 0 days: {(activity_agg["n_activity_days"]==0).sum():>5}  ({(activity_agg["n_activity_days"]==0).mean()*100:.1f}%)')
log(f'    Windows < {MIN_DAYS} days:   {(activity_agg["n_activity_days"]<MIN_DAYS).sum():>5}  ({(activity_agg["n_activity_days"]<MIN_DAYS).mean()*100:.1f}%)  -> set to NaN')
log(f'    Windows >= {MIN_DAYS} days:  {(activity_agg["n_activity_days"]>=MIN_DAYS).sum():>5}  ({(activity_agg["n_activity_days"]>=MIN_DAYS).mean()*100:.1f}%)  -> valid')

# -- MISSING DATA PER OUTCOME -----------------------------------------------
section('MISSING DATA PER OUTCOME VARIABLE')

outcome_cols = [
    'sleep_total_hrs', 'sleep_efficiency', 'sleep_rem_hrs', 'sleep_deep_hrs',
    'sleep_rmssd', 'sleep_hr_lowest', 'sleep_restless', 'sleep_onset_latency_min',
    'act_steps', 'act_cal_active', 'act_inactive', 'act_inactivity_alerts',
    'act_average_met', 'act_met_min_medium', 'act_met_min_high'
]

log(f'  {"Variable":<28} {"N valid":>8}  {"N missing":>10}  {"% missing":>10}')
log(f'  {"-"*28}  {"-"*8}  {"-"*10}  {"-"*10}')
for col in outcome_cols:
    if col not in analytic.columns:
        continue
    n_valid   = analytic[col].notna().sum()
    n_missing = analytic[col].isna().sum()
    pct       = n_missing / len(analytic) * 100
    flag      = '  <- HIGH' if pct > 20 else ''
    log(f'  {col:<28} {n_valid:>8,}  {n_missing:>10,}  {pct:>9.1f}%{flag}')

# -- DESCRIPTIVE STATISTICS -------------------------------------------------
section('OUTCOME DESCRIPTIVES (valid windows only)')

log(f'  {"Variable":<28} {"Mean":>8}  {"SD":>8}  {"Min":>8}  {"Max":>8}')
log(f'  {"-"*28}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}')
for col in outcome_cols:
    if col not in analytic.columns:
        continue
    s = analytic[col].dropna()
    if len(s) == 0:
        continue
    log(f'  {col:<28} {s.mean():>8.2f}  {s.std():>8.2f}  {s.min():>8.2f}  {s.max():>8.2f}')

# -- SAVE -------------------------------------------------------------------
section('SAVING')

analytic.to_csv(OUTPUT_FILE, index=False)
log(f'  Saved: {OUTPUT_FILE}')
log(f'  Shape: {analytic.shape[0]} rows x {analytic.shape[1]} columns')
log()
log(f'  Columns in output:')
for col in analytic.columns:
    log(f'    {col}')

# -- NEXT STEP NOTE ---------------------------------------------------------
section('NEXT STEPS')
log('  1. Run SEANCE locally on concern_matched.csv to get seance_features.csv')
log('  2. Merge seance_features.csv onto oura_aggregated.csv by record_id + Week')
log('  3. Run mixed effects models: SEANCE features -> Oura outcomes')

# -- SAVE LOG ---------------------------------------------------------------
with open(LOG_OUT, 'w') as f:
    f.write('\n'.join(lines))

log(f'\n  Log saved to: {LOG_OUT}')