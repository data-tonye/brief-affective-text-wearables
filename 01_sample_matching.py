#!/usr/bin/env python3
"""
TEXT LEMURS - Sample Matching
==============================
Matches concern text participants to Oura (sleep + activity) participants
by record_id, applies the minimum timepoint threshold, deduplicates to one
row per person-wave, and saves three matched output files ready for
aggregation and analysis.

Inputs:  concern_clean.csv, sleep_clean.csv, activity_clean.csv
Outputs: concern_matched.csv, sleep_matched.csv, activity_matched.csv
         sample_matching_report.txt
"""

import pandas as pd

# -- FILE PATHS - update if running locally ---------------------------------
import config
CONCERN_FILE     = config.data_path('concern_cleaned.csv')
SLEEP_FILE       = config.data_path('sleep_data.csv')
ACTIVITY_FILE    = config.data_path('activity_data.csv')

CONCERN_OUT      = config.out_path('concern_matched.csv')
SLEEP_OUT        = config.out_path('sleep_matched.csv')
ACTIVITY_OUT     = config.out_path('activity_matched.csv')
REPORT_OUT       = config.out_path('sample_matching_report.txt')

MIN_TIMEPOINTS   = 3   # minimum bimonthly waves per participant for LME
# ---------------------------------------------------------------------------

lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# -- LOAD -------------------------------------------------------------------
log('Loading files...')
concern  = pd.read_csv(CONCERN_FILE)
sleep    = pd.read_csv(SLEEP_FILE)
activity = pd.read_csv(ACTIVITY_FILE)

# Normalise concern column names to match pipeline conventions
# (raw file uses 'Record.ID' and 'Date' instead of 'record_id' and 'survey_date')
# Rename columns to pipeline conventions
# Handles both raw file formats (Record.ID or record_id)
concern = concern.rename(columns={
    'Record.ID': 'record_id',    # raw file variant
    'startDate': 'survey_date',  # actual timestamp used for deduplication
    'Date':      'survey_id',    # numeric survey ID, not a date
})
# Verify survey_date exists after rename
if 'survey_date' not in concern.columns:
    raise KeyError(
        f"Could not find 'startDate' to rename. "
        f"Available columns: {list(concern.columns)}"
    )

log(f'  Concern:  {concern["record_id"].nunique():>4} participants, {len(concern):>6,} rows')
log(f'  Sleep:    {sleep["record_id"].nunique():>4} participants, {len(sleep):>6,} rows')
log(f'  Activity: {activity["record_id"].nunique():>4} participants, {len(activity):>6,} rows')

# -- STEP 1: FIND MATCHED IDs -----------------------------------------------
log('\nStep 1: Intersecting record_ids...')

concern_ids  = set(concern['record_id'].unique())
oura_ids     = set(sleep['record_id'].unique()) | set(activity['record_id'].unique())
matched_ids  = concern_ids & oura_ids

log(f'  Concern only (no Oura):      {len(concern_ids - oura_ids):>4}')
log(f'  Oura only (no concern text): {len(oura_ids - concern_ids):>4}')
log(f'  Matched both:                {len(matched_ids):>4}')

# -- STEP 2: APPLY MINIMUM TIMEPOINT THRESHOLD -----------------------------
log(f'\nStep 2: Applying minimum timepoint threshold (>= {MIN_TIMEPOINTS} waves)...')

concern_matched = concern[concern['record_id'].isin(matched_ids)].copy()
timepoints = concern_matched.groupby('record_id')['Week'].nunique()

sufficient_ids = set(timepoints[timepoints >= MIN_TIMEPOINTS].index)
excluded_ids   = set(timepoints[timepoints < MIN_TIMEPOINTS].index)

log(f'  Participants with >= {MIN_TIMEPOINTS} timepoints: {len(sufficient_ids):>4}  <- analytic sample')
log(f'  Participants with <  {MIN_TIMEPOINTS} timepoints: {len(excluded_ids):>4}  <- excluded from LME')

# -- STEP 2.5: DEDUPLICATE TO ONE ROW PER PERSON-WAVE ----------------------
log('\nStep 2.5: Deduplicating to one row per person-wave...')

# Concern: keep most recent survey response per wave
concern_before = len(concern_matched)
concern_matched['survey_date'] = pd.to_datetime(concern_matched['survey_date'])
concern_matched = (concern_matched
    .sort_values(['record_id', 'Week', 'survey_date'],
                 ascending=[True, True, False])
    .drop_duplicates(subset=['record_id', 'Week'], keep='first'))
# Note: survey_date = startDate (actual response timestamp)
# Most recent response per wave is kept
log(f'  Concern:  {concern_before:,} -> {len(concern_matched):,} rows '
    f'({concern_before - len(concern_matched)} duplicates removed)')
log(f'  Rule: kept most recent survey_date per person-wave')

# NOTE: Sleep and activity files are daily-level at this stage - they do not
# have a Week column yet. Wave-level deduplication (one row per person-wave)
# must be applied in the aggregation script after Week windows are computed.

# Verify concern dedup only - sleep/activity dedup happens in aggregation script
dupes = concern_matched.groupby(['record_id', 'Week']).size()
n_dupes = (dupes > 1).sum()
if n_dupes > 0:
    log(f'  WARNING: Concern still has {n_dupes} person-waves with >1 row - check data')
else:
    log(f'  Concern: confirmed 1 row per person-wave')

# -- STEP 3: FILTER ALL THREE FILES TO ANALYTIC SAMPLE ---------------------
log('\nStep 3: Filtering all files to analytic sample...')

concern_out  = concern_matched[concern_matched['record_id'].isin(sufficient_ids)].copy()
sleep_out    = sleep[sleep['record_id'].isin(sufficient_ids)].copy()
activity_out = activity[activity['record_id'].isin(sufficient_ids)].copy()

log(f'  Concern:  {concern_out["record_id"].nunique():>4} participants, {len(concern_out):>6,} rows')
log(f'  Sleep:    {sleep_out["record_id"].nunique():>4} participants, {len(sleep_out):>6,} rows')
log(f'  Activity: {activity_out["record_id"].nunique():>4} participants, {len(activity_out):>6,} rows')

# -- STEP 4: SAVE -----------------------------------------------------------
concern_out.to_csv(CONCERN_OUT,  index=False)
sleep_out.to_csv(SLEEP_OUT,      index=False)
activity_out.to_csv(ACTIVITY_OUT, index=False)

log(f'\nSaved:')
log(f'  {CONCERN_OUT}')
log(f'  {SLEEP_OUT}')
log(f'  {ACTIVITY_OUT}')

# -- REPORT -----------------------------------------------------------------
log('\n-- Timepoint distribution (analytic sample) --')
tp_final = concern_out.groupby('record_id')['Week'].nunique()
log(f'  Mean:   {tp_final.mean():.1f}')
log(f'  Median: {tp_final.median():.0f}')
log(f'  Min:    {tp_final.min()}')
log(f'  Max:    {tp_final.max()}')

log(f'\n-- Excluded participant IDs (< {MIN_TIMEPOINTS} timepoints) --')
log(f'  {sorted(excluded_ids)}')

with open(REPORT_OUT, 'w') as f:
    f.write('\n'.join(lines))

log(f'\nReport saved to: {REPORT_OUT}')
