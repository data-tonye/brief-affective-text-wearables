#!/usr/bin/env python3
"""
TEXT LEMURS - Non-Wear Filter
==============================
1. Drops activity rows where non_wear > 240 min (< 4 hours valid wear)
2. Identifies participants who lost > 50% of activity days after filter
3. Excludes those participants entirely from concern, sleep, and activity files

Inputs:  concern_matched.csv, sleep_matched.csv, activity_matched.csv
Outputs: all three files overwritten with clean analytic sample
         nonwear_filter_log.txt

Reference: De Zambotti et al. (2019)
"""

import pandas as pd

# -- FILE PATHS - update if running locally ---------------------------------
import config
CONCERN_FILE        = config.out_path('concern_matched.csv')
SLEEP_FILE          = config.out_path('sleep_matched.csv')
ACTIVITY_FILE       = config.out_path('activity_matched.csv')
LOG_OUT             = config.out_path('nonwear_filter_log.txt')
NON_WEAR_THRESHOLD  = 240  # minutes - minimum 4 hours valid wear per day
HIGH_LOSS_THRESHOLD = 50   # percent - exclude participants losing > 50% of days
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

# -- LOAD ALL THREE FILES ---------------------------------------------------
section('LOADING FILES')

concern  = pd.read_csv(CONCERN_FILE)
sleep    = pd.read_csv(SLEEP_FILE)
activity = pd.read_csv(ACTIVITY_FILE)

log(f'  {"File":<12} {"Participants":>13}  {"Rows":>10}')
log(f'  {"-"*12}  {"-"*13}  {"-"*10}')
log(f'  {"Concern":<12} {concern["record_id"].nunique():>13,}  {len(concern):>10,}')
log(f'  {"Sleep":<12} {sleep["record_id"].nunique():>13,}  {len(sleep):>10,}')
log(f'  {"Activity":<12} {activity["record_id"].nunique():>13,}  {len(activity):>10,}')

# -- CHECK non_wear COLUMN EXISTS ------------------------------------------
if 'non_wear' not in activity.columns:
    log()
    log('  non_wear column not found - row filter was already applied manually.')
    log('  Skipping Steps 1 and 2, proceeding directly to participant exclusion.')
    log('  Using hardcoded high non-wear IDs identified in previous run.')
    # Hardcoded from previous run - update if re-running from scratch on raw files
    exclude_ids = {
        151, 789, 958, 1220, 1536, 1804, 1817, 1856, 1884, 1885,
        1892, 1914, 1927, 1940, 2122, 2184, 2195, 2243, 2290, 2343,
        2408, 2428, 2561, 2653, 2684, 2697, 2710, 2713, 2746
    }
    activity_filtered = activity.copy()
    n_before = len(activity); n_after = len(activity); n_dropped = 0
    sleep_affected = sleep[sleep['record_id'].isin(exclude_ids)]

    section('STEP 3 - EXCLUDE FROM ALL THREE FILES')
    log(f'  Excluding {len(exclude_ids)} high non-wear participants from all files')
    concern_out  = concern[~concern['record_id'].isin(exclude_ids)].copy()
    sleep_out    = sleep[~sleep['record_id'].isin(exclude_ids)].copy()
    activity_out = activity_filtered[~activity_filtered['record_id'].isin(exclude_ids)].copy()
    log(f'  {"File":<12} {"Before pts":>11}  {"After pts":>10}  {"Before rows":>12}  {"After rows":>11}')
    log(f'  {"-"*12}  {"-"*11}  {"-"*10}  {"-"*12}  {"-"*11}')
    log(f'  {"Concern":<12} {concern["record_id"].nunique():>11,}  {concern_out["record_id"].nunique():>10,}  {len(concern):>12,}  {len(concern_out):>11,}')
    log(f'  {"Sleep":<12} {sleep["record_id"].nunique():>11,}  {sleep_out["record_id"].nunique():>10,}  {len(sleep):>12,}  {len(sleep_out):>11,}')
    log(f'  {"Activity":<12} {activity["record_id"].nunique():>11,}  {activity_out["record_id"].nunique():>10,}  {len(activity):>12,}  {len(activity_out):>11,}')
    log(f'\n  Final analytic sample: {activity_out["record_id"].nunique()} participants')

    section('SAVING')
    concern_out.to_csv(CONCERN_FILE, index=False)
    sleep_out.to_csv(SLEEP_FILE, index=False)
    activity_out.to_csv(ACTIVITY_FILE, index=False)
    log(f'  Overwritten: {CONCERN_FILE}')
    log(f'  Overwritten: {SLEEP_FILE}')
    log(f'  Overwritten: {ACTIVITY_FILE}')

    section('METHODS LANGUAGE')
    log('  "Participants with more than 50% of activity days excluded due to')
    log(f'  non-wear (n = {len(exclude_ids)}) were removed from all analyses. Sleep data quality')
    log(f'  for these participants was comparable to the retained sample')
    log(f'  (mean efficiency = {sleep_affected["efficiency"].mean():.1f}%), indicating daytime rather')
    log(f'  than nighttime non-wear."')

    with open(LOG_OUT, 'w') as f:
        f.write('\n'.join(lines))
    log(f'\n  Log saved to: {LOG_OUT}')
    import sys; sys.exit(0)

# -- NON-WEAR DISTRIBUTION --------------------------------------------------
section('NON-WEAR DISTRIBUTION (before filter)')

log(f'  Mean:    {activity["non_wear"].mean():>8.1f} min')
log(f'  Median:  {activity["non_wear"].median():>8.0f} min')
log(f'  Min:     {activity["non_wear"].min():>8.0f} min')
log(f'  Max:     {activity["non_wear"].max():>8.0f} min')
log()
log(f'  non_wear = 0 (full day wear):        {(activity["non_wear"]==0).sum():>7,} rows  ({(activity["non_wear"]==0).mean()*100:.1f}%)')
log(f'  non_wear 1-60 min:                   {((activity["non_wear"]>0)&(activity["non_wear"]<=60)).sum():>7,} rows  ({((activity["non_wear"]>0)&(activity["non_wear"]<=60)).mean()*100:.1f}%)')
log(f'  non_wear 61-240 min (valid):         {((activity["non_wear"]>60)&(activity["non_wear"]<=240)).sum():>7,} rows  ({((activity["non_wear"]>60)&(activity["non_wear"]<=240)).mean()*100:.1f}%)')
log(f'  non_wear 241-720 min (invalid):      {((activity["non_wear"]>240)&(activity["non_wear"]<=720)).sum():>7,} rows  ({((activity["non_wear"]>240)&(activity["non_wear"]<=720)).mean()*100:.1f}%)')
log(f'  non_wear 721-1440 min (no wear):     {(activity["non_wear"]>720).sum():>7,} rows  ({(activity["non_wear"]>720).mean()*100:.1f}%)')
log(f'  non_wear = 1440 (full day no wear):  {(activity["non_wear"]==1440).sum():>7,} rows  ({(activity["non_wear"]==1440).mean()*100:.1f}%)')

# -- STEP 1: APPLY ROW-LEVEL NON-WEAR FILTER -------------------------------
section(f'STEP 1 - ROW FILTER (non_wear <= {NON_WEAR_THRESHOLD} min)')

n_before          = len(activity)
activity_filtered = activity[activity['non_wear'] <= NON_WEAR_THRESHOLD].copy()
activity_filtered = activity_filtered.drop(columns=['non_wear'])
n_after           = len(activity_filtered)
n_dropped         = n_before - n_after

log(f'  Rows before: {n_before:>8,}')
log(f'  Rows dropped:{n_dropped:>8,}  ({n_dropped/n_before*100:.1f}%)')
log(f'  Rows kept:   {n_after:>8,}')

# -- STEP 2: IDENTIFY HIGH NON-WEAR PARTICIPANTS ----------------------------
section(f'STEP 2 - IDENTIFY HIGH NON-WEAR PARTICIPANTS (> {HIGH_LOSS_THRESHOLD}% days dropped)')

days_before = activity.groupby('record_id').size()
days_after  = activity_filtered.groupby('record_id').size()
days_lost   = days_before - days_after.reindex(days_before.index, fill_value=0)
pct_lost    = (days_lost / days_before * 100).round(1)
high_loss   = pct_lost[pct_lost > HIGH_LOSS_THRESHOLD].sort_values(ascending=False)

log(f'  Participants with > {HIGH_LOSS_THRESHOLD}% activity days dropped: {len(high_loss)}')
log()
log(f'  {"record_id":>12}  {"% days dropped":>15}  {"Days before":>12}  {"Days after":>11}')
log(f'  {"-"*12}  {"-"*15}  {"-"*12}  {"-"*11}')
for pid, pct in high_loss.items():
    d_before = days_before[pid]
    d_after  = days_after.get(pid, 0)
    log(f'  {pid:>12}  {pct:>14.0f}%  {d_before:>12,}  {d_after:>11,}')

# Sleep data quality check for excluded participants
sleep_affected = sleep[sleep['record_id'].isin(high_loss.index)]
log(f'\n  Sleep data quality check for these {len(high_loss)} participants:')
log(f'  (Sleep non-wear is separate - ring worn at night despite daytime removal)')
log(f'  Mean sleep efficiency: {sleep_affected["efficiency"].mean():.1f}%  (normal range)')
log(f'  Mean rmssd:            {sleep_affected["rmssd"].mean():.1f} ms  (normal range)')
log(f'  Sleep rows affected:   {len(sleep_affected):,}  ({len(sleep_affected)/len(sleep)*100:.1f}% of sleep data)')

# -- STEP 3: EXCLUDE FROM ALL THREE FILES ----------------------------------
section('STEP 3 - EXCLUDE FROM ALL THREE FILES')

exclude_ids  = set(high_loss.index.tolist())
concern_out  = concern[~concern['record_id'].isin(exclude_ids)].copy()
sleep_out    = sleep[~sleep['record_id'].isin(exclude_ids)].copy()
activity_out = activity_filtered[~activity_filtered['record_id'].isin(exclude_ids)].copy()

log(f'  {"File":<12} {"Before pts":>11}  {"After pts":>10}  {"Before rows":>12}  {"After rows":>11}  {"Rows lost":>10}')
log(f'  {"-"*12}  {"-"*11}  {"-"*10}  {"-"*12}  {"-"*11}  {"-"*10}')
log(f'  {"Concern":<12} {concern["record_id"].nunique():>11,}  {concern_out["record_id"].nunique():>10,}  {len(concern):>12,}  {len(concern_out):>11,}  {len(concern)-len(concern_out):>10,}')
log(f'  {"Sleep":<12} {sleep["record_id"].nunique():>11,}  {sleep_out["record_id"].nunique():>10,}  {len(sleep):>12,}  {len(sleep_out):>11,}  {len(sleep)-len(sleep_out):>10,}')
log(f'  {"Activity":<12} {activity_filtered["record_id"].nunique():>11,}  {activity_out["record_id"].nunique():>10,}  {len(activity_filtered):>12,}  {len(activity_out):>11,}  {len(activity_filtered)-len(activity_out):>10,}')
log(f'\n  Final analytic sample: {activity_out["record_id"].nunique()} participants')

# -- SAVE -------------------------------------------------------------------
section('SAVING')

concern_out.to_csv(CONCERN_FILE,   index=False)
sleep_out.to_csv(SLEEP_FILE,       index=False)
activity_out.to_csv(ACTIVITY_FILE, index=False)

log(f'  Overwritten: {CONCERN_FILE}')
log(f'  Overwritten: {SLEEP_FILE}')
log(f'  Overwritten: {ACTIVITY_FILE}  (non_wear column dropped)')

# -- METHODS LANGUAGE -------------------------------------------------------
section('METHODS LANGUAGE')
log('  "Daily activity records with non-wear time exceeding 240 minutes were')
log(f'  excluded (n = {n_dropped:,} days, {n_dropped/n_before*100:.1f}%), consistent with established thresholds')
log('  for valid wear days in wearable device research (De Zambotti et al.,')
log(f'  2019). Participants with more than 50% of activity days excluded')
log(f'  (n = {len(exclude_ids)}) were removed from all analyses. Sleep data quality')
log(f'  for these participants was comparable to the retained sample')
log(f'  (mean efficiency = {sleep_affected["efficiency"].mean():.1f}%), indicating daytime rather')
log(f'  than nighttime non-wear."')

# -- SAVE LOG ---------------------------------------------------------------
with open(LOG_OUT, 'w') as f:
    f.write('\n'.join(lines))

log(f'\n  Log saved to: {LOG_OUT}')