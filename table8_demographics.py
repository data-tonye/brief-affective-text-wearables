#!/usr/bin/env python3
"""
Y23 Cohort - Demographic Summary for Wearable Subsample
Cross-references oura_aggregated.csv with Y23 baseline data
to produce demographics for the 458 wearable participants.
"""

import pandas as pd

# -- Load data ------------------------------------------------------------------
import config
oura = pd.read_csv(config.out_path('oura_aggregated.csv'))
baseline = pd.read_csv(config.BASELINE_FILE)

# -- Cross-reference ------------------------------------------------------------
oura_ids = oura['record_id'].unique()
df = baseline[baseline['Record.ID'].isin(oura_ids)].copy()

n_oura = len(oura_ids)
n_matched = len(df)
n_unmatched = n_oura - n_matched

print("=" * 60)
print("Y23 WEARABLE SUBSAMPLE - DEMOGRAPHIC SUMMARY")
print("=" * 60)
print(f"\nTotal wearable participants (oura_aggregated): {n_oura}")
print(f"Matched to baseline demographics:             {n_matched}")
print(f"No baseline record found:                     {n_unmatched}")

# -- Age ------------------------------------------------------------------------
print("\n--- AGE ---")
age = df['Y23_BL_Demographics_Age'].dropna()
print(f"  Mean: {age.mean():.1f} years")
print(f"  SD:   {age.std():.1f}")
print(f"  Min:  {age.min():.0f}  |  Max: {age.max():.0f}")

# -- Gender ---------------------------------------------------------------------
print("\n--- GENDER (participants may select multiple) ---")
gender_cols = {
    'Woman':       'Y23_BL_Demographics_Woman',
    'Man':         'Y23_BL_Demographics_Man',
    'Transgender': 'Y23_BL_Demographics_Transgender',
    'Genderqueer': 'Y23_BL_Demographics_Genderqueer',
    'Agender':     'Y23_BL_Demographics_Agender',
    'Other':       'Y23_BL_Demographics_OtherGender',
}
for label, col in gender_cols.items():
    count = df[col].sum()
    pct = count / n_matched * 100
    print(f"  {label:<15} {count:>4}  ({pct:.1f}%)")

# -- Race/Ethnicity -------------------------------------------------------------
print("\n--- RACE / ETHNICITY (participants may select multiple) ---")
race_cols = {
    'White':           'Y23_BL_Demographics_White',
    'Hispanic':        'Y23_BL_Demographics_Hispanic',
    'African American':'Y23_BL_Demographics_AfricanAmerican',
    'African Caribbean':'Y23_BL_Demographics_AfricanCaribbean',
    'East Asian':      'Y23_BL_Demographics_EastAsian',
    'Southeast Asian': 'Y23_BL_Demographics_SoutheastAsian',
    'Middle Eastern':  'Y23_BL_Demographics_MiddleEastern',
    'Native American': 'Y23_BL_Demographics_NativeAmerican',
    'European':        'Y23_BL_Demographics_European',
    'Other':           'Y23_BL_Demographics_OtherRace',
    'Pacific Islander': 'Y23_BL_Demographics_PacificIslander',
    'Other African':    'Y23_BL_Demographics_OtherAfrican',
}
for label, col in race_cols.items():
    count = df[col].sum()
    pct = count / n_matched * 100
    print(f"  {label:<22} {count:>4}  ({pct:.1f}%)")

# -- Sexual Orientation ---------------------------------------------------------
print("\n--- SEXUAL ORIENTATION (participants may select multiple) ---")
orient_cols = {
    'Straight':    'Y23_BL_Demographics_Straight',
    'Gay':         'Y23_BL_Demographics_Gay',
    'Bisexual':    'Y23_BL_Demographics_Bisexual',
    'Pansexual':   'Y23_BL_Demographics_Pansexual',
    'Asexual':     'Y23_BL_Demographics_Asexual',
    'Queer':       'Y23_BL_Demographics_Queer',
    'Questioning': 'Y23_BL_Demographics_Questioning',
    'Unsure':      'Y23_BL_Demographics_UnsureOrientation',
    'Other':       'Y23_BL_Demographics_AnotherOrientation',
}
for label, col in orient_cols.items():
    count = df[col].sum()
    pct = count / n_matched * 100
    print(f"  {label:<15} {count:>4}  ({pct:.1f}%)")

print("\n" + "=" * 60)
print("Note: Percentages calculated out of matched N =", n_matched)
print("Race/gender/orientation are multi-select - totals may exceed 100%.")
print("=" * 60)