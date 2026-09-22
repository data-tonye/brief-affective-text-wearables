#!/usr/bin/env python3
"""
TEXT LEMURS - SEANCE Feature Extraction
=========================================
Runs concern text through SEANCE and saves one row per participant
per wave with all linguistic features, ready to merge with oura_aggregated.csv.

Input:   concern_matched.csv
Output:  seance_features.csv

Place this file in the same folder as:
  - concern_matched.csv
  - seance_adapter.py
  - SEANCE_1_2_0_Py3/ directory
"""

import pandas as pd
from seance_adapter import ConcernSeanceProcessor

# -- CONFIG -----------------------------------------------------------------
import config
CONCERN_FILE = config.out_path('concern_matched.csv')
OUTPUT_FILE  = config.out_path('seance_features.csv')
SEANCE_DIR   = config.SEANCE_DIR
BATCH_SIZE   = 50
# ---------------------------------------------------------------------------

# -- LOAD -------------------------------------------------------------------
print("Loading concern text...")
df = pd.read_csv(CONCERN_FILE)
print(f"  {len(df)} rows | {df['record_id'].nunique()} participants")

# -- PREPARE CONVERSATIONS --------------------------------------------------
# SEANCE adapter expects a list of dicts with 'user_conversations' key
# We include record_id and Week so we can merge back after processing
print("\nPreparing texts for SEANCE...")

conversations = []
for _, row in df.iterrows():
    text = str(row['concern_text_cleaned']).strip() if pd.notna(row['concern_text_cleaned']) else ''
    conversations.append({
        'user': f"{row['record_id']}__W{row['Week']}",  # unique key: ID + wave
        'user_conversations': text if text else 'nothing'  # SEANCE needs non-empty string
    })

print(f"  {len(conversations)} concern responses prepared")

# -- RUN SEANCE -------------------------------------------------------------
print("\nInitialising SEANCE...")
processor = ConcernSeanceProcessor(seance_dir=SEANCE_DIR)

print("Preparing data...")
prepared = processor.prepare_texts(conversations, min_words=0)

print(f"Running SEANCE on {len(prepared)} texts (batch size {BATCH_SIZE})...")
seance_results = processor.process_conversations(
    prepared,
    output_path=None,
    batch_size=BATCH_SIZE
)

# -- RESTORE record_id AND Week ---------------------------------------------
# Split the composite key back into record_id and Week
print("\nRestoring record_id and Week columns...")
seance_results[['record_id', 'Week']] = (
    seance_results['user']
    .str.split('__W', expand=True)
    .rename(columns={0: 'record_id', 1: 'Week'})
)
seance_results['record_id'] = seance_results['record_id'].astype(int)
seance_results['Week']      = seance_results['Week'].astype(int)
seance_results = seance_results.drop(columns=['user'])

# -- MERGE BACK ORIGINAL METADATA ------------------------------------------
print("Merging back concern metadata...")
final = df[['record_id', 'Week', 'concern_present', 'concern_text_cleaned']].merge(
    seance_results,
    on=['record_id', 'Week'],
    how='left'
)

# -- SAVE -------------------------------------------------------------------
final.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*60}")
print("Done.")
print(f"  Rows:     {len(final)}")
print(f"  Columns:  {final.shape[1]}")
print(f"  Saved to: {OUTPUT_FILE}")
print('='*60)