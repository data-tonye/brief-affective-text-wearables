#!/usr/bin/env python3
"""
Concern text cleaning and concern_present flagging (pipeline step 0).
Produces: concern_cleaned.csv (feeds every downstream step; §3.2.1)

Applies exact-match spelling corrections and "no concern" coding to produce
concern_cleaned.csv from the raw bimonthly survey export.

Provenance: this file is a reconstruction of the preprocessing (the original
interactive script was not saved). It was verified to reproduce
dataset/concern_cleaned.csv row for row (concern_text_cleaned and
concern_present identical on all 4,052 rows). The annotated .xlsx it writes
differs from the earlier hand-reviewed workbook on 14 rows of cleaned text;
the .csv is what the pipeline reads.

Participant text: the two correction dictionaries this script applies
(`typo_corrections`, `nothing_equiv`) are built from short fragments of
participants' free-text responses, so they are not committed to this
repository - see corrections.example.json for the format. Running this
script without your own corrections.json still produces concern_cleaned.csv,
just without the typo-correction and equivalence-class steps applied (the
"no concern" flag will undercount responses typed as e.g. "nuthin bbg"
rather than "nothing").

Input:  raw bimonthly survey export (e.g. concern_text_new.csv), with a
        free-text concern column, a record_id column, and a survey-week
        column.
Output: concern_cleaned.csv with concern_text_cleaned and concern_present
        columns added, plus an annotated concern_text_cleaned.xlsx with a
        correction log sheet.
"""

import json
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

import config

# -- CONFIG - adjust to match your actual raw file --------------------------
INPUT_FILE = config.data_path("concern_text_new.csv")   # raw survey export
CONCERN_COL = "Y23_BiMonthly_CurrentConcern"
OUTPUT_CSV = config.out_path("concern_cleaned.csv")
OUTPUT_XLSX = config.out_path("concern_text_cleaned.xlsx")

# -- Participant-text correction dictionaries (not committed) ---------------
# See corrections.example.json for the expected format. Point
# TEXT_LEMURS_CORRECTIONS_FILE at your own copy (available from the
# corresponding author under the data use agreement), or leave unset to run
# without typo correction / "no concern" equivalence classing.
CORRECTIONS_FILE = os.environ.get("TEXT_LEMURS_CORRECTIONS_FILE", "corrections.json")
if os.path.exists(CORRECTIONS_FILE):
    with open(CORRECTIONS_FILE) as f:
        _corrections = json.load(f)
    typo_corrections = _corrections.get("typo_corrections", {})
    nothing_equiv = set(_corrections.get("nothing_equiv", []))
else:
    print(f"No corrections file at {CORRECTIONS_FILE} - running without "
          f"typo correction or 'no concern' equivalence classing.")
    typo_corrections = {}
    nothing_equiv = {
        "no", "nope", "n/a", "na", "none", "nothing", "not much",
        "not sure", "unsure", "idk",
    }


def clean_text(text):
    """Apply exact-match typo correction to a single response."""
    if pd.isna(text):
        return text
    stripped = str(text).strip()
    lower = stripped.lower()
    if lower in typo_corrections:
        return typo_corrections[lower]
    return stripped


def flag_concern(text):
    """0 = blank or a 'nothing'-equivalent response, 1 = substantive concern."""
    if pd.isna(text) or str(text).strip() == "":
        return 0
    return 0 if str(text).strip().lower() in nothing_equiv else 1


def main():
    df = pd.read_csv(INPUT_FILE)
    df["concern_text_original"] = df[CONCERN_COL]
    df["concern_text_cleaned"] = df[CONCERN_COL].apply(clean_text)
    df["concern_present"] = df["concern_text_cleaned"].apply(flag_concern)

    # -- Correction log --------------------------------------------------
    corrections = []
    for _, row in df.iterrows():
        orig = row["concern_text_original"]
        cleaned = row["concern_text_cleaned"]
        if pd.notna(orig) and pd.notna(cleaned) and str(orig).strip() != str(cleaned).strip():
            corrections.append({
                "record_id": row.get("record_id", row.get("Record.ID")),
                "Week": row.get("Week"),
                "original_text": orig,
                "corrected_text": cleaned,
            })
    corr_df = pd.DataFrame(corrections)

    print(f"Total rows: {len(df)}")
    print(f"Total corrections made: {len(corr_df)}")
    print(f"concern_present=1: {df['concern_present'].sum()}")
    print(f"concern_present=0: {(df['concern_present'] == 0).sum()}")

    # -- Write CSV (this is what the downstream pipeline reads) ----------
    df.to_csv(OUTPUT_CSV, index=False)

    # -- Write annotated Excel workbook for human review ------------------
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Cleaned Data"

    out_cols = ["record_id", "Week", "concern_text_original",
                "concern_text_cleaned", "concern_present"]
    out_cols = [c for c in out_cols if c in df.columns]

    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=11)
    header_fill = PatternFill("solid", start_color="4472C4")
    body_font = Font(name="Arial", size=10)

    for c_idx, col_name in enumerate(out_cols, 1):
        cell = ws1.cell(row=1, column=c_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for r_idx, row in enumerate(df[out_cols].itertuples(index=False), 2):
        for c_idx, val in enumerate(row, 1):
            cell = ws1.cell(row=r_idx, column=c_idx, value=val)
            cell.font = body_font
            if out_cols[c_idx - 1] == "concern_present" and val == 0:
                cell.fill = PatternFill("solid", start_color="FFF2CC")
            if out_cols[c_idx - 1] == "concern_text_cleaned":
                orig_col_idx = out_cols.index("concern_text_original") + 1
                orig_val = ws1.cell(row=r_idx, column=orig_col_idx).value
                if orig_val and val and str(orig_val).strip() != str(val).strip():
                    cell.fill = PatternFill("solid", start_color="E2EFDA")

    for i, w in enumerate([12, 8, 45, 45, 16][:len(out_cols)], 1):
        ws1.column_dimensions[get_column_letter(i)].width = w
    ws1.freeze_panes = "A2"

    ws2 = wb.create_sheet("Correction Log")
    log_cols = ["record_id", "Week", "original_text", "corrected_text"]
    log_cols = [c for c in log_cols if c in corr_df.columns]
    for c_idx, col_name in enumerate(log_cols, 1):
        cell = ws2.cell(row=1, column=c_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for r_idx, row in enumerate(corr_df[log_cols].itertuples(index=False), 2):
        for c_idx, val in enumerate(row, 1):
            cell = ws2.cell(row=r_idx, column=c_idx, value=val)
            cell.font = body_font
    for i, w in enumerate([12, 8, 45, 45][:len(log_cols)], 1):
        ws2.column_dimensions[get_column_letter(i)].width = w
    ws2.freeze_panes = "A2"

    wb.save(OUTPUT_XLSX)
    print(f"Wrote {OUTPUT_CSV} and {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
