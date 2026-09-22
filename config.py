"""Path configuration. Set env vars to point at your own (non-public) data.

TEXT_LEMURS_DATA_DIR      raw input files supplied by you (default: ./data)
TEXT_LEMURS_OUTPUT_DIR    all intermediate files and results (default: ./outputs)
TEXT_LEMURS_BASELINE_FILE baseline demographics CSV (default: <DATA_DIR>/baseline_demographics.csv)
SEANCE_DIR                local copy of SEANCE 1.2.0 (default: ./third_party/SEANCE_1_2_0_Py3)
"""
import os

DATA_DIR = os.environ.get("TEXT_LEMURS_DATA_DIR", "data")
OUTPUT_DIR = os.environ.get("TEXT_LEMURS_OUTPUT_DIR", "outputs")
SEANCE_DIR = os.environ.get("SEANCE_DIR", os.path.join("third_party", "SEANCE_1_2_0_Py3"))
BASELINE_FILE = os.environ.get(
    "TEXT_LEMURS_BASELINE_FILE", os.path.join(DATA_DIR, "baseline_demographics.csv")
)


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


def out_path(*parts):
    p = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p
