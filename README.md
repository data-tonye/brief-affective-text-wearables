# Brief Affective Text as a Complement to Wearable Sensing - Analysis Code

Analysis code for **"A Formative Study of Brief Affective Text as a Complement to Wearable Sensing for Longitudinal Student Health Monitoring"** (Harry, Hidalgo, Price, Feng, Stanton, Tompkins, Dodds, Fudolig, Bloomfield, Danforth).

This repository contains **code only**. No participant data, derived data, model outputs, or participant text are included - see [Data policy](#data-policy).

## Pipeline order

Run from the repository root. Every intermediate file is written under `TEXT_LEMURS_OUTPUT_DIR` (default `outputs/`); see `config.py`.

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 0 | `00_clean_concern_text.py` | raw survey export, `corrections.json` (optional, see below) | `concern_cleaned.csv` |
| 1 | `01_sample_matching.py` | `concern_cleaned.csv`, raw sleep/activity files | `*_matched.csv` |
| 2 | `02_nonwear_filter.py` | `*_matched.csv` | `*_matched.csv` (overwritten) |
| 3 | `03_aggregation.py` | `*_matched.csv` | `oura_aggregated.csv` |
| 4 | `04_seance_features.py` (uses `seance_adapter.py`) | `concern_matched.csv`, local SEANCE install | `seance_features.csv` |
| 5 | `table8_demographics.py` | `oura_aggregated.csv`, baseline demographics CSV | Table 8 (printed) |
| 6 | `table5_semester_decline.py` | `oura_aggregated.csv`, `seance_features.csv` | Table 5, word-count descriptives |
| 7 | `table6_seance_feature_models.py` | `seance_features.csv`, `oura_aggregated.csv` | Table 6, `model_results*.csv` |
| 8 | `seance_variance_and_dominance.py` (uses `r2_tools.py`) | `seance_features.csv`, `oura_aggregated.csv` | Table 7 (SEANCE column), Table 9 |
| 9 | `embeddings_pca_domain.py` | `oura_aggregated.csv` | `embedding_outputs/*` (embeddings, PCA, zero-shot domain scores) |
| 10 | `table12_13_pc_coefficient_models.py` | `embedding_outputs/*` | Tables 12–13 |
| 11 | `table14_domain_models.py` | `embedding_outputs/domain_classifications.csv`, `oura_aggregated.csv` | Table 14 |
| 12 | `embedding_variance_and_dominance.py` (uses `r2_tools.py`) | steps 7–10 outputs | Table 7 (RoBERTa/MentalRoBERTa columns), Tables 10–11 |
| 13 | `bootstrap_best_method_ci.py` | steps 8, 12 outputs | Table 7 "Best" column bootstrap CIs |
| 14 | `pc_interpretation.py` | embedding, results and SEANCE files | §4.9 PC interpretation, appendix interpretation tables |
| 15 | `figure2.py` -> `figure3.py`, `figure4.py`, `figure6.py` | embedding files, `figures/tsne_data.csv` | Figures 2–4, 6, Supplementary Figure 5 |

`figure1_diagram.js` (pptx generator, run with `node`) produces Figure 1 from summary statistics only - it does not read participant data.

Note for step 12: `embedding_variance_and_dominance.py` reads the SEANCE decomposition written by step 8 (`outputs/variance_decomposition.csv`); run step 8 first.

## Table / Figure -> script

| Paper item | Script(s) |
|---|---|
| Table 5 | `table5_semester_decline.py` |
| Table 6 | `table6_seance_feature_models.py` |
| Table 7 | `seance_variance_and_dominance.py` (SEANCE column) + `embedding_variance_and_dominance.py` (RoBERTa/MentalRoBERTa columns) + `bootstrap_best_method_ci.py` (Best column CIs) |
| Table 8 | `table8_demographics.py` |
| Table 9 | `seance_variance_and_dominance.py` |
| Tables 10–11 | `embedding_variance_and_dominance.py` |
| Tables 12–13 | `table12_13_pc_coefficient_models.py` |
| Table 14 | `table14_domain_models.py` |
| Figure 1 | `figure1_diagram.js` |
| Figure 2 | `figure2.py` |
| Figure 3 | `figure3.py` |
| Figure 4 | `figure4.py` |
| Figure 5 (Supplementary) | `figure3.py` (both figures share plotting logic) |
| Figure 6 | `figure6.py` |
| §4.9 PC interpretation | `pc_interpretation.py` |

## Supplying your own data

Set the paths with environment variables (defaults in parentheses):

- `TEXT_LEMURS_DATA_DIR` (`data/`) - raw inputs
- `TEXT_LEMURS_OUTPUT_DIR` (`outputs/`) - everything the pipeline creates
- `TEXT_LEMURS_BASELINE_FILE` (`<DATA_DIR>/baseline_demographics.csv`)
- `SEANCE_DIR` (`third_party/SEANCE_1_2_0_Py3`) - a local copy of SEANCE 1.2.0, downloaded from
  [linguisticanalysistools.org/seance.html](https://www.linguisticanalysistools.org/seance.html)
  (Kyle & Crossley). Not included here: SEANCE is licensed CC BY-NC-SA 4.0 and bundles several
  third-party lexicons (e.g. ANEW) with their own separate usage terms, so download and license it
  directly from the source rather than from a copy in this repo.
- `TEXT_LEMURS_CORRECTIONS_FILE` (`corrections.json`) - see below

## Data policy

No participant data, no derived output files (CSVs, figures, model results), and no participant text are committed to this repository - `.gitignore` blocks all of it, and every table and figure number reported here is the one already published in the paper.

The one place participant text would otherwise appear in code is `00_clean_concern_text.py`'s spelling-correction step: two small dictionaries (`typo_corrections`, `nothing_equiv`) built from short fragments of participants' free-text responses (e.g. correcting "wchool" to "school"). Those dictionaries are **not included**. `00_clean_concern_text.py` loads them at runtime from `corrections.json` if present (format documented in `corrections.example.json`); without that file it still runs, just without typo correction or "no concern" equivalence classing applied. Researchers with legitimate data access can request the corresponding author's copy under the data use agreement.

De-identified aggregate data sharing beyond what is reported in the paper is subject to institutional review board restrictions; requests may be directed to the corresponding author.

## Requirements

```
pip install -r requirements.txt
```

`figure1_diagram.js` additionally requires Node.js and `pptxgenjs` (`npm install pptxgenjs`).
