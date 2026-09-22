#!/usr/bin/env python3
"""
Participant-cluster bootstrap of the language-block delta R2 for RoBERTa, MentalRoBERTa and SEANCE
(step 12b; run after embed_domanalysis.py).

For each outcome, participants are resampled with replacement and the language-block delta R2
(model with language features minus model with nwords_z + Week_z) is refitted for each method.
Reported: point estimates, 95% percentile intervals, and the interval of each between-method difference.

Two embedding language blocks are bootstrapped:
  allPCs  - all 127 / 100 principal components (the block used in embed_domanalysis.py); 100 resamples
  sigPCs  - only the PCs significant in embed_mixed2.py; 200 resamples. The PC list is held fixed, so
            selection variability is not reflected in these intervals.
Both are in-sample R2 values and are upper-biased relative to a cross-validated comparison.

Output: variance_dominance_outputs/bootstrap_R_vs_M.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from multiprocessing import Pool

import config
import r2_tools as rt

OUTCOMES = {
    "sleep_total_hrs": "Sleep Duration (hrs)", "sleep_efficiency": "Sleep Efficiency (%)",
    "sleep_rem_hrs": "REM Sleep (hrs)", "sleep_deep_hrs": "Deep Sleep (hrs)", "sleep_rmssd": "RMSSD (ms)",
    "sleep_onset_latency_min": "Sleep Onset Latency (min)", "act_steps": "Steps/day",
    "act_met_min_medium": "MET-min Medium", "act_met_min_high": "MET-min High",
}
SEANCE_FEATURES = ["Strong_GI", "Negative_EmoLex", "Sadness_EmoLex", "Fear_EmoLex",
                   "Valence", "Academ_GI", "Work_GI"]      # same seven features as decomposition.py
N_BOOT = {"sigPCs": 200, "allPCs": 100}


def resample(data, rng):
    ids = data["record_id"].unique()
    pick = rng.choice(ids, len(ids), replace=True)
    groups = {i: d for i, d in data.groupby("record_id")}
    parts = []
    for k, i in enumerate(pick):
        d = groups[i].copy()
        d["record_id"] = k
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def delta(data, outcome, terms, fitter):
    base = rt.marginal_r2(fitter(f"{outcome} ~ nwords_z + Week_z", data))
    full = rt.marginal_r2(fitter(f"{outcome} ~ nwords_z + Week_z + " + " + ".join(terms), data))
    return full - base


def job(args):
    variant, outcome, seed = args
    rng = np.random.default_rng(seed)
    emb = rt.load_embed_frame()
    sea = rt.load_seance_frame()
    for c in SEANCE_FEATURES + ["Week", "nwords"]:
        sea[f"{c}_z"] = rt.zscore(sea[c])
    pcs = {mk: (rt.sig_pcs(mk, outcome) if variant == "sigPCs" else rt.all_pcs(emb, mk)) for mk in rt.MODELS}
    s_terms = [f"{f}_z" for f in SEANCE_FEATURES]
    d_emb = {mk: emb[[outcome, "record_id", "Week_z", "nwords_z"] + pcs[mk]].dropna() for mk in rt.MODELS}
    d_sea = sea[[outcome, "record_id", "Week_z", "nwords_z"] + s_terms].dropna()
    point = {mk: delta(d_emb[mk], outcome, pcs[mk], rt.fit_ml) for mk in rt.MODELS}
    point["seance"] = delta(d_sea, outcome, s_terms, rt.fit_reml)

    draws = []
    for _ in range(N_BOOT[variant]):
        s = int(rng.integers(1e9))
        try:
            v = {mk: delta(resample(d_emb[mk], np.random.default_rng(s)), outcome, pcs[mk], rt.fit_ml)
                 for mk in rt.MODELS}
            v["seance"] = delta(resample(d_sea, np.random.default_rng(s)), outcome, s_terms, rt.fit_reml)
            draws.append(v)
        except Exception:
            continue
    v = pd.DataFrame(draws)

    def ci(x):
        return x.quantile(.025), x.quantile(.975)

    out = dict(variant=variant, outcome_var=outcome, outcome=OUTCOMES[outcome], n_boot=len(v))
    for mk in ["roberta", "mentalroberta", "seance"]:
        out[f"delta_{mk}"] = point[mk]
        out[f"{mk}_lo"], out[f"{mk}_hi"] = ci(v[mk])
    for tag, a, b in [("MR", "mentalroberta", "roberta"), ("RS", "roberta", "seance"), ("MS", "mentalroberta", "seance")]:
        lo, hi = ci(v[a] - v[b])
        out[f"pt_{a[0].upper() if a != 'roberta' else 'R'}_minus_{b[0].upper() if b != 'roberta' else 'R'}"] = point[a] - point[b]
        out[f"{tag}_lo"], out[f"{tag}_hi"] = lo, hi
    out["MR_excludes0"] = bool(out["MR_lo"] > 0 or out["MR_hi"] < 0)
    return out


if __name__ == "__main__":
    jobs = [(v, o, 1000 + i) for v in ["sigPCs", "allPCs"] for i, o in enumerate(OUTCOMES)]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(job, jobs, chunksize=1)
    out_file = config.out_path("variance_dominance_outputs/bootstrap_R_vs_M.csv")
    pd.DataFrame(res).to_csv(out_file, index=False)
    print("saved", out_file)
