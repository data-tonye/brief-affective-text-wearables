#!/usr/bin/env python3
"""
TEXT LEMURS - Figure 6: Language-Block ΔR² by NLP Method (Appendix H)
=======================================================================
Graphical summary of Table 7 (main text). Requires
seance_variance_and_dominance.py and embedding_variance_and_dominance.py
to have been run first.

Inputs:
    outputs/variance_decomposition.csv                        (SEANCE column)
    outputs/variance_dominance_outputs/variance_decomposition.csv  (RoBERTa /
                                                                  MentalRoBERTa columns)

Output:
    outputs/figures/figure6.png / .pdf

Caption:
    Figure 6. Incremental variance explained (delta R^2) by each NLP method
    above semester timing across nine wearable outcomes. Bars show the
    language-block delta R^2 for SEANCE (blue), RoBERTa-base (orange), and
    MentalRoBERTa (aqua). Stars indicate the best-performing method per
    outcome. All models restricted to concern-present waves.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

OUT_DIR = config.out_path("figures")
os.makedirs(OUT_DIR, exist_ok=True)

seance = pd.read_csv(config.out_path("variance_decomposition.csv"))
embed = pd.read_csv(config.out_path("variance_dominance_outputs/variance_decomposition.csv"))

# outcome_var (decomposition.py) and outcome (embed_domanalysis.py) use the
# same snake_case outcome keys across both scripts (e.g. "sleep_total_hrs",
# "act_steps") - see each script's OUTCOMES dict.
colors = {"SEANCE": "#2a78d6", "RoBERTa-base": "#eb6834", "MentalRoBERTa": "#1baf7a"}
sleep = [("sleep_total_hrs", "Sleep\nduration"), ("sleep_efficiency", "Sleep\nefficiency"),
         ("sleep_rem_hrs", "REM\nsleep"), ("sleep_deep_hrs", "Deep\nsleep"),
         ("sleep_rmssd", "RMSSD"), ("sleep_onset_latency_min", "Onset\nlatency")]
act = [("act_steps", "Steps/day"), ("act_met_min_medium", "MET-min\nmedium"),
       ("act_met_min_high", "MET-min\nhigh")]


def val(outcome_var, method):
    if method == "SEANCE":
        r = seance[seance.outcome_var == outcome_var]
        return float(r.delta_language.iloc[0])
    model_key = "roberta" if method == "RoBERTa-base" else "mentalroberta"
    r = embed[(embed.outcome == outcome_var) & (embed.model == model_key)]
    return float(r.deltaR2_language.iloc[0])


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                      "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True,
                          gridspec_kw={"width_ratios": [6, 3], "wspace": 0.06})
bw = 0.26
for ax, groups, title in [(axes[0], sleep, "Sleep"), (axes[1], act, "Physical activity")]:
    for gi, (outcome_var, lab) in enumerate(groups):
        vs = {m: val(outcome_var, m) for m in colors}
        best = max(vs, key=vs.get)
        for k, m in enumerate(colors):
            x = gi + (k - 1) * (bw + 0.02)
            ax.bar(x, vs[m], bw, color=colors[m], edgecolor="#fcfcfb", linewidth=1.2,
                   label=m if gi == 0 else None, zorder=3)
            if m == best:
                ax.text(x, vs[m] + 0.0009, "★", ha="center", va="bottom",
                        fontsize=10, color="#1a1a19", zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups], fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.yaxis.grid(True, color="#e3e2dc", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9.5, length=0)
    ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#8a897f")
axes[0].set_ylim(0, 0.042)
axes[0].set_ylabel("Language-block Δ$R^2$", fontsize=10.5)
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
axes[1].tick_params(axis="y", left=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.8)
fig.subplots_adjust(top=0.84, bottom=0.17, left=0.11, right=0.985)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"figure6.{ext}"), dpi=300, facecolor="white")
print(f"Saved: {os.path.join(OUT_DIR, 'figure6.png')}")
