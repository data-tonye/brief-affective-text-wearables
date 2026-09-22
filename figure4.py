#!/usr/bin/env python3
"""
TEXT LEMURS - Figure 4: Single-Outcome Zoomed Panel (Steps/Day)
================================================================
Requires Figure 2 to have been run first (uses saved t-SNE coordinates).

Label placement: each annotation's full text box (not just its anchor
point) is checked against every dot before being placed, so labels land in
open white space next to the cluster rather than over the scatter.

Cluster locations (confirmed from actual t-SNE output):
  Academic cluster  : center (-18.7, -3.2) - LEFT side of embedding space
  Emotional cluster : center (~+8,   +2.0) - RIGHT side of embedding space

Inputs:
    outputs/figures/tsne_data.csv          (written by figure2.py)
    outputs/embedding_outputs/text_lemurs_embeddings_merged.csv

Outputs:
    outputs/figures/figure4.png

Caption:
    Figure 4. Within-person daily step count across two semantic neighborhoods
    of the RoBERTa-base embedding space. Each dot represents one student-week,
    colored by within-person z-scored steps/day (green = above student's own
    weekly average; red = below). Left panel: academic/workload cluster (left
    of embedding space), where weeks dominated by academic concern language
    (school, finals, exams) are predominantly red - students walk less in weeks
    they express academic concerns. Right panel: emotional/relational cluster
    (right of embedding space), where the pattern is more heterogeneous,
    reflecting the mixed relationship between emotional concern framing and
    physical activity. Extreme dots are drawn on top so annotation arrows
    always point to visible, correctly colored dots. Annotations sampled 4:2
    (low-scoring : high-scoring) per panel. Annotated concern texts are selected 
    from the extremes of the within-person step distribution within each cluster. 
    Embedding placement reflects high-dimensional structural similarity and does 
    not always align with zero-shot domain classification - for example, 'Went on vacation' 
    (z = -2.33, academic cluster) was classified by the zero-shot model as 
    relationships/social life with low confidence (p = .31), illustrating 
    the inherent ambiguity of brief naturalistic text.

    

Run from the project root directory.
Requirements: pip install pandas numpy matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['pdf.fonttype'] = 42

import config
os.makedirs(config.out_path('figures'), exist_ok=True)

# -- CONFIG --
TSNE_FILE     = config.out_path('figures/tsne_data.csv')
EMBEDDING_DIR = config.out_path('embedding_outputs')
OUTPUT_PATH   = config.out_path('figures/figure4.png')

# Confirmed from actual t-SNE output:
#   Academic:  LEFT  - center (-18.7, -3.2)
#   Emotional: RIGHT - center (~+8,   +2.0)
ACAD_BOX = dict(xmin=-45, xmax=-4,  ymin=-28, ymax=18)
EMOT_BOX = dict(xmin=-4,  xmax=35,  ymin=-18, ymax=22)

SLOTS = [
    ( 90,  'center', 'bottom'),
    ( 45,  'left',   'bottom'),
    (  0,  'left',   'center'),
    (-45,  'left',   'top'),
    (-90,  'center', 'top'),
    (-135, 'right',  'top'),
    ( 180, 'right',  'center'),
    ( 135, 'right',  'bottom'),
]

# -- LOAD DATA --
print('Loading data...')
tsne   = pd.read_csv(TSNE_FILE)
merged = pd.read_csv(os.path.join(EMBEDDING_DIR, 'text_lemurs_embeddings_merged.csv'))

concern = merged[merged['concern_present'] == 1].copy().reset_index(drop=True)
concern['tsne_x'] = tsne['tsne_x'].values
concern['tsne_y'] = tsne['tsne_y'].values
concern['concern_text_cleaned'] = tsne['concern_text_cleaned'].values

# Within-person z-score steps
pm = concern.groupby('record_id')['act_steps'].transform('mean')
ps = concern.groupby('record_id')['act_steps'].transform('std')
concern['steps_z'] = (concern['act_steps'] - pm) / ps.replace(0, np.nan)

# -- HELPERS --
def get_cluster(df, box):
    return df[
        df['tsne_x'].between(box['xmin'], box['xmax']) &
        df['tsne_y'].between(box['ymin'], box['ymax'])
    ].copy()


def get_annotations(cluster, n_low=4, n_high=2, max_chars=35):
    sub = cluster.dropna(subset=['steps_z']).copy()
    sub = sub[sub['concern_text_cleaned'].notna()]
    sub = sub[sub['concern_text_cleaned'].str.len().between(3, max_chars)]
    sub['_txt'] = sub['concern_text_cleaned'].str.strip().str.lower()
    sub = sub.drop_duplicates(subset=['_txt'])
    bad  = sub.nsmallest(n_low  * 4, 'steps_z').head(n_low)
    good = sub.nlargest( n_high * 4, 'steps_z').head(n_high)
    return pd.concat([bad, good])


def text_bbox(lx, ly, ha, va, tw, th):
    """Bounding box of a 2-line annotation anchored at (lx, ly)."""
    x0 = lx if ha == 'left' else (lx - tw if ha == 'right' else lx - tw / 2)
    y0 = ly if va == 'bottom' else (ly - th if va == 'top' else ly - th / 2)
    return x0, x0 + tw, y0, y0 + th


def bbox_is_clear(bbox, dots_xy, pad):
    """True if no dot center (padded by the marker radius) falls inside the
    given text bounding box - i.e. the whole label, not just its anchor
    point, would sit in open space."""
    x0, x1, y0, y1 = bbox
    inside = ((dots_xy[:, 0] >= x0 - pad) & (dots_xy[:, 0] <= x1 + pad) &
              (dots_xy[:, 1] >= y0 - pad) & (dots_xy[:, 1] <= y1 + pad))
    return not inside.any()


def find_clear_anchor(cx, cy, ang_rad, dots_xy, pad, ha, va, tw, th,
                       r_start, r_step, r_max):
    """Walk outward along a fixed angle from the cluster centroid until the
    candidate label's full text box (not just its anchor point) is clear
    of every dot, i.e. sits in open white space rather than over the
    scatter."""
    r = r_start
    lx, ly = cx + np.cos(ang_rad) * r, cy + np.sin(ang_rad) * r
    bbox = text_bbox(lx, ly, ha, va, tw, th)
    while r <= r_max:
        lx, ly = cx + np.cos(ang_rad) * r, cy + np.sin(ang_rad) * r
        bbox = text_bbox(lx, ly, ha, va, tw, th)
        if bbox_is_clear(bbox, dots_xy, pad):
            return lx, ly, bbox
        r += r_step
    return lx, ly, bbox  # fall back to the furthest point tried


def assign_slots(annots_df, cx, cy, slots=SLOTS):
    slot_angles = np.array([s[0] for s in slots])
    used   = [False] * len(slots)
    result = []
    annots_df = annots_df.copy()
    annots_df['_d'] = np.hypot(annots_df['tsne_x'] - cx,
                                annots_df['tsne_y'] - cy)
    annots_df = annots_df.sort_values('_d', ascending=False)
    for _, row in annots_df.iterrows():
        ang = np.degrees(np.arctan2(row['tsne_y'] - cy, row['tsne_x'] - cx))
        diffs = [(i, abs(((ang - sa + 180) % 360) - 180))
                 for i, sa in enumerate(slot_angles) if not used[i]]
        if not diffs:
            break
        bi = min(diffs, key=lambda x: x[1])[0]
        used[bi] = True
        _, ha, va = slots[bi]
        result.append((row, np.radians(slot_angles[bi]), ha, va))
    return result


def draw_panel(ax, cluster, title, show_colorbar=True):
    sub = cluster.dropna(subset=['steps_z']).copy()

    # Sort by abs z ascending so extreme dots render on top
    sub['_abs_z'] = sub['steps_z'].abs()
    sub = sub.sort_values('_abs_z', ascending=True)

    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    sc   = ax.scatter(
        sub['tsne_x'], sub['tsne_y'],
        c=sub['steps_z'], cmap='RdYlGn', norm=norm,
        s=45, alpha=0.88, linewidths=0.35,
        edgecolors='white', zorder=3, rasterized=True
    )

    annots = get_annotations(cluster)
    cx = sub['tsne_x'].mean()
    cy = sub['tsne_y'].mean()
    xspan = sub['tsne_x'].max() - sub['tsne_x'].min()
    yspan = sub['tsne_y'].max() - sub['tsne_y'].min()
    r = min(xspan, yspan) * 0.55

    # Gap-seeking: push each label outward along its assigned direction
    # until its full text box (both lines, not just the anchor point) is
    # clear of every dot, so it reads as sitting in open white space next
    # to the cluster rather than over the scatter.
    dots_xy = sub[['tsne_x', 'tsne_y']].values
    nn_tree = cKDTree(dots_xy)
    nn_dist,_ = nn_tree.query(dots_xy, k=2)
    med_nn  = np.median(nn_dist[:, 1])
    pad     = med_nn * 0.65       # ~ dot radius, so a dot can't sit flush against the text
    r_step  = med_nn * 0.8
    r_max   = min(xspan, yspan) * 1.35
    char_w  = xspan * 0.0135
    line_h  = yspan * 0.045

    # On the colorbar panel, tilt the due-east slot up toward the top-right
    # corner: a label sent straight right competes with the colorbar (which
    # sits at mid-height, flush against the axis), forcing a lot of
    # reserved padding that visually squeezes the scatter. The due-east
    # (0°) and southeast (-45°) slots are the two nearest a point heading
    # roughly east, so both need to move - retilting only one still leaves
    # the other as the nearer (unwanted, southeast) option.
    if show_colorbar:
        panel_slots = [(40, 'left', 'bottom') if s[0] == 0 else
                        (-80, 'center', 'top') if s[0] == -45 else s
                        for s in SLOTS]
    else:
        panel_slots = SLOTS

    # First pass: find every label's clear anchor + bbox before drawing, so
    # axis limits can be sized from where the text actually lands (not a
    # guessed padding factor) - otherwise a label can be pushed past the
    # axis edge and render on top of the colorbar.
    placed, xs_ext, ys_ext = [], [], []
    for row, slot_rad, ha, va in assign_slots(annots, cx, cy, panel_slots):
        tw = max(len(str(row['concern_text_cleaned'])) + 2, 10) * char_w
        th = 2 * line_h
        lx, ly, bbox = find_clear_anchor(cx, cy, slot_rad, dots_xy, pad,
                                          ha, va, tw, th, r, r_step, r_max)
        placed.append((row, lx, ly, ha, va))
        xs_ext += [bbox[0], bbox[1]]
        ys_ext += [bbox[2], bbox[3]]

    for row, lx, ly, ha, va in placed:
        z     = row['steps_z']
        px,py = row['tsne_x'], row['tsne_y']
        text  = str(row['concern_text_cleaned']).strip()
        color = '#712B13' if z < 0 else '#27500A'
        ax.annotate(
            f'"{text}"\nz={z:+.2f}',
            xy=(px, py), xytext=(lx, ly),
            fontsize=8.5, color=color, fontweight='500',
            ha=ha, va=va,
            arrowprops=dict(
                arrowstyle='->', color='#AAAAAA',
                lw=0.8, mutation_scale=7,
                connectionstyle='arc3,rad=0.1'
            ),
            path_effects=[pe.withStroke(linewidth=1.6, foreground='white')],
            annotation_clip=False, zorder=6
        )

    dot_margin = 1.0
    xmin = min([sub['tsne_x'].min()] + xs_ext) - dot_margin
    xmax = max([sub['tsne_x'].max()] + xs_ext) + dot_margin
    ymin = min([sub['tsne_y'].min()] + ys_ext) - dot_margin
    ymax = max([sub['tsne_y'].max()] + ys_ext) + dot_margin
    if show_colorbar:
        # The colorbar sits flush against the right spine (pad=0.02); with
        # the due-east slot removed no label reaches that far right anymore,
        # so only a small reserve is needed (vs. the large one this used to
        # require) - this keeps the scatter from looking artificially
        # compressed.
        xmax += xspan * 0.08
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_yticks([-2, 0, 2])
        cbar.set_label('Steps/day\n(within-person z)', fontsize=8.5, labelpad=4)
        cbar.ax.text(0.5, -0.06, 'fewer steps', fontsize=7, color='#712B13',
                     transform=cbar.ax.transAxes, ha='center', va='top')
        cbar.ax.text(0.5,  1.06, 'more steps',  fontsize=7, color='#27500A',
                     transform=cbar.ax.transAxes, ha='center', va='bottom')

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=8, color='#2C2C2A', fontweight='500')
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color('#D3D1C7')
    ax.text(0.5, -0.025, f'n = {len(sub):,} student-weeks',
            transform=ax.transAxes, fontsize=8, color='#888780',
            ha='center', va='top', style='italic')


# -- BUILD CLUSTERS --
acad = get_cluster(concern, ACAD_BOX)
emot = get_cluster(concern, EMOT_BOX)
print(f'Academic cluster : {len(acad):,}  '
      f'(center {acad["tsne_x"].mean():.1f}, {acad["tsne_y"].mean():.1f})')
print(f'Emotional cluster: {len(emot):,}  '
      f'(center {emot["tsne_x"].mean():.1f}, {emot["tsne_y"].mean():.1f})')

# Sanity check
assert acad['tsne_x'].mean() < emot['tsne_x'].mean(), (
    "Cluster assignment wrong - academic centre should have lower x. "
    "Check ACAD_BOX / EMOT_BOX."
)
print('  Cluster sanity check passed.')

# -- PLOT --
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='white',
                                gridspec_kw={'wspace': 0.35})

# Academic on LEFT, emotional on RIGHT
draw_panel(ax1, acad, 'Academic / workload cluster', show_colorbar=False)
draw_panel(ax2, emot, 'Emotional / relational cluster', show_colorbar=True)

plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight', facecolor='white',
            metadata={'Title': 'TEXT LEMURS Figure 4'})
plt.close()
print(f'Saved: {OUTPUT_PATH}')
