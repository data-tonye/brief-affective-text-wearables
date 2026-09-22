"""
TEXT LEMURS - Figure 3 / Supplementary Figure 5: Zoomed Embedding Neighborhoods
================================================================================
Requires Figure 2 to have been run first (uses saved t-SNE coordinates).

Produces both the main-text figure and its all-outcomes supplementary
version from one script, since they share all plotting logic:
  outputs/figures/figure3.png       - Figure 3 (2x3: Steps/day, Sleep
                                       efficiency, RMSSD only)
  outputs/figures/figure5_full.png  - Supplementary Figure 5 (2x5: all outcomes)

Label placement: each annotation's full text box (not just its anchor
point) is checked against every dot, every other label, and the colorbar's
reserved zone before being placed, so labels land in open white space next
to the cluster rather than over the scatter or each other. Long annotation
text wraps onto a second line rather than being truncated.

Cluster locations (confirmed from actual t-SNE output):
  Academic cluster  : center (-18.7, -3.2) - LEFT side of embedding space
  Emotional cluster : center (~+8,   +2.0) - RIGHT side of embedding space

Inputs:
    outputs/figures/tsne_data.csv          (written by figure2.py)
    outputs/embedding_outputs/text_lemurs_embeddings_merged.csv

Run from the project root directory.
Requirements: pip install pandas numpy matplotlib scipy
"""

import os
import textwrap
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

# -- CONFIG ----------------------------------------------------------------
TSNE_FILE      = config.out_path('figures/tsne_data.csv')
EMBEDDING_DIR  = config.out_path('embedding_outputs')
OUTPUT_FULL    = config.out_path('figures/figure5_full.png')   # 2x5
OUTPUT_3COL    = config.out_path('figures/figure3.png')        # 2x3

ACAD_BOX = dict(xmin=-45, xmax=-4,  ymin=-28, ymax=18)
EMOT_BOX = dict(xmin=-4,  xmax=35,  ymin=-18, ymax=22)

OUTCOMES_ALL = [
    ('act_steps',               'Steps/day',           'steps z',    'fewer steps',  'more steps',   False),
    ('sleep_efficiency',        'Sleep efficiency',    'effic. z',   'worse sleep',  'better sleep', False),
    ('sleep_rmssd',             'RMSSD (HRV)',         'RMSSD z',    'lower HRV',    'higher HRV',   False),
    ('sleep_onset_latency_min', 'Sleep onset latency', 'onset z',    'faster onset', 'slower onset', True),
    ('sleep_total_hrs',         'Sleep duration',      'duration z', 'less sleep',   'more sleep',   False),
]

OUTCOMES_3COL = [
    ('act_steps',        'Steps/day',        'steps z',  'fewer steps', 'more steps',   False),
    ('sleep_efficiency', 'Sleep efficiency', 'effic. z', 'worse sleep', 'better sleep', False),
    ('sleep_rmssd',      'RMSSD (HRV)',      'RMSSD z',  'lower HRV',   'higher HRV',   False),
]

# Base slot ring; the due-east (0) and southeast (-45) slots are retilted
# toward the top-right corner per panel below, since every panel here has a
# colorbar flush against its right edge.
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
PANEL_SLOTS = [(25, 'left', 'bottom') if s[0] == 0 else
               (-80, 'center', 'top') if s[0] == -45 else s
               for s in SLOTS]

# -- LOAD DATA ---------------------------------------------------------------
print('Loading data...')
tsne   = pd.read_csv(TSNE_FILE)
merged = pd.read_csv(os.path.join(EMBEDDING_DIR, 'text_lemurs_embeddings_merged.csv'))

concern = merged[merged['concern_present'] == 1].copy().reset_index(drop=True)
concern['tsne_x'] = tsne['tsne_x'].values
concern['tsne_y'] = tsne['tsne_y'].values
concern['concern_text_cleaned'] = tsne['concern_text_cleaned'].values

for col, *_ in OUTCOMES_ALL:
    if col in concern.columns:
        pm = concern.groupby('record_id')[col].transform('mean')
        ps = concern.groupby('record_id')[col].transform('std')
        concern[col + '_z'] = (concern[col] - pm) / ps.replace(0, np.nan)

print('  Within-person z-scores computed.')

# -- HELPERS ------------------------------------------------------------------
def get_cluster(df, box):
    return df[
        df['tsne_x'].between(box['xmin'], box['xmax']) &
        df['tsne_y'].between(box['ymin'], box['ymax'])
    ].copy()


def wrap_label(s, width=20):
    """Wrap displayed annotation text onto a second line at a word boundary
    instead of truncating it - no content is lost. Keeps labels compact
    regardless of font size (a long string on one line at this font, in a
    narrow 3-column panel, is wider than the panel itself)."""
    return '\n'.join(textwrap.wrap(str(s).strip(), width=width, break_long_words=False))


def get_annotations(cluster, zcol, n_low=4, n_high=2, max_chars=35, invert=False):
    sub = cluster.dropna(subset=[zcol]).copy()
    sub = sub[sub['concern_text_cleaned'].notna()]
    sub = sub[sub['concern_text_cleaned'].str.len().between(3, max_chars)]
    sub['_txt'] = sub['concern_text_cleaned'].str.strip().str.lower()
    sub = sub.drop_duplicates(subset=['_txt'])
    if not invert:
        bad  = sub.nsmallest(n_low  * 4, zcol).head(n_low)
        good = sub.nlargest( n_high * 4, zcol).head(n_high)
    else:
        bad  = sub.nlargest( n_low  * 4, zcol).head(n_low)
        good = sub.nsmallest(n_high * 4, zcol).head(n_high)
    return bad, good


def assign_slots(annots_df, cx, cy, slots):
    slot_angles = np.array([s[0] for s in slots])
    used   = [False] * len(slots)
    result = []
    annots_df = annots_df.copy()
    annots_df['_dist'] = np.hypot(annots_df['tsne_x'] - cx, annots_df['tsne_y'] - cy)
    annots_df = annots_df.sort_values('_dist', ascending=False)
    for _, row in annots_df.iterrows():
        pt_angle = np.degrees(np.arctan2(row['tsne_y'] - cy, row['tsne_x'] - cx))
        diffs = [(i, abs(((pt_angle - sa + 180) % 360) - 180))
                 for i, sa in enumerate(slot_angles) if not used[i]]
        if not diffs:
            break
        best_i = min(diffs, key=lambda x: x[1])[0]
        used[best_i] = True
        _, ha, va = slots[best_i]
        result.append((row, np.radians(slot_angles[best_i]), ha, va))
    return result


def text_bbox(lx, ly, ha, va, tw, th):
    x0 = lx if ha == 'left' else (lx - tw if ha == 'right' else lx - tw / 2)
    y0 = ly if va == 'bottom' else (ly - th if va == 'top' else ly - th / 2)
    return x0, x0 + tw, y0, y0 + th


def rects_overlap(a, b):
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def bbox_is_clear(bbox, dots_xy, pad, exclude_rects=()):
    x0, x1, y0, y1 = bbox
    inside = ((dots_xy[:, 0] >= x0 - pad) & (dots_xy[:, 0] <= x1 + pad) &
              (dots_xy[:, 1] >= y0 - pad) & (dots_xy[:, 1] <= y1 + pad))
    if inside.any():
        return False
    return not any(rects_overlap(bbox, r) for r in exclude_rects)


def find_clear_anchor(cx, cy, ang_rad, dots_xy, pad, ha, va, tw, th,
                       r_start, r_step, r_max, exclude_rects=()):
    """Walk outward until the label's text box clears every dot AND every
    rectangle in `exclude_rects` - the colorbar's reserved zone and every
    label already placed in this panel, so labels don't overlap the scatter,
    the colorbar, or each other."""
    r = r_start
    lx, ly = cx + np.cos(ang_rad) * r, cy + np.sin(ang_rad) * r
    bbox = text_bbox(lx, ly, ha, va, tw, th)
    while r <= r_max:
        lx, ly = cx + np.cos(ang_rad) * r, cy + np.sin(ang_rad) * r
        bbox = text_bbox(lx, ly, ha, va, tw, th)
        if bbox_is_clear(bbox, dots_xy, pad, exclude_rects):
            return lx, ly, bbox
        r += r_step
    return lx, ly, bbox  # fall back to the furthest point tried


def draw_panel(ax, cluster, zcol, title, cbar_label, low_txt, high_txt,
               invert=False, annot_fs=8.0, title_fs=12, cbar_fs=10,
               tick_fs=9, n_label_fs=9):
    sub = cluster.dropna(subset=[zcol]).copy()
    sub['_abs_z'] = sub[zcol].abs()
    sub = sub.sort_values('_abs_z', ascending=True)

    cmap = 'RdYlGn_r' if invert else 'RdYlGn'
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    sc = ax.scatter(
        sub['tsne_x'], sub['tsne_y'],
        c=sub[zcol], cmap=cmap, norm=norm,
        s=40, alpha=0.85, linewidths=0.3,
        edgecolors='white', zorder=3, rasterized=True
    )

    bad_df, good_df = get_annotations(cluster, zcol, invert=invert)
    all_annots = pd.concat([bad_df, good_df])
    cx = sub['tsne_x'].mean()
    cy = sub['tsne_y'].mean()
    xspan = sub['tsne_x'].max() - sub['tsne_x'].min()
    yspan = sub['tsne_y'].max() - sub['tsne_y'].min()
    r = min(xspan, yspan) * 0.40

    dots_xy = sub[['tsne_x', 'tsne_y']].values
    nn_dist, _ = cKDTree(dots_xy).query(dots_xy, k=2)
    med_nn = np.median(nn_dist[:, 1])
    pad    = med_nn * 0.65
    r_step = med_nn * 0.8
    r_max  = min(xspan, yspan) * 1.6
    char_w = xspan * 0.0135 * (annot_fs / 8.0)
    line_h = yspan * 0.045 * (annot_fs / 8.0)

    # Every panel here carries its own colorbar, with "more/fewer" labels
    # above and below it - reserve that whole zone up front so no annotation
    # is even considered there, on top of avoiding the scatter itself.
    cbar_zone = (
        sub['tsne_x'].max() - xspan * 0.02, sub['tsne_x'].max() + xspan * 0.42,
        cy - yspan * 0.47, cy + yspan * 0.47,
    )
    exclude_rects = [cbar_zone]

    # Placed sequentially (nearest-first is already how assign_slots orders
    # ties) so each new label also avoids every label already anchored in
    # this panel, not just the dots and the colorbar.
    placed, xs_ext, ys_ext = [], [], []
    for row, slot_rad, ha, va in assign_slots(all_annots, cx, cy, PANEL_SLOTS):
        wrapped = wrap_label(row['concern_text_cleaned'])
        n_lines = wrapped.count('\n') + 2  # wrapped text lines + the z=.. line
        tw = max(max(len(ln) for ln in wrapped.split('\n')) + 3, 12) * char_w
        th = n_lines * line_h
        lx, ly, bbox = find_clear_anchor(cx, cy, slot_rad, dots_xy, pad,
                                          ha, va, tw, th, r, r_step, r_max,
                                          exclude_rects)
        placed.append((row, lx, ly, ha, va))
        exclude_rects.append(bbox)
        xs_ext += [bbox[0], bbox[1]]
        ys_ext += [bbox[2], bbox[3]]

    dot_margin = 1.0
    xmin = min([sub['tsne_x'].min()] + xs_ext) - dot_margin
    xmax = max([sub['tsne_x'].max()] + xs_ext) + dot_margin
    ymin = min([sub['tsne_y'].min()] + ys_ext) - dot_margin
    ymax = max([sub['tsne_y'].max()] + ys_ext) + dot_margin
    xmax += xspan * 0.08
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    anns = []
    for row, lx, ly, ha, va in placed:
        z      = row[zcol]
        px, py = row['tsne_x'], row['tsne_y']
        text   = wrap_label(row['concern_text_cleaned'])
        is_bad    = (z < 0) if not invert else (z > 0)
        dot_color = '#712B13' if is_bad else '#27500A'
        ann = ax.annotate(
            f'"{text}"\nz={z:+.2f}',
            xy=(px, py), xytext=(lx, ly),
            fontsize=annot_fs, color=dot_color, fontweight='500',
            ha=ha, va=va,
            arrowprops=dict(
                arrowstyle='->', color='#AAAAAA',
                lw=0.7, mutation_scale=6,
                connectionstyle='arc3,rad=0.1'
            ),
            path_effects=[pe.withStroke(linewidth=1.6, foreground='white')],
            annotation_clip=False, zorder=6
        )
        anns.append(ann)


    cbar = plt.colorbar(sc, ax=ax, shrink=0.48, pad=0.01)
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.ax.set_yticks([-2, 0, 2])
    cbar.set_label(cbar_label, fontsize=cbar_fs, labelpad=3)
    low_c  = '#712B13' if not invert else '#27500A'
    high_c = '#27500A' if not invert else '#712B13'
    cbar.ax.text(0.5, -0.05, low_txt,  fontsize=cbar_fs - 1.5, color=low_c,
                 transform=cbar.ax.transAxes, ha='center', va='top')
    cbar.ax.text(0.5,  1.05, high_txt, fontsize=cbar_fs - 1.5, color=high_c,
                 transform=cbar.ax.transAxes, ha='center', va='bottom')

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=title_fs, pad=6, color='#2C2C2A')
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#D3D1C7')
    ax.text(0.5, -0.02, f'n = {len(sub):,} student-weeks',
            transform=ax.transAxes, fontsize=n_label_fs,
            color='#888780', ha='center', va='top', style='italic')


# -- BUILD CLUSTERS ------------------------------------------------------------
acad = get_cluster(concern, ACAD_BOX)
emot = get_cluster(concern, EMOT_BOX)
print(f'Academic cluster : {len(acad):,} rows  '
      f'(center {acad["tsne_x"].mean():.1f}, {acad["tsne_y"].mean():.1f})')
print(f'Emotional cluster: {len(emot):,} rows  '
      f'(center {emot["tsne_x"].mean():.1f}, {emot["tsne_y"].mean():.1f})')

assert acad['tsne_x'].mean() < emot['tsne_x'].mean(), (
    "Cluster assignment wrong - academic centre should have lower x than "
    "emotional centre. Check ACAD_BOX / EMOT_BOX."
)
print('  Cluster sanity check passed.')


# ============================================================================
# OUTPUT 1 - supplementary 2x5 (all outcomes); no top heading
# ============================================================================
print('\nGenerating Supplementary Figure 5 (2x5)...')
fig, axes = plt.subplots(2, 5, figsize=(26, 12), facecolor='white',
                         gridspec_kw={'hspace': 0.16, 'wspace': 0.36})
fig.subplots_adjust(left=0.09)

for ci, (col, title, cbar_lbl, low_lbl, high_lbl, inv) in enumerate(OUTCOMES_ALL):
    zcol = col + '_z'
    draw_panel(axes[0, ci], acad, zcol, title, cbar_lbl, low_lbl, high_lbl, inv,
               annot_fs=6.2, title_fs=9.5, cbar_fs=8, tick_fs=7.5, n_label_fs=7)
    draw_panel(axes[1, ci], emot, zcol, title, cbar_lbl, low_lbl, high_lbl, inv,
               annot_fs=6.2, title_fs=9.5, cbar_fs=8, tick_fs=7.5, n_label_fs=7)

for ri, lbl in enumerate([
    'Academic / workload cluster\n(left of embedding space)',
    'Emotional / relational cluster\n(right of embedding space)',
]):
    fig.text(0.025, 0.75 - ri * 0.50, lbl,
             fontsize=9, fontweight='500', color='#444441',
             ha='center', va='center', rotation=90, linespacing=1.5)

plt.savefig(OUTPUT_FULL, dpi=150, bbox_inches='tight', facecolor='white',
            metadata={'Title': 'TEXT LEMURS Supplementary Figure 5 (2x5)'})
plt.close()
print(f'  Saved: {OUTPUT_FULL}')


# ============================================================================
# OUTPUT 2 - 2x3 publication version; no top heading
# ============================================================================
print('\nGenerating Figure 3 (2x3)...')
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 13), facecolor='white',
                            gridspec_kw={'hspace': 0.16, 'wspace': 0.38})
fig3.subplots_adjust(left=0.11)

PUB = dict(annot_fs=8.8, title_fs=13, cbar_fs=10.5, tick_fs=9.5, n_label_fs=9.5)

for ci, (col, title, cbar_lbl, low_lbl, high_lbl, inv) in enumerate(OUTCOMES_3COL):
    zcol = col + '_z'
    draw_panel(axes3[0, ci], acad, zcol, title, cbar_lbl, low_lbl, high_lbl, inv, **PUB)
    draw_panel(axes3[1, ci], emot, zcol, title, cbar_lbl, low_lbl, high_lbl, inv, **PUB)

for ri, lbl in enumerate([
    'Academic / workload cluster\n(left of embedding space)',
    'Emotional / relational cluster\n(right of embedding space)',
]):
    fig3.text(0.028, 0.74 - ri * 0.50, lbl,
              fontsize=10.5, fontweight='500', color='#444441',
              ha='center', va='center', rotation=90, linespacing=1.5)

plt.savefig(OUTPUT_3COL, dpi=200, bbox_inches='tight', facecolor='white',
            metadata={'Title': 'TEXT LEMURS Figure 3 (2x3)'})
plt.close()
print(f'  Saved: {OUTPUT_3COL}')

print('\nDone. Both figures saved.')
print(f'  Paper figure : {OUTPUT_3COL}')
print(f'  Supplementary: {OUTPUT_FULL}')
