#!/usr/bin/env python3
"""
TEXT LEMURS - Figure 2: t-SNE Embedding Space Colored by Zero-Shot Domain
==========================================================================
Inputs:
    embedding_outputs/text_lemurs_embeddings_merged.csv  - merged data with RoBERTa PCs
    embedding_outputs/domain_classifications.csv         - zero-shot domain scores

Outputs:
    figures/figure2_tsne_domain.png
    figures/tsne_data.csv  (saved for Figures 3 and 4)

Caption:
    Figure 2. Two-dimensional t-SNE projection of RoBERTa-base embeddings for 3,073
    concern-present person-waves, colored by zero-shot domain classification (top domain
    per wave). Each dot represents one student-week. The global structure reveals two
    primary clusters: an academic/workload cluster (left of the embedding space, center
    approximately -19, -3) and an emotional/relational cluster (right of the embedding
    space, center approximately +8, +2), with general stress and financial concerns
    distributed across the space. Domain labels are assigned by the zero-shot classifier;
    color intensity reflects classifier confidence.

Run from the project root directory.
Requirements: pip install pandas numpy scikit-learn matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['pdf.fonttype'] = 42
import config
os.makedirs(config.out_path('figures'), exist_ok=True)

# -- CONFIG --------------------------------------------------------------------
EMBEDDING_DIR  = config.out_path('embedding_outputs')
DOMAIN_FILE    = config.out_path('embedding_outputs/domain_classifications.csv')
OUTPUT_PATH    = config.out_path('figures/figure2_tsne_domain.png')
TSNE_DATA_OUT  = config.out_path('figures/tsne_data.csv')
RANDOM_SEED    = 42
PERPLEXITY     = 40
N_ITER         = 1000

DOMAIN_CONFIG = {
    'academic workload and exams':   ('#1a5fa8', 'Academic workload'),
    'money and finances':            ('#d4a017', 'Money & finances'),
    'physical health and illness':   ('#c0392b', 'Physical health'),
    'mental health and anxiety':     ('#8e44ad', 'Mental health'),
    'relationships and social life': ('#27ae60', 'Relationships'),
    'housing and living situation':  ('#e67e22', 'Housing'),
    'future plans and career':       ('#16a085', 'Future plans'),
    'general stress':                ('#7f8c8d', 'General stress'),
    'no concern':                    ('#bdc3c7', 'No concern'),
}

# -- LOAD DATA -----------------------------------------------------------------
print('Loading data...')
merged = pd.read_csv(os.path.join(EMBEDDING_DIR, 'text_lemurs_embeddings_merged.csv'))
domain = pd.read_csv(DOMAIN_FILE)
print(f'  Domain columns: {domain.columns.tolist()}')

# Filter to concern-present first
concern = merged[merged['concern_present'] == 1].copy().reset_index(drop=True)

# Drop any existing domain columns to avoid merge conflicts
drop_cols = [c for c in concern.columns
             if c.startswith('domain_') or c in ('top_domain', 'top_score')]
concern = concern.drop(columns=drop_cols, errors='ignore')

# Merge domain file
df = concern.merge(domain, on=['record_id', 'Week'], how='inner').reset_index(drop=True)
print(f'  Rows after merge: {len(df):,}')
print(f'  Domain cols: {[c for c in df.columns if "domain" in c.lower() or "top" in c.lower()]}')

# -- EXTRACT ROBERTA PCs -------------------------------------------------------
pc_cols = [c for c in df.columns if c.startswith('roberta_PC')]
print(f'  RoBERTa PCs: {len(pc_cols)}')
X = StandardScaler().fit_transform(df[pc_cols].values)

# -- t-SNE --------------------------------------------------------------------
print(f'Running t-SNE (perplexity={PERPLEXITY}, max_iter={N_ITER})...')
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, max_iter=N_ITER,
            random_state=RANDOM_SEED, n_jobs=-1)
coords = tsne.fit_transform(X)
df['tsne_x'] = coords[:, 0]
df['tsne_y'] = coords[:, 1]
print('  t-SNE complete.')
print(f'  x range: {df["tsne_x"].min():.1f} to {df["tsne_x"].max():.1f}')
print(f'  y range: {df["tsne_y"].min():.1f} to {df["tsne_y"].max():.1f}')

# Print cluster centres to verify for Figures 3 and 4
print('\n  Domain centres (for box calibration):')
for dom in df['top_domain'].unique():
    sub = df[df['top_domain'] == dom]
    print(f'    {dom}: ({sub["tsne_x"].mean():.1f}, {sub["tsne_y"].mean():.1f})  n={len(sub)}')

# Save coordinates for reuse
save_cols = ['record_id', 'Week', 'tsne_x', 'tsne_y', 'concern_text_cleaned']
for col in ['top_domain', 'top_score']:
    if col in df.columns:
        save_cols.append(col)
    else:
        print(f'  Warning: {col} not in merged df - skipping')
df[save_cols].to_csv(TSNE_DATA_OUT, index=False)
print(f'  Coordinates saved: {TSNE_DATA_OUT}')

# -- PLOT ---------------------------------------------------------------------
print('Plotting...')
fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')

for domain_key, (color, label) in DOMAIN_CONFIG.items():
    mask = df['top_domain'] == domain_key
    if mask.sum() == 0:
        continue
    sub = df[mask]
    alpha = np.clip(sub['top_score'].values * 0.85 + 0.10, 0.15, 0.95).mean()
    ax.scatter(
        sub['tsne_x'], sub['tsne_y'],
        c=color, s=12, alpha=0.7,
        label=f'{label} (n={mask.sum():,})',
        linewidths=0, zorder=3, rasterized=True
    )

ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_linewidth(0.5); sp.set_color('#D3D1C7')

ax.legend(
    loc='upper right', framealpha=0.92, fontsize=9,
    markerscale=1.8, handletextpad=0.5,
    title='Zero-shot domain', title_fontsize=9.5,
    edgecolor='#D3D1C7'
)
ax.text(0.01, 0.01,
        f'N = {len(df):,} student-weeks  ·  RoBERTa-base embeddings  ·  '
        f't-SNE perplexity={PERPLEXITY}',
        transform=ax.transAxes, fontsize=7.5, color='#888780',
        ha='left', va='bottom', style='italic')

plt.tight_layout(pad=1.0)
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight',
            facecolor='white', metadata={'Title': 'TEXT LEMURS Figure 2'})
plt.close()
print(f'Saved: {OUTPUT_PATH}')
