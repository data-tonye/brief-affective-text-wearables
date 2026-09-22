#!/usr/bin/env python3
"""
TEXT LEMURS - Embedding Extraction + Zero-Shot Domain Classification Pipeline

Runs three steps:
  1. RoBERTa-base embeddings       -> PCA components
  2. MentalRoBERTa embeddings      -> PCA components
  3. Zero-shot domain classifier   -> domain probability scores per text

All steps are checkpointed - if the script is interrupted, it resumes
from where it left off. Safe to run overnight with screen.

Requirements:
    pip install transformers torch pandas numpy scikit-learn

screen usage:
    screen -S textlemurs
    python embed.py
    Ctrl+A then D          (detach - leave running)
    screen -r textlemurs   (reattach next day)
    screen -ls             (list all sessions)

Outputs (all saved to ./embedding_outputs/):
    checkpoints/                        - intermediate progress files
    roberta_embeddings_raw.csv          - 768-dim raw embeddings
    mentalroberta_embeddings_raw.csv    - 768-dim raw embeddings
    roberta_pca_components.csv          - PCA-reduced components
    mentalroberta_pca_components.csv    - PCA-reduced components
    pca_variance_explained.csv          - variance explained per model
    domain_classifications.csv          - domain probability scores per text
    text_lemurs_embeddings_merged.csv   - everything merged, one row per person-wave
"""

import os
import sys
import time
import warnings
import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, pipeline
except ImportError:
    print("ERROR: transformers and torch are required.")
    print("Run: pip install transformers torch")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONFIGURATION - edit these if needed
# -----------------------------------------------------------------------------

import config
DATA_PATH      = config.out_path('oura_aggregated.csv')
OUTPUT_DIR     = config.out_path('embedding_outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BATCH_SIZE     = 32      # reduce to 8 if you run out of RAM
MAX_TOKEN_LEN  = 64      # generous for 3-word texts
PCA_VARIANCE   = 0.80    # keep components explaining 80% variance
RANDOM_SEED    = 42
LOG_EVERY      = 10      # print progress every N batches

MODELS = {
    "roberta":       "roberta-base",
    "mentalroberta": "mental/mental-roberta-base",
}

ZEROSHOT_MODEL = "facebook/bart-large-mnli"

# Domain labels informed by frequency analysis of actual concern texts.
# Edit freely - the classifier requires no retraining when you change these.
DOMAINS = [
    "academic workload and exams",
    "money and finances",
    "physical health and illness",
    "mental health and anxiety",
    "relationships and social life",
    "housing and living situation",
    "future plans and career",
    "general stress",
    "no concern",
]

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# CHECKPOINTING
# -----------------------------------------------------------------------------

def ckpt_path(name: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{name}.npy")

def save_ckpt(name: str, data: np.ndarray) -> None:
    np.save(ckpt_path(name), data)
    log(f"  Checkpoint saved: {name}")

def load_ckpt(name: str):
    path = ckpt_path(name)
    if os.path.exists(path):
        log(f"  Checkpoint found - resuming: {name}")
        return np.load(path)
    return None


# -----------------------------------------------------------------------------
# STEP 1 - Load data
# -----------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    log("=" * 55)
    log("STEP 1 - Loading data")
    log("=" * 55)

    df = pd.read_csv(DATA_PATH)
    log(f"Total rows         : {len(df):,}")

    required = {"record_id", "Week", "concern_text_cleaned", "concern_present"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep all rows where concern_present == 1.
    # "nothing", "nothin", "not much" are intentionally kept -
    # they represent valid zero-concern states -> "no concern" domain.
    df_text = df[df["concern_present"] == 1].copy()
    df_text = df_text[df_text["concern_text_cleaned"].notna()]
    df_text = df_text[df_text["concern_text_cleaned"].str.strip() != ""]
    df_text = df_text.reset_index(drop=True)

    wc = df_text["concern_text_cleaned"].str.split().str.len()
    log(f"Rows with text     : {len(df_text):,}")
    log(f"Median word count  : {wc.median():.1f}")
    log(f"Mean word count    : {wc.mean():.1f}")
    log(f"Max word count     : {wc.max()}")

    return df_text


# -----------------------------------------------------------------------------
# STEP 2 - Embedding extraction
# -----------------------------------------------------------------------------

def mean_pool(token_embeddings: torch.Tensor,
              attention_mask: torch.Tensor) -> np.ndarray:
    mask_expanded  = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask       = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask).detach().cpu().numpy()


def extract_embeddings(texts: list, model_key: str, model_name: str) -> np.ndarray:
    log("=" * 55)
    log(f"STEP 2 - Embeddings: {model_key.upper()}")
    log(f"  Model : {model_name}")
    log("=" * 55)

    cached = load_ckpt(f"{model_key}_embeddings")
    if cached is not None and cached.shape[0] == len(texts):
        log(f"  Full checkpoint loaded: shape {cached.shape}")
        return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device     : {device}")
    log(f"  Texts      : {len(texts):,}")
    log(f"  Batch size : {BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    n_batches      = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeddings = []
    t_start        = time.time()

    for i in range(n_batches):
        batch   = texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        emb = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(emb)

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == n_batches:
            elapsed   = time.time() - t_start
            done      = (i + 1) * BATCH_SIZE
            remaining = (elapsed / max(done, 1)) * max(0, len(texts) - done)
            log(f"  Batch {i+1:>4}/{n_batches} | "
                f"elapsed {elapsed/60:.1f}m | "
                f"est. remaining {remaining/60:.1f}m")
            save_ckpt(f"{model_key}_embeddings_partial", np.vstack(all_embeddings))

    result = np.vstack(all_embeddings)
    save_ckpt(f"{model_key}_embeddings", result)
    log(f"  Final matrix: {result.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# -----------------------------------------------------------------------------
# STEP 3 - PCA reduction
# -----------------------------------------------------------------------------

def run_pca(embeddings: np.ndarray, model_key: str) -> tuple:
    log("=" * 55)
    log(f"STEP 3 - PCA: {model_key.upper()}")
    log("=" * 55)

    scaler     = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    pca_full     = PCA(random_state=RANDOM_SEED)
    pca_full.fit(emb_scaled)
    cumvar       = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, PCA_VARIANCE) + 1)

    log(f"  Components for {PCA_VARIANCE*100:.0f}% variance : {n_components}")
    log(f"  Top-5 components explain             : "
        f"{pca_full.explained_variance_ratio_[:5].sum()*100:.1f}%")

    pca        = PCA(n_components=n_components, random_state=RANDOM_SEED)
    components = pca.fit_transform(emb_scaled)

    return components, pca


# -----------------------------------------------------------------------------
# STEP 4 - Save embedding outputs
# -----------------------------------------------------------------------------

def save_embedding_outputs(df_meta: pd.DataFrame,
                           raw_embeddings: np.ndarray,
                           pca_components: np.ndarray,
                           pca_obj: PCA,
                           model_key: str) -> pd.DataFrame:
    log("=" * 55)
    log(f"STEP 4 - Saving outputs: {model_key.upper()}")
    log("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta = df_meta[["record_id", "Week"]].reset_index(drop=True)

    # Raw embeddings
    raw_cols = [f"{model_key}_emb_{i+1}" for i in range(raw_embeddings.shape[1])]
    df_raw   = pd.concat([meta, pd.DataFrame(raw_embeddings, columns=raw_cols)], axis=1)
    raw_path = os.path.join(OUTPUT_DIR, f"{model_key}_embeddings_raw.csv")
    df_raw.to_csv(raw_path, index=False)
    log(f"  Raw embeddings : {raw_path}  {df_raw.shape}")

    # PCA components
    n       = pca_components.shape[1]
    pc_cols = [f"{model_key}_PC{i+1}" for i in range(n)]
    df_pca  = pd.concat([meta, pd.DataFrame(pca_components, columns=pc_cols)], axis=1)
    pca_path = os.path.join(OUTPUT_DIR, f"{model_key}_pca_components.csv")
    df_pca.to_csv(pca_path, index=False)
    log(f"  PCA components : {pca_path}  {df_pca.shape}")

    # Variance explained
    var_df = pd.DataFrame({
        "model":               model_key,
        "component":           [f"PC{i+1}" for i in range(n)],
        "variance_explained":  pca_obj.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca_obj.explained_variance_ratio_)
    })
    var_df.to_csv(os.path.join(OUTPUT_DIR, f"{model_key}_pca_variance.csv"), index=False)
    log(f"  Variance table saved")

    return df_pca


# -----------------------------------------------------------------------------
# STEP 5 - Zero-shot domain classification
# -----------------------------------------------------------------------------

def run_zero_shot(df_meta: pd.DataFrame, texts: list) -> pd.DataFrame:
    log("=" * 55)
    log("STEP 5 - Zero-shot domain classification")
    log(f"  Model  : {ZEROSHOT_MODEL}")
    log(f"  Domains ({len(DOMAINS)}):")
    for d in DOMAINS:
        log(f"    - {d}")
    log("=" * 55)

    domain_path = os.path.join(OUTPUT_DIR, "domain_classifications.csv")

    if os.path.exists(domain_path):
        df_done = pd.read_csv(domain_path)
        if len(df_done) == len(texts):
            log(f"  Full checkpoint found - skipping")
            return df_done
        log(f"  Partial checkpoint ({len(df_done)}/{len(texts)}) - resuming")

    device = 0 if torch.cuda.is_available() else -1
    log(f"  Device : {'cuda' if device == 0 else 'cpu'}")
    log(f"  Loading model (~1.6GB on first run)...")

    classifier = pipeline(
        "zero-shot-classification",
        model=ZEROSHOT_MODEL,
        device=device
    )

    results      = []
    partial_path = os.path.join(CHECKPOINT_DIR, "domain_partial.csv")
    start_idx    = 0

    if os.path.exists(partial_path):
        partial_df = pd.read_csv(partial_path)
        start_idx  = len(partial_df)
        results    = partial_df.to_dict("records")
        log(f"  Resuming from text {start_idx} / {len(texts)}")

    t_start = time.time()

    for i in range(start_idx, len(texts)):
        text = texts[i]
        out  = classifier(text, DOMAINS, multi_label=False)

        row = {
            "record_id":  df_meta["record_id"].iloc[i],
            "Week":       df_meta["Week"].iloc[i],
            "top_domain": out["labels"][0],
            "top_score":  round(out["scores"][0], 4),
        }
        for label, score in zip(out["labels"], out["scores"]):
            col      = "domain_" + label.lower() \
                                        .replace(" ", "_") \
                                        .replace("/", "_") \
                                        .replace("and_", "")
            row[col] = round(score, 4)

        results.append(row)

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(texts):
            elapsed   = time.time() - t_start
            done      = i - start_idx + 1
            remaining = (elapsed / max(done, 1)) * (len(texts) - i - 1)
            log(f"  Text {i+1:>4}/{len(texts)} | "
                f"elapsed {elapsed/60:.1f}m | "
                f"est. remaining {remaining/60:.1f}m")
            pd.DataFrame(results).to_csv(partial_path, index=False)

    df_domains = pd.DataFrame(results)
    df_domains.to_csv(domain_path, index=False)
    log(f"  Saved: {domain_path}  {df_domains.shape}")

    return df_domains


# -----------------------------------------------------------------------------
# STEP 6 - Merge everything
# -----------------------------------------------------------------------------

def build_merged_file(pca_dfs: dict, df_domains: pd.DataFrame) -> pd.DataFrame:
    log("=" * 55)
    log("STEP 6 - Building merged output file")
    log("=" * 55)

    df_full = pd.read_csv(DATA_PATH)

    for model_key, df_pca in pca_dfs.items():
        df_full = df_full.merge(df_pca, on=["record_id", "Week"], how="left")

    df_full = df_full.merge(df_domains, on=["record_id", "Week"], how="left")

    out_path = os.path.join(OUTPUT_DIR, "text_lemurs_embeddings_merged.csv")
    df_full.to_csv(out_path, index=False)
    log(f"  Merged file : {out_path}")
    log(f"  Shape       : {df_full.shape}")

    for model_key in pca_dfs:
        n_filled = df_full[f"{model_key}_PC1"].notna().sum()
        log(f"  {model_key} coverage : {n_filled:,} / {len(df_full):,} rows")

    if "top_domain" in df_full.columns:
        log(f"\n  Domain distribution:")
        for domain, count in df_full["top_domain"].value_counts().items():
            pct = count / df_full["top_domain"].notna().sum() * 100
            log(f"    {count:>4} ({pct:4.1f}%)  {domain}")

    return df_full


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    t_total = time.time()

    log("=" * 55)
    log("TEXT LEMURS - Embedding + Domain Classification Pipeline")
    log("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df    = load_data()
    texts = df["concern_text_cleaned"].tolist()

    pca_dfs           = {}
    all_variance_rows = []

    for model_key, model_name in MODELS.items():
        embeddings          = extract_embeddings(texts, model_key, model_name)
        components, pca_obj = run_pca(embeddings, model_key)
        df_pca              = save_embedding_outputs(df, embeddings, components,
                                                     pca_obj, model_key)
        pca_dfs[model_key]  = df_pca

        for i, (v, cv) in enumerate(zip(
            pca_obj.explained_variance_ratio_,
            np.cumsum(pca_obj.explained_variance_ratio_)
        )):
            all_variance_rows.append({
                "model":               model_key,
                "component":           f"PC{i+1}",
                "variance_explained":  round(v, 6),
                "cumulative_variance": round(cv, 6)
            })

    pd.DataFrame(all_variance_rows).to_csv(
        os.path.join(OUTPUT_DIR, "pca_variance_explained.csv"), index=False)
    log("Combined variance table saved")

    df_domains = run_zero_shot(df, texts)
    build_merged_file(pca_dfs, df_domains)

    elapsed = time.time() - t_total
    log("=" * 55)
    log(f"ALL DONE - total time: {elapsed/60:.1f} minutes")
    log(f"Outputs in: ./{OUTPUT_DIR}/")
    log("=" * 55)


if __name__ == "__main__":
    main()