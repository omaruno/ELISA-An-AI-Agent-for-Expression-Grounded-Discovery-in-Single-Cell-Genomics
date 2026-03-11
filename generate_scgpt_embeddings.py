#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scGPT CLS Embedding Generator — Parameterized
===============================================
Generates per-cell CLS embeddings from scGPT whole-human model,
then aggregates them into per-cluster embeddings for ELISA retrieval.

Usage:
    python generate_scgpt_embeddings.py \
        --h5ad /path/to/data.h5ad \
        --cluster-key cell_type \
        --name MyDataset \
        --out /path/to/output/

    python generate_scgpt_embeddings.py \
        --h5ad /path/to/data.h5ad \
        --cluster-key cell_type \
        --model-dir /path/to/scgpt_model/ \
        --gene-map /path/to/gene_info.csv \
        --name P2_BreastTissue \
        --out /path/to/output/ \
        --batch-size 64

Outputs:
    {out}/scgpt_cluster_embeddings_by_{cluster_key}_{name}.pt
        Contains: cluster_ids, cluster_scgpt_embeddings, cluster_cell_ids

    {out}/scgpt_cell_embeddings_{name}.npy  (optional, --save-cell-emb)
        Per-cell embeddings (N_cells x 512)

    {out}/metadata_cells_{name}.csv
        Cell metadata with cluster assignments
"""

import os
import sys
import json
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import torch
from collections import defaultdict

# Suppress TF/Flax warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import functools
print = functools.partial(print, flush=True)


# ============================================================
# DEFAULTS
# ============================================================
DEFAULT_MODEL_DIR = "/lustre/home/ocoser/aiagents/scgptHuman"
DEFAULT_GENE_MAP = None  # will look for {model_dir}/gene_info.csv
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_TOKENS = 3000
DEFAULT_N_BINS = 51


# ============================================================
# GENE MAP: Ensembl → Symbol
# ============================================================
def load_gene_map(path_csv):
    """Load ENSG → gene symbol mapping from CSV."""
    print(f"[INFO] Loading gene map from: {path_csv}")
    df = pd.read_csv(path_csv)
    df["ensembl"] = df["feature_id"].astype(str).str.split(".").str[0]
    df["symbol"] = df["feature_name"].astype(str)
    mapping = dict(zip(df["ensembl"], df["symbol"]))
    print(f"[INFO] Loaded {len(mapping)} gene mappings")
    return mapping


def apply_gene_map(adata, mapping):
    """Rename ENSG var_names to gene symbols."""
    cleaned = []
    for g in adata.var_names:
        base = g.split(".")[0]
        cleaned.append(mapping.get(base, base))
    adata.var_names = cleaned
    adata.var["gene_name"] = cleaned
    adata.var_names_make_unique()
    n_mapped = sum(1 for g in cleaned if not g.startswith("ENSG"))
    print(f"[INFO] Mapped {n_mapped}/{len(cleaned)} genes to symbols")
    return adata


def check_if_ensembl(adata):
    """Check if var_names are ENSEMBL IDs."""
    sample = list(adata.var_names[:10])
    n_ensg = sum(1 for g in sample if str(g).startswith("ENSG"))
    return n_ensg >= 5


# ============================================================
# LOAD scGPT MODEL
# ============================================================
def load_scgpt(model_dir):
    """Load scGPT whole-human model."""
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.model import TransformerModel

    vocab_path = os.path.join(model_dir, "vocab.json")
    args_path = os.path.join(model_dir, "args.json")
    ckpt_path = os.path.join(model_dir, "best_model.pt")

    for p in [vocab_path, args_path, ckpt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    vocab_dict = json.load(open(vocab_path))
    if "<pad>" not in vocab_dict:
        vocab_dict = {k: v + 1 for k, v in vocab_dict.items()}
        vocab_dict["<pad>"] = 0

    vocab = GeneVocab.from_dict(vocab_dict)
    vocab.pad_token = "<pad>"
    vocab.pad_id = vocab_dict["<pad>"]

    args = json.load(open(args_path))

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args["embsize"],
        nhead=args["nheads"],
        d_hid=args["d_hid"],
        nlayers=args["nlayers"],
        dropout=args["dropout"],
        pad_token=vocab.pad_token,
        pad_value=-2,
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        n_input_bins=args["n_bins"],
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
        vocab=vocab,
    )

    print("[INFO] Loading checkpoint...")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[INFO] scGPT model loaded on: {device}")
    print(f"[INFO] Vocab size: {len(vocab)}, d_model: {args['embsize']}, "
          f"n_bins: {args['n_bins']}")

    return model, vocab, device, args["n_bins"]


# ============================================================
# TOKENIZE CELL
# ============================================================
def tokenize_cell(expr_row, gene_names, stoi, n_bins, vocab,
                  max_tokens=DEFAULT_MAX_TOKENS):
    """Tokenize a single cell's expression profile for scGPT."""
    nz = np.nonzero(expr_row)[0]
    if len(nz) == 0:
        return [vocab["<cls>"]], [0]

    pairs = []
    for i in nz:
        g = gene_names[i]
        if g not in stoi:
            continue
        pairs.append((g, expr_row[i]))

    if len(pairs) == 0:
        return [vocab["<cls>"]], [0]

    # Sort by descending expression, truncate
    pairs = sorted(pairs, key=lambda x: -x[1])
    if len(pairs) > max_tokens:
        pairs = pairs[:max_tokens]

    gene_ids = [stoi[g] for g, v in pairs]
    value_bins = [min(int(v), n_bins - 1) for g, v in pairs]

    # Prepend CLS token
    CLS = vocab["<cls>"]
    gene_ids = [CLS] + gene_ids
    value_bins = [0] + value_bins

    return gene_ids, value_bins


# ============================================================
# GENERATE PER-CELL EMBEDDINGS
# ============================================================
def generate_cell_embeddings(adata, model, vocab, device, n_bins,
                             batch_size=DEFAULT_BATCH_SIZE,
                             max_tokens=DEFAULT_MAX_TOKENS):
    """Generate CLS embeddings for all cells."""
    stoi = vocab.get_stoi()
    gene_names = list(adata.var_names)

    X = adata.layers["X_binned"]
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    all_embeddings = []
    N = X.shape[0]

    print(f"[INFO] Generating CLS embeddings for {N} cells "
          f"(batch_size={batch_size})...")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_expr = X[start:end]

        batch_gene_ids = []
        batch_value_bins = []

        for i in range(batch_expr.shape[0]):
            ids, bins = tokenize_cell(
                batch_expr[i], gene_names, stoi, n_bins, vocab, max_tokens
            )
            batch_gene_ids.append(ids)
            batch_value_bins.append(bins)

        # Pad sequences
        max_len = max(len(ids) for ids in batch_gene_ids)
        id_mat = np.full((len(batch_gene_ids), max_len),
                         vocab["<pad>"], dtype=np.int64)
        val_mat = np.zeros((len(batch_gene_ids), max_len), dtype=np.float32)

        for i, (ids, bins) in enumerate(zip(batch_gene_ids, batch_value_bins)):
            id_mat[i, :len(ids)] = ids
            val_mat[i, :len(bins)] = bins

        ids_t = torch.tensor(id_mat).to(device)
        vals_t = torch.tensor(val_mat).to(device)
        mask = ids_t.eq(vocab["<pad>"])

        # Forward pass
        with torch.no_grad():
            out = model.encode_batch(
                ids_t, vals_t,
                src_key_padding_mask=mask,
                batch_size=ids_t.shape[0],
                time_step=0,
                batch_labels=None,
                return_np=False,
            )

        # Extract CLS token embedding
        if out.dim() == 3:
            cls_vecs = out[:, 0, :].detach().cpu().numpy()
        elif out.dim() == 2:
            if out.shape[0] == ids_t.shape[0]:
                cls_vecs = out.detach().cpu().numpy()
            else:
                cls_vecs = out[0, :].detach().cpu().numpy().reshape(1, -1)
        elif out.dim() == 1:
            cls_vecs = out.detach().cpu().numpy().reshape(1, -1)
        else:
            raise RuntimeError(f"Unexpected embedding shape: {out.shape}")

        all_embeddings.append(cls_vecs)

        if (end % (batch_size * 10) == 0) or end == N:
            print(f"  [{end}/{N}] cells processed")

        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


# ============================================================
# AGGREGATE: cell embeddings → cluster embeddings
# ============================================================
def aggregate_to_clusters(cell_embeddings, adata, cluster_key):
    """Mean-pool cell embeddings per cluster."""
    labels = adata.obs[cluster_key].values
    unique_labels = sorted(set(str(l) for l in labels))

    cluster_ids = []
    cluster_embs = []
    cluster_cell_ids = {}

    for cl in unique_labels:
        mask = np.array([str(l) == cl for l in labels])
        n_cells = mask.sum()
        if n_cells == 0:
            continue

        mean_emb = cell_embeddings[mask].mean(axis=0)
        cluster_ids.append(cl)
        cluster_embs.append(mean_emb)

        cell_indices = np.where(mask)[0]
        cluster_cell_ids[cl] = adata.obs_names[cell_indices].tolist()

        print(f"  {cl}: {n_cells} cells → emb dim {mean_emb.shape[0]}")

    cluster_embs = np.stack(cluster_embs)
    print(f"[INFO] Aggregated {len(cluster_ids)} clusters, "
          f"shape: {cluster_embs.shape}")

    return cluster_ids, cluster_embs, cluster_cell_ids


# ============================================================
# DETECT CLUSTER KEY
# ============================================================
def detect_cluster_key(adata):
    """Auto-detect the cluster column in adata.obs."""
    candidates = [
        "cell_type", "celltype", "CellType", "cell_type_ontology_term_id",
        "author_cell_type", "Annotations", "leiden", "louvain",
        "seurat_clusters", "cluster",
    ]
    for c in candidates:
        if c in adata.obs.columns:
            n_groups = adata.obs[c].nunique()
            if 2 <= n_groups <= 200:
                print(f"[INFO] Auto-detected cluster key: '{c}' "
                      f"({n_groups} groups)")
                return c

    raise ValueError(
        f"Could not auto-detect cluster key. "
        f"Available columns: {list(adata.obs.columns)}\n"
        f"Use --cluster-key to specify."
    )


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="scGPT CLS Embedding Generator for ELISA"
    )
    parser.add_argument("--h5ad", required=True,
                        help="Path to .h5ad file")
    parser.add_argument("--out", default=".",
                        help="Output directory")
    parser.add_argument("--name", default=None,
                        help="Dataset name (default: from h5ad filename)")
    parser.add_argument("--cluster-key", default=None,
                        help="obs column for clusters (auto-detected if not set)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"Path to scGPT model dir (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--gene-map", default=None,
                        help="Path to gene_info.csv (default: {model-dir}/gene_info.csv)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens per cell (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--save-cell-emb", action="store_true",
                        help="Also save per-cell embeddings as .npy")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (if data already has X_binned)")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Dataset name
    if args.name:
        ds_name = args.name
    else:
        ds_name = os.path.splitext(os.path.basename(args.h5ad))[0]

    # Gene map path
    gene_map_path = args.gene_map or os.path.join(args.model_dir, "gene_info.csv")

    print(f"\n{'=' * 60}")
    print(f"scGPT CLS Embedding Generator")
    print(f"{'=' * 60}")
    print(f"  Dataset:     {args.h5ad}")
    print(f"  Name:        {ds_name}")
    print(f"  Output:      {args.out}")
    print(f"  Model:       {args.model_dir}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"{'=' * 60}\n")

    # ── Load data ──
    print(f"[INFO] Loading: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"[INFO] Shape: {adata.n_obs} cells x {adata.n_vars} genes")

    # ── Cluster key ──
    if args.cluster_key:
        cluster_key = args.cluster_key
        if cluster_key not in adata.obs.columns:
            raise ValueError(
                f"'{cluster_key}' not in obs columns: "
                f"{list(adata.obs.columns)}"
            )
        n_groups = adata.obs[cluster_key].nunique()
        print(f"[INFO] Using cluster key: '{cluster_key}' ({n_groups} groups)")
    else:
        cluster_key = detect_cluster_key(adata)

    # ── Gene mapping (ENSG → symbols) ──
    if check_if_ensembl(adata):
        if os.path.exists(gene_map_path):
            gene_map = load_gene_map(gene_map_path)
            adata = apply_gene_map(adata, gene_map)
        else:
            # Try using feature_name column from var
            if "feature_name" in adata.var.columns:
                print("[INFO] Mapping ENSG → symbols using var['feature_name']")
                new_names = []
                for idx, row in adata.var.iterrows():
                    sym = str(row.get("feature_name", idx))
                    new_names.append(sym if sym and sym != "nan" else str(idx))
                adata.var_names = new_names
                adata.var_names_make_unique()
                n_mapped = sum(1 for g in adata.var_names
                               if not g.startswith("ENSG"))
                print(f"[INFO] Mapped {n_mapped}/{adata.n_vars} genes")
            else:
                print("[WARN] No gene map found and no feature_name column. "
                      "Using ENSG IDs (many will be missing from scGPT vocab).")
    else:
        print("[INFO] var_names appear to be gene symbols already")

    # ── Preprocess ──
    if args.skip_preprocess and "X_binned" in adata.layers:
        print("[INFO] Skipping preprocessing (X_binned already exists)")
    else:
        from scgpt.preprocess import Preprocessor
        print("[INFO] Preprocessing with scGPT Preprocessor...")
        pre = Preprocessor(
            use_key="X",
            filter_gene_by_counts=1,
            normalize_total=1e4,
            log1p=True,
            subset_hvg=None,
            binning=DEFAULT_N_BINS,
            result_binned_key="X_binned",
        )
        pre(adata)

    # ── Load model ──
    model, vocab, device, n_bins = load_scgpt(args.model_dir)

    # ── Generate per-cell embeddings ──
    cell_emb = generate_cell_embeddings(
        adata, model, vocab, device, n_bins,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
    print(f"[INFO] Cell embeddings shape: {cell_emb.shape}")

    # ── Quality check ──
    print("\n── Quality Check ──")
    var = cell_emb.var(axis=0)
    print(f"  Variance: min={var.min():.4e}, max={var.max():.4e}")

    # Sample cosine similarity
    n_sample = min(500, cell_emb.shape[0])
    idx = np.random.choice(cell_emb.shape[0], n_sample, replace=False)
    sample = cell_emb[idx]
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sample_norm = sample / norms
    sim = sample_norm @ sample_norm.T
    tri = sim[np.triu_indices(n_sample, k=1)]
    print(f"  Cosine sim (sample): mean={tri.mean():.4f}, "
          f"min={tri.min():.4f}, max={tri.max():.4f}")

    # ── Aggregate to clusters ──
    print(f"\n[INFO] Aggregating to clusters by '{cluster_key}'...")
    cluster_ids, cluster_embs, cluster_cell_ids = aggregate_to_clusters(
        cell_emb, adata, cluster_key
    )

    # ── Save cluster embeddings ──
    cluster_pt_path = os.path.join(
        args.out,
        f"scgpt_cluster_embeddings_by_{cluster_key}_{ds_name}.pt"
    )
    torch.save({
        "cluster_ids": cluster_ids,
        "cluster_scgpt_embeddings": torch.tensor(
            cluster_embs, dtype=torch.float32
        ),
        "cluster_cell_ids": cluster_cell_ids,
    }, cluster_pt_path)
    print(f"\n[SAVED] Cluster embeddings: {cluster_pt_path}")
    print(f"  Clusters: {len(cluster_ids)}")
    print(f"  Shape: {cluster_embs.shape}")

    # ── Save per-cell embeddings (optional) ──
    if args.save_cell_emb:
        cell_npy_path = os.path.join(
            args.out, f"scgpt_cell_embeddings_{ds_name}.npy"
        )
        np.save(cell_npy_path, cell_emb)
        print(f"[SAVED] Cell embeddings: {cell_npy_path}")

    # ── Save metadata CSV ──
    meta_path = os.path.join(args.out, f"metadata_cells_{ds_name}.csv")
    df = pd.DataFrame({"cell_id": list(adata.obs_names)})
    for col in adata.obs.columns:
        try:
            df[col] = list(adata.obs[col])
        except Exception:
            pass
    df.to_csv(meta_path, index=False)
    print(f"[SAVED] Cell metadata: {meta_path}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"DONE — scGPT embeddings for '{ds_name}'")
    print(f"{'=' * 60}")
    print(f"  Cells:    {adata.n_obs}")
    print(f"  Clusters: {len(cluster_ids)}")
    print(f"  Emb dim:  {cluster_embs.shape[1]}")
    print(f"\n  To fuse with semantic embeddings:")
    print(f"  python Unire_scGPTEmb_SemanticEmbClaude.py \\")
    print(f"      --semantic {args.out}/hybrid_cluster_embeddings_with_cell_metadata_{ds_name}.pt \\")
    print(f"      --scgpt {cluster_pt_path} \\")
    print(f"      --cells-csv {meta_path} \\")
    print(f"      --out {args.out}/fused_{ds_name}.pt")
    print()


if __name__ == "__main__":
    main()
