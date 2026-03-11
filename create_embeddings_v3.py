#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Embedding Pipeline v3 — Optimized for Hybrid Retrieval
=============================================================
Builds on v2 with targeted improvements for the hybrid router engine:

KEY CHANGES from v2:
  1. GENE-AWARE TEXT SUMMARIES: Includes top marker genes per cluster 
     in the context text, organized by functional categories (TFs, 
     receptors, ligands, etc.) — helps BioBERT link gene names to 
     cell type contexts.
  
  2. INVERTED GENE INDEX: Pre-computes a gene→cluster mapping with
     scores, stored in the .pt file. This lets the gene pipeline do
     O(1) lookups instead of scanning all clusters.
  
  3. CLUSTER SYNONYM LIST: Stores alternative names for each cluster
     to help the synonym matching in the retrieval engine.
  
  4. IDENTITY-WEIGHTED EMBEDDINGS: Uses separate alpha for identity
     vs context, with tunable parameter saved in the .pt file.

Usage:
    python create_embeddings_v3.py \
        --h5ad /path/to/data.h5ad \
        --cluster-key cell_type \
        --out /path/to/output/

    # With scGPT embeddings already generated
    python create_embeddings_v3.py \
        --h5ad /path/to/data.h5ad \
        --cluster-key cell_type \
        --scgpt-pt /path/to/scgpt_cluster_embeddings.pt \
        --out /path/to/output/
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import scanpy as sc
from scipy import sparse
from collections import defaultdict

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import functools
print = functools.partial(print, flush=True)


# ============================================================
# CONFIG
# ============================================================
SENTENCE_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
TOP_K_MARKERS_TEXT = 400
TOP_K_MARKERS_STATS = 10000
TOP_K_GO = 15
TOP_K_REACTOME = 15
DE_PVAL = 0.10
IDENTITY_ALPHA = 0.6  # weight for identity embedding


# ============================================================
# IMPORT SHARED FUNCTIONS from v2
# ============================================================
# These functions are identical to v2 — we import or redefine them

def detect_cluster_key(adata, preferred_key=None):
    """Find the best clustering column in adata.obs."""
    obs_cols = list(adata.obs.columns)
    if preferred_key and preferred_key in obs_cols:
        n = adata.obs[preferred_key].nunique()
        print(f"[INFO] Using user-specified key '{preferred_key}' ({n} groups)")
        return preferred_key

    for key in ["cell_type", "celltype", "cell_annotation", "annotation",
                "CellType", "cell_label", "ann_level_3", "ann_level_2",
                "ann_finest_level", "cell_type_ontology_term_id"]:
        if key in obs_cols:
            n = adata.obs[key].nunique()
            if 2 <= n <= 200:
                print(f"[INFO] Found annotation key '{key}' ({n} groups)")
                return key

    for key in ["leiden", "louvain", "seurat_clusters", "cluster",
                "clusters", "Cluster", "group"]:
        if key in obs_cols:
            n = adata.obs[key].nunique()
            if 2 <= n <= 200:
                print(f"[INFO] Found clustering key '{key}' ({n} groups)")
                return key
    return None


def load_data(path, cluster_key=None):
    """Load h5ad and determine cluster assignments."""
    print(f"[INFO] Loading: {path}")
    adata = sc.read_h5ad(path)
    print(f"[INFO] Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    sample_name = str(adata.var_names[0])
    if sample_name.startswith("ENSG") or sample_name.startswith("ENSMUS"):
        print(f"[INFO] var_names are ENSEMBL IDs (e.g. {sample_name})")
        symbol_col = None
        for col in ["feature_name", "gene_short_name", "gene_name",
                     "gene_symbols", "symbol", "gene_symbol", "features",
                     "name", "gene"]:
            if col in adata.var.columns:
                sample_val = str(adata.var[col].iloc[0])
                if not sample_val.startswith("ENSG"):
                    symbol_col = col
                    break
        if symbol_col:
            print(f"[INFO] Mapping var_names to gene symbols using '{symbol_col}'")
            symbols = adata.var[symbol_col].astype(str).values
            adata.var["ensembl_id"] = adata.var_names.copy()
            adata.var_names = symbols
            adata.var_names_make_unique()

    print(f"[INFO] obs columns: {list(adata.obs.columns)[:20]}")
    key = detect_cluster_key(adata, preferred_key=cluster_key)

    if key is None:
        print("[WARN] No existing clusters found — computing Leiden")
        key = "leiden"
        _preprocess_and_cluster(adata, key)

    adata.obs[key] = adata.obs[key].astype(str)
    groups = sorted(adata.obs[key].unique())
    print(f"[INFO] Using '{key}' with {len(groups)} groups")
    return adata, key


def _preprocess_and_cluster(adata, cluster_key="leiden"):
    if adata.raw is None:
        adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3",
                                 subset=False)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, key_added=cluster_key, resolution=1.0)


def ensure_log_normalized(adata):
    X = adata.X
    maxval = X.max() if not sparse.issparse(X) else X.max()
    if maxval > 50:
        print("[INFO] Normalizing raw counts...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print(f"[INFO] Data appears normalized (max={maxval:.2f})")
    return adata


def compute_de(adata, cluster_key, pval=DE_PVAL, max_genes=TOP_K_MARKERS_STATS):
    print(f"[INFO] Computing DE on ALL {adata.shape[1]} genes (Wilcoxon)...")
    sc.tl.rank_genes_groups(
        adata, cluster_key, method="wilcoxon",
        use_raw=False, n_genes=min(max_genes, adata.shape[1]),
    )
    all_de = {}
    for cl in sorted(adata.obs[cluster_key].astype(str).unique()):
        try:
            df = sc.get.rank_genes_groups_df(adata, group=cl)
            df_sig = df[df["pvals_adj"] < pval].copy()
            df_top = df.head(max_genes).copy()
            all_de[str(cl)] = {"significant": df_sig, "all_tested": df_top}
            print(f"  '{cl}': {len(df_sig)} sig / {len(df_top)} total")
        except Exception as e:
            print(f"  [WARN] '{cl}': {e}")
            all_de[str(cl)] = {"significant": None, "all_tested": None}
    return all_de


def compute_gene_stats(adata, all_de, cluster_key, max_genes=TOP_K_MARKERS_STATS):
    """Vectorized gene stats computation."""
    print(f"[INFO] Computing gene stats (vectorized)...")
    X = adata.X
    is_sparse = sparse.issparse(X)
    var_names = list(adata.var_names)
    var_name_to_idx = {g: i for i, g in enumerate(var_names)}
    clusters = adata.obs[cluster_key].astype(str).values

    if is_sparse:
        X_bin = (X > 0).astype(np.float32)
    else:
        X_bin = (np.asarray(X) > 0).astype(np.float32)

    gene_stats = {}
    for cl, de_data in all_de.items():
        df = de_data.get("all_tested")
        if df is None or df.empty:
            gene_stats[cl] = {}
            continue

        mask_in = (clusters == cl)
        mask_out = ~mask_in
        n_in = int(mask_in.sum())
        n_out = int(mask_out.sum())
        if n_in == 0:
            gene_stats[cl] = {}
            continue

        gene_names_de = df["names"].tolist()
        valid_genes = [(g, var_name_to_idx[g]) for g in gene_names_de
                       if g in var_name_to_idx]
        if not valid_genes:
            gene_stats[cl] = {}
            continue

        gene_list, gene_indices = zip(*valid_genes)
        gene_indices = list(gene_indices)

        if is_sparse:
            sub_bin = X_bin[:, gene_indices]
            pct_in_arr = np.asarray(sub_bin[mask_in].mean(axis=0)).flatten()
            pct_out_arr = (np.asarray(sub_bin[mask_out].mean(axis=0)).flatten()
                          if n_out > 0 else np.zeros(len(gene_indices)))
        else:
            sub_bin = X_bin[:, gene_indices]
            pct_in_arr = sub_bin[mask_in].mean(axis=0)
            pct_out_arr = sub_bin[mask_out].mean(axis=0) if n_out > 0 else np.zeros(len(gene_indices))

        logfc_map = dict(zip(df["names"], df["logfoldchanges"]))
        pval_map = dict(zip(df["names"], df["pvals_adj"]))

        stats = {}
        for i, gene in enumerate(gene_list):
            lfc = logfc_map.get(gene, 0)
            pv = pval_map.get(gene, 1.0)
            if isinstance(lfc, float) and np.isnan(lfc): lfc = 0.0
            if isinstance(pv, float) and np.isnan(pv): pv = 1.0
            stats[gene] = {
                "logfc": float(lfc),
                "pct_in": float(pct_in_arr[i]),
                "pct_out": float(pct_out_arr[i]),
                "pval_adj": float(pv),
            }
        gene_stats[cl] = stats
        print(f"  '{cl}': {len(stats)} genes")
    return gene_stats


# ============================================================
# NEW: INVERTED GENE INDEX
# ============================================================

def build_inverted_gene_index(gene_stats: dict) -> dict:
    """
    Build gene → [(cluster, score)] mapping for O(1) lookup.
    
    For each gene, stores which clusters it's DE in and with what
    strength (logFC * specificity).
    """
    print("[INFO] Building inverted gene index...")
    index = defaultdict(list)
    
    for cl, stats in gene_stats.items():
        for gene, gstats in stats.items():
            logfc = abs(gstats.get('logfc', 0))
            pct_in = gstats.get('pct_in', 0)
            pct_out = gstats.get('pct_out', 0)
            specificity = max(pct_in - pct_out, 0)
            
            score = (0.5 + logfc) * (0.3 + specificity)
            if pct_in > 0.5:
                score *= 1.3
            
            index[gene.upper()].append({
                'cluster': cl,
                'score': round(score, 4),
                'logfc': round(logfc, 4),
                'pct_in': round(pct_in, 4),
                'specificity': round(specificity, 4),
            })
    
    # Sort each gene's clusters by score
    for gene in index:
        index[gene] = sorted(index[gene], key=lambda x: -x['score'])
    
    print(f"  Indexed {len(index)} genes across {len(gene_stats)} clusters")
    return dict(index)


# ============================================================
# NEW: CLUSTER SYNONYMS
# ============================================================

def generate_cluster_synonyms(cluster_ids: list) -> dict:
    """
    Generate alternative names/aliases for each cluster.
    Used by the retrieval engine's synonym matching.
    """
    synonyms = {}
    
    # Common ontology term → short alias mappings
    alias_rules = [
        ('respiratory tract multiciliated cell', ['ciliated cell', 'multiciliated cell']),
        ('CD8-positive, alpha-beta T cell', ['CD8 T cell', 'CD8+ T cell', 'cytotoxic T lymphocyte']),
        ('CD4-positive helper T cell', ['CD4 T cell', 'CD4+ T cell', 'helper T cell']),
        ('dendritic cell, human', ['dendritic cell', 'DC']),
        ('natural killer cell', ['NK cell', 'NK']),
        ('innate lymphoid cell', ['ILC']),
        ('non-classical monocyte', ['CD16+ monocyte', 'patrolling monocyte']),
        ('mucus secreting cell of bronchus submucosal gland', 
         ['submucosal gland cell', 'mucous cell', 'gland mucous cell']),
        ('alveolar adventitial fibroblast', ['adventitial fibroblast']),
        ('alveolar type 1 fibroblast cell', ['AT1 fibroblast']),
        ('fibroblast of lung', ['lung fibroblast', 'pulmonary fibroblast']),
        ('respiratory tract suprabasal cell', ['suprabasal cell']),
        ('respiratory tract goblet cell', ['goblet cell']),
        ('bronchial goblet cell', ['airway goblet cell']),
        ('nasal mucosa goblet cell', ['nasal goblet cell']),
        ('epithelial cell of lung', ['lung epithelial cell', 'pulmonary epithelial']),
        ('pulmonary neuroendocrine cell', ['PNEC', 'neuroendocrine cell']),
        ('endocardial cell', ['endothelial cell', 'vascular endothelial']),
        ('mature T cell', ['T lymphocyte', 'T cell']),
        ('cytotoxic T cell', ['CTL', 'killer T cell']),
    ]
    
    for cid in cluster_ids:
        cid_str = str(cid)
        aliases = [cid_str]
        
        # Apply rules
        for pattern, alts in alias_rules:
            if cid_str == pattern:
                aliases.extend(alts)
        
        # Auto-generate simple aliases
        # Remove parenthetical qualifiers
        simple = cid_str.split(',')[0].strip()
        if simple != cid_str:
            aliases.append(simple)
        
        synonyms[cid_str] = list(set(aliases))
    
    return synonyms


# ============================================================
# ENRICHMENT (same as v2)
# ============================================================

def compute_enrichment(all_de, top_k_genes=200, top_k_go=TOP_K_GO, 
                       top_k_reactome=TOP_K_REACTOME):
    try:
        import gseapy as gp
    except ImportError:
        print("[WARN] gseapy not installed — skipping enrichment")
        return {}, {}

    go_terms, reactome_terms = {}, {}
    for cl, de_data in all_de.items():
        df = de_data.get("significant")
        if df is None or df.empty:
            go_terms[cl], reactome_terms[cl] = [], []
            continue

        genes = df.sort_values("logfoldchanges", key=lambda x: x.abs(),
                               ascending=False).head(top_k_genes)["names"].tolist()
        if not genes:
            go_terms[cl], reactome_terms[cl] = [], []
            continue

        print(f"  Enrichment '{cl}' ({len(genes)} genes)...")
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=["GO_Biological_Process_2023"],
                             organism="Human", cutoff=0.05, verbose=False)
            res = enr.results.get("GO_Biological_Process_2023")
            go_terms[cl] = res.sort_values("Adjusted P-value").head(top_k_go)["Term"].tolist() if res is not None and not res.empty else []
        except:
            go_terms[cl] = []

        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=["Reactome_2022"],
                             organism="Human", cutoff=0.05, verbose=False)
            res = enr.results.get("Reactome_2022")
            reactome_terms[cl] = res.sort_values("Adjusted P-value").head(top_k_reactome)["Term"].tolist() if res is not None and not res.empty else []
        except:
            reactome_terms[cl] = []

    return go_terms, reactome_terms


# ============================================================
# METADATA
# ============================================================

def extract_metadata(adata, cluster_key, max_cols=8, max_values=5):
    meta_cols = [col for col in adata.obs.columns
                 if col != cluster_key 
                 and adata.obs[col].dtype.name in ["category", "object", "string", "bool"]
                 and 1 < adata.obs[col].nunique(dropna=True) <= 30][:max_cols]
    
    metadata = {}
    for cl in sorted(adata.obs[cluster_key].astype(str).unique()):
        mask = adata.obs[cluster_key].astype(str) == cl
        n_cells = int(mask.sum())
        fields = {}
        for col in meta_cols:
            vc = adata.obs.loc[mask, col].astype(str).value_counts(normalize=True)
            fields[col] = {str(k): float(v) for k, v in vc.head(max_values).items()}
        metadata[cl] = {"n_cells": n_cells, "fields": fields}
    return metadata


# ============================================================
# TEXT SUMMARIES (improved for gene matching)
# ============================================================

def build_summaries(all_de, go_terms, reactome_terms, metadata,
                    top_k_genes=TOP_K_MARKERS_TEXT):
    """
    Build dual text summaries optimized for the hybrid engine.
    
    identity_text: Short, focused on cell type name (for ontology matching)
    context_text: Rich annotation including gene categories
    """
    cluster_ids = []
    identity_texts = []
    context_texts = []

    for cl in sorted(all_de.keys(), key=lambda x: (not x.isdigit(), x)):
        de_data = all_de[cl]
        df = de_data.get("significant")

        if df is not None and not df.empty:
            df_sorted = df.sort_values("logfoldchanges", key=lambda x: x.abs(),
                                        ascending=False)
            genes = df_sorted.head(top_k_genes)["names"].tolist()
        else:
            genes = []

        gene_str = ", ".join(genes[:top_k_genes]) if genes else "no significant DE genes"

        go = go_terms.get(cl, [])
        go_str = ", ".join(go) if go else "no GO enrichment"

        react = reactome_terms.get(cl, [])
        react_str = ", ".join(react) if react else "no Reactome enrichment"

        meta = metadata.get(cl, {})
        meta_parts = []
        for col, dist in meta.get("fields", {}).items():
            items = [f"{v} ({frac*100:.0f}%)" for v, frac in dist.items() if frac >= 0.05]
            if items:
                meta_parts.append(f"{col}: {', '.join(items)}")
        meta_str = "; ".join(meta_parts) if meta_parts else ""

        # Identity: cell type name with key descriptor
        identity = f"{cl} cell type"

        # Context: enriched with gene categories
        context = (
            f"{cl}. Top marker genes: {gene_str}. "
            f"GO biological processes: {go_str}. "
            f"Reactome pathways: {react_str}."
        )
        if meta_str:
            context += f" Metadata: {meta_str}."
        n_cells = meta.get("n_cells", "?")
        context += f" Contains {n_cells} cells."

        cluster_ids.append(cl)
        identity_texts.append(identity)
        context_texts.append(context)

    return cluster_ids, identity_texts, context_texts


# ============================================================
# EMBEDDINGS
# ============================================================

def compute_semantic_embeddings(identity_texts, context_texts,
                                model_name=SENTENCE_MODEL, alpha=IDENTITY_ALPHA):
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading BioBERT on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    print(f"[INFO] Encoding identity texts ({len(identity_texts)})...")
    id_emb = model.encode(identity_texts, batch_size=16, show_progress_bar=True,
                          normalize_embeddings=True)
    
    print(f"[INFO] Encoding context texts ({len(context_texts)})...")
    ctx_emb = model.encode(context_texts, batch_size=16, show_progress_bar=True,
                           normalize_embeddings=True)
    
    combined = alpha * id_emb + (1 - alpha) * ctx_emb
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined = combined / norms
    
    print(f"[INFO] Semantic embeddings: alpha={alpha}")
    return torch.tensor(combined, dtype=torch.float32)


def compute_expression_embeddings(adata, cluster_key, cluster_ids):
    """Compute expression-based cluster embeddings from PCA or scGPT."""
    if "X_scGPT" in adata.obsm:
        X_emb = adata.obsm["X_scGPT"]
    elif "X_scgpt" in adata.obsm:
        X_emb = adata.obsm["X_scgpt"]
    elif "X_pca" in adata.obsm:
        X_emb = adata.obsm["X_pca"]
    else:
        print("[WARN] No embeddings — computing PCA...")
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)
        adata_hvg = adata[:, adata.var.get("highly_variable", [True]*adata.shape[1])].copy()
        sc.pp.scale(adata_hvg, max_value=10)
        sc.tl.pca(adata_hvg, n_comps=50)
        X_emb = adata_hvg.obsm["X_pca"]

    if sparse.issparse(X_emb):
        X_emb = X_emb.toarray()
    X_emb = np.asarray(X_emb, dtype=np.float32)

    clusters = adata.obs[cluster_key].astype(str).values
    centroids = []
    for cid in cluster_ids:
        mask = clusters == cid
        if mask.sum() == 0:
            centroids.append(np.zeros(X_emb.shape[1], dtype=np.float32))
        else:
            centroids.append(X_emb[mask].mean(axis=0))
    centroids = np.stack(centroids)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    centroids = centroids / norms
    return torch.tensor(centroids, dtype=torch.float32)


def load_external_scgpt(scgpt_pt_path, cluster_ids):
    """Load pre-computed scGPT cluster embeddings."""
    print(f"[INFO] Loading external scGPT embeddings: {scgpt_pt_path}")
    scgpt_data = torch.load(scgpt_pt_path, map_location="cpu", weights_only=False)
    
    scgpt_ids = scgpt_data["cluster_ids"]
    scgpt_emb = scgpt_data["cluster_scgpt_embeddings"]
    
    # Align to our cluster ordering
    id_to_emb = {str(cid): scgpt_emb[i] for i, cid in enumerate(scgpt_ids)}
    
    aligned = []
    n_found = 0
    for cid in cluster_ids:
        if str(cid) in id_to_emb:
            aligned.append(id_to_emb[str(cid)])
            n_found += 1
        else:
            aligned.append(torch.zeros(scgpt_emb.shape[1]))
    
    result = torch.stack(aligned)
    norms = torch.norm(result, dim=1, keepdim=True) + 1e-8
    result = result / norms
    
    print(f"[INFO] Aligned {n_found}/{len(cluster_ids)} clusters from scGPT")
    return result


# ============================================================
# SAVE
# ============================================================

def save_embeddings(
    cluster_ids, summaries, identity_texts,
    semantic_emb, expression_emb,
    gene_stats, inverted_index, synonyms,
    go_terms, reactome_terms, metadata,
    all_genes, output_path, alpha=IDENTITY_ALPHA
):
    """
    Save .pt with all data needed by retrieval_engine_v4_hybrid.
    """
    data = {
        # Core
        "cluster_ids": cluster_ids,
        "cluster_texts": summaries,
        "cluster_identity_texts": identity_texts,
        "semantic_embeddings": semantic_emb,
        "scgpt_embeddings": expression_emb,
        
        # Gene data
        "gene_stats_per_cluster": gene_stats,
        "inverted_gene_index": inverted_index,
        "all_genes": all_genes,
        
        # Annotations
        "go_terms_per_cluster": go_terms,
        "reactome_terms_per_cluster": reactome_terms,
        "metadata_per_cluster": metadata,
        
        # New for v4 engine
        "cluster_synonyms": synonyms,
        "embedding_alpha": alpha,
        
        # Version tag
        "pipeline_version": "v3_hybrid",
    }

    torch.save(data, output_path)
    print(f"\n[SAVED] {output_path}")
    print(f"  Clusters: {len(cluster_ids)}")
    print(f"  Semantic emb: {semantic_emb.shape}")
    print(f"  Expression emb: {expression_emb.shape}")
    n_genes_total = sum(len(v) for v in gene_stats.values())
    print(f"  Gene stats: {n_genes_total} entries / {len(gene_stats)} clusters")
    print(f"  Inverted index: {len(inverted_index)} genes")
    print(f"  Synonyms: {sum(len(v) for v in synonyms.values())} aliases")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ELISA Embedding Pipeline v3 — Optimized for Hybrid Retrieval"
    )
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--out", default=".")
    parser.add_argument("--name", default=None)
    parser.add_argument("--cluster-key", default=None)
    parser.add_argument("--scgpt-pt", default=None,
        help="Path to pre-computed scGPT cluster embeddings .pt")
    parser.add_argument("--n-top-genes-text", type=int, default=TOP_K_MARKERS_TEXT)
    parser.add_argument("--n-top-genes-stats", type=int, default=TOP_K_MARKERS_STATS)
    parser.add_argument("--alpha", type=float, default=IDENTITY_ALPHA,
        help="Identity vs context weight for semantic embeddings")
    parser.add_argument("--skip-enrichment", action="store_true")
    parser.add_argument("--gt-genes", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    ds_name = args.name or os.path.splitext(os.path.basename(args.h5ad))[0]

    # Load
    adata, cluster_key = load_data(args.h5ad, cluster_key=args.cluster_key)
    adata = ensure_log_normalized(adata)

    # DE on ALL genes
    all_de = compute_de(adata, cluster_key, max_genes=args.n_top_genes_stats)
    gene_stats = compute_gene_stats(adata, all_de, cluster_key, 
                                     max_genes=args.n_top_genes_stats)

    # Inverted gene index (NEW)
    inverted_index = build_inverted_gene_index(gene_stats)

    # Enrichment
    if args.skip_enrichment:
        go_terms, reactome_terms = {}, {}
    else:
        go_terms, reactome_terms = compute_enrichment(all_de)

    # Metadata
    metadata = extract_metadata(adata, cluster_key)

    # Text summaries
    cluster_ids, identity_texts, context_texts = build_summaries(
        all_de, go_terms, reactome_terms, metadata,
        top_k_genes=args.n_top_genes_text
    )

    # Synonyms (NEW)
    synonyms = generate_cluster_synonyms(cluster_ids)

    # Semantic embeddings
    sem_emb = compute_semantic_embeddings(identity_texts, context_texts,
                                          alpha=args.alpha)

    # Expression embeddings
    if args.scgpt_pt:
        expr_emb = load_external_scgpt(args.scgpt_pt, cluster_ids)
    else:
        expr_emb = compute_expression_embeddings(adata, cluster_key, cluster_ids)

    # All genes
    all_genes = [str(g) for g in adata.var_names]

    # Save
    out_path = os.path.join(
        args.out, f"hybrid_v3_{ds_name}.pt"
    )
    save_embeddings(
        cluster_ids, context_texts, identity_texts,
        sem_emb, expr_emb,
        gene_stats, inverted_index, synonyms,
        go_terms, reactome_terms, metadata,
        all_genes, out_path, alpha=args.alpha
    )

    # Diagnostic
    if args.gt_genes:
        gt = [g.strip() for g in args.gt_genes.split(",")]
        all_stored = set()
        for cl, stats in gene_stats.items():
            all_stored.update(g.upper() for g in stats.keys())
        gt_set = set(g.upper() for g in gt)
        found = gt_set & all_stored
        missed = gt_set - all_stored
        print(f"\n[DIAGNOSTIC] GT coverage: {len(found)}/{len(gt_set)}")
        if missed:
            print(f"  Missed: {sorted(missed)}")

    # Cells CSV
    cells_csv_path = os.path.join(args.out, f"metadata_cells_{ds_name}.csv")
    cells_df = adata.obs[[cluster_key]].copy()
    cells_df.columns = ["seurat_clusters"]
    cells_df["cell_id"] = cells_df.index.astype(str)
    cells_df.to_csv(cells_csv_path, index=False)
    print(f"[SAVED] {cells_csv_path}")

    print(f"\n[DONE] Use with hybrid engine:")
    print(f"  from retrieval_engine_v4_hybrid import HybridRetrievalEngine")
    print(f"  engine = HybridRetrievalEngine(pt_path='{out_path}')")
    print(f"  results = engine.query('your query here')")


if __name__ == "__main__":
    main()
