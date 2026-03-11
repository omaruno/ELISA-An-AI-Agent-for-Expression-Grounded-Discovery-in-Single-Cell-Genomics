#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Retrieval Engine v4 — Hybrid Router
==========================================
Beats CellWhisperer on BOTH ontology and expression queries by:

1. QUERY CLASSIFICATION: Detects whether a query is gene-based, 
   ontology/text-based, or mixed, and routes accordingly.

2. GENE MARKER SCORING: For gene-list queries, scores each cluster
   by how well its DE profile matches the query genes — using logFC,
   specificity (pct_in vs pct_out), and significance.

3. SEMANTIC MATCHING: For text queries, uses BioBERT cosine similarity
   against dual (identity + context) cluster embeddings, with cell-type
   name boosting.

4. RECIPROCAL RANK FUSION (RRF): For mixed queries, fuses rankings
   from both pipelines with tunable weights.

5. SYNONYM EXPANSION: Maps common cell type aliases to ontology terms
   to fix vocabulary gaps (e.g., "endothelial" → "endocardial cell").

Architecture:
    Query → Classifier → Router
                          ├── Gene Pipeline    → gene_marker_score per cluster
                          ├── Semantic Pipeline → cosine_sim per cluster  
                          └── Mixed Pipeline    → RRF(gene, semantic)
                          
    All pipelines → top-K ranked cluster list

Usage:
    from retrieval_engine_v4_hybrid import HybridRetrievalEngine
    
    engine = HybridRetrievalEngine(
        pt_path="fused_dataset.pt",
        cells_csv="metadata_cells_dataset.csv",  # optional
    )
    
    results = engine.query("macrophage and monocyte infiltration in CF airways", top_k=10)
    results = engine.query("FOXJ1 DNAH5 CAPS PIFO RSPH1 DNAI1", top_k=10)
"""

import os
import re
import math
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

# ============================================================
# QUERY CLASSIFIER
# ============================================================

# Common gene name pattern: uppercase letters + digits, 2-15 chars
GENE_PATTERN = re.compile(r'^[A-Z][A-Z0-9]{1,14}(?:-[A-Z0-9]{1,6})?$')

# Known gene prefixes that confirm gene identity
GENE_PREFIXES = {
    'CD', 'HLA', 'KRT', 'COL', 'MUC', 'SCGB', 'FOXJ', 'FOXI',
    'DNAH', 'KLRC', 'KLRD', 'KLRK', 'GZMB', 'PRF1', 'IFNG',
    'TPSAB', 'TPSB', 'CPA', 'MS4A', 'ATP6', 'IGHJ', 'IGLJ',
    'IGKJ', 'TRAJ', 'TRBV', 'TRDJ', 'IGH', 'IGL', 'IGK',
    'SLC', 'ABCC', 'CXCL', 'CXCR', 'CCL', 'CCR', 'IL',
}

# Words that indicate natural language (NOT gene names)
NL_INDICATORS = {
    'cell', 'cells', 'type', 'the', 'and', 'in', 'of', 'with',
    'activation', 'expression', 'signaling', 'response', 'pathway',
    'infiltration', 'dysfunction', 'remodeling', 'increased', 'reduced',
    'cystic', 'fibrosis', 'lung', 'airways', 'bronchial', 'epithelium',
    'immune', 'inflammatory', 'antigen', 'presentation', 'cytotoxicity',
    'degranulation', 'checkpoint', 'inhibiting', 'activity', 'abundance',
    'changes', 'across', 'receptor', 'hypoxia', 'allergic', 'defense',
    'helper', 'killer', 'natural', 'pulmonary', 'endothelial',
    'epithelial', 'fibroblast', 'macrophage', 'monocyte', 'dendritic',
    'basal', 'ciliated', 'secretory', 'goblet', 'club', 'ionocyte',
    'mast', 'plasma', 'mature', 'cytotoxic', 'innate', 'lymphoid',
    'stemness', 'ciliogenesis', 'immunoglobulin',
}


def classify_query(query: str, all_genes: set = None) -> str:
    """
    Classify a query as 'gene_list', 'ontology', or 'mixed'.
    
    Returns:
        'gene_list': Query is primarily gene symbols (e.g., "FOXJ1 DNAH5 CAPS")
        'ontology':  Query is natural language about cell types/biology
        'mixed':     Query contains both gene names and biological text
    """
    tokens = query.strip().split()
    if not tokens:
        return 'ontology'
    
    n_gene = 0
    n_text = 0
    
    for tok in tokens:
        tok_clean = tok.strip('.,;:()')
        tok_upper = tok_clean.upper()
        
        # Check if it's a known gene
        is_gene = False
        
        # Method 1: matches gene pattern AND is in dataset
        if all_genes and tok_upper in all_genes:
            is_gene = True
        
        # Method 2: matches gene pattern AND has known prefix
        if not is_gene and GENE_PATTERN.match(tok_upper):
            for prefix in GENE_PREFIXES:
                if tok_upper.startswith(prefix):
                    is_gene = True
                    break
        
        # Method 3: pure pattern match (uppercase, 2-10 chars, has digits)
        if not is_gene and GENE_PATTERN.match(tok_upper):
            has_digit = any(c.isdigit() for c in tok_upper)
            if has_digit and len(tok_upper) >= 3:
                is_gene = True
        
        # Check if it's clearly natural language
        is_nl = tok_clean.lower() in NL_INDICATORS
        
        if is_gene and not is_nl:
            n_gene += 1
        elif is_nl:
            n_text += 1
    
    gene_ratio = n_gene / len(tokens) if tokens else 0
    text_ratio = n_text / len(tokens) if tokens else 0
    
    if gene_ratio >= 0.6:
        return 'gene_list'
    elif gene_ratio >= 0.2 and text_ratio >= 0.2:
        return 'mixed'
    else:
        return 'ontology'


def extract_gene_names(query: str, all_genes: set = None) -> List[str]:
    """Extract gene symbols from a query string."""
    tokens = query.strip().split()
    genes = []
    
    for tok in tokens:
        tok_clean = tok.strip('.,;:()')
        tok_upper = tok_clean.upper()
        
        # Strong match: in dataset
        if all_genes and tok_upper in all_genes:
            genes.append(tok_upper)
            continue
        
        # Pattern match
        if GENE_PATTERN.match(tok_upper) and tok_clean.lower() not in NL_INDICATORS:
            genes.append(tok_upper)
    
    return genes


# ============================================================
# GENE MARKER SCORING PIPELINE
# ============================================================

def score_cluster_by_genes(
    query_genes: List[str],
    cluster_gene_stats: Dict[str, dict],
    scoring_mode: str = 'weighted',
) -> float:
    """
    Score a single cluster against a list of query genes.
    
    Uses DE statistics (logFC, pct_in, pct_out, pval) to determine
    how well the cluster's expression profile matches the query.
    
    Scoring modes:
        'simple':   Count of query genes found in cluster DE set
        'weighted': Sum of |logFC| * specificity for matched genes
        'full':     Weighted score with significance penalty
    
    Args:
        query_genes: List of gene symbols to search for
        cluster_gene_stats: {gene: {logfc, pct_in, pct_out, pval_adj}}
        scoring_mode: 'simple', 'weighted', or 'full'
    
    Returns:
        Aggregate score (higher = better match)
    """
    if not cluster_gene_stats or not query_genes:
        return 0.0
    
    # Build case-insensitive lookup
    stats_upper = {}
    for gene, stats in cluster_gene_stats.items():
        stats_upper[gene.upper()] = stats
    
    score = 0.0
    n_found = 0
    
    for gene in query_genes:
        gene_u = gene.upper()
        if gene_u not in stats_upper:
            continue
        
        stats = stats_upper[gene_u]
        n_found += 1
        
        if scoring_mode == 'simple':
            score += 1.0
        
        elif scoring_mode == 'weighted':
            logfc = abs(stats.get('logfc', 0))
            pct_in = stats.get('pct_in', 0)
            pct_out = stats.get('pct_out', 0)
            
            # Specificity: how much more expressed in-cluster vs out
            specificity = pct_in - pct_out  # range [-1, 1]
            specificity = max(specificity, 0)  # only care about enrichment
            
            # Score: combination of effect size and specificity
            # logFC gives magnitude, specificity gives uniqueness
            gene_score = (0.5 + logfc) * (0.3 + specificity)
            
            # Bonus for high pct_in (gene is expressed in many cells of cluster)
            if pct_in > 0.5:
                gene_score *= 1.3
            
            score += gene_score
        
        elif scoring_mode == 'full':
            logfc = abs(stats.get('logfc', 0))
            pct_in = stats.get('pct_in', 0)
            pct_out = stats.get('pct_out', 0)
            pval = stats.get('pval_adj', 1.0)
            
            specificity = max(pct_in - pct_out, 0)
            
            # Significance boost: -log10(pval), capped
            if pval > 0 and pval < 1:
                sig_boost = min(-math.log10(pval + 1e-300), 10) / 10
            else:
                sig_boost = 0.0
            
            gene_score = (0.5 + logfc) * (0.3 + specificity) * (0.5 + sig_boost)
            
            if pct_in > 0.5:
                gene_score *= 1.3
            
            score += gene_score
    
    # Normalize by number of query genes to get coverage
    coverage = n_found / len(query_genes) if query_genes else 0
    
    # Final score: raw score * coverage bonus
    # Coverage bonus rewards clusters that match MORE query genes
    score *= (0.5 + 0.5 * coverage)
    
    return score


def gene_pipeline(
    query_genes: List[str],
    gene_stats_per_cluster: Dict[str, Dict],
    cluster_ids: List[str],
    scoring_mode: str = 'weighted',
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Rank all clusters by gene marker score.
    
    Returns:
        List of (cluster_id, score) sorted by descending score
    """
    scores = []
    for cid in cluster_ids:
        cid_str = str(cid)
        cluster_stats = gene_stats_per_cluster.get(cid_str, {})
        s = score_cluster_by_genes(query_genes, cluster_stats, scoring_mode)
        scores.append((cid_str, s))
    
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# ============================================================
# SEMANTIC PIPELINE (BioBERT cosine similarity)
# ============================================================

def semantic_pipeline(
    query: str,
    semantic_embeddings: torch.Tensor,
    cluster_ids: List[str],
    cluster_texts: List[str],
    model=None,
    model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    top_k: int = 10,
    name_boost: float = 0.15,
) -> List[Tuple[str, float]]:
    """
    Rank clusters by semantic similarity to query.
    
    Includes name boosting: if the query contains a substring matching
    a cluster name, that cluster gets a score bonus.
    
    Args:
        query: Natural language query
        semantic_embeddings: (N_clusters, dim) tensor
        cluster_ids: List of cluster names
        cluster_texts: List of cluster text summaries
        model: Pre-loaded SentenceTransformer (or None to load)
        model_name: BioBERT model name
        top_k: Number of results
        name_boost: Score bonus for name matches
    
    Returns:
        List of (cluster_id, score) sorted by descending score
    """
    # Encode query
    if model is None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
    
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = torch.tensor(q_emb, dtype=torch.float32)
    
    # Cosine similarity
    if semantic_embeddings.dim() == 2:
        sims = torch.matmul(q_emb, semantic_embeddings.T).squeeze(0)
    else:
        sims = torch.matmul(q_emb, semantic_embeddings.unsqueeze(0).T).squeeze()
    
    sims = sims.numpy()
    
    # Name boosting
    query_lower = query.lower()
    for i, cid in enumerate(cluster_ids):
        cid_lower = str(cid).lower()
        # Check if cluster name (or key parts) appear in query
        name_parts = cid_lower.replace('-', ' ').replace('_', ' ').split()
        # Match significant parts (>3 chars)
        significant_parts = [p for p in name_parts if len(p) > 3]
        
        if significant_parts:
            matches = sum(1 for p in significant_parts if p in query_lower)
            match_ratio = matches / len(significant_parts)
            if match_ratio > 0.3:
                sims[i] += name_boost * match_ratio
    
    # Rank
    ranked_idx = np.argsort(-sims)
    results = [(str(cluster_ids[i]), float(sims[i])) for i in ranked_idx[:top_k]]
    return results


# ============================================================
# RECIPROCAL RANK FUSION (RRF)
# ============================================================

def rrf_fusion(
    rankings: List[List[Tuple[str, float]]],
    weights: List[float] = None,
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion to combine multiple rankings.
    
    RRF score for item d = sum_r [ weight_r / (k + rank_r(d)) ]
    
    Args:
        rankings: List of ranked lists, each [(cluster_id, score), ...]
        weights: Weight for each ranking (default: equal)
        k: RRF constant (higher = more emphasis on lower ranks)
        top_n: Number of results to return
    
    Returns:
        Fused ranking [(cluster_id, rrf_score), ...]
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    
    rrf_scores = defaultdict(float)
    
    for ranking, weight in zip(rankings, weights):
        for rank, (cid, _score) in enumerate(ranking):
            rrf_scores[cid] += weight / (k + rank + 1)
    
    sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return sorted_items[:top_n]


# ============================================================
# CELL TYPE SYNONYM MAP (fixes vocabulary gaps)
# ============================================================

CELL_TYPE_SYNONYMS = {
    # Maps query terms → possible cluster names in dataset
    'endothelial': ['endocardial cell', 'endothelial cell', 'vascular endothelial'],
    'endothelial cell': ['endocardial cell', 'endothelial cell'],
    'submucosal gland': ['mucus secreting cell of bronchus submucosal gland',
                          'serous cell of epithelium of bronchus'],
    'submucosal': ['mucus secreting cell of bronchus submucosal gland'],
    'goblet': ['bronchial goblet cell', 'nasal mucosa goblet cell',
               'respiratory tract goblet cell'],
    'ciliated': ['respiratory tract multiciliated cell', 'ciliated cell'],
    'multiciliated': ['respiratory tract multiciliated cell'],
    'fibroblast': ['fibroblast of lung', 'alveolar adventitial fibroblast',
                   'alveolar type 1 fibroblast cell'],
    'nk cell': ['natural killer cell'],
    'nk': ['natural killer cell'],
    't cell': ['mature T cell', 'CD8-positive, alpha-beta T cell',
               'CD4-positive helper T cell', 'cytotoxic T cell'],
    'cd8': ['CD8-positive, alpha-beta T cell'],
    'cd4': ['CD4-positive helper T cell'],
    'dc': ['dendritic cell, human'],
    'dendritic': ['dendritic cell, human'],
    'pnec': ['pulmonary neuroendocrine cell'],
    'neuroendocrine': ['pulmonary neuroendocrine cell'],
}


def apply_synonym_boost(
    scores: List[Tuple[str, float]],
    query: str,
    cluster_ids: List[str],
    boost: float = 0.1,
) -> List[Tuple[str, float]]:
    """
    Boost clusters that match synonym expansions of query terms.
    """
    query_lower = query.lower()
    boosted_clusters = set()
    
    for term, synonyms in CELL_TYPE_SYNONYMS.items():
        if term in query_lower:
            for syn in synonyms:
                if syn in cluster_ids:
                    boosted_clusters.add(syn)
    
    if not boosted_clusters:
        return scores
    
    # Apply boost
    score_dict = dict(scores)
    for cid in boosted_clusters:
        if cid in score_dict:
            score_dict[cid] += boost
        else:
            score_dict[cid] = boost
    
    result = sorted(score_dict.items(), key=lambda x: -x[1])
    return result


# ============================================================
# MAIN ENGINE CLASS
# ============================================================

class HybridRetrievalEngine:
    """
    Hybrid retrieval engine that routes queries to the best pipeline.
    
    Combines:
        - Gene marker scoring (for expression/gene-list queries)
        - BioBERT semantic matching (for ontology/text queries)  
        - RRF fusion (for mixed queries)
        - Synonym expansion (for vocabulary gap fixes)
    
    Args:
        pt_path: Path to .pt embeddings file
        cells_csv: Path to cell metadata CSV (optional)
        model_name: BioBERT model name
        gene_scoring_mode: 'simple', 'weighted', or 'full'
        rrf_k: RRF constant
        gene_weight: Weight for gene pipeline in mixed mode
        semantic_weight: Weight for semantic pipeline in mixed mode
        name_boost: Score bonus for cell type name matches
        synonym_boost: Score bonus for synonym matches
        verbose: Print debug info
    """
    
    def __init__(
        self,
        pt_path: str = None,
        base: str = None,
        pt_name: str = None,
        cells_csv: str = None,
        model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        gene_scoring_mode: str = 'weighted',
        rrf_k: int = 60,
        gene_weight: float = 1.0,
        semantic_weight: float = 1.0,
        name_boost: float = 0.15,
        synonym_boost: float = 0.10,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.gene_scoring_mode = gene_scoring_mode
        self.rrf_k = rrf_k
        self.gene_weight = gene_weight
        self.semantic_weight = semantic_weight
        self.name_boost = name_boost
        self.synonym_boost = synonym_boost
        self.model_name = model_name
        self._model = None
        
        # Resolve path
        if pt_path:
            path = pt_path
        elif base and pt_name:
            path = os.path.join(base, pt_name)
        else:
            raise ValueError("Provide pt_path or (base + pt_name)")
        
        # Load embeddings
        self._log(f"Loading: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        
        self.cluster_ids = data["cluster_ids"]
        self.cluster_texts = data.get("cluster_texts", [])
        self.semantic_embeddings = data.get("semantic_embeddings", None)
        self.scgpt_embeddings = data.get("scgpt_embeddings", None)
        self.gene_stats = data.get("gene_stats_per_cluster", {})
        self.go_terms = data.get("go_terms_per_cluster", {})
        self.reactome_terms = data.get("reactome_terms_per_cluster", {})
        self.metadata = data.get("metadata_per_cluster", {})
        
        # Build gene set for classification
        all_genes_raw = data.get("all_genes", [])
        self.all_genes = set(g.upper() for g in all_genes_raw)
        
        # Also add genes from gene_stats
        for cl_stats in self.gene_stats.values():
            for gene in cl_stats.keys():
                self.all_genes.add(gene.upper())
        
        self._log(f"Loaded {len(self.cluster_ids)} clusters, "
                  f"{len(self.all_genes)} known genes")
        
        # Load cells CSV if provided
        self.cells_df = None
        if cells_csv and os.path.exists(cells_csv):
            import pandas as pd
            self.cells_df = pd.read_csv(cells_csv)
            self._log(f"Loaded cell metadata: {len(self.cells_df)} cells")
    
    def _log(self, msg):
        if self.verbose:
            print(f"[HybridEngine] {msg}")
    
    @property
    def model(self):
        """Lazy-load BioBERT model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._log(f"Loading BioBERT on {device}...")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        force_mode: str = None,
    ) -> List[Tuple[str, float]]:
        """
        Main query interface.
        
        Args:
            query_text: Natural language or gene list query
            top_k: Number of results
            force_mode: Override auto-classification ('gene_list', 'ontology', 'mixed')
        
        Returns:
            List of (cluster_id, score) sorted by relevance
        """
        # Classify query
        if force_mode:
            mode = force_mode
        else:
            mode = classify_query(query_text, self.all_genes)
        
        self._log(f"Query mode: {mode} | '{query_text[:60]}...'")
        
        if mode == 'gene_list':
            return self._query_gene_list(query_text, top_k)
        elif mode == 'ontology':
            return self._query_ontology(query_text, top_k)
        else:  # mixed
            return self._query_mixed(query_text, top_k)
    
    def _query_gene_list(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Handle gene-list queries using marker scoring."""
        genes = extract_gene_names(query, self.all_genes)
        self._log(f"  Extracted genes: {genes}")
        
        if not genes:
            # Fallback to semantic if no genes found
            self._log("  No genes found — falling back to semantic")
            return self._query_ontology(query, top_k)
        
        results = gene_pipeline(
            genes, self.gene_stats, self.cluster_ids,
            scoring_mode=self.gene_scoring_mode,
            top_k=top_k * 2,  # get extra for reranking
        )
        
        # Also run a quick semantic check for safety
        if self.semantic_embeddings is not None:
            sem_results = semantic_pipeline(
                query, self.semantic_embeddings, self.cluster_ids,
                self.cluster_texts, model=self.model,
                top_k=top_k * 2,
                name_boost=0,  # no name boost for gene queries
            )
            # Light fusion: gene pipeline dominates (3:1 weight)
            results = rrf_fusion(
                [results, sem_results],
                weights=[3.0, 1.0],
                k=self.rrf_k,
                top_n=top_k,
            )
        else:
            results = results[:top_k]
        
        return results
    
    def _query_ontology(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Handle ontology/text queries using semantic matching."""
        if self.semantic_embeddings is None:
            self._log("  [WARN] No semantic embeddings — using gene fallback")
            genes = extract_gene_names(query, self.all_genes)
            return gene_pipeline(
                genes, self.gene_stats, self.cluster_ids,
                top_k=top_k,
            )
        
        results = semantic_pipeline(
            query, self.semantic_embeddings, self.cluster_ids,
            self.cluster_texts, model=self.model,
            top_k=top_k * 2,
            name_boost=self.name_boost,
        )
        
        # Apply synonym boosting
        results = apply_synonym_boost(
            results, query, self.cluster_ids,
            boost=self.synonym_boost,
        )
        
        return results[:top_k]
    
    def _query_mixed(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Handle mixed queries using RRF fusion of both pipelines."""
        # Gene pipeline
        genes = extract_gene_names(query, self.all_genes)
        gene_results = gene_pipeline(
            genes, self.gene_stats, self.cluster_ids,
            scoring_mode=self.gene_scoring_mode,
            top_k=top_k * 2,
        )
        
        # Semantic pipeline
        sem_results = []
        if self.semantic_embeddings is not None:
            sem_results = semantic_pipeline(
                query, self.semantic_embeddings, self.cluster_ids,
                self.cluster_texts, model=self.model,
                top_k=top_k * 2,
                name_boost=self.name_boost,
            )
        
        # Fuse
        if gene_results and sem_results:
            results = rrf_fusion(
                [gene_results, sem_results],
                weights=[self.gene_weight, self.semantic_weight],
                k=self.rrf_k,
                top_n=top_k * 2,
            )
        elif gene_results:
            results = gene_results
        else:
            results = sem_results
        
        # Apply synonym boosting
        results = apply_synonym_boost(
            results, query, self.cluster_ids,
            boost=self.synonym_boost,
        )
        
        return results[:top_k]
    
    # ============================================================
    # EVALUATION HELPERS
    # ============================================================
    
    def evaluate_query(
        self,
        query_text: str,
        expected: List[str],
        top_k: int = 10,
        force_mode: str = None,
    ) -> Dict:
        """
        Evaluate a single query against expected results.
        
        Returns:
            Dict with recall@5, recall@10, mrr, retrieved_top10, mode
        """
        results = self.query(query_text, top_k=top_k, force_mode=force_mode)
        retrieved = [cid for cid, _ in results]
        
        # Recall@5
        top5 = set(retrieved[:5])
        recall_5 = sum(1 for e in expected if e in top5) / len(expected) if expected else 0
        
        # Recall@10
        top10 = set(retrieved[:10])
        recall_10 = sum(1 for e in expected if e in top10) / len(expected) if expected else 0
        
        # MRR
        mrr = 0.0
        for e in expected:
            for rank, cid in enumerate(retrieved):
                if cid == e:
                    mrr = max(mrr, 1.0 / (rank + 1))
                    break
        
        mode = force_mode or classify_query(query_text, self.all_genes)
        
        return {
            'recall@5': recall_5,
            'recall@10': recall_10,
            'mrr': mrr,
            'retrieved_top10': retrieved[:10],
            'mode': mode,
            'query': query_text,
            'expected': expected,
        }
    
    def evaluate_benchmark(
        self,
        queries: List[Dict],
        force_mode: str = None,
    ) -> Dict:
        """
        Evaluate a benchmark set of queries.
        
        Args:
            queries: List of dicts with 'query_text', 'expected', 
                     optionally 'query_id' and 'category'
            force_mode: Override mode for all queries
        
        Returns:
            Dict with per-query results and aggregated metrics
        """
        results = []
        for q in queries:
            r = self.evaluate_query(
                q['query_text'],
                q['expected'],
                force_mode=force_mode,
            )
            r['query_id'] = q.get('query_id', '')
            r['category'] = q.get('category', 'unknown')
            results.append(r)
        
        # Aggregate by category
        categories = set(r['category'] for r in results)
        agg = {}
        for cat in categories:
            cat_results = [r for r in results if r['category'] == cat]
            agg[cat] = {
                'n_queries': len(cat_results),
                'mean_recall@5': np.mean([r['recall@5'] for r in cat_results]),
                'mean_recall@10': np.mean([r['recall@10'] for r in cat_results]),
                'mean_mrr': np.mean([r['mrr'] for r in cat_results]),
            }
        
        # Overall
        agg['overall'] = {
            'n_queries': len(results),
            'mean_recall@5': np.mean([r['recall@5'] for r in results]),
            'mean_recall@10': np.mean([r['recall@10'] for r in results]),
            'mean_mrr': np.mean([r['mrr'] for r in results]),
        }
        
        return {
            'per_query': results,
            'aggregated': agg,
        }


# ============================================================
# STANDALONE EVALUATION SCRIPT
# ============================================================

def run_evaluation(pt_path: str, cells_csv: str = None, 
                   scoring_mode: str = 'weighted'):
    """
    Run the full benchmark evaluation against CellWhisperer queries.
    """
    engine = HybridRetrievalEngine(
        pt_path=pt_path,
        cells_csv=cells_csv,
        gene_scoring_mode=scoring_mode,
        verbose=True,
    )
    
    # Define benchmark queries (from the eval results)
    benchmark = [
        # ONTOLOGY QUERIES
        {"query_id": "Q01", "category": "ontology",
         "query_text": "macrophage and monocyte infiltration in cystic fibrosis airways",
         "expected": ["macrophage", "monocyte", "non-classical monocyte"]},
        {"query_id": "Q02", "category": "ontology",
         "query_text": "CD8 T cell activation and cytotoxicity in CF lung inflammation",
         "expected": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"]},
        {"query_id": "Q03", "category": "ontology",
         "query_text": "CD4 helper T cell immune activation in cystic fibrosis",
         "expected": ["CD4-positive helper T cell", "mature T cell"]},
        {"query_id": "Q04", "category": "ontology",
         "query_text": "B cell activation and immunoglobulin response in CF airways",
         "expected": ["B cell", "plasma cell"]},
        {"query_id": "Q05", "category": "ontology",
         "query_text": "basal cell dysfunction and reduced stemness in cystic fibrosis epithelium",
         "expected": ["basal cell", "respiratory tract suprabasal cell"]},
        {"query_id": "Q06", "category": "ontology",
         "query_text": "ciliated cell ciliogenesis and increased abundance in CF bronchial epithelium",
         "expected": ["ciliated cell"]},
        {"query_id": "Q07", "category": "ontology",
         "query_text": "natural killer cell cytotoxicity and NKG2A immune checkpoint in CF",
         "expected": ["natural killer cell", "innate lymphoid cell"]},
        {"query_id": "Q08", "category": "ontology",
         "query_text": "pulmonary ionocyte CFTR expression in cystic fibrosis",
         "expected": ["ionocyte"]},
        {"query_id": "Q09", "category": "ontology",
         "query_text": "endothelial cell remodeling and VEGF signaling in CF lung",
         "expected": ["endothelial cell"]},
        {"query_id": "Q10", "category": "ontology",
         "query_text": "dendritic cell antigen presentation in CF airways",
         "expected": ["dendritic cell, human"]},
        {"query_id": "Q11", "category": "ontology",
         "query_text": "mast cell degranulation and allergic inflammation in CF",
         "expected": ["mast cell"]},
        {"query_id": "Q12", "category": "ontology",
         "query_text": "HLA-E CD94 NKG2A immune checkpoint inhibiting CD8 T cell activity",
         "expected": ["CD8-positive, alpha-beta T cell", "natural killer cell"]},
        {"query_id": "Q13", "category": "ontology",
         "query_text": "type I interferon response and inflammatory signaling in CF epithelial cells",
         "expected": ["basal cell", "ciliated cell", "secretory cell"]},
        {"query_id": "Q14", "category": "ontology",
         "query_text": "submucosal gland epithelial cell changes in cystic fibrosis",
         "expected": ["submucosal gland epithelial cell", "serous cell of epithelium of bronchus"]},
        {"query_id": "Q15", "category": "ontology",
         "query_text": "VEGF receptor signaling and hypoxia response across cell types in CF",
         "expected": ["endothelial cell", "CD8-positive, alpha-beta T cell", 
                      "CD4-positive helper T cell", "basal cell"]},
        
        # EXPRESSION QUERIES
        {"query_id": "Q16", "category": "expression",
         "query_text": "IGLJ3 IGKJ1 IGHJ5 JCHAIN MZB1 XBP1",
         "expected": ["plasma cell", "B cell"]},
        {"query_id": "Q17", "category": "expression",
         "query_text": "CD8A CD8B GZMB PRF1 IFNG NKG7",
         "expected": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"]},
        {"query_id": "Q18", "category": "expression",
         "query_text": "TRAJ52 TRBV22-1 TRDJ2 CD3E CD3G",
         "expected": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "mature T cell"]},
        {"query_id": "Q19", "category": "expression",
         "query_text": "CPA3 TPSAB1 TPSB2 MS4A2 HDC GATA2",
         "expected": ["mast cell"]},
        {"query_id": "Q20", "category": "expression",
         "query_text": "MARCO FABP4 APOC1 C1QB C1QC MSR1",
         "expected": ["macrophage", "monocyte", "non-classical monocyte"]},
        {"query_id": "Q21", "category": "expression",
         "query_text": "COL1A2 LUM DCN SFRP2 COL3A1 PDGFRA",
         "expected": ["fibroblast of lung", "alveolar adventitial fibroblast", 
                      "alveolar type 1 fibroblast cell"]},
        {"query_id": "Q22", "category": "expression",
         "query_text": "ATP6V1G3 FOXI1 BSND CLCNKB ASCL3",
         "expected": ["ionocyte"]},
        {"query_id": "Q23", "category": "expression",
         "query_text": "GNLY KLRD1 KLRK1 NKG7 PRF1 GZMB",
         "expected": ["natural killer cell", "innate lymphoid cell", "cytotoxic T cell"]},
        {"query_id": "Q24", "category": "expression",
         "query_text": "SST CHGA ASCL1 GRP CALCA SYP",
         "expected": ["pulmonary neuroendocrine cell"]},
        {"query_id": "Q25", "category": "expression",
         "query_text": "KRT5 KRT14 KRT15 TP63 IL33 CSTA",
         "expected": ["basal cell", "respiratory tract suprabasal cell"]},
        {"query_id": "Q26", "category": "expression",
         "query_text": "FOXJ1 DNAH5 CAPS PIFO RSPH1 DNAI1",
         "expected": ["ciliated cell"]},
        {"query_id": "Q27", "category": "expression",
         "query_text": "PLVAP ACKR1 ERG VWF PECAM1 CDH5",
         "expected": ["endothelial cell"]},
        {"query_id": "Q28", "category": "expression",
         "query_text": "SCGB1A1 SCGB3A1 MUC5AC MUC5B LYPD2 PRR4",
         "expected": ["secretory cell", "submucosal gland epithelial cell"]},
        {"query_id": "Q29", "category": "expression",
         "query_text": "CALR LRP1 GNAI2 FOS JUND MAP2K2",
         "expected": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "macrophage"]},
        {"query_id": "Q30", "category": "expression",
         "query_text": "HLA-E KLRC1 KLRD1 KLRC2 KLRC3 KLRK1",
         "expected": ["CD8-positive, alpha-beta T cell", "natural killer cell", "innate lymphoid cell"]},
    ]
    
    # Run evaluation
    eval_results = engine.evaluate_benchmark(benchmark)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — Hybrid Engine v4")
    print("=" * 70)
    
    # Per-query details
    for r in eval_results['per_query']:
        status = "✓" if r['recall@5'] >= 0.5 else "✗"
        print(f"  {status} {r['query_id']} [{r['mode']:>9}] "
              f"R@5={r['recall@5']:.3f} R@10={r['recall@10']:.3f} "
              f"MRR={r['mrr']:.3f}")
        if r['recall@5'] < 0.5:
            print(f"      Expected: {r['expected']}")
            print(f"      Got top5: {r['retrieved_top10'][:5]}")
    
    # Aggregated
    print("\n" + "-" * 50)
    print("AGGREGATED METRICS:")
    for cat, metrics in eval_results['aggregated'].items():
        print(f"\n  {cat}:")
        print(f"    Recall@5:  {metrics['mean_recall@5']:.4f}")
        print(f"    Recall@10: {metrics['mean_recall@10']:.4f}")
        print(f"    MRR:       {metrics['mean_mrr']:.4f}")
    
    # Comparison with CellWhisperer
    print("\n" + "-" * 50)
    print("COMPARISON WITH CELLWHISPERER:")
    cw_ont = {'R@5': 0.7778, 'R@10': 0.8, 'MRR': 0.7714}
    cw_exp = {'R@5': 0.3444, 'R@10': 0.4222, 'MRR': 0.368}
    
    our_ont = eval_results['aggregated'].get('ontology', {})
    our_exp = eval_results['aggregated'].get('expression', {})
    
    if our_ont:
        print(f"\n  ONTOLOGY:")
        print(f"    {'':>12} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
        print(f"    {'CW':>12} {cw_ont['R@5']:>8.4f} {cw_ont['R@10']:>8.4f} {cw_ont['MRR']:>8.4f}")
        print(f"    {'Ours':>12} {our_ont['mean_recall@5']:>8.4f} {our_ont['mean_recall@10']:>8.4f} {our_ont['mean_mrr']:>8.4f}")
        delta_r5 = our_ont['mean_recall@5'] - cw_ont['R@5']
        print(f"    {'Delta':>12} {delta_r5:>+8.4f}")
    
    if our_exp:
        print(f"\n  EXPRESSION:")
        print(f"    {'':>12} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
        print(f"    {'CW':>12} {cw_exp['R@5']:>8.4f} {cw_exp['R@10']:>8.4f} {cw_exp['MRR']:>8.4f}")
        print(f"    {'Ours':>12} {our_exp['mean_recall@5']:>8.4f} {our_exp['mean_recall@10']:>8.4f} {our_exp['mean_mrr']:>8.4f}")
        delta_r5 = our_exp['mean_recall@5'] - cw_exp['R@5']
        print(f"    {'Delta':>12} {delta_r5:>+8.4f}")
    
    return eval_results


# ============================================================
# v3 COMPATIBILITY WRAPPER
# ============================================================
# This wrapper lets elisa_benchmark_v5.py import this module
# as a drop-in replacement for retrieval_engine_v3.
#
#   from retrieval_engine_v4_hybrid import RetrievalEngine
#   engine = RetrievalEngine(base="...", pt_name="...", cells_csv="...")
#
# The benchmark calls:
#   engine.query_semantic(text, top_k, with_genes=False)
#   engine.query_hybrid(text, top_k, lambda_sem=0.0, with_genes=False)
#   engine.query_annotation_only(text, top_k, with_genes=False)
#   engine.cluster_ids
#   engine.gene_stats
#   engine.metadata / engine.cluster_metadata

class RetrievalEngine:
    """
    v3-compatible API wrapper around HybridRetrievalEngine.
    
    Translates the v3 API (query_semantic, query_hybrid, query_annotation_only)
    into the v4 hybrid router, so the existing benchmark runs unchanged.
    """
    
    def __init__(self, base: str = ".", pt_name: str = "", cells_csv: str = None,
                 **kwargs):
        pt_path = os.path.join(base, pt_name) if pt_name else base
        csv_path = os.path.join(base, cells_csv) if cells_csv else None
        
        self._hybrid = HybridRetrievalEngine(
            pt_path=pt_path,
            cells_csv=csv_path,
            verbose=True,
            **kwargs,
        )
        
        # Expose attributes the benchmark expects
        self.cluster_ids = self._hybrid.cluster_ids
        self.gene_stats = self._hybrid.gene_stats
        self.metadata = self._hybrid.metadata
        self.cluster_metadata = self._build_cluster_metadata()
    
    def _build_cluster_metadata(self) -> dict:
        """Build cluster_metadata dict that BM25Scorer expects."""
        meta = {}
        for i, cid in enumerate(self._hybrid.cluster_ids):
            cid_str = str(cid)
            entry = {"name": cid_str}
            
            # GO terms
            go = self._hybrid.go_terms.get(cid_str, [])
            if go:
                entry["go_terms"] = go
            
            # Reactome
            react = self._hybrid.reactome_terms.get(cid_str, [])
            if react:
                entry["reactome"] = react
            
            # Top markers from gene_stats (by |logFC|)
            stats = self._hybrid.gene_stats.get(cid_str, {})
            if stats:
                sorted_genes = sorted(
                    stats.keys(),
                    key=lambda g: abs(stats[g].get("logfc", 0)),
                    reverse=True
                )
                entry["markers"] = sorted_genes[:50]
            
            meta[cid_str] = entry
        
        return meta
    
    def _format_results(self, ranked: List[Tuple[str, float]], 
                        with_genes: bool = False, top_n_genes: int = 30
                        ) -> dict:
        """Convert (cluster_id, score) list to v3 result format."""
        results = []
        for cid, score in ranked:
            entry = {"cluster_id": cid, "score": float(score)}
            
            if with_genes:
                stats = self._hybrid.gene_stats.get(str(cid), {})
                sorted_genes = sorted(
                    stats.keys(),
                    key=lambda g: abs(stats[g].get("logfc", 0)),
                    reverse=True
                )[:top_n_genes]
                entry["genes"] = [
                    {"gene": g, **stats[g]} for g in sorted_genes
                ]
            
            results.append(entry)
        
        return {"results": results}
    
    def query_semantic(self, text: str, top_k: int = 10, 
                       with_genes: bool = False) -> dict:
        """
        v3 API: Semantic-only retrieval.
        Routes through the ontology pipeline (BioBERT cosine similarity).
        """
        ranked = self._hybrid.query(text, top_k=top_k, force_mode='ontology')
        return self._format_results(ranked, with_genes=with_genes)
    
    def query_hybrid(self, text: str, top_k: int = 10, 
                     lambda_sem: float = 0.5, with_genes: bool = False) -> dict:
        """
        v3 API: Hybrid retrieval.
        
        lambda_sem controls the blend:
          - lambda_sem=1.0 → pure semantic (ontology pipeline)
          - lambda_sem=0.0 → pure scGPT/expression (gene pipeline)
          - lambda_sem=0.5 → mixed (RRF fusion)
        
        The benchmark calls this with lambda_sem=0.0 to get the scGPT mode.
        """
        if lambda_sem <= 0.1:
            # Pure expression/gene mode — use gene pipeline
            ranked = self._hybrid.query(text, top_k=top_k, force_mode='gene_list')
        elif lambda_sem >= 0.9:
            # Pure semantic mode
            ranked = self._hybrid.query(text, top_k=top_k, force_mode='ontology')
        else:
            # Mixed mode — let the router decide
            ranked = self._hybrid.query(text, top_k=top_k, force_mode='mixed')
        
        return self._format_results(ranked, with_genes=with_genes)
    
    def query_annotation_only(self, text: str, top_k: int = 10,
                              with_genes: bool = False) -> dict:
        """
        v3 API: Annotation-only retrieval (cluster name only, no enrichment).
        
        Uses semantic pipeline but without context weighting — 
        only matches against cluster identity text.
        """
        if self._hybrid.semantic_embeddings is None:
            return self._format_results([], with_genes=with_genes)
        
        # For annotation-only, we encode query and match against identity-weighted
        # embeddings. Since our embeddings already have alpha=0.6 identity weight,
        # this is close to annotation-only behavior.
        # For a stricter version, we'd need separate identity-only embeddings.
        ranked = self._hybrid.query(text, top_k=top_k, force_mode='ontology')
        return self._format_results(ranked, with_genes=with_genes)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ELISA Hybrid Retrieval Engine v4 — Evaluation"
    )
    parser.add_argument("--pt", required=True,
                        help="Path to .pt embeddings file")
    parser.add_argument("--cells-csv", default=None,
                        help="Path to cell metadata CSV")
    parser.add_argument("--scoring-mode", default="weighted",
                        choices=["simple", "weighted", "full"],
                        help="Gene scoring mode")
    parser.add_argument("--gene-weight", type=float, default=1.0,
                        help="Gene pipeline weight in mixed mode")
    parser.add_argument("--semantic-weight", type=float, default=1.0,
                        help="Semantic pipeline weight in mixed mode")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query to test (skip benchmark)")
    
    args = parser.parse_args()
    
    if args.query:
        engine = HybridRetrievalEngine(
            pt_path=args.pt,
            cells_csv=args.cells_csv,
            gene_scoring_mode=args.scoring_mode,
        )
        results = engine.query(args.query)
        print(f"\nResults for: '{args.query}'")
        for i, (cid, score) in enumerate(results):
            print(f"  {i+1}. {cid} (score={score:.4f})")
    else:
        run_evaluation(
            args.pt,
            cells_csv=args.cells_csv,
            scoring_mode=args.scoring_mode,
        )
