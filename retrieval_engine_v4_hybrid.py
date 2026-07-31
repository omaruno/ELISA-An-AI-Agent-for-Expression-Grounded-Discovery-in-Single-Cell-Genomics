#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
elisa_engine_compat.py
======================
Compatibility layer between `retrieval_engine_v4_hybrid.RetrievalEngine`
(a thin v3-benchmark shim) and `elisa_chat_v4.py`, which expects a much
larger engine surface.

Usage — change ONE line in elisa_chat_v4.py:

    # from retrieval_engine_v4_hybrid import RetrievalEngine
    from elisa_engine_compat import RetrievalEngine

Optionally, right after the h5ad is loaded in main(), add:

    if adata is not None:
        engine.attach_adata(adata, cluster_key=cluster_key)

That unlocks proportions/compare using real per-cell metadata instead of
whatever happens to be in metadata_per_cluster.

IMPORTANT CAVEATS
-----------------
* `gene_stats` holds per-cluster DE/marker statistics, not a full expression
  matrix. Anything derived from it (pathway scores, interactions) is a
  *marker-enrichment proxy*, not a measurement. Treated as such throughout.
* `interactions()` uses a small built-in ligand-receptor table. It predicts
  co-enrichment of a ligand in one cluster and its receptor in another.
  It is not CellPhoneDB / CellChat and makes no permutation-based claims.
* `compare()` requires per-cell condition labels (attached h5ad or cells_csv).
  Without them it returns {"error": ...} rather than fabricating a result.
"""

import os
import math
from typing import List, Dict, Optional, Any

import numpy as np

from retrieval_engine_v4_hybrid import (
    RetrievalEngine as _V3Wrapper,
    classify_query,
    extract_gene_names,
    gene_pipeline,
    semantic_pipeline,
)


# ============================================================
# SMALL UTILITIES
# ============================================================

def _pct(v) -> float:
    """Normalize a percentage that may be stored as 0-1 or 0-100."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if v > 1.5:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


CONDITION_KEY_CANDIDATES = [
    "condition", "disease", "disease_state", "disease__ontology_label",
    "status", "group", "treatment", "genotype", "phenotype", "diagnosis",
    "sample_type", "stimulation", "timepoint", "age_group", "development_stage",
]

# Columns that look like conditions but are really identifiers
CONDITION_KEY_BLOCKLIST = {
    "donor_id", "sample_id", "cell_id", "barcode", "batch", "library_id",
}


# ============================================================
# BUILT-IN PATHWAY GENE SETS
# ============================================================

PATHWAYS: Dict[str, List[str]] = {
    "type_i_interferon": [
        "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "MX2", "OAS1", "OAS2",
        "OAS3", "IFI6", "IFI27", "IFI44L", "STAT1", "STAT2", "IRF7", "IRF9",
        "RSAD2", "BST2", "XAF1",
    ],
    "antigen_presentation_mhc_i": [
        "HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "B2M", "TAP1", "TAP2",
        "TAPBP", "PSMB8", "PSMB9", "NLRC5", "CALR",
    ],
    "antigen_presentation_mhc_ii": [
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
        "HLA-DMA", "HLA-DMB", "CD74", "CIITA",
    ],
    "cytotoxicity": [
        "GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "GNLY", "NKG7", "KLRD1",
        "KLRK1", "FASLG", "IFNG", "CTSW",
    ],
    "inflammatory_cytokines": [
        "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4", "CXCL1",
        "CXCL2", "CXCL10", "NFKB1", "NFKBIA", "PTGS2", "SOD2",
    ],
    "hypoxia": [
        "HIF1A", "VEGFA", "SLC2A1", "LDHA", "PGK1", "ALDOA", "ENO1", "CA9",
        "ADM", "BNIP3", "NDRG1", "P4HA1",
    ],
    "angiogenesis_vegf": [
        "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "FLT4", "PGF", "NRP1",
        "NRP2", "ANGPT1", "ANGPT2", "TEK", "PECAM1", "CDH5", "ESM1",
    ],
    "epithelial_mesenchymal_transition": [
        "VIM", "FN1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2",
        "SPARC", "TAGLN", "ACTA2", "COL1A1", "COL1A2", "TGFB1",
    ],
    "cell_cycle": [
        "MKI67", "TOP2A", "CCNB1", "CCNA2", "CDK1", "PCNA", "TYMS", "BIRC5",
        "AURKB", "UBE2C", "RRM2", "TUBB4B", "STMN1", "HMGB2",
    ],
    "apoptosis": [
        "BAX", "BAK1", "BID", "CASP3", "CASP7", "CASP8", "CASP9", "BCL2",
        "BCL2L1", "MCL1", "TP53", "CDKN1A", "PMAIP1", "BBC3",
    ],
    "oxidative_phosphorylation": [
        "NDUFA4", "NDUFB2", "COX4I1", "COX5A", "COX6C", "COX7C", "ATP5F1A",
        "ATP5F1B", "ATP5MC2", "UQCRB", "UQCRQ", "SDHB", "CYC1",
    ],
    "wnt_signaling": [
        "WNT2", "WNT2B", "WNT5A", "WNT7B", "WNT11", "FZD1", "FZD2", "FZD4",
        "LRP5", "LRP6", "CTNNB1", "AXIN2", "LEF1", "TCF7L2", "RSPO2", "SFRP1",
    ],
    "notch_signaling": [
        "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1",
        "DLL3", "DLL4", "HES1", "HEY1", "HEYL", "RBPJ",
    ],
    "tgf_beta_bmp": [
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
        "SMAD4", "BMP2", "BMP4", "BMPR1A", "BMPR2", "ID1", "ID2", "ID3",
    ],
    "alveolar_surfactant": [
        "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "SFTPD", "NAPSA", "LAMP3",
        "PGC", "SLC34A2", "ABCA3", "NKX2-1", "AGER", "PDPN", "HOPX",
    ],
    "ciliogenesis": [
        "FOXJ1", "DNAH5", "DNAI1", "DNAI2", "RSPH1", "RSPH4A", "PIFO",
        "CAPS", "TPPP3", "SPAG6", "TEKT1", "CFAP43", "IFT81", "TUBA1A",
    ],
    "mucin_secretion": [
        "MUC5AC", "MUC5B", "MUC1", "MUC16", "SCGB1A1", "SCGB3A1", "SCGB3A2",
        "BPIFA1", "BPIFB1", "TFF3", "AGR2", "SPDEF", "CREB3L1",
    ],
    "extracellular_matrix": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL6A2", "FN1", "LUM",
        "DCN", "ELN", "FBN1", "MMP2", "MMP9", "TIMP1", "SPARC", "POSTN",
    ],
}


# ============================================================
# BUILT-IN LIGAND-RECEPTOR TABLE
# ============================================================
# (ligand, receptor, pathway)
LR_PAIRS = [
    ("WNT2", "FZD1", "WNT"), ("WNT5A", "FZD4", "WNT"),
    ("WNT7B", "FZD1", "WNT"), ("RSPO2", "LGR5", "WNT"),
    ("FGF7", "FGFR2", "FGF"), ("FGF10", "FGFR2", "FGF"),
    ("FGF9", "FGFR3", "FGF"), ("FGF2", "FGFR1", "FGF"),
    ("TGFB1", "TGFBR2", "TGFB"), ("TGFB2", "TGFBR2", "TGFB"),
    ("TGFB1", "TGFBR1", "TGFB"), ("BMP4", "BMPR1A", "BMP"),
    ("BMP2", "BMPR2", "BMP"), ("BMP5", "BMPR1B", "BMP"),
    ("JAG1", "NOTCH1", "NOTCH"), ("JAG2", "NOTCH2", "NOTCH"),
    ("DLL1", "NOTCH1", "NOTCH"), ("DLL4", "NOTCH4", "NOTCH"),
    ("VEGFA", "KDR", "VEGF"), ("VEGFA", "FLT1", "VEGF"),
    ("VEGFB", "FLT1", "VEGF"), ("PGF", "FLT1", "VEGF"),
    ("VEGFC", "FLT4", "VEGF"), ("ANGPT1", "TEK", "ANGIOPOIETIN"),
    ("ANGPT2", "TEK", "ANGIOPOIETIN"),
    ("PDGFA", "PDGFRA", "PDGF"), ("PDGFB", "PDGFRB", "PDGF"),
    ("SHH", "PTCH1", "HEDGEHOG"), ("IHH", "PTCH1", "HEDGEHOG"),
    ("IGF1", "IGF1R", "IGF"), ("IGF2", "IGF2R", "IGF"),
    ("HGF", "MET", "HGF"), ("NRG1", "ERBB3", "ERBB"),
    ("EREG", "EGFR", "EGF"), ("HBEGF", "EGFR", "EGF"),
    ("TGFA", "EGFR", "EGF"), ("AREG", "EGFR", "EGF"),
    ("CXCL12", "CXCR4", "CHEMOKINE"), ("CCL2", "CCR2", "CHEMOKINE"),
    ("CXCL8", "CXCR1", "CHEMOKINE"), ("CXCL8", "CXCR2", "CHEMOKINE"),
    ("CCL5", "CCR5", "CHEMOKINE"), ("CXCL9", "CXCR3", "CHEMOKINE"),
    ("CXCL10", "CXCR3", "CHEMOKINE"), ("CCL19", "CCR7", "CHEMOKINE"),
    ("IL6", "IL6R", "CYTOKINE"), ("IL1B", "IL1R1", "CYTOKINE"),
    ("TNF", "TNFRSF1A", "CYTOKINE"), ("IFNG", "IFNGR1", "CYTOKINE"),
    ("IL10", "IL10RA", "CYTOKINE"), ("IL33", "IL1RL1", "CYTOKINE"),
    ("CSF1", "CSF1R", "CYTOKINE"), ("TSLP", "CRLF2", "CYTOKINE"),
    ("ICAM1", "ITGAL", "ADHESION"), ("VCAM1", "ITGA4", "ADHESION"),
    ("COL4A1", "ITGB1", "ADHESION"), ("LAMA3", "ITGB4", "ADHESION"),
    ("FN1", "ITGA5", "ADHESION"), ("SPP1", "CD44", "ADHESION"),
    ("HLA-E", "KLRC1", "IMMUNE_CHECKPOINT"),
    ("CD274", "PDCD1", "IMMUNE_CHECKPOINT"),
    ("HLA-DRA", "CD4", "ANTIGEN_PRESENTATION"),
    ("EDN1", "EDNRA", "ENDOTHELIN"), ("SEMA3A", "NRP1", "SEMAPHORIN"),
    ("PTN", "PTPRZ1", "PLEIOTROPHIN"), ("MDK", "LRP1", "MIDKINE"),
    ("GRN", "SORT1", "GROWTH_FACTOR"), ("APOE", "LRP1", "LIPID"),
]


# ============================================================
# ENGINE
# ============================================================

class RetrievalEngine(_V3Wrapper):
    """
    Drop-in replacement for retrieval_engine_v4_hybrid.RetrievalEngine that
    implements the full surface elisa_chat_v4.py calls.
    """

    def __init__(self, base: str = ".", pt_name: str = "",
                 cells_csv: str = None, **kwargs):
        super().__init__(base=base, pt_name=pt_name, cells_csv=cells_csv,
                         **kwargs)

        self.cluster_texts = getattr(self._hybrid, "cluster_texts", [])
        self.go_terms = getattr(self._hybrid, "go_terms", {})
        self.reactome_terms = getattr(self._hybrid, "reactome_terms", {})
        self.all_genes = getattr(self._hybrid, "all_genes", set())

        # Diagnostics consumed by viz (plot:scatter)
        self._last_sem_sims = None
        self._last_expr_sims = None

        # Optional per-cell backing
        self._adata = None
        self._adata_cluster_key = None
        self._caps_cache = None

    # --------------------------------------------------------
    # BASIC ATTRIBUTES
    # --------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.cluster_ids)

    @property
    def semantic_emb(self):
        emb = getattr(self._hybrid, "semantic_embeddings", None)
        return self._as_numpy(emb)

    @property
    def scgpt_emb(self):
        emb = getattr(self._hybrid, "scgpt_embeddings", None)
        if emb is None:
            return self.semantic_emb
        return self._as_numpy(emb)

    @staticmethod
    def _as_numpy(emb):
        if emb is None:
            return None
        if hasattr(emb, "detach"):
            return emb.detach().cpu().numpy()
        return np.asarray(emb)

    def _cid(self, cid) -> str:
        """Resolve a user-supplied cluster name to the canonical id."""
        cid = str(cid).strip()
        for c in self.cluster_ids:
            if str(c) == cid:
                return str(c)
        low = cid.lower()
        for c in self.cluster_ids:
            if str(c).lower() == low:
                return str(c)
        for c in self.cluster_ids:
            if low in str(c).lower():
                return str(c)
        return cid

    def get_metadata(self, cid: str) -> dict:
        key = self._cid(cid)
        meta = dict(self.metadata.get(key, {}))
        meta.setdefault("cluster_id", key)
        if key in self.go_terms:
            meta["go_terms"] = self.go_terms[key][:25]
        if key in self.reactome_terms:
            meta["reactome_terms"] = self.reactome_terms[key][:25]
        stats = self.gene_stats.get(key, {})
        if stats:
            top = sorted(stats.keys(),
                         key=lambda g: abs(_f(stats[g].get("logfc"))),
                         reverse=True)[:25]
            meta["top_markers"] = top
            meta["n_de_genes"] = len(stats)
        if not meta.get("n_cells"):
            counts = self._cluster_counts()
            if key in counts:
                meta["n_cells"] = int(counts[key])
        return meta

    def get_cells(self, cid: str) -> List[str]:
        key = self._cid(cid)

        if self._adata is not None and self._adata_cluster_key:
            obs = self._adata.obs
            mask = obs[self._adata_cluster_key].astype(str) == key
            return list(obs.index[mask].astype(str))

        df = getattr(self._hybrid, "cells_df", None)
        if df is not None:
            for col in ("cell_type", "cluster", "cluster_id", "label"):
                if col in df.columns:
                    sub = df[df[col].astype(str) == key]
                    for idcol in ("cell_id", "barcode", "index"):
                        if idcol in sub.columns:
                            return list(sub[idcol].astype(str))
                    return list(sub.index.astype(str))

        meta = self.metadata.get(key, {})
        for k in ("cells", "cell_ids", "barcodes"):
            if k in meta:
                return list(meta[k])
        return []

    # --------------------------------------------------------
    # PER-CELL BACKING (optional)
    # --------------------------------------------------------

    def attach_adata(self, adata, cluster_key: str = "cell_type"):
        """Attach a loaded AnnData so proportions/compare use real cell data."""
        if adata is None:
            return
        if cluster_key not in adata.obs.columns:
            print(f"[Compat] cluster_key '{cluster_key}' not in adata.obs — "
                  f"available: {list(adata.obs.columns)[:15]}")
            return
        self._adata = adata
        self._adata_cluster_key = cluster_key
        self._caps_cache = None
        print(f"[Compat] Attached h5ad: {adata.shape[0]} cells "
              f"on obs['{cluster_key}']")

    def _cluster_counts(self) -> Dict[str, int]:
        """Cells per cluster, from the best source available."""
        if self._adata is not None:
            vc = self._adata.obs[self._adata_cluster_key].astype(str).value_counts()
            return {str(k): int(v) for k, v in vc.items()}

        df = getattr(self._hybrid, "cells_df", None)
        if df is not None:
            for col in ("cell_type", "cluster", "cluster_id", "label"):
                if col in df.columns:
                    vc = df[col].astype(str).value_counts()
                    return {str(k): int(v) for k, v in vc.items()}

        counts = {}
        for cid in self.cluster_ids:
            meta = self.metadata.get(str(cid), {})
            for k in ("n_cells", "num_cells", "cell_count", "size"):
                if k in meta:
                    counts[str(cid)] = int(_f(meta[k]))
                    break
        return counts

    def _obs(self):
        if self._adata is not None:
            return self._adata.obs
        return getattr(self._hybrid, "cells_df", None)

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    def detect_capabilities(self) -> dict:
        if self._caps_cache is not None:
            return self._caps_cache

        caps = {
            "n_clusters": self.n,
            "n_genes": len(self.all_genes),
            "has_semantic_embeddings": self.semantic_emb is not None,
            "has_scgpt_embeddings":
                getattr(self._hybrid, "scgpt_embeddings", None) is not None,
            "has_gene_stats": bool(self.gene_stats),
            "has_go_terms": bool(self.go_terms),
            "has_reactome_terms": bool(self.reactome_terms),
            "has_cell_level_data": self._adata is not None
                                   or getattr(self._hybrid, "cells_df", None) is not None,
            "has_conditions": False,
            "condition_column": None,
            "condition_values": [],
            "condition_source": None,
            "available_pathways": sorted(PATHWAYS.keys()),
        }

        obs = self._obs()
        if obs is not None:
            best = None
            for col in obs.columns:
                if col.lower() in CONDITION_KEY_BLOCKLIST:
                    continue
                if col.lower() not in CONDITION_KEY_CANDIDATES:
                    continue
                try:
                    vals = obs[col].astype(str).unique().tolist()
                except Exception:
                    continue
                if 2 <= len(vals) <= 12:
                    best = (col, sorted(vals))
                    break
            if best:
                caps["has_conditions"] = True
                caps["condition_column"] = best[0]
                caps["condition_values"] = best[1]
                caps["condition_source"] = ("h5ad" if self._adata is not None
                                            else "cells_csv")

        # Fall back to cluster-level metadata
        if not caps["has_conditions"] and self.metadata:
            keysets = {}
            for cid in self.cluster_ids:
                for k, v in (self.metadata.get(str(cid), {}) or {}).items():
                    if k.lower() in CONDITION_KEY_CANDIDATES and isinstance(v, (str, int, float)):
                        keysets.setdefault(k, set()).add(str(v))
            for k, vals in keysets.items():
                if 2 <= len(vals) <= 12:
                    caps["has_conditions"] = True
                    caps["condition_column"] = k
                    caps["condition_values"] = sorted(vals)
                    caps["condition_source"] = "cluster_metadata"
                    break

        counts = self._cluster_counts()
        caps["total_cells"] = int(sum(counts.values())) if counts else None
        caps["has_proportions"] = bool(counts)

        self._caps_cache = caps
        return caps

    # --------------------------------------------------------
    # RESULT FORMATTING
    # --------------------------------------------------------

    def _gene_evidence(self, cid: str, top_n: int = 25) -> List[dict]:
        stats = self.gene_stats.get(str(cid), {})
        if not stats:
            return []
        ordered = sorted(stats.keys(),
                         key=lambda g: abs(_f(stats[g].get("logfc"))),
                         reverse=True)[:top_n]
        out = []
        for g in ordered:
            s = stats[g]
            out.append({
                "gene": g,
                "logfc": round(_f(s.get("logfc")), 4),
                "pct_in": round(_pct(s.get("pct_in")), 4),
                "pct_out": round(_pct(s.get("pct_out")), 4),
                "pval_adj": _f(s.get("pval_adj"), 1.0),
            })
        return out

    def _format_results(self, ranked, with_genes: bool = False,
                        top_n_genes: int = 25, mode: str = "semantic",
                        query: str = "") -> dict:
        sem_map = dict(zip([str(c) for c in self.cluster_ids],
                           self._last_sem_sims)) if self._last_sem_sims is not None else {}
        expr_map = dict(zip([str(c) for c in self.cluster_ids],
                            self._last_expr_sims)) if self._last_expr_sims is not None else {}

        results = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            cid = str(cid)
            entry = {
                "rank": rank,
                "cluster_id": cid,
                "name": cid,
                "score": float(score),
                "semantic_similarity": float(sem_map.get(cid, score)),
                "expr_similarity": float(expr_map.get(cid, 0.0)),
                "hybrid_similarity": float(score),
            }
            if with_genes:
                ev = self._gene_evidence(cid, top_n_genes)
                entry["gene_evidence"] = ev          # chat + viz key
                entry["genes"] = ev                  # v3 benchmark key
                entry["n_de_genes"] = len(self.gene_stats.get(cid, {}))
            results.append(entry)

        return {
            "query": query,
            "mode": mode,
            "n_results": len(results),
            "results": results,
        }

    # --------------------------------------------------------
    # SIMILARITY DIAGNOSTICS
    # --------------------------------------------------------

    def _compute_all_sem_sims(self, text: str):
        emb = getattr(self._hybrid, "semantic_embeddings", None)
        if emb is None:
            return None
        try:
            import torch
            q = self._hybrid.model.encode([text], normalize_embeddings=True)
            q = torch.tensor(np.asarray(q), dtype=torch.float32)
            sims = torch.matmul(q, emb.float().T).squeeze(0).cpu().numpy()
            return np.asarray(sims, dtype=float)
        except Exception as e:
            print(f"[Compat] semantic sim computation failed: {e}")
            return None

    def _compute_all_expr_scores(self, text: str):
        genes = extract_gene_names(text, self.all_genes)
        if not genes:
            return None
        ranked = gene_pipeline(genes, self.gene_stats,
                               [str(c) for c in self.cluster_ids],
                               scoring_mode=self._hybrid.gene_scoring_mode,
                               top_k=len(self.cluster_ids))
        d = {cid: s for cid, s in ranked}
        arr = np.array([d.get(str(c), 0.0) for c in self.cluster_ids], dtype=float)
        mx = arr.max()
        if mx > 0:
            arr = arr / mx
        return arr

    def _refresh_diagnostics(self, text: str):
        self._last_sem_sims = self._compute_all_sem_sims(text)
        self._last_expr_sims = self._compute_all_expr_scores(text)

    # --------------------------------------------------------
    # RETRIEVAL (v3 API, extended)
    # --------------------------------------------------------

    def query_semantic(self, text: str, top_k: int = 10,
                       with_genes: bool = False, **_ignored) -> dict:
        self._refresh_diagnostics(text)
        ranked = self._hybrid.query(text, top_k=top_k, force_mode="ontology")
        return self._format_results(ranked, with_genes=with_genes,
                                    mode="semantic", query=text)

    def query_hybrid(self, text: str, top_k: int = 10, lambda_sem: float = 0.5,
                     with_genes: bool = False, pre_k: int = None,
                     gamma: float = None, **_ignored) -> dict:
        """
        pre_k / gamma are accepted for v3 call-signature compatibility.
        The v4 router does not use them; they are recorded in the payload
        so reports remain honest about what was actually run.
        """
        self._refresh_diagnostics(text)
        if lambda_sem <= 0.1:
            mode = "gene_list"
        elif lambda_sem >= 0.9:
            mode = "ontology"
        else:
            mode = "mixed"
        ranked = self._hybrid.query(text, top_k=top_k, force_mode=mode)
        payload = self._format_results(ranked, with_genes=with_genes,
                                       mode="hybrid", query=text)
        payload["router_mode"] = mode
        payload["lambda_sem"] = lambda_sem
        if pre_k is not None or gamma is not None:
            payload["ignored_params"] = {"pre_k": pre_k, "gamma": gamma}
        return payload

    def query_annotation_only(self, text: str, top_k: int = 10,
                              with_genes: bool = False, **_ignored) -> dict:
        self._refresh_diagnostics(text)
        ranked = self._hybrid.query(text, top_k=top_k, force_mode="ontology")
        return self._format_results(ranked, with_genes=with_genes,
                                    mode="annotation", query=text)

    def discover(self, text: str, top_k: int = 5, lambda_sem: float = 0.5,
                 pre_k: int = None, gamma: float = None, **_ignored) -> dict:
        """Discovery mode: mixed routing plus enrichment context."""
        self._refresh_diagnostics(text)
        auto_mode = classify_query(text, self.all_genes)
        force = "mixed" if auto_mode == "mixed" else auto_mode
        ranked = self._hybrid.query(text, top_k=top_k, force_mode=force)
        payload = self._format_results(ranked, with_genes=True,
                                       mode="discovery", query=text)
        payload["classified_as"] = auto_mode
        payload["query_genes"] = extract_gene_names(text, self.all_genes)

        for entry in payload["results"]:
            cid = entry["cluster_id"]
            go = self.go_terms.get(cid, [])
            react = self.reactome_terms.get(cid, [])
            if go:
                entry["go_terms"] = go[:10]
            if react:
                entry["reactome_terms"] = react[:10]
        return payload

    def lambda_sweep(self, query: str, top_k: int = 5) -> dict:
        """
        Sweep the semantic/expression blend and report, at each lambda,
        the fraction of query genes covered by the returned clusters'
        marker sets. Falls back to rank-overlap when the query has no genes.
        """
        lambdas = [round(x * 0.1, 1) for x in range(11)]
        query_genes = [g.upper() for g in extract_gene_names(query, self.all_genes)]
        coverages = []
        baseline = None

        for lam in lambdas:
            payload = self.query_hybrid(query, top_k=top_k, lambda_sem=lam,
                                        with_genes=False)
            cids = [r["cluster_id"] for r in payload["results"]]

            if query_genes:
                covered = set()
                for cid in cids:
                    for g in self.gene_stats.get(cid, {}):
                        if g.upper() in query_genes:
                            covered.add(g.upper())
                coverages.append(len(covered) / len(query_genes))
            else:
                if baseline is None:
                    baseline = set(cids)
                    coverages.append(1.0)
                else:
                    coverages.append(len(baseline & set(cids)) / max(len(baseline), 1))

        return {
            "query": query,
            "lambdas": lambdas,
            "coverages": coverages,
            "metric": "query_gene_coverage" if query_genes else "rank_overlap_vs_lambda0",
            "top_k": top_k,
        }

    # --------------------------------------------------------
    # PROPORTIONS
    # --------------------------------------------------------

    def proportions(self) -> dict:
        counts = self._cluster_counts()
        if not counts:
            return {
                "query": "Cell type proportions",
                "error": "No per-cell counts available. Attach an h5ad "
                         "(engine.attach_adata(adata)) or pass cells_csv.",
            }

        total = sum(counts.values())
        rows = [{
            "cluster_id": cid,
            "n_cells": int(n),
            "fraction": round(n / total, 5),
            "percent": round(100.0 * n / total, 3),
        } for cid, n in counts.items()]
        rows.sort(key=lambda r: -r["n_cells"])

        payload = {
            "query": "Cell type proportions",
            "mode": "proportions",
            "total_cells": int(total),
            "n_clusters": len(rows),
            "proportions": rows,
            "source": "h5ad" if self._adata is not None else "metadata",
        }

        caps = self.detect_capabilities()
        obs = self._obs()
        if caps["has_conditions"] and obs is not None and self._adata_cluster_key:
            col = caps["condition_column"]
            ckey = self._adata_cluster_key
            cond_props = {}
            for cond in caps["condition_values"]:
                sub = obs[obs[col].astype(str) == cond]
                if len(sub) == 0:
                    continue
                vc = sub[ckey].astype(str).value_counts()
                sub_total = int(vc.sum())
                clusters = [{
                    "cluster_id": str(k),
                    "n_cells": int(v),
                    "fraction": round(v / sub_total, 5),
                } for k, v in vc.items()]
                clusters.sort(key=lambda r: -r["n_cells"])
                cond_props[cond] = {"total_cells": sub_total, "clusters": clusters}

            payload["condition_column"] = col
            payload["condition_proportions"] = cond_props

            if len(cond_props) == 2:
                a, b = list(cond_props.keys())
                fa = {c["cluster_id"]: c["fraction"] for c in cond_props[a]["clusters"]}
                fb = {c["cluster_id"]: c["fraction"] for c in cond_props[b]["clusters"]}
                fcs = []
                for cid in set(fa) | set(fb):
                    va, vb = fa.get(cid, 0.0), fb.get(cid, 0.0)
                    fcs.append({
                        "cluster_id": cid,
                        f"fraction_{a}": round(va, 5),
                        f"fraction_{b}": round(vb, 5),
                        "log2_fold_change": round(
                            math.log2((va + 1e-6) / (vb + 1e-6)), 4),
                    })
                fcs.sort(key=lambda r: -abs(r["log2_fold_change"]))
                payload["proportion_fold_changes"] = fcs
                payload["fold_change_direction"] = f"positive = enriched in {a}"

        return payload

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    def compare(self, group_a: str, group_b: str,
                genes: Optional[List[str]] = None) -> dict:
        caps = self.detect_capabilities()
        if not caps["has_conditions"]:
            return {
                "error": "No condition column detected. Comparative analysis "
                         "needs per-cell labels — attach an h5ad with a "
                         "condition/disease/group column, or pass cells_csv.",
                "checked_columns": CONDITION_KEY_CANDIDATES,
            }

        col = caps["condition_column"]
        obs = self._obs()
        ckey = self._adata_cluster_key
        if obs is None or ckey is None:
            return {"error": "Condition metadata found but no per-cell table "
                             "is attached; cannot compare."}

        vals = {str(v).lower(): str(v) for v in caps["condition_values"]}
        a = vals.get(str(group_a).lower())
        b = vals.get(str(group_b).lower())
        if a is None or b is None:
            return {"error": f"Unknown condition(s). Available: "
                             f"{caps['condition_values']}"}

        labels = obs[col].astype(str)
        clusters = obs[ckey].astype(str)
        mask_a = labels == a
        mask_b = labels == b
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())

        va = clusters[mask_a].value_counts()
        vb = clusters[mask_b].value_counts()

        cluster_block = {}
        for cid in sorted(set(va.index) | set(vb.index)):
            ca, cb = int(va.get(cid, 0)), int(vb.get(cid, 0))
            fa = ca / n_a if n_a else 0.0
            fb = cb / n_b if n_b else 0.0
            cluster_block[cid] = {
                f"n_{a}": ca,
                f"n_{b}": cb,
                f"fraction_{a}": round(fa, 5),
                f"fraction_{b}": round(fb, 5),
                "log2_fold_change": round(math.log2((fa + 1e-6) / (fb + 1e-6)), 4),
                "genes": [],
            }

        # Optional per-gene comparison, only if we have the matrix
        gene_note = None
        if genes:
            if self._adata is None:
                gene_note = ("Gene-level comparison skipped: no expression "
                             "matrix attached.")
            else:
                gene_note = ("Mean log-normalized expression and percent "
                             "expressing, per cluster per condition. "
                             "Descriptive only — no statistical test applied.")
                for g in genes:
                    gu = g.upper()
                    if gu not in {v.upper() for v in self._adata.var_names}:
                        continue
                    idx = [i for i, v in enumerate(self._adata.var_names)
                           if str(v).upper() == gu]
                    if not idx:
                        continue
                    vec = self._adata.X[:, idx[0]]
                    try:
                        from scipy import sparse
                        if sparse.issparse(vec):
                            vec = np.asarray(vec.todense()).ravel()
                    except ImportError:
                        pass
                    vec = np.asarray(vec).ravel()

                    for cid, block in cluster_block.items():
                        cm = (clusters == cid).values
                        sa = vec[cm & mask_a.values]
                        sb = vec[cm & mask_b.values]
                        if sa.size == 0 and sb.size == 0:
                            continue
                        ma = float(sa.mean()) if sa.size else 0.0
                        mb = float(sb.mean()) if sb.size else 0.0
                        block["genes"].append({
                            "gene": g,
                            f"mean_{a}": round(ma, 4),
                            f"mean_{b}": round(mb, 4),
                            f"pct_expressing_{a}": round(
                                float((sa > 0).mean()) if sa.size else 0.0, 4),
                            f"pct_expressing_{b}": round(
                                float((sb > 0).mean()) if sb.size else 0.0, 4),
                            "delta": round(ma - mb, 4),
                        })

        # Fall back to marker overlap so the LLM has gene context either way
        if not genes:
            for cid, block in cluster_block.items():
                block["genes"] = [e["gene"] for e in self._gene_evidence(cid, 15)]

        return {
            "query": f"{a} vs {b}",
            "mode": "compare",
            "group_a": a,
            "group_b": b,
            "condition_column": col,
            f"n_cells_{a}": n_a,
            f"n_cells_{b}": n_b,
            "clusters": cluster_block,
            "fold_change_direction": f"positive = enriched in {a}",
            "caveat": "Cluster fold changes are compositional. "
                      "No formal differential-abundance test was run.",
            "gene_note": gene_note,
        }

    # --------------------------------------------------------
    # PATHWAY
    # --------------------------------------------------------

    def _score_pathway_in_cluster(self, cid: str, pathway_genes: List[str]):
        stats = self.gene_stats.get(str(cid), {})
        if not stats:
            return 0.0, []
        upper = {g.upper(): s for g, s in stats.items()}
        hits = []
        for g in pathway_genes:
            s = upper.get(g.upper())
            if s is None:
                continue
            lf = _f(s.get("logfc"))
            pin = _pct(s.get("pct_in"))
            pout = _pct(s.get("pct_out"))
            spec = max(pin - pout, 0.0)
            hits.append({
                "gene": g,
                "logfc": round(lf, 4),
                "pct_in": round(pin, 4),
                "contribution": round(max(lf, 0.0) * (0.3 + spec), 4),
            })
        if not hits:
            return 0.0, []
        total = sum(h["contribution"] for h in hits)
        score = total / len(pathway_genes)
        hits.sort(key=lambda h: -h["contribution"])
        return score, hits

    def pathway(self, pathway_name: str = None) -> dict:
        metric = ("mean marker-enrichment score over pathway genes "
                  "(max(logFC,0) x (0.3 + specificity)), normalized by "
                  "pathway size")

        if pathway_name is None or str(pathway_name).lower() == "all":
            ranked = []
            for name, genes in PATHWAYS.items():
                per_cluster = []
                for cid in self.cluster_ids:
                    sc, hits = self._score_pathway_in_cluster(str(cid), genes)
                    per_cluster.append((str(cid), sc, len(hits)))
                per_cluster.sort(key=lambda t: -t[1])
                top = per_cluster[0] if per_cluster else (None, 0.0, 0)
                ranked.append({
                    "pathway": name,
                    "top_cluster": top[0],
                    "top_score": round(top[1], 4),
                    "n_genes_found": top[2],
                    "pathway_size": len(genes),
                })
            ranked.sort(key=lambda r: -r["top_score"])
            return {
                "query": "all pathways",
                "mode": "pathway",
                "metric": metric,
                "ranked": ranked,
                "caveat": "Scores derive from per-cluster marker statistics, "
                          "not per-cell expression. Not AUCell/ssGSEA.",
            }

        key = str(pathway_name).strip().lower().replace(" ", "_").replace("-", "_")
        if key not in PATHWAYS:
            matches = [p for p in PATHWAYS if key in p]
            if len(matches) == 1:
                key = matches[0]
            else:
                return {
                    "error": f"Unknown pathway '{pathway_name}'",
                    "available": sorted(PATHWAYS.keys()),
                }

        genes = PATHWAYS[key]
        scores = []
        for cid in self.cluster_ids:
            sc, hits = self._score_pathway_in_cluster(str(cid), genes)
            scores.append({
                "cluster_id": str(cid),
                "score": round(sc, 4),
                "n_genes_found": len(hits),
                "top_genes": hits[:8],
            })
        scores.sort(key=lambda r: -r["score"])

        return {
            "query": key,
            "mode": "pathway",
            "pathway": key,
            "metric": metric,
            "genes_in_pathway": genes,
            "scores": scores,
            "caveat": "Scores derive from per-cluster marker statistics, "
                      "not per-cell expression. Not AUCell/ssGSEA.",
        }

    # --------------------------------------------------------
    # INTERACTIONS
    # --------------------------------------------------------

    def _enriched_genes(self, cid: str, min_pct: float = 0.20,
                        min_logfc: float = 0.10) -> Dict[str, float]:
        out = {}
        for g, s in self.gene_stats.get(str(cid), {}).items():
            pin = _pct(s.get("pct_in"))
            lf = _f(s.get("logfc"))
            if pin >= min_pct and lf >= min_logfc:
                out[g.upper()] = pin
        return out

    def interactions(self, source: str = None, target: str = None,
                     max_results: int = 200) -> dict:
        expressed = {str(c): self._enriched_genes(str(c)) for c in self.cluster_ids}

        src_list = [self._cid(source)] if source else [str(c) for c in self.cluster_ids]
        tgt_list = [self._cid(target)] if target else [str(c) for c in self.cluster_ids]

        found = []
        for lig, rec, pw in LR_PAIRS:
            for s in src_list:
                pl = expressed.get(s, {}).get(lig)
                if not pl:
                    continue
                for t in tgt_list:
                    pr = expressed.get(t, {}).get(rec)
                    if not pr:
                        continue
                    found.append({
                        "source": s,
                        "target": t,
                        "ligand": lig,
                        "receptor": rec,
                        "pathway": pw,
                        "ligand_pct_in": round(pl, 4),
                        "receptor_pct_in": round(pr, 4),
                        "score": round(pl * pr, 5),
                    })

        found.sort(key=lambda r: -r["score"])
        n_total = len(found)
        found = found[:max_results]

        by_pw = {}
        for it in found:
            b = by_pw.setdefault(it["pathway"], {"pathway": it["pathway"],
                                                 "n": 0, "total": 0.0})
            b["n"] += 1
            b["total"] += it["score"]
        pathway_summary = [{
            "pathway": v["pathway"], "n_interactions": v["n"],
            "mean_score": round(v["total"] / v["n"], 5),
        } for v in by_pw.values()]
        pathway_summary.sort(key=lambda r: -r["n_interactions"])

        by_pair = {}
        for it in found:
            k = (it["source"], it["target"])
            b = by_pair.setdefault(k, {"source": k[0], "target": k[1],
                                       "n": 0, "total": 0.0})
            b["n"] += 1
            b["total"] += it["score"]
        pair_summary = [{
            "source": v["source"], "target": v["target"],
            "n_interactions": v["n"],
            "mean_score": round(v["total"] / v["n"], 5),
        } for v in by_pair.values()]
        pair_summary.sort(key=lambda r: -r["n_interactions"])

        label = "Cell-cell interactions"
        if source and target:
            label = f"Interactions {self._cid(source)} -> {self._cid(target)}"
        elif source:
            label = f"Interactions from {self._cid(source)}"
        elif target:
            label = f"Interactions into {self._cid(target)}"

        payload = {
            "query": label,
            "mode": "interactions",
            "n_total": n_total,
            "n_returned": len(found),
            "interactions": found,
            "pathway_summary": pathway_summary,
            "pair_summary": pair_summary,
            "lr_database_size": len(LR_PAIRS),
            "caveat": "Predicted from co-enrichment of ligand and receptor in "
                      "per-cluster marker statistics. Not a validated "
                      "interaction inference method; no null model or "
                      "permutation testing was performed.",
        }
        if n_total == 0:
            payload["note"] = ("No ligand-receptor pairs from the built-in "
                               "table were co-enriched. With 5 clusters and a "
                               "restricted marker set this is expected; "
                               "lower min_pct in _enriched_genes to loosen.")
        return payload
