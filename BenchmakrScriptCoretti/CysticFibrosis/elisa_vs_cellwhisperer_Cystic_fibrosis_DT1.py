#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA vs CellWhisperer — Head-to-Head Comparison
=================================================
Loads ELISA benchmark results (from benchmark JSON output),
then runs CellWhisperer REAL CLIP model on the same queries, and compares.

Step 1: Run ELISA benchmark first:
    conda activate scgpt_env
    python elisa_benchmark_v5_1.py \
        --base /path/to/embeddings \
        --pt-name fused_file.pt \
        --paper P1 \
        --out results/

Step 2: Run this script (in cellwhisperer env):
    conda activate cellwhisperer
    python elisa_vs_cellwhisperer.py \
        --elisa-results results/benchmark_v5_results.json \
        --cw-npz /path/to/full_output.npz \
        --cw-leiden /path/to/leiden_umap_embeddings.h5ad \
        --cw-ckpt /path/to/cellwhisperer_clip_v1.ckpt \
        --cf-h5ad /path/to/read_count_table.h5ad \
        --out comparison_results/
"""

import os, sys, json, argparse
from datetime import datetime
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# QUERIES
# ============================================================

QUERIES = [
    # --- ONTOLOGY (Q1-Q15) ---
    {"id": "Q01", "text": "macrophage and monocyte infiltration in cystic fibrosis airways",
     "category": "ontology",
     "expected_clusters": ["macrophage", "monocyte", "non-classical monocyte"],
     "expected_genes": ["CD68", "CD14", "CSF1R", "MARCO", "C1QB"]},

    {"id": "Q02", "text": "CD8 T cell activation and cytotoxicity in CF lung inflammation",
     "category": "ontology",
     "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"],
     "expected_genes": ["IFNG", "CD69", "CD81", "CD3G", "GZMB", "PRF1"]},

    {"id": "Q03", "text": "CD4 helper T cell immune activation in cystic fibrosis",
     "category": "ontology",
     "expected_clusters": ["CD4-positive helper T cell", "mature T cell"],
     "expected_genes": ["KLF2", "IL7R", "CD48", "TXNIP", "ETS1"]},

    {"id": "Q04", "text": "B cell activation and immunoglobulin response in CF airways",
     "category": "ontology",
     "expected_clusters": ["B cell", "plasma cell"],
     "expected_genes": ["CD79A", "IGHG3", "IGLC2", "JCHAIN", "SYK", "CD81"]},

    {"id": "Q05", "text": "basal cell dysfunction and reduced stemness in cystic fibrosis epithelium",
     "category": "ontology",
     "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
     "expected_genes": ["KRT5", "KRT14", "TP63", "CSTA", "HSPB1"]},

    {"id": "Q06", "text": "ciliated cell ciliogenesis and increased abundance in CF bronchial epithelium",
     "category": "ontology",
     "expected_clusters": ["ciliated cell"],
     "expected_genes": ["FOXJ1", "DNAH5", "SYNE1", "SYNE2", "CAPS"]},

    {"id": "Q07", "text": "natural killer cell cytotoxicity and NKG2A immune checkpoint in CF",
     "category": "ontology",
     "expected_clusters": ["natural killer cell", "innate lymphoid cell"],
     "expected_genes": ["GNLY", "NKG7", "KLRD1", "KLRK1", "KLRC1", "PRF1"]},

    {"id": "Q08", "text": "pulmonary ionocyte CFTR expression in cystic fibrosis",
     "category": "ontology",
     "expected_clusters": ["ionocyte"],
     "expected_genes": ["FOXI1", "CFTR", "ATP6V1G3", "BSND", "ASCL3"]},

    {"id": "Q09", "text": "endothelial cell remodeling and VEGF signaling in CF lung",
     "category": "ontology",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["VEGFA", "PLVAP", "ACKR1", "VIM", "ERG"]},

    {"id": "Q10", "text": "dendritic cell antigen presentation in CF airways",
     "category": "ontology",
     "expected_clusters": ["dendritic cell, human"],
     "expected_genes": ["HLA-DPA1", "HLA-DRB1", "CD74", "CD80", "CD86"]},

    {"id": "Q11", "text": "mast cell degranulation and allergic inflammation in CF",
     "category": "ontology",
     "expected_clusters": ["mast cell"],
     "expected_genes": ["CPA3", "TPSAB1", "TPSB2", "MS4A2", "KIT"]},

    {"id": "Q12", "text": "HLA-E CD94 NKG2A immune checkpoint inhibiting CD8 T cell activity",
     "category": "ontology",
     "expected_clusters": ["CD8-positive, alpha-beta T cell", "natural killer cell"],
     "expected_genes": ["HLA-E", "KLRC1", "KLRD1", "KLRC2", "CD8A"]},

    {"id": "Q13", "text": "type I interferon response and inflammatory signaling in CF epithelial cells",
     "category": "ontology",
     "expected_clusters": ["basal cell", "ciliated cell", "secretory cell"],
     "expected_genes": ["IFIT1", "MX1", "OAS2", "ISG15", "IFITM3"]},

    {"id": "Q14", "text": "submucosal gland epithelial cell changes in cystic fibrosis",
     "category": "ontology",
     "expected_clusters": ["submucosal gland epithelial cell",
                           "serous cell of epithelium of bronchus"],
     "expected_genes": ["MUC5AC", "MUC5B", "SCGB1A1", "LYZ", "SCGB3A1"]},

    {"id": "Q15", "text": "VEGF receptor signaling and hypoxia response across cell types in CF",
     "category": "ontology",
     "expected_clusters": ["endothelial cell", "CD8-positive, alpha-beta T cell",
                           "CD4-positive helper T cell", "basal cell"],
     "expected_genes": ["TXNIP", "MAP2K2", "ETS1", "VEGFA"]},

    # --- EXPRESSION (Q16-Q30) ---
    {"id": "Q16", "text": "IGLJ3 IGKJ1 IGHJ5 JCHAIN MZB1 XBP1",
     "category": "expression",
     "expected_clusters": ["plasma cell", "B cell"],
     "expected_genes": ["JCHAIN", "IGHG3", "IGLC2", "MZB1", "XBP1"]},

    {"id": "Q17", "text": "CD8A CD8B GZMB PRF1 IFNG NKG7",
     "category": "expression",
     "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"],
     "expected_genes": ["CD8A", "GZMB", "PRF1", "IFNG", "NKG7"]},

    {"id": "Q18", "text": "TRAJ52 TRBV22-1 TRDJ2 CD3E CD3G",
     "category": "expression",
     "expected_clusters": ["CD8-positive, alpha-beta T cell",
                           "CD4-positive helper T cell", "mature T cell"],
     "expected_genes": ["CD3G", "CD3E", "CD69", "IL7R"]},

    {"id": "Q19", "text": "CPA3 TPSAB1 TPSB2 MS4A2 HDC GATA2",
     "category": "expression",
     "expected_clusters": ["mast cell"],
     "expected_genes": ["CPA3", "TPSAB1", "TPSB2", "MS4A2", "KIT"]},

    {"id": "Q20", "text": "MARCO FABP4 APOC1 C1QB C1QC MSR1",
     "category": "expression",
     "expected_clusters": ["macrophage", "monocyte", "non-classical monocyte"],
     "expected_genes": ["C1QB", "MARCO", "FABP4", "CD68"]},

    {"id": "Q21", "text": "COL1A2 LUM DCN SFRP2 COL3A1 PDGFRA",
     "category": "expression",
     "expected_clusters": ["fibroblast of lung", "alveolar adventitial fibroblast",
                           "alveolar type 1 fibroblast cell"],
     "expected_genes": ["COL1A2", "LUM", "DCN", "COL3A1"]},

    {"id": "Q22", "text": "ATP6V1G3 FOXI1 BSND CLCNKB ASCL3",
     "category": "expression",
     "expected_clusters": ["ionocyte"],
     "expected_genes": ["FOXI1", "ATP6V1G3", "BSND", "CFTR"]},

    {"id": "Q23", "text": "GNLY KLRD1 KLRK1 NKG7 PRF1 GZMB",
     "category": "expression",
     "expected_clusters": ["natural killer cell", "innate lymphoid cell", "cytotoxic T cell"],
     "expected_genes": ["GNLY", "NKG7", "PRF1", "GZMB", "KLRD1"]},

    {"id": "Q24", "text": "SST CHGA ASCL1 GRP CALCA SYP",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["CHGA", "ASCL1", "GRP", "SYP"]},

    {"id": "Q25", "text": "KRT5 KRT14 KRT15 TP63 IL33 CSTA",
     "category": "expression",
     "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
     "expected_genes": ["KRT5", "KRT14", "TP63", "CSTA", "HSPB1"]},

    {"id": "Q26", "text": "FOXJ1 DNAH5 CAPS PIFO RSPH1 DNAI1",
     "category": "expression",
     "expected_clusters": ["ciliated cell"],
     "expected_genes": ["FOXJ1", "DNAH5", "CAPS", "SYNE1", "SYNE2"]},

    {"id": "Q27", "text": "PLVAP ACKR1 ERG VWF PECAM1 CDH5",
     "category": "expression",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["PLVAP", "ACKR1", "ERG", "VIM", "VEGFA"]},

    {"id": "Q28", "text": "SCGB1A1 SCGB3A1 MUC5AC MUC5B LYPD2 PRR4",
     "category": "expression",
     "expected_clusters": ["secretory cell", "submucosal gland epithelial cell"],
     "expected_genes": ["SCGB1A1", "MUC5AC", "MUC5B", "SCGB3A1"]},

    {"id": "Q29", "text": "CALR LRP1 GNAI2 FOS JUND MAP2K2",
     "category": "expression",
     "expected_clusters": ["CD8-positive, alpha-beta T cell",
                           "CD4-positive helper T cell", "macrophage"],
     "expected_genes": ["CALR", "GNAI2", "FOS", "JUND", "MAP2K2"]},

    {"id": "Q30", "text": "HLA-E KLRC1 KLRD1 KLRC2 KLRC3 KLRK1",
     "category": "expression",
     "expected_clusters": ["CD8-positive, alpha-beta T cell",
                           "natural killer cell", "innate lymphoid cell"],
     "expected_genes": ["HLA-E", "KLRC1", "KLRD1", "KLRC2", "KLRK1"]},
]


# ============================================================
# METRICS
# ============================================================

def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def cluster_recall_at_k(expected, retrieved, k=5):
    if not expected:
        return 0.0
    ret_lower = [r.lower() for r in retrieved[:k]]
    found = 0
    for exp in expected:
        exp_l = exp.lower()
        if any(exp_l in r or r in exp_l or _word_overlap(exp_l, r) >= 0.5
               for r in ret_lower):
            found += 1
    return found / len(expected)


def mrr(expected, retrieved):
    for rank, ret in enumerate(retrieved, 1):
        ret_l = ret.lower()
        for exp in expected:
            exp_l = exp.lower()
            if exp_l in ret_l or ret_l in exp_l or _word_overlap(exp_l, ret_l) >= 0.5:
                return 1.0 / rank
    return 0.0


# ============================================================
# CELLWHISPERER SCORER  — real CLIP embeddings
# ============================================================

class CellWhispererScorer:
    """Score queries using real CellWhisperer CLIP embeddings."""

    def __init__(self, npz_path, leiden_h5ad_path, ckpt_path, cf_h5ad_path):
        import anndata as ad

        print("[CW] Loading precomputed cell embeddings...")
        data = np.load(npz_path, allow_pickle=True)
        self.cell_embeds = data["transcriptome_embeds"]
        self.orig_ids = data["orig_ids"]
        print(f"     shape: {self.cell_embeds.shape}")

        print("[CW] Loading leiden cluster assignments...")
        leiden_adata = ad.read_h5ad(leiden_h5ad_path)
        self.leiden_labels = leiden_adata.obs["leiden"].values
        print(f"     {len(np.unique(self.leiden_labels))} leiden clusters")

        print("[CW] Loading original cell type annotations...")
        cf_adata = ad.read_h5ad(cf_h5ad_path)
        self.cell_types = cf_adata.obs["cell_type"].values

        # Cell type -> mean embedding
        ct_embed_accum = {}
        ct_counts = {}
        for i, ct in enumerate(self.cell_types):
            if ct not in ct_embed_accum:
                ct_embed_accum[ct] = np.zeros(self.cell_embeds.shape[1], dtype=np.float64)
                ct_counts[ct] = 0
            ct_embed_accum[ct] += self.cell_embeds[i].astype(np.float64)
            ct_counts[ct] += 1

        self.celltype_embeds = {}
        for ct in ct_embed_accum:
            self.celltype_embeds[ct] = (ct_embed_accum[ct] / ct_counts[ct]).astype(np.float32)

        print(f"[CW] {len(self.celltype_embeds)} unique cell types:")
        for ct in sorted(self.celltype_embeds.keys()):
            print(f"     {ct} ({ct_counts[ct]} cells)")

        # Load CLIP model
        self.model = None
        self._load_model(ckpt_path)

    def _load_model(self, ckpt_path):
        try:
            import torch
            cw_src = os.path.join(os.path.dirname(ckpt_path), "..", "..", "src")
            if os.path.isdir(cw_src):
                sys.path.insert(0, cw_src)

            from cellwhisperer.jointemb.cellwhisperer_lightning import (
                TranscriptomeTextDualEncoderLightning,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(ckpt_path)
            pl_model.eval().to(self.device)
            pl_model.freeze()
            self.model = pl_model.model
            print(f"[CW] CLIP model loaded on {self.device}")
        except Exception as e:
            print(f"[CW] WARNING: Could not load CLIP model: {e}")
            print(f"[CW] Will use random ranking as fallback!")
            self.model = None

    def embed_text(self, query_text):
        import torch
        if self.model is not None:
            with torch.no_grad():
                return self.model.embed_texts([query_text]).cpu().numpy()[0]
        return None

    def score_query(self, query_text, top_k=10):
        """Rank cell types by cosine similarity to query. Returns list of names."""
        text_embed = self.embed_text(query_text)
        if text_embed is None:
            ct_list = list(self.celltype_embeds.keys())
            np.random.shuffle(ct_list)
            return ct_list[:top_k]

        t_norm = text_embed / (np.linalg.norm(text_embed) + 1e-8)
        scores = {}
        for ct, ct_embed in self.celltype_embeds.items():
            c_norm = ct_embed / (np.linalg.norm(ct_embed) + 1e-8)
            scores[ct] = float(np.dot(t_norm, c_norm))

        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        self._last_scores = scores
        return ranked[:top_k]


# ============================================================
# MAIN COMPARISON
# ============================================================

def run_comparison(cw_scorer, elisa_json_path, paper_id, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ── Load ELISA results ──
    print(f"\n[ELISA] Loading results from {elisa_json_path}")
    with open(elisa_json_path) as f:
        elisa_all = json.load(f)

    paper = elisa_all[paper_id]
    elisa_detail  = paper["retrieval_detail"]
    elisa_summary = paper["retrieval_summary"]
    elisa_analytical = paper.get("analytical", {})

    # ── Run CellWhisperer ──
    print("\n" + "=" * 70)
    print("[CW] Running CellWhisperer on 30 queries...")
    print("=" * 70)

    cw_results = {"ontology": [], "expression": []}
    for q in QUERIES:
        ranked = cw_scorer.score_query(q["text"], top_k=15)
        r5  = cluster_recall_at_k(q["expected_clusters"], ranked, k=5)
        r10 = cluster_recall_at_k(q["expected_clusters"], ranked, k=10)
        m   = mrr(q["expected_clusters"], ranked)

        cw_results[q["category"]].append({
            "query_id": q["id"],
            "query_text": q["text"],
            "expected": q["expected_clusters"],
            "retrieved_top10": ranked[:10],
            "recall@5": round(r5, 4),
            "recall@10": round(r10, 4),
            "mrr": round(m, 4),
        })
        print(f"  [{q['id']}] R@5={r5:.2f}  MRR={m:.2f}  Top3={ranked[:3]}")

    cw_agg = {}
    for cat in ["ontology", "expression"]:
        e = cw_results[cat]
        cw_agg[cat] = {
            "mean_recall@5":    round(np.mean([x["recall@5"]  for x in e]), 4),
            "mean_recall@10":   round(np.mean([x["recall@10"] for x in e]), 4),
            "mean_mrr":         round(np.mean([x["mrr"]       for x in e]), 4),
            "mean_gene_recall": 0.0,
        }

    # ── Build unified mode list: random, CW real, semantic, scgpt, union ──
    # Only keep modes that exist in ELISA results and are in our target set
    ELISA_TARGET_MODES = ["random", "semantic", "scgpt", "union"]
    ALL_MODES = ["random", "cellwhisperer_real", "semantic", "scgpt", "union"]

    MODE_COLORS = {
        "random": "#9E9E9E",
        "cellwhisperer_real": "#E91E63",
        "semantic": "#2196F3",
        "scgpt": "#FF9800",
        "union": "#4CAF50",
    }
    MODE_LABELS = {
        "random": "Random",
        "cellwhisperer_real": "CellWhisp.",
        "semantic": "Semantic",
        "scgpt": "scGPT",
        "union": "Union(S+G)",
    }

    def gm(cat, mode, mk):
        if mode == "cellwhisperer_real":
            return cw_agg[cat].get(mk, 0)
        return elisa_summary.get(f"{cat}_{mode}", {}).get(mk, 0)

    # ── Console output ──
    print("\n" + "=" * 90)
    print("ELISA vs CellWhisperer — BENCHMARK RESULTS (30 queries)")
    print("=" * 90)
    print(f"\n{'Category':<14} {'Mode':<16} {'Recall@5':>10} {'Recall@10':>10} "
          f"{'MRR':>10} {'GeneRec':>10}")
    print("-" * 72)
    for cat in ["ontology", "expression"]:
        for mode in ALL_MODES:
            r5  = gm(cat, mode, "mean_recall@5")
            r10 = gm(cat, mode, "mean_recall@10")
            mmr = gm(cat, mode, "mean_mrr")
            gr  = gm(cat, mode, "mean_gene_recall")
            gr_s = "  N/A" if mode == "cellwhisperer_real" else f"{gr:.3f}"
            print(f"{cat:<14} {MODE_LABELS.get(mode,mode):<16} "
                  f"{r5:>10.3f} {r10:>10.3f} {mmr:>10.3f} {gr_s:>10}")
        print()

    print("── Overall ──")
    for mode in ALL_MODES:
        r5 = np.mean([gm(c, mode, "mean_recall@5") for c in ["ontology","expression"]])
        print(f"  {MODE_LABELS.get(mode,mode):<16} Recall@5={r5:.3f}")

    ana = elisa_analytical
    print("\n── Analytical Modules (ELISA only) ──")
    print(f"  Pathways:     {ana.get('pathways',{}).get('alignment',0):.1f}%")
    print(f"  Interactions: {ana.get('interactions',{}).get('lr_recovery_rate',0):.1f}% LR")
    print(f"  Proportions:  {ana.get('proportions',{}).get('consistency_rate',0):.1f}%")
    print(f"  Compare:      {ana.get('compare',{}).get('compare_recall',0):.1f}%")
    print("=" * 90)

    # ============================================================
    # FIGURES
    # ============================================================

    # Fig 1: Recall@5 bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cat, ttl in zip(axes, ["ontology","expression"],
                             ["Ontology Queries (concept-level)",
                              "Expression Queries (gene-signature)"]):
        x = np.arange(len(ALL_MODES))
        vals = [gm(cat, m, "mean_recall@5") for m in ALL_MODES]
        bars = ax.bar(x, vals, color=[MODE_COLORS.get(m,"#666") for m in ALL_MODES],
                      alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS.get(m,m) for m in ALL_MODES],
                           fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean Cluster Recall@5")
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("ELISA vs CellWhisperer — CF Airways (30 queries)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_recall5.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig1_recall5.pdf"), bbox_inches="tight")
    plt.close()
    print(f"\n[FIG] fig1_recall5.png")

    # Fig 2: All metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, mk, mt in zip(axes,
                           ["mean_recall@5","mean_recall@10","mean_mrr"],
                           ["Recall@5","Recall@10","MRR"]):
        x = np.arange(len(ALL_MODES)); w = 0.35
        ont = [gm("ontology",m,mk) for m in ALL_MODES]
        exp = [gm("expression",m,mk) for m in ALL_MODES]
        ax.bar(x-w/2, ont, w, label="Ontology", alpha=0.85,
               color=[MODE_COLORS.get(m,"#666") for m in ALL_MODES], edgecolor="white")
        ax.bar(x+w/2, exp, w, label="Expression", alpha=0.45,
               color=[MODE_COLORS.get(m,"#666") for m in ALL_MODES],
               edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS.get(m,m) for m in ALL_MODES],
                           fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.15); ax.set_title(mt, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_all_metrics.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_all_metrics.pdf"), bbox_inches="tight")
    plt.close()
    print("[FIG] fig2_all_metrics.png")

    # Fig 3: Per-query (Union vs CW)
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for ax, cat in zip(axes, ["ontology","expression"]):
        cw_e = cw_results[cat]
        qids = [e["query_id"] for e in cw_e]
        cw_r5 = [e["recall@5"] for e in cw_e]
        el_union = elisa_detail.get(cat,{}).get("union",[])
        el_r5 = [e.get("recall@5",0) for e in el_union] if el_union else [0]*len(qids)
        n = min(len(qids), len(el_r5))
        y = np.arange(n); h = 0.35
        ax.barh(y-h/2, el_r5[:n], h, label="ELISA Union", color="#4CAF50", alpha=0.8)
        ax.barh(y+h/2, cw_r5[:n], h, label="CellWhisperer", color="#E91E63", alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(qids[:n], fontsize=9)
        ax.set_xlabel("Recall@5"); ax.set_xlim(0, 1.1)
        ax.set_title(f"{cat.capitalize()} Queries", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    plt.suptitle("Per-Query: ELISA Union vs CellWhisperer", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_perquery.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_perquery.pdf"), bbox_inches="tight")
    plt.close()
    print("[FIG] fig3_perquery.png")

    # Fig 4: Radar
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    rl = ["Ont R@5","Ont R@10","Ont MRR","Exp R@5","Exp R@10","Exp MRR"]
    N = len(rl); angles = [n/float(N)*2*np.pi for n in range(N)]; angles += angles[:1]
    for mode, color, ls in [("cellwhisperer_real","#E91E63","--"),
                             ("semantic","#2196F3","-"),
                             ("scgpt","#FF9800","-"),
                             ("union","#4CAF50","-")]:
        v = []
        for cat in ["ontology","expression"]:
            for mk in ["mean_recall@5","mean_recall@10","mean_mrr"]:
                v.append(gm(cat, mode, mk))
        v += v[:1]
        ax.plot(angles, v, linewidth=2, linestyle=ls, label=MODE_LABELS.get(mode,mode), color=color)
        ax.fill(angles, v, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rl, fontsize=10)
    ax.set_ylim(0,1); ax.set_title("ELISA vs CellWhisperer", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_radar.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig4_radar.pdf"), bbox_inches="tight")
    plt.close()
    print("[FIG] fig4_radar.png")

    # Fig 5: Gene recall + analytical
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    # 5a: Gene recall — show ELISA modes + CW (which has 0 gene recall)
    gm_modes = ALL_MODES
    x = np.arange(len(gm_modes)); w = 0.35
    og = [gm("ontology",m,"mean_gene_recall") for m in gm_modes]
    eg = [gm("expression",m,"mean_gene_recall") for m in gm_modes]
    clrs = [MODE_COLORS.get(m,"#666") for m in gm_modes]
    ax1.bar(x-w/2, og, w, label="Ontology", alpha=0.85, color=clrs, edgecolor="white")
    ax1.bar(x+w/2, eg, w, label="Expression", alpha=0.45, color=clrs,
            edgecolor="black", linewidth=0.5, hatch="//")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODE_LABELS.get(m,m) for m in gm_modes], fontsize=8, rotation=20, ha="right")
    ax1.set_ylim(0,1.1); ax1.set_ylabel("Gene Recall")
    ax1.set_title("Gene-Level Evidence Delivery", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # 5b: Analytical modules
    ax2.set_axis_off()
    ax2b = fig.add_axes([0.55, 0.1, 0.4, 0.75], polar=True)
    al = ["Pathways","Interactions\n(LR)","Proportions","Compare"]
    av = [ana.get("pathways",{}).get("alignment",0)/100,
          ana.get("interactions",{}).get("lr_recovery_rate",0)/100,
          ana.get("proportions",{}).get("consistency_rate",0)/100,
          ana.get("compare",{}).get("compare_recall",0)/100]
    aa = np.linspace(0, 2*np.pi, len(al), endpoint=False).tolist()
    av_c = av + av[:1]; aa_c = aa + aa[:1]
    ax2b.fill(aa_c, av_c, alpha=0.25, color="#4CAF50")
    ax2b.plot(aa_c, av_c, "o-", color="#4CAF50", linewidth=2, label="ELISA")
    ax2b.plot(aa_c, [0]*len(aa_c), "--", color="#E91E63", linewidth=1.5, label="CellWhisp. (N/A)")
    ax2b.set_xticks(aa); ax2b.set_xticklabels(al, fontsize=9)
    ax2b.set_ylim(0,1.05); ax2b.set_title("Analytical Modules\n(ELISA only)", fontsize=11, fontweight="bold", pad=15)
    ax2b.legend(loc="upper right", bbox_to_anchor=(1.35,1.05), fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.pdf"), bbox_inches="tight")
    plt.close()
    print("[FIG] fig5_gene_analytical.png")

    # ── Save JSON ──
    output = {
        "cellwhisperer": {"results": cw_results, "aggregated": cw_agg},
        "elisa_summary": elisa_summary,
        "elisa_analytical": elisa_analytical,
        "comparison": {
            cat: {mode: {mk: gm(cat,mode,mk)
                         for mk in ["mean_recall@5","mean_recall@10","mean_mrr"]}
                  for mode in ALL_MODES}
            for cat in ["ontology","expression"]
        },
        "modes": ALL_MODES,
        "n_queries": len(QUERIES),
        "timestamp": datetime.now().isoformat(),
    }
    rp = os.path.join(out_dir, "elisa_vs_cellwhisperer_results.json")
    with open(rp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")


def main():
    parser = argparse.ArgumentParser(description="ELISA vs CellWhisperer comparison")
    parser.add_argument("--elisa-results", required=True,
                        help="ELISA benchmark_v5_results.json")
    parser.add_argument("--cw-npz", required=True,
                        help="CellWhisperer full_output.npz")
    parser.add_argument("--cw-leiden", required=True,
                        help="CellWhisperer leiden_umap_embeddings.h5ad")
    parser.add_argument("--cw-ckpt", required=True,
                        help="cellwhisperer_clip_v1.ckpt")
    parser.add_argument("--cf-h5ad", required=True,
                        help="Original CF read_count_table.h5ad")
    parser.add_argument("--paper", default="P1")
    parser.add_argument("--out", default="comparison_results/")
    args = parser.parse_args()

    cw = CellWhispererScorer(args.cw_npz, args.cw_leiden, args.cw_ckpt, args.cf_h5ad)
    run_comparison(cw, args.elisa_results, args.paper, args.out)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
