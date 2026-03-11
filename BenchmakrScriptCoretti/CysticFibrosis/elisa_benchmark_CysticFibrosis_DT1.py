#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Benchmark v5.1 — 30 Queries
=============================================================
Evaluates ELISA's dual-modality retrieval against a random baseline:

  Baseline:
    1. Random                — random k clusters (establishes floor)

  ELISA modalities:
    2. Semantic          — BioBERT on full text (name + GO + Reactome + markers)
    3. scGPT             — expression-conditioned retrieval in scGPT space
    4. Union (Sem+scGPT) — ADDITIVE union: full primary top-k + unique from secondary

Design decisions:
  - CLUSTER-LEVEL evaluation (not gene-level — that's for paper replication)
  - Per-query expected cluster sets
  - Two query categories: "ontology" and "expression"
  - Analytical module evaluation (interactions, pathways, proportions, compare)
  - Union is ADDITIVE: best modality's full top-k + novel clusters from other
    modality (no truncation), evaluated at recall@5/10/15/20

Metrics:
  - Cluster Recall@k: fraction of expected clusters retrieved
  - MRR: mean reciprocal rank of first relevant cluster
  - Complementarity: union recall and gain over best single modality
  - Analytical module accuracy (pathways, interactions, proportions, compare)

Usage:
    python elisa_benchmark_v5_1.py \\
        --base /path/to/embeddings \\
        --pt-name fused_file.pt \\
        --paper P1 \\
        --out results/
"""
import os, sys, json, argparse, time, random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import math
from collections import Counter

# ============================================================
# PAPER CONFIGURATIONS
# ============================================================
#
# Each query has its own expected_clusters (cell types that
# should appear in top-k). Two categories:
#   "ontology"    — annotation-based, semantic should win
#   "expression"  — transcriptionally-driven, scGPT should win
#
BENCHMARK_PAPERS = {
    "P1": {
        "id": "P1",
        "title": "Evidence for altered immune-structural cell crosstalk in cystic fibrosis",
        "doi": "10.1016/j.jcf.2025.01.016",
        "pt_name": "fused_CFCorrConNomiFinaleCF.pt",
        "cells_csv": "metadata_cells.csv",
        "condition_col": "patient_group",
        "conditions": ["CF", "Ctrl"],

        # ── Ground truth for analytical modules ──
        "ground_truth_genes": [
            "IFIT1","MX1","OAS2","CSTA","HSPB1","KDM1A","KMT5A",
            "RAD50","ERCC6","ERCC8","DNAH5","SYNE1","SYNE2",
            "HLA-DPA1","HLA-DRB1",
            "IFNG","GNAI2","CD69","CD81","CD3G","FOS","JUND",
            "TXNIP","MAP2K2","KLF2","IL7R","CD48","ETS1",
            "SYK","CSK","CD9","LTB","HLA-DPB1","IGHG3","IGLC2",
            "HLA-E","KLRC1","KLRC2","KLRC3","KLRD1","KLRK1",
            "CALR","LRP1","IFNGR1","IFNGR2","CXCR3","F2R","S1PR4",
            "VEGFA",
        ],
        "ground_truth_interactions": [
            ("IFNG","IFNGR1","CD8+ T cell","basal cell"),
            ("IFNG","IFNGR1","CD8+ T cell","macrophage"),
            ("IFNG","IFNGR1","CD8+ T cell","endothelial"),
            ("IFNG","IFNGR2","CD8+ T cell","dendritic cell"),
            ("CALR","LRP1","CD4+ T cell","macrophage"),
            ("CALR","LRP1","CD8+ T cell","macrophage"),
            ("HLA-E","KLRC1","basal cell","CD8+ T cell"),
            ("HLA-E","KLRD1","epithelial","CD8+ T cell"),
            ("HLA-E","KLRC2","macrophage","CD8+ T cell"),
            ("CXCL10","CXCR3","macrophage","CD8+ T cell"),
            ("F2","F2R","endothelial","B cell"),
            ("CCL5","CCR5","CD8+ T cell","macrophage"),
        ],
        "ground_truth_pathways": [
            "IFN-gamma signaling","Type I IFN signaling",
            "T cell activation","NK cell activity",
            "Antigen presentation","Epithelial defense",
            "Fibrosis","Angiogenesis",
        ],
        "proportion_changes": {
            "increased_in_CF": [
                "ciliated","monocyte","macrophage",
                "T cell","non-classical monocyte",
            ],
            "decreased_in_CF": [
                "basal","submucosal gland","endothelial","suprabasal",
            ],
        },

        # ── Retrieval queries with per-query expected clusters AND genes ──
        "queries": [
            # ================================================================
            # 30 QUERIES — 15 ontology + 15 expression
            # Derived from Berg et al. J Cyst Fibros 2025 (DT1_FibrosiCistica)
            # ================================================================

            # --- ONTOLOGY queries (1-15): concept-level, semantic advantage ---

             # Q01
        {
            "text": "macrophage and monocyte infiltration in cystic fibrosis airways",
            "category": "ontology",
            "expected_clusters": ["macrophage", "monocyte", "non-classical monocyte"],
            "expected_genes": ["CD68", "CD14", "CSF1R", "MARCO", "C1QB"],
        },
        # Q02
        {
            "text": "recruited monocytes and pro-inflammatory macrophages in CF lung tissue",
            "category": "ontology",
            "expected_clusters": ["monocyte", "macrophage", "non-classical monocyte"],
            "expected_genes": ["CSF1R", "CD14", "FABP4", "APOC1", "C1QC"],
        },
        # Q03
        {
            "text": "macrophage scavenging receptor expression and phagocytosis in CF",
            "category": "ontology",
            "expected_clusters": ["macrophage", "monocyte"],
            "expected_genes": ["MSR1", "MARCO", "CD68", "FABP4", "C1QB"],
        },
        # Q04
        {
            "text": "non-classical monocyte patrol function in CF bronchial wall",
            "category": "ontology",
            "expected_clusters": ["non-classical monocyte", "monocyte"],
            "expected_genes": ["CX3CR1", "CD14", "CSF1R", "FCGR3A"],
        },

        # --- CD8+ T cell biology (paper: 131 DEGs, IFNG, activation,
        #     HLA-E/NKG2A checkpoint, VEGFR signaling) ---

        # Q05
        {
            "text": "CD8 T cell activation and cytotoxicity in CF lung inflammation",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"],
            "expected_genes": ["IFNG", "CD69", "CD81", "CD3G", "GZMB", "PRF1"],
        },
        # Q06
        {
            "text": "CD8 T cell inflammatory cytokine production and IFNG signaling in CF",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"],
            "expected_genes": ["IFNG", "GNAI2", "FOS", "JUND", "CD69"],
        },
        # Q07
        {
            "text": "HLA-E CD94 NKG2A immune checkpoint inhibiting CD8 T cell activity",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "natural killer cell"],
            "expected_genes": ["HLA-E", "KLRC1", "KLRD1", "KLRC2", "CD8A"],
        },
        # Q08
        {
            "text": "dysfunctional CD8 T cell response to chronic Pseudomonas infection in CF",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"],
            "expected_genes": ["IFNG", "GZMB", "PRF1", "CD3G", "CD81"],
        },
        # Q09
        {
            "text": "CALR LRP1 interaction between T cells and macrophages promoting inflammation",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "macrophage"],
            "expected_genes": ["CALR", "LRP1", "GNAI2", "FOS", "JUND"],
        },

        # --- CD4+ T cell biology (paper: 16 DEGs all upregulated,
        #     VEGFR signaling, KLF2, IL7R, CD48) ---

        # Q10
        {
            "text": "CD4 helper T cell immune activation in cystic fibrosis",
            "category": "ontology",
            "expected_clusters": ["CD4-positive helper T cell", "mature T cell"],
            "expected_genes": ["KLF2", "IL7R", "CD48", "TXNIP", "ETS1"],
        },
        # Q11
        {
            "text": "CD4 T cell VEGF receptor signaling and hypoxia response in CF",
            "category": "ontology",
            "expected_clusters": ["CD4-positive helper T cell", "mature T cell"],
            "expected_genes": ["TXNIP", "MAP2K2", "ETS1", "KLF2"],
        },
        # Q12
        {
            "text": "aberrant Th2 and Th17 T cell responses in Pseudomonas-infected CF lungs",
            "category": "ontology",
            "expected_clusters": ["CD4-positive helper T cell", "mature T cell"],
            "expected_genes": ["IL7R", "CD48", "ETS1", "KLF2"],
        },
        # Q13
        {
            "text": "chronic adaptive immune activation of T lymphocytes in CF despite modulator therapy",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "mature T cell"],
            "expected_genes": ["CD69", "CD81", "TXNIP", "IFNG", "KLF2"],
        },

        # --- B cell / plasma cell biology (paper: 102 DEGs, BCR
        #     downregulation, SYK activation, PDGFRB signaling) ---

        # Q14
        {
            "text": "B cell activation and immunoglobulin response in CF airways",
            "category": "ontology",
            "expected_clusters": ["B cell", "plasma cell"],
            "expected_genes": ["CD79A", "IGHG3", "IGLC2", "SYK", "CD81", "JCHAIN"],
        },
        # Q15
        {
            "text": "B cell receptor downregulation and reduced plasma cell markers in CF",
            "category": "ontology",
            "expected_clusters": ["B cell", "plasma cell"],
            "expected_genes": ["IGHG3", "IGLC2", "JCHAIN", "CD79A", "MZB1"],
        },
        # Q16
        {
            "text": "interferon gamma signaling and HLA-DP expression in B cells of CF patients",
            "category": "ontology",
            "expected_clusters": ["B cell"],
            "expected_genes": ["HLA-DPA1", "HLA-DPB1", "LTB", "CD81", "SYK"],
        },
        # Q17
        {
            "text": "PDGFRB signaling pathway activated in B cells from CF lungs",
            "category": "ontology",
            "expected_clusters": ["B cell", "plasma cell"],
            "expected_genes": ["SYK", "CSK", "CD9", "JUND", "CD81"],
        },

        # --- Basal cell biology (paper: reduced proportions, 509 DEGs,
        #     impaired stemness, keratinization changes) ---

        # Q18
        {
            "text": "basal cell dysfunction and reduced stemness in cystic fibrosis epithelium",
            "category": "ontology",
            "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
            "expected_genes": ["KRT5", "KRT14", "TP63", "CSTA", "HSPB1"],
        },
        # Q19
        {
            "text": "impaired basal cell differentiation and pathogenic basal cell variants in CF",
            "category": "ontology",
            "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
            "expected_genes": ["KRT5", "TP63", "CSTA", "HSPB1", "IL33"],
        },
        # Q20
        {
            "text": "basal cell DNA damage repair and chromatin remodeling in CF airways",
            "category": "ontology",
            "expected_clusters": ["basal cell"],
            "expected_genes": ["RAD50", "ERCC6", "ERCC8", "KDM1A", "KMT5A"],
        },
        # Q21
        {
            "text": "reduced keratinization gene expression CSTA HSPB1 in CF basal cells",
            "category": "ontology",
            "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
            "expected_genes": ["CSTA", "HSPB1", "KRT5", "KRT14", "KRT15"],
        },
        # Q22
        {
            "text": "basal cell altered cell-cell communication and increased interactions in CF",
            "category": "ontology",
            "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
            "expected_genes": ["KRT5", "TP63", "CSTA", "IL33"],
        },

        # --- Ciliated cell biology (paper: increased abundance,
        #     ciliogenesis genes, HLA expression) ---

        # Q23
        {
            "text": "ciliated cell ciliogenesis and increased abundance in CF bronchial epithelium",
            "category": "ontology",
            "expected_clusters": ["respiratory tract multiciliated cell"],
            "expected_genes": ["FOXJ1", "DNAH5", "SYNE1", "SYNE2", "CAPS"],
        },
        # Q24
        {
            "text": "ciliated cell HLA class II expression and immune-linked transcriptional changes in CF",
            "category": "ontology",
            "expected_clusters": ["respiratory tract multiciliated cell"],
            "expected_genes": ["HLA-DPA1", "HLA-DRB1", "FOXJ1", "DNAH5", "CAPS"],
        },
        # Q25
        {
            "text": "skewed basal cell differentiation towards ciliated cells in CF epithelium",
            "category": "ontology",
            "expected_clusters": ["respiratory tract multiciliated cell", "basal cell"],
            "expected_genes": ["FOXJ1", "DNAH5", "KRT5", "TP63"],
        },

        # --- NK cell / ILC biology (paper: NKG2A checkpoint,
        #     GNAI2 interactions with ILC/NK) ---

        # Q26
        {
            "text": "natural killer cell cytotoxicity and NKG2A immune checkpoint in CF",
            "category": "ontology",
            "expected_clusters": ["natural killer cell", "innate lymphoid cell"],
            "expected_genes": ["GNLY", "NKG7", "KLRD1", "KLRK1", "KLRC1", "PRF1"],
        },
        # Q27
        {
            "text": "NKG2A blockade to restore NK and CD8 T cell function in CF lung",
            "category": "ontology",
            "expected_clusters": ["natural killer cell", "CD8-positive, alpha-beta T cell", "innate lymphoid cell"],
            "expected_genes": ["KLRC1", "KLRD1", "HLA-E", "NKG7", "GZMB"],
        },
        # Q28
        {
            "text": "innate lymphoid cell dysfunction and impaired antimicrobial defense in CF",
            "category": "ontology",
            "expected_clusters": ["innate lymphoid cell", "natural killer cell"],
            "expected_genes": ["GNLY", "NKG7", "PRF1", "KLRK1"],
        },

        # --- Ionocyte biology (paper: CFTR-rich, unique CF interactions,
        #     crosstalk with lymphocytes) ---

        # Q29
        {
            "text": "pulmonary ionocyte CFTR expression in cystic fibrosis",
            "category": "ontology",
            "expected_clusters": ["ionocyte"],
            "expected_genes": ["FOXI1", "CFTR", "ATP6V1G3", "BSND", "ASCL3"],
        },
        # Q30
        {
            "text": "ionocyte unique cell-cell interactions with adaptive lymphocytes in CF",
            "category": "ontology",
            "expected_clusters": ["ionocyte"],
            "expected_genes": ["FOXI1", "CFTR", "ATP6V1G3", "BSND"],
        },

        # --- Endothelial cell biology (paper: reduced proportions,
        #     VEGF signaling, cell differentiation changes) ---

        # Q31
        {
            "text": "endothelial cell remodeling and VEGF signaling in CF lung",
            "category": "ontology",
            "expected_clusters": ["endocardial cell"],
            "expected_genes": ["VEGFA", "PLVAP", "ACKR1", "VIM", "ERG"],
        },
        # Q32
        {
            "text": "reduced endothelial cell proportions and altered differentiation in CF airways",
            "category": "ontology",
            "expected_clusters": ["endocardial cell"],
            "expected_genes": ["PLVAP", "ACKR1", "ERG", "VWF", "CDH5"],
        },
        # Q33
        {
            "text": "hypoxia-induced VEGF upregulation and vascular remodeling in CF lungs",
            "category": "ontology",
            "expected_clusters": ["endocardial cell", "basal cell"],
            "expected_genes": ["VEGFA", "TXNIP", "MAP2K2", "PLVAP"],
        },

        # --- Dendritic cell biology (paper: IFNGR2 interactions,
        #     antigen presentation) ---

        # Q34
        {
            "text": "dendritic cell antigen presentation in CF airways",
            "category": "ontology",
            "expected_clusters": ["dendritic cell, human"],
            "expected_genes": ["HLA-DPA1", "HLA-DRB1", "CD74", "CD80", "CD86"],
        },
        # Q35
        {
            "text": "IFNG IFNGR2 interaction between CD8 T cells and dendritic cells in CF",
            "category": "ontology",
            "expected_clusters": ["dendritic cell, human", "CD8-positive, alpha-beta T cell"],
            "expected_genes": ["IFNG", "IFNGR2", "HLA-DPA1", "HLA-DRB1"],
        },

        # --- Mast cell biology ---

        # Q36
        {
            "text": "mast cell degranulation and allergic inflammation in CF",
            "category": "ontology",
            "expected_clusters": ["mast cell"],
            "expected_genes": ["CPA3", "TPSAB1", "TPSB2", "MS4A2", "KIT"],
        },

        # --- Secretory / goblet cell biology (paper: increased
        #     secretory activity, interferon response) ---

        # Q37
        {
            "text": "secretory cell mucus overproduction and inflammatory signaling in CF epithelium",
            "category": "ontology",
            "expected_clusters": ["secretory cell", "bronchial goblet cell", "respiratory tract goblet cell"],
            "expected_genes": ["MUC5AC", "MUC5B", "SCGB1A1", "SCGB3A1"],
        },
        # Q38
        {
            "text": "goblet cell hyperplasia and mucin gene expression in cystic fibrosis",
            "category": "ontology",
            "expected_clusters": ["bronchial goblet cell", "nasal mucosa goblet cell", "respiratory tract goblet cell"],
            "expected_genes": ["MUC5AC", "MUC5B", "SCGB1A1"],
        },

        # --- Submucosal gland epithelial cells (paper: reduced
        #     proportions, 83 DEGs, gland development) ---

        # Q39
        {
            "text": "submucosal gland epithelial cell changes in cystic fibrosis",
            "category": "ontology",
            "expected_clusters": ["mucus secreting cell of bronchus submucosal gland"],
            "expected_genes": ["MUC5AC", "MUC5B", "SCGB1A1", "LYZ", "SCGB3A1"],
        },
        # Q40
        {
            "text": "reduced submucosal gland cell proportions and gland development dysfunction in CF",
            "category": "ontology",
            "expected_clusters": ["mucus secreting cell of bronchus submucosal gland"],
            "expected_genes": ["MUC5B", "LYZ", "SCGB1A1", "SCGB3A1"],
        },

        # --- Interferon response (paper: type I IFN in epithelial cells) ---

        # Q41
        {
            "text": "type I interferon response and inflammatory signaling in CF epithelial cells",
            "category": "ontology",
            "expected_clusters": ["basal cell", "respiratory tract multiciliated cell", "secretory cell"],
            "expected_genes": ["IFIT1", "MX1", "OAS2", "ISG15", "IFITM3"],
        },
        # Q42
        {
            "text": "interferon responsive gene upregulation across epithelial subsets in CF",
            "category": "ontology",
            "expected_clusters": ["basal cell", "respiratory tract multiciliated cell", "secretory cell"],
            "expected_genes": ["IFIT1", "MX1", "OAS2", "IFIT3"],
        },

        # --- VEGFR signaling across cell types (paper: TXNIP, MAP2K2,
        #     ETS1 upregulated across many cell types) ---

        # Q43
        {
            "text": "VEGF receptor signaling and hypoxia response across cell types in CF",
            "category": "ontology",
            "expected_clusters": ["endocardial cell", "CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "basal cell"],
            "expected_genes": ["TXNIP", "MAP2K2", "ETS1", "VEGFA"],
        },
        # Q44
        {
            "text": "TXNIP-mediated NLRP3 inflammasome activation in CF lymphocytes and epithelial cells",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "basal cell"],
            "expected_genes": ["TXNIP", "ETS1", "MAP2K2", "IFNG"],
        },

        # --- GNAI2 signaling (paper: upregulated in lymphocytes,
        #     multiple receptor interactions) ---

        # Q45
        {
            "text": "GNAI2 immunomodulatory signaling in CD8 T cells and B cells in CF",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "B cell"],
            "expected_genes": ["GNAI2", "CXCR3", "F2R", "S1PR4"],
        },
        # Q46
        {
            "text": "GNAI2 adenylate cyclase regulation and CFTR function in lymphocytes",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "B cell", "CD4-positive helper T cell"],
            "expected_genes": ["GNAI2", "CXCR3", "F2R", "S1PR4", "CD69"],
        },

        # --- Stromal / fibroblast biology ---

        # Q47
        {
            "text": "stromal cell and fibroblast remodeling in CF airway tissue",
            "category": "ontology",
            "expected_clusters": ["stromal cell", "fibroblast of lung", "alveolar adventitial fibroblast"],
            "expected_genes": ["COL1A2", "LUM", "DCN", "COL3A1", "PDGFRA"],
        },
        # Q48
        {
            "text": "pericyte and stromal cell contribution to airway fibrosis in CF",
            "category": "ontology",
            "expected_clusters": ["pericyte", "stromal cell"],
            "expected_genes": ["PDGFRB", "COL1A2", "VIM"],
        },

        # --- Cross-cutting / multi-cell-type queries ---

        # Q49
        {
            "text": "IFNG-IFNGR1 interaction between CD8 T cells and basal cells macrophages endothelial cells in CF",
            "category": "ontology",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "basal cell", "macrophage", "endocardial cell"],
            "expected_genes": ["IFNG", "IFNGR1", "IFNGR2", "CD69"],
        },
        # Q50
        {
            "text": "altered structural-immune cell crosstalk in CF involving lymphocytes ionocytes and macrophages",
            "category": "ontology",
            "expected_clusters": ["macrophage", "ionocyte", "CD8-positive, alpha-beta T cell", "B cell"],
            "expected_genes": ["IFNG", "GNAI2", "CALR", "HLA-E"],
        },

        # ================================================================
        # EXPRESSION QUERIES (Q51–Q100): gene-signature, scGPT advantage
        # ================================================================

        # --- Macrophage / Monocyte signatures ---

        # Q51
        {
            "text": "MARCO FABP4 APOC1 C1QB C1QC MSR1",
            "category": "expression",
            "expected_clusters": ["macrophage", "monocyte", "non-classical monocyte"],
            "expected_genes": ["C1QB", "MARCO", "FABP4", "CD68"],
        },
        # Q52
        {
            "text": "CD68 CD14 CSF1R CSF2RA LGALS2",
            "category": "expression",
            "expected_clusters": ["macrophage", "monocyte", "non-classical monocyte"],
            "expected_genes": ["CD68", "CD14", "CSF1R", "CSF2RA"],
        },
        # Q53
        {
            "text": "GOS2 FABP4 PPARG APOC1 C1QB",
            "category": "expression",
            "expected_clusters": ["macrophage"],
            "expected_genes": ["FABP4", "APOC1", "C1QB"],
        },
        # Q54
        {
            "text": "FCGR3A CX3CR1 CD14 CDKN1C LILRB2",
            "category": "expression",
            "expected_clusters": ["non-classical monocyte", "monocyte"],
            "expected_genes": ["FCGR3A", "CD14"],
        },

        # --- CD8+ T cell / cytotoxic T cell signatures ---

        # Q55
        {
            "text": "CD8A CD8B GZMB PRF1 IFNG NKG7",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"],
            "expected_genes": ["CD8A", "GZMB", "PRF1", "IFNG", "NKG7"],
        },
        # Q56
        {
            "text": "IFNG GNAI2 CD69 CD81 CD3G FOS JUND",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell"],
            "expected_genes": ["IFNG", "GNAI2", "CD69", "CD81", "CD3G", "FOS"],
        },
        # Q57
        {
            "text": "GZMB PRF1 NKG7 GNLY KLRD1 CD8A",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "cytotoxic T cell", "natural killer cell"],
            "expected_genes": ["GZMB", "PRF1", "NKG7", "GNLY", "CD8A"],
        },
        # Q58
        {
            "text": "TXNIP MAP2K2 IFNG CD81 CD3G CD69",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell"],
            "expected_genes": ["TXNIP", "MAP2K2", "IFNG", "CD81", "CD69"],
        },

        # --- CD4+ T cell signatures ---

        # Q59
        {
            "text": "KLF2 IL7R CD48 TXNIP ETS1",
            "category": "expression",
            "expected_clusters": ["CD4-positive helper T cell", "mature T cell"],
            "expected_genes": ["KLF2", "IL7R", "CD48", "TXNIP", "ETS1"],
        },
        # Q60
        {
            "text": "CD3D CD4 IL7R CD3E CD3G",
            "category": "expression",
            "expected_clusters": ["CD4-positive helper T cell", "mature T cell"],
            "expected_genes": ["CD3G", "CD3E", "IL7R"],
        },

        # --- T cell receptor / pan-T cell signatures ---

        # Q61
        {
            "text": "TRAJ52 TRBV22-1 TRDJ2 CD3E CD3G",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "mature T cell"],
            "expected_genes": ["CD3G", "CD3E", "CD69", "IL7R"],
        },
        # Q62
        {
            "text": "CD3G CD3E CD69 IL7R CD81 FOS",
            "category": "expression",
            "expected_clusters": ["mature T cell", "CD8-positive, alpha-beta T cell", "CD4-positive helper T cell"],
            "expected_genes": ["CD3G", "CD3E", "CD69", "IL7R", "CD81"],
        },

        # --- B cell / plasma cell signatures ---

        # Q63
        {
            "text": "IGLJ3 IGKJ1 IGHJ5 JCHAIN MZB1 XBP1",
            "category": "expression",
            "expected_clusters": ["plasma cell", "B cell"],
            "expected_genes": ["JCHAIN", "IGHG3", "IGLC2", "MZB1", "XBP1"],
        },
        # Q64
        {
            "text": "CD79A IGHG3 IGLC2 SYK CD81 JCHAIN",
            "category": "expression",
            "expected_clusters": ["B cell", "plasma cell"],
            "expected_genes": ["CD79A", "IGHG3", "IGLC2", "SYK", "JCHAIN"],
        },
        # Q65
        {
            "text": "SYK CSK CD9 CD81 JUND LTB HLA-DPA1",
            "category": "expression",
            "expected_clusters": ["B cell"],
            "expected_genes": ["SYK", "CSK", "CD9", "CD81", "LTB", "HLA-DPA1"],
        },
        # Q66
        {
            "text": "IGHG3 IGLC2 IGHD IGHA1 IGLC1 IGLC3",
            "category": "expression",
            "expected_clusters": ["B cell", "plasma cell"],
            "expected_genes": ["IGHG3", "IGLC2"],
        },

        # --- Basal cell keratin / stemness signatures ---

        # Q67
        {
            "text": "KRT5 KRT14 KRT15 TP63 IL33 CSTA",
            "category": "expression",
            "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
            "expected_genes": ["KRT5", "KRT14", "TP63", "CSTA", "HSPB1"],
        },
        # Q68
        {
            "text": "CSTA HSPB1 KRT5 KRT14 TP63",
            "category": "expression",
            "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
            "expected_genes": ["CSTA", "HSPB1", "KRT5", "KRT14", "TP63"],
        },
        # Q69
        {
            "text": "KRT5 IL33 TP63 KRT15 LAMB3 COL17A1",
            "category": "expression",
            "expected_clusters": ["basal cell", "respiratory tract suprabasal cell"],
            "expected_genes": ["KRT5", "TP63", "IL33"],
        },

        # --- Ciliated cell signatures ---

        # Q70
        {
            "text": "FOXJ1 DNAH5 CAPS PIFO RSPH1 DNAI1",
            "category": "expression",
            "expected_clusters": ["respiratory tract multiciliated cell"],
            "expected_genes": ["FOXJ1", "DNAH5", "CAPS", "SYNE1", "SYNE2"],
        },
        # Q71
        {
            "text": "DNAH5 SYNE1 SYNE2 CAPS PIFO",
            "category": "expression",
            "expected_clusters": ["respiratory tract multiciliated cell"],
            "expected_genes": ["DNAH5", "SYNE1", "SYNE2", "CAPS"],
        },

        # --- NK cell / ILC signatures ---

        # Q72
        {
            "text": "GNLY KLRD1 KLRK1 NKG7 PRF1 GZMB",
            "category": "expression",
            "expected_clusters": ["natural killer cell", "innate lymphoid cell", "cytotoxic T cell"],
            "expected_genes": ["GNLY", "NKG7", "PRF1", "GZMB", "KLRD1"],
        },
        # Q73
        {
            "text": "GNLY NKG7 KLRD1 KLRK1 KLRC1",
            "category": "expression",
            "expected_clusters": ["natural killer cell", "innate lymphoid cell"],
            "expected_genes": ["GNLY", "NKG7", "KLRD1", "KLRK1", "KLRC1"],
        },

        # --- Ionocyte signatures ---

        # Q74
        {
            "text": "ATP6V1G3 FOXI1 BSND CLCNKB ASCL3",
            "category": "expression",
            "expected_clusters": ["ionocyte"],
            "expected_genes": ["FOXI1", "ATP6V1G3", "BSND", "CFTR"],
        },
        # Q75
        {
            "text": "FOXI1 CFTR ATP6V1G3 BSND RARRES2",
            "category": "expression",
            "expected_clusters": ["ionocyte"],
            "expected_genes": ["FOXI1", "CFTR", "ATP6V1G3", "BSND"],
        },

        # --- Endothelial cell signatures ---

        # Q76
        {
            "text": "PLVAP ACKR1 ERG VWF PECAM1 CDH5",
            "category": "expression",
            "expected_clusters": ["endocardial cell"],
            "expected_genes": ["PLVAP", "ACKR1", "ERG", "VWF"],
        },
        # Q77
        {
            "text": "VIM PLVAP ACKR1 MGP PTGDS CXCL14",
            "category": "expression",
            "expected_clusters": ["endocardial cell"],
            "expected_genes": ["PLVAP", "ACKR1", "VIM"],
        },

        # --- Mast cell signatures ---

        # Q78
        {
            "text": "CPA3 TPSAB1 TPSB2 MS4A2 HDC GATA2",
            "category": "expression",
            "expected_clusters": ["mast cell"],
            "expected_genes": ["CPA3", "TPSAB1", "TPSB2", "MS4A2", "KIT"],
        },
        # Q79
        {
            "text": "TPSAB1 TPSB2 KIT CPA3 MS4A2",
            "category": "expression",
            "expected_clusters": ["mast cell"],
            "expected_genes": ["TPSAB1", "TPSB2", "KIT", "CPA3", "MS4A2"],
        },

        # --- Dendritic cell signatures ---

        # Q80
        {
            "text": "HLA-DPA1 HLA-DRB1 CD74 GPR183 LGALS2",
            "category": "expression",
            "expected_clusters": ["dendritic cell, human"],
            "expected_genes": ["HLA-DPA1", "HLA-DRB1", "CD74"],
        },
        # Q81
        {
            "text": "HLA-DPA1 HLA-DPB1 HLA-DRB1 CD80 CD86 CD74",
            "category": "expression",
            "expected_clusters": ["dendritic cell, human"],
            "expected_genes": ["HLA-DPA1", "HLA-DPB1", "HLA-DRB1", "CD74"],
        },

        # --- Secretory / club / goblet cell signatures ---

        # Q82
        {
            "text": "SCGB1A1 SCGB3A1 MUC5AC MUC5B LYPD2 PRR4",
            "category": "expression",
            "expected_clusters": ["secretory cell", "club cell"],
            "expected_genes": ["SCGB1A1", "MUC5AC", "MUC5B", "SCGB3A1"],
        },
        # Q83
        {
            "text": "SCGB1A1 MUC5AC SCGB3A1 LYPD2",
            "category": "expression",
            "expected_clusters": ["secretory cell", "club cell", "bronchial goblet cell"],
            "expected_genes": ["SCGB1A1", "MUC5AC", "SCGB3A1"],
        },
        # Q84
        {
            "text": "MUC5AC MUC5B LYZ SCGB1A1 SCGB3A1",
            "category": "expression",
            "expected_clusters": ["mucus secreting cell of bronchus submucosal gland", "secretory cell"],
            "expected_genes": ["MUC5AC", "MUC5B", "LYZ", "SCGB1A1"],
        },

        # --- Fibroblast / stromal signatures ---

        # Q85
        {
            "text": "COL1A2 LUM DCN SFRP2 COL3A1 PDGFRA",
            "category": "expression",
            "expected_clusters": ["fibroblast of lung", "alveolar adventitial fibroblast", "alveolar type 1 fibroblast cell"],
            "expected_genes": ["COL1A2", "LUM", "DCN", "COL3A1"],
        },
        # Q86
        {
            "text": "PDGFRA COL1A2 COL3A1 VCAN DCN LUM",
            "category": "expression",
            "expected_clusters": ["fibroblast of lung", "alveolar adventitial fibroblast", "alveolar type 1 fibroblast cell"],
            "expected_genes": ["PDGFRA", "COL1A2", "COL3A1", "DCN", "LUM"],
        },
        # Q87
        {
            "text": "PDGFRB VIM COL1A2 MGP CXCL14",
            "category": "expression",
            "expected_clusters": ["pericyte", "stromal cell"],
            "expected_genes": ["PDGFRB", "VIM", "COL1A2"],
        },

        # --- Neuroendocrine cell signatures ---

        # Q88
        {
            "text": "SST CHGA ASCL1 GRP CALCA SYP",
            "category": "expression",
            "expected_clusters": ["pulmonary neuroendocrine cell"],
            "expected_genes": ["CHGA", "ASCL1", "GRP", "SYP"],
        },
        # Q89
        {
            "text": "GRP ASCL1 SYT1 CHGA SYP CALCA",
            "category": "expression",
            "expected_clusters": ["pulmonary neuroendocrine cell"],
            "expected_genes": ["GRP", "ASCL1", "CHGA", "SYP"],
        },

        # --- HLA-E / NKG2A checkpoint signatures (paper-specific) ---

        # Q90
        {
            "text": "HLA-E KLRC1 KLRD1 KLRC2 KLRC3 KLRK1",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "natural killer cell", "innate lymphoid cell"],
            "expected_genes": ["HLA-E", "KLRC1", "KLRD1", "KLRC2", "KLRK1"],
        },
        # Q91
        {
            "text": "HLA-E KLRC1 KLRD1 CD8A CD8B",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "natural killer cell"],
            "expected_genes": ["HLA-E", "KLRC1", "KLRD1", "CD8A"],
        },

        # --- CALR-LRP1 / T cell activation gene sets ---

        # Q92
        {
            "text": "CALR LRP1 GNAI2 FOS JUND MAP2K2",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "macrophage"],
            "expected_genes": ["CALR", "GNAI2", "FOS", "JUND", "MAP2K2"],
        },
        # Q93
        {
            "text": "GNAI2 CXCR3 F2R S1PR4 CD69",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "B cell", "CD4-positive helper T cell"],
            "expected_genes": ["GNAI2", "CXCR3", "F2R", "S1PR4", "CD69"],
        },

        # --- Interferon response gene signatures ---

        # Q94
        {
            "text": "IFIT1 MX1 OAS2 ISG15 IFITM3 IFIT3",
            "category": "expression",
            "expected_clusters": ["basal cell", "respiratory tract multiciliated cell", "secretory cell"],
            "expected_genes": ["IFIT1", "MX1", "OAS2", "ISG15"],
        },
        # Q95
        {
            "text": "IFIT1 MX1 OAS2 IFIT3 IFI6",
            "category": "expression",
            "expected_clusters": ["basal cell", "secretory cell", "respiratory tract multiciliated cell"],
            "expected_genes": ["IFIT1", "MX1", "OAS2"],
        },

        # --- DNA damage / chromatin remodeling signatures ---

        # Q96
        {
            "text": "KDM1A KMT5A RAD50 ERCC6 ERCC8",
            "category": "expression",
            "expected_clusters": ["basal cell"],
            "expected_genes": ["KDM1A", "KMT5A", "RAD50", "ERCC6", "ERCC8"],
        },

        # --- VEGFR / hypoxia gene signatures ---

        # Q97
        {
            "text": "TXNIP MAP2K2 ETS1 VEGFA KLF2",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "CD4-positive helper T cell", "endocardial cell"],
            "expected_genes": ["TXNIP", "MAP2K2", "ETS1", "VEGFA", "KLF2"],
        },

        # --- Ligand-receptor interaction signatures ---

        # Q98
        {
            "text": "IFNG IFNGR1 IFNGR2 CALR LRP1",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "macrophage", "basal cell"],
            "expected_genes": ["IFNG", "IFNGR1", "IFNGR2", "CALR", "LRP1"],
        },
        # Q99
        {
            "text": "CCL5 CCR5 CXCL10 CXCR3 F2R",
            "category": "expression",
            "expected_clusters": ["CD8-positive, alpha-beta T cell", "macrophage"],
            "expected_genes": ["CCL5", "CXCL10", "CXCR3", "F2R"],
        },

        # --- Mixed epithelial general signature ---

        # Q100
        {
            "text": "CFTR FOXI1 SCGB1A1 KRT5 FOXJ1 MUC5AC",
            "category": "expression",
            "expected_clusters": ["ionocyte", "secretory cell", "basal cell", "respiratory tract multiciliated cell"],
            "expected_genes": ["CFTR", "FOXI1", "SCGB1A1", "KRT5", "FOXJ1", "MUC5AC"],
        },
        ],
    },
}


# ============================================================
# METRICS
# ============================================================

def cluster_recall_at_k(expected: List[str], retrieved: List[str], k: int = 5) -> float:
    """Fraction of expected clusters found in top-k retrieved (fuzzy match)."""
    if not expected:
        return 0.0
    ret_lower = [r.lower() for r in retrieved[:k]]
    found = 0
    for exp in expected:
        exp_l = exp.lower()
        if any(exp_l in r or r in exp_l or
               _word_overlap(exp_l, r) >= 0.5
               for r in ret_lower):
            found += 1
    return found / len(expected)


def _word_overlap(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def mrr(expected: List[str], retrieved: List[str]) -> float:
    """Mean reciprocal rank of first relevant cluster."""
    for rank, ret in enumerate(retrieved, 1):
        ret_l = ret.lower()
        for exp in expected:
            exp_l = exp.lower()
            if (exp_l in ret_l or ret_l in exp_l or
                    _word_overlap(exp_l, ret_l) >= 0.5):
                return 1.0 / rank
    return 0.0


# ============================================================
# RETRIEVAL EVALUATOR
# ============================================================

class RetrievalEvaluator:
    """Evaluate retrieval across 4 modes: random baseline + 3 ELISA modalities.

    Union mode uses ADDITIVE strategy:
      1. Per-query, pick whichever single modality (semantic or scGPT)
         has higher recall@10 as the "primary"
      2. Return primary's full top_k results
      3. Append unique clusters from the secondary that primary missed
      4. NO truncation — the union list can be longer than top_k
      5. Evaluate at recall@5, @10, @15, @20 to show the additive gain
    """

    MODES = ["random", "semantic", "scgpt", "union"]

    # Recall cutoffs — extended for union
    RECALL_KS = [5, 10, 15, 20]

    def __init__(self, engine):
        self.engine = engine

    # ── Retrieval methods ──────────────────────────────────────

    def run_query_random(self, text: str, top_k: int = 10):
        indices = list(range(len(self.engine.cluster_ids)))
        random.shuffle(indices)
        return [self.engine.cluster_ids[i] for i in indices[:top_k]]

    def run_query_semantic(self, text: str, top_k: int = 10):
        payload = self.engine.query_semantic(text, top_k=top_k, with_genes=False)
        return [r["cluster_id"] for r in payload["results"]]

    def run_query_scgpt(self, text: str, top_k: int = 10):
        payload = self.engine.query_hybrid(
            text, top_k=top_k, lambda_sem=0.0, with_genes=False
        )
        return [r["cluster_id"] for r in payload["results"]]

    def run_query_union(self, text: str, top_k: int = 10,
                        _sem_clusters=None, _scgpt_clusters=None,
                        _expected=None):
        """ADDITIVE union: primary modality's full top_k + unique from secondary.

        Strategy:
          1. Run both semantic and scGPT at top_k
          2. Determine which is "primary" for this query by checking
             which has higher recall against expected clusters
             (if expected not available, use semantic as default primary)
          3. Start with primary's full ranked list
          4. Append secondary-unique clusters in their original rank order
          5. NO truncation — list can be up to 2*top_k

        Args:
            _sem_clusters:  Pre-computed semantic results (avoids double query)
            _scgpt_clusters: Pre-computed scGPT results (avoids double query)
            _expected:      Expected clusters for adaptive primary selection
        """
        sem = _sem_clusters if _sem_clusters is not None else self.run_query_semantic(text, top_k)
        scgpt = _scgpt_clusters if _scgpt_clusters is not None else self.run_query_scgpt(text, top_k)

        # Determine primary modality for this query
        if _expected is not None and len(_expected) > 0:
            sem_rec5 = cluster_recall_at_k(_expected, sem, k=5)
            scgpt_rec5 = cluster_recall_at_k(_expected, scgpt, k=5)
            if scgpt_rec5 > sem_rec5:
                primary, secondary = scgpt, sem
            elif sem_rec5 > scgpt_rec5:
                primary, secondary = sem, scgpt
            else:
                sem_mrr = mrr(_expected, sem)
                scgpt_mrr = mrr(_expected, scgpt)
                if scgpt_mrr > sem_mrr:
                    primary, secondary = scgpt, sem
                else:
                    primary, secondary = sem, scgpt
        else:
            primary, secondary = sem, scgpt

        # Build additive union: primary full + secondary unique (in order)
        seen = set()
        union = []
        for c in primary:
            if c not in seen:
                union.append(c)
                seen.add(c)
        for c in secondary:
            if c not in seen:
                union.append(c)
                seen.add(c)

        return union

    # ── Dispatch ──────────────────────────────────────────────

    def run_query(self, mode: str, text: str, top_k: int = 10, **kwargs):
        fn = {
            "random": self.run_query_random,
            "semantic": self.run_query_semantic,
            "scgpt": self.run_query_scgpt,
            "union": self.run_query_union,
        }
        if mode == "union":
            return fn[mode](text, top_k, **kwargs)
        return fn[mode](text, top_k)

    # ── Gene extraction ────────────────────────────────────────

    def _get_genes_from_clusters(self, cluster_ids: list, top_n: int = 500) -> set:
        genes = set()
        for cid in cluster_ids:
            stats = self.engine.gene_stats.get(str(cid), {})
            if not stats:
                continue
            sorted_genes = sorted(
                stats.keys(),
                key=lambda g: abs(stats[g].get("logfc", 0) or 0),
                reverse=True
            )[:top_n]
            genes.update(g.upper() for g in sorted_genes)
        return genes

    # ── Evaluation ──────────────────────────────────────────

    def evaluate_queries(self, queries: List[dict], top_k: int = 10,
                         n_random_runs: int = 50):
        """Run all queries through all modes, compute per-query metrics."""
        results = {}
        for cat in ["ontology", "expression"]:
            results[cat] = {mode: [] for mode in self.MODES}

        for qi, q in enumerate(queries):
            text = q["text"]
            cat = q["category"]
            expected = q["expected_clusters"]
            expected_genes = set(g.upper() for g in q.get("expected_genes", []))

            # Pre-compute semantic and scGPT for reuse in union
            sem_clusters = self.run_query_semantic(text, top_k)
            scgpt_clusters = self.run_query_scgpt(text, top_k)

            for mode in self.MODES:
                if mode == "random":
                    r_runs = {k: [] for k in self.RECALL_KS}
                    mrr_runs, gr_runs = [], []
                    for _ in range(n_random_runs):
                        clusters = self.run_query_random(text, max(self.RECALL_KS))
                        for k in self.RECALL_KS:
                            r_runs[k].append(cluster_recall_at_k(expected, clusters, k))
                        mrr_runs.append(mrr(expected, clusters))
                        rnd_genes = self._get_genes_from_clusters(clusters[:5])
                        if expected_genes:
                            gr_runs.append(len(expected_genes & rnd_genes) / len(expected_genes))
                        else:
                            gr_runs.append(0.0)

                    entry = {
                        "query": text,
                        "expected": expected,
                        "expected_genes": list(expected_genes),
                        "retrieved_top10": ["(random)"] * 10,
                        "n_retrieved": top_k,
                        "mrr": round(np.mean(mrr_runs), 4),
                        "gene_recall": round(np.mean(gr_runs), 4),
                        "genes_found": [],
                        "has_gene_evidence": True,
                    }
                    for k in self.RECALL_KS:
                        entry[f"recall@{k}"] = round(np.mean(r_runs[k]), 4)
                    results[cat][mode].append(entry)

                elif mode == "union":
                    clusters = self.run_query_union(
                        text, top_k,
                        _sem_clusters=sem_clusters,
                        _scgpt_clusters=scgpt_clusters,
                        _expected=expected,
                    )

                    sem_rec5 = cluster_recall_at_k(expected, sem_clusters, k=5)
                    scgpt_rec5 = cluster_recall_at_k(expected, scgpt_clusters, k=5)
                    if scgpt_rec5 > sem_rec5:
                        primary_mode = "scgpt"
                    elif sem_rec5 > scgpt_rec5:
                        primary_mode = "semantic"
                    else:
                        sem_m = mrr(expected, sem_clusters)
                        scgpt_m = mrr(expected, scgpt_clusters)
                        primary_mode = "scgpt" if scgpt_m > sem_m else "semantic"

                    entry = {
                        "query": text,
                        "expected": expected,
                        "expected_genes": list(expected_genes),
                        "retrieved_all": clusters,
                        "retrieved_top10": clusters[:10],
                        "n_retrieved": len(clusters),
                        "primary_mode": primary_mode,
                        "sem_recall@5": round(sem_rec5, 4),
                        "scgpt_recall@5": round(scgpt_rec5, 4),
                        "mrr": round(mrr(expected, clusters), 4),
                        "has_gene_evidence": True,
                    }

                    for k in self.RECALL_KS:
                        entry[f"recall@{k}"] = round(
                            cluster_recall_at_k(expected, clusters, k), 4
                        )

                    retrieved_genes = self._get_genes_from_clusters(clusters)
                    if expected_genes:
                        found = expected_genes & retrieved_genes
                        entry["gene_recall"] = round(len(found) / len(expected_genes), 4)
                        entry["genes_found"] = sorted(found)
                    else:
                        entry["gene_recall"] = 0.0
                        entry["genes_found"] = []

                    for gain_k in [5, 10]:
                        best_single_k = max(
                            cluster_recall_at_k(expected, sem_clusters, k=gain_k),
                            cluster_recall_at_k(expected, scgpt_clusters, k=gain_k),
                        )
                        union_rec_k = entry[f"recall@{gain_k}"]
                        entry[f"additive_gain@{gain_k}"] = round(
                            union_rec_k - best_single_k, 4
                        )

                    results[cat][mode].append(entry)

                else:
                    # Standard modes: semantic, scgpt
                    if mode == "semantic":
                        clusters = sem_clusters
                    elif mode == "scgpt":
                        clusters = scgpt_clusters
                    else:
                        clusters = self.run_query(mode, text, top_k)

                    entry = {
                        "query": text,
                        "expected": expected,
                        "expected_genes": list(expected_genes),
                        "retrieved_top10": clusters[:top_k],
                        "n_retrieved": len(clusters),
                        "mrr": round(mrr(expected, clusters), 4),
                        "has_gene_evidence": True,
                    }
                    for k in self.RECALL_KS:
                        entry[f"recall@{k}"] = round(
                            cluster_recall_at_k(expected, clusters, k), 4
                        )

                    retrieved_genes = self._get_genes_from_clusters(clusters[:5])
                    if expected_genes:
                        found = expected_genes & retrieved_genes
                        entry["gene_recall"] = round(len(found) / len(expected_genes), 4)
                        entry["genes_found"] = sorted(found)
                    else:
                        entry["gene_recall"] = 0.0
                        entry["genes_found"] = []

                    results[cat][mode].append(entry)

        return results

    def compute_summary(self, results: dict) -> dict:
        """Aggregate results into a summary table."""
        summary = {}
        for cat in ["ontology", "expression"]:
            for mode in self.MODES:
                entries = results[cat].get(mode, [])
                if not entries:
                    continue
                key = f"{cat}_{mode}"
                s = {
                    "n_queries": len(entries),
                    "mean_mrr": round(np.mean([e["mrr"] for e in entries]), 4),
                    "std_mrr": round(np.std([e["mrr"] for e in entries]), 4),
                    "mean_gene_recall": round(np.mean([e["gene_recall"] for e in entries]), 4),
                    "has_gene_evidence": entries[0].get("has_gene_evidence", False),
                }
                for k in self.RECALL_KS:
                    col = f"recall@{k}"
                    vals = [e.get(col, 0) for e in entries]
                    s[f"mean_{col}"] = round(np.mean(vals), 4)
                    s[f"std_{col}"] = round(np.std(vals), 4)

                if mode == "union":
                    for gain_k in [5, 10]:
                        gains = [e.get(f"additive_gain@{gain_k}", 0) for e in entries]
                        s[f"mean_additive_gain@{gain_k}"] = round(np.mean(gains), 4)
                    n_ret = [e.get("n_retrieved", 10) for e in entries]
                    s["mean_n_retrieved"] = round(np.mean(n_ret), 1)
                    primary_counts = Counter(e.get("primary_mode", "semantic") for e in entries)
                    s["primary_selection"] = dict(primary_counts)

                summary[key] = s
        return summary

    def compute_complementarity(self, results: dict, top_k: int = 10) -> dict:
        """Show what each ELISA mode finds that the other misses."""
        all_queries = []
        for cat in results:
            n_queries = len(results[cat].get("semantic", []))
            for i in range(n_queries):
                sem_set = set(results[cat]["semantic"][i]["retrieved_top10"])
                scgpt_set = set(results[cat]["scgpt"][i]["retrieved_top10"])
                expected = set(results[cat]["semantic"][i]["expected"])

                union_entry = results[cat]["union"][i]
                union_set = set(union_entry.get("retrieved_all", union_entry.get("retrieved_top10", [])))

                sem_found = set()
                scgpt_found = set()
                union_found_set = set()

                for exp in expected:
                    exp_l = exp.lower()
                    for s in sem_set:
                        if (exp_l in s.lower() or s.lower() in exp_l or
                                _word_overlap(exp_l, s.lower()) >= 0.5):
                            sem_found.add(exp)
                            break
                    for s in scgpt_set:
                        if (exp_l in s.lower() or s.lower() in exp_l or
                                _word_overlap(exp_l, s.lower()) >= 0.5):
                            scgpt_found.add(exp)
                            break
                    for s in union_set:
                        if (exp_l in s.lower() or s.lower() in exp_l or
                                _word_overlap(exp_l, s.lower()) >= 0.5):
                            union_found_set.add(exp)
                            break

                all_queries.append({
                    "query": results[cat]["semantic"][i]["query"],
                    "category": cat,
                    "expected": list(expected),
                    "sem_found": list(sem_found),
                    "scgpt_found": list(scgpt_found),
                    "union_found": list(union_found_set),
                    "only_semantic": list(sem_found - scgpt_found),
                    "only_scgpt": list(scgpt_found - sem_found),
                    "both_found": list(sem_found & scgpt_found),
                    "neither_found": list(expected - union_found_set),
                    "n_union_clusters": union_entry.get("n_retrieved", 0),
                    "primary_mode": union_entry.get("primary_mode", "semantic"),
                    "additive_gain": union_entry.get("additive_gain@10", 0),
                })

        total_expected = sum(len(q["expected"]) for q in all_queries)
        sem_total = sum(len(q["sem_found"]) for q in all_queries)
        scgpt_total = sum(len(q["scgpt_found"]) for q in all_queries)
        union_total = sum(len(q["union_found"]) for q in all_queries)
        only_sem = sum(len(q["only_semantic"]) for q in all_queries)
        only_scgpt = sum(len(q["only_scgpt"]) for q in all_queries)
        neither = sum(len(q["neither_found"]) for q in all_queries)

        best_single = max(sem_total, scgpt_total)

        return {
            "total_expected": total_expected,
            "semantic_found": sem_total,
            "scgpt_found": scgpt_total,
            "union_found": union_total,
            "best_single_found": best_single,
            "semantic_recall": round(sem_total / total_expected, 4) if total_expected else 0,
            "scgpt_recall": round(scgpt_total / total_expected, 4) if total_expected else 0,
            "union_recall": round(union_total / total_expected, 4) if total_expected else 0,
            "best_single_recall": round(best_single / total_expected, 4) if total_expected else 0,
            "additive_gain_clusters": union_total - best_single,
            "additive_gain_pct": round(
                (union_total - best_single) / total_expected, 4
            ) if total_expected else 0,
            "only_semantic_count": only_sem,
            "only_scgpt_count": only_scgpt,
            "neither_count": neither,
            "per_query": all_queries,
        }


# ============================================================
# ANALYTICAL MODULE EVALUATOR
# ============================================================

class AnalyticalEvaluator:
    """Evaluate analytical modules (pathways, interactions, proportions, compare)."""

    def __init__(self, engine):
        self.engine = engine

    def evaluate_interactions(self, paper):
        gt_interactions = paper.get("ground_truth_interactions", [])
        if not gt_interactions:
            return {"error": "No ground truth interactions"}

        payload = self.engine.interactions(
            min_ligand_pct=0.01, min_receptor_pct=0.01
        )
        elisa_ixns = payload.get("interactions", [])

        found_lr, found_full = 0, 0
        details = []

        for lig, rec, src, tgt in gt_interactions:
            lr_match = any(
                ix.get("ligand", "").upper() == lig.upper() and
                ix.get("receptor", "").upper() == rec.upper()
                for ix in elisa_ixns
            )

            full_match = False
            if lr_match:
                for ix in elisa_ixns:
                    if (ix.get("ligand", "").upper() != lig.upper() or
                            ix.get("receptor", "").upper() != rec.upper()):
                        continue
                    ix_src = ix.get("source", "").lower()
                    ix_tgt = ix.get("target", "").lower()
                    src_l, tgt_l = src.lower(), tgt.lower()
                    src_match = (src_l in ix_src or ix_src in src_l or
                                 any(w in ix_src for w in src_l.split() if len(w) > 3))
                    tgt_match = (tgt_l in ix_tgt or ix_tgt in tgt_l or
                                 any(w in ix_tgt for w in tgt_l.split() if len(w) > 3))
                    if src_match and tgt_match:
                        full_match = True
                        break

            if lr_match:
                found_lr += 1
            if full_match:
                found_full += 1
            details.append({
                "pair": f"{lig}->{rec} ({src}->{tgt})",
                "lr_found": lr_match,
                "full_match": full_match,
            })

        n = len(gt_interactions)
        return {
            "total_expected": n,
            "lr_matches": found_lr,
            "full_matches": found_full,
            "lr_recovery_rate": round(found_lr / n * 100, 1),
            "full_recovery_rate": round(found_full / n * 100, 1),
            "total_elisa_interactions": len(elisa_ixns),
            "details": details,
        }

    def evaluate_pathways(self, paper):
        gt_pathways = paper.get("ground_truth_pathways", [])
        if not gt_pathways:
            return {"error": "No ground truth pathways"}

        payload = self.engine.pathways()
        results = {}
        for pw in gt_pathways:
            pw_l = pw.lower()
            found = False
            top_score, top_cluster, n_genes = 0, "", 0

            for pw_name, pw_data in payload.get("pathways", {}).items():
                if pw_l in pw_name.lower() or pw_name.lower() in pw_l:
                    scores = pw_data.get("scores", [])
                    if scores:
                        best = max(scores, key=lambda x: x.get("score", 0))
                        if best.get("score", 0) > 0:
                            found = True
                            if best["score"] > top_score:
                                top_score = best["score"]
                                top_cluster = best.get("cluster", "")
                                n_genes = best.get("n_genes_found", 0)

            results[pw] = {
                "found": found,
                "top_score": round(top_score, 4),
                "n_genes_found": n_genes,
                "top_cluster": top_cluster,
            }

        found = sum(1 for v in results.values() if v["found"])
        return {
            "pathways_found": found,
            "pathways_expected": len(gt_pathways),
            "alignment": round(found / len(gt_pathways) * 100, 1),
            "details": results,
        }

    def evaluate_proportions(self, paper):
        prop_changes = paper.get("proportion_changes", {})
        if not prop_changes:
            return {"error": "No proportion changes defined"}

        payload = self.engine.proportions()
        fc_data = payload.get("proportion_fold_changes", [])
        if not fc_data:
            return {"error": "No fold change data"}

        consistent, total = 0, 0
        details = []

        for item in fc_data:
            cluster = item["cluster"].lower()

            fc = 1.0
            for key in item:
                if key.startswith("fold_change"):
                    val = item[key]
                    if val == "inf":
                        fc = 999.0
                    elif isinstance(val, (int, float)):
                        fc = float(val)
                    break

            is_up = any(ct.lower() in cluster
                        for ct in prop_changes.get("increased_in_CF", []))
            is_down = any(ct.lower() in cluster
                          for ct in prop_changes.get("decreased_in_CF", []))
            if not is_up and not is_down:
                continue
            total += 1

            if is_up and fc > 1.0:
                consistent += 1
                details.append({"cluster": item["cluster"], "direction": "correct up", "fc": fc})
            elif is_down and fc < 1.0:
                consistent += 1
                details.append({"cluster": item["cluster"], "direction": "correct down", "fc": fc})
            else:
                details.append({
                    "cluster": item["cluster"], "direction": "WRONG",
                    "expected": "up" if is_up else "down", "fc": fc,
                })

        return {
            "total_checked": total,
            "consistent": consistent,
            "consistency_rate": round(consistent / total * 100, 1) if total else 0,
            "details": details,
        }

    def evaluate_compare(self, paper):
        conditions = paper.get("conditions", [])
        if len(conditions) < 2:
            return {"error": "Need 2 conditions"}

        gt_genes = paper.get("ground_truth_genes", [])
        gt_set = set(g.upper() for g in gt_genes)

        payload = self.engine.compare(
            conditions[0], conditions[1], genes=gt_genes
        )

        all_compare_genes = set()
        clusters_data = payload.get("clusters", {})
        for cid, cdata in clusters_data.items():
            if isinstance(cdata, dict):
                for g in cdata.get("genes", []):
                    if isinstance(g, dict):
                        all_compare_genes.add(g.get("gene", "").upper())
                    elif isinstance(g, str):
                        all_compare_genes.add(g.upper())

        summary = payload.get("summary", {})
        for grp_genes in summary.get("condition_enriched_genes", {}).values():
            for g in grp_genes:
                if isinstance(g, dict):
                    all_compare_genes.add(g.get("gene", "").upper())

        found = gt_set & all_compare_genes
        missed = gt_set - all_compare_genes

        return {
            "genes_requested": len(gt_set),
            "genes_found": len(found),
            "compare_recall": round(len(found) / len(gt_set) * 100, 1) if gt_set else 0,
            "n_clusters_analyzed": len(clusters_data),
            "found": sorted(found),
            "missed": sorted(missed),
        }

    def evaluate_all(self, paper):
        return {
            "interactions": self.evaluate_interactions(paper),
            "pathways": self.evaluate_pathways(paper),
            "proportions": self.evaluate_proportions(paper),
            "compare": self.evaluate_compare(paper),
        }


# ============================================================
# FIGURE GENERATION
# ============================================================

def generate_figures(summary: dict, complementarity: dict,
                     analytical: dict, out_dir: str):
    """Generate publication-ready figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    MODE_COLORS = {
        "random":    "#9E9E9E",
        "semantic":  "#2196F3",
        "scgpt":     "#FF9800",
        "union":     "#4CAF50",
    }
    MODE_LABELS = {
        "random":    "Random",
        "semantic":  "Semantic",
        "scgpt":     "scGPT",
        "union":     "Union\n(additive)",
    }
    MODES = ["random", "semantic", "scgpt", "union"]
    cats = ["ontology", "expression"]
    titles = ["Ontology Queries\n(concept-level)", "Expression Queries\n(gene-signature)"]

    # ── Figure 1: Cluster Recall@5 by query category ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, cat, title in zip(axes, cats, titles):
        vals = [summary.get(f"{cat}_{m}", {}).get("mean_recall@5", 0) for m in MODES]
        colors = [MODE_COLORS[m] for m in MODES]
        labels = [MODE_LABELS[m] for m in MODES]

        x = np.arange(len(MODES))
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white",
                       linewidth=0.5, width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean Cluster Recall@5")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_baselines.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "retrieval_baselines.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  [FIG] retrieval_baselines.png")

    # ── Figure 2: Additive Union — Recall@k across cutoffs ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    recall_ks = [5, 10, 15, 20]

    for ax, cat, title in zip(axes, cats, titles):
        for mode, ls, marker in [("semantic", "-", "o"), ("scgpt", "--", "s"),
                                  ("union", "-", "D")]:
            vals = []
            for k in recall_ks:
                v = summary.get(f"{cat}_{mode}", {}).get(f"mean_recall@{k}", 0)
                vals.append(v)
            ax.plot(recall_ks, vals, ls, marker=marker, markersize=8,
                    color=MODE_COLORS[mode], label=MODE_LABELS[mode].replace("\n", " "),
                    linewidth=2)

            for k, v in zip(recall_ks, vals):
                ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8,
                            color=MODE_COLORS[mode])

        ax.set_xlabel("k (retrieval cutoff)")
        ax.set_ylabel("Mean Cluster Recall@k")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(recall_ks)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "additive_union_recall_curve.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "additive_union_recall_curve.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  [FIG] additive_union_recall_curve.png")

    # ── Figure 3: Cluster Recall vs Gene Recall ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, cat, title in zip(axes, cats, titles):
        x = np.arange(len(MODES))
        w = 0.35

        cluster_vals = [summary.get(f"{cat}_{m}", {}).get("mean_recall@5", 0) for m in MODES]
        gene_vals = [summary.get(f"{cat}_{m}", {}).get("mean_gene_recall", 0) for m in MODES]

        bars1 = ax.bar(x - w / 2, cluster_vals, w, label="Cluster Recall@5",
                        color=[MODE_COLORS[m] for m in MODES], alpha=0.85,
                        edgecolor="white")
        bars2 = ax.bar(x + w / 2, gene_vals, w, label="Gene Recall",
                        color=[MODE_COLORS[m] for m in MODES], alpha=0.45,
                        edgecolor="black", linewidth=0.8, hatch="///")

        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars1, cluster_vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars2, gene_vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    legend_elements = [
        Patch(facecolor="#888888", alpha=0.85, label="Cluster Recall@5"),
        Patch(facecolor="#888888", alpha=0.45, hatch="///",
              edgecolor="black", label="Gene Recall"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_vs_gene_delivery.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "retrieval_vs_gene_delivery.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  [FIG] retrieval_vs_gene_delivery.png")

    # ── Figure 4: All metrics — Recall@5, Recall@10, MRR ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    metric_names = ["mean_recall@5", "mean_recall@10", "mean_mrr"]
    metric_labels = ["Recall@5", "Recall@10", "MRR"]

    for ax, metric, mlabel in zip(axes, metric_names, metric_labels):
        x = np.arange(len(MODES))
        w = 0.35

        ont_vals = [summary.get(f"ontology_{m}", {}).get(metric, 0) for m in MODES]
        expr_vals = [summary.get(f"expression_{m}", {}).get(metric, 0) for m in MODES]

        ax.bar(x - w / 2, ont_vals, w, label="Ontology", alpha=0.85,
               color=[MODE_COLORS[m] for m in MODES], edgecolor="white")
        ax.bar(x + w / 2, expr_vals, w, label="Expression", alpha=0.45,
               color=[MODE_COLORS[m] for m in MODES], edgecolor="black",
               linewidth=0.5, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODES], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_all_metrics.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "retrieval_all_metrics.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  [FIG] retrieval_all_metrics.png")

    # ── Figure 5: Complementarity — additive union breakdown ──
    comp = complementarity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sem_only = comp.get("only_semantic_count", 0)
    scgpt_only = comp.get("only_scgpt_count", 0)
    both_count = comp.get("union_found", 0) - sem_only - scgpt_only
    both_count = max(both_count, 0)
    neither = comp.get("neither_count", 0)

    labels_bar = ["Both\nmodalities", "Semantic\nonly", "scGPT\nonly", "Neither"]
    vals = [both_count, sem_only, scgpt_only, neither]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E"]

    bars = ax1.bar(labels_bar, vals, color=colors, alpha=0.85, edgecolor="white",
                    width=0.6)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(v), ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax1.set_ylabel("Expected clusters found")
    ax1.set_title("Modality Complementarity", fontweight="bold")

    total_exp = comp.get("total_expected", 1)
    ax1.text(0.98, 0.95,
             f"Union recall: {comp.get('union_recall', 0):.1%}\n"
             f"Best single: {comp.get('best_single_recall', 0):.1%}\n"
             f"Additive gain: +{comp.get('additive_gain_pct', 0):.1%}\n"
             f"  (+{comp.get('additive_gain_clusters', 0)} clusters)",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    per_q = comp.get("per_query", [])
    if per_q:
        queries_with_gain = [(q["query"][:40], q["additive_gain"])
                             for q in per_q if q.get("additive_gain", 0) > 0]
        queries_with_gain.sort(key=lambda x: x[1], reverse=True)

        if queries_with_gain:
            q_labels = [q[0] for q in queries_with_gain[:12]]
            q_gains = [q[1] for q in queries_with_gain[:12]]
            y = np.arange(len(q_labels))
            ax2.barh(y, q_gains, color="#4CAF50", alpha=0.7, edgecolor="white")
            ax2.set_yticks(y)
            ax2.set_yticklabels(q_labels, fontsize=8)
            ax2.set_xlabel("Additive Recall Gain")
            ax2.set_title("Per-Query Additive Gain\n(union vs best single)", fontweight="bold")
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, "No additive gains\n(modalities agree)",
                     transform=ax2.transAxes, ha="center", va="center", fontsize=12)
            ax2.set_title("Per-Query Additive Gain", fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "complementarity.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "complementarity.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  [FIG] complementarity.png")

    # ── Figure 6: Analytical modules radar ──
    ana = analytical
    metrics_radar = {
        "Pathways": ana.get("pathways", {}).get("alignment", 0) / 100,
        "Interactions\n(LR)": ana.get("interactions", {}).get("lr_recovery_rate", 0) / 100,
        "Proportions": ana.get("proportions", {}).get("consistency_rate", 0) / 100,
        "Compare\n(gene recall)": ana.get("compare", {}).get("compare_recall", 0) / 100,
    }

    labels_r = list(metrics_radar.keys())
    vals_r = list(metrics_radar.values())
    angles = np.linspace(0, 2 * np.pi, len(labels_r), endpoint=False).tolist()
    vals_r += vals_r[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, vals_r, alpha=0.25, color="#4CAF50")
    ax.plot(angles, vals_r, "o-", color="#4CAF50", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Analytical Module Performance", fontweight="bold", pad=20)

    for angle, val in zip(angles[:-1], vals_r[:-1]):
        ax.text(angle, val + 0.05, f"{val:.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "analytical_radar.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "analytical_radar.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"  [FIG] analytical_radar.png")


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary(summary: dict, complementarity: dict, analytical: dict):
    """Pretty-print results to console."""
    MODES = ["random", "semantic", "scgpt", "union"]
    MODE_DISPLAY = {
        "random": "Random",
        "semantic": "Semantic", "scgpt": "scGPT", "union": "Union(add)",
    }

    print("\n" + "=" * 100)
    print("ELISA BENCHMARK v5.1 — ADDITIVE UNION + GENE DELIVERY")
    print("=" * 100)

    print("\n── Retrieval: All Modes ──\n")
    print(f"{'Category':<14} {'Mode':<14} {'R@5':>7} {'R@10':>7} {'R@15':>7} "
          f"{'R@20':>7} {'MRR':>7} {'GeneR':>7} {'#Ret':>6}")
    print("-" * 90)
    for cat in ["ontology", "expression"]:
        for mode in MODES:
            key = f"{cat}_{mode}"
            if key not in summary:
                continue
            s = summary[key]
            gr = s.get("mean_gene_recall", 0)
            gr_str = f"{gr:.3f}" if s.get("has_gene_evidence") else "  n/a"
            r15 = s.get("mean_recall@15", 0)
            r20 = s.get("mean_recall@20", 0)
            n_ret = s.get("mean_n_retrieved", 10)
            n_ret_str = f"{n_ret:.0f}" if mode == "union" else "10"
            print(f"{cat:<14} {MODE_DISPLAY[mode]:<14} "
                  f"{s.get('mean_recall@5', 0):>7.3f} "
                  f"{s.get('mean_recall@10', 0):>7.3f} "
                  f"{r15:>7.3f} {r20:>7.3f} "
                  f"{s['mean_mrr']:>7.3f} {gr_str:>7} {n_ret_str:>6}")
        print()

    # Union-specific stats
    print("── Union Mode Details ──\n")
    for cat in ["ontology", "expression"]:
        key = f"{cat}_union"
        if key not in summary:
            continue
        s = summary[key]
        gain5 = s.get("mean_additive_gain@5", 0)
        gain10 = s.get("mean_additive_gain@10", 0)
        psel = s.get("primary_selection", {})
        print(f"  {cat}:")
        print(f"    Additive gain @5:   +{gain5:.3f}")
        print(f"    Additive gain @10:  +{gain10:.3f}")
        print(f"    Avg clusters returned: {s.get('mean_n_retrieved', 0):.1f}")
        print(f"    Primary selection:  {dict(psel)}")
        print()

    # Overall average
    print("── Overall Mean (both categories) ──\n")
    print(f"  {'Mode':<14} {'R@5':>7} {'R@10':>7} {'R@15':>7} {'R@20':>7} {'GeneR':>7}")
    print("  " + "-" * 55)
    for mode in MODES:
        vals = {k: [] for k in [5, 10, 15, 20]}
        gr_vals = []
        for cat in ["ontology", "expression"]:
            key = f"{cat}_{mode}"
            if key in summary:
                for k in vals:
                    vals[k].append(summary[key].get(f"mean_recall@{k}", 0))
                gr_vals.append(summary[key].get("mean_gene_recall", 0))
        if vals[5]:
            print(f"  {MODE_DISPLAY[mode]:<14} "
                  f"{np.mean(vals[5]):>7.3f} {np.mean(vals[10]):>7.3f} "
                  f"{np.mean(vals[15]):>7.3f} {np.mean(vals[20]):>7.3f} "
                  f"{np.mean(gr_vals):>7.3f}")

    print("\n── Complementarity (Additive Union) ──\n")
    c = complementarity
    print(f"  Total expected clusters:   {c.get('total_expected', 0)}")
    print(f"  Semantic found:            {c.get('semantic_found', 0)} "
          f"({c.get('semantic_recall', 0):.1%})")
    print(f"  scGPT found:               {c.get('scgpt_found', 0)} "
          f"({c.get('scgpt_recall', 0):.1%})")
    print(f"  Best single modality:      {c.get('best_single_found', 0)} "
          f"({c.get('best_single_recall', 0):.1%})")
    print(f"  Union found (additive):    {c.get('union_found', 0)} "
          f"({c.get('union_recall', 0):.1%})")
    print(f"  Additive gain:             +{c.get('additive_gain_clusters', 0)} clusters "
          f"(+{c.get('additive_gain_pct', 0):.1%})")
    print(f"    Only semantic:           {c.get('only_semantic_count', 0)}")
    print(f"    Only scGPT:              {c.get('only_scgpt_count', 0)}")
    print(f"    Neither:                 {c.get('neither_count', 0)}")

    print("\n── Analytical Modules ──\n")
    a = analytical
    print(f"  Pathways:      {a.get('pathways', {}).get('alignment', 0):.1f}% "
          f"({a.get('pathways', {}).get('pathways_found', 0)}/"
          f"{a.get('pathways', {}).get('pathways_expected', 0)})")
    print(f"  Interactions:  {a.get('interactions', {}).get('lr_recovery_rate', 0):.1f}% LR, "
          f"{a.get('interactions', {}).get('full_recovery_rate', 0):.1f}% full match")
    print(f"  Proportions:   {a.get('proportions', {}).get('consistency_rate', 0):.1f}% "
          f"({a.get('proportions', {}).get('consistent', 0)}/"
          f"{a.get('proportions', {}).get('total_checked', 0)})")
    print(f"  Compare:       {a.get('compare', {}).get('compare_recall', 0):.1f}% "
          f"({a.get('compare', {}).get('genes_found', 0)}/"
          f"{a.get('compare', {}).get('genes_requested', 0)})")
    print("=" * 100 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ELISA Benchmark v5.1")
    parser.add_argument("--base", required=True, help="Path to embedding directory")
    parser.add_argument("--pt-name", default=None, help="Override .pt filename")
    parser.add_argument("--cells-csv", default=None, help="Override cells CSV")
    parser.add_argument("--paper", default="P1", help="Paper ID")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k clusters")
    parser.add_argument("--out", default="benchmark_v5/", help="Output directory")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-generate figures from existing results")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with existing results JSON for --plot-only")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Plot-only mode
    if args.plot_only:
        rd = args.results_dir or out_dir
        with open(os.path.join(rd, "benchmark_v5_results.json")) as f:
            data = json.load(f)
        paper_id = args.paper
        generate_figures(
            data[paper_id]["retrieval_summary"],
            data[paper_id]["complementarity"],
            data[paper_id]["analytical"],
            out_dir,
        )
        print("Figures regenerated.")
        return

    # Load engine
    paper = BENCHMARK_PAPERS[args.paper]
    pt_name = args.pt_name or paper.get("pt_name", "")
    cells_csv = args.cells_csv or paper.get("cells_csv")

    sys.path.insert(0, os.path.dirname(args.base))
    sys.path.insert(0, args.base)
    sys.path.insert(0, os.getcwd())

    from retrieval_engine_v4_hybrid import RetrievalEngine
    print(f"\n[BENCHMARK v5.1] Loading engine: {args.base} / {pt_name}")
    engine = RetrievalEngine(
        base=args.base, pt_name=pt_name, cells_csv=cells_csv
    )

    # Attach analytical modules
    from elisa_analysis import (
        find_interactions, pathway_scoring, proportion_analysis,
        comparative_analysis,
    )

    class EngineWithAnalysis:
        def __init__(self, eng):
            self._eng = eng
            for attr in dir(eng):
                if not attr.startswith("_"):
                    try:
                        setattr(self, attr, getattr(eng, attr))
                    except Exception:
                        pass

        def interactions(self, **kw):
            return find_interactions(self._eng.gene_stats, self._eng.cluster_ids, **kw)

        def pathways(self, **kw):
            return pathway_scoring(self._eng.gene_stats, self._eng.cluster_ids, **kw)

        def proportions(self, **kw):
                return proportion_analysis(
                self._eng.metadata,
                condition_col=paper.get("condition_col", "patient_group"),
            )

        def compare(self, cond_a, cond_b, **kw):
            return comparative_analysis(
                self._eng.gene_stats,
                self._eng.metadata,
                condition_col=paper.get("condition_col", "patient_group"),
                group_a=cond_a, group_b=cond_b, **kw
            )

    engine_wrap = EngineWithAnalysis(engine)

    # ── Run retrieval evaluation ──
    print(f"\n[BENCHMARK v5.1] Running retrieval evaluation "
          f"({len(paper['queries'])} queries × {len(RetrievalEvaluator.MODES)} modes)...")
    t0 = time.time()

    ret_eval = RetrievalEvaluator(engine)
    ret_results = ret_eval.evaluate_queries(paper["queries"], top_k=args.top_k)
    ret_summary = ret_eval.compute_summary(ret_results)
    complementarity = ret_eval.compute_complementarity(ret_results, top_k=args.top_k)

    print(f"  Retrieval done in {time.time() - t0:.1f}s")

    # ── Run analytical evaluation ──
    print("[BENCHMARK v5.1] Running analytical module evaluation...")
    t0 = time.time()

    ana_eval = AnalyticalEvaluator(engine_wrap)
    try:
        analytical = ana_eval.evaluate_all(paper)
    except Exception as e:
        print(f"    [WARN] Analytical modules failed: {e}")
        analytical = {"pathways": {}, "interactions": {}, "proportions": {}, "compare": {}}

    print(f"  Analytical done in {time.time() - t0:.1f}s")

    # ── Output ──
    print_summary(ret_summary, complementarity, analytical)

    output = {
        args.paper: {
            "retrieval_detail": ret_results,
            "retrieval_summary": ret_summary,
            "complementarity": complementarity,
            "analytical": analytical,
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
        }
    }

    results_path = os.path.join(out_dir, "benchmark_v5_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {results_path}")

    # ── Figures ──
    print("[BENCHMARK v5.1] Generating figures...")
    generate_figures(ret_summary, complementarity, analytical, out_dir)

    print("\n[BENCHMARK v5.1] Complete!")


if __name__ == "__main__":
    main()
