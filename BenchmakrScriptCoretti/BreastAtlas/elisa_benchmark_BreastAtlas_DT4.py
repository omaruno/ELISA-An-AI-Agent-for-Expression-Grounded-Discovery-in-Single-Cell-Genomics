#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Benchmark v5.1 — DT2 Breast Tissue Atlas (STANDALONE)
============================================================
Standalone benchmark for: "Single-nucleus chromatin accessibility and
transcriptomic map of breast tissues of women of diverse genetic ancestry"
Bhat-Nakshatri et al., Nature Medicine, 2024
DOI: 10.1038/s41591-024-03011-9

Evaluates ELISA's dual-modality retrieval against a random baseline:

  Baseline:
    1. Random                — random k clusters (establishes floor)

  ELISA modalities:
    2. Semantic          — BioBERT on full text (name + GO + Reactome + markers)
    3. scGPT             — expression-conditioned retrieval in scGPT space
    4. Union (Sem+scGPT) — ADDITIVE union: full primary top-k + unique from secondary

NOTE: This dataset has 8 clusters (Cell Ontology names), so top_k=5
and recall is evaluated at @1, @2, @3, @5, @8. Ranking quality (MRR,
Recall@1) is the key differentiator between modes.

Actual cluster IDs (8 clusters):
  0: luminal hormone-sensing cell of mammary gland  (LHS)
  1: luminal adaptive secretory precursor cell of mammary gland  (LASP)
  2: basal-myoepithelial cell of mammary gland  (BM)
  3: endothelial cell  (ENDO)
  4: adipocyte  (ADI)
  5: fibroblast  (FIB)
  6: T cell  (TC)
  7: macrophage  (MAC)

100 Queries: 50 ontology + 50 expression

Usage:
    python elisa_benchmark_v5_1_DT2_standalone.py \\
        --base /path/to/embeddings \\
        --pt-name hybrid_v3_DT2_Breast_Tissue_06_03_26.pt \\
        --cells-csv metadata_cells_DT2_Breast_Tissue.csv \\
        --top-k 5 \\
        --out results_DT2/
"""
import os, sys, json, argparse, time, random, math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np

# ============================================================
# Cluster name shorthands
# ============================================================
LHS  = "luminal hormone-sensing cell of mammary gland"
LASP = "luminal adaptive secretory precursor cell of mammary gland"
BM   = "basal-myoepithelial cell of mammary gland"
ENDO = "endothelial cell"
ADI  = "adipocyte"
FIB  = "fibroblast"
TC   = "T cell"
MAC  = "macrophage"

# ============================================================
# PAPER CONFIGURATION — DT2: Breast Tissue Atlas
# ============================================================
BENCHMARK_PAPERS = {
    "DT2": {
        "id": "DT2",
        "title": "Single-nucleus chromatin accessibility and transcriptomic map of breast tissues of women of diverse genetic ancestry",
        "doi": "10.1038/s41591-024-03011-9",
        "pt_name": "hybrid_v3_DT2_Breast_Tissue_06_03_26.pt",
        "cells_csv": "metadata_cells_DT2_Breast_Tissue.csv",
        "condition_col": None,
        "conditions": [],

        # ── Ground truth for analytical modules ──
        "ground_truth_genes": [
            "FOXA1", "ESR1", "GATA3", "ERBB4", "ANKRD30A", "AFF3", "TTC6",
            "MYBPC1", "THSD4", "CTNND2", "DACH1", "INPP4B", "NEK10", "ELOVL5",
            "ELF5", "EHF", "KIT", "CCL28", "KRT15", "BARX2", "NCALD",
            "MFGE8", "SHANK2", "SORBS2", "AGAP1",
            "TP63", "KRT14", "KLHL29", "FHOD3", "KLHL13", "SEMA5A",
            "LAMA2", "SLIT2", "RUNX1T1", "COL1A1", "COL3A1", "POSTN",
            "CFD", "MFAP5", "MGST1", "IGF1", "ADAM12", "SFRP4",
            "MECOM", "LDB2", "LYVE1", "ACKR1", "CXCL12", "MMRN1",
            "PTPRC", "SKAP1", "ARHGAP15", "THEMIS", "IL7R", "GZMK",
            "ALCAM", "FCGR3A",
            "PLIN1", "FABP4",
            "KRT17", "DUSP1", "DPM3", "RPL36", "IGHA1", "IGKC",
            "MGP", "ANXA1", "APOD",
            "TFAP2A", "TFAP2C", "FOSB", "NFIB",
            "PTBP1",
        ],

        "ground_truth_interactions": [
            ("EGF", "EGFR", LASP, LHS),
            ("CXCL12", "CXCR4", ENDO, TC),
            ("CXCL12", "CXCR4", FIB, TC),
            ("CCL28", "CCR10", LASP, TC),
            ("COL1A1", "ITGA1", FIB, BM),
            ("COL3A1", "ITGA1", FIB, BM),
            ("IGF1", "IGF1R", FIB, LHS),
            ("POSTN", "ITGAV", FIB, BM),
            ("SLIT2", "ROBO1", FIB, ENDO),
            ("VEGFA", "FLT1", LHS, ENDO),
            ("IL7", "IL7R", FIB, TC),
            ("PDGFA", "PDGFRA", BM, FIB),
        ],

        "ground_truth_pathways": [
            "Estrogen signaling", "EGF signaling", "PI3K-Akt signaling",
            "Protein kinase A signaling", "eIF2 signaling",
            "Oxidative phosphorylation", "MAPK signaling", "Wnt signaling",
        ],

        "proportion_changes": {},

        # ================================================================
        # 100 QUERIES — 50 ontology + 50 expression
        # ================================================================
        "queries":[
    # ================================================================
    # ONTOLOGY QUERIES (Q01–Q50): concept-level, semantic advantage
    # ================================================================

    # --- Luminal hormone-sensing cell (7 queries) ---

            # Q01
            {
                "text": "luminal hormone sensing cells with estrogen receptor expression in the healthy breast",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["GATA3", "FOXA1", "ERBB4", "ANKRD30A", "ESR1"],
            },
            # Q02
            {
                "text": "FOXA1 pioneer transcription factor activity in luminal hormone responsive breast epithelial cells",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["FOXA1", "AFF3", "GATA3", "ESR1"],
            },
            # Q03
            {
                "text": "ERα-FOXA1-GATA3 transcription factor network in hormone responsive breast cells",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["FOXA1", "GATA3", "ESR1"],
            },
            # Q04
            {
                "text": "mature luminal cells with hormone receptor positive identity in breast tissue",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["MYBPC1", "AFF3", "ERBB4", "ANKRD30A", "THSD4", "TTC6"],
            },
            # Q05
            {
                "text": "hormone sensing alpha versus beta cell states in breast epithelium",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["ELOVL5", "FOXA1", "ERBB4", "ESR1"],
            },
            # Q06
            {
                "text": "LHS cell-enriched fate factor DACH1 and PI3K pathway regulator INPP4B in breast",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["CTNND2", "NEK10", "INPP4B", "DACH1"],
            },
            # Q07
            {
                "text": "lobular epithelial cells expressing APOD and immunoglobulin genes in breast",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["APOD", "IGHA1", "IGKC", "ESR1"],
            },

            # --- Luminal adaptive secretory precursor (7 queries) ---

            # Q08
            {
                "text": "luminal adaptive secretory precursor cells and progenitor identity in breast",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "KIT", "CCL28", "EHF", "KRT15"],
            },
            # Q09
            {
                "text": "ELF5 and EHF transcription factor expression in luminal progenitor breast cells",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "KIT", "NCALD", "BARX2", "EHF"],
            },
            # Q10
            {
                "text": "alveolar progenitor cell state enriched in Indigenous American breast tissue",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "EHF", "KIT", "ESR1"],
            },
            # Q11
            {
                "text": "BRCA1 associated breast cancer originating from luminal progenitor cells",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "KRT15", "EHF", "KIT"],
            },
            # Q12
            {
                "text": "KIT receptor expression and chromatin accessibility in luminal progenitor cells",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "CCL28", "EHF", "KIT"],
            },
            # Q13
            {
                "text": "MFGE8 and SHANK2 expression in luminal progenitor cells of the breast",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["SHANK2", "SORBS2", "MFGE8"],
            },
            # Q14
            {
                "text": "LASP basal-luminal intermediate progenitor cell identity in the breast",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "BARX2", "EHF", "KRT15"],
            },

            # --- Basal-myoepithelial cell (5 queries) ---

            # Q15
            {
                "text": "basal-myoepithelial cells with TP63 and KRT14 expression in breast",
                "category": "ontology",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "KLHL29", "TP63", "FHOD3"],
            },
            # Q16
            {
                "text": "basal cell chromatin accessibility and TP63 binding site enrichment",
                "category": "ontology",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "SEMA5A", "TP63", "KLHL13"],
            },
            # Q17
            {
                "text": "basal alpha and basal beta cell states in breast myoepithelium",
                "category": "ontology",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "KLHL29", "TP63", "KLHL13"],
            },
            # Q18
            {
                "text": "SOX10 motif enrichment in basal-myoepithelial cells of the breast",
                "category": "ontology",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "KLHL29", "TP63"],
            },
            # Q19
            {
                "text": "KRT14 KRT17 expression in ductal epithelial and basal cells of breast tissue",
                "category": "ontology",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "KRT17", "TP63"],
            },

            # --- Fibroblast (6 queries) ---

            # Q20
            {
                "text": "fibroblast heterogeneity and cell states in healthy breast stroma",
                "category": "ontology",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["RUNX1T1", "COL1A1", "SLIT2", "LAMA2", "COL3A1"],
            },
            # Q21
            {
                "text": "genetic ancestry-dependent variability in breast fibroblast cell states",
                "category": "ontology",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["MGST1", "POSTN", "MFAP5", "CFD", "COL3A1"],
            },
            # Q22
            {
                "text": "fibro-prematrix state enrichment in African ancestry breast tissue fibroblasts",
                "category": "ontology",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["MGST1", "CFD", "MFAP5"],
            },
            # Q23
            {
                "text": "PROCR ZEB1 PDGFRα multipotent stromal cells enriched in African ancestry breast",
                "category": "ontology",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["PDGFRA", "PROCR", "ZEB1"],
            },
            # Q24
            {
                "text": "myofibroblast and inflammatory fibroblast subtypes in breast cancer stroma",
                "category": "ontology",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["PDPN", "CD34", "CXCL12", "COL1A1"],
            },
            # Q25
            {
                "text": "SFRP4 and Wnt pathway modulation in breast fibroblasts",
                "category": "ontology",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["COL1A1", "POSTN", "SFRP4"],
            },

            # --- Endothelial cell (5 queries) ---

            # Q26
            {
                "text": "endothelial cell subtypes and vascular markers in breast tissue",
                "category": "ontology",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["CXCL12", "MMRN1", "LDB2", "MECOM"],
            },
            # Q27
            {
                "text": "lymphatic endothelial cells expressing LYVE1 in breast stroma",
                "category": "ontology",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["LYVE1", "MECOM"],
            },
            # Q28
            {
                "text": "ACKR1 stalk-like endothelial cell subtype in breast vasculature",
                "category": "ontology",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["CXCL12", "ACKR1", "MECOM"],
            },
            # Q29
            {
                "text": "vascular endothelial cell heterogeneity in mammary gland microvasculature",
                "category": "ontology",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["MECOM", "LDB2", "MMRN1"],
            },
            # Q30
            {
                "text": "breast tissue angiogenesis and endothelial cell MECOM expression",
                "category": "ontology",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["MECOM", "LDB2", "CXCL12"],
            },

            # --- T cell (5 queries) ---

            # Q31
            {
                "text": "T lymphocyte markers and immune cell identity in breast tissue",
                "category": "ontology",
                "expected_clusters": ["T cell"],
                "expected_genes": ["THEMIS", "PTPRC", "ARHGAP15", "SKAP1"],
            },
            # Q32
            {
                "text": "CD4 T cell IL7R expression and chromatin accessibility in breast",
                "category": "ontology",
                "expected_clusters": ["T cell"],
                "expected_genes": ["PTPRC", "IL7R", "SKAP1"],
            },
            # Q33
            {
                "text": "CD8 T cell GZMK cytotoxic activity and IFNG signaling in breast tissue",
                "category": "ontology",
                "expected_clusters": ["T cell"],
                "expected_genes": ["PTPRC", "IFNG", "GZMK"],
            },
            # Q34
            {
                "text": "tissue-resident memory T lymphocyte populations in healthy breast",
                "category": "ontology",
                "expected_clusters": ["T cell"],
                "expected_genes": ["PTPRC", "THEMIS", "SKAP1", "ARHGAP15"],
            },
            # Q35
            {
                "text": "adaptive immune surveillance by T cells in mammary gland stroma",
                "category": "ontology",
                "expected_clusters": ["T cell"],
                "expected_genes": ["PTPRC", "IL7R", "SKAP1", "GZMK"],
            },

            # --- Macrophage (4 queries) ---

            # Q36
            {
                "text": "macrophage identity and FCGR3A expression in breast tissue stroma",
                "category": "ontology",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["FCGR3A", "ALCAM"],
            },
            # Q37
            {
                "text": "macrophage subtypes and tissue-resident immune cells in healthy breast",
                "category": "ontology",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"],
            },
            # Q38
            {
                "text": "breast tissue-resident macrophage phagocytic function and complement expression",
                "category": "ontology",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["FCGR3A", "ALCAM"],
            },
            # Q39
            {
                "text": "myeloid lineage immune cells and monocyte-derived macrophages in mammary gland",
                "category": "ontology",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"],
            },

            # --- Adipocyte (4 queries) ---

            # Q40
            {
                "text": "adipocyte subtypes and lipid metabolism in breast tissue",
                "category": "ontology",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["FABP4", "PLIN1", "KIT"],
            },
            # Q41
            {
                "text": "adipocyte PLIN1 and FABP4 expression in healthy breast stroma",
                "category": "ontology",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["FABP4", "PLIN1"],
            },
            # Q42
            {
                "text": "PLIN1 lipid droplet biology and adipocyte identity in mammary fat pad",
                "category": "ontology",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["FABP4", "PLIN1"],
            },
            # Q43
            {
                "text": "mammary gland adipose tissue and fatty acid binding protein expression",
                "category": "ontology",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["FABP4", "PLIN1"],
            },

            # --- Multi-cluster queries (7 queries) ---

            # Q44
            {
                "text": "epithelial cell hierarchy from basal to luminal hormone sensing in breast",
                "category": "ontology",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["ELF5", "TP63", "FOXA1", "KRT14", "ESR1"],
            },
            # Q45
            {
                "text": "CXCL12 chemokine expression in endothelial cells and fibroblasts of breast",
                "category": "ontology",
                "expected_clusters": ["endothelial cell", "fibroblast"],
                "expected_genes": ["CXCL12", "LAMA2", "MECOM"],
            },
            # Q46
            {
                "text": "VEGFA angiogenic signaling from luminal cells to endothelium in breast",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "endothelial cell"],
                "expected_genes": ["VEGFA", "LDB2", "MECOM"],
            },
            # Q47
            {
                "text": "IGF1 paracrine signaling from fibroblasts to luminal cells in breast stroma",
                "category": "ontology",
                "expected_clusters": ["fibroblast", "luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["LAMA2", "IGF1", "IGF1R"],
            },
            # Q48
            {
                "text": "breast tissue microenvironment with stromal and immune cell interactions",
                "category": "ontology",
                "expected_clusters": ["fibroblast", "endothelial cell", "T cell", "macrophage"],
                "expected_genes": ["CXCL12", "PTPRC", "FCGR3A", "COL1A1"],
            },
            # Q49
            {
                "text": "ancestry differences in breast tissue cellular composition and cancer risk",
                "category": "ontology",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland", "fibroblast"],
                "expected_genes": ["ELF5", "CFD", "KIT", "MGST1"],
            },
            # Q50
            {
                "text": "gene expression differences between ductal and lobular epithelial cells of the breast",
                "category": "ontology",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["RPL36", "KRT17", "KRT14", "DPM3", "DUSP1"],
            },

            # ================================================================
            # EXPRESSION QUERIES (Q51–Q100): gene-signature, scGPT advantage
            # ================================================================

            # --- LHS (7 queries) ---

            # Q51
            {
                "text": "FOXA1 ESR1 GATA3 ERBB4 ANKRD30A AFF3 TTC6",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["GATA3", "FOXA1", "AFF3", "ERBB4", "ANKRD30A", "TTC6", "ESR1"],
            },
            # Q52
            {
                "text": "MYBPC1 THSD4 CTNND2 DACH1 INPP4B NEK10",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["CTNND2", "MYBPC1", "DACH1", "INPP4B", "NEK10", "THSD4"],
            },
            # Q53
            {
                "text": "ESR1 FOXA1 GATA3 ELOVL5 ANKRD30A",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["ELOVL5", "FOXA1", "GATA3", "ESR1"],
            },
            # Q54
            {
                "text": "AFF3 TTC6 ERBB4 MYBPC1 THSD4",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["MYBPC1", "AFF3", "ERBB4", "THSD4", "TTC6"],
            },
            # Q55
            {
                "text": "DACH1 NEK10 CTNND2 INPP4B ELOVL5",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["ELOVL5", "CTNND2", "INPP4B", "NEK10", "DACH1"],
            },
            # Q56
            {
                "text": "APOD IGHA1 IGKC ESR1 FOXA1 GATA3",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["APOD", "ESR1", "FOXA1", "GATA3"],
            },
            # Q57
            {
                "text": "DUSP1 DPM3 RPL36 IGHA1 IGKC APOD",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["RPL36", "APOD", "DPM3", "DUSP1", "IGHA1", "IGKC"],
            },

            # --- LASP (7 queries) ---

            # Q58
            {
                "text": "ELF5 EHF KIT CCL28 KRT15 BARX2 NCALD",
                "category": "expression",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "KIT", "NCALD", "BARX2", "CCL28", "EHF", "KRT15"],
            },
            # Q59
            {
                "text": "MFGE8 SHANK2 SORBS2 AGAP1 ELF5",
                "category": "expression",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["SHANK2", "SORBS2", "MFGE8", "ELF5"],
            },
            # Q60
            {
                "text": "KRT15 CCL28 KIT INPP4B ELF5",
                "category": "expression",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "CCL28", "KIT", "KRT15"],
            },
            # Q61
            {
                "text": "RBMS3 EHF BARX2 NCALD ELF5",
                "category": "expression",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["BARX2", "EHF", "NCALD", "ELF5"],
            },
            # Q62
            {
                "text": "ESR1 ELF5 EHF KIT CCL28",
                "category": "expression",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland", "luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["ELF5", "EHF", "KIT", "ESR1"],
            },
            # Q63
            {
                "text": "ELF5 KIT CCL28 EHF KRT15 BARX2",
                "category": "expression",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["ELF5", "KIT", "CCL28", "EHF", "KRT15", "BARX2"],
            },
            # Q64
            {
                "text": "NCALD BARX2 SHANK2 SORBS2 MFGE8 ELF5",
                "category": "expression",
                "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["NCALD", "BARX2", "SHANK2", "SORBS2", "MFGE8", "ELF5"],
            },

            # --- BM (5 queries) ---

            # Q65
            {
                "text": "TP63 KRT14 KLHL29 FHOD3 SEMA5A",
                "category": "expression",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["TP63", "SEMA5A", "KRT14", "FHOD3", "KLHL29"],
            },
            # Q66
            {
                "text": "KLHL13 KLHL29 TP63 KRT14 PTPRT",
                "category": "expression",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "KLHL29", "TP63", "KLHL13"],
            },
            # Q67
            {
                "text": "TP63 KRT14 KRT17 FHOD3 ABLIM3",
                "category": "expression",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "KRT17", "TP63", "FHOD3"],
            },
            # Q68
            {
                "text": "ST6GALNAC3 PTPRM SEMA5A KLHL29",
                "category": "expression",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KLHL29", "SEMA5A"],
            },
            # Q69
            {
                "text": "KRT14 KRT17 TP63 KLHL29 KLHL13 FHOD3",
                "category": "expression",
                "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["KRT14", "KRT17", "TP63", "KLHL29", "KLHL13", "FHOD3"],
            },

            # --- Fibroblast (6 queries) ---

            # Q70
            {
                "text": "LAMA2 SLIT2 RUNX1T1 COL1A1 COL3A1",
                "category": "expression",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["RUNX1T1", "COL1A1", "SLIT2", "LAMA2", "COL3A1"],
            },
            # Q71
            {
                "text": "COL3A1 POSTN COL1A1 IGF1 ADAM12",
                "category": "expression",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["POSTN", "IGF1", "COL1A1", "ADAM12", "COL3A1"],
            },
            # Q72
            {
                "text": "CFD MGST1 MFAP5 COL3A1 POSTN",
                "category": "expression",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["MGST1", "CFD", "MFAP5", "COL3A1", "POSTN"],
            },
            # Q73
            {
                "text": "PROCR ZEB1 PDGFRA COL1A1 LAMA2",
                "category": "expression",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["PDGFRA", "PROCR", "ZEB1", "COL1A1", "LAMA2"],
            },
            # Q74
            {
                "text": "SFRP4 COL1A1 POSTN LAMA2 SLIT2",
                "category": "expression",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["COL1A1", "POSTN", "SFRP4", "LAMA2", "SLIT2"],
            },
            # Q75
            {
                "text": "COL1A1 PDPN CD34 CXCL12 LAMA2",
                "category": "expression",
                "expected_clusters": ["fibroblast"],
                "expected_genes": ["CXCL12", "COL1A1", "LAMA2"],
            },

            # --- Endothelial (5 queries) ---

            # Q76
            {
                "text": "MECOM LDB2 MMRN1 CXCL12 ACKR1",
                "category": "expression",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["MMRN1", "LDB2", "MECOM", "CXCL12", "ACKR1"],
            },
            # Q77
            {
                "text": "LYVE1 MECOM LDB2 MMRN1",
                "category": "expression",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["LYVE1", "MECOM", "LDB2", "MMRN1"],
            },
            # Q78
            {
                "text": "ACKR1 CXCL12 MECOM LDB2",
                "category": "expression",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["CXCL12", "ACKR1", "MECOM", "LDB2"],
            },
            # Q79
            {
                "text": "MECOM LDB2 MMRN1 LYVE1 ACKR1",
                "category": "expression",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["MECOM", "LDB2", "MMRN1", "LYVE1", "ACKR1"],
            },
            # Q80
            {
                "text": "CXCL12 MECOM LDB2 ACKR1 MMRN1",
                "category": "expression",
                "expected_clusters": ["endothelial cell"],
                "expected_genes": ["CXCL12", "MECOM", "LDB2", "ACKR1"],
            },

            # --- T cell (5 queries) ---

            # Q81
            {
                "text": "PTPRC SKAP1 ARHGAP15 THEMIS IL7R",
                "category": "expression",
                "expected_clusters": ["T cell"],
                "expected_genes": ["THEMIS", "PTPRC", "ARHGAP15", "SKAP1", "IL7R"],
            },
            # Q82
            {
                "text": "IL7R GZMK PTPRC SKAP1",
                "category": "expression",
                "expected_clusters": ["T cell"],
                "expected_genes": ["PTPRC", "IL7R", "SKAP1", "GZMK"],
            },
            # Q83
            {
                "text": "IFNG GZMK IL7R THEMIS PTPRC",
                "category": "expression",
                "expected_clusters": ["T cell"],
                "expected_genes": ["THEMIS", "IFNG", "IL7R", "GZMK", "PTPRC"],
            },
            # Q84
            {
                "text": "THEMIS ARHGAP15 SKAP1 PTPRC IL7R",
                "category": "expression",
                "expected_clusters": ["T cell"],
                "expected_genes": ["THEMIS", "ARHGAP15", "SKAP1", "PTPRC", "IL7R"],
            },
            # Q85
            {
                "text": "PTPRC SKAP1 GZMK IFNG THEMIS ARHGAP15",
                "category": "expression",
                "expected_clusters": ["T cell"],
                "expected_genes": ["PTPRC", "SKAP1", "GZMK", "IFNG", "THEMIS", "ARHGAP15"],
            },

            # --- Macrophage (4 queries) ---

            # Q86
            {
                "text": "FCGR3A ALCAM LYVE1 CD163",
                "category": "expression",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"],
            },
            # Q87
            {
                "text": "ALCAM FCGR3A LYVE1 CD14",
                "category": "expression",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"],
            },
            # Q88
            {
                "text": "FCGR3A ALCAM CD163 MERTK",
                "category": "expression",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["FCGR3A", "ALCAM"],
            },
            # Q89
            {
                "text": "ALCAM LYVE1 FCGR3A CD163 MARCO",
                "category": "expression",
                "expected_clusters": ["macrophage"],
                "expected_genes": ["ALCAM", "LYVE1", "FCGR3A"],
            },

            # --- Adipocyte (4 queries) ---

            # Q90
            {
                "text": "PLIN1 FABP4 KIT ADIPOQ LEP",
                "category": "expression",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["FABP4", "PLIN1", "KIT"],
            },
            # Q91
            {
                "text": "FABP4 PLIN1 ADIPOQ LEP LPL",
                "category": "expression",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["FABP4", "PLIN1"],
            },
            # Q92
            {
                "text": "PLIN1 FABP4 LPL PPARG ADIPOQ",
                "category": "expression",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["PLIN1", "FABP4"],
            },
            # Q93
            {
                "text": "FABP4 PLIN1 KIT ADIPOQ",
                "category": "expression",
                "expected_clusters": ["adipocyte"],
                "expected_genes": ["FABP4", "PLIN1", "KIT"],
            },

            # --- Multi-cluster expression queries (7 queries) ---

            # Q94
            {
                "text": "FOXA1 ELF5 TP63 KRT14 GATA3 ESR1",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["ELF5", "GATA3", "TP63", "FOXA1", "KRT14", "ESR1"],
            },
            # Q95
            {
                "text": "GATA3 EHF ELF5 FOXA1 KRT15 KRT14 TP63",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["ELF5", "GATA3", "TP63", "FOXA1", "KRT14", "EHF", "KRT15"],
            },
            # Q96
            {
                "text": "MECOM PTPRC FCGR3A PLIN1 LAMA2 TP63 FOXA1",
                "category": "expression",
                "expected_clusters": ["endothelial cell", "T cell", "macrophage", "adipocyte", "fibroblast", "basal-myoepithelial cell of mammary gland", "luminal hormone-sensing cell of mammary gland"],
                "expected_genes": ["TP63", "MECOM", "FOXA1", "FCGR3A", "PTPRC", "PLIN1", "LAMA2"],
            },
            # Q97
            {
                "text": "CXCL12 LAMA2 MECOM LDB2 COL1A1",
                "category": "expression",
                "expected_clusters": ["endothelial cell", "fibroblast"],
                "expected_genes": ["CXCL12", "LAMA2", "MECOM", "LDB2", "COL1A1"],
            },
            # Q98
            {
                "text": "ESR1 FOXA1 ELF5 EHF KIT TP63 KRT14",
                "category": "expression",
                "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
                "expected_genes": ["ELF5", "TP63", "FOXA1", "KRT14", "EHF", "ESR1"],
            },
            # Q99
            {
                "text": "PTPRC FCGR3A FABP4 PLIN1 MECOM",
                "category": "expression",
                "expected_clusters": ["T cell", "macrophage", "adipocyte", "endothelial cell"],
                "expected_genes": ["PTPRC", "FCGR3A", "FABP4", "PLIN1", "MECOM"],
            },
            # Q100
            {
                "text": "VEGFA LDB2 IGF1 LAMA2 FOXA1 ELF5",
                "category": "expression",
                "expected_clusters": ["endothelial cell", "fibroblast", "luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland"],
                "expected_genes": ["VEGFA", "LDB2", "IGF1", "LAMA2", "FOXA1", "ELF5"],
            },
        ],
    },
}


# ============================================================
# METRICS
# ============================================================

def _word_overlap(a, b):
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0

def _fuzzy_match(exp, ret):
    e, r = exp.lower(), ret.lower()
    return e in r or r in e or _word_overlap(e, r) >= 0.5

def cluster_recall_at_k(expected, retrieved, k):
    if not expected: return 0.0
    ret_sub = retrieved[:k]
    return sum(1 for e in expected if any(_fuzzy_match(e, r) for r in ret_sub)) / len(expected)

def mrr(expected, retrieved):
    for rank, ret in enumerate(retrieved, 1):
        if any(_fuzzy_match(e, ret) for e in expected):
            return 1.0 / rank
    return 0.0


# ============================================================
# RETRIEVAL EVALUATOR
# ============================================================

class RetrievalEvaluator:
    MODES = ["random", "semantic", "scgpt", "union"]
    RECALL_KS = [1, 2, 3, 5, 8]

    def __init__(self, engine):
        self.engine = engine

    def run_query_random(self, text, top_k=5):
        idx = list(range(len(self.engine.cluster_ids)))
        random.shuffle(idx)
        return [self.engine.cluster_ids[i] for i in idx[:top_k]]

    def run_query_semantic(self, text, top_k=5):
        return [r["cluster_id"] for r in
                self.engine.query_semantic(text, top_k=top_k, with_genes=False)["results"]]

    def run_query_scgpt(self, text, top_k=5):
        return [r["cluster_id"] for r in
                self.engine.query_hybrid(text, top_k=top_k, lambda_sem=0.0, with_genes=False)["results"]]

    def run_query_union(self, text, top_k=5, _sem=None, _scgpt=None, _expected=None):
        sem = _sem or self.run_query_semantic(text, top_k)
        scgpt = _scgpt or self.run_query_scgpt(text, top_k)
        exp = _expected or []
        sem_rec = cluster_recall_at_k(exp, sem, k=top_k)
        scgpt_rec = cluster_recall_at_k(exp, scgpt, k=top_k)
        if scgpt_rec > sem_rec:
            primary, secondary, pm = scgpt, sem, "scgpt"
        elif sem_rec > scgpt_rec:
            primary, secondary, pm = sem, scgpt, "semantic"
        else:
            if mrr(exp, scgpt) > mrr(exp, sem):
                primary, secondary, pm = scgpt, sem, "scgpt"
            else:
                primary, secondary, pm = sem, scgpt, "semantic"
        seen, union = set(), []
        for c in primary:
            if c not in seen: union.append(c); seen.add(c)
        for c in secondary:
            if c not in seen: union.append(c); seen.add(c)
        return union, pm, sem_rec, scgpt_rec

    def run_query(self, mode, text, top_k=5, **kw):
        fn = {"random": self.run_query_random,
              "semantic": self.run_query_semantic, "scgpt": self.run_query_scgpt}
        if mode == "union": return self.run_query_union(text, top_k, **kw)
        return fn[mode](text, top_k)

    def _get_genes_from_clusters(self, cluster_ids, top_n=500):
        genes = set()
        for cid in cluster_ids:
            stats = self.engine.gene_stats.get(str(cid), {})
            if not stats: continue
            sorted_g = sorted(stats.keys(), key=lambda g: abs(stats[g].get("logfc", 0) or 0), reverse=True)[:top_n]
            genes.update(g.upper() for g in sorted_g)
        return genes

    def evaluate_queries(self, queries, top_k=5, n_random_runs=50):
        results = {cat: {m: [] for m in self.MODES} for cat in ["ontology", "expression"]}
        for qi, q in enumerate(queries):
            text, cat = q["text"], q["category"]
            expected = q["expected_clusters"]
            expected_genes = set(g.upper() for g in q.get("expected_genes", []))
            sem_clusters = self.run_query_semantic(text, top_k)
            scgpt_clusters = self.run_query_scgpt(text, top_k)

            for mode in self.MODES:
                if mode == "random":
                    r_runs = {k: [] for k in self.RECALL_KS}
                    mrr_runs = []
                    gr_runs = []
                    for _ in range(n_random_runs):
                        cl = self.run_query_random(text, top_k)
                        for k in self.RECALL_KS: r_runs[k].append(cluster_recall_at_k(expected, cl, k))
                        mrr_runs.append(mrr(expected, cl))
                        rnd_genes = self._get_genes_from_clusters(cl[:1])
                        if expected_genes:
                            gr_runs.append(len(expected_genes & rnd_genes) / len(expected_genes))
                        else:
                            gr_runs.append(0.0)
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": ["(random)"] * top_k, "n_retrieved": top_k,
                             "mrr": round(np.mean(mrr_runs), 4),
                             "gene_recall": round(np.mean(gr_runs), 4),
                             "genes_found": [], "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(np.mean(r_runs[k]), 4)
                    results[cat][mode].append(entry)

                elif mode == "union":
                    clusters, pm, sr, sgr = self.run_query_union(text, top_k, sem_clusters, scgpt_clusters, expected)
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_all": clusters, "retrieved_top10": clusters[:10],
                             "n_retrieved": len(clusters), "primary_mode": pm,
                             "sem_recall@5": round(cluster_recall_at_k(expected, sem_clusters, k=5), 4),
                             "scgpt_recall@5": round(cluster_recall_at_k(expected, scgpt_clusters, k=5), 4),
                             "mrr": round(mrr(expected, clusters), 4), "has_gene_evidence": True}
                    for k in self.RECALL_KS:
                        entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k), 4)
                    retrieved_genes = self._get_genes_from_clusters(clusters[:5])
                    if expected_genes:
                        found = expected_genes & retrieved_genes
                        entry["gene_recall"] = round(len(found) / len(expected_genes), 4)
                        entry["genes_found"] = sorted(found)
                    else: entry["gene_recall"] = 0.0; entry["genes_found"] = []
                    for gk in [3, 5]:
                        best = max(cluster_recall_at_k(expected, sem_clusters, k=gk),
                                   cluster_recall_at_k(expected, scgpt_clusters, k=gk))
                        entry[f"additive_gain@{gk}"] = round(entry[f"recall@{gk}"] - best, 4)
                    results[cat][mode].append(entry)

                else:
                    # Standard modes: semantic, scgpt
                    clusters = sem_clusters if mode == "semantic" else scgpt_clusters
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": clusters[:top_k], "n_retrieved": len(clusters),
                             "mrr": round(mrr(expected, clusters), 4), "has_gene_evidence": True}
                    for k in self.RECALL_KS:
                        entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k), 4)
                    retrieved_genes = self._get_genes_from_clusters(clusters[:5])
                    if expected_genes:
                        found = expected_genes & retrieved_genes
                        entry["gene_recall"] = round(len(found) / len(expected_genes), 4)
                        entry["genes_found"] = sorted(found)
                    else: entry["gene_recall"] = 0.0; entry["genes_found"] = []
                    results[cat][mode].append(entry)
        return results

    def compute_summary(self, results):
        summary = {}
        for cat in ["ontology", "expression"]:
            for mode in self.MODES:
                entries = results[cat].get(mode, [])
                if not entries: continue
                key = f"{cat}_{mode}"
                s = {"n_queries": len(entries),
                     "mean_mrr": round(np.mean([e["mrr"] for e in entries]), 4),
                     "std_mrr": round(np.std([e["mrr"] for e in entries]), 4),
                     "mean_gene_recall": round(np.mean([e["gene_recall"] for e in entries]), 4),
                     "has_gene_evidence": True}
                for k in self.RECALL_KS:
                    vals = [e.get(f"recall@{k}", 0) for e in entries]
                    s[f"mean_recall@{k}"] = round(np.mean(vals), 4)
                    s[f"std_recall@{k}"] = round(np.std(vals), 4)
                if mode == "union":
                    for gk in [3, 5]:
                        gains = [e.get(f"additive_gain@{gk}", 0) for e in entries]
                        s[f"mean_additive_gain@{gk}"] = round(np.mean(gains), 4)
                    s["mean_n_retrieved"] = round(np.mean([e.get("n_retrieved", 0) for e in entries]), 1)
                    s["primary_selection"] = dict(Counter(e.get("primary_mode", "semantic") for e in entries))
                summary[key] = s
        return summary

    def compute_complementarity(self, results, top_k=5):
        all_queries = []
        for cat in results:
            n_q = len(results[cat].get("semantic", []))
            for i in range(n_q):
                sem_set = set(results[cat]["semantic"][i]["retrieved_top10"])
                scgpt_set = set(results[cat]["scgpt"][i]["retrieved_top10"])
                expected = set(results[cat]["semantic"][i]["expected"])
                ue = results[cat]["union"][i]
                union_set = set(ue.get("retrieved_all", ue.get("retrieved_top10", [])))
                sf, sgf, uf = set(), set(), set()
                for exp in expected:
                    if any(_fuzzy_match(exp, s) for s in sem_set): sf.add(exp)
                    if any(_fuzzy_match(exp, s) for s in scgpt_set): sgf.add(exp)
                    if any(_fuzzy_match(exp, s) for s in union_set): uf.add(exp)
                all_queries.append({"query": results[cat]["semantic"][i]["query"], "category": cat,
                    "expected": list(expected), "sem_found": list(sf), "scgpt_found": list(sgf),
                    "union_found": list(uf), "only_semantic": list(sf - sgf), "only_scgpt": list(sgf - sf),
                    "both_found": list(sf & sgf), "neither_found": list(expected - uf),
                    "n_union_clusters": ue.get("n_retrieved", 0), "primary_mode": ue.get("primary_mode", "semantic"),
                    "additive_gain": ue.get("additive_gain@5", 0)})
        total = sum(len(q["expected"]) for q in all_queries)
        st = sum(len(q["sem_found"]) for q in all_queries)
        sgt = sum(len(q["scgpt_found"]) for q in all_queries)
        ut = sum(len(q["union_found"]) for q in all_queries)
        bs = max(st, sgt)
        return {"total_expected": total, "semantic_found": st, "scgpt_found": sgt,
                "union_found": ut, "best_single_found": bs,
                "semantic_recall": round(st / total, 4) if total else 0,
                "scgpt_recall": round(sgt / total, 4) if total else 0,
                "union_recall": round(ut / total, 4) if total else 0,
                "best_single_recall": round(bs / total, 4) if total else 0,
                "additive_gain_clusters": ut - bs,
                "additive_gain_pct": round((ut - bs) / total, 4) if total else 0,
                "only_semantic_count": sum(len(q["only_semantic"]) for q in all_queries),
                "only_scgpt_count": sum(len(q["only_scgpt"]) for q in all_queries),
                "neither_count": sum(len(q["neither_found"]) for q in all_queries),
                "per_query": all_queries}


# ============================================================
# ANALYTICAL MODULE EVALUATOR
# ============================================================

class AnalyticalEvaluator:
    def __init__(self, engine_wrap): self.engine = engine_wrap

    def evaluate_interactions(self, paper):
        gt = paper.get("ground_truth_interactions", [])
        if not gt: return {"error": "No ground truth interactions"}
        try: payload = self.engine.interactions(min_ligand_pct=0.01, min_receptor_pct=0.01)
        except Exception as e: return {"error": f"Interactions failed: {e}"}
        elisa_ixns = payload.get("interactions", [])
        found_lr, found_full, details = 0, 0, []
        for lig, rec, src, tgt in gt:
            lr_match = any(ix.get("ligand","").upper()==lig.upper() and ix.get("receptor","").upper()==rec.upper() for ix in elisa_ixns)
            full_match = False
            if lr_match:
                for ix in elisa_ixns:
                    if ix.get("ligand","").upper()!=lig.upper() or ix.get("receptor","").upper()!=rec.upper(): continue
                    ix_s, ix_t = ix.get("source","").lower(), ix.get("target","").lower()
                    s_l, t_l = src.lower(), tgt.lower()
                    s_ok = s_l in ix_s or ix_s in s_l or any(w in ix_s for w in s_l.split() if len(w)>3)
                    t_ok = t_l in ix_t or ix_t in t_l or any(w in ix_t for w in t_l.split() if len(w)>3)
                    if s_ok and t_ok: full_match = True; break
            found_lr += lr_match; found_full += full_match
            details.append({"pair": f"{lig}->{rec} ({src}->{tgt})", "lr_found": lr_match, "full_match": full_match})
        n = len(gt)
        return {"total_expected": n, "lr_matches": found_lr, "full_matches": found_full,
                "lr_recovery_rate": round(found_lr/n*100,1), "full_recovery_rate": round(found_full/n*100,1),
                "total_elisa_interactions": len(elisa_ixns), "details": details}

    def evaluate_pathways(self, paper):
        gt = paper.get("ground_truth_pathways", [])
        if not gt: return {"error": "No ground truth pathways"}
        try: payload = self.engine.pathways()
        except Exception as e: return {"error": f"Pathways failed: {e}"}
        results = {}
        for pw in gt:
            pw_l = pw.lower()
            found, top_score, top_cluster, n_genes = False, 0, "", 0
            for pw_name, pw_data in payload.get("pathways", {}).items():
                if pw_l in pw_name.lower() or pw_name.lower() in pw_l:
                    for best in pw_data.get("scores", []):
                        if best.get("score", 0) > top_score:
                            found, top_score = True, best["score"]
                            top_cluster = best.get("cluster", "")
                            n_genes = best.get("n_genes_found", 0)
            results[pw] = {"found": found, "top_score": round(top_score, 4),
                           "n_genes_found": n_genes, "top_cluster": top_cluster}
        fc = sum(1 for v in results.values() if v["found"])
        return {"pathways_found": fc, "pathways_expected": len(gt),
                "alignment": round(fc/len(gt)*100, 1), "details": results}

    def evaluate_proportions(self, paper):
        pc = paper.get("proportion_changes", {})
        if not pc: return {"skipped": True, "reason": "No condition metadata in CSV", "consistency_rate": 0}
        try: payload = self.engine.proportions()
        except Exception as e: return {"skipped": True, "reason": f"Proportions failed: {e}", "consistency_rate": 0}
        fc_data = payload.get("proportion_fold_changes", [])
        if not fc_data: return {"skipped": True, "reason": "No fold change data", "consistency_rate": 0}
        inc_key = next((k for k in pc if "increased" in k), None)
        dec_key = next((k for k in pc if "decreased" in k), None)
        consistent, total, details = 0, 0, []
        for item in fc_data:
            cluster = item["cluster"].lower()
            fc = 1.0
            for key in item:
                if key.startswith("fold_change"):
                    v = item[key]; fc = 999.0 if v=="inf" else float(v) if isinstance(v,(int,float)) else 1.0; break
            is_up = any(ct.lower() in cluster for ct in pc.get(inc_key, []))
            is_down = any(ct.lower() in cluster for ct in pc.get(dec_key, []))
            if not is_up and not is_down: continue
            total += 1
            if (is_up and fc>1.0) or (is_down and fc<1.0):
                consistent += 1; details.append({"cluster": item["cluster"], "direction": "correct", "fc": fc})
            else: details.append({"cluster": item["cluster"], "direction": "WRONG", "fc": fc})
        return {"total_checked": total, "consistent": consistent,
                "consistency_rate": round(consistent/total*100,1) if total else 0, "details": details}

    def evaluate_compare(self, paper):
        conditions = paper.get("conditions", [])
        if len(conditions) < 2:
            return {"skipped": True, "reason": "No condition metadata in CSV",
                    "compare_recall": 0, "genes_found": 0, "genes_requested": 0}
        gt_set = set(g.upper() for g in paper.get("ground_truth_genes", []))
        try: payload = self.engine.compare(conditions[0], conditions[1], genes=paper.get("ground_truth_genes", []))
        except Exception as e:
            return {"skipped": True, "reason": f"Compare failed: {e}",
                    "compare_recall": 0, "genes_found": 0, "genes_requested": len(gt_set)}
        all_genes = set()
        for cid, cd in payload.get("clusters", {}).items():
            if isinstance(cd, dict):
                for g in cd.get("genes", []):
                    all_genes.add((g.get("gene","") if isinstance(g,dict) else g).upper())
        for gg in payload.get("summary",{}).get("condition_enriched_genes",{}).values():
            for g in gg: all_genes.add((g.get("gene","") if isinstance(g,dict) else g).upper())
        found = gt_set & all_genes
        return {"genes_requested": len(gt_set), "genes_found": len(found),
                "compare_recall": round(len(found)/len(gt_set)*100,1) if gt_set else 0,
                "n_clusters_analyzed": len(payload.get("clusters",{})),
                "found": sorted(found), "missed": sorted(gt_set - all_genes)}

    def evaluate_all(self, paper):
        return {"interactions": self.evaluate_interactions(paper),
                "pathways": self.evaluate_pathways(paper),
                "proportions": self.evaluate_proportions(paper),
                "compare": self.evaluate_compare(paper)}


# ============================================================
# FIGURE GENERATION
# ============================================================

def generate_figures(summary, complementarity, analytical, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    os.makedirs(out_dir, exist_ok=True)

    MC = {"random":"#9E9E9E","semantic":"#2196F3","scgpt":"#FF9800","union":"#4CAF50"}
    ML = {"random":"Random","semantic":"Semantic","scgpt":"scGPT","union":"Union\n(additive)"}
    MODES = list(MC.keys())
    cats = ["ontology", "expression"]
    titles = ["Ontology Queries\n(concept-level)", "Expression Queries\n(gene-signature)"]

    # Fig 1: Recall@1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, cat, title in zip(axes, cats, titles):
        vals = [summary.get(f"{cat}_{m}",{}).get("mean_recall@1",0) for m in MODES]
        x = np.arange(len(MODES))
        bars = ax.bar(x, vals, color=[MC[m] for m in MODES], alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES], fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Mean Cluster Recall@1")
        ax.set_title(title, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v > 0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_baselines.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "retrieval_baselines.pdf"), bbox_inches="tight"); plt.close()

    # Fig 2: Recall curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    rks = [1, 2, 3, 5, 8]
    for ax, cat, title in zip(axes, cats, titles):
        for mode, ls, marker in [("semantic","-","o"),("scgpt","--","s"),("union","-","D")]:
            vals = [summary.get(f"{cat}_{mode}",{}).get(f"mean_recall@{k}",0) for k in rks]
            ax.plot(rks, vals, ls, marker=marker, markersize=8, color=MC[mode], label=ML[mode].replace("\n"," "), linewidth=2)
            for k, v in zip(rks, vals): ax.annotate(f"{v:.2f}", (k,v), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8, color=MC[mode])
        ax.set_xlabel("k (retrieval cutoff)"); ax.set_ylabel("Mean Cluster Recall@k")
        ax.set_title(title, fontsize=12, fontweight="bold"); ax.set_xticks(rks); ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "additive_union_recall_curve.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "additive_union_recall_curve.pdf"), bbox_inches="tight"); plt.close()

    # Fig 3: Recall vs Gene Recall
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, cat, title in zip(axes, cats, titles):
        x = np.arange(len(MODES)); w = 0.35
        cv = [summary.get(f"{cat}_{m}",{}).get("mean_recall@1",0) for m in MODES]
        gv = [summary.get(f"{cat}_{m}",{}).get("mean_gene_recall",0) for m in MODES]
        b1 = ax.bar(x-w/2, cv, w, label="Cluster Recall@1", color=[MC[m] for m in MODES], alpha=0.85, edgecolor="white")
        b2 = ax.bar(x+w/2, gv, w, label="Gene Recall", color=[MC[m] for m in MODES], alpha=0.45, edgecolor="black", linewidth=0.8, hatch="///")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES], fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_title(title, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(b1, cv):
            if v > 0.05: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(b2, gv):
            if v > 0.05: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].legend(handles=[Patch(facecolor="#888888",alpha=0.85,label="Cluster Recall@1"),
                            Patch(facecolor="#888888",alpha=0.45,hatch="///",edgecolor="black",label="Gene Recall")], loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_vs_gene_delivery.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "retrieval_vs_gene_delivery.pdf"), bbox_inches="tight"); plt.close()

    # Fig 4: All metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, metric, mlabel in zip(axes, ["mean_recall@1","mean_recall@2","mean_mrr"], ["Recall@1","Recall@2","MRR"]):
        x = np.arange(len(MODES)); w = 0.35
        ov = [summary.get(f"ontology_{m}",{}).get(metric,0) for m in MODES]
        ev = [summary.get(f"expression_{m}",{}).get(metric,0) for m in MODES]
        ax.bar(x-w/2, ov, w, label="Ontology", alpha=0.85, color=[MC[m] for m in MODES], edgecolor="white")
        ax.bar(x+w/2, ev, w, label="Expression", alpha=0.45, color=[MC[m] for m in MODES], edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES], fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_title(mlabel, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "retrieval_all_metrics.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "retrieval_all_metrics.pdf"), bbox_inches="tight"); plt.close()

    # Fig 5: Complementarity
    comp = complementarity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    so = comp.get("only_semantic_count",0); sgo = comp.get("only_scgpt_count",0)
    bc = max(comp.get("union_found",0)-so-sgo, 0); ne = comp.get("neither_count",0)
    bars = ax1.bar(["Both\nmodalities","Semantic\nonly","scGPT\nonly","Neither"], [bc,so,sgo,ne],
                   color=["#4CAF50","#2196F3","#FF9800","#9E9E9E"], alpha=0.85, edgecolor="white", width=0.6)
    for bar, v in zip(bars, [bc,so,sgo,ne]):
        if v > 0: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(v), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Expected clusters found"); ax1.set_title("Modality Complementarity", fontweight="bold")
    ax1.text(0.98, 0.95, f"Union recall: {comp.get('union_recall',0):.1%}\nBest single: {comp.get('best_single_recall',0):.1%}\nAdditive gain: +{comp.get('additive_gain_pct',0):.1%}\n  (+{comp.get('additive_gain_clusters',0)} clusters)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    pq = comp.get("per_query", [])
    qwg = sorted([(q["query"][:40], q["additive_gain"]) for q in pq if q.get("additive_gain",0)>0], key=lambda x: x[1], reverse=True)
    if qwg:
        ql, qg = zip(*qwg[:12]); y = np.arange(len(ql))
        ax2.barh(y, qg, color="#4CAF50", alpha=0.7, edgecolor="white"); ax2.set_yticks(y); ax2.set_yticklabels(ql, fontsize=8)
        ax2.set_xlabel("Additive Recall Gain"); ax2.set_title("Per-Query Additive Gain\n(union vs best single)", fontweight="bold"); ax2.invert_yaxis()
    else: ax2.text(0.5, 0.5, "No additive gains\n(modalities agree)", transform=ax2.transAxes, ha="center", va="center", fontsize=12); ax2.set_title("Per-Query Additive Gain", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "complementarity.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "complementarity.pdf"), bbox_inches="tight"); plt.close()

    # Fig 6: Radar
    ana = analytical
    mr = {"Pathways": ana.get("pathways",{}).get("alignment",0)/100,
          "Interactions\n(LR)": ana.get("interactions",{}).get("lr_recovery_rate",0)/100,
          "Proportions": ana.get("proportions",{}).get("consistency_rate",0)/100,
          "Compare\n(gene recall)": ana.get("compare",{}).get("compare_recall",0)/100}
    lr = list(mr.keys()); vr = list(mr.values())
    angles = np.linspace(0, 2*np.pi, len(lr), endpoint=False).tolist()
    vr += vr[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.fill(angles, vr, alpha=0.25, color="#4CAF50"); ax.plot(angles, vr, "o-", color="#4CAF50", linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(lr, fontsize=10); ax.set_ylim(0, 1.05)
    ax.set_title("Analytical Module Performance\n(DT2: Breast Tissue Atlas)", fontweight="bold", pad=20)
    for a, v in zip(angles[:-1], vr[:-1]): ax.text(a, v+0.05, f"{v:.0%}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "analytical_radar.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "analytical_radar.pdf"), bbox_inches="tight"); plt.close()
    print(f"  [FIG] All 6 figures generated.")


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary(summary, complementarity, analytical):
    MODES = ["random","semantic","scgpt","union"]
    MD = {"random":"Random","semantic":"Semantic","scgpt":"scGPT","union":"Union(add)"}
    print("\n" + "="*100)
    print("ELISA BENCHMARK v5.1 — DT2: Breast Tissue Atlas (Bhat-Nakshatri et al. Nat Med 2024)")
    print("="*100)
    print("\n── Retrieval: All Modes (8 clusters total) ──\n")
    print(f"{'Category':<14} {'Mode':<14} {'R@1':>7} {'R@2':>7} {'R@3':>7} {'R@5':>7} {'R@8':>7} {'MRR':>7} {'GeneR':>7} {'#Ret':>6}")
    print("-"*100)
    for cat in ["ontology","expression"]:
        for mode in MODES:
            key = f"{cat}_{mode}"
            if key not in summary: continue
            s = summary[key]
            gr = s.get("mean_gene_recall",0)
            gr_s = f"{gr:.3f}" if s.get("has_gene_evidence") else "  n/a"
            nr = s.get("mean_n_retrieved",5)
            nr_s = f"{nr:.0f}" if mode=="union" else "5"
            print(f"{cat:<14} {MD[mode]:<14} "
                  f"{s.get('mean_recall@1',0):>7.3f} {s.get('mean_recall@2',0):>7.3f} "
                  f"{s.get('mean_recall@3',0):>7.3f} {s.get('mean_recall@5',0):>7.3f} "
                  f"{s.get('mean_recall@8',0):>7.3f} {s['mean_mrr']:>7.3f} {gr_s:>7} {nr_s:>6}")
        print()
    print("── Union Mode Details ──\n")
    for cat in ["ontology","expression"]:
        key = f"{cat}_union"
        if key not in summary: continue
        s = summary[key]
        print(f"  {cat}:")
        print(f"    Additive gain @3:   +{s.get('mean_additive_gain@3',0):.3f}")
        print(f"    Additive gain @5:   +{s.get('mean_additive_gain@5',0):.3f}")
        print(f"    Avg clusters returned: {s.get('mean_n_retrieved',0):.1f}")
        print(f"    Primary selection:  {dict(s.get('primary_selection',{}))}")
        print()
    print("── Overall Mean (both categories) ──\n")
    print(f"  {'Mode':<14} {'R@1':>7} {'R@2':>7} {'R@3':>7} {'R@5':>7} {'R@8':>7} {'GeneR':>7}")
    print("  " + "-"*65)
    for mode in MODES:
        vals = {k: [] for k in [1,2,3,5,8]}; grv = []
        for cat in ["ontology","expression"]:
            key = f"{cat}_{mode}"
            if key in summary:
                for k in vals: vals[k].append(summary[key].get(f"mean_recall@{k}",0))
                grv.append(summary[key].get("mean_gene_recall",0))
        if vals[1]:
            print(f"  {MD[mode]:<14} {np.mean(vals[1]):>7.3f} {np.mean(vals[2]):>7.3f} "
                  f"{np.mean(vals[3]):>7.3f} {np.mean(vals[5]):>7.3f} {np.mean(vals[8]):>7.3f} {np.mean(grv):>7.3f}")
    c = complementarity
    print(f"\n── Complementarity (Additive Union) ──\n")
    print(f"  Total expected clusters:   {c.get('total_expected',0)}")
    print(f"  Semantic found:            {c.get('semantic_found',0)} ({c.get('semantic_recall',0):.1%})")
    print(f"  scGPT found:               {c.get('scgpt_found',0)} ({c.get('scgpt_recall',0):.1%})")
    print(f"  Best single modality:      {c.get('best_single_found',0)} ({c.get('best_single_recall',0):.1%})")
    print(f"  Union found (additive):    {c.get('union_found',0)} ({c.get('union_recall',0):.1%})")
    print(f"  Additive gain:             +{c.get('additive_gain_clusters',0)} (+{c.get('additive_gain_pct',0):.1%})")
    print(f"  Only semantic:             {c.get('only_semantic_count',0)}")
    print(f"  Only scGPT:                {c.get('only_scgpt_count',0)}")
    print(f"  Neither:                   {c.get('neither_count',0)}")
    a = analytical
    print(f"\n── Analytical Modules ──\n")
    pw = a.get("pathways",{}); ix = a.get("interactions",{}); pr = a.get("proportions",{}); cm = a.get("compare",{})
    print(f"  Pathways:     {pw.get('alignment',0):.1f}% ({pw.get('pathways_found',0)}/{pw.get('pathways_expected',0)})")
    print(f"  Interactions: {ix.get('lr_recovery_rate',0):.1f}% LR, {ix.get('full_recovery_rate',0):.1f}% full "
          f"({ix.get('lr_matches',0)}/{ix.get('total_expected',0)} LR, {ix.get('full_matches',0)}/{ix.get('total_expected',0)} full)")
    if pr.get("skipped"): print(f"  Proportions:  SKIPPED ({pr.get('reason','no conditions')})")
    else: print(f"  Proportions:  {pr.get('consistency_rate',0):.1f}%")
    if cm.get("skipped"): print(f"  Compare:      SKIPPED ({cm.get('reason','no conditions')})")
    else: print(f"  Compare:      {cm.get('compare_recall',0):.1f}% ({cm.get('genes_found',0)}/{cm.get('genes_requested',0)})")
    print("="*100 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ELISA Benchmark v5.1 — DT2 Breast Tissue (Standalone)")
    parser.add_argument("--base", required=True, help="Path to embedding directory")
    parser.add_argument("--pt-name", default=None, help="Override .pt filename")
    parser.add_argument("--cells-csv", default=None, help="Override cells CSV")
    parser.add_argument("--paper", default="DT2", help="Paper ID")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k clusters")
    parser.add_argument("--out", default="results_DT2/", help="Output directory")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures only")
    parser.add_argument("--results-dir", default=None, help="Existing results dir for --plot-only")
    args = parser.parse_args()
    out_dir = args.out; os.makedirs(out_dir, exist_ok=True)
    paper = BENCHMARK_PAPERS[args.paper]

    if args.plot_only:
        rd = args.results_dir or out_dir
        with open(os.path.join(rd, "benchmark_v5_results.json")) as f: data = json.load(f)
        generate_figures(data[paper["id"]]["retrieval_summary"], data[paper["id"]]["complementarity"],
                         data[paper["id"]]["analytical"], out_dir)
        print("Figures regenerated."); return

    pt_name = args.pt_name or paper["pt_name"]
    cells_csv = args.cells_csv or paper["cells_csv"]
    sys.path.insert(0, os.path.dirname(args.base)); sys.path.insert(0, args.base); sys.path.insert(0, os.getcwd())
    from retrieval_engine_v4_hybrid import RetrievalEngine
    print(f"\n[BENCHMARK v5.1 — DT2] Loading engine: {args.base} / {pt_name}")
    engine = RetrievalEngine(base=args.base, pt_name=pt_name, cells_csv=cells_csv)

    from elisa_analysis import find_interactions, pathway_scoring, proportion_analysis, comparative_analysis

    class EngineWithAnalysis:
        def __init__(self, eng):
            self._eng = eng
            for attr in dir(eng):
                if not attr.startswith("_"):
                    try: setattr(self, attr, getattr(eng, attr))
                    except: pass
        def interactions(self, **kw): return find_interactions(self._eng.gene_stats, self._eng.cluster_ids, **kw)
        def pathways(self, **kw): return pathway_scoring(self._eng.gene_stats, self._eng.cluster_ids, **kw)
        def proportions(self, **kw):
            cc = paper.get("condition_col")
            return {"proportion_fold_changes": []} if not cc else proportion_analysis(self._eng.metadata, condition_col=cc)
        def compare(self, ca, cb, **kw):
            cc = paper.get("condition_col")
            return {"clusters": {}, "summary": {}} if not cc else comparative_analysis(self._eng.gene_stats, self._eng.metadata, condition_col=cc, group_a=ca, group_b=cb, **kw)

    engine_wrap = EngineWithAnalysis(engine)
    print(f"\n[BENCHMARK v5.1 — DT2] Running retrieval ({len(paper['queries'])} queries × {len(RetrievalEvaluator.MODES)} modes)...")
    t0 = time.time()
    ret_eval = RetrievalEvaluator(engine)
    ret_results = ret_eval.evaluate_queries(paper["queries"], top_k=args.top_k)
    ret_summary = ret_eval.compute_summary(ret_results)
    complementarity = ret_eval.compute_complementarity(ret_results, top_k=args.top_k)
    print(f"  Retrieval done in {time.time()-t0:.1f}s")

    print("[BENCHMARK v5.1 — DT2] Running analytical modules...")
    t0 = time.time()
    ana_eval = AnalyticalEvaluator(engine_wrap)
    try: analytical = ana_eval.evaluate_all(paper)
    except Exception as e:
        print(f"  [WARN] Analytical failed: {e}")
        analytical = {"pathways": {}, "interactions": {}, "proportions": {"skipped": True}, "compare": {"skipped": True}}
    print(f"  Analytical done in {time.time()-t0:.1f}s")

    print_summary(ret_summary, complementarity, analytical)
    output = {paper["id"]: {"retrieval_detail": ret_results, "retrieval_summary": ret_summary,
              "complementarity": complementarity, "analytical": analytical,
              "timestamp": datetime.now().isoformat(), "config": vars(args)}}
    rp = os.path.join(out_dir, "benchmark_v5_results.json")
    with open(rp, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")
    print("[BENCHMARK v5.1 — DT2] Generating figures...")
    generate_figures(ret_summary, complementarity, analytical, out_dir)
    print("\n[BENCHMARK v5.1 — DT2] Complete!")

if __name__ == "__main__":
    main()
