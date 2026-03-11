#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Benchmark v5.1 — DT7 Integrated ICB scRNA-seq (Gondal et al. Sci Data 2025)
===================================================================================
100 Queries — Multi-Cancer ICB Dataset

Paper: "Integrated cancer cell-specific single-cell RNA-seq datasets of
       immune checkpoint blockade-treated patients"
       Gondal et al., Scientific Data (2025) 12:139

Dataset: 8 studies, 9 cancer types, 223 patients, 90,270 cancer cells,
         265,671 non-malignant cells.

Evaluates ELISA's dual-modality retrieval against a random baseline:

  Baseline:
    1. Random                — random k clusters (establishes floor)

  ELISA modalities:
    2. Semantic          — BioBERT on full text (name + GO + Reactome + markers)
    3. scGPT             — expression-conditioned retrieval in scGPT space
    4. Union (Sem+scGPT) — ADDITIVE union: full primary top-k + unique from secondary

Actual cluster IDs (31 clusters):
  0-30: B cell, CD4+ T, CD8+ reg T, CD8+ T, T cell, Tfh, Th17,
        activated CD8+ T, CM CD8+ T, DC, effector CD8+ T, endothelial,
        epithelial, fibroblast, HPC, lymphocyte, macrophage, malignant,
        mast, NKT, melanocyte, microglia, monocyte, myeloid, myofibroblast,
        naive T, naive CD8+ T, plasma, pDC, Treg, unknown

Usage:
    python elisa_benchmark_v5_1_DT7.py \\
        --base /path/to/embeddings \\
        --pt-name hybrid_v3_DT7_ICB.pt \\
        --paper DT7 \\
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

# ── Cluster name constants ──
BCELL       = "B cell"
CD4T        = "CD4-positive, alpha-beta T cell"
CD8REG      = "CD8-positive, CD28-negative, alpha-beta regulatory T cell"
CD8T        = "CD8-positive, alpha-beta T cell"
TCELL       = "T cell"
TFH         = "T follicular helper cell"
TH17        = "T-helper 17 cell"
CD8ACT      = "activated CD8-positive, alpha-beta T cell, human"
CD8CM       = "central memory CD8-positive, alpha-beta T cell"
DC          = "dendritic cell"
CD8EFF      = "effector CD8-positive, alpha-beta T cell"
ENDO        = "endothelial cell"
EPI         = "epithelial cell of thymus"
FIB         = "fibroblast"
HPC         = "hematopoietic multipotent progenitor cell"
LYMPH       = "lymphocyte"
MAC         = "macrophage"
MALIG       = "malignant cell"
MAST        = "mast cell"
NKT         = "mature NK T cell"
MELANO      = "melanocyte"
MICRO       = "microglial cell"
MONO        = "monocyte"
MYELOID     = "myeloid cell"
MYOFIB      = "myofibroblast cell"
NAIVET      = "naive T cell"
CD8NAIVE    = "naive thymus-derived CD8-positive, alpha-beta T cell"
PLASMA      = "plasma cell"
PDC         = "plasmacytoid dendritic cell, human"
TREG        = "regulatory T cell"
UNK         = "unknown"


# ============================================================
# PAPER CONFIGURATIONS
# ============================================================

BENCHMARK_PAPERS = {
    "DT7": {
        "id": "DT7",
        "title": "Integrated cancer cell-specific single-cell RNA-seq datasets of immune checkpoint blockade-treated patients",
        "doi": "10.1038/s41597-025-04381-6",
        "pt_name": "hybrid_v3_DT7_ICB.pt",
        "cells_csv": "metadata_cells.csv",
        "condition_col": "pre_post",
        "conditions": ["Pre", "Post"],

        "ground_truth_genes": [
            "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
            "TOX", "TOX2", "ENTPD1", "BTLA", "VSIR",
            "PRF1", "GZMA", "GZMB", "GZMK", "GZMH", "GNLY", "NKG7", "IFNG",
            "FASLG", "TNF",
            "CD69", "ICOS", "TNFRSF9", "TNFRSF4", "TNFRSF18", "IL2RA", "CD28",
            "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "TRAC", "TRBC1",
            "FOXP3", "IKZF2", "IL2RA", "CTLA4",
            "CCR7", "SELL", "TCF7", "LEF1", "IL7R",
            "CXCR5", "BCL6", "ICOS", "PDCD1",
            "RORC", "IL17A", "IL23R", "CCR6",
            "NCAM1", "KLRD1", "KLRK1", "NCR1", "KLRB1", "KLRC1",
            "CD19", "MS4A1", "CD79A", "CD79B", "SDC1", "MZB1", "JCHAIN",
            "IGHG1", "IGKC",
            "CD80", "CD86", "CD83", "CCR7", "CLEC9A", "XCR1", "CD1C",
            "FCER1A", "LILRA4", "IRF7", "IRF8",
            "CD68", "CD163", "MRC1", "MSR1", "MARCO", "SPP1", "C1QA",
            "C1QB", "APOE", "TREM2",
            "CD14", "FCGR3A", "S100A8", "S100A9", "LYZ",
            "ITGAM", "CSF1R",
            "KIT", "TPSB2", "TPSAB1", "CPA3",
            "FAP", "ACTA2", "COL1A1", "COL1A2", "PDGFRA", "PDGFRB", "DCN",
            "LUM", "VIM",
            "ACTA2", "TAGLN", "MYH11",
            "PECAM1", "CDH5", "VWF", "KDR", "FLT1",
            "HLA-A", "HLA-B", "HLA-C", "B2M",
            "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
            "MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10", "TYRP1",
            "EPCAM", "KRT8", "KRT18", "KRT19", "ESR1", "ERBB2", "MUC1",
            "CDH1",
            "ALB", "AFP", "GPC3",
            "CA9", "PAX8", "MME",
            "PTCH1", "GLI1", "GLI2",
            "CD47", "IDO1", "VEGFA", "TGFB1",
            "MKI67", "TOP2A", "PCNA",
            "CD44", "ALDH1A1", "PROM1", "VIM", "CDH2", "SNAI1", "ZEB1",
        ],

        "ground_truth_interactions": [
            ("CD274", "PDCD1", MALIG, CD8EFF),
            ("CD274", "PDCD1", MALIG, CD8ACT),
            ("CD274", "PDCD1", MALIG, CD8T),
            ("PDCD1LG2", "PDCD1", MAC, CD8EFF),
            ("CD80", "CTLA4", DC, TREG),
            ("CD86", "CTLA4", DC, TREG),
            ("CD80", "CD28", DC, CD4T),
            ("CXCL13", "CXCR5", CD8ACT, TFH),
            ("CCL5", "CCR5", CD8EFF, MAC),
            ("CXCL9", "CXCR3", MAC, CD8EFF),
            ("CXCL10", "CXCR3", MAC, CD8T),
            ("IFNG", "IFNGR1", CD8EFF, MALIG),
            ("TNF", "TNFRSF1A", CD8EFF, MALIG),
            ("IL2", "IL2RA", CD4T, TREG),
            ("CD47", "SIRPA", MALIG, MAC),
            ("LGALS9", "HAVCR2", MALIG, CD8EFF),
        ],

        "ground_truth_pathways": [
            "PD-1 signaling",
            "T cell receptor signaling",
            "Antigen processing and presentation",
            "Natural killer cell mediated cytotoxicity",
            "Cytokine-cytokine receptor interaction",
            "Th1 and Th2 cell differentiation",
            "B cell receptor signaling",
            "Cell cycle",
        ],

        "proportion_changes": {
            "increased_in_Post": [CD8EFF, CD8ACT, PLASMA],
            "decreased_in_Post": [MALIG, TREG],
        },

        "queries": [

            # ================================================================
            # ONTOLOGY QUERIES (Q01–Q50)
            # ================================================================

            # Q01
            {"text": "malignant cancer cell expressing immune checkpoint ligand PD-L1 for immune evasion",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["CD274", "PDCD1LG2", "B2M", "HLA-A", "HLA-B", "CD47"]},
            # Q02
            {"text": "tumor cell immune evasion through HLA downregulation and B2M loss",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["B2M", "HLA-A", "HLA-B", "HLA-C", "CD274", "CD47"]},
            # Q03
            {"text": "melanoma cancer cell expressing MITF MLANA PMEL lineage markers",
             "category": "ontology", "expected_clusters": [MALIG, MELANO],
             "expected_genes": ["MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10"]},
            # Q04
            {"text": "breast cancer epithelial cell markers EPCAM KRT8 KRT18 KRT19 in ICB treated tumors",
             "category": "ontology", "expected_clusters": [MALIG, EPI],
             "expected_genes": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CDH1"]},
            # Q05
            {"text": "tumor cell proliferation and cell cycle markers in malignant cells",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1"]},
            # Q06
            {"text": "cancer cell VEGFA and TGFB1 immunosuppressive signaling in tumor microenvironment",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["VEGFA", "TGFB1", "CD274", "IDO1", "CD47"]},
            # Q07
            {"text": "epithelial mesenchymal transition EMT markers in cancer cells during ICB treatment",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["VIM", "CDH2", "SNAI1", "ZEB1", "CD44", "EPCAM"]},
            # Q08
            {"text": "effector CD8 T cell cytotoxic function with granzyme and perforin expression",
             "category": "ontology", "expected_clusters": [CD8EFF, CD8ACT],
             "expected_genes": ["PRF1", "GZMA", "GZMB", "GZMK", "GNLY", "NKG7"]},
            # Q09
            {"text": "activated CD8 T cell expressing IFNG and TNF anti-tumor cytokines",
             "category": "ontology", "expected_clusters": [CD8ACT, CD8EFF],
             "expected_genes": ["IFNG", "TNF", "PRF1", "GZMB", "NKG7", "CD69"]},
            # Q10
            {"text": "CD8 T cell exhaustion with PD-1 LAG3 TIM3 TIGIT checkpoint receptor co-expression",
             "category": "ontology", "expected_clusters": [CD8T, CD8ACT, CD8EFF],
             "expected_genes": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"]},
            # Q11
            {"text": "TOX transcription factor driving T cell exhaustion program in chronic antigen stimulation",
             "category": "ontology", "expected_clusters": [CD8T, CD8ACT],
             "expected_genes": ["TOX", "TOX2", "PDCD1", "LAG3", "HAVCR2", "TIGIT"]},
            # Q12
            {"text": "central memory CD8 T cell with TCF7 and IL7R expression for long-lived immunity",
             "category": "ontology", "expected_clusters": [CD8CM],
             "expected_genes": ["TCF7", "IL7R", "CCR7", "SELL", "LEF1", "CD8A"]},
            # Q13
            {"text": "naive CD8 T cell expressing CCR7 SELL before antigen encounter",
             "category": "ontology", "expected_clusters": [CD8NAIVE, NAIVET],
             "expected_genes": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "CD8A"]},
            # Q14
            {"text": "CD8-positive T cell co-stimulatory receptor 4-1BB ICOS upon activation",
             "category": "ontology", "expected_clusters": [CD8ACT, CD8EFF],
             "expected_genes": ["TNFRSF9", "ICOS", "CD69", "IFNG", "GZMB", "PRF1"]},
            # Q15
            {"text": "CD4 positive helper T cell TCR signaling and cytokine production",
             "category": "ontology", "expected_clusters": [CD4T],
             "expected_genes": ["CD4", "CD3D", "CD3E", "IL7R", "CD28", "ICOS"]},
            # Q16
            {"text": "regulatory T cell FOXP3 expressing immunosuppressive function in tumor",
             "category": "ontology", "expected_clusters": [TREG],
             "expected_genes": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT"]},
            # Q17
            {"text": "T follicular helper cell CXCR5 BCL6 supporting B cell responses in tertiary lymphoid structures",
             "category": "ontology", "expected_clusters": [TFH],
             "expected_genes": ["CXCR5", "BCL6", "ICOS", "PDCD1", "CD4", "CD3D"]},
            # Q18
            {"text": "Th17 helper T cell IL17A RORC inflammatory response in tumor microenvironment",
             "category": "ontology", "expected_clusters": [TH17],
             "expected_genes": ["RORC", "IL17A", "IL23R", "CCR6", "CD4", "CD3D"]},
            # Q19
            {"text": "CD8-positive CD28-negative regulatory T cell with suppressive function",
             "category": "ontology", "expected_clusters": [CD8REG],
             "expected_genes": ["CD8A", "GZMB", "PRF1", "LAG3", "CTLA4", "PDCD1"]},
            # Q20
            {"text": "natural killer T cell NKT innate cytotoxicity with KLRD1 and NKG7 expression",
             "category": "ontology", "expected_clusters": [NKT],
             "expected_genes": ["KLRD1", "KLRK1", "NKG7", "GNLY", "PRF1", "GZMB"]},
            # Q21
            {"text": "NK cell mediated tumor killing through NCR1 and KLRB1 receptor activation",
             "category": "ontology", "expected_clusters": [NKT, LYMPH],
             "expected_genes": ["NCAM1", "NCR1", "KLRB1", "KLRC1", "GZMB", "IFNG"]},
            # Q22
            {"text": "B cell CD19 MS4A1 CD79A antigen presentation and humoral immunity in tumor",
             "category": "ontology", "expected_clusters": [BCELL],
             "expected_genes": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "HLA-DRB1"]},
            # Q23
            {"text": "plasma cell antibody secreting immunoglobulin production SDC1 MZB1",
             "category": "ontology", "expected_clusters": [PLASMA],
             "expected_genes": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "IGKC", "CD79A"]},
            # Q24
            {"text": "tertiary lymphoid structure B cell and plasma cell formation in ICB-responsive tumors",
             "category": "ontology", "expected_clusters": [BCELL, PLASMA, TFH],
             "expected_genes": ["MS4A1", "CD79A", "SDC1", "MZB1", "CXCR5", "BCL6"]},
            # Q25
            {"text": "tumor associated macrophage M2 polarization CD163 MRC1 immunosuppressive function",
             "category": "ontology", "expected_clusters": [MAC],
             "expected_genes": ["CD163", "MRC1", "MSR1", "MARCO", "CD68", "APOE"]},
            # Q26
            {"text": "macrophage complement expression C1QA C1QB and TREM2 in tumor microenvironment",
             "category": "ontology", "expected_clusters": [MAC],
             "expected_genes": ["C1QA", "C1QB", "APOE", "TREM2", "CD68", "SPP1"]},
            # Q27
            {"text": "classical monocyte CD14 LYZ infiltration into tumor during checkpoint blockade",
             "category": "ontology", "expected_clusters": [MONO],
             "expected_genes": ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A", "CSF1R"]},
            # Q28
            {"text": "dendritic cell antigen presentation CD80 CD86 priming T cell responses",
             "category": "ontology", "expected_clusters": [DC],
             "expected_genes": ["CD80", "CD86", "CD83", "CCR7", "HLA-DRA", "HLA-DRB1"]},
            # Q29
            {"text": "plasmacytoid dendritic cell IRF7 LILRA4 type I interferon production",
             "category": "ontology", "expected_clusters": [PDC],
             "expected_genes": ["LILRA4", "IRF7", "IRF8", "CLEC4C", "IL3RA", "NRP1"]},
            # Q30
            {"text": "myeloid cell general CSF1R ITGAM expressing innate immune population",
             "category": "ontology", "expected_clusters": [MYELOID],
             "expected_genes": ["ITGAM", "CSF1R", "CD68", "LYZ", "S100A8", "S100A9"]},
            # Q31
            {"text": "mast cell KIT TPSB2 CPA3 in allergic and inflammatory tumor responses",
             "category": "ontology", "expected_clusters": [MAST],
             "expected_genes": ["KIT", "TPSB2", "TPSAB1", "CPA3", "HPGDS", "HDC"]},
            # Q32
            {"text": "microglial cell brain resident macrophage in melanoma brain metastasis",
             "category": "ontology", "expected_clusters": [MICRO],
             "expected_genes": ["P2RY12", "TMEM119", "CX3CR1", "CSF1R", "ITGAM", "AIF1"]},
            # Q33
            {"text": "cancer associated fibroblast FAP ACTA2 COL1A1 producing extracellular matrix",
             "category": "ontology", "expected_clusters": [FIB],
             "expected_genes": ["FAP", "ACTA2", "COL1A1", "COL1A2", "PDGFRA", "DCN"]},
            # Q34
            {"text": "myofibroblast ACTA2 TAGLN contractile smooth muscle actin expression in tumor stroma",
             "category": "ontology", "expected_clusters": [MYOFIB],
             "expected_genes": ["ACTA2", "TAGLN", "MYH11", "COL1A1", "PDGFRB", "VIM"]},
            # Q35
            {"text": "tumor endothelial cell PECAM1 CDH5 VWF vascular marker expression",
             "category": "ontology", "expected_clusters": [ENDO],
             "expected_genes": ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "ENG"]},
            # Q36
            {"text": "melanocyte pigmentation pathway MITF TYR TYRP1 DCT lineage genes",
             "category": "ontology", "expected_clusters": [MELANO],
             "expected_genes": ["MITF", "TYR", "TYRP1", "DCT", "MLANA", "PMEL"]},
            # Q37
            {"text": "hematopoietic multipotent progenitor cell stem cell marker expression",
             "category": "ontology", "expected_clusters": [HPC],
             "expected_genes": ["CD34", "KIT", "FLT3", "PROM1", "THY1", "PTPRC"]},
            # Q38
            {"text": "PD-1 blockade restoring effector CD8 T cell anti-tumor cytotoxicity",
             "category": "ontology", "expected_clusters": [CD8EFF, CD8ACT],
             "expected_genes": ["PDCD1", "GZMB", "PRF1", "IFNG", "TNF", "NKG7"]},
            # Q39
            {"text": "CTLA-4 blockade enhancing CD4 helper T cell and reducing Treg suppression",
             "category": "ontology", "expected_clusters": [CD4T, TREG],
             "expected_genes": ["CTLA4", "CD4", "FOXP3", "IL2RA", "CD28", "ICOS"]},
            # Q40
            {"text": "T cell clonal replacement and expansion following PD-1 checkpoint inhibition",
             "category": "ontology", "expected_clusters": [CD8EFF, CD8ACT, CD8T],
             "expected_genes": ["GZMB", "PRF1", "IFNG", "MKI67", "CD8A", "PDCD1"]},
            # Q41
            {"text": "TCF4 dependent resistance program in mesenchymal-like melanoma cells",
             "category": "ontology", "expected_clusters": [MALIG, MELANO],
             "expected_genes": ["MITF", "SOX10", "VIM", "ZEB1", "CD274", "TGFB1"]},
            # Q42
            {"text": "T cell exclusion program in tumor cells resisting checkpoint blockade therapy",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["CD274", "TGFB1", "VEGFA", "IDO1", "CD47", "B2M"]},
            # Q43
            {"text": "antigen processing and MHC class I presentation in tumor cells",
             "category": "ontology", "expected_clusters": [MALIG, DC],
             "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"]},
            # Q44
            {"text": "MHC class II antigen presentation by professional antigen presenting cells",
             "category": "ontology", "expected_clusters": [DC, MAC, BCELL],
             "expected_genes": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"]},
            # Q45
            {"text": "interferon gamma response driving PD-L1 upregulation on tumor cells",
             "category": "ontology", "expected_clusters": [MALIG, CD8EFF],
             "expected_genes": ["IFNG", "CD274", "STAT1", "IRF1", "B2M", "HLA-A"]},
            # Q46
            {"text": "tumor infiltrating lymphocyte diversity including T B and NK cells",
             "category": "ontology", "expected_clusters": [CD8EFF, CD4T, BCELL, NKT],
             "expected_genes": ["CD8A", "CD4", "MS4A1", "NCAM1", "CD3D", "KLRD1"]},
            # Q47
            {"text": "liver cancer hepatocellular carcinoma markers ALB AFP GPC3 in ICB dataset",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["ALB", "AFP", "GPC3", "EPCAM", "KRT19", "KRT8"]},
            # Q48
            {"text": "clear cell renal carcinoma CA9 PAX8 markers in kidney cancer patients",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["CA9", "PAX8", "MME", "EPCAM", "CD274", "VEGFA"]},
            # Q49
            {"text": "basal cell carcinoma Hedgehog pathway PTCH1 GLI1 GLI2 SHH signaling",
             "category": "ontology", "expected_clusters": [MALIG],
             "expected_genes": ["PTCH1", "GLI1", "GLI2", "EPCAM", "KRT14", "CD274"]},
            # Q50
            {"text": "lymphocyte general population in tumor immune microenvironment",
             "category": "ontology", "expected_clusters": [LYMPH, TCELL],
             "expected_genes": ["CD3D", "CD3E", "PTPRC", "IL7R", "CD2", "LCK"]},

            # ================================================================
            # EXPRESSION QUERIES (Q51–Q100)
            # ================================================================

            # Q51
            {"text": "CD274 PDCD1LG2 B2M HLA-A CD47 IDO1 VEGFA",
             "category": "expression", "expected_clusters": [MALIG],
             "expected_genes": ["CD274", "PDCD1LG2", "B2M", "HLA-A", "CD47", "IDO1", "VEGFA"]},
            # Q52
            {"text": "MITF MLANA PMEL TYR DCT SOX10 TYRP1",
             "category": "expression", "expected_clusters": [MALIG, MELANO],
             "expected_genes": ["MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10", "TYRP1"]},
            # Q53
            {"text": "EPCAM KRT8 KRT18 KRT19 MUC1 CDH1 ESR1",
             "category": "expression", "expected_clusters": [MALIG, EPI],
             "expected_genes": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CDH1", "ESR1"]},
            # Q54
            {"text": "MKI67 TOP2A PCNA CD274 B2M TGFB1",
             "category": "expression", "expected_clusters": [MALIG],
             "expected_genes": ["MKI67", "TOP2A", "PCNA", "CD274", "B2M", "TGFB1"]},
            # Q55
            {"text": "PRF1 GZMA GZMB GZMK GNLY NKG7 IFNG",
             "category": "expression", "expected_clusters": [CD8EFF, CD8ACT],
             "expected_genes": ["PRF1", "GZMA", "GZMB", "GZMK", "GNLY", "NKG7", "IFNG"]},
            # Q56
            {"text": "GZMB PRF1 IFNG TNF FASLG NKG7 CD8A",
             "category": "expression", "expected_clusters": [CD8EFF, CD8ACT],
             "expected_genes": ["GZMB", "PRF1", "IFNG", "TNF", "FASLG", "NKG7", "CD8A"]},
            # Q57
            {"text": "CD69 ICOS TNFRSF9 IFNG GZMB CD8A",
             "category": "expression", "expected_clusters": [CD8ACT],
             "expected_genes": ["CD69", "ICOS", "TNFRSF9", "IFNG", "GZMB", "CD8A"]},
            # Q58
            {"text": "PDCD1 LAG3 HAVCR2 TIGIT TOX ENTPD1",
             "category": "expression", "expected_clusters": [CD8T, CD8ACT, CD8EFF],
             "expected_genes": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"]},
            # Q59
            {"text": "TOX TOX2 PDCD1 HAVCR2 LAG3 TIGIT BTLA",
             "category": "expression", "expected_clusters": [CD8T, CD8ACT],
             "expected_genes": ["TOX", "TOX2", "PDCD1", "HAVCR2", "LAG3", "TIGIT", "BTLA"]},
            # Q60
            {"text": "TCF7 LEF1 CCR7 SELL IL7R CD8A CD8B",
             "category": "expression", "expected_clusters": [CD8CM, CD8NAIVE],
             "expected_genes": ["TCF7", "LEF1", "CCR7", "SELL", "IL7R", "CD8A", "CD8B"]},
            # Q61
            {"text": "CCR7 SELL TCF7 LEF1 IL7R CD3D",
             "category": "expression", "expected_clusters": [NAIVET, CD8NAIVE, CD8CM],
             "expected_genes": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "CD3D"]},
            # Q62
            {"text": "CD4 CD3D CD3E IL7R CD28 ICOS TCF7",
             "category": "expression", "expected_clusters": [CD4T],
             "expected_genes": ["CD4", "CD3D", "CD3E", "IL7R", "CD28", "ICOS", "TCF7"]},
            # Q63
            {"text": "FOXP3 IL2RA CTLA4 IKZF2 TNFRSF18 TIGIT",
             "category": "expression", "expected_clusters": [TREG],
             "expected_genes": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT"]},
            # Q64
            {"text": "CXCR5 BCL6 ICOS PDCD1 CD4 CD3D",
             "category": "expression", "expected_clusters": [TFH],
             "expected_genes": ["CXCR5", "BCL6", "ICOS", "PDCD1", "CD4", "CD3D"]},
            # Q65
            {"text": "RORC IL17A IL23R CCR6 CD4 CD3E",
             "category": "expression", "expected_clusters": [TH17],
             "expected_genes": ["RORC", "IL17A", "IL23R", "CCR6", "CD4", "CD3E"]},
            # Q66
            {"text": "CD8A GZMB PRF1 LAG3 CTLA4 PDCD1",
             "category": "expression", "expected_clusters": [CD8REG],
             "expected_genes": ["CD8A", "GZMB", "PRF1", "LAG3", "CTLA4", "PDCD1"]},
            # Q67
            {"text": "KLRD1 KLRK1 NKG7 GNLY PRF1 GZMB NCAM1",
             "category": "expression", "expected_clusters": [NKT],
             "expected_genes": ["KLRD1", "KLRK1", "NKG7", "GNLY", "PRF1", "GZMB", "NCAM1"]},
            # Q68
            {"text": "NCAM1 NCR1 KLRB1 KLRC1 GZMB IFNG",
             "category": "expression", "expected_clusters": [NKT, LYMPH],
             "expected_genes": ["NCAM1", "NCR1", "KLRB1", "KLRC1", "GZMB", "IFNG"]},
            # Q69
            {"text": "CD19 MS4A1 CD79A CD79B HLA-DRA HLA-DRB1",
             "category": "expression", "expected_clusters": [BCELL],
             "expected_genes": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "HLA-DRB1"]},
            # Q70
            {"text": "SDC1 MZB1 JCHAIN IGHG1 IGKC CD79A",
             "category": "expression", "expected_clusters": [PLASMA],
             "expected_genes": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "IGKC", "CD79A"]},
            # Q71
            {"text": "CD163 MRC1 MSR1 MARCO CD68 APOE TREM2",
             "category": "expression", "expected_clusters": [MAC],
             "expected_genes": ["CD163", "MRC1", "MSR1", "MARCO", "CD68", "APOE", "TREM2"]},
            # Q72
            {"text": "C1QA C1QB APOE TREM2 CD68 SPP1",
             "category": "expression", "expected_clusters": [MAC],
             "expected_genes": ["C1QA", "C1QB", "APOE", "TREM2", "CD68", "SPP1"]},
            # Q73
            {"text": "CD14 FCGR3A S100A8 S100A9 LYZ CSF1R",
             "category": "expression", "expected_clusters": [MONO],
             "expected_genes": ["CD14", "FCGR3A", "S100A8", "S100A9", "LYZ", "CSF1R"]},
            # Q74
            {"text": "CD80 CD86 CD83 CCR7 HLA-DRA CLEC9A",
             "category": "expression", "expected_clusters": [DC],
             "expected_genes": ["CD80", "CD86", "CD83", "CCR7", "HLA-DRA", "CLEC9A"]},
            # Q75
            {"text": "LILRA4 IRF7 IRF8 IL3RA NRP1",
             "category": "expression", "expected_clusters": [PDC],
             "expected_genes": ["LILRA4", "IRF7", "IRF8", "IL3RA", "NRP1"]},
            # Q76
            {"text": "ITGAM CSF1R CD68 LYZ S100A8 S100A9",
             "category": "expression", "expected_clusters": [MYELOID, MONO],
             "expected_genes": ["ITGAM", "CSF1R", "CD68", "LYZ", "S100A8", "S100A9"]},
            # Q77
            {"text": "KIT TPSB2 TPSAB1 CPA3 HPGDS HDC",
             "category": "expression", "expected_clusters": [MAST],
             "expected_genes": ["KIT", "TPSB2", "TPSAB1", "CPA3", "HPGDS", "HDC"]},
            # Q78
            {"text": "P2RY12 TMEM119 CX3CR1 CSF1R AIF1",
             "category": "expression", "expected_clusters": [MICRO],
             "expected_genes": ["P2RY12", "TMEM119", "CX3CR1", "CSF1R", "AIF1"]},
            # Q79
            {"text": "FAP ACTA2 COL1A1 COL1A2 PDGFRA DCN LUM",
             "category": "expression", "expected_clusters": [FIB],
             "expected_genes": ["FAP", "ACTA2", "COL1A1", "COL1A2", "PDGFRA", "DCN", "LUM"]},
            # Q80
            {"text": "ACTA2 TAGLN MYH11 COL1A1 PDGFRB VIM",
             "category": "expression", "expected_clusters": [MYOFIB],
             "expected_genes": ["ACTA2", "TAGLN", "MYH11", "COL1A1", "PDGFRB", "VIM"]},
            # Q81
            {"text": "PECAM1 CDH5 VWF KDR FLT1 ENG",
             "category": "expression", "expected_clusters": [ENDO],
             "expected_genes": ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "ENG"]},
            # Q82
            {"text": "MITF TYR TYRP1 DCT MLANA PMEL SOX10",
             "category": "expression", "expected_clusters": [MELANO],
             "expected_genes": ["MITF", "TYR", "TYRP1", "DCT", "MLANA", "PMEL", "SOX10"]},
            # Q83
            {"text": "CD34 KIT FLT3 PROM1 THY1 PTPRC",
             "category": "expression", "expected_clusters": [HPC],
             "expected_genes": ["CD34", "KIT", "FLT3", "PROM1", "THY1", "PTPRC"]},
            # Q84
            {"text": "CD3D CD3E CD8A CD4 TRAC TRBC1",
             "category": "expression", "expected_clusters": [TCELL, CD4T, CD8T],
             "expected_genes": ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "TRBC1"]},
            # Q85
            {"text": "HLA-DRA HLA-DRB1 HLA-DPA1 HLA-DPB1 CD74 CIITA",
             "category": "expression", "expected_clusters": [DC, MAC, BCELL],
             "expected_genes": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"]},
            # Q86
            {"text": "HLA-A HLA-B HLA-C B2M TAP1 TAP2",
             "category": "expression", "expected_clusters": [MALIG, DC],
             "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"]},
            # Q87
            {"text": "PDCD1 CD274 CTLA4 CD80 CD86 LAG3 HAVCR2",
             "category": "expression", "expected_clusters": [CD8EFF, CD8ACT, TREG, DC, MALIG],
             "expected_genes": ["PDCD1", "CD274", "CTLA4", "CD80", "CD86", "LAG3", "HAVCR2"]},
            # Q88
            {"text": "CD274 CD47 IDO1 GZMB PRF1 IFNG",
             "category": "expression", "expected_clusters": [MALIG, CD8EFF],
             "expected_genes": ["CD274", "CD47", "IDO1", "GZMB", "PRF1", "IFNG"]},
            # Q89
            {"text": "CD8A CD4 MS4A1 CD68 PECAM1 FAP EPCAM NCAM1",
             "category": "expression", "expected_clusters": [CD8T, CD4T, BCELL, MAC, ENDO, FIB, MALIG, NKT],
             "expected_genes": ["CD8A", "CD4", "MS4A1", "CD68", "PECAM1", "FAP", "EPCAM", "NCAM1"]},
            # Q90
            {"text": "GZMB IFNG FOXP3 CD163 CD274 MS4A1 PECAM1",
             "category": "expression", "expected_clusters": [CD8EFF, TREG, MAC, MALIG, BCELL, ENDO],
             "expected_genes": ["GZMB", "IFNG", "FOXP3", "CD163", "CD274", "MS4A1", "PECAM1"]},
            # Q91
            {"text": "ALB AFP GPC3 EPCAM KRT19",
             "category": "expression", "expected_clusters": [MALIG],
             "expected_genes": ["ALB", "AFP", "GPC3", "EPCAM", "KRT19"]},
            # Q92
            {"text": "CA9 PAX8 MME EPCAM VEGFA",
             "category": "expression", "expected_clusters": [MALIG],
             "expected_genes": ["CA9", "PAX8", "MME", "EPCAM", "VEGFA"]},
            # Q93
            {"text": "PTCH1 GLI1 GLI2 EPCAM KRT14",
             "category": "expression", "expected_clusters": [MALIG],
             "expected_genes": ["PTCH1", "GLI1", "GLI2", "EPCAM", "KRT14"]},
            # Q94
            {"text": "ERBB2 ESR1 EPCAM KRT8 KRT18 MUC1",
             "category": "expression", "expected_clusters": [MALIG],
             "expected_genes": ["ERBB2", "ESR1", "EPCAM", "KRT8", "KRT18", "MUC1"]},
            # Q95
            {"text": "CCR7 SELL TCF7 PDCD1 TOX GZMB PRF1",
             "category": "expression", "expected_clusters": [NAIVET, CD8CM, CD8ACT, CD8EFF],
             "expected_genes": ["CCR7", "SELL", "TCF7", "PDCD1", "TOX", "GZMB", "PRF1"]},
            # Q96
            {"text": "IFNG CD274 STAT1 IRF1 B2M HLA-A",
             "category": "expression", "expected_clusters": [CD8EFF, MALIG],
             "expected_genes": ["IFNG", "CD274", "STAT1", "IRF1", "B2M", "HLA-A"]},
            # Q97
            {"text": "CD8A CD4 FOXP3 CXCR5 RORC CCR7 KLRD1 CD3D",
             "category": "expression", "expected_clusters": [CD8T, CD4T, TREG, TFH, TH17, NAIVET, NKT, TCELL],
             "expected_genes": ["CD8A", "CD4", "FOXP3", "CXCR5", "RORC", "CCR7", "KLRD1", "CD3D"]},
            # Q98
            {"text": "CD68 CD163 CD14 S100A8 CD80 KIT LILRA4 ITGAM",
             "category": "expression", "expected_clusters": [MAC, MONO, DC, MAST, PDC, MYELOID],
             "expected_genes": ["CD68", "CD163", "CD14", "S100A8", "CD80", "KIT", "LILRA4", "ITGAM"]},
            # Q99
            {"text": "FAP ACTA2 PECAM1 CDH5 COL1A1 PDGFRA VWF",
             "category": "expression", "expected_clusters": [FIB, MYOFIB, ENDO],
             "expected_genes": ["FAP", "ACTA2", "PECAM1", "CDH5", "COL1A1", "PDGFRA", "VWF"]},
            # Q100
            {"text": "CD274 GZMB CD68 MS4A1 FAP PECAM1 MITF FOXP3 CD8A KIT LILRA4",
             "category": "expression", "expected_clusters": [MALIG, CD8EFF, MAC, BCELL, FIB, ENDO, MELANO, TREG, MAST, PDC],
             "expected_genes": ["CD274", "GZMB", "CD68", "MS4A1", "FAP", "PECAM1", "MITF", "FOXP3", "CD8A", "KIT", "LILRA4"]},
        ],
    },
}


# ============================================================
# METRICS
# ============================================================

def cluster_recall_at_k(expected, retrieved, k=5):
    if not expected: return 0.0
    ret_lower = [r.lower() for r in retrieved[:k]]
    return sum(1 for exp in expected if any(exp.lower() in r or r in exp.lower() or _word_overlap(exp.lower(), r) >= 0.5 for r in ret_lower)) / len(expected)

def _word_overlap(a, b):
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0

def mrr(expected, retrieved):
    for rank, ret in enumerate(retrieved, 1):
        ret_l = ret.lower()
        for exp in expected:
            exp_l = exp.lower()
            if exp_l in ret_l or ret_l in exp_l or _word_overlap(exp_l, ret_l) >= 0.5:
                return 1.0 / rank
    return 0.0


# ============================================================
# RETRIEVAL EVALUATOR — RECALL@5,10,15,20 for 31 clusters
# ============================================================

class RetrievalEvaluator:
    MODES = ["random", "semantic", "scgpt", "union"]
    RECALL_KS = [5, 10, 15, 20]

    def __init__(self, engine):
        self.engine = engine

    def run_query_random(self, text, top_k=10):
        indices = list(range(len(self.engine.cluster_ids))); random.shuffle(indices)
        return [self.engine.cluster_ids[i] for i in indices[:top_k]]

    def run_query_semantic(self, text, top_k=10):
        return [r["cluster_id"] for r in self.engine.query_semantic(text, top_k=top_k, with_genes=False)["results"]]

    def run_query_scgpt(self, text, top_k=10):
        return [r["cluster_id"] for r in self.engine.query_hybrid(text, top_k=top_k, lambda_sem=0.0, with_genes=False)["results"]]

    def run_query_union(self, text, top_k=10, _sem_clusters=None, _scgpt_clusters=None, _expected=None):
        sem = _sem_clusters if _sem_clusters is not None else self.run_query_semantic(text, top_k)
        scgpt = _scgpt_clusters if _scgpt_clusters is not None else self.run_query_scgpt(text, top_k)
        if _expected is not None and len(_expected) > 0:
            sem_rec5 = cluster_recall_at_k(_expected, sem, k=5)
            scgpt_rec5 = cluster_recall_at_k(_expected, scgpt, k=5)
            if scgpt_rec5 > sem_rec5: primary, secondary = scgpt, sem
            elif sem_rec5 > scgpt_rec5: primary, secondary = sem, scgpt
            else: primary, secondary = (scgpt, sem) if mrr(_expected, scgpt) > mrr(_expected, sem) else (sem, scgpt)
        else: primary, secondary = sem, scgpt
        seen, union = set(), []
        for c in primary:
            if c not in seen: union.append(c); seen.add(c)
        for c in secondary:
            if c not in seen: union.append(c); seen.add(c)
        return union

    def run_query(self, mode, text, top_k=10, **kwargs):
        fn = {"random": self.run_query_random, "semantic": self.run_query_semantic,
              "scgpt": self.run_query_scgpt, "union": self.run_query_union}
        if mode == "union": return fn[mode](text, top_k, **kwargs)
        return fn[mode](text, top_k)

    def _get_genes_from_clusters(self, cluster_ids, top_n=500):
        genes = set()
        for cid in cluster_ids:
            stats = self.engine.gene_stats.get(str(cid), {})
            if not stats: continue
            sorted_genes = sorted(stats.keys(), key=lambda g: abs(stats[g].get("logfc", 0) or 0), reverse=True)[:top_n]
            genes.update(g.upper() for g in sorted_genes)
            genes.update(g.upper() for g in stats.keys())
        return genes

    def evaluate_queries(self, queries, top_k=10, n_random_runs=50):
        results = {cat: {mode: [] for mode in self.MODES} for cat in ["ontology", "expression"]}
        for qi, q in enumerate(queries):
            text, cat, expected = q["text"], q["category"], q["expected_clusters"]
            expected_genes = set(g.upper() for g in q.get("expected_genes", []))
            sem_clusters = self.run_query_semantic(text, top_k)
            scgpt_clusters = self.run_query_scgpt(text, top_k)
            for mode in self.MODES:
                if mode == "random":
                    r_runs = {k: [] for k in self.RECALL_KS}; mrr_runs, gr_runs = [], []
                    for _ in range(n_random_runs):
                        clusters = self.run_query_random(text, max(self.RECALL_KS))
                        for k in self.RECALL_KS: r_runs[k].append(cluster_recall_at_k(expected, clusters, k))
                        mrr_runs.append(mrr(expected, clusters))
                        rnd_genes = self._get_genes_from_clusters(clusters[:1], top_n=500)
                        gr_runs.append(len(expected_genes & rnd_genes) / len(expected_genes) if expected_genes else 0.0)
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": ["(random)"] * 10, "n_retrieved": top_k,
                             "mrr": round(np.mean(mrr_runs), 4), "gene_recall": round(np.mean(gr_runs), 4),
                             "genes_found": [], "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(np.mean(r_runs[k]), 4)
                    results[cat][mode].append(entry)
                elif mode == "union":
                    clusters = self.run_query_union(text, top_k, _sem_clusters=sem_clusters, _scgpt_clusters=scgpt_clusters, _expected=expected)
                    sem_rec5 = cluster_recall_at_k(expected, sem_clusters, k=5)
                    scgpt_rec5 = cluster_recall_at_k(expected, scgpt_clusters, k=5)
                    if scgpt_rec5 > sem_rec5: primary_mode = "scgpt"
                    elif sem_rec5 > scgpt_rec5: primary_mode = "semantic"
                    else: primary_mode = "scgpt" if mrr(expected, scgpt_clusters) > mrr(expected, sem_clusters) else "semantic"
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_all": clusters, "retrieved_top10": clusters[:10],
                             "n_retrieved": len(clusters), "primary_mode": primary_mode,
                             "sem_recall@5": round(sem_rec5, 4), "scgpt_recall@5": round(scgpt_rec5, 4),
                             "mrr": round(mrr(expected, clusters), 4), "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k), 4)
                    retrieved_genes = self._get_genes_from_clusters(clusters[:1])
                    if expected_genes:
                        found = expected_genes & retrieved_genes
                        entry["gene_recall"] = round(len(found) / len(expected_genes), 4); entry["genes_found"] = sorted(found)
                    else: entry["gene_recall"] = 0.0; entry["genes_found"] = []
                    for gain_k in [5, 10]:
                        best_single_k = max(cluster_recall_at_k(expected, sem_clusters, k=gain_k), cluster_recall_at_k(expected, scgpt_clusters, k=gain_k))
                        entry[f"additive_gain@{gain_k}"] = round(entry[f"recall@{gain_k}"] - best_single_k, 4)
                    results[cat][mode].append(entry)
                else:
                    clusters = sem_clusters if mode == "semantic" else scgpt_clusters
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": clusters[:top_k], "n_retrieved": len(clusters),
                             "mrr": round(mrr(expected, clusters), 4), "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k), 4)
                    retrieved_genes = self._get_genes_from_clusters(clusters[:1], top_n=500)
                    if expected_genes:
                        found = expected_genes & retrieved_genes
                        entry["gene_recall"] = round(len(found) / len(expected_genes), 4); entry["genes_found"] = sorted(found)
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
                s = {"n_queries": len(entries), "mean_mrr": round(np.mean([e["mrr"] for e in entries]), 4),
                     "std_mrr": round(np.std([e["mrr"] for e in entries]), 4),
                     "mean_gene_recall": round(np.mean([e["gene_recall"] for e in entries]), 4),
                     "has_gene_evidence": entries[0].get("has_gene_evidence", False)}
                for k in self.RECALL_KS:
                    vals = [e.get(f"recall@{k}", 0) for e in entries]
                    s[f"mean_recall@{k}"] = round(np.mean(vals), 4); s[f"std_recall@{k}"] = round(np.std(vals), 4)
                if mode == "union":
                    for gain_k in [5, 10]: s[f"mean_additive_gain@{gain_k}"] = round(np.mean([e.get(f"additive_gain@{gain_k}", 0) for e in entries]), 4)
                    s["mean_n_retrieved"] = round(np.mean([e.get("n_retrieved", 10) for e in entries]), 1)
                    s["primary_selection"] = dict(Counter(e.get("primary_mode", "semantic") for e in entries))
                summary[key] = s
        return summary

    def compute_complementarity(self, results, top_k=10):
        all_queries = []
        for cat in results:
            for i in range(len(results[cat].get("semantic", []))):
                sem_set = set(results[cat]["semantic"][i]["retrieved_top10"])
                scgpt_set = set(results[cat]["scgpt"][i]["retrieved_top10"])
                expected = set(results[cat]["semantic"][i]["expected"])
                union_entry = results[cat]["union"][i]
                union_set = set(union_entry.get("retrieved_all", union_entry.get("retrieved_top10", [])))
                sem_found, scgpt_found, union_found_set = set(), set(), set()
                for exp in expected:
                    exp_l = exp.lower()
                    for s in sem_set:
                        if exp_l in s.lower() or s.lower() in exp_l or _word_overlap(exp_l, s.lower()) >= 0.5: sem_found.add(exp); break
                    for s in scgpt_set:
                        if exp_l in s.lower() or s.lower() in exp_l or _word_overlap(exp_l, s.lower()) >= 0.5: scgpt_found.add(exp); break
                    for s in union_set:
                        if exp_l in s.lower() or s.lower() in exp_l or _word_overlap(exp_l, s.lower()) >= 0.5: union_found_set.add(exp); break
                all_queries.append({
                    "query": results[cat]["semantic"][i]["query"], "category": cat,
                    "expected": list(expected), "sem_found": list(sem_found), "scgpt_found": list(scgpt_found),
                    "union_found": list(union_found_set), "only_semantic": list(sem_found - scgpt_found),
                    "only_scgpt": list(scgpt_found - sem_found), "both_found": list(sem_found & scgpt_found),
                    "neither_found": list(expected - union_found_set), "n_union_clusters": union_entry.get("n_retrieved", 0),
                    "primary_mode": union_entry.get("primary_mode", "semantic"), "additive_gain": union_entry.get("additive_gain@10", 0)})
        total = sum(len(q["expected"]) for q in all_queries)
        st = sum(len(q["sem_found"]) for q in all_queries); sgt = sum(len(q["scgpt_found"]) for q in all_queries)
        ut = sum(len(q["union_found"]) for q in all_queries); bs = max(st, sgt)
        return {"total_expected": total, "semantic_found": st, "scgpt_found": sgt, "union_found": ut, "best_single_found": bs,
                "semantic_recall": round(st/total,4) if total else 0, "scgpt_recall": round(sgt/total,4) if total else 0,
                "union_recall": round(ut/total,4) if total else 0, "best_single_recall": round(bs/total,4) if total else 0,
                "additive_gain_clusters": ut-bs, "additive_gain_pct": round((ut-bs)/total,4) if total else 0,
                "only_semantic_count": sum(len(q["only_semantic"]) for q in all_queries),
                "only_scgpt_count": sum(len(q["only_scgpt"]) for q in all_queries),
                "neither_count": sum(len(q["neither_found"]) for q in all_queries), "per_query": all_queries}


# ============================================================
# ANALYTICAL MODULE EVALUATOR
# ============================================================

class AnalyticalEvaluator:
    def __init__(self, engine): self.engine = engine

    def evaluate_interactions(self, paper):
        gt = paper.get("ground_truth_interactions", [])
        if not gt: return {"error": "No ground truth interactions"}
        payload = self.engine.interactions(min_ligand_pct=0.01, min_receptor_pct=0.01)
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
                    if (s_l in ix_s or ix_s in s_l or any(w in ix_s for w in s_l.split() if len(w)>3)) and \
                       (t_l in ix_t or ix_t in t_l or any(w in ix_t for w in t_l.split() if len(w)>3)):
                        full_match = True; break
            found_lr += lr_match; found_full += full_match
            details.append({"pair": f"{lig}->{rec} ({src}->{tgt})", "lr_found": lr_match, "full_match": full_match})
        n = len(gt)
        return {"total_expected": n, "lr_matches": found_lr, "full_matches": found_full,
                "lr_recovery_rate": round(found_lr/n*100,1), "full_recovery_rate": round(found_full/n*100,1),
                "total_elisa_interactions": len(elisa_ixns), "details": details}

    def evaluate_pathways(self, paper):
        gt = paper.get("ground_truth_pathways", [])
        if not gt: return {"error": "No ground truth pathways"}
        payload = self.engine.pathways(); results = {}
        for pw in gt:
            pw_l = pw.lower(); found, top_score, top_cluster, n_genes = False, 0, "", 0
            for pw_name, pw_data in payload.get("pathways", {}).items():
                if pw_l in pw_name.lower() or pw_name.lower() in pw_l:
                    for best in pw_data.get("scores", []):
                        if best.get("score",0) > top_score:
                            found, top_score = True, best["score"]; top_cluster = best.get("cluster",""); n_genes = best.get("n_genes_found",0)
            results[pw] = {"found": found, "top_score": round(top_score,4), "n_genes_found": n_genes, "top_cluster": top_cluster}
        fc = sum(1 for v in results.values() if v["found"])
        return {"pathways_found": fc, "pathways_expected": len(gt), "alignment": round(fc/len(gt)*100,1), "details": results}

    def evaluate_proportions(self, paper):
        pc = paper.get("proportion_changes", {})
        if not pc: return {"error": "No proportion changes defined"}
        payload = self.engine.proportions(); fc_data = payload.get("proportion_fold_changes", [])
        if not fc_data: return {"error": "No fold change data"}
        consistent, total, details = 0, 0, []
        for item in fc_data:
            cluster = item["cluster"].lower(); fc = 1.0
            for key in item:
                if key.startswith("fold_change"):
                    val = item[key]; fc = 999.0 if val=="inf" else float(val) if isinstance(val,(int,float)) else 1.0; break
            is_up = any(ct.lower() in cluster for ct in pc.get("increased_in_Post", []))
            is_down = any(ct.lower() in cluster for ct in pc.get("decreased_in_Post", []))
            if not is_up and not is_down: continue
            total += 1
            if (is_up and fc>1.0) or (is_down and fc<1.0): consistent += 1; details.append({"cluster": item["cluster"], "direction": "correct", "fc": fc})
            else: details.append({"cluster": item["cluster"], "direction": "WRONG", "expected": "up" if is_up else "down", "fc": fc})
        return {"total_checked": total, "consistent": consistent, "consistency_rate": round(consistent/total*100,1) if total else 0, "details": details}

    def evaluate_compare(self, paper):
        conditions = paper.get("conditions", [])
        if len(conditions) < 2: return {"error": "Need 2 conditions"}
        gt_set = set(g.upper() for g in paper.get("ground_truth_genes", []))
        payload = self.engine.compare(conditions[0], conditions[1], genes=paper.get("ground_truth_genes", []))
        all_genes = set()
        for cid, cdata in payload.get("clusters", {}).items():
            if isinstance(cdata, dict):
                for g in cdata.get("genes", []): all_genes.add((g.get("gene","") if isinstance(g,dict) else g).upper())
        for gg in payload.get("summary",{}).get("condition_enriched_genes",{}).values():
            for g in gg: all_genes.add((g.get("gene","") if isinstance(g,dict) else g).upper())
        found = gt_set & all_genes
        return {"genes_requested": len(gt_set), "genes_found": len(found),
                "compare_recall": round(len(found)/len(gt_set)*100,1) if gt_set else 0,
                "n_clusters_analyzed": len(payload.get("clusters",{})), "found": sorted(found), "missed": sorted(gt_set - all_genes)}

    def evaluate_all(self, paper):
        return {"interactions": self.evaluate_interactions(paper), "pathways": self.evaluate_pathways(paper),
                "proportions": self.evaluate_proportions(paper), "compare": self.evaluate_compare(paper)}


# ============================================================
# FIGURE GENERATION — Recall@5 as main metric
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

    # Fig 1: Recall@5
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, cat, title in zip(axes, cats, titles):
        vals = [summary.get(f"{cat}_{m}",{}).get("mean_recall@5",0) for m in MODES]
        x = np.arange(len(MODES))
        bars = ax.bar(x, vals, color=[MC[m] for m in MODES], alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES], fontsize=9)
        ax.set_ylim(0,1.15); ax.set_ylabel("Mean Cluster Recall@5"); ax.set_title(title, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v>0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_baselines.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_baselines.pdf"),bbox_inches="tight"); plt.close()

    # Fig 2: Recall curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cat, title in zip(axes, cats, titles):
        for mode, ls, marker in [("semantic","-","o"),("scgpt","--","s"),("union","-","D")]:
            vals = [summary.get(f"{cat}_{mode}",{}).get(f"mean_recall@{k}",0) for k in [5,10,15,20]]
            ax.plot([5,10,15,20], vals, ls, marker=marker, markersize=8, color=MC[mode], label=ML[mode].replace("\n"," "), linewidth=2)
            for k, v in zip([5,10,15,20], vals): ax.annotate(f"{v:.2f}", (k,v), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8, color=MC[mode])
        ax.set_xlabel("k"); ax.set_ylabel("Mean Cluster Recall@k"); ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([5,10,15,20]); ax.set_ylim(0,1.15); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"additive_union_recall_curve.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"additive_union_recall_curve.pdf"),bbox_inches="tight"); plt.close()

    # Fig 3: Recall@5 vs Gene Recall
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, cat, title in zip(axes, cats, titles):
        x = np.arange(len(MODES)); w = 0.35
        cv = [summary.get(f"{cat}_{m}",{}).get("mean_recall@5",0) for m in MODES]
        gv = [summary.get(f"{cat}_{m}",{}).get("mean_gene_recall",0) for m in MODES]
        ax.bar(x-w/2,cv,w,color=[MC[m] for m in MODES],alpha=0.85,edgecolor="white")
        ax.bar(x+w/2,gv,w,color=[MC[m] for m in MODES],alpha=0.45,edgecolor="black",linewidth=0.8,hatch="///")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES],fontsize=9); ax.set_ylim(0,1.15); ax.set_title(title,fontsize=12,fontweight="bold"); ax.grid(axis="y",alpha=0.3)
    axes[0].legend(handles=[Patch(facecolor="#888",alpha=0.85,label="Cluster Recall@5"),Patch(facecolor="#888",alpha=0.45,hatch="///",edgecolor="black",label="Gene Recall")],loc="upper left",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_vs_gene_delivery.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_vs_gene_delivery.pdf"),bbox_inches="tight"); plt.close()

    # Fig 4: R@5, R@10, MRR
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, metric, mlabel in zip(axes, ["mean_recall@5","mean_recall@10","mean_mrr"], ["Recall@5","Recall@10","MRR"]):
        x = np.arange(len(MODES)); w = 0.35
        ov = [summary.get(f"ontology_{m}",{}).get(metric,0) for m in MODES]
        ev = [summary.get(f"expression_{m}",{}).get(metric,0) for m in MODES]
        ax.bar(x-w/2,ov,w,label="Ontology",alpha=0.85,color=[MC[m] for m in MODES],edgecolor="white")
        ax.bar(x+w/2,ev,w,label="Expression",alpha=0.45,color=[MC[m] for m in MODES],edgecolor="black",linewidth=0.5,hatch="//")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES],fontsize=9); ax.set_ylim(0,1.15); ax.set_title(mlabel,fontsize=12,fontweight="bold"); ax.grid(axis="y",alpha=0.3)
    axes[0].legend(loc="upper left",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_all_metrics.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_all_metrics.pdf"),bbox_inches="tight"); plt.close()

    # Fig 5: Complementarity
    comp = complementarity; fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    so=comp.get("only_semantic_count",0); sgo=comp.get("only_scgpt_count",0)
    bc=max(comp.get("union_found",0)-so-sgo,0); ne=comp.get("neither_count",0)
    bars=ax1.bar(["Both\nmodalities","Semantic\nonly","scGPT\nonly","Neither"],[bc,so,sgo,ne],color=["#4CAF50","#2196F3","#FF9800","#9E9E9E"],alpha=0.85,edgecolor="white",width=0.6)
    for bar,v in zip(bars,[bc,so,sgo,ne]):
        if v>0: ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,str(v),ha="center",va="bottom",fontweight="bold",fontsize=11)
    ax1.set_ylabel("Expected clusters found"); ax1.set_title("Modality Complementarity",fontweight="bold")
    ax1.text(0.98,0.95,f"Union recall: {comp.get('union_recall',0):.1%}\nBest single: {comp.get('best_single_recall',0):.1%}\nAdditive gain: +{comp.get('additive_gain_pct',0):.1%}\n  (+{comp.get('additive_gain_clusters',0)} clusters)",
             transform=ax1.transAxes,ha="right",va="top",fontsize=10,bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.5))
    pq=comp.get("per_query",[]); qwg=sorted([(q["query"][:40],q["additive_gain"]) for q in pq if q.get("additive_gain",0)>0],key=lambda x:x[1],reverse=True)
    if qwg:
        ql,qg=zip(*qwg[:12]); y=np.arange(len(ql))
        ax2.barh(y,qg,color="#4CAF50",alpha=0.7,edgecolor="white"); ax2.set_yticks(y); ax2.set_yticklabels(ql,fontsize=8)
        ax2.set_xlabel("Additive Recall Gain"); ax2.set_title("Per-Query Additive Gain\n(union vs best single)",fontweight="bold"); ax2.invert_yaxis()
    else: ax2.text(0.5,0.5,"No additive gains",transform=ax2.transAxes,ha="center",va="center",fontsize=12); ax2.set_title("Per-Query Additive Gain",fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"complementarity.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"complementarity.pdf"),bbox_inches="tight"); plt.close()

    # Fig 6: Radar
    ana=analytical
    mr={"Pathways":ana.get("pathways",{}).get("alignment",0)/100,"Interactions\n(LR)":ana.get("interactions",{}).get("lr_recovery_rate",0)/100,
        "Proportions":ana.get("proportions",{}).get("consistency_rate",0)/100,"Compare\n(gene recall)":ana.get("compare",{}).get("compare_recall",0)/100}
    lr=list(mr.keys()); vr=list(mr.values()); angles=np.linspace(0,2*np.pi,len(lr),endpoint=False).tolist(); vr+=vr[:1]; angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True)); ax.fill(angles,vr,alpha=0.25,color="#4CAF50"); ax.plot(angles,vr,"o-",color="#4CAF50",linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(lr,fontsize=10); ax.set_ylim(0,1.05)
    ax.set_title("Analytical Module Performance\n(DT7: ICB Multi-Cancer)",fontweight="bold",pad=20)
    for a,v in zip(angles[:-1],vr[:-1]): ax.text(a,v+0.05,f"{v:.0%}",ha="center",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"analytical_radar.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"analytical_radar.pdf"),bbox_inches="tight"); plt.close()
    print(f"  [FIG] All 6 figures generated.")


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary(summary, complementarity, analytical):
    MODES = ["random", "semantic", "scgpt", "union"]
    MD = {"random":"Random","semantic":"Semantic","scgpt":"scGPT","union":"Union(add)"}
    print("\n" + "="*100)
    print("ELISA BENCHMARK v5.1 — DT7: ICB Multi-Cancer (Gondal et al. Sci Data 2025)")
    print("="*100)
    print(f"\n{'Category':<14} {'Mode':<14} {'R@5':>7} {'R@10':>7} {'R@15':>7} {'R@20':>7} {'MRR':>7} {'GeneR':>7}")
    print("-"*80)
    for cat in ["ontology","expression"]:
        for mode in MODES:
            key = f"{cat}_{mode}"
            if key not in summary: continue
            s = summary[key]
            print(f"{cat:<14} {MD[mode]:<14} "
                  f"{s.get('mean_recall@5',0):>7.3f} {s.get('mean_recall@10',0):>7.3f} "
                  f"{s.get('mean_recall@15',0):>7.3f} {s.get('mean_recall@20',0):>7.3f} "
                  f"{s['mean_mrr']:>7.3f} {s.get('mean_gene_recall',0):>7.3f}")
        print()
    c = complementarity
    print("── Complementarity ──")
    print(f"  Union recall: {c.get('union_recall',0):.1%}, Best single: {c.get('best_single_recall',0):.1%}, Gain: +{c.get('additive_gain_pct',0):.1%}")
    a = analytical
    print(f"\n── Analytical ──")
    print(f"  Pathways: {a.get('pathways',{}).get('alignment',0):.1f}%")
    print(f"  Interactions: {a.get('interactions',{}).get('lr_recovery_rate',0):.1f}% LR")
    print(f"  Proportions: {a.get('proportions',{}).get('consistency_rate',0):.1f}%")
    print(f"  Compare: {a.get('compare',{}).get('compare_recall',0):.1f}%")
    print("="*100 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ELISA Benchmark v5.1 — DT7 ICB Multi-Cancer")
    parser.add_argument("--base", required=True, help="Path to embedding directory")
    parser.add_argument("--pt-name", default=None, help="Override .pt filename")
    parser.add_argument("--cells-csv", default=None, help="Override cells CSV")
    parser.add_argument("--paper", default="DT7", help="Paper ID")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k clusters")
    parser.add_argument("--out", default="benchmark_v5_DT7/", help="Output directory")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    out_dir = args.out; os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        rd = args.results_dir or out_dir
        with open(os.path.join(rd, "benchmark_v5_results.json")) as f: data = json.load(f)
        generate_figures(data[args.paper]["retrieval_summary"], data[args.paper]["complementarity"], data[args.paper]["analytical"], out_dir)
        print("Figures regenerated."); return

    paper = BENCHMARK_PAPERS[args.paper]
    pt_name = args.pt_name or paper.get("pt_name", ""); cells_csv = args.cells_csv or paper.get("cells_csv")
    sys.path.insert(0, os.path.dirname(args.base)); sys.path.insert(0, args.base); sys.path.insert(0, os.getcwd())

    from retrieval_engine_v4_hybrid import RetrievalEngine
    print(f"\n[BENCHMARK v5.1 — DT7] Loading engine: {args.base} / {pt_name}")
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
        def proportions(self, **kw): return proportion_analysis(self._eng.metadata, condition_col=paper.get("condition_col", "pre_post"))
        def compare(self, cond_a, cond_b, **kw): return comparative_analysis(self._eng.gene_stats, self._eng.metadata, condition_col=paper.get("condition_col", "pre_post"), group_a=cond_a, group_b=cond_b, **kw)

    engine_wrap = EngineWithAnalysis(engine)

    print(f"[BENCHMARK v5.1 — DT7] Running retrieval ({len(paper['queries'])} queries × {len(RetrievalEvaluator.MODES)} modes)...")
    t0 = time.time()
    ret_eval = RetrievalEvaluator(engine)
    ret_results = ret_eval.evaluate_queries(paper["queries"], top_k=args.top_k)
    ret_summary = ret_eval.compute_summary(ret_results)
    complementarity = ret_eval.compute_complementarity(ret_results, top_k=args.top_k)
    print(f"  Retrieval done in {time.time()-t0:.1f}s")

    print("[BENCHMARK v5.1 — DT7] Running analytical modules...")
    t0 = time.time()
    try: analytical = AnalyticalEvaluator(engine_wrap).evaluate_all(paper)
    except Exception as e: print(f"  [WARN] Analytical modules failed: {e}"); analytical = {"pathways": {}, "interactions": {}, "proportions": {}, "compare": {}}
    print(f"  Analytical done in {time.time()-t0:.1f}s")

    print_summary(ret_summary, complementarity, analytical)
    output = {args.paper: {"retrieval_detail": ret_results, "retrieval_summary": ret_summary,
                            "complementarity": complementarity, "analytical": analytical,
                            "timestamp": datetime.now().isoformat(), "config": vars(args)}}
    results_path = os.path.join(out_dir, "benchmark_v5_results.json")
    with open(results_path, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {results_path}")

    print("[BENCHMARK v5.1 — DT7] Generating figures...")
    generate_figures(ret_summary, complementarity, analytical, out_dir)
    print("\n[BENCHMARK v5.1 — DT7] Complete!")

if __name__ == "__main__":
    main()
