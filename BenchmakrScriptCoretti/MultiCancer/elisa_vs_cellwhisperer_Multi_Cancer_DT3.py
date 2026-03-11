#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA vs CellWhisperer — DT7 Integrated ICB scRNA-seq (Gondal et al. Sci Data 2025)
====================================================================================
Head-to-Head comparison on multi-cancer ICB scRNA-seq data.

31 clusters:
  0:  B cell                                          16: macrophage
  1:  CD4-positive, alpha-beta T cell                 17: malignant cell
  2:  CD8-positive, CD28-negative, alpha-beta reg T   18: mast cell
  3:  CD8-positive, alpha-beta T cell                 19: mature NK T cell
  4:  T cell                                          20: melanocyte
  5:  T follicular helper cell                        21: microglial cell
  6:  T-helper 17 cell                                22: monocyte
  7:  activated CD8-positive, alpha-beta T cell       23: myeloid cell
  8:  central memory CD8-positive, alpha-beta T cell  24: myofibroblast cell
  9:  dendritic cell                                  25: naive T cell
  10: effector CD8-positive, alpha-beta T cell        26: naive thymus-derived CD8+ T cell
  11: endothelial cell                                27: plasma cell
  12: epithelial cell of thymus                       28: plasmacytoid dendritic cell
  13: fibroblast                                      29: regulatory T cell
  14: hematopoietic multipotent progenitor cell       30: unknown
  15: lymphocyte

Step 1: Run ELISA benchmark first:
    python elisa_benchmark_v5_1_DT7.py \\
        --base /path/to/embeddings \\
        --pt-name hybrid_v3_DT7_ICB.pt \\
        --paper DT7 \\
        --out results_DT7/

Step 2: Run this script (in cellwhisperer env):
    conda activate cellwhisperer
    python elisa_vs_cellwhisperer_DT7.py \\
        --elisa-results results_DT7/benchmark_v5_results.json \\
        --cw-npz /path/to/cellwhisperer/full_output.npz \\
        --cw-leiden /path/to/leiden_umap_embeddings.h5ad \\
        --cw-ckpt /path/to/cellwhisperer_clip_v1.ckpt \\
        --cf-h5ad /path/to/read_count_table.h5ad \\
        --out comparison_DT7/
"""

import os, sys, json, argparse
from datetime import datetime
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Cluster name constants ──
BCELL    = "B cell"
CD4T     = "CD4-positive, alpha-beta T cell"
CD8REG   = "CD8-positive, CD28-negative, alpha-beta regulatory T cell"
CD8T     = "CD8-positive, alpha-beta T cell"
TCELL    = "T cell"
TFH      = "T follicular helper cell"
TH17     = "T-helper 17 cell"
CD8ACT   = "activated CD8-positive, alpha-beta T cell, human"
CD8CM    = "central memory CD8-positive, alpha-beta T cell"
DC       = "dendritic cell"
CD8EFF   = "effector CD8-positive, alpha-beta T cell"
ENDO     = "endothelial cell"
EPI      = "epithelial cell of thymus"
FIB      = "fibroblast"
HPC      = "hematopoietic multipotent progenitor cell"
LYMPH    = "lymphocyte"
MAC      = "macrophage"
MALIG    = "malignant cell"
MAST     = "mast cell"
NKT      = "mature NK T cell"
MELANO   = "melanocyte"
MICRO    = "microglial cell"
MONO     = "monocyte"
MYELOID  = "myeloid cell"
MYOFIB   = "myofibroblast cell"
NAIVET   = "naive T cell"
CD8NAIVE = "naive thymus-derived CD8-positive, alpha-beta T cell"
PLASMA   = "plasma cell"
PDC      = "plasmacytoid dendritic cell, human"
TREG     = "regulatory T cell"
UNK      = "unknown"


# ============================================================
# QUERIES — 50 ontology + 50 expression
# ============================================================

QUERIES = [
    # ================================================================
    # ONTOLOGY QUERIES (Q01–Q50)
    # ================================================================

    {"id": "Q01", "text": "malignant cancer cell expressing immune checkpoint ligand PD-L1 for immune evasion",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["CD274", "PDCD1LG2", "B2M", "HLA-A", "HLA-B", "CD47"]},
    {"id": "Q02", "text": "tumor cell immune evasion through HLA downregulation and B2M loss",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["B2M", "HLA-A", "HLA-B", "HLA-C", "CD274", "CD47"]},
    {"id": "Q03", "text": "melanoma cancer cell expressing MITF MLANA PMEL lineage markers",
     "category": "ontology", "expected_clusters": [MALIG, MELANO],
     "expected_genes": ["MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10"]},
    {"id": "Q04", "text": "breast cancer epithelial cell markers EPCAM KRT8 KRT18 KRT19 in ICB treated tumors",
     "category": "ontology", "expected_clusters": [MALIG, EPI],
     "expected_genes": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CDH1"]},
    {"id": "Q05", "text": "tumor cell proliferation and cell cycle markers in malignant cells",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1"]},
    {"id": "Q06", "text": "cancer cell VEGFA and TGFB1 immunosuppressive signaling in tumor microenvironment",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["VEGFA", "TGFB1", "CD274", "IDO1", "CD47"]},
    {"id": "Q07", "text": "epithelial mesenchymal transition EMT markers in cancer cells during ICB treatment",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["VIM", "CDH2", "SNAI1", "ZEB1", "CD44", "EPCAM"]},
    {"id": "Q08", "text": "effector CD8 T cell cytotoxic function with granzyme and perforin expression",
     "category": "ontology", "expected_clusters": [CD8EFF, CD8ACT],
     "expected_genes": ["PRF1", "GZMA", "GZMB", "GZMK", "GNLY", "NKG7"]},
    {"id": "Q09", "text": "activated CD8 T cell expressing IFNG and TNF anti-tumor cytokines",
     "category": "ontology", "expected_clusters": [CD8ACT, CD8EFF],
     "expected_genes": ["IFNG", "TNF", "PRF1", "GZMB", "NKG7", "CD69"]},
    {"id": "Q10", "text": "CD8 T cell exhaustion with PD-1 LAG3 TIM3 TIGIT checkpoint receptor co-expression",
     "category": "ontology", "expected_clusters": [CD8T, CD8ACT, CD8EFF],
     "expected_genes": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"]},
    {"id": "Q11", "text": "TOX transcription factor driving T cell exhaustion program in chronic antigen stimulation",
     "category": "ontology", "expected_clusters": [CD8T, CD8ACT],
     "expected_genes": ["TOX", "TOX2", "PDCD1", "LAG3", "HAVCR2", "TIGIT"]},
    {"id": "Q12", "text": "central memory CD8 T cell with TCF7 and IL7R expression for long-lived immunity",
     "category": "ontology", "expected_clusters": [CD8CM],
     "expected_genes": ["TCF7", "IL7R", "CCR7", "SELL", "LEF1", "CD8A"]},
    {"id": "Q13", "text": "naive CD8 T cell expressing CCR7 SELL before antigen encounter",
     "category": "ontology", "expected_clusters": [CD8NAIVE, NAIVET],
     "expected_genes": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "CD8A"]},
    {"id": "Q14", "text": "CD8-positive T cell co-stimulatory receptor 4-1BB ICOS upon activation",
     "category": "ontology", "expected_clusters": [CD8ACT, CD8EFF],
     "expected_genes": ["TNFRSF9", "ICOS", "CD69", "IFNG", "GZMB", "PRF1"]},
    {"id": "Q15", "text": "CD4 positive helper T cell TCR signaling and cytokine production",
     "category": "ontology", "expected_clusters": [CD4T],
     "expected_genes": ["CD4", "CD3D", "CD3E", "IL7R", "CD28", "ICOS"]},
    {"id": "Q16", "text": "regulatory T cell FOXP3 expressing immunosuppressive function in tumor",
     "category": "ontology", "expected_clusters": [TREG],
     "expected_genes": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT"]},
    {"id": "Q17", "text": "T follicular helper cell CXCR5 BCL6 supporting B cell responses in tertiary lymphoid structures",
     "category": "ontology", "expected_clusters": [TFH],
     "expected_genes": ["CXCR5", "BCL6", "ICOS", "PDCD1", "CD4", "CD3D"]},
    {"id": "Q18", "text": "Th17 helper T cell IL17A RORC inflammatory response in tumor microenvironment",
     "category": "ontology", "expected_clusters": [TH17],
     "expected_genes": ["RORC", "IL17A", "IL23R", "CCR6", "CD4", "CD3D"]},
    {"id": "Q19", "text": "CD8-positive CD28-negative regulatory T cell with suppressive function",
     "category": "ontology", "expected_clusters": [CD8REG],
     "expected_genes": ["CD8A", "GZMB", "PRF1", "LAG3", "CTLA4", "PDCD1"]},
    {"id": "Q20", "text": "natural killer T cell NKT innate cytotoxicity with KLRD1 and NKG7 expression",
     "category": "ontology", "expected_clusters": [NKT],
     "expected_genes": ["KLRD1", "KLRK1", "NKG7", "GNLY", "PRF1", "GZMB"]},
    {"id": "Q21", "text": "NK cell mediated tumor killing through NCR1 and KLRB1 receptor activation",
     "category": "ontology", "expected_clusters": [NKT, LYMPH],
     "expected_genes": ["NCAM1", "NCR1", "KLRB1", "KLRC1", "GZMB", "IFNG"]},
    {"id": "Q22", "text": "B cell CD19 MS4A1 CD79A antigen presentation and humoral immunity in tumor",
     "category": "ontology", "expected_clusters": [BCELL],
     "expected_genes": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "HLA-DRB1"]},
    {"id": "Q23", "text": "plasma cell antibody secreting immunoglobulin production SDC1 MZB1",
     "category": "ontology", "expected_clusters": [PLASMA],
     "expected_genes": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "IGKC", "CD79A"]},
    {"id": "Q24", "text": "tertiary lymphoid structure B cell and plasma cell formation in ICB-responsive tumors",
     "category": "ontology", "expected_clusters": [BCELL, PLASMA, TFH],
     "expected_genes": ["MS4A1", "CD79A", "SDC1", "MZB1", "CXCR5", "BCL6"]},
    {"id": "Q25", "text": "tumor associated macrophage M2 polarization CD163 MRC1 immunosuppressive function",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["CD163", "MRC1", "MSR1", "MARCO", "CD68", "APOE"]},
    {"id": "Q26", "text": "macrophage complement expression C1QA C1QB and TREM2 in tumor microenvironment",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["C1QA", "C1QB", "APOE", "TREM2", "CD68", "SPP1"]},
    {"id": "Q27", "text": "classical monocyte CD14 LYZ infiltration into tumor during checkpoint blockade",
     "category": "ontology", "expected_clusters": [MONO],
     "expected_genes": ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A", "CSF1R"]},
    {"id": "Q28", "text": "dendritic cell antigen presentation CD80 CD86 priming T cell responses",
     "category": "ontology", "expected_clusters": [DC],
     "expected_genes": ["CD80", "CD86", "CD83", "CCR7", "HLA-DRA", "HLA-DRB1"]},
    {"id": "Q29", "text": "plasmacytoid dendritic cell IRF7 LILRA4 type I interferon production",
     "category": "ontology", "expected_clusters": [PDC],
     "expected_genes": ["LILRA4", "IRF7", "IRF8", "CLEC4C", "IL3RA", "NRP1"]},
    {"id": "Q30", "text": "myeloid cell general CSF1R ITGAM expressing innate immune population",
     "category": "ontology", "expected_clusters": [MYELOID],
     "expected_genes": ["ITGAM", "CSF1R", "CD68", "LYZ", "S100A8", "S100A9"]},
    {"id": "Q31", "text": "mast cell KIT TPSB2 CPA3 in allergic and inflammatory tumor responses",
     "category": "ontology", "expected_clusters": [MAST],
     "expected_genes": ["KIT", "TPSB2", "TPSAB1", "CPA3", "HPGDS", "HDC"]},
    {"id": "Q32", "text": "microglial cell brain resident macrophage in melanoma brain metastasis",
     "category": "ontology", "expected_clusters": [MICRO],
     "expected_genes": ["P2RY12", "TMEM119", "CX3CR1", "CSF1R", "ITGAM", "AIF1"]},
    {"id": "Q33", "text": "cancer associated fibroblast FAP ACTA2 COL1A1 producing extracellular matrix",
     "category": "ontology", "expected_clusters": [FIB],
     "expected_genes": ["FAP", "ACTA2", "COL1A1", "COL1A2", "PDGFRA", "DCN"]},
    {"id": "Q34", "text": "myofibroblast ACTA2 TAGLN contractile smooth muscle actin expression in tumor stroma",
     "category": "ontology", "expected_clusters": [MYOFIB],
     "expected_genes": ["ACTA2", "TAGLN", "MYH11", "COL1A1", "PDGFRB", "VIM"]},
    {"id": "Q35", "text": "tumor endothelial cell PECAM1 CDH5 VWF vascular marker expression",
     "category": "ontology", "expected_clusters": [ENDO],
     "expected_genes": ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "ENG"]},
    {"id": "Q36", "text": "melanocyte pigmentation pathway MITF TYR TYRP1 DCT lineage genes",
     "category": "ontology", "expected_clusters": [MELANO],
     "expected_genes": ["MITF", "TYR", "TYRP1", "DCT", "MLANA", "PMEL"]},
    {"id": "Q37", "text": "hematopoietic multipotent progenitor cell stem cell marker expression",
     "category": "ontology", "expected_clusters": [HPC],
     "expected_genes": ["CD34", "KIT", "FLT3", "PROM1", "THY1", "PTPRC"]},
    {"id": "Q38", "text": "PD-1 blockade restoring effector CD8 T cell anti-tumor cytotoxicity",
     "category": "ontology", "expected_clusters": [CD8EFF, CD8ACT],
     "expected_genes": ["PDCD1", "GZMB", "PRF1", "IFNG", "TNF", "NKG7"]},
    {"id": "Q39", "text": "CTLA-4 blockade enhancing CD4 helper T cell and reducing Treg suppression",
     "category": "ontology", "expected_clusters": [CD4T, TREG],
     "expected_genes": ["CTLA4", "CD4", "FOXP3", "IL2RA", "CD28", "ICOS"]},
    {"id": "Q40", "text": "T cell clonal replacement and expansion following PD-1 checkpoint inhibition",
     "category": "ontology", "expected_clusters": [CD8EFF, CD8ACT, CD8T],
     "expected_genes": ["GZMB", "PRF1", "IFNG", "MKI67", "CD8A", "PDCD1"]},
    {"id": "Q41", "text": "TCF4 dependent resistance program in mesenchymal-like melanoma cells",
     "category": "ontology", "expected_clusters": [MALIG, MELANO],
     "expected_genes": ["MITF", "SOX10", "VIM", "ZEB1", "CD274", "TGFB1"]},
    {"id": "Q42", "text": "T cell exclusion program in tumor cells resisting checkpoint blockade therapy",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["CD274", "TGFB1", "VEGFA", "IDO1", "CD47", "B2M"]},
    {"id": "Q43", "text": "antigen processing and MHC class I presentation in tumor cells",
     "category": "ontology", "expected_clusters": [MALIG, DC],
     "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"]},
    {"id": "Q44", "text": "MHC class II antigen presentation by professional antigen presenting cells",
     "category": "ontology", "expected_clusters": [DC, MAC, BCELL],
     "expected_genes": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"]},
    {"id": "Q45", "text": "interferon gamma response driving PD-L1 upregulation on tumor cells",
     "category": "ontology", "expected_clusters": [MALIG, CD8EFF],
     "expected_genes": ["IFNG", "CD274", "STAT1", "IRF1", "B2M", "HLA-A"]},
    {"id": "Q46", "text": "tumor infiltrating lymphocyte diversity including T B and NK cells",
     "category": "ontology", "expected_clusters": [CD8EFF, CD4T, BCELL, NKT],
     "expected_genes": ["CD8A", "CD4", "MS4A1", "NCAM1", "CD3D", "KLRD1"]},
    {"id": "Q47", "text": "liver cancer hepatocellular carcinoma markers ALB AFP GPC3 in ICB dataset",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["ALB", "AFP", "GPC3", "EPCAM", "KRT19", "KRT8"]},
    {"id": "Q48", "text": "clear cell renal carcinoma CA9 PAX8 markers in kidney cancer patients",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["CA9", "PAX8", "MME", "EPCAM", "CD274", "VEGFA"]},
    {"id": "Q49", "text": "basal cell carcinoma Hedgehog pathway PTCH1 GLI1 GLI2 SHH signaling",
     "category": "ontology", "expected_clusters": [MALIG],
     "expected_genes": ["PTCH1", "GLI1", "GLI2", "EPCAM", "KRT14", "CD274"]},
    {"id": "Q50", "text": "lymphocyte general population in tumor immune microenvironment",
     "category": "ontology", "expected_clusters": [LYMPH, TCELL],
     "expected_genes": ["CD3D", "CD3E", "PTPRC", "IL7R", "CD2", "LCK"]},

    # ================================================================
    # EXPRESSION QUERIES (Q51–Q100)
    # ================================================================

    {"id": "Q51", "text": "CD274 PDCD1LG2 B2M HLA-A CD47 IDO1 VEGFA",
     "category": "expression", "expected_clusters": [MALIG],
     "expected_genes": ["CD274", "PDCD1LG2", "B2M", "HLA-A", "CD47", "IDO1", "VEGFA"]},
    {"id": "Q52", "text": "MITF MLANA PMEL TYR DCT SOX10 TYRP1",
     "category": "expression", "expected_clusters": [MALIG, MELANO],
     "expected_genes": ["MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10", "TYRP1"]},
    {"id": "Q53", "text": "EPCAM KRT8 KRT18 KRT19 MUC1 CDH1 ESR1",
     "category": "expression", "expected_clusters": [MALIG, EPI],
     "expected_genes": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "CDH1", "ESR1"]},
    {"id": "Q54", "text": "MKI67 TOP2A PCNA CD274 B2M TGFB1",
     "category": "expression", "expected_clusters": [MALIG],
     "expected_genes": ["MKI67", "TOP2A", "PCNA", "CD274", "B2M", "TGFB1"]},
    {"id": "Q55", "text": "PRF1 GZMA GZMB GZMK GNLY NKG7 IFNG",
     "category": "expression", "expected_clusters": [CD8EFF, CD8ACT],
     "expected_genes": ["PRF1", "GZMA", "GZMB", "GZMK", "GNLY", "NKG7", "IFNG"]},
    {"id": "Q56", "text": "GZMB PRF1 IFNG TNF FASLG NKG7 CD8A",
     "category": "expression", "expected_clusters": [CD8EFF, CD8ACT],
     "expected_genes": ["GZMB", "PRF1", "IFNG", "TNF", "FASLG", "NKG7", "CD8A"]},
    {"id": "Q57", "text": "CD69 ICOS TNFRSF9 IFNG GZMB CD8A",
     "category": "expression", "expected_clusters": [CD8ACT],
     "expected_genes": ["CD69", "ICOS", "TNFRSF9", "IFNG", "GZMB", "CD8A"]},
    {"id": "Q58", "text": "PDCD1 LAG3 HAVCR2 TIGIT TOX ENTPD1",
     "category": "expression", "expected_clusters": [CD8T, CD8ACT, CD8EFF],
     "expected_genes": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"]},
    {"id": "Q59", "text": "TOX TOX2 PDCD1 HAVCR2 LAG3 TIGIT BTLA",
     "category": "expression", "expected_clusters": [CD8T, CD8ACT],
     "expected_genes": ["TOX", "TOX2", "PDCD1", "HAVCR2", "LAG3", "TIGIT", "BTLA"]},
    {"id": "Q60", "text": "TCF7 LEF1 CCR7 SELL IL7R CD8A CD8B",
     "category": "expression", "expected_clusters": [CD8CM, CD8NAIVE],
     "expected_genes": ["TCF7", "LEF1", "CCR7", "SELL", "IL7R", "CD8A", "CD8B"]},
    {"id": "Q61", "text": "CCR7 SELL TCF7 LEF1 IL7R CD3D",
     "category": "expression", "expected_clusters": [NAIVET, CD8NAIVE, CD8CM],
     "expected_genes": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "CD3D"]},
    {"id": "Q62", "text": "CD4 CD3D CD3E IL7R CD28 ICOS TCF7",
     "category": "expression", "expected_clusters": [CD4T],
     "expected_genes": ["CD4", "CD3D", "CD3E", "IL7R", "CD28", "ICOS", "TCF7"]},
    {"id": "Q63", "text": "FOXP3 IL2RA CTLA4 IKZF2 TNFRSF18 TIGIT",
     "category": "expression", "expected_clusters": [TREG],
     "expected_genes": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT"]},
    {"id": "Q64", "text": "CXCR5 BCL6 ICOS PDCD1 CD4 CD3D",
     "category": "expression", "expected_clusters": [TFH],
     "expected_genes": ["CXCR5", "BCL6", "ICOS", "PDCD1", "CD4", "CD3D"]},
    {"id": "Q65", "text": "RORC IL17A IL23R CCR6 CD4 CD3E",
     "category": "expression", "expected_clusters": [TH17],
     "expected_genes": ["RORC", "IL17A", "IL23R", "CCR6", "CD4", "CD3E"]},
    {"id": "Q66", "text": "CD8A GZMB PRF1 LAG3 CTLA4 PDCD1",
     "category": "expression", "expected_clusters": [CD8REG],
     "expected_genes": ["CD8A", "GZMB", "PRF1", "LAG3", "CTLA4", "PDCD1"]},
    {"id": "Q67", "text": "KLRD1 KLRK1 NKG7 GNLY PRF1 GZMB NCAM1",
     "category": "expression", "expected_clusters": [NKT],
     "expected_genes": ["KLRD1", "KLRK1", "NKG7", "GNLY", "PRF1", "GZMB", "NCAM1"]},
    {"id": "Q68", "text": "NCAM1 NCR1 KLRB1 KLRC1 GZMB IFNG",
     "category": "expression", "expected_clusters": [NKT, LYMPH],
     "expected_genes": ["NCAM1", "NCR1", "KLRB1", "KLRC1", "GZMB", "IFNG"]},
    {"id": "Q69", "text": "CD19 MS4A1 CD79A CD79B HLA-DRA HLA-DRB1",
     "category": "expression", "expected_clusters": [BCELL],
     "expected_genes": ["CD19", "MS4A1", "CD79A", "CD79B", "HLA-DRA", "HLA-DRB1"]},
    {"id": "Q70", "text": "SDC1 MZB1 JCHAIN IGHG1 IGKC CD79A",
     "category": "expression", "expected_clusters": [PLASMA],
     "expected_genes": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "IGKC", "CD79A"]},
    {"id": "Q71", "text": "CD163 MRC1 MSR1 MARCO CD68 APOE TREM2",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["CD163", "MRC1", "MSR1", "MARCO", "CD68", "APOE", "TREM2"]},
    {"id": "Q72", "text": "C1QA C1QB APOE TREM2 CD68 SPP1",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["C1QA", "C1QB", "APOE", "TREM2", "CD68", "SPP1"]},
    {"id": "Q73", "text": "CD14 FCGR3A S100A8 S100A9 LYZ CSF1R",
     "category": "expression", "expected_clusters": [MONO],
     "expected_genes": ["CD14", "FCGR3A", "S100A8", "S100A9", "LYZ", "CSF1R"]},
    {"id": "Q74", "text": "CD80 CD86 CD83 CCR7 HLA-DRA CLEC9A",
     "category": "expression", "expected_clusters": [DC],
     "expected_genes": ["CD80", "CD86", "CD83", "CCR7", "HLA-DRA", "CLEC9A"]},
    {"id": "Q75", "text": "LILRA4 IRF7 IRF8 IL3RA NRP1",
     "category": "expression", "expected_clusters": [PDC],
     "expected_genes": ["LILRA4", "IRF7", "IRF8", "IL3RA", "NRP1"]},
    {"id": "Q76", "text": "ITGAM CSF1R CD68 LYZ S100A8 S100A9",
     "category": "expression", "expected_clusters": [MYELOID, MONO],
     "expected_genes": ["ITGAM", "CSF1R", "CD68", "LYZ", "S100A8", "S100A9"]},
    {"id": "Q77", "text": "KIT TPSB2 TPSAB1 CPA3 HPGDS HDC",
     "category": "expression", "expected_clusters": [MAST],
     "expected_genes": ["KIT", "TPSB2", "TPSAB1", "CPA3", "HPGDS", "HDC"]},
    {"id": "Q78", "text": "P2RY12 TMEM119 CX3CR1 CSF1R AIF1",
     "category": "expression", "expected_clusters": [MICRO],
     "expected_genes": ["P2RY12", "TMEM119", "CX3CR1", "CSF1R", "AIF1"]},
    {"id": "Q79", "text": "FAP ACTA2 COL1A1 COL1A2 PDGFRA DCN LUM",
     "category": "expression", "expected_clusters": [FIB],
     "expected_genes": ["FAP", "ACTA2", "COL1A1", "COL1A2", "PDGFRA", "DCN", "LUM"]},
    {"id": "Q80", "text": "ACTA2 TAGLN MYH11 COL1A1 PDGFRB VIM",
     "category": "expression", "expected_clusters": [MYOFIB],
     "expected_genes": ["ACTA2", "TAGLN", "MYH11", "COL1A1", "PDGFRB", "VIM"]},
    {"id": "Q81", "text": "PECAM1 CDH5 VWF KDR FLT1 ENG",
     "category": "expression", "expected_clusters": [ENDO],
     "expected_genes": ["PECAM1", "CDH5", "VWF", "KDR", "FLT1", "ENG"]},
    {"id": "Q82", "text": "MITF TYR TYRP1 DCT MLANA PMEL SOX10",
     "category": "expression", "expected_clusters": [MELANO],
     "expected_genes": ["MITF", "TYR", "TYRP1", "DCT", "MLANA", "PMEL", "SOX10"]},
    {"id": "Q83", "text": "CD34 KIT FLT3 PROM1 THY1 PTPRC",
     "category": "expression", "expected_clusters": [HPC],
     "expected_genes": ["CD34", "KIT", "FLT3", "PROM1", "THY1", "PTPRC"]},
    {"id": "Q84", "text": "CD3D CD3E CD8A CD4 TRAC TRBC1",
     "category": "expression", "expected_clusters": [TCELL, CD4T, CD8T],
     "expected_genes": ["CD3D", "CD3E", "CD8A", "CD4", "TRAC", "TRBC1"]},
    {"id": "Q85", "text": "HLA-DRA HLA-DRB1 HLA-DPA1 HLA-DPB1 CD74 CIITA",
     "category": "expression", "expected_clusters": [DC, MAC, BCELL],
     "expected_genes": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"]},
    {"id": "Q86", "text": "HLA-A HLA-B HLA-C B2M TAP1 TAP2",
     "category": "expression", "expected_clusters": [MALIG, DC],
     "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"]},
    {"id": "Q87", "text": "PDCD1 CD274 CTLA4 CD80 CD86 LAG3 HAVCR2",
     "category": "expression", "expected_clusters": [CD8EFF, CD8ACT, TREG, DC, MALIG],
     "expected_genes": ["PDCD1", "CD274", "CTLA4", "CD80", "CD86", "LAG3", "HAVCR2"]},
    {"id": "Q88", "text": "CD274 CD47 IDO1 GZMB PRF1 IFNG",
     "category": "expression", "expected_clusters": [MALIG, CD8EFF],
     "expected_genes": ["CD274", "CD47", "IDO1", "GZMB", "PRF1", "IFNG"]},
    {"id": "Q89", "text": "CD8A CD4 MS4A1 CD68 PECAM1 FAP EPCAM NCAM1",
     "category": "expression", "expected_clusters": [CD8T, CD4T, BCELL, MAC, ENDO, FIB, MALIG, NKT],
     "expected_genes": ["CD8A", "CD4", "MS4A1", "CD68", "PECAM1", "FAP", "EPCAM", "NCAM1"]},
    {"id": "Q90", "text": "GZMB IFNG FOXP3 CD163 CD274 MS4A1 PECAM1",
     "category": "expression", "expected_clusters": [CD8EFF, TREG, MAC, MALIG, BCELL, ENDO],
     "expected_genes": ["GZMB", "IFNG", "FOXP3", "CD163", "CD274", "MS4A1", "PECAM1"]},
    {"id": "Q91", "text": "ALB AFP GPC3 EPCAM KRT19",
     "category": "expression", "expected_clusters": [MALIG],
     "expected_genes": ["ALB", "AFP", "GPC3", "EPCAM", "KRT19"]},
    {"id": "Q92", "text": "CA9 PAX8 MME EPCAM VEGFA",
     "category": "expression", "expected_clusters": [MALIG],
     "expected_genes": ["CA9", "PAX8", "MME", "EPCAM", "VEGFA"]},
    {"id": "Q93", "text": "PTCH1 GLI1 GLI2 EPCAM KRT14",
     "category": "expression", "expected_clusters": [MALIG],
     "expected_genes": ["PTCH1", "GLI1", "GLI2", "EPCAM", "KRT14"]},
    {"id": "Q94", "text": "ERBB2 ESR1 EPCAM KRT8 KRT18 MUC1",
     "category": "expression", "expected_clusters": [MALIG],
     "expected_genes": ["ERBB2", "ESR1", "EPCAM", "KRT8", "KRT18", "MUC1"]},
    {"id": "Q95", "text": "CCR7 SELL TCF7 PDCD1 TOX GZMB PRF1",
     "category": "expression", "expected_clusters": [NAIVET, CD8CM, CD8ACT, CD8EFF],
     "expected_genes": ["CCR7", "SELL", "TCF7", "PDCD1", "TOX", "GZMB", "PRF1"]},
    {"id": "Q96", "text": "IFNG CD274 STAT1 IRF1 B2M HLA-A",
     "category": "expression", "expected_clusters": [CD8EFF, MALIG],
     "expected_genes": ["IFNG", "CD274", "STAT1", "IRF1", "B2M", "HLA-A"]},
    {"id": "Q97", "text": "CD8A CD4 FOXP3 CXCR5 RORC CCR7 KLRD1 CD3D",
     "category": "expression", "expected_clusters": [CD8T, CD4T, TREG, TFH, TH17, NAIVET, NKT, TCELL],
     "expected_genes": ["CD8A", "CD4", "FOXP3", "CXCR5", "RORC", "CCR7", "KLRD1", "CD3D"]},
    {"id": "Q98", "text": "CD68 CD163 CD14 S100A8 CD80 KIT LILRA4 ITGAM",
     "category": "expression", "expected_clusters": [MAC, MONO, DC, MAST, PDC, MYELOID],
     "expected_genes": ["CD68", "CD163", "CD14", "S100A8", "CD80", "KIT", "LILRA4", "ITGAM"]},
    {"id": "Q99", "text": "FAP ACTA2 PECAM1 CDH5 COL1A1 PDGFRA VWF",
     "category": "expression", "expected_clusters": [FIB, MYOFIB, ENDO],
     "expected_genes": ["FAP", "ACTA2", "PECAM1", "CDH5", "COL1A1", "PDGFRA", "VWF"]},
    {"id": "Q100", "text": "CD274 GZMB CD68 MS4A1 FAP PECAM1 MITF FOXP3 CD8A KIT LILRA4",
     "category": "expression", "expected_clusters": [MALIG, CD8EFF, MAC, BCELL, FIB, ENDO, MELANO, TREG, MAST, PDC],
     "expected_genes": ["CD274", "GZMB", "CD68", "MS4A1", "FAP", "PECAM1", "MITF", "FOXP3", "CD8A", "KIT", "LILRA4"]},
]


# ============================================================
# METRICS
# ============================================================

def _word_overlap(a, b):
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb: return 0.0
    return len(wa & wb) / len(wa | wb)

def cluster_recall_at_k(expected, retrieved, k=5):
    if not expected: return 0.0
    ret_lower = [r.lower() for r in retrieved[:k]]
    found = 0
    for exp in expected:
        exp_l = exp.lower()
        if any(exp_l in r or r in exp_l or _word_overlap(exp_l, r) >= 0.5 for r in ret_lower):
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
# CELLWHISPERER SCORER
# ============================================================

class CellWhispererScorer:
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

        ct_embed_accum, ct_counts = {}, {}
        for i, ct in enumerate(self.cell_types):
            if ct not in ct_embed_accum:
                ct_embed_accum[ct] = np.zeros(self.cell_embeds.shape[1], dtype=np.float64); ct_counts[ct] = 0
            ct_embed_accum[ct] += self.cell_embeds[i].astype(np.float64); ct_counts[ct] += 1

        self.celltype_embeds = {}
        for ct in ct_embed_accum:
            self.celltype_embeds[ct] = (ct_embed_accum[ct] / ct_counts[ct]).astype(np.float32)
        print(f"[CW] {len(self.celltype_embeds)} unique cell types:")
        for ct in sorted(self.celltype_embeds.keys()): print(f"     {ct} ({ct_counts[ct]} cells)")

        self.model = None; self._load_model(ckpt_path)

    def _load_model(self, ckpt_path):
        try:
            import torch
            cw_src = os.path.join(os.path.dirname(ckpt_path), "..", "..", "src")
            if os.path.isdir(cw_src): sys.path.insert(0, cw_src)
            from cellwhisperer.jointemb.cellwhisperer_lightning import TranscriptomeTextDualEncoderLightning
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(ckpt_path)
            pl_model.eval().to(self.device); pl_model.freeze()
            self.model = pl_model.model
            print(f"[CW] CLIP model loaded on {self.device}")
        except Exception as e:
            print(f"[CW] WARNING: Could not load CLIP model: {e}")
            print(f"[CW] Will use random ranking as fallback!"); self.model = None

    def embed_text(self, query_text):
        import torch
        if self.model is not None:
            with torch.no_grad(): return self.model.embed_texts([query_text]).cpu().numpy()[0]
        return None

    def score_query(self, query_text, top_k=10):
        text_embed = self.embed_text(query_text)
        if text_embed is None:
            ct_list = list(self.celltype_embeds.keys()); np.random.shuffle(ct_list); return ct_list[:top_k]
        t_norm = text_embed / (np.linalg.norm(text_embed) + 1e-8)
        scores = {}
        for ct, ct_embed in self.celltype_embeds.items():
            c_norm = ct_embed / (np.linalg.norm(ct_embed) + 1e-8)
            scores[ct] = float(np.dot(t_norm, c_norm))
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        self._last_scores = scores
        return ranked[:top_k]


# ============================================================
# MAIN COMPARISON — Recall@5,10,15,20 for 31 clusters
# ============================================================

def run_comparison(cw_scorer, elisa_json_path, paper_id, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[ELISA] Loading results from {elisa_json_path}")
    with open(elisa_json_path) as f: elisa_all = json.load(f)
    paper = elisa_all[paper_id]
    elisa_detail = paper["retrieval_detail"]
    elisa_summary = paper["retrieval_summary"]
    elisa_analytical = paper.get("analytical", {})

    # DT7 uses recall@5,10,15,20 (31 clusters)
    RECALL_KS = [5, 10, 15, 20]

    print("\n" + "=" * 70)
    print("[CW] Running CellWhisperer on 100 queries (DT7)...")
    print("=" * 70)

    cw_results = {"ontology": [], "expression": []}
    for q in QUERIES:
        ranked = cw_scorer.score_query(q["text"], top_k=31)
        entry = {"query_id": q["id"], "query_text": q["text"], "expected": q["expected_clusters"],
                 "retrieved_top20": ranked[:20], "mrr": round(mrr(q["expected_clusters"], ranked), 4)}
        for k in RECALL_KS:
            entry[f"recall@{k}"] = round(cluster_recall_at_k(q["expected_clusters"], ranked, k), 4)
        cw_results[q["category"]].append(entry)
        print(f"  [{q['id']}] R@5={entry['recall@5']:.2f}  MRR={entry['mrr']:.2f}  Top3={ranked[:3]}")

    cw_agg = {}
    for cat in ["ontology", "expression"]:
        e = cw_results[cat]
        cw_agg[cat] = {"mean_mrr": round(np.mean([x["mrr"] for x in e]), 4), "mean_gene_recall": 0.0}
        for k in RECALL_KS:
            cw_agg[cat][f"mean_recall@{k}"] = round(np.mean([x[f"recall@{k}"] for x in e]), 4)

    ALL_MODES = ["random", "cellwhisperer_real", "semantic", "scgpt", "union"]
    MODE_COLORS = {"random": "#9E9E9E", "cellwhisperer_real": "#E91E63", "semantic": "#2196F3", "scgpt": "#FF9800", "union": "#4CAF50"}
    MODE_LABELS = {"random": "Random", "cellwhisperer_real": "CellWhisp.", "semantic": "Semantic", "scgpt": "scGPT", "union": "Union(S+G)"}

    def gm(cat, mode, mk):
        if mode == "cellwhisperer_real": return cw_agg[cat].get(mk, 0)
        return elisa_summary.get(f"{cat}_{mode}", {}).get(mk, 0)

    # ── Console output ──
    print("\n" + "=" * 90)
    print("ELISA vs CellWhisperer — DT7: ICB Multi-Cancer (100 queries)")
    print("=" * 90)
    print(f"\n{'Category':<14} {'Mode':<16} {'R@5':>7} {'R@10':>7} {'R@15':>7} {'R@20':>7} {'MRR':>7} {'GeneR':>7}")
    print("-" * 85)
    for cat in ["ontology", "expression"]:
        for mode in ALL_MODES:
            r5, r10, r15, r20 = gm(cat, mode, "mean_recall@5"), gm(cat, mode, "mean_recall@10"), gm(cat, mode, "mean_recall@15"), gm(cat, mode, "mean_recall@20")
            mmr, gr = gm(cat, mode, "mean_mrr"), gm(cat, mode, "mean_gene_recall")
            gr_s = "  N/A" if mode == "cellwhisperer_real" else f"{gr:.3f}"
            print(f"{cat:<14} {MODE_LABELS.get(mode, mode):<16} {r5:>7.3f} {r10:>7.3f} {r15:>7.3f} {r20:>7.3f} {mmr:>7.3f} {gr_s:>7}")
        print()

    print("── Overall ──")
    for mode in ALL_MODES:
        r5 = np.mean([gm(c, mode, "mean_recall@5") for c in ["ontology", "expression"]])
        print(f"  {MODE_LABELS.get(mode, mode):<16} Recall@5={r5:.3f}")

    ana = elisa_analytical
    print("\n── Analytical Modules (ELISA only) ──")
    print(f"  Pathways:     {ana.get('pathways', {}).get('alignment', 0):.1f}%")
    print(f"  Interactions: {ana.get('interactions', {}).get('lr_recovery_rate', 0):.1f}% LR")
    print(f"  Proportions:  {ana.get('proportions', {}).get('consistency_rate', 0):.1f}%")
    print(f"  Compare:      {ana.get('compare', {}).get('compare_recall', 0):.1f}%")
    print("=" * 90)

    # ── FIGURES ──
    cats = ["ontology", "expression"]
    cat_titles = ["Ontology Queries\n(concept-level)", "Expression Queries\n(gene-signature)"]

    # Fig 1: Recall@5
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES)); vals = [gm(cat, m, "mean_recall@5") for m in ALL_MODES]
        bars = ax.bar(x, vals, color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15); ax.set_ylabel("Mean Cluster Recall@5"); ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v > 0: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("ELISA vs CellWhisperer — DT7: ICB Multi-Cancer (100 queries)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig1_recall5.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig1_recall5.pdf"), bbox_inches="tight"); plt.close(); print(f"\n[FIG] fig1_recall5.png")

    # Fig 2: All metrics R@5, R@10, MRR
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, mk, mt in zip(axes, ["mean_recall@5", "mean_recall@10", "mean_mrr"], ["Recall@5", "Recall@10", "MRR"]):
        x = np.arange(len(ALL_MODES)); w = 0.35
        ont = [gm("ontology", m, mk) for m in ALL_MODES]; exp = [gm("expression", m, mk) for m in ALL_MODES]
        ax.bar(x - w / 2, ont, w, label="Ontology", alpha=0.85, color=[MODE_COLORS[m] for m in ALL_MODES], edgecolor="white")
        ax.bar(x + w / 2, exp, w, label="Expression", alpha=0.45, color=[MODE_COLORS[m] for m in ALL_MODES], edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15); ax.set_title(mt, fontsize=13, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig2_all_metrics.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig2_all_metrics.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig2_all_metrics.png")

    # Fig 3: Cluster Recall@5 vs Gene Recall
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES)); w = 0.35
        cv = [gm(cat, m, "mean_recall@5") for m in ALL_MODES]; gv = [gm(cat, m, "mean_gene_recall") for m in ALL_MODES]
        bars1 = ax.bar(x - w / 2, cv, w, label="Cluster Recall@5", color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + w / 2, gv, w, label="Gene Recall", color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.45, edgecolor="black", linewidth=0.8, hatch="///")
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9); ax.set_ylim(0, 1.15); ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars1, cv):
            if v > 0.05: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars2, gv):
            if v > 0.05: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    legend_elements = [Patch(facecolor="#888888", alpha=0.85, label="Cluster Recall@5"), Patch(facecolor="#888888", alpha=0.45, hatch="///", edgecolor="black", label="Gene Recall")]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=9)
    cw_idx = ALL_MODES.index("cellwhisperer_real")
    for ax in axes: ax.text(cw_idx + 0.175, 0.05, "N/A", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#E91E63")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig3_cluster_vs_gene.png")

    # Fig 4: Radar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    rl = ["Ont R@5", "Ont R@10", "Ont MRR", "Exp R@5", "Exp R@10", "Exp MRR"]; N = len(rl)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]; angles += angles[:1]
    for mode, color, ls in [("cellwhisperer_real", "#E91E63", "--"), ("semantic", "#2196F3", "-"), ("scgpt", "#FF9800", "-"), ("union", "#4CAF50", "-")]:
        v = []
        for cat in ["ontology", "expression"]:
            for mk in ["mean_recall@5", "mean_recall@10", "mean_mrr"]: v.append(gm(cat, mode, mk))
        v += v[:1]
        ax.plot(angles, v, linewidth=2, linestyle=ls, label=MODE_LABELS.get(mode, mode), color=color)
        ax.fill(angles, v, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rl, fontsize=10); ax.set_ylim(0, 1)
    ax.set_title("ELISA vs CellWhisperer\n(DT7: ICB Multi-Cancer)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig4_radar.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig4_radar.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig4_radar.png")

    # Fig 5: Gene recall + Analytical
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(ALL_MODES)); w = 0.35
    og = [gm("ontology", m, "mean_gene_recall") for m in ALL_MODES]; eg = [gm("expression", m, "mean_gene_recall") for m in ALL_MODES]
    ax1.bar(x - w / 2, og, w, label="Ontology", alpha=0.85, color=[MODE_COLORS[m] for m in ALL_MODES], edgecolor="white")
    ax1.bar(x + w / 2, eg, w, label="Expression", alpha=0.45, color=[MODE_COLORS[m] for m in ALL_MODES], edgecolor="black", linewidth=0.5, hatch="//")
    ax1.set_xticks(x); ax1.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("Gene Recall"); ax1.set_title("Gene-Level Evidence Delivery", fontsize=12, fontweight="bold"); ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    ax1.text(cw_idx, 0.05, "N/A", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#E91E63")

    ax2.set_axis_off()
    ax2b = fig.add_axes([0.55, 0.1, 0.4, 0.75], polar=True)
    al = ["Pathways", "Interactions\n(LR)", "Proportions", "Compare"]
    av = [ana.get("pathways", {}).get("alignment", 0) / 100, ana.get("interactions", {}).get("lr_recovery_rate", 0) / 100,
          ana.get("proportions", {}).get("consistency_rate", 0) / 100, ana.get("compare", {}).get("compare_recall", 0) / 100]
    aa = np.linspace(0, 2 * np.pi, len(al), endpoint=False).tolist(); av_c = av + av[:1]; aa_c = aa + aa[:1]
    ax2b.fill(aa_c, av_c, alpha=0.25, color="#4CAF50"); ax2b.plot(aa_c, av_c, "o-", color="#4CAF50", linewidth=2, label="ELISA")
    ax2b.plot(aa_c, [0] * len(aa_c), "--", color="#E91E63", linewidth=1.5, label="CellWhisp. (N/A)")
    ax2b.set_xticks(aa); ax2b.set_xticklabels(al, fontsize=9); ax2b.set_ylim(0, 1.05)
    ax2b.set_title("Analytical Modules\n(ELISA only)", fontsize=11, fontweight="bold", pad=15)
    ax2b.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig5_gene_analytical.png")

    # Fig 6: Per-query (first 25)
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for ax, cat in zip(axes, ["ontology", "expression"]):
        cw_e = cw_results[cat]; qids = [e["query_id"] for e in cw_e]; cw_r5 = [e["recall@5"] for e in cw_e]
        el_union = elisa_detail.get(cat, {}).get("union", [])
        el_r5 = [e.get("recall@5", 0) for e in el_union] if el_union else [0] * len(qids)
        n = min(25, len(qids), len(el_r5)); y = np.arange(n); h = 0.35
        ax.barh(y - h / 2, el_r5[:n], h, label="ELISA Union", color="#4CAF50", alpha=0.8)
        ax.barh(y + h / 2, cw_r5[:n], h, label="CellWhisperer", color="#E91E63", alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(qids[:n], fontsize=8); ax.set_xlabel("Recall@5"); ax.set_xlim(0, 1.1)
        ax.set_title(f"{cat.capitalize()} Queries (first 25)", fontsize=13, fontweight="bold"); ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    plt.suptitle("Per-Query: ELISA Union vs CellWhisperer (DT7)", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig6_perquery.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig6_perquery.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig6_perquery.png")

    # ── Save JSON ──
    output = {
        "cellwhisperer": {"results": cw_results, "aggregated": cw_agg},
        "elisa_summary": elisa_summary, "elisa_analytical": elisa_analytical,
        "comparison": {cat: {mode: {mk: gm(cat, mode, mk) for mk in [f"mean_recall@{k}" for k in RECALL_KS] + ["mean_mrr"]} for mode in ALL_MODES} for cat in cats},
        "modes": ALL_MODES, "n_queries": len(QUERIES), "dataset": "DT7_ICB_MultiCancer",
        "paper": "Gondal et al. Sci Data 2025", "n_clusters": 31,
        "cluster_names": [BCELL, CD4T, CD8REG, CD8T, TCELL, TFH, TH17, CD8ACT, CD8CM, DC, CD8EFF, ENDO, EPI, FIB, HPC, LYMPH, MAC, MALIG, MAST, NKT, MELANO, MICRO, MONO, MYELOID, MYOFIB, NAIVET, CD8NAIVE, PLASMA, PDC, TREG, UNK],
        "timestamp": datetime.now().isoformat(),
    }
    rp = os.path.join(out_dir, "elisa_vs_cellwhisperer_DT7_results.json")
    with open(rp, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")


def main():
    parser = argparse.ArgumentParser(description="ELISA vs CellWhisperer — DT7 ICB Multi-Cancer")
    parser.add_argument("--elisa-results", required=True, help="ELISA benchmark_v5_results.json from DT7")
    parser.add_argument("--cw-npz", required=True, help="CellWhisperer full_output.npz")
    parser.add_argument("--cw-leiden", required=True, help="CellWhisperer leiden_umap_embeddings.h5ad")
    parser.add_argument("--cw-ckpt", required=True, help="cellwhisperer_clip_v1.ckpt")
    parser.add_argument("--cf-h5ad", required=True, help="Original read_count_table.h5ad")
    parser.add_argument("--paper", default="DT7")
    parser.add_argument("--out", default="comparison_DT7/")
    args = parser.parse_args()
    cw = CellWhispererScorer(args.cw_npz, args.cw_leiden, args.cw_ckpt, args.cf_h5ad)
    run_comparison(cw, args.elisa_results, args.paper, args.out)
    print("\n[DONE]")

if __name__ == "__main__":
    main()
