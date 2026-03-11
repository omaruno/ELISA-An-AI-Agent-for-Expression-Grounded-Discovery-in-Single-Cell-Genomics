#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA vs CellWhisperer — DT6 High-Risk Neuroblastoma (Yu et al. Nat Genet 2025)
================================================================================
Head-to-Head comparison on longitudinal neuroblastoma snRNA-seq data.

11 clusters:
  0:  B cell
  1:  Schwann cell
  2:  T cell
  3:  cortical cell of adrenal gland
  4:  dendritic cell
  5:  endothelial cell
  6:  fibroblast
  7:  hepatocyte
  8:  kidney cell
  9:  macrophage
  10: neuroblast (sensu Vertebrata)

Step 1: Run ELISA benchmark first:
    python elisa_benchmark_v5_1_DT6.py \\
        --base /path/to/embeddings \\
        --pt-name hybrid_v3_DT6_NB.pt \\
        --paper DT6 \\
        --out results_DT6/

Step 2: Run this script (in cellwhisperer env):
    conda activate cellwhisperer
    python elisa_vs_cellwhisperer_DT6.py \\
        --elisa-results results_DT6/benchmark_v5_results.json \\
        --cw-npz /path/to/cellwhisperer/full_output.npz \\
        --cw-leiden /path/to/leiden_umap_embeddings.h5ad \\
        --cw-ckpt /path/to/cellwhisperer_clip_v1.ckpt \\
        --cf-h5ad /path/to/read_count_table.h5ad \\
        --out comparison_DT6/
"""

import os, sys, json, argparse
from datetime import datetime
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Cluster name constants ──
BCELL   = "B cell"
SCHWANN = "Schwann cell"
TCELL   = "T cell"
ADRENAL = "cortical cell of adrenal gland"
DC      = "dendritic cell"
ENDO    = "endothelial cell"
FIB     = "fibroblast"
HEPATO  = "hepatocyte"
KIDNEY  = "kidney cell"
MAC     = "macrophage"
NB      = "neuroblast (sensu Vertebrata)"


# ============================================================
# QUERIES — 50 ontology + 50 expression (same as benchmark)
# ============================================================

QUERIES = [
    # ================================================================
    # ONTOLOGY QUERIES (Q01–Q50)
    # ================================================================

    {"id": "Q01", "text": "neuroblast neoplastic cell of sympathetic nervous system expressing PHOX2B and ISL1",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["PHOX2B", "ISL1", "HAND2", "TH", "DBH", "CHGA"]},
    {"id": "Q02", "text": "neuroblastoma tumor cell with MYCN amplification and proliferative phenotype",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["MYCN", "MKI67", "TOP2A", "EZH2", "PCNA", "BIRC5"]},
    {"id": "Q03", "text": "adrenergic neuroblast expressing catecholamine biosynthesis enzymes tyrosine hydroxylase",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["TH", "DBH", "DDC", "PHOX2B", "PHOX2A", "GATA3"]},
    {"id": "Q04", "text": "neuroblastoma cell with calcium and synaptic signaling pathway enrichment",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["CACNA1B", "CACNA2D1", "SYN2", "KCNMA1", "KCNQ3", "CREB5"]},
    {"id": "Q05", "text": "dopaminergic neuroblast expressing dopamine transporter and metabolic genes",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["SLC18A2", "TH", "DDC", "AGTR2", "ATP2A2", "PHOX2B"]},
    {"id": "Q06", "text": "proliferating neuroblastoma cell with cell cycle and DNA replication markers",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5", "BUB1B"]},
    {"id": "Q07", "text": "mesenchymal neuroblastoma cell state expressing extracellular matrix genes and YAP1",
     "category": "ontology", "expected_clusters": [NB, FIB],
     "expected_genes": ["YAP1", "FN1", "VIM", "COL1A1", "SERPINE1", "SPARC"]},
    {"id": "Q08", "text": "intermediate OXPHOS neuroblast with ribosomal gene expression and oxidative phosphorylation",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["RPL3", "RPL4", "RPS6", "RPS3", "MYCN", "PHOX2B"]},
    {"id": "Q09", "text": "EZH2 expressing neuroblastoma cell PRC2 polycomb repressive complex chromatin regulation",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["EZH2", "SMC4", "TOP2A", "MKI67", "PHOX2B", "MYCN"]},
    {"id": "Q10", "text": "neuroblastoma cell ERBB4 receptor expressing epidermal growth factor signaling",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["ERBB4", "EGFR", "HBEGF", "TGFA", "EREG", "MAPK1"]},
    {"id": "Q11", "text": "neuroblast with adrenergic transcription factor PHOX2A PHOX2B GATA3 expression",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["PHOX2A", "PHOX2B", "GATA3", "ASCL1", "ISL1", "HAND2"]},
    {"id": "Q12", "text": "neural crest derived neoplastic cell in pediatric tumor expressing chromogranin",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["CHGA", "CHGB", "PHOX2B", "ISL1", "TH", "DBH"]},
    {"id": "Q13", "text": "neuroblastoma cell immune evasion NECTIN2 and checkpoint ligand expression",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["NECTIN2", "CD274", "B2M", "HLA-A", "HLA-B", "HLA-C"]},
    {"id": "Q14", "text": "mesenchymal transition state in neuroblastoma with AP-1 transcription factors",
     "category": "ontology", "expected_clusters": [NB, FIB],
     "expected_genes": ["JUN", "FOS", "JUNB", "JUND", "FOSL2", "BACH1"]},

    # --- Macrophage ---
    {"id": "Q15", "text": "tumor associated macrophage in neuroblastoma microenvironment CD68 CD163 expressing",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["CD68", "CD163", "CD86", "CSF1R", "MRC1", "SPP1"]},
    {"id": "Q16", "text": "pro-inflammatory macrophage IL18 expressing anti-tumor immune response",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["IL18", "CD68", "CD163", "CD86", "CSF1R", "HLA-DRA"]},
    {"id": "Q17", "text": "pro-angiogenic macrophage VCAN expressing promoting tumor vascularization",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["VCAN", "VEGFA", "CD68", "CD163", "EGFR", "SPP1"]},
    {"id": "Q18", "text": "immunosuppressive macrophage C1QC SPP1 complement expressing in tumor",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["C1QC", "SPP1", "CD68", "CD163", "APOE", "TREM2"]},
    {"id": "Q19", "text": "tissue resident macrophage F13A1 expressing phagocytic function in neuroblastoma",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["F13A1", "CD68", "CD163", "MRC1", "LYVE1", "CSF1R"]},
    {"id": "Q20", "text": "lipid associated macrophage HS3ST2 with metabolic phenotype in tumor",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["HS3ST2", "CYP27A1", "CD68", "CD163", "APOE", "LPL"]},
    {"id": "Q21", "text": "macrophage secreting HB-EGF ligand for ERBB4 receptor activation on neuroblasts",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["HBEGF", "CD68", "CD163", "TGFA", "EREG", "AREG"]},
    {"id": "Q22", "text": "CCL4 expressing pro-angiogenic macrophage chemokine signaling in tumor",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["CCL4", "CD68", "CD163", "VEGFA", "CSF1R", "CCL3"]},
    {"id": "Q23", "text": "proliferating macrophage MKI67 TOP2A expanding after chemotherapy",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["MKI67", "TOP2A", "CD68", "CD163", "CSF1R", "PCNA"]},
    {"id": "Q24", "text": "THY1 positive macrophage undefined myeloid phenotype in neuroblastoma",
     "category": "ontology", "expected_clusters": [MAC],
     "expected_genes": ["THY1", "CD68", "CD163", "MRC1", "CSF1R", "CD86"]},

    # --- T cell ---
    {"id": "Q25", "text": "T cell lymphocyte infiltrating neuroblastoma tumor expressing CD247 CD96",
     "category": "ontology", "expected_clusters": [TCELL],
     "expected_genes": ["CD247", "CD96", "CD3D", "CD3E", "CD8A", "CD4"]},
    {"id": "Q26", "text": "cytotoxic T cell with granzyme perforin mediated tumor cell killing",
     "category": "ontology", "expected_clusters": [TCELL],
     "expected_genes": ["GZMA", "GZMB", "PRF1", "IFNG", "CD8A", "CD3D"]},
    {"id": "Q27", "text": "tumor infiltrating T lymphocyte immune response to neuroblastoma",
     "category": "ontology", "expected_clusters": [TCELL],
     "expected_genes": ["CD3D", "CD3E", "CD247", "CD96", "CD8A", "CD4"]},

    # --- B cell ---
    {"id": "Q28", "text": "B cell lymphocyte PAX5 MS4A1 in neuroblastoma tumor immune microenvironment",
     "category": "ontology", "expected_clusters": [BCELL],
     "expected_genes": ["PAX5", "MS4A1", "CD19", "CD79A", "HLA-DRA", "HLA-DRB1"]},
    {"id": "Q29", "text": "B lymphocyte humoral immunity and antigen presentation in pediatric tumor",
     "category": "ontology", "expected_clusters": [BCELL],
     "expected_genes": ["MS4A1", "CD79A", "CD19", "PAX5", "HLA-DRA", "CD74"]},

    # --- Dendritic cell ---
    {"id": "Q30", "text": "dendritic cell IRF8 FLT3 antigen presentation priming T cell responses in tumor",
     "category": "ontology", "expected_clusters": [DC],
     "expected_genes": ["IRF8", "FLT3", "CLEC9A", "CD1C", "CD80", "HLA-DRA"]},
    {"id": "Q31", "text": "professional antigen presenting dendritic cell MHC class II expression",
     "category": "ontology", "expected_clusters": [DC],
     "expected_genes": ["HLA-DRA", "HLA-DRB1", "IRF8", "FLT3", "CD80", "CD86"]},

    # --- Fibroblast ---
    {"id": "Q32", "text": "fibroblast stromal cell PDGFRB DCN extracellular matrix production in neuroblastoma",
     "category": "ontology", "expected_clusters": [FIB],
     "expected_genes": ["PDGFRB", "DCN", "LUM", "COL1A1", "COL1A2", "VIM"]},
    {"id": "Q33", "text": "cancer associated fibroblast FAP ACTA2 expressing in tumor stroma",
     "category": "ontology", "expected_clusters": [FIB],
     "expected_genes": ["FAP", "ACTA2", "COL1A1", "COL1A2", "PDGFRA", "DCN"]},
    {"id": "Q34", "text": "neural crest derived endoneurial fibroblast in neuroblastoma tissue",
     "category": "ontology", "expected_clusters": [FIB],
     "expected_genes": ["PDGFRB", "DCN", "COL1A1", "VIM", "LUM", "PDGFRA"]},

    # --- Schwann cell ---
    {"id": "Q35", "text": "Schwann cell PLP1 CDH19 myelinating glial cell in neuroblastoma microenvironment",
     "category": "ontology", "expected_clusters": [SCHWANN],
     "expected_genes": ["PLP1", "CDH19", "SOX10", "MPZ", "MBP", "S100B"]},
    {"id": "Q36", "text": "Schwann cell precursor neural crest lineage expanding after therapy",
     "category": "ontology", "expected_clusters": [SCHWANN],
     "expected_genes": ["PLP1", "CDH19", "SOX10", "S100B", "MBP", "MPZ"]},

    # --- Endothelial ---
    {"id": "Q37", "text": "endothelial cell PECAM1 PTPRB vascular marker in neuroblastoma tumor vasculature",
     "category": "ontology", "expected_clusters": [ENDO],
     "expected_genes": ["PECAM1", "PTPRB", "CDH5", "VWF", "KDR", "FLT1"]},
    {"id": "Q38", "text": "tumor endothelium blood vessel lining cell expressing vascular endothelial markers",
     "category": "ontology", "expected_clusters": [ENDO],
     "expected_genes": ["PECAM1", "CDH5", "VWF", "PTPRB", "KDR", "FLT1"]},

    # --- Adrenal cortex ---
    {"id": "Q39", "text": "adrenal cortex cell steroidogenesis CYP11A1 CYP11B1 adjacent normal tissue",
     "category": "ontology", "expected_clusters": [ADRENAL],
     "expected_genes": ["CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1"]},
    {"id": "Q40", "text": "cortical cell of adrenal gland steroid hormone biosynthesis normal adjacent tissue",
     "category": "ontology", "expected_clusters": [ADRENAL],
     "expected_genes": ["CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1"]},

    # --- Hepatocyte ---
    {"id": "Q41", "text": "hepatocyte ALB expressing liver cell from adjacent normal tissue in neuroblastoma biopsy",
     "category": "ontology", "expected_clusters": [HEPATO],
     "expected_genes": ["ALB", "DCDC2", "HNF4A", "APOB", "CYP3A4"]},

    # --- Kidney ---
    {"id": "Q42", "text": "kidney cell renal tissue PKHD1 from adjacent normal tissue in neuroblastoma specimen",
     "category": "ontology", "expected_clusters": [KIDNEY],
     "expected_genes": ["PKHD1", "PAX2", "WT1", "SLC12A1"]},

    # --- Cross-cutting ---
    {"id": "Q43", "text": "chemotherapy induced tumor microenvironment rewiring macrophage expansion after therapy",
     "category": "ontology", "expected_clusters": [MAC, NB],
     "expected_genes": ["CD68", "CD163", "HBEGF", "PHOX2B", "MKI67", "VCAN"]},
    {"id": "Q44", "text": "HB-EGF ERBB4 paracrine signaling axis between macrophage and neuroblast promoting ERK",
     "category": "ontology", "expected_clusters": [MAC, NB],
     "expected_genes": ["HBEGF", "ERBB4", "MAPK1", "MAPK3", "CD68", "PHOX2B"]},
    {"id": "Q45", "text": "tumor immune evasion and antigen presentation in neuroblastoma",
     "category": "ontology", "expected_clusters": [NB, DC],
     "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "CD274", "NECTIN2"]},
    {"id": "Q46", "text": "VEGFA angiogenesis signaling in neuroblastoma tumor microenvironment",
     "category": "ontology", "expected_clusters": [MAC, ENDO],
     "expected_genes": ["VEGFA", "KDR", "FLT1", "GPC1", "NRP1", "PECAM1"]},
    {"id": "Q47", "text": "immune cell infiltration in high-risk neuroblastoma T cell B cell macrophage",
     "category": "ontology", "expected_clusters": [TCELL, BCELL, MAC],
     "expected_genes": ["CD3D", "CD247", "MS4A1", "CD68", "CD163", "CD8A"]},
    {"id": "Q48", "text": "THBS1 CD47 dont eat me signal between macrophage and neuroblastoma cell",
     "category": "ontology", "expected_clusters": [MAC, NB],
     "expected_genes": ["THBS1", "CD47", "ITGB1", "ITGA3", "CD68", "PHOX2B"]},
    {"id": "Q49", "text": "neuroblastoma cell expressing ALK receptor tyrosine kinase oncogenic driver",
     "category": "ontology", "expected_clusters": [NB],
     "expected_genes": ["ALK", "MYCN", "PHOX2B", "RET", "NTRK1", "NTRK2"]},
    {"id": "Q50", "text": "tumor microenvironment cell diversity neuroblasts fibroblasts Schwann endothelial macrophages",
     "category": "ontology", "expected_clusters": [NB, FIB, SCHWANN, ENDO, MAC],
     "expected_genes": ["PHOX2B", "DCN", "PLP1", "PECAM1", "CD68"]},

    # ================================================================
    # EXPRESSION QUERIES (Q51–Q100)
    # ================================================================

    {"id": "Q51", "text": "PHOX2B ISL1 HAND2 TH DBH DDC CHGA",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["PHOX2B", "ISL1", "HAND2", "TH", "DBH", "DDC", "CHGA"]},
    {"id": "Q52", "text": "MYCN MKI67 TOP2A EZH2 SMC4 BIRC5",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["MYCN", "MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5"]},
    {"id": "Q53", "text": "PHOX2A PHOX2B GATA3 ASCL1 ISL1 HAND2",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["PHOX2A", "PHOX2B", "GATA3", "ASCL1", "ISL1", "HAND2"]},
    {"id": "Q54", "text": "CACNA1B SYN2 KCNMA1 KCNQ3 GPC5 CREB5",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["CACNA1B", "SYN2", "KCNMA1", "KCNQ3", "GPC5", "CREB5"]},
    {"id": "Q55", "text": "SLC18A2 TH DDC AGTR2 ATP2A2 PHOX2B",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["SLC18A2", "TH", "DDC", "AGTR2", "ATP2A2", "PHOX2B"]},
    {"id": "Q56", "text": "MKI67 TOP2A EZH2 SMC4 BIRC5 BUB1B ASPM KIF11",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5", "BUB1B", "ASPM", "KIF11"]},
    {"id": "Q57", "text": "YAP1 FN1 VIM COL1A1 SERPINE1 SPARC THBS2",
     "category": "expression", "expected_clusters": [NB, FIB],
     "expected_genes": ["YAP1", "FN1", "VIM", "COL1A1", "SERPINE1", "SPARC", "THBS2"]},
    {"id": "Q58", "text": "ERBB4 EGFR HBEGF TGFA EREG AREG",
     "category": "expression", "expected_clusters": [NB, MAC],
     "expected_genes": ["ERBB4", "EGFR", "HBEGF", "TGFA", "EREG", "AREG"]},
    {"id": "Q59", "text": "NECTIN2 CD274 B2M HLA-A HLA-B PHOX2B",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["NECTIN2", "CD274", "B2M", "HLA-A", "HLA-B", "PHOX2B"]},
    {"id": "Q60", "text": "JUN FOS JUNB JUND FOSL2 BACH1 BACH2",
     "category": "expression", "expected_clusters": [NB, FIB],
     "expected_genes": ["JUN", "FOS", "JUNB", "JUND", "FOSL2", "BACH1", "BACH2"]},
    {"id": "Q61", "text": "CHGA CHGB PHOX2B ISL1 NTRK1 RET",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["CHGA", "CHGB", "PHOX2B", "ISL1", "NTRK1", "RET"]},
    {"id": "Q62", "text": "ETS1 ETV6 ELF1 KLF6 KLF7 RUNX1 ZNF148",
     "category": "expression", "expected_clusters": [NB, FIB],
     "expected_genes": ["ETS1", "ETV6", "ELF1", "KLF6", "KLF7", "RUNX1", "ZNF148"]},
    {"id": "Q63", "text": "ALK MYCN NTRK2 PHOX2B TH",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["ALK", "MYCN", "NTRK2", "PHOX2B", "TH"]},

    # --- Macrophage ---
    {"id": "Q64", "text": "CD68 CD163 CD86 CSF1R MRC1 SPP1",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["CD68", "CD163", "CD86", "CSF1R", "MRC1", "SPP1"]},
    {"id": "Q65", "text": "IL18 CD68 CD163 CD86 HLA-DRA CSF1R",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["IL18", "CD68", "CD163", "CD86", "HLA-DRA", "CSF1R"]},
    {"id": "Q66", "text": "VCAN VEGFA CD68 CD163 SPP1 EGFR",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["VCAN", "VEGFA", "CD68", "CD163", "SPP1", "EGFR"]},
    {"id": "Q67", "text": "C1QC SPP1 CD68 CD163 APOE TREM2",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["C1QC", "SPP1", "CD68", "CD163", "APOE", "TREM2"]},
    {"id": "Q68", "text": "F13A1 CD68 CD163 MRC1 LYVE1 CSF1R",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["F13A1", "CD68", "CD163", "MRC1", "LYVE1", "CSF1R"]},
    {"id": "Q69", "text": "HS3ST2 CYP27A1 CD68 CD163 APOE LPL",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["HS3ST2", "CYP27A1", "CD68", "CD163", "APOE", "LPL"]},
    {"id": "Q70", "text": "HBEGF TGFA EREG AREG CD68 CD163",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["HBEGF", "TGFA", "EREG", "AREG", "CD68", "CD163"]},
    {"id": "Q71", "text": "CCL4 CD68 CD163 VEGFA CSF1R CCL3",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["CCL4", "CD68", "CD163", "VEGFA", "CSF1R", "CCL3"]},
    {"id": "Q72", "text": "THY1 CD68 CD163 MRC1 CSF1R CD86",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["THY1", "CD68", "CD163", "MRC1", "CSF1R", "CD86"]},

    # --- T cell ---
    {"id": "Q73", "text": "CD247 CD96 CD3D CD3E CD8A CD4",
     "category": "expression", "expected_clusters": [TCELL],
     "expected_genes": ["CD247", "CD96", "CD3D", "CD3E", "CD8A", "CD4"]},
    {"id": "Q74", "text": "GZMA GZMB PRF1 IFNG CD8A CD3D",
     "category": "expression", "expected_clusters": [TCELL],
     "expected_genes": ["GZMA", "GZMB", "PRF1", "IFNG", "CD8A", "CD3D"]},

    # --- B cell ---
    {"id": "Q75", "text": "PAX5 MS4A1 CD19 CD79A HLA-DRA HLA-DRB1",
     "category": "expression", "expected_clusters": [BCELL],
     "expected_genes": ["PAX5", "MS4A1", "CD19", "CD79A", "HLA-DRA", "HLA-DRB1"]},

    # --- DC ---
    {"id": "Q76", "text": "IRF8 FLT3 CLEC9A CD1C CD80 HLA-DRA",
     "category": "expression", "expected_clusters": [DC],
     "expected_genes": ["IRF8", "FLT3", "CLEC9A", "CD1C", "CD80", "HLA-DRA"]},

    # --- Fibroblast ---
    {"id": "Q77", "text": "PDGFRB DCN LUM COL1A1 COL1A2 VIM",
     "category": "expression", "expected_clusters": [FIB],
     "expected_genes": ["PDGFRB", "DCN", "LUM", "COL1A1", "COL1A2", "VIM"]},
    {"id": "Q78", "text": "FAP ACTA2 COL1A1 PDGFRA DCN LUM",
     "category": "expression", "expected_clusters": [FIB],
     "expected_genes": ["FAP", "ACTA2", "COL1A1", "PDGFRA", "DCN", "LUM"]},

    # --- Schwann ---
    {"id": "Q79", "text": "PLP1 CDH19 SOX10 MPZ MBP S100B",
     "category": "expression", "expected_clusters": [SCHWANN],
     "expected_genes": ["PLP1", "CDH19", "SOX10", "MPZ", "MBP", "S100B"]},

    # --- Endothelial ---
    {"id": "Q80", "text": "PECAM1 PTPRB CDH5 VWF KDR FLT1",
     "category": "expression", "expected_clusters": [ENDO],
     "expected_genes": ["PECAM1", "PTPRB", "CDH5", "VWF", "KDR", "FLT1"]},

    # --- Adrenal cortex ---
    {"id": "Q81", "text": "CYP11A1 CYP11B1 CYP17A1 STAR NR5A1",
     "category": "expression", "expected_clusters": [ADRENAL],
     "expected_genes": ["CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1"]},

    # --- Hepatocyte ---
    {"id": "Q82", "text": "ALB DCDC2 HNF4A APOB",
     "category": "expression", "expected_clusters": [HEPATO],
     "expected_genes": ["ALB", "DCDC2", "HNF4A", "APOB"]},

    # --- Kidney ---
    {"id": "Q83", "text": "PKHD1 PAX2 WT1 SLC12A1",
     "category": "expression", "expected_clusters": [KIDNEY],
     "expected_genes": ["PKHD1", "PAX2", "WT1", "SLC12A1"]},

    # --- Cross-lineage ---
    {"id": "Q84", "text": "PHOX2B CD68 CD3D MS4A1 PECAM1 DCN PLP1",
     "category": "expression", "expected_clusters": [NB, MAC, TCELL, BCELL, ENDO, FIB, SCHWANN],
     "expected_genes": ["PHOX2B", "CD68", "CD3D", "MS4A1", "PECAM1", "DCN", "PLP1"]},
    {"id": "Q85", "text": "HBEGF ERBB4 CD68 PHOX2B MAPK1",
     "category": "expression", "expected_clusters": [MAC, NB],
     "expected_genes": ["HBEGF", "ERBB4", "CD68", "PHOX2B", "MAPK1"]},
    {"id": "Q86", "text": "VCAN THBS1 CD47 ITGB1 CD68 PHOX2B",
     "category": "expression", "expected_clusters": [MAC, NB],
     "expected_genes": ["VCAN", "THBS1", "CD47", "ITGB1", "CD68", "PHOX2B"]},
    {"id": "Q87", "text": "HLA-A HLA-B HLA-C B2M HLA-DRA HLA-DRB1",
     "category": "expression", "expected_clusters": [NB, DC, MAC, BCELL],
     "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "HLA-DRA", "HLA-DRB1"]},
    {"id": "Q88", "text": "VEGFA KDR FLT1 NRP1 GPC1 PECAM1",
     "category": "expression", "expected_clusters": [MAC, ENDO],
     "expected_genes": ["VEGFA", "KDR", "FLT1", "NRP1", "GPC1", "PECAM1"]},
    {"id": "Q89", "text": "CD68 IL18 VCAN C1QC SPP1 F13A1 HS3ST2 CCL4 THY1",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["CD68", "IL18", "VCAN", "C1QC", "SPP1", "F13A1", "HS3ST2", "CCL4", "THY1"]},
    {"id": "Q90", "text": "PHOX2B MKI67 TOP2A YAP1 CACNA1B SLC18A2",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["PHOX2B", "MKI67", "TOP2A", "YAP1", "CACNA1B", "SLC18A2"]},
    {"id": "Q91", "text": "APOE LDLR VLDLR LPL HS3ST2 CD68",
     "category": "expression", "expected_clusters": [MAC],
     "expected_genes": ["APOE", "LDLR", "VLDLR", "LPL", "HS3ST2", "CD68"]},
    {"id": "Q92", "text": "THBS1 ITGB1 ITGA3 LRP5 CD47 FN1",
     "category": "expression", "expected_clusters": [MAC, NB, FIB],
     "expected_genes": ["THBS1", "ITGB1", "ITGA3", "LRP5", "CD47", "FN1"]},
    {"id": "Q93", "text": "COL1A1 COL1A2 COL4A1 COL4A2 FN1 VIM SPARC",
     "category": "expression", "expected_clusters": [FIB, NB],
     "expected_genes": ["COL1A1", "COL1A2", "COL4A1", "COL4A2", "FN1", "VIM", "SPARC"]},
    {"id": "Q94", "text": "MAPK1 MAPK3 AKT1 ERBB4 EGFR HBEGF",
     "category": "expression", "expected_clusters": [NB, MAC],
     "expected_genes": ["MAPK1", "MAPK3", "AKT1", "ERBB4", "EGFR", "HBEGF"]},
    {"id": "Q95", "text": "CD274 PDCD1 CTLA4 TIGIT LAG3 NECTIN2",
     "category": "expression", "expected_clusters": [NB, TCELL],
     "expected_genes": ["CD274", "PDCD1", "CTLA4", "TIGIT", "LAG3", "NECTIN2"]},
    {"id": "Q96", "text": "PHOX2B CD68 PLP1 PECAM1 DCN IRF8 PAX5 CD247",
     "category": "expression", "expected_clusters": [NB, MAC, SCHWANN, ENDO, FIB, DC, BCELL, TCELL],
     "expected_genes": ["PHOX2B", "CD68", "PLP1", "PECAM1", "DCN", "IRF8", "PAX5", "CD247"]},
    {"id": "Q97", "text": "CYP11A1 ALB PKHD1 PHOX2B CD68",
     "category": "expression", "expected_clusters": [ADRENAL, HEPATO, KIDNEY, NB, MAC],
     "expected_genes": ["CYP11A1", "ALB", "PKHD1", "PHOX2B", "CD68"]},
    {"id": "Q98", "text": "PHOX2B HBEGF ERBB4 VCAN SPP1 CD163 VEGFA",
     "category": "expression", "expected_clusters": [NB, MAC],
     "expected_genes": ["PHOX2B", "HBEGF", "ERBB4", "VCAN", "SPP1", "CD163", "VEGFA"]},
    {"id": "Q99", "text": "MKI67 TOP2A PCNA CDK1 CCNB1 EZH2 MELK",
     "category": "expression", "expected_clusters": [NB],
     "expected_genes": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "EZH2", "MELK"]},
    {"id": "Q100", "text": "PHOX2B ISL1 CD68 CD163 CD3D MS4A1 PLP1 PECAM1 DCN CYP11A1 ALB",
     "category": "expression", "expected_clusters": [NB, MAC, TCELL, BCELL, SCHWANN, ENDO, FIB, ADRENAL, HEPATO],
     "expected_genes": ["PHOX2B", "ISL1", "CD68", "CD163", "CD3D", "MS4A1", "PLP1", "PECAM1", "DCN", "CYP11A1", "ALB"]},
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
# MAIN COMPARISON
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

    # DT6 uses recall@1,2,3,5 (11 clusters)
    RECALL_KS = [1, 2, 3, 5]

    print("\n" + "=" * 70)
    print("[CW] Running CellWhisperer on 100 queries (DT6)...")
    print("=" * 70)

    cw_results = {"ontology": [], "expression": []}
    for q in QUERIES:
        ranked = cw_scorer.score_query(q["text"], top_k=11)
        entry = {"query_id": q["id"], "query_text": q["text"], "expected": q["expected_clusters"],
                 "retrieved_top10": ranked[:10], "mrr": round(mrr(q["expected_clusters"], ranked), 4)}
        for k in RECALL_KS:
            entry[f"recall@{k}"] = round(cluster_recall_at_k(q["expected_clusters"], ranked, k), 4)
        cw_results[q["category"]].append(entry)
        print(f"  [{q['id']}] R@1={entry['recall@1']:.2f}  MRR={entry['mrr']:.2f}  Top3={ranked[:3]}")

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
    print("ELISA vs CellWhisperer — DT6: High-Risk Neuroblastoma (100 queries)")
    print("=" * 90)
    print(f"\n{'Category':<14} {'Mode':<16} {'R@1':>7} {'R@2':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'GeneR':>7}")
    print("-" * 80)
    for cat in ["ontology", "expression"]:
        for mode in ALL_MODES:
            r1, r2, r3, r5 = gm(cat, mode, "mean_recall@1"), gm(cat, mode, "mean_recall@2"), gm(cat, mode, "mean_recall@3"), gm(cat, mode, "mean_recall@5")
            mmr, gr = gm(cat, mode, "mean_mrr"), gm(cat, mode, "mean_gene_recall")
            gr_s = "  N/A" if mode == "cellwhisperer_real" else f"{gr:.3f}"
            print(f"{cat:<14} {MODE_LABELS.get(mode, mode):<16} {r1:>7.3f} {r2:>7.3f} {r3:>7.3f} {r5:>7.3f} {mmr:>7.3f} {gr_s:>7}")
        print()

    print("── Overall ──")
    for mode in ALL_MODES:
        r1 = np.mean([gm(c, mode, "mean_recall@1") for c in ["ontology", "expression"]])
        print(f"  {MODE_LABELS.get(mode, mode):<16} Recall@1={r1:.3f}")

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

    # Fig 1: Recall@1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES)); vals = [gm(cat, m, "mean_recall@1") for m in ALL_MODES]
        bars = ax.bar(x, vals, color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15); ax.set_ylabel("Mean Cluster Recall@1"); ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v > 0: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("ELISA vs CellWhisperer — DT6: High-Risk Neuroblastoma (100 queries)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig1_recall1.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig1_recall1.pdf"), bbox_inches="tight"); plt.close(); print(f"\n[FIG] fig1_recall1.png")

    # Fig 2: All metrics R@1, R@2, MRR
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, mk, mt in zip(axes, ["mean_recall@1", "mean_recall@2", "mean_mrr"], ["Recall@1", "Recall@2", "MRR"]):
        x = np.arange(len(ALL_MODES)); w = 0.35
        ont = [gm("ontology", m, mk) for m in ALL_MODES]; exp = [gm("expression", m, mk) for m in ALL_MODES]
        ax.bar(x - w / 2, ont, w, label="Ontology", alpha=0.85, color=[MODE_COLORS[m] for m in ALL_MODES], edgecolor="white")
        ax.bar(x + w / 2, exp, w, label="Expression", alpha=0.45, color=[MODE_COLORS[m] for m in ALL_MODES], edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15); ax.set_title(mt, fontsize=13, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig2_all_metrics.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig2_all_metrics.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig2_all_metrics.png")

    # Fig 3: Cluster Recall vs Gene Recall
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES)); w = 0.35
        cv = [gm(cat, m, "mean_recall@1") for m in ALL_MODES]; gv = [gm(cat, m, "mean_gene_recall") for m in ALL_MODES]
        bars1 = ax.bar(x - w / 2, cv, w, label="Cluster Recall@1", color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + w / 2, gv, w, label="Gene Recall", color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.45, edgecolor="black", linewidth=0.8, hatch="///")
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9); ax.set_ylim(0, 1.15); ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars1, cv):
            if v > 0.05: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars2, gv):
            if v > 0.05: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    legend_elements = [Patch(facecolor="#888888", alpha=0.85, label="Cluster Recall@1"), Patch(facecolor="#888888", alpha=0.45, hatch="///", edgecolor="black", label="Gene Recall")]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=9)
    cw_idx = ALL_MODES.index("cellwhisperer_real")
    for ax in axes: ax.text(cw_idx + 0.175, 0.05, "N/A", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#E91E63")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig3_cluster_vs_gene.png")

    # Fig 4: Radar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    rl = ["Ont R@1", "Ont R@2", "Ont MRR", "Exp R@1", "Exp R@2", "Exp MRR"]; N = len(rl)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]; angles += angles[:1]
    for mode, color, ls in [("cellwhisperer_real", "#E91E63", "--"), ("semantic", "#2196F3", "-"), ("scgpt", "#FF9800", "-"), ("union", "#4CAF50", "-")]:
        v = []
        for cat in ["ontology", "expression"]:
            for mk in ["mean_recall@1", "mean_recall@2", "mean_mrr"]: v.append(gm(cat, mode, mk))
        v += v[:1]
        ax.plot(angles, v, linewidth=2, linestyle=ls, label=MODE_LABELS.get(mode, mode), color=color)
        ax.fill(angles, v, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rl, fontsize=10); ax.set_ylim(0, 1)
    ax.set_title("ELISA vs CellWhisperer\n(DT6: High-Risk Neuroblastoma)", fontsize=13, fontweight="bold", pad=20)
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
        cw_e = cw_results[cat]; qids = [e["query_id"] for e in cw_e]; cw_r1 = [e["recall@1"] for e in cw_e]
        el_union = elisa_detail.get(cat, {}).get("union", [])
        el_r1 = [e.get("recall@1", 0) for e in el_union] if el_union else [0] * len(qids)
        n = min(25, len(qids), len(el_r1)); y = np.arange(n); h = 0.35
        ax.barh(y - h / 2, el_r1[:n], h, label="ELISA Union", color="#4CAF50", alpha=0.8)
        ax.barh(y + h / 2, cw_r1[:n], h, label="CellWhisperer", color="#E91E63", alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(qids[:n], fontsize=8); ax.set_xlabel("Recall@1"); ax.set_xlim(0, 1.1)
        ax.set_title(f"{cat.capitalize()} Queries (first 25)", fontsize=13, fontweight="bold"); ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    plt.suptitle("Per-Query: ELISA Union vs CellWhisperer (DT6)", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "fig6_perquery.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(out_dir, "fig6_perquery.pdf"), bbox_inches="tight"); plt.close(); print("[FIG] fig6_perquery.png")

    # ── Save JSON ──
    output = {
        "cellwhisperer": {"results": cw_results, "aggregated": cw_agg},
        "elisa_summary": elisa_summary, "elisa_analytical": elisa_analytical,
        "comparison": {cat: {mode: {mk: gm(cat, mode, mk) for mk in [f"mean_recall@{k}" for k in RECALL_KS] + ["mean_mrr"]} for mode in ALL_MODES} for cat in cats},
        "modes": ALL_MODES, "n_queries": len(QUERIES), "dataset": "DT6_Neuroblastoma",
        "paper": "Yu et al. Nat Genet 2025", "n_clusters": 11,
        "cluster_names": [BCELL, SCHWANN, TCELL, ADRENAL, DC, ENDO, FIB, HEPATO, KIDNEY, MAC, NB],
        "timestamp": datetime.now().isoformat(),
    }
    rp = os.path.join(out_dir, "elisa_vs_cellwhisperer_DT6_results.json")
    with open(rp, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")


def main():
    parser = argparse.ArgumentParser(description="ELISA vs CellWhisperer — DT6 High-Risk Neuroblastoma")
    parser.add_argument("--elisa-results", required=True, help="ELISA benchmark_v5_results.json from DT6")
    parser.add_argument("--cw-npz", required=True, help="CellWhisperer full_output.npz")
    parser.add_argument("--cw-leiden", required=True, help="CellWhisperer leiden_umap_embeddings.h5ad")
    parser.add_argument("--cw-ckpt", required=True, help="cellwhisperer_clip_v1.ckpt")
    parser.add_argument("--cf-h5ad", required=True, help="Original read_count_table.h5ad")
    parser.add_argument("--paper", default="DT6")
    parser.add_argument("--out", default="comparison_DT6/")
    args = parser.parse_args()
    cw = CellWhispererScorer(args.cw_npz, args.cw_leiden, args.cw_ckpt, args.cf_h5ad)
    run_comparison(cw, args.elisa_results, args.paper, args.out)
    print("\n[DONE]")

if __name__ == "__main__":
    main()
