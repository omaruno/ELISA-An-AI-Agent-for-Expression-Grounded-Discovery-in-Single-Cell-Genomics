#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Benchmark v5.1 — DT6 Longitudinal High-Risk Neuroblastoma (Yu et al. Nat Genet 2025)
============================================================================================
100 Queries — Longitudinal Neuroblastoma Multiomics

Paper: "Longitudinal single-cell multiomic atlas of high-risk neuroblastoma
       reveals chemotherapy-induced tumor microenvironment rewiring"
       Yu, Biyik-Sit, Uzun, Chen et al., Nature Genetics (2025) 57:1142–1154

Dataset: 22 patients with high-risk neuroblastoma, paired diagnostic biopsy
         and post-induction chemotherapy surgical resection.
         snRNA-seq (372,619 cells), snATAC-seq (144,366 cells), WGS (22 pairs).

Evaluates ELISA's dual-modality retrieval against a random baseline:

  Baseline:
    1. Random                — random k clusters (establishes floor)

  ELISA modalities:
    2. Semantic          — BioBERT on full text (name + GO + Reactome + markers)
    3. scGPT             — expression-conditioned retrieval in scGPT space
    4. Union (Sem+scGPT) — ADDITIVE union: full primary top-k + unique from secondary

Actual cluster IDs (11 clusters):
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

Usage:
    python elisa_benchmark_v5_1_DT6.py \\
        --base /path/to/embeddings \\
        --pt-name hybrid_v3_DT6_NB.pt \\
        --paper DT6 \\
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
SCHWANN     = "Schwann cell"
TCELL       = "T cell"
ADRENAL     = "cortical cell of adrenal gland"
DC          = "dendritic cell"
ENDO        = "endothelial cell"
FIB         = "fibroblast"
HEPATO      = "hepatocyte"
KIDNEY      = "kidney cell"
MAC         = "macrophage"
NB          = "neuroblast (sensu Vertebrata)"


# ============================================================
# PAPER CONFIGURATIONS
# ============================================================

BENCHMARK_PAPERS = {
    "DT6": {
        "id": "DT6",
        "title": "Longitudinal single-cell multiomic atlas of high-risk neuroblastoma reveals chemotherapy-induced tumor microenvironment rewiring",
        "doi": "10.1038/s41588-025-02158-6",
        "pt_name": "hybrid_v3_DT6_NB.pt",
        "cells_csv": "metadata_cells.csv",
        "condition_col": "timepoint",
        "conditions": ["DX", "PTX"],

        "ground_truth_genes": [
            "PHOX2B", "ISL1", "HAND2", "TH", "DBH", "DDC", "CHGA", "CHGB",
            "MYCN", "ALK", "NTRK1", "NTRK2", "RET",
            "PHOX2A", "GATA3", "ASCL1",
            "CACNA1B", "CACNA2D1", "SYN2", "KCNMA1", "KCNQ3", "GPC5",
            "CREB5", "GRIK4",
            "SLC18A2", "SLC6A3", "AGTR2", "ATP2A2",
            "MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5", "BUB1B", "PCNA",
            "CDK1", "CCNB1", "ASPM", "KIF11", "KIF14", "MELK",
            "RPL3", "RPL4", "RPS6", "RPS3",
            "YAP1", "FN1", "SPARC", "VIM", "COL1A1", "COL1A2", "COL4A1",
            "COL4A2", "SERPINE1", "THBS2", "NECTIN2", "NNMT",
            "ETS1", "ETV6", "ELF1", "KLF6", "KLF7", "RUNX1", "ZNF148",
            "JUN", "FOS", "JUNB", "JUND", "FOSL2", "BACH1", "BACH2",
            "FOSB", "ATF3",
            "CD68", "CD163", "CD86", "CSF1R", "MRC1",
            "IL18", "VCAN", "CCL4", "C1QC", "SPP1", "F13A1", "HS3ST2",
            "THY1", "LYVE1", "CYP27A1", "VEGFA",
            "HBEGF", "ERBB4", "EREG", "TGFA", "AREG", "EGFR", "ICAM1",
            "MAPK1", "MAPK3", "AKT1",
            "CD247", "CD96", "CD3D", "CD3E", "CD8A", "CD4",
            "GZMA", "GZMB", "PRF1", "IFNG",
            "PAX5", "MS4A1", "CD19", "CD79A",
            "IRF8", "FLT3", "CLEC9A", "CD1C", "CD80",
            "PDGFRB", "DCN", "LUM", "ACTA2", "FAP", "PDGFRA",
            "PLP1", "CDH19", "SOX10", "MPZ", "MBP", "S100B",
            "PECAM1", "PTPRB", "CDH5", "VWF", "KDR", "FLT1",
            "CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1",
            "ALB", "DCDC2", "HNF4A", "APOB",
            "PKHD1", "PAX2", "WT1",
            "CD274", "PDCD1", "CTLA4", "TIGIT", "LAG3",
            "THBS1", "CD47", "ITGB1", "ITGA3",
            "VEGFA", "GPC1", "NRP1",
            "MMP2", "MMP9", "TIMP1",
            "HLA-A", "HLA-B", "HLA-C", "B2M",
            "HLA-DRA", "HLA-DRB1",
        ],

        "ground_truth_interactions": [
            ("HBEGF", "ERBB4", MAC, NB),
            ("HBEGF", "EGFR", MAC, NB),
            ("HBEGF", "CD9", MAC, NB),
            ("TGFA", "ERBB4", MAC, NB),
            ("EREG", "ERBB4", MAC, NB),
            ("EREG", "EGFR", MAC, NB),
            ("AREG", "EGFR", MAC, NB),
            ("VCAN", "EGFR", MAC, NB),
            ("VCAN", "ITGB1", MAC, NB),
            ("THBS1", "CD47", MAC, NB),
            ("THBS1", "ITGB1", MAC, NB),
            ("THBS1", "ITGA3", MAC, NB),
            ("THBS1", "LRP5", MAC, NB),
            ("VEGFA", "GPC1", MAC, NB),
            ("APOE", "LDLR", MAC, NB),
            ("APOE", "VLDLR", MAC, NB),
        ],

        "ground_truth_pathways": [
            "ErbB signaling pathway",
            "MAPK signaling pathway",
            "PI3K-Akt signaling pathway",
            "Cell cycle",
            "Axon guidance",
            "Calcium signaling pathway",
            "Dopaminergic synapse",
            "Oxidative phosphorylation",
            "ECM-receptor interaction",
            "Focal adhesion",
        ],

        "proportion_changes": {
            "increased_in_PTX": [MAC, SCHWANN],
            "decreased_in_PTX": [BCELL],
        },

        "queries": [

            # ================================================================
            # ONTOLOGY QUERIES (Q01–Q50)
            # ================================================================

            # Q01
            {
                "text": "neuroblast neoplastic cell of sympathetic nervous system expressing PHOX2B and ISL1",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["PHOX2B", "ISL1", "HAND2", "TH", "DBH", "CHGA"],
            },
            # Q02
            {
                "text": "neuroblastoma tumor cell with MYCN amplification and proliferative phenotype",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["MYCN", "MKI67", "TOP2A", "EZH2", "PCNA", "BIRC5"],
            },
            # Q03
            {
                "text": "adrenergic neuroblast expressing catecholamine biosynthesis enzymes tyrosine hydroxylase",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["TH", "DBH", "DDC", "PHOX2B", "PHOX2A", "GATA3"],
            },
            # Q04
            {
                "text": "neuroblastoma cell with calcium and synaptic signaling pathway enrichment",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["CACNA1B", "CACNA2D1", "SYN2", "KCNMA1", "KCNQ3", "CREB5"],
            },
            # Q05
            {
                "text": "dopaminergic neuroblast expressing dopamine transporter and metabolic genes",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["SLC18A2", "TH", "DDC", "AGTR2", "ATP2A2", "PHOX2B"],
            },
            # Q06
            {
                "text": "proliferating neuroblastoma cell with cell cycle and DNA replication markers",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5", "BUB1B"],
            },
            # Q07
            {
                "text": "mesenchymal neuroblastoma cell state expressing extracellular matrix genes and YAP1",
                "category": "ontology",
                "expected_clusters": [NB, FIB],
                "expected_genes": ["YAP1", "FN1", "VIM", "COL1A1", "SERPINE1", "SPARC"],
            },
            # Q08
            {
                "text": "intermediate OXPHOS neuroblast with ribosomal gene expression and oxidative phosphorylation",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["RPL3", "RPL4", "RPS6", "RPS3", "MYCN", "PHOX2B"],
            },
            # Q09
            {
                "text": "EZH2 expressing neuroblastoma cell PRC2 polycomb repressive complex chromatin regulation",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["EZH2", "SMC4", "TOP2A", "MKI67", "PHOX2B", "MYCN"],
            },
            # Q10
            {
                "text": "neuroblastoma cell ERBB4 receptor expressing epidermal growth factor signaling",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["ERBB4", "EGFR", "HBEGF", "TGFA", "EREG", "MAPK1"],
            },
            # Q11
            {
                "text": "neuroblast with adrenergic transcription factor PHOX2A PHOX2B GATA3 expression",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["PHOX2A", "PHOX2B", "GATA3", "ASCL1", "ISL1", "HAND2"],
            },
            # Q12
            {
                "text": "neural crest derived neoplastic cell in pediatric tumor expressing chromogranin",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["CHGA", "CHGB", "PHOX2B", "ISL1", "TH", "DBH"],
            },
            # Q13
            {
                "text": "neuroblastoma cell immune evasion NECTIN2 and checkpoint ligand expression",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["NECTIN2", "CD274", "B2M", "HLA-A", "HLA-B", "HLA-C"],
            },
            # Q14
            {
                "text": "mesenchymal transition state in neuroblastoma with AP-1 transcription factors",
                "category": "ontology",
                "expected_clusters": [NB, FIB],
                "expected_genes": ["JUN", "FOS", "JUNB", "JUND", "FOSL2", "BACH1"],
            },
            # Q15
            {
                "text": "tumor associated macrophage in neuroblastoma microenvironment CD68 CD163 expressing",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["CD68", "CD163", "CD86", "CSF1R", "MRC1", "SPP1"],
            },
            # Q16
            {
                "text": "pro-inflammatory macrophage IL18 expressing anti-tumor immune response",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["IL18", "CD68", "CD163", "CD86", "CSF1R", "HLA-DRA"],
            },
            # Q17
            {
                "text": "pro-angiogenic macrophage VCAN expressing promoting tumor vascularization",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["VCAN", "VEGFA", "CD68", "CD163", "EGFR", "SPP1"],
            },
            # Q18
            {
                "text": "immunosuppressive macrophage C1QC SPP1 complement expressing in tumor",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["C1QC", "SPP1", "CD68", "CD163", "APOE", "TREM2"],
            },
            # Q19
            {
                "text": "tissue resident macrophage F13A1 expressing phagocytic function in neuroblastoma",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["F13A1", "CD68", "CD163", "MRC1", "LYVE1", "CSF1R"],
            },
            # Q20
            {
                "text": "lipid associated macrophage HS3ST2 with metabolic phenotype in tumor",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["HS3ST2", "CYP27A1", "CD68", "CD163", "APOE", "LPL"],
            },
            # Q21
            {
                "text": "macrophage secreting HB-EGF ligand for ERBB4 receptor activation on neuroblasts",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["HBEGF", "CD68", "CD163", "TGFA", "EREG", "AREG"],
            },
            # Q22
            {
                "text": "CCL4 expressing pro-angiogenic macrophage chemokine signaling in tumor",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["CCL4", "CD68", "CD163", "VEGFA", "CSF1R", "CCL3"],
            },
            # Q23
            {
                "text": "proliferating macrophage MKI67 TOP2A expanding after chemotherapy",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["MKI67", "TOP2A", "CD68", "CD163", "CSF1R", "PCNA"],
            },
            # Q24
            {
                "text": "THY1 positive macrophage undefined myeloid phenotype in neuroblastoma",
                "category": "ontology",
                "expected_clusters": [MAC],
                "expected_genes": ["THY1", "CD68", "CD163", "MRC1", "CSF1R", "CD86"],
            },
            # Q25
            {
                "text": "T cell lymphocyte infiltrating neuroblastoma tumor expressing CD247 CD96",
                "category": "ontology",
                "expected_clusters": [TCELL],
                "expected_genes": ["CD247", "CD96", "CD3D", "CD3E", "CD8A", "CD4"],
            },
            # Q26
            {
                "text": "cytotoxic T cell with granzyme perforin mediated tumor cell killing",
                "category": "ontology",
                "expected_clusters": [TCELL],
                "expected_genes": ["GZMA", "GZMB", "PRF1", "IFNG", "CD8A", "CD3D"],
            },
            # Q27
            {
                "text": "tumor infiltrating T lymphocyte immune response to neuroblastoma",
                "category": "ontology",
                "expected_clusters": [TCELL],
                "expected_genes": ["CD3D", "CD3E", "CD247", "CD96", "CD8A", "CD4"],
            },
            # Q28
            {
                "text": "B cell lymphocyte PAX5 MS4A1 in neuroblastoma tumor immune microenvironment",
                "category": "ontology",
                "expected_clusters": [BCELL],
                "expected_genes": ["PAX5", "MS4A1", "CD19", "CD79A", "HLA-DRA", "HLA-DRB1"],
            },
            # Q29
            {
                "text": "B lymphocyte humoral immunity and antigen presentation in pediatric tumor",
                "category": "ontology",
                "expected_clusters": [BCELL],
                "expected_genes": ["MS4A1", "CD79A", "CD19", "PAX5", "HLA-DRA", "CD74"],
            },
            # Q30
            {
                "text": "dendritic cell IRF8 FLT3 antigen presentation priming T cell responses in tumor",
                "category": "ontology",
                "expected_clusters": [DC],
                "expected_genes": ["IRF8", "FLT3", "CLEC9A", "CD1C", "CD80", "HLA-DRA"],
            },
            # Q31
            {
                "text": "professional antigen presenting dendritic cell MHC class II expression",
                "category": "ontology",
                "expected_clusters": [DC],
                "expected_genes": ["HLA-DRA", "HLA-DRB1", "IRF8", "FLT3", "CD80", "CD86"],
            },
            # Q32
            {
                "text": "fibroblast stromal cell PDGFRB DCN extracellular matrix production in neuroblastoma",
                "category": "ontology",
                "expected_clusters": [FIB],
                "expected_genes": ["PDGFRB", "DCN", "LUM", "COL1A1", "COL1A2", "VIM"],
            },
            # Q33
            {
                "text": "cancer associated fibroblast FAP ACTA2 expressing in tumor stroma",
                "category": "ontology",
                "expected_clusters": [FIB],
                "expected_genes": ["FAP", "ACTA2", "COL1A1", "COL1A2", "PDGFRA", "DCN"],
            },
            # Q34
            {
                "text": "neural crest derived endoneurial fibroblast in neuroblastoma tissue",
                "category": "ontology",
                "expected_clusters": [FIB],
                "expected_genes": ["PDGFRB", "DCN", "COL1A1", "VIM", "LUM", "PDGFRA"],
            },
            # Q35
            {
                "text": "Schwann cell PLP1 CDH19 myelinating glial cell in neuroblastoma microenvironment",
                "category": "ontology",
                "expected_clusters": [SCHWANN],
                "expected_genes": ["PLP1", "CDH19", "SOX10", "MPZ", "MBP", "S100B"],
            },
            # Q36
            {
                "text": "Schwann cell precursor neural crest lineage expanding after therapy",
                "category": "ontology",
                "expected_clusters": [SCHWANN],
                "expected_genes": ["PLP1", "CDH19", "SOX10", "S100B", "MBP", "MPZ"],
            },
            # Q37
            {
                "text": "endothelial cell PECAM1 PTPRB vascular marker in neuroblastoma tumor vasculature",
                "category": "ontology",
                "expected_clusters": [ENDO],
                "expected_genes": ["PECAM1", "PTPRB", "CDH5", "VWF", "KDR", "FLT1"],
            },
            # Q38
            {
                "text": "tumor endothelium blood vessel lining cell expressing vascular endothelial markers",
                "category": "ontology",
                "expected_clusters": [ENDO],
                "expected_genes": ["PECAM1", "CDH5", "VWF", "PTPRB", "KDR", "FLT1"],
            },
            # Q39
            {
                "text": "adrenal cortex cell steroidogenesis CYP11A1 CYP11B1 adjacent normal tissue",
                "category": "ontology",
                "expected_clusters": [ADRENAL],
                "expected_genes": ["CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1"],
            },
            # Q40
            {
                "text": "cortical cell of adrenal gland steroid hormone biosynthesis normal adjacent tissue",
                "category": "ontology",
                "expected_clusters": [ADRENAL],
                "expected_genes": ["CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1"],
            },
            # Q41
            {
                "text": "hepatocyte ALB expressing liver cell from adjacent normal tissue in neuroblastoma biopsy",
                "category": "ontology",
                "expected_clusters": [HEPATO],
                "expected_genes": ["ALB", "DCDC2", "HNF4A", "APOB", "CYP3A4"],
            },
            # Q42
            {
                "text": "kidney cell renal tissue PKHD1 from adjacent normal tissue in neuroblastoma specimen",
                "category": "ontology",
                "expected_clusters": [KIDNEY],
                "expected_genes": ["PKHD1", "PAX2", "WT1", "SLC12A1"],
            },
            # Q43
            {
                "text": "chemotherapy induced tumor microenvironment rewiring macrophage expansion after therapy",
                "category": "ontology",
                "expected_clusters": [MAC, NB],
                "expected_genes": ["CD68", "CD163", "HBEGF", "PHOX2B", "MKI67", "VCAN"],
            },
            # Q44
            {
                "text": "HB-EGF ERBB4 paracrine signaling axis between macrophage and neuroblast promoting ERK",
                "category": "ontology",
                "expected_clusters": [MAC, NB],
                "expected_genes": ["HBEGF", "ERBB4", "MAPK1", "MAPK3", "CD68", "PHOX2B"],
            },
            # Q45
            {
                "text": "tumor immune evasion and antigen presentation in neuroblastoma",
                "category": "ontology",
                "expected_clusters": [NB, DC],
                "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "CD274", "NECTIN2"],
            },
            # Q46
            {
                "text": "VEGFA angiogenesis signaling in neuroblastoma tumor microenvironment",
                "category": "ontology",
                "expected_clusters": [MAC, ENDO],
                "expected_genes": ["VEGFA", "KDR", "FLT1", "GPC1", "NRP1", "PECAM1"],
            },
            # Q47
            {
                "text": "immune cell infiltration in high-risk neuroblastoma T cell B cell macrophage",
                "category": "ontology",
                "expected_clusters": [TCELL, BCELL, MAC],
                "expected_genes": ["CD3D", "CD247", "MS4A1", "CD68", "CD163", "CD8A"],
            },
            # Q48
            {
                "text": "THBS1 CD47 dont eat me signal between macrophage and neuroblastoma cell",
                "category": "ontology",
                "expected_clusters": [MAC, NB],
                "expected_genes": ["THBS1", "CD47", "ITGB1", "ITGA3", "CD68", "PHOX2B"],
            },
            # Q49
            {
                "text": "neuroblastoma cell expressing ALK receptor tyrosine kinase oncogenic driver",
                "category": "ontology",
                "expected_clusters": [NB],
                "expected_genes": ["ALK", "MYCN", "PHOX2B", "RET", "NTRK1", "NTRK2"],
            },
            # Q50
            {
                "text": "tumor microenvironment cell diversity neuroblasts fibroblasts Schwann endothelial macrophages",
                "category": "ontology",
                "expected_clusters": [NB, FIB, SCHWANN, ENDO, MAC],
                "expected_genes": ["PHOX2B", "DCN", "PLP1", "PECAM1", "CD68"],
            },

            # ================================================================
            # EXPRESSION QUERIES (Q51–Q100)
            # ================================================================

            # Q51
            {
                "text": "PHOX2B ISL1 HAND2 TH DBH DDC CHGA",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["PHOX2B", "ISL1", "HAND2", "TH", "DBH", "DDC", "CHGA"],
            },
            # Q52
            {
                "text": "MYCN MKI67 TOP2A EZH2 SMC4 BIRC5",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["MYCN", "MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5"],
            },
            # Q53
            {
                "text": "PHOX2A PHOX2B GATA3 ASCL1 ISL1 HAND2",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["PHOX2A", "PHOX2B", "GATA3", "ASCL1", "ISL1", "HAND2"],
            },
            # Q54
            {
                "text": "CACNA1B SYN2 KCNMA1 KCNQ3 GPC5 CREB5",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["CACNA1B", "SYN2", "KCNMA1", "KCNQ3", "GPC5", "CREB5"],
            },
            # Q55
            {
                "text": "SLC18A2 TH DDC AGTR2 ATP2A2 PHOX2B",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["SLC18A2", "TH", "DDC", "AGTR2", "ATP2A2", "PHOX2B"],
            },
            # Q56
            {
                "text": "MKI67 TOP2A EZH2 SMC4 BIRC5 BUB1B ASPM KIF11",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5", "BUB1B", "ASPM", "KIF11"],
            },
            # Q57
            {
                "text": "YAP1 FN1 VIM COL1A1 SERPINE1 SPARC THBS2",
                "category": "expression",
                "expected_clusters": [NB, FIB],
                "expected_genes": ["YAP1", "FN1", "VIM", "COL1A1", "SERPINE1", "SPARC", "THBS2"],
            },
            # Q58
            {
                "text": "ERBB4 EGFR HBEGF TGFA EREG AREG",
                "category": "expression",
                "expected_clusters": [NB, MAC],
                "expected_genes": ["ERBB4", "EGFR", "HBEGF", "TGFA", "EREG", "AREG"],
            },
            # Q59
            {
                "text": "NECTIN2 CD274 B2M HLA-A HLA-B PHOX2B",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["NECTIN2", "CD274", "B2M", "HLA-A", "HLA-B", "PHOX2B"],
            },
            # Q60
            {
                "text": "JUN FOS JUNB JUND FOSL2 BACH1 BACH2",
                "category": "expression",
                "expected_clusters": [NB, FIB],
                "expected_genes": ["JUN", "FOS", "JUNB", "JUND", "FOSL2", "BACH1", "BACH2"],
            },
            # Q61
            {
                "text": "CHGA CHGB PHOX2B ISL1 NTRK1 RET",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["CHGA", "CHGB", "PHOX2B", "ISL1", "NTRK1", "RET"],
            },
            # Q62
            {
                "text": "ETS1 ETV6 ELF1 KLF6 KLF7 RUNX1 ZNF148",
                "category": "expression",
                "expected_clusters": [NB, FIB],
                "expected_genes": ["ETS1", "ETV6", "ELF1", "KLF6", "KLF7", "RUNX1", "ZNF148"],
            },
            # Q63
            {
                "text": "ALK MYCN NTRK2 PHOX2B TH",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["ALK", "MYCN", "NTRK2", "PHOX2B", "TH"],
            },
            # Q64
            {
                "text": "CD68 CD163 CD86 CSF1R MRC1 SPP1",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["CD68", "CD163", "CD86", "CSF1R", "MRC1", "SPP1"],
            },
            # Q65
            {
                "text": "IL18 CD68 CD163 CD86 HLA-DRA CSF1R",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["IL18", "CD68", "CD163", "CD86", "HLA-DRA", "CSF1R"],
            },
            # Q66
            {
                "text": "VCAN VEGFA CD68 CD163 SPP1 EGFR",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["VCAN", "VEGFA", "CD68", "CD163", "SPP1", "EGFR"],
            },
            # Q67
            {
                "text": "C1QC SPP1 CD68 CD163 APOE TREM2",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["C1QC", "SPP1", "CD68", "CD163", "APOE", "TREM2"],
            },
            # Q68
            {
                "text": "F13A1 CD68 CD163 MRC1 LYVE1 CSF1R",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["F13A1", "CD68", "CD163", "MRC1", "LYVE1", "CSF1R"],
            },
            # Q69
            {
                "text": "HS3ST2 CYP27A1 CD68 CD163 APOE LPL",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["HS3ST2", "CYP27A1", "CD68", "CD163", "APOE", "LPL"],
            },
            # Q70
            {
                "text": "HBEGF TGFA EREG AREG CD68 CD163",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["HBEGF", "TGFA", "EREG", "AREG", "CD68", "CD163"],
            },
            # Q71
            {
                "text": "CCL4 CD68 CD163 VEGFA CSF1R CCL3",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["CCL4", "CD68", "CD163", "VEGFA", "CSF1R", "CCL3"],
            },
            # Q72
            {
                "text": "THY1 CD68 CD163 MRC1 CSF1R CD86",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["THY1", "CD68", "CD163", "MRC1", "CSF1R", "CD86"],
            },
            # Q73
            {
                "text": "CD247 CD96 CD3D CD3E CD8A CD4",
                "category": "expression",
                "expected_clusters": [TCELL],
                "expected_genes": ["CD247", "CD96", "CD3D", "CD3E", "CD8A", "CD4"],
            },
            # Q74
            {
                "text": "GZMA GZMB PRF1 IFNG CD8A CD3D",
                "category": "expression",
                "expected_clusters": [TCELL],
                "expected_genes": ["GZMA", "GZMB", "PRF1", "IFNG", "CD8A", "CD3D"],
            },
            # Q75
            {
                "text": "PAX5 MS4A1 CD19 CD79A HLA-DRA HLA-DRB1",
                "category": "expression",
                "expected_clusters": [BCELL],
                "expected_genes": ["PAX5", "MS4A1", "CD19", "CD79A", "HLA-DRA", "HLA-DRB1"],
            },
            # Q76
            {
                "text": "IRF8 FLT3 CLEC9A CD1C CD80 HLA-DRA",
                "category": "expression",
                "expected_clusters": [DC],
                "expected_genes": ["IRF8", "FLT3", "CLEC9A", "CD1C", "CD80", "HLA-DRA"],
            },
            # Q77
            {
                "text": "PDGFRB DCN LUM COL1A1 COL1A2 VIM",
                "category": "expression",
                "expected_clusters": [FIB],
                "expected_genes": ["PDGFRB", "DCN", "LUM", "COL1A1", "COL1A2", "VIM"],
            },
            # Q78
            {
                "text": "FAP ACTA2 COL1A1 PDGFRA DCN LUM",
                "category": "expression",
                "expected_clusters": [FIB],
                "expected_genes": ["FAP", "ACTA2", "COL1A1", "PDGFRA", "DCN", "LUM"],
            },
            # Q79
            {
                "text": "PLP1 CDH19 SOX10 MPZ MBP S100B",
                "category": "expression",
                "expected_clusters": [SCHWANN],
                "expected_genes": ["PLP1", "CDH19", "SOX10", "MPZ", "MBP", "S100B"],
            },
            # Q80
            {
                "text": "PECAM1 PTPRB CDH5 VWF KDR FLT1",
                "category": "expression",
                "expected_clusters": [ENDO],
                "expected_genes": ["PECAM1", "PTPRB", "CDH5", "VWF", "KDR", "FLT1"],
            },
            # Q81
            {
                "text": "CYP11A1 CYP11B1 CYP17A1 STAR NR5A1",
                "category": "expression",
                "expected_clusters": [ADRENAL],
                "expected_genes": ["CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1"],
            },
            # Q82
            {
                "text": "ALB DCDC2 HNF4A APOB",
                "category": "expression",
                "expected_clusters": [HEPATO],
                "expected_genes": ["ALB", "DCDC2", "HNF4A", "APOB"],
            },
            # Q83
            {
                "text": "PKHD1 PAX2 WT1 SLC12A1",
                "category": "expression",
                "expected_clusters": [KIDNEY],
                "expected_genes": ["PKHD1", "PAX2", "WT1", "SLC12A1"],
            },
            # Q84
            {
                "text": "PHOX2B CD68 CD3D MS4A1 PECAM1 DCN PLP1",
                "category": "expression",
                "expected_clusters": [NB, MAC, TCELL, BCELL, ENDO, FIB, SCHWANN],
                "expected_genes": ["PHOX2B", "CD68", "CD3D", "MS4A1", "PECAM1", "DCN", "PLP1"],
            },
            # Q85
            {
                "text": "HBEGF ERBB4 CD68 PHOX2B MAPK1",
                "category": "expression",
                "expected_clusters": [MAC, NB],
                "expected_genes": ["HBEGF", "ERBB4", "CD68", "PHOX2B", "MAPK1"],
            },
            # Q86
            {
                "text": "VCAN THBS1 CD47 ITGB1 CD68 PHOX2B",
                "category": "expression",
                "expected_clusters": [MAC, NB],
                "expected_genes": ["VCAN", "THBS1", "CD47", "ITGB1", "CD68", "PHOX2B"],
            },
            # Q87
            {
                "text": "HLA-A HLA-B HLA-C B2M HLA-DRA HLA-DRB1",
                "category": "expression",
                "expected_clusters": [NB, DC, MAC, BCELL],
                "expected_genes": ["HLA-A", "HLA-B", "HLA-C", "B2M", "HLA-DRA", "HLA-DRB1"],
            },
            # Q88
            {
                "text": "VEGFA KDR FLT1 NRP1 GPC1 PECAM1",
                "category": "expression",
                "expected_clusters": [MAC, ENDO],
                "expected_genes": ["VEGFA", "KDR", "FLT1", "NRP1", "GPC1", "PECAM1"],
            },
            # Q89
            {
                "text": "CD68 IL18 VCAN C1QC SPP1 F13A1 HS3ST2 CCL4 THY1",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["CD68", "IL18", "VCAN", "C1QC", "SPP1", "F13A1", "HS3ST2", "CCL4", "THY1"],
            },
            # Q90
            {
                "text": "PHOX2B MKI67 TOP2A YAP1 CACNA1B SLC18A2",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["PHOX2B", "MKI67", "TOP2A", "YAP1", "CACNA1B", "SLC18A2"],
            },
            # Q91
            {
                "text": "APOE LDLR VLDLR LPL HS3ST2 CD68",
                "category": "expression",
                "expected_clusters": [MAC],
                "expected_genes": ["APOE", "LDLR", "VLDLR", "LPL", "HS3ST2", "CD68"],
            },
            # Q92
            {
                "text": "THBS1 ITGB1 ITGA3 LRP5 CD47 FN1",
                "category": "expression",
                "expected_clusters": [MAC, NB, FIB],
                "expected_genes": ["THBS1", "ITGB1", "ITGA3", "LRP5", "CD47", "FN1"],
            },
            # Q93
            {
                "text": "COL1A1 COL1A2 COL4A1 COL4A2 FN1 VIM SPARC",
                "category": "expression",
                "expected_clusters": [FIB, NB],
                "expected_genes": ["COL1A1", "COL1A2", "COL4A1", "COL4A2", "FN1", "VIM", "SPARC"],
            },
            # Q94
            {
                "text": "MAPK1 MAPK3 AKT1 ERBB4 EGFR HBEGF",
                "category": "expression",
                "expected_clusters": [NB, MAC],
                "expected_genes": ["MAPK1", "MAPK3", "AKT1", "ERBB4", "EGFR", "HBEGF"],
            },
            # Q95
            {
                "text": "CD274 PDCD1 CTLA4 TIGIT LAG3 NECTIN2",
                "category": "expression",
                "expected_clusters": [NB, TCELL],
                "expected_genes": ["CD274", "PDCD1", "CTLA4", "TIGIT", "LAG3", "NECTIN2"],
            },
            # Q96
            {
                "text": "PHOX2B CD68 PLP1 PECAM1 DCN IRF8 PAX5 CD247",
                "category": "expression",
                "expected_clusters": [NB, MAC, SCHWANN, ENDO, FIB, DC, BCELL, TCELL],
                "expected_genes": ["PHOX2B", "CD68", "PLP1", "PECAM1", "DCN", "IRF8", "PAX5", "CD247"],
            },
            # Q97
            {
                "text": "CYP11A1 ALB PKHD1 PHOX2B CD68",
                "category": "expression",
                "expected_clusters": [ADRENAL, HEPATO, KIDNEY, NB, MAC],
                "expected_genes": ["CYP11A1", "ALB", "PKHD1", "PHOX2B", "CD68"],
            },
            # Q98
            {
                "text": "PHOX2B HBEGF ERBB4 VCAN SPP1 CD163 VEGFA",
                "category": "expression",
                "expected_clusters": [NB, MAC],
                "expected_genes": ["PHOX2B", "HBEGF", "ERBB4", "VCAN", "SPP1", "CD163", "VEGFA"],
            },
            # Q99
            {
                "text": "MKI67 TOP2A PCNA CDK1 CCNB1 EZH2 MELK",
                "category": "expression",
                "expected_clusters": [NB],
                "expected_genes": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "EZH2", "MELK"],
            },
            # Q100
            {
                "text": "PHOX2B ISL1 CD68 CD163 CD3D MS4A1 PLP1 PECAM1 DCN CYP11A1 ALB",
                "category": "expression",
                "expected_clusters": [NB, MAC, TCELL, BCELL, SCHWANN, ENDO, FIB, ADRENAL, HEPATO],
                "expected_genes": ["PHOX2B", "ISL1", "CD68", "CD163", "CD3D", "MS4A1", "PLP1", "PECAM1", "DCN", "CYP11A1", "ALB"],
            },
        ],
    },
}


# ============================================================
# METRICS
# ============================================================

def cluster_recall_at_k(expected, retrieved, k=5):
    if not expected: return 0.0
    ret_lower = [r.lower() for r in retrieved[:k]]
    found = 0
    for exp in expected:
        exp_l = exp.lower()
        if any(exp_l in r or r in exp_l or _word_overlap(exp_l, r) >= 0.5 for r in ret_lower):
            found += 1
    return found / len(expected)

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
# RETRIEVAL EVALUATOR
# ============================================================

class RetrievalEvaluator:
    MODES = ["random", "semantic", "scgpt", "union"]
    RECALL_KS = [1, 2, 3, 5]

    def __init__(self, engine):
        self.engine = engine

    def run_query_random(self, text, top_k=10):
        indices = list(range(len(self.engine.cluster_ids)))
        random.shuffle(indices)
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
            else:
                primary, secondary = (scgpt, sem) if mrr(_expected, scgpt) > mrr(_expected, sem) else (sem, scgpt)
        else:
            primary, secondary = sem, scgpt
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

    def _get_genes_from_clusters(self, cluster_ids, top_n=50):
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
                    r_runs = {k: [] for k in self.RECALL_KS}
                    mrr_runs, gr_runs = [], []
                    for _ in range(n_random_runs):
                        clusters = self.run_query_random(text, max(self.RECALL_KS))
                        for k in self.RECALL_KS: r_runs[k].append(cluster_recall_at_k(expected, clusters, k))
                        mrr_runs.append(mrr(expected, clusters))
                        rnd_genes = self._get_genes_from_clusters(clusters[:2], top_n=50)
                        gr_runs.append(len(expected_genes & rnd_genes) / len(expected_genes) if expected_genes else 0.0)
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": ["(random)"] * 10, "n_retrieved": top_k,
                             "mrr": round(np.mean(mrr_runs), 4), "gene_recall": round(np.mean(gr_runs), 4),
                             "genes_found": [], "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(np.mean(r_runs[k]), 4)
                    results[cat][mode].append(entry)
                elif mode == "union":
                    clusters = self.run_query_union(text, top_k, _sem_clusters=sem_clusters,
                                                     _scgpt_clusters=scgpt_clusters, _expected=expected)
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
                    for k in self.RECALL_KS:
                        entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k), 4)
                    retrieved_genes = self._get_genes_from_clusters(clusters[:3], top_n=50)
                    if expected_genes:
                        found = expected_genes & retrieved_genes
                        entry["gene_recall"] = round(len(found) / len(expected_genes), 4)
                        entry["genes_found"] = sorted(found)
                    else: entry["gene_recall"] = 0.0; entry["genes_found"] = []
                    for gain_k in [2, 3]:
                        best_single_k = max(cluster_recall_at_k(expected, sem_clusters, k=gain_k),
                                            cluster_recall_at_k(expected, scgpt_clusters, k=gain_k))
                        entry[f"additive_gain@{gain_k}"] = round(entry[f"recall@{gain_k}"] - best_single_k, 4)
                    results[cat][mode].append(entry)
                else:
                    clusters = sem_clusters if mode == "semantic" else scgpt_clusters
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": clusters[:top_k], "n_retrieved": len(clusters),
                             "mrr": round(mrr(expected, clusters), 4), "has_gene_evidence": True}
                    for k in self.RECALL_KS:
                        entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k), 4)
                    retrieved_genes = self._get_genes_from_clusters(clusters[:3], top_n=50)
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
                     "has_gene_evidence": entries[0].get("has_gene_evidence", False)}
                for k in self.RECALL_KS:
                    col = f"recall@{k}"
                    vals = [e.get(col, 0) for e in entries]
                    s[f"mean_{col}"] = round(np.mean(vals), 4)
                    s[f"std_{col}"] = round(np.std(vals), 4)
                if mode == "union":
                    for gain_k in [2, 3]:
                        s[f"mean_additive_gain@{gain_k}"] = round(np.mean([e.get(f"additive_gain@{gain_k}", 0) for e in entries]), 4)
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
                        if exp_l in s.lower() or s.lower() in exp_l or _word_overlap(exp_l, s.lower()) >= 0.5:
                            sem_found.add(exp); break
                    for s in scgpt_set:
                        if exp_l in s.lower() or s.lower() in exp_l or _word_overlap(exp_l, s.lower()) >= 0.5:
                            scgpt_found.add(exp); break
                    for s in union_set:
                        if exp_l in s.lower() or s.lower() in exp_l or _word_overlap(exp_l, s.lower()) >= 0.5:
                            union_found_set.add(exp); break
                all_queries.append({
                    "query": results[cat]["semantic"][i]["query"], "category": cat,
                    "expected": list(expected), "sem_found": list(sem_found),
                    "scgpt_found": list(scgpt_found), "union_found": list(union_found_set),
                    "only_semantic": list(sem_found - scgpt_found), "only_scgpt": list(scgpt_found - sem_found),
                    "both_found": list(sem_found & scgpt_found), "neither_found": list(expected - union_found_set),
                    "n_union_clusters": union_entry.get("n_retrieved", 0),
                    "primary_mode": union_entry.get("primary_mode", "semantic"),
                    "additive_gain": union_entry.get("additive_gain@3", 0),
                })
        total_expected = sum(len(q["expected"]) for q in all_queries)
        sem_total = sum(len(q["sem_found"]) for q in all_queries)
        scgpt_total = sum(len(q["scgpt_found"]) for q in all_queries)
        union_total = sum(len(q["union_found"]) for q in all_queries)
        best_single = max(sem_total, scgpt_total)
        return {
            "total_expected": total_expected, "semantic_found": sem_total, "scgpt_found": scgpt_total,
            "union_found": union_total, "best_single_found": best_single,
            "semantic_recall": round(sem_total / total_expected, 4) if total_expected else 0,
            "scgpt_recall": round(scgpt_total / total_expected, 4) if total_expected else 0,
            "union_recall": round(union_total / total_expected, 4) if total_expected else 0,
            "best_single_recall": round(best_single / total_expected, 4) if total_expected else 0,
            "additive_gain_clusters": union_total - best_single,
            "additive_gain_pct": round((union_total - best_single) / total_expected, 4) if total_expected else 0,
            "only_semantic_count": sum(len(q["only_semantic"]) for q in all_queries),
            "only_scgpt_count": sum(len(q["only_scgpt"]) for q in all_queries),
            "neither_count": sum(len(q["neither_found"]) for q in all_queries),
            "per_query": all_queries,
        }


# ============================================================
# ANALYTICAL MODULE EVALUATOR
# ============================================================

class AnalyticalEvaluator:
    def __init__(self, engine):
        self.engine = engine

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
        payload = self.engine.pathways()
        results = {}
        for pw in gt:
            pw_l = pw.lower()
            found, top_score, top_cluster, n_genes = False, 0, "", 0
            for pw_name, pw_data in payload.get("pathways", {}).items():
                if pw_l in pw_name.lower() or pw_name.lower() in pw_l:
                    for best in pw_data.get("scores", []):
                        if best.get("score", 0) > top_score:
                            found, top_score = True, best["score"]
                            top_cluster = best.get("cluster", ""); n_genes = best.get("n_genes_found", 0)
            results[pw] = {"found": found, "top_score": round(top_score, 4), "n_genes_found": n_genes, "top_cluster": top_cluster}
        fc = sum(1 for v in results.values() if v["found"])
        return {"pathways_found": fc, "pathways_expected": len(gt), "alignment": round(fc/len(gt)*100,1), "details": results}

    def evaluate_proportions(self, paper):
        pc = paper.get("proportion_changes", {})
        if not pc: return {"error": "No proportion changes defined"}
        payload = self.engine.proportions()
        fc_data = payload.get("proportion_fold_changes", [])
        if not fc_data: return {"error": "No fold change data"}
        consistent, total, details = 0, 0, []
        for item in fc_data:
            cluster = item["cluster"].lower()
            fc = 1.0
            for key in item:
                if key.startswith("fold_change"):
                    val = item[key]; fc = 999.0 if val=="inf" else float(val) if isinstance(val,(int,float)) else 1.0; break
            is_up = any(ct.lower() in cluster for ct in pc.get("increased_in_PTX", []))
            is_down = any(ct.lower() in cluster for ct in pc.get("decreased_in_PTX", []))
            if not is_up and not is_down: continue
            total += 1
            if (is_up and fc>1.0) or (is_down and fc<1.0):
                consistent += 1; details.append({"cluster": item["cluster"], "direction": "correct", "fc": fc})
            else: details.append({"cluster": item["cluster"], "direction": "WRONG", "expected": "up" if is_up else "down", "fc": fc})
        return {"total_checked": total, "consistent": consistent,
                "consistency_rate": round(consistent/total*100,1) if total else 0, "details": details}

    def evaluate_compare(self, paper):
        conditions = paper.get("conditions", [])
        if len(conditions) < 2: return {"error": "Need 2 conditions"}
        gt_set = set(g.upper() for g in paper.get("ground_truth_genes", []))
        payload = self.engine.compare(conditions[0], conditions[1], genes=paper.get("ground_truth_genes", []))
        all_genes = set()
        for cid, cdata in payload.get("clusters", {}).items():
            if isinstance(cdata, dict):
                for g in cdata.get("genes", []):
                    all_genes.add((g.get("gene","") if isinstance(g,dict) else g).upper())
        for gg in payload.get("summary",{}).get("condition_enriched_genes",{}).values():
            for g in gg: all_genes.add((g.get("gene","") if isinstance(g,dict) else g).upper())
        found = gt_set & all_genes
        return {"genes_requested": len(gt_set), "genes_found": len(found),
                "compare_recall": round(len(found)/len(gt_set)*100,1) if gt_set else 0,
                "n_clusters_analyzed": len(payload.get("clusters",{})),
                "found": sorted(found), "missed": sorted(gt_set - all_genes)}

    def evaluate_all(self, paper):
        return {"interactions": self.evaluate_interactions(paper), "pathways": self.evaluate_pathways(paper),
                "proportions": self.evaluate_proportions(paper), "compare": self.evaluate_compare(paper)}


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
        ax.set_ylim(0,1.15); ax.set_ylabel("Mean Cluster Recall@1"); ax.set_title(title, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v>0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_baselines.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_baselines.pdf"),bbox_inches="tight"); plt.close()

    # Fig 2: Recall curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cat, title in zip(axes, cats, titles):
        for mode, ls, marker in [("semantic","-","o"),("scgpt","--","s"),("union","-","D")]:
            vals = [summary.get(f"{cat}_{mode}",{}).get(f"mean_recall@{k}",0) for k in [1,2,3,5]]
            ax.plot([1,2,3,5], vals, ls, marker=marker, markersize=8, color=MC[mode], label=ML[mode].replace("\n"," "), linewidth=2)
            for k, v in zip([1,2,3,5], vals): ax.annotate(f"{v:.2f}", (k,v), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8, color=MC[mode])
        ax.set_xlabel("k"); ax.set_ylabel("Mean Cluster Recall@k"); ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([1,2,3,5]); ax.set_ylim(0,1.15); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"additive_union_recall_curve.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"additive_union_recall_curve.pdf"),bbox_inches="tight"); plt.close()

    # Fig 3: Recall vs Gene Recall
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, cat, title in zip(axes, cats, titles):
        x = np.arange(len(MODES)); w = 0.35
        cv = [summary.get(f"{cat}_{m}",{}).get("mean_recall@1",0) for m in MODES]
        gv = [summary.get(f"{cat}_{m}",{}).get("mean_gene_recall",0) for m in MODES]
        ax.bar(x-w/2,cv,w,color=[MC[m] for m in MODES],alpha=0.85,edgecolor="white")
        ax.bar(x+w/2,gv,w,color=[MC[m] for m in MODES],alpha=0.45,edgecolor="black",linewidth=0.8,hatch="///")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES],fontsize=9); ax.set_ylim(0,1.15); ax.set_title(title,fontsize=12,fontweight="bold"); ax.grid(axis="y",alpha=0.3)
    axes[0].legend(handles=[Patch(facecolor="#888",alpha=0.85,label="Cluster Recall@1"),Patch(facecolor="#888",alpha=0.45,hatch="///",edgecolor="black",label="Gene Recall")],loc="upper left",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_vs_gene_delivery.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_vs_gene_delivery.pdf"),bbox_inches="tight"); plt.close()

    # Fig 4: All metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, metric, mlabel in zip(axes, ["mean_recall@1","mean_recall@2","mean_mrr"], ["Recall@1","Recall@2","MRR"]):
        x = np.arange(len(MODES)); w = 0.35
        ov = [summary.get(f"ontology_{m}",{}).get(metric,0) for m in MODES]
        ev = [summary.get(f"expression_{m}",{}).get(metric,0) for m in MODES]
        ax.bar(x-w/2,ov,w,label="Ontology",alpha=0.85,color=[MC[m] for m in MODES],edgecolor="white")
        ax.bar(x+w/2,ev,w,label="Expression",alpha=0.45,color=[MC[m] for m in MODES],edgecolor="black",linewidth=0.5,hatch="//")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES],fontsize=9); ax.set_ylim(0,1.15); ax.set_title(mlabel,fontsize=12,fontweight="bold"); ax.grid(axis="y",alpha=0.3)
    axes[0].legend(loc="upper left",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_all_metrics.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_all_metrics.pdf"),bbox_inches="tight"); plt.close()

    # Fig 5: Complementarity
    comp = complementarity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    so=comp.get("only_semantic_count",0); sgo=comp.get("only_scgpt_count",0)
    bc=max(comp.get("union_found",0)-so-sgo,0); ne=comp.get("neither_count",0)
    bars=ax1.bar(["Both\nmodalities","Semantic\nonly","scGPT\nonly","Neither"],[bc,so,sgo,ne],color=["#4CAF50","#2196F3","#FF9800","#9E9E9E"],alpha=0.85,edgecolor="white",width=0.6)
    for bar,v in zip(bars,[bc,so,sgo,ne]):
        if v>0: ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,str(v),ha="center",va="bottom",fontweight="bold",fontsize=11)
    ax1.set_ylabel("Expected clusters found"); ax1.set_title("Modality Complementarity",fontweight="bold")
    ax1.text(0.98,0.95,f"Union recall: {comp.get('union_recall',0):.1%}\nBest single: {comp.get('best_single_recall',0):.1%}\nAdditive gain: +{comp.get('additive_gain_pct',0):.1%}\n  (+{comp.get('additive_gain_clusters',0)} clusters)",
             transform=ax1.transAxes,ha="right",va="top",fontsize=10,bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.5))
    pq=comp.get("per_query",[])
    qwg=sorted([(q["query"][:40],q["additive_gain"]) for q in pq if q.get("additive_gain",0)>0],key=lambda x:x[1],reverse=True)
    if qwg:
        ql,qg=zip(*qwg[:12]); y=np.arange(len(ql))
        ax2.barh(y,qg,color="#4CAF50",alpha=0.7,edgecolor="white"); ax2.set_yticks(y); ax2.set_yticklabels(ql,fontsize=8)
        ax2.set_xlabel("Additive Recall Gain"); ax2.set_title("Per-Query Additive Gain\n(union vs best single)",fontweight="bold"); ax2.invert_yaxis()
    else: ax2.text(0.5,0.5,"No additive gains\n(modalities agree)",transform=ax2.transAxes,ha="center",va="center",fontsize=12); ax2.set_title("Per-Query Additive Gain",fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"complementarity.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"complementarity.pdf"),bbox_inches="tight"); plt.close()

    # Fig 6: Radar
    ana=analytical
    mr={"Pathways":ana.get("pathways",{}).get("alignment",0)/100,"Interactions\n(LR)":ana.get("interactions",{}).get("lr_recovery_rate",0)/100,
        "Proportions":ana.get("proportions",{}).get("consistency_rate",0)/100,"Compare\n(gene recall)":ana.get("compare",{}).get("compare_recall",0)/100}
    lr=list(mr.keys()); vr=list(mr.values()); angles=np.linspace(0,2*np.pi,len(lr),endpoint=False).tolist(); vr+=vr[:1]; angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True)); ax.fill(angles,vr,alpha=0.25,color="#4CAF50"); ax.plot(angles,vr,"o-",color="#4CAF50",linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(lr,fontsize=10); ax.set_ylim(0,1.05)
    ax.set_title("Analytical Module Performance\n(DT6: High-Risk Neuroblastoma)",fontweight="bold",pad=20)
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
    print("ELISA BENCHMARK v5.1 — DT6: High-Risk Neuroblastoma (Yu et al. Nat Genet 2025)")
    print("="*100)
    print(f"\n{'Category':<14} {'Mode':<14} {'R@1':>7} {'R@2':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'GeneR':>7}")
    print("-"*80)
    for cat in ["ontology","expression"]:
        for mode in MODES:
            key = f"{cat}_{mode}"
            if key not in summary: continue
            s = summary[key]
            print(f"{cat:<14} {MD[mode]:<14} "
                  f"{s.get('mean_recall@1',0):>7.3f} {s.get('mean_recall@2',0):>7.3f} "
                  f"{s.get('mean_recall@3',0):>7.3f} {s.get('mean_recall@5',0):>7.3f} "
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
    parser = argparse.ArgumentParser(description="ELISA Benchmark v5.1 — DT6 High-Risk Neuroblastoma")
    parser.add_argument("--base", required=True, help="Path to embedding directory")
    parser.add_argument("--pt-name", default=None, help="Override .pt filename")
    parser.add_argument("--cells-csv", default=None, help="Override cells CSV")
    parser.add_argument("--paper", default="DT6", help="Paper ID")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k clusters")
    parser.add_argument("--out", default="benchmark_v5_DT6/", help="Output directory")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        rd = args.results_dir or out_dir
        with open(os.path.join(rd, "benchmark_v5_results.json")) as f: data = json.load(f)
        generate_figures(data[args.paper]["retrieval_summary"], data[args.paper]["complementarity"],
                         data[args.paper]["analytical"], out_dir)
        print("Figures regenerated."); return

    paper = BENCHMARK_PAPERS[args.paper]
    pt_name = args.pt_name or paper.get("pt_name", "")
    cells_csv = args.cells_csv or paper.get("cells_csv")
    sys.path.insert(0, os.path.dirname(args.base)); sys.path.insert(0, args.base); sys.path.insert(0, os.getcwd())

    from retrieval_engine_v4_hybrid import RetrievalEngine
    print(f"\n[BENCHMARK v5.1 — DT6] Loading engine: {args.base} / {pt_name}")
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
        def proportions(self, **kw): return proportion_analysis(self._eng.metadata, condition_col=paper.get("condition_col", "timepoint"))
        def compare(self, cond_a, cond_b, **kw): return comparative_analysis(self._eng.gene_stats, self._eng.metadata, condition_col=paper.get("condition_col", "timepoint"), group_a=cond_a, group_b=cond_b, **kw)

    engine_wrap = EngineWithAnalysis(engine)

    print(f"[BENCHMARK v5.1 — DT6] Running retrieval ({len(paper['queries'])} queries × {len(RetrievalEvaluator.MODES)} modes)...")
    t0 = time.time()
    ret_eval = RetrievalEvaluator(engine)
    ret_results = ret_eval.evaluate_queries(paper["queries"], top_k=args.top_k)
    ret_summary = ret_eval.compute_summary(ret_results)
    complementarity = ret_eval.compute_complementarity(ret_results, top_k=args.top_k)
    print(f"  Retrieval done in {time.time()-t0:.1f}s")

    print("[BENCHMARK v5.1 — DT6] Running analytical modules...")
    t0 = time.time()
    try: analytical = AnalyticalEvaluator(engine_wrap).evaluate_all(paper)
    except Exception as e:
        print(f"  [WARN] Analytical modules failed: {e}")
        analytical = {"pathways": {}, "interactions": {}, "proportions": {}, "compare": {}}
    print(f"  Analytical done in {time.time()-t0:.1f}s")

    print_summary(ret_summary, complementarity, analytical)

    output = {args.paper: {"retrieval_detail": ret_results, "retrieval_summary": ret_summary,
                            "complementarity": complementarity, "analytical": analytical,
                            "timestamp": datetime.now().isoformat(), "config": vars(args)}}
    results_path = os.path.join(out_dir, "benchmark_v5_results.json")
    with open(results_path, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {results_path}")

    print("[BENCHMARK v5.1 — DT6] Generating figures...")
    generate_figures(ret_summary, complementarity, analytical, out_dir)
    print("\n[BENCHMARK v5.1 — DT6] Complete!")


if __name__ == "__main__":
    main()
