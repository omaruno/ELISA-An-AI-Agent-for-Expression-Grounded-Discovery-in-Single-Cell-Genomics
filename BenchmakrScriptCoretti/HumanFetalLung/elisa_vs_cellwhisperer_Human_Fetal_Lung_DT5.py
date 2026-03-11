#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA vs CellWhisperer — DT3 Human Fetal Lung (Lim et al. EMBO J 2025)
=======================================================================
Head-to-Head comparison on fdAT2 organoid scRNA-seq data.

5 clusters (Cell Ontology IDs):
  0: basal cell
  1: lung multiciliated epithelial cell
  2: pulmonary alveolar type 2 cell
  3: pulmonary neuroendocrine cell
  4: respiratory tract epithelial cell

Step 1: Run ELISA benchmark first:
    python elisa_benchmark_v5_1_DT3.py \
        --base /path/to/embeddings \
        --pt-name fused_DT3_HumanFetalLung.pt \
        --paper DT3 \
        --out results_DT3/

Step 2: Run this script (in cellwhisperer env):
    conda activate cellwhisperer
    python elisa_vs_cellwhisperer_DT3.py \
        --elisa-results results_DT3/benchmark_v5_results.json \
        --cw-npz /path/to/cellwhisperer/full_output.npz \
        --cw-leiden /path/to/leiden_umap_embeddings.h5ad \
        --cw-ckpt /path/to/cellwhisperer_clip_v1.ckpt \
        --cf-h5ad /path/to/read_count_table.h5ad \
        --out comparison_DT3/
"""

import os, sys, json, argparse
from datetime import datetime
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# QUERIES — 50 ontology + 50 expression
# 5 clusters, balanced distribution
# ============================================================

QUERIES = [
    # ================================================================
    # ONTOLOGY QUERIES (Q01–Q50): concept-level, semantic advantage
    # ================================================================

    # --- Pulmonary alveolar type 2 cell (15 queries) ---

    {"id": "Q01", "text": "alveolar type 2 cell identity and surfactant protein production in fetal lung organoids",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "SFTPB", "SFTPA1", "NAPSA", "LAMP3", "NKX2-1"]},

    {"id": "Q02", "text": "mature AT2 cell markers and lamellar body formation in human lung",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "SFTPB", "ABCA3", "LAMP3", "HOPX", "NAPSA"]},

    {"id": "Q03", "text": "surfactant protein C maturation and intracellular trafficking in alveolar epithelium",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "ITCH", "UBE2N", "LAMP3", "ABCA3"]},

    {"id": "Q04", "text": "SFTPC processing through endosomal compartments and multivesicular bodies",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "EEA1", "MICALL1", "HRS", "VPS28", "LAMP3"]},

    {"id": "Q05", "text": "ITCH E3 ubiquitin ligase role in SFTPC trafficking and ubiquitination",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["ITCH", "SFTPC", "UBE2N", "NEDD4", "RABGEF1"]},

    {"id": "Q06", "text": "K63 ubiquitination of surfactant protein C for ESCRT recognition and MVB entry",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["UBE2N", "ITCH", "HRS", "VPS28", "SFTPC"]},

    {"id": "Q07", "text": "AT2 stem cell self-renewal and proliferation in fetal lung organoids",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["MKI67", "PCNA", "NKX2-1", "SFTPC", "HOPX"]},

    {"id": "Q08", "text": "surfactant metabolism and lipid transport in fetal alveolar epithelium",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["ABCA3", "SFTPB", "SFTPC", "SFTPD", "SFTA3", "CD36"]},

    {"id": "Q09", "text": "CXCL chemokine expressing AT2 subpopulation in fetal lung organoids",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["CXCL1", "CXCL2", "CXCL3", "SFTPC", "CCL2"]},

    {"id": "Q10", "text": "Wnt signaling pathway maintaining AT2 identity and inhibiting differentiation",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["CTNNB1", "SFTPC", "NKX2-1", "HOPX"]},

    {"id": "Q11", "text": "SFTPC-I73T pathogenic variant causing interstitial lung disease and AT2 dysfunction",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "ITCH", "EEA1", "MICALL1", "LAMP3"]},

    {"id": "Q12", "text": "vesicle-mediated transport and lysosome localization in AT2 surfactant processing",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["LAMP3", "ABCA3", "NAPSA", "EEA1", "CKAP4"]},

    {"id": "Q13", "text": "immune response gene expression and MHC class II in alveolar type 2 cells",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["CXCL1", "CCL2", "HLA-DPA1", "HLA-DPB1", "HLA-DRA"]},

    {"id": "Q14", "text": "NKX2-1 transcription factor regulation of surfactant gene expression",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["NKX2-1", "SFTPC", "SFTPB", "SFTPA1", "HOPX"]},

    {"id": "Q15", "text": "CRISPRi-mediated depletion of ITCH and UBE2N in alveolar organoids",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["ITCH", "UBE2N", "SFTPC", "LAMP3"]},

    # --- Respiratory tract epithelial cell / Intermediate (10 queries) ---

    {"id": "Q16", "text": "alveolar type 1 cell differentiation from AT2 organoids via YAP activation",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AQP5", "CAV1", "AGER", "HOPX"]},

    {"id": "Q17", "text": "AT2 to AT1 lineage transition through Wnt withdrawal and LATS inhibition",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["CAV1", "AGER", "AQP5", "SFTPC", "CTNNB1"]},

    {"id": "Q18", "text": "AT1 cell fate markers AQP5 CAV1 AGER in differentiated organoids",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AQP5", "CAV1", "AGER", "HOPX"]},

    {"id": "Q19", "text": "intermediate transitional cell state between AT2 and differentiated lineages",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["SOX2", "TP63", "SFTPC", "NKX2-1"]},

    {"id": "Q20", "text": "fetal lung tip progenitor differentiation into mature alveolar cells",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell", "pulmonary alveolar type 2 cell"],
     "expected_genes": ["NKX2-1", "SFTPC", "HOPX", "SOX9"]},

    {"id": "Q21", "text": "organoid engraftment in precision-cut lung slices and AT1 differentiation",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["SFTPC", "CAV1", "AGER", "AQP5"]},

    {"id": "Q22", "text": "epithelial cell plasticity and lineage transition in alveolar organoid culture",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AQP5", "AGER", "CAV1", "NKX2-1"]},

    {"id": "Q23", "text": "gas exchange surface epithelial cell differentiation and PDPN expression",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AQP5", "AGER", "CAV1", "PDPN"]},

    {"id": "Q24", "text": "SOX2 expression in transitional epithelial progenitor cells of fetal lung",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["SOX2", "NKX2-1", "HOPX", "CAV1"]},

    {"id": "Q25", "text": "flattened type 1 pneumocyte morphology and reduced surfactant expression",
     "category": "ontology",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AQP5", "AGER", "HOPX", "CAV1"]},

    # --- Basal cell (8 queries) ---

    {"id": "Q26", "text": "aberrant basal cell differentiation from AT2 cells in organoid culture",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT14"]},

    {"id": "Q27", "text": "airway basal stem cell markers TP63 and KRT5 in fetal lung organoids",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT14", "KRT15"]},

    {"id": "Q28", "text": "basal cell emergence and squamous metaplasia in alveolar organoid culture",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT14", "LAMB3"]},

    {"id": "Q29", "text": "TP63 positive progenitor cell with airway differentiation potential",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT14", "SOX2"]},

    {"id": "Q30", "text": "keratin 5 and keratin 14 expressing cells in fetal distal lung epithelium",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["KRT5", "KRT14", "TP63", "KRT15"]},

    {"id": "Q31", "text": "hypoxia-induced airway differentiation of alveolar epithelial cells",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT14"]},

    {"id": "Q32", "text": "basal-like cell laminin and collagen XVII adhesion molecule expression",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["LAMB3", "COL17A1", "KRT5", "TP63"]},

    {"id": "Q33", "text": "SOX2 driven airway progenitor identity in differentiating basal-like cells",
     "category": "ontology",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["SOX2", "TP63", "KRT5", "KRT14"]},

    # --- Pulmonary neuroendocrine cell (8 queries) ---

    {"id": "Q34", "text": "pulmonary neuroendocrine cell differentiation in AT2 organoids",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["ASCL1", "NEUROD1", "GRP", "CHGA", "SYP"]},

    {"id": "Q35", "text": "neuroendocrine progenitor cells co-expressing SFTPC and NE markers",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["ASCL1", "GRP", "SFTPC", "NKX2-1"]},

    {"id": "Q36", "text": "ASCL1 master regulator of neuroendocrine cell fate in fetal lung",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["ASCL1", "NEUROD1", "GRP", "CHGA"]},

    {"id": "Q37", "text": "gastrin releasing peptide and chromogranin A in pulmonary NE cells",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["GRP", "CHGA", "SYP", "ASCL1"]},

    {"id": "Q38", "text": "synaptophysin and neuropeptide secretion in lung neuroendocrine bodies",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["SYP", "CHGA", "GRP", "ASCL1", "CALCA"]},

    {"id": "Q39", "text": "neuroendocrine cell oxygen sensing and airway chemoreception",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["ASCL1", "GRP", "CHGA", "SYP"]},

    {"id": "Q40", "text": "NEUROD1 transcription factor activity in pulmonary NE lineage commitment",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["NEUROD1", "ASCL1", "GRP", "CHGA"]},

    {"id": "Q41", "text": "rare neuroendocrine cell population arising during organoid differentiation",
     "category": "ontology",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["ASCL1", "GRP", "SYP", "CHGA", "NEUROD1"]},

    # --- Lung multiciliated epithelial cell (7 queries) ---

    {"id": "Q42", "text": "ciliated cell-like differentiation in fetal AT2 organoid culture",
     "category": "ontology",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "DNAH5"]},

    {"id": "Q43", "text": "FOXJ1 master regulator of ciliogenesis in lung epithelial cells",
     "category": "ontology",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "DNAH5", "CAPS"]},

    {"id": "Q44", "text": "multiciliated cell motile cilia assembly and dynein arm formation",
     "category": "ontology",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["DNAH5", "FOXJ1", "DNAI1"]},

    {"id": "Q45", "text": "mucociliary clearance function in differentiated airway epithelium",
     "category": "ontology",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "DNAH5", "CAPS"]},

    {"id": "Q46", "text": "radial spoke head protein expression in motile ciliated cells of the lung",
     "category": "ontology",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["RSPH1", "FOXJ1", "DNAH5", "CAPS"]},

    {"id": "Q47", "text": "primary ciliary dyskinesia genes in pulmonary multiciliated epithelial cells",
     "category": "ontology",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["DNAH5", "DNAI1", "FOXJ1", "RSPH1"]},

    {"id": "Q48", "text": "cilia beating frequency regulation and calcium signaling in airway cells",
     "category": "ontology",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "CAPS", "DNAH5"]},

    # --- Multi-cluster queries (2 queries) ---

    {"id": "Q49", "text": "cell type heterogeneity across fdAT2 organoid lines including AT2 basal and NE cells",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell", "basal cell", "pulmonary neuroendocrine cell", "lung multiciliated epithelial cell", "respiratory tract epithelial cell"],
     "expected_genes": ["SFTPC", "TP63", "ASCL1", "FOXJ1", "AQP5"]},

    {"id": "Q50", "text": "multi-lineage differentiation potential of fetal lung tip progenitor organoids",
     "category": "ontology",
     "expected_clusters": ["pulmonary alveolar type 2 cell", "respiratory tract epithelial cell", "basal cell", "pulmonary neuroendocrine cell"],
     "expected_genes": ["SFTPC", "NKX2-1", "TP63", "ASCL1", "AQP5"]},

    # ================================================================
    # EXPRESSION QUERIES (Q51–Q100): gene-signature, scGPT advantage
    # ================================================================

    # --- Pulmonary alveolar type 2 cell (15 queries) ---

    {"id": "Q51", "text": "SFTPC SFTPB SFTPA1 SFTPA2 NAPSA LAMP3",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "NAPSA", "LAMP3"]},

    {"id": "Q52", "text": "SFTPC SFTPB ABCA3 LAMP3 HOPX NKX2-1",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "SFTPB", "ABCA3", "LAMP3", "HOPX", "NKX2-1"]},

    {"id": "Q53", "text": "NKX2-1 SLC34A2 HOPX SFTPC SFTA3",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["NKX2-1", "SLC34A2", "HOPX", "SFTPC", "SFTA3"]},

    {"id": "Q54", "text": "SFTPC SFTPD SFTA3 CD36 SLC34A2",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "SFTPD", "SFTA3", "CD36", "SLC34A2"]},

    {"id": "Q55", "text": "SFTPA1 SFTPA2 SFTPB SFTPC SFTPD",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPA1", "SFTPA2", "SFTPB", "SFTPC", "SFTPD"]},

    {"id": "Q56", "text": "ITCH UBE2N HRS VPS28 RABGEF1 EEA1",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["ITCH", "UBE2N", "HRS", "VPS28", "RABGEF1", "EEA1"]},

    {"id": "Q57", "text": "ABCA3 LAMP3 NAPSA CKAP4 ZDHHC2 CTSH",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["ABCA3", "LAMP3", "NAPSA", "CKAP4", "ZDHHC2", "CTSH"]},

    {"id": "Q58", "text": "CXCL1 CXCL2 CXCL3 CCL2 SFTPC",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["CXCL1", "CXCL2", "CXCL3", "CCL2", "SFTPC"]},

    {"id": "Q59", "text": "SFTPC ITCH EEA1 LAMP3 MICALL1 ABCA3",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "ITCH", "EEA1", "LAMP3", "MICALL1", "ABCA3"]},

    {"id": "Q60", "text": "CTNNB1 NKX2-1 SFTPC SFTPB LAMP3",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["CTNNB1", "NKX2-1", "SFTPC", "SFTPB", "LAMP3"]},

    {"id": "Q61", "text": "MKI67 PCNA TOP2A SFTPC NKX2-1",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["MKI67", "PCNA", "SFTPC", "NKX2-1"]},

    {"id": "Q62", "text": "HLA-DPA1 HLA-DPB1 HLA-DRA CXCL1 CCL2",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["HLA-DPA1", "HLA-DPB1", "HLA-DRA", "CXCL1", "CCL2"]},

    {"id": "Q63", "text": "NAPSA ABCA3 SFTA3 SFTPD LAMP3 HOPX",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["NAPSA", "ABCA3", "SFTA3", "SFTPD", "LAMP3", "HOPX"]},

    {"id": "Q64", "text": "UBE2I UBA2 PIAS1 ITCH RABGEF1",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["UBE2I", "UBA2", "PIAS1", "ITCH", "RABGEF1"]},

    {"id": "Q65", "text": "ITCH SFTPC LAMP3 ABCA3 UBE2N NAPSA",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell"],
     "expected_genes": ["ITCH", "SFTPC", "LAMP3", "ABCA3", "UBE2N", "NAPSA"]},

    # --- Respiratory tract epithelial cell (10 queries) ---

    {"id": "Q66", "text": "AQP5 CAV1 AGER HOPX",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AQP5", "CAV1", "AGER", "HOPX"]},

    {"id": "Q67", "text": "CAV1 AGER AQP5 PDPN",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["CAV1", "AGER", "AQP5"]},

    {"id": "Q68", "text": "SOX2 NKX2-1 HOPX CAV1",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["SOX2", "NKX2-1", "HOPX", "CAV1"]},

    {"id": "Q69", "text": "AGER AQP5 PDPN CLIC5 EMP2",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AGER", "AQP5"]},

    {"id": "Q70", "text": "CAV1 CAV2 AQP5 AGER HOPX",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["CAV1", "AQP5", "AGER", "HOPX"]},

    {"id": "Q71", "text": "SFTPC CAV1 AGER AQP5 HOPX NKX2-1",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell", "pulmonary alveolar type 2 cell"],
     "expected_genes": ["SFTPC", "CAV1", "AGER", "AQP5", "HOPX", "NKX2-1"]},

    {"id": "Q72", "text": "SOX9 NKX2-1 SFTPC HOPX CAV1",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell", "pulmonary alveolar type 2 cell"],
     "expected_genes": ["NKX2-1", "SFTPC", "HOPX", "CAV1"]},

    {"id": "Q73", "text": "AQP5 AGER PDPN RTKN2 GPRC5A",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["AQP5", "AGER"]},

    {"id": "Q74", "text": "EMP2 CLIC5 CAV1 AGER AQP5",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell"],
     "expected_genes": ["CAV1", "AGER", "AQP5"]},

    {"id": "Q75", "text": "SOX2 SOX9 NKX2-1 SFTPC TP63",
     "category": "expression",
     "expected_clusters": ["respiratory tract epithelial cell", "basal cell"],
     "expected_genes": ["SOX2", "NKX2-1", "SFTPC", "TP63"]},

    # --- Basal cell (8 queries) ---

    {"id": "Q76", "text": "TP63 KRT5 KRT14 KRT15 SOX2",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT14"]},

    {"id": "Q77", "text": "KRT5 KRT14 TP63 LAMB3 COL17A1",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["KRT5", "KRT14", "TP63"]},

    {"id": "Q78", "text": "TP63 KRT5 KRT14 KRT15 ITGA6 ITGB4",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT14", "KRT15"]},

    {"id": "Q79", "text": "KRT5 SOX2 TP63 NGFR DLK2",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["KRT5", "SOX2", "TP63"]},

    {"id": "Q80", "text": "LAMB3 COL17A1 KRT5 KRT14 TP63 KRT15",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["KRT5", "KRT14", "TP63", "KRT15"]},

    {"id": "Q81", "text": "KRT14 KRT5 ITGA6 ITGB4 TP63",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["KRT14", "KRT5", "TP63"]},

    {"id": "Q82", "text": "TP63 KRT5 S100A2 DST KRT15",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["TP63", "KRT5", "KRT15"]},

    {"id": "Q83", "text": "KRT14 KRT5 TP63 SOX2 DAPL1",
     "category": "expression",
     "expected_clusters": ["basal cell"],
     "expected_genes": ["KRT14", "KRT5", "TP63", "SOX2"]},

    # --- Pulmonary neuroendocrine cell (8 queries) ---

    {"id": "Q84", "text": "ASCL1 NEUROD1 GRP CHGA SYP CALCA",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["ASCL1", "NEUROD1", "GRP", "CHGA", "SYP"]},

    {"id": "Q85", "text": "GRP ASCL1 SYT1 CHGA SYP",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["GRP", "ASCL1", "CHGA", "SYP"]},

    {"id": "Q86", "text": "ASCL1 GRP CHGA SYP NEUROD1",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["ASCL1", "GRP", "CHGA", "SYP", "NEUROD1"]},

    {"id": "Q87", "text": "CHGA SYP CALCA GRP ASCL1 SCG5",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["CHGA", "SYP", "GRP", "ASCL1"]},

    {"id": "Q88", "text": "ASCL1 GRP SFTPC NKX2-1",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell", "pulmonary alveolar type 2 cell"],
     "expected_genes": ["ASCL1", "GRP", "SFTPC", "NKX2-1"]},

    {"id": "Q89", "text": "SYP SYT1 CHGA GRP CALCA PCSK1",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["SYP", "CHGA", "GRP"]},

    {"id": "Q90", "text": "NEUROD1 INSM1 ASCL1 GRP DDC",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["NEUROD1", "ASCL1", "GRP"]},

    {"id": "Q91", "text": "GRP CHGA ASCL1 NEUROD1 SCG2 SYP",
     "category": "expression",
     "expected_clusters": ["pulmonary neuroendocrine cell"],
     "expected_genes": ["GRP", "CHGA", "ASCL1", "NEUROD1", "SYP"]},

    # --- Lung multiciliated epithelial cell (7 queries) ---

    {"id": "Q92", "text": "FOXJ1 DNAH5 CAPS PIFO RSPH1",
     "category": "expression",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "DNAH5"]},

    {"id": "Q93", "text": "FOXJ1 DNAH5 DNAI1 RSPH1 CAPS",
     "category": "expression",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "DNAH5"]},

    {"id": "Q94", "text": "DNAH5 DNAI1 DNAI2 FOXJ1 RSPH1",
     "category": "expression",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["DNAH5", "FOXJ1"]},

    {"id": "Q95", "text": "FOXJ1 CAPS PIFO DNAH5 TPPP3",
     "category": "expression",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "CAPS", "DNAH5"]},

    {"id": "Q96", "text": "DNAH5 RSPH1 RSPH4A FOXJ1 CAPS",
     "category": "expression",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["DNAH5", "FOXJ1", "CAPS"]},

    {"id": "Q97", "text": "FOXJ1 DNAH5 CCDC39 CCDC40 CAPS",
     "category": "expression",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "DNAH5"]},

    {"id": "Q98", "text": "PIFO FOXJ1 DNAH5 CAPS TUBA1A",
     "category": "expression",
     "expected_clusters": ["lung multiciliated epithelial cell"],
     "expected_genes": ["FOXJ1", "DNAH5", "CAPS"]},

    # --- Multi-cluster expression queries (2 queries) ---

    {"id": "Q99", "text": "SFTPC CXCL1 MKI67 TP63 ASCL1 FOXJ1 AQP5",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell", "basal cell", "pulmonary neuroendocrine cell", "lung multiciliated epithelial cell", "respiratory tract epithelial cell"],
     "expected_genes": ["SFTPC", "CXCL1", "MKI67", "TP63", "ASCL1", "FOXJ1", "AQP5"]},

    {"id": "Q100", "text": "SFTPC TP63 ASCL1 FOXJ1 NKX2-1 CAV1",
     "category": "expression",
     "expected_clusters": ["pulmonary alveolar type 2 cell", "basal cell", "pulmonary neuroendocrine cell", "lung multiciliated epithelial cell", "respiratory tract epithelial cell"],
     "expected_genes": ["SFTPC", "TP63", "ASCL1", "FOXJ1", "NKX2-1", "CAV1"]},
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
# CELLWHISPERER SCORER — real CLIP embeddings
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
        """Rank cell types by cosine similarity to query."""
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
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    # ── Load ELISA results ──
    print(f"\n[ELISA] Loading results from {elisa_json_path}")
    with open(elisa_json_path) as f:
        elisa_all = json.load(f)

    paper = elisa_all[paper_id]
    elisa_detail = paper["retrieval_detail"]
    elisa_summary = paper["retrieval_summary"]
    elisa_analytical = paper.get("analytical", {})

    # ── Run CellWhisperer on all 100 queries ──
    print("\n" + "=" * 70)
    print("[CW] Running CellWhisperer on 100 queries (DT3)...")
    print("=" * 70)

    # DT3 uses recall@1,2,3,5 (5 clusters)
    RECALL_KS = [1, 2, 3, 5]

    cw_results = {"ontology": [], "expression": []}
    for q in QUERIES:
        ranked = cw_scorer.score_query(q["text"], top_k=5)
        entry = {
            "query_id": q["id"],
            "query_text": q["text"],
            "expected": q["expected_clusters"],
            "retrieved_top5": ranked[:5],
            "mrr": round(mrr(q["expected_clusters"], ranked), 4),
        }
        for k in RECALL_KS:
            entry[f"recall@{k}"] = round(
                cluster_recall_at_k(q["expected_clusters"], ranked, k), 4
            )
        cw_results[q["category"]].append(entry)
        print(f"  [{q['id']}] R@1={entry['recall@1']:.2f}  MRR={entry['mrr']:.2f}  Top3={ranked[:3]}")

    cw_agg = {}
    for cat in ["ontology", "expression"]:
        e = cw_results[cat]
        cw_agg[cat] = {
            "mean_mrr": round(np.mean([x["mrr"] for x in e]), 4),
            "mean_gene_recall": 0.0,  # CW has no gene delivery
        }
        for k in RECALL_KS:
            cw_agg[cat][f"mean_recall@{k}"] = round(
                np.mean([x[f"recall@{k}"] for x in e]), 4
            )

    # ── Unified mode list: Random, CW real, Semantic, scGPT, Union ──
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
    print("ELISA vs CellWhisperer — DT3: fdAT2 Organoids (100 queries)")
    print("=" * 90)
    print(f"\n{'Category':<14} {'Mode':<16} {'R@1':>7} {'R@2':>7} {'R@3':>7} "
          f"{'R@5':>7} {'MRR':>7} {'GeneR':>7}")
    print("-" * 80)
    for cat in ["ontology", "expression"]:
        for mode in ALL_MODES:
            r1 = gm(cat, mode, "mean_recall@1")
            r2 = gm(cat, mode, "mean_recall@2")
            r3 = gm(cat, mode, "mean_recall@3")
            r5 = gm(cat, mode, "mean_recall@5")
            mmr = gm(cat, mode, "mean_mrr")
            gr = gm(cat, mode, "mean_gene_recall")
            gr_s = "  N/A" if mode == "cellwhisperer_real" else f"{gr:.3f}"
            print(f"{cat:<14} {MODE_LABELS.get(mode, mode):<16} "
                  f"{r1:>7.3f} {r2:>7.3f} {r3:>7.3f} "
                  f"{r5:>7.3f} {mmr:>7.3f} {gr_s:>7}")
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

    # ============================================================
    # FIGURES
    # ============================================================

    cats = ["ontology", "expression"]
    cat_titles = [
        "Ontology Queries\n(concept-level)",
        "Expression Queries\n(gene-signature)",
    ]

    # ── Fig 1: Recall@1 bars (primary metric for 5 clusters) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES))
        vals = [gm(cat, m, "mean_recall@1") for m in ALL_MODES]
        bars = ax.bar(x, vals,
                      color=[MODE_COLORS[m] for m in ALL_MODES],
                      alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES],
                           fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean Cluster Recall@1")
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

    plt.suptitle("ELISA vs CellWhisperer — DT3: fdAT2 Organoids (100 queries)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_recall1.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig1_recall1.pdf"), bbox_inches="tight")
    plt.close()
    print(f"\n[FIG] fig1_recall1.png")

    # ── Fig 2: All metrics — R@1, R@2, MRR ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    metric_keys = ["mean_recall@1", "mean_recall@2", "mean_mrr"]
    metric_labels = ["Recall@1", "Recall@2", "MRR"]

    for ax, mk, mt in zip(axes, metric_keys, metric_labels):
        x = np.arange(len(ALL_MODES))
        w = 0.35
        ont = [gm("ontology", m, mk) for m in ALL_MODES]
        exp = [gm("expression", m, mk) for m in ALL_MODES]

        ax.bar(x - w / 2, ont, w, label="Ontology", alpha=0.85,
               color=[MODE_COLORS[m] for m in ALL_MODES], edgecolor="white")
        ax.bar(x + w / 2, exp, w, label="Expression", alpha=0.45,
               color=[MODE_COLORS[m] for m in ALL_MODES],
               edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES],
                           fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(mt, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_all_metrics.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_all_metrics.pdf"), bbox_inches="tight")
    plt.close()
    print("[FIG] fig2_all_metrics.png")

    # ── Fig 3: Cluster Recall vs Gene Recall ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES))
        w = 0.35
        cluster_vals = [gm(cat, m, "mean_recall@1") for m in ALL_MODES]
        gene_vals = [gm(cat, m, "mean_gene_recall") for m in ALL_MODES]

        bars1 = ax.bar(x - w / 2, cluster_vals, w, label="Cluster Recall@1",
                        color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.85,
                        edgecolor="white")
        bars2 = ax.bar(x + w / 2, gene_vals, w, label="Gene Recall",
                        color=[MODE_COLORS[m] for m in ALL_MODES], alpha=0.45,
                        edgecolor="black", linewidth=0.8, hatch="///")

        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars1, cluster_vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars2, gene_vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    legend_elements = [
        Patch(facecolor="#888888", alpha=0.85, label="Cluster Recall@1"),
        Patch(facecolor="#888888", alpha=0.45, hatch="///",
              edgecolor="black", label="Gene Recall"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.pdf"), bbox_inches="tight")
    plt.close()
    print("[FIG] fig3_cluster_vs_gene.png")

    # ── Fig 4: Radar — ELISA modalities vs CW ──
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    rl = ["Ont R@1", "Ont R@2", "Ont MRR", "Exp R@1", "Exp R@2", "Exp MRR"]
    N = len(rl)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for mode, color, ls in [
        ("cellwhisperer_real", "#E91E63", "--"),
        ("semantic", "#2196F3", "-"),
        ("scgpt", "#FF9800", "-"),
        ("union", "#4CAF50", "-"),
    ]:
        v = []
        for cat in ["ontology", "expression"]:
            for mk in ["mean_recall@1", "mean_recall@2", "mean_mrr"]:
                v.append(gm(cat, mode, mk))
        v += v[:1]
        ax.plot(angles, v, linewidth=2, linestyle=ls,
                label=MODE_LABELS.get(mode, mode), color=color)
        ax.fill(angles, v, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rl, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("ELISA vs CellWhisperer\n(DT3: fdAT2 Organoids)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_radar.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig4_radar.pdf"), bbox_inches="tight")
    plt.close()
    print("[FIG] fig4_radar.png")

    # ── Fig 5: Gene recall + Analytical modules ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 5a: Gene recall (CW = N/A shown as 0)
    x = np.arange(len(ALL_MODES))
    w = 0.35
    og = [gm("ontology", m, "mean_gene_recall") for m in ALL_MODES]
    eg = [gm("expression", m, "mean_gene_recall") for m in ALL_MODES]
    clrs = [MODE_COLORS[m] for m in ALL_MODES]
    ax1.bar(x - w / 2, og, w, label="Ontology", alpha=0.85,
            color=clrs, edgecolor="white")
    ax1.bar(x + w / 2, eg, w, label="Expression", alpha=0.45,
            color=clrs, edgecolor="black", linewidth=0.5, hatch="//")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODE_LABELS[m] for m in ALL_MODES],
                        fontsize=9, rotation=15, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Gene Recall")
    ax1.set_title("Gene-Level Evidence Delivery", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Add "N/A" label on CW bars
    cw_idx = ALL_MODES.index("cellwhisperer_real")
    ax1.text(cw_idx, 0.05, "N/A", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#E91E63")

    # 5b: Analytical modules radar
    ax2.set_axis_off()
    ax2b = fig.add_axes([0.55, 0.1, 0.4, 0.75], polar=True)
    al = ["Pathways", "Interactions\n(LR)", "Proportions", "Compare"]
    av = [
        ana.get("pathways", {}).get("alignment", 0) / 100,
        ana.get("interactions", {}).get("lr_recovery_rate", 0) / 100,
        ana.get("proportions", {}).get("consistency_rate", 0) / 100,
        ana.get("compare", {}).get("compare_recall", 0) / 100,
    ]
    aa = np.linspace(0, 2 * np.pi, len(al), endpoint=False).tolist()
    av_c = av + av[:1]
    aa_c = aa + aa[:1]
    ax2b.fill(aa_c, av_c, alpha=0.25, color="#4CAF50")
    ax2b.plot(aa_c, av_c, "o-", color="#4CAF50", linewidth=2, label="ELISA")
    ax2b.plot(aa_c, [0] * len(aa_c), "--", color="#E91E63", linewidth=1.5,
              label="CellWhisp. (N/A)")
    ax2b.set_xticks(aa)
    ax2b.set_xticklabels(al, fontsize=9)
    ax2b.set_ylim(0, 1.05)
    ax2b.set_title("Analytical Modules\n(ELISA only)",
                   fontsize=11, fontweight="bold", pad=15)
    ax2b.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.pdf"),
                bbox_inches="tight")
    plt.close()
    print("[FIG] fig5_gene_analytical.png")

    # ── Fig 6: Per-query comparison (Union vs CW) — top 20 per category ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for ax, cat in zip(axes, ["ontology", "expression"]):
        cw_e = cw_results[cat]
        qids = [e["query_id"] for e in cw_e]
        cw_r1 = [e["recall@1"] for e in cw_e]
        el_union = elisa_detail.get(cat, {}).get("union", [])
        el_r1 = [e.get("recall@1", 0) for e in el_union] if el_union else [0] * len(qids)

        # Show first 25 queries
        n = min(25, len(qids), len(el_r1))
        y = np.arange(n)
        h = 0.35
        ax.barh(y - h / 2, el_r1[:n], h, label="ELISA Union",
                color="#4CAF50", alpha=0.8)
        ax.barh(y + h / 2, cw_r1[:n], h, label="CellWhisperer",
                color="#E91E63", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(qids[:n], fontsize=8)
        ax.set_xlabel("Recall@1")
        ax.set_xlim(0, 1.1)
        ax.set_title(f"{cat.capitalize()} Queries (first 25)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    plt.suptitle("Per-Query: ELISA Union vs CellWhisperer (DT3)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_perquery.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig6_perquery.pdf"),
                bbox_inches="tight")
    plt.close()
    print("[FIG] fig6_perquery.png")

    # ── Save JSON ──
    output = {
        "cellwhisperer": {"results": cw_results, "aggregated": cw_agg},
        "elisa_summary": elisa_summary,
        "elisa_analytical": elisa_analytical,
        "comparison": {
            cat: {mode: {mk: gm(cat, mode, mk)
                         for mk in [f"mean_recall@{k}" for k in RECALL_KS] + ["mean_mrr"]}
                  for mode in ALL_MODES}
            for cat in ["ontology", "expression"]
        },
        "modes": ALL_MODES,
        "n_queries": len(QUERIES),
        "dataset": "DT3_HumanFetalLung",
        "paper": "Lim et al. EMBO J 2025",
        "n_clusters": 5,
        "cluster_names": [
            "basal cell",
            "lung multiciliated epithelial cell",
            "pulmonary alveolar type 2 cell",
            "pulmonary neuroendocrine cell",
            "respiratory tract epithelial cell",
        ],
        "timestamp": datetime.now().isoformat(),
    }
    rp = os.path.join(out_dir, "elisa_vs_cellwhisperer_DT3_results.json")
    with open(rp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")


def main():
    parser = argparse.ArgumentParser(
        description="ELISA vs CellWhisperer — DT3 Human Fetal Lung"
    )
    parser.add_argument("--elisa-results", required=True,
                        help="ELISA benchmark_v5_results.json from DT3")
    parser.add_argument("--cw-npz", required=True,
                        help="CellWhisperer full_output.npz")
    parser.add_argument("--cw-leiden", required=True,
                        help="CellWhisperer leiden_umap_embeddings.h5ad")
    parser.add_argument("--cw-ckpt", required=True,
                        help="cellwhisperer_clip_v1.ckpt")
    parser.add_argument("--cf-h5ad", required=True,
                        help="Original read_count_table.h5ad")
    parser.add_argument("--paper", default="DT3")
    parser.add_argument("--out", default="comparison_DT3/")
    args = parser.parse_args()

    cw = CellWhispererScorer(args.cw_npz, args.cw_leiden, args.cw_ckpt, args.cf_h5ad)
    run_comparison(cw, args.elisa_results, args.paper, args.out)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
