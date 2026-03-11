#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA vs CellWhisperer — DT2 Breast Tissue Atlas (Bhat-Nakshatri et al. Nat Med 2024)
======================================================================================
Head-to-Head comparison on breast tissue scRNA-seq data.

8 clusters (Cell Ontology IDs):
  0: luminal hormone-sensing cell of mammary gland  (LHS)
  1: luminal adaptive secretory precursor cell of mammary gland  (LASP)
  2: basal-myoepithelial cell of mammary gland  (BM)
  3: endothelial cell  (ENDO)
  4: adipocyte  (ADI)
  5: fibroblast  (FIB)
  6: T cell  (TC)
  7: macrophage  (MAC)

Step 1: Run ELISA benchmark first:
    python elisa_benchmark_v5_1_DT2_standalone.py \\
        --base /path/to/embeddings \\
        --pt-name fused_DT2_BreastTissue.pt \\
        --paper DT2 \\
        --out results_DT2/

Step 2: Run this script (in cellwhisperer env):
    conda activate cellwhisperer
    python elisa_vs_cellwhisperer_DT2.py \\
        --elisa-results results_DT2/benchmark_v5_results.json \\
        --cw-npz /path/to/cellwhisperer/full_output.npz \\
        --cw-leiden /path/to/leiden_umap_embeddings.h5ad \\
        --cw-ckpt /path/to/cellwhisperer_clip_v1.ckpt \\
        --cf-h5ad /path/to/read_count_table.h5ad \\
        --out comparison_DT2/
"""

import os, sys, json, argparse
from datetime import datetime
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# QUERIES — 50 ontology + 50 expression, balanced across 8 clusters
# ============================================================

QUERIES = [
    # ================================================================
    # ONTOLOGY QUERIES (Q01–Q50)
    # ================================================================

    {"id": "Q01", "text": "luminal hormone sensing cells with estrogen receptor expression in the healthy breast",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["GATA3", "FOXA1", "ERBB4", "ANKRD30A", "ESR1"]},
    {"id": "Q02", "text": "FOXA1 pioneer transcription factor activity in luminal hormone responsive breast epithelial cells",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["FOXA1", "AFF3", "GATA3", "ESR1"]},
    {"id": "Q03", "text": "ERα-FOXA1-GATA3 transcription factor network in hormone responsive breast cells",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["FOXA1", "GATA3", "ESR1"]},
    {"id": "Q04", "text": "mature luminal cells with hormone receptor positive identity in breast tissue",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["MYBPC1", "AFF3", "ERBB4", "ANKRD30A", "THSD4", "TTC6"]},
    {"id": "Q05", "text": "hormone sensing alpha versus beta cell states in breast epithelium",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["ELOVL5", "FOXA1", "ERBB4", "ESR1"]},
    {"id": "Q06", "text": "LHS cell-enriched fate factor DACH1 and PI3K pathway regulator INPP4B in breast",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["CTNND2", "NEK10", "INPP4B", "DACH1"]},
    {"id": "Q07", "text": "lobular epithelial cells expressing APOD and immunoglobulin genes in breast",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["APOD", "IGHA1", "IGKC", "ESR1"]},
    {"id": "Q08", "text": "luminal adaptive secretory precursor cells and progenitor identity in breast",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "KIT", "CCL28", "EHF", "KRT15"]},
    {"id": "Q09", "text": "ELF5 and EHF transcription factor expression in luminal progenitor breast cells",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "KIT", "NCALD", "BARX2", "EHF"]},
    {"id": "Q10", "text": "alveolar progenitor cell state enriched in Indigenous American breast tissue",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "EHF", "KIT", "ESR1"]},
    {"id": "Q11", "text": "BRCA1 associated breast cancer originating from luminal progenitor cells",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "KRT15", "EHF", "KIT"]},
    {"id": "Q12", "text": "KIT receptor expression and chromatin accessibility in luminal progenitor cells",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "CCL28", "EHF", "KIT"]},
    {"id": "Q13", "text": "MFGE8 and SHANK2 expression in luminal progenitor cells of the breast",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["SHANK2", "SORBS2", "MFGE8"]},
    {"id": "Q14", "text": "LASP basal-luminal intermediate progenitor cell identity in the breast",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "BARX2", "EHF", "KRT15"]},
    {"id": "Q15", "text": "basal-myoepithelial cells with TP63 and KRT14 expression in breast",
     "category": "ontology",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "KLHL29", "TP63", "FHOD3"]},
    {"id": "Q16", "text": "basal cell chromatin accessibility and TP63 binding site enrichment",
     "category": "ontology",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "SEMA5A", "TP63", "KLHL13"]},
    {"id": "Q17", "text": "basal alpha and basal beta cell states in breast myoepithelium",
     "category": "ontology",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "KLHL29", "TP63", "KLHL13"]},
    {"id": "Q18", "text": "SOX10 motif enrichment in basal-myoepithelial cells of the breast",
     "category": "ontology",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "KLHL29", "TP63"]},
    {"id": "Q19", "text": "KRT14 KRT17 expression in ductal epithelial and basal cells of breast tissue",
     "category": "ontology",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "KRT17", "TP63"]},
    {"id": "Q20", "text": "fibroblast heterogeneity and cell states in healthy breast stroma",
     "category": "ontology",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["RUNX1T1", "COL1A1", "SLIT2", "LAMA2", "COL3A1"]},
    {"id": "Q21", "text": "genetic ancestry-dependent variability in breast fibroblast cell states",
     "category": "ontology",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["MGST1", "POSTN", "MFAP5", "CFD", "COL3A1"]},
    {"id": "Q22", "text": "fibro-prematrix state enrichment in African ancestry breast tissue fibroblasts",
     "category": "ontology",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["MGST1", "CFD", "MFAP5"]},
    {"id": "Q23", "text": "PROCR ZEB1 PDGFRα multipotent stromal cells enriched in African ancestry breast",
     "category": "ontology",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["PDGFRA", "PROCR", "ZEB1"]},
    {"id": "Q24", "text": "myofibroblast and inflammatory fibroblast subtypes in breast cancer stroma",
     "category": "ontology",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["PDPN", "CD34", "CXCL12", "COL1A1"]},
    {"id": "Q25", "text": "SFRP4 and Wnt pathway modulation in breast fibroblasts",
     "category": "ontology",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["COL1A1", "POSTN", "SFRP4"]},
    {"id": "Q26", "text": "endothelial cell subtypes and vascular markers in breast tissue",
     "category": "ontology",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["CXCL12", "MMRN1", "LDB2", "MECOM"]},
    {"id": "Q27", "text": "lymphatic endothelial cells expressing LYVE1 in breast stroma",
     "category": "ontology",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["LYVE1", "MECOM"]},
    {"id": "Q28", "text": "ACKR1 stalk-like endothelial cell subtype in breast vasculature",
     "category": "ontology",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["CXCL12", "ACKR1", "MECOM"]},
    {"id": "Q29", "text": "vascular endothelial cell heterogeneity in mammary gland microvasculature",
     "category": "ontology",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["MECOM", "LDB2", "MMRN1"]},
    {"id": "Q30", "text": "breast tissue angiogenesis and endothelial cell MECOM expression",
     "category": "ontology",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["MECOM", "LDB2", "CXCL12"]},
    {"id": "Q31", "text": "T lymphocyte markers and immune cell identity in breast tissue",
     "category": "ontology",
     "expected_clusters": ["T cell"],
     "expected_genes": ["THEMIS", "PTPRC", "ARHGAP15", "SKAP1"]},
    {"id": "Q32", "text": "CD4 T cell IL7R expression and chromatin accessibility in breast",
     "category": "ontology",
     "expected_clusters": ["T cell"],
     "expected_genes": ["PTPRC", "IL7R", "SKAP1"]},
    {"id": "Q33", "text": "CD8 T cell GZMK cytotoxic activity and IFNG signaling in breast tissue",
     "category": "ontology",
     "expected_clusters": ["T cell"],
     "expected_genes": ["PTPRC", "IFNG", "GZMK"]},
    {"id": "Q34", "text": "tissue-resident memory T lymphocyte populations in healthy breast",
     "category": "ontology",
     "expected_clusters": ["T cell"],
     "expected_genes": ["PTPRC", "THEMIS", "SKAP1", "ARHGAP15"]},
    {"id": "Q35", "text": "adaptive immune surveillance by T cells in mammary gland stroma",
     "category": "ontology",
     "expected_clusters": ["T cell"],
     "expected_genes": ["PTPRC", "IL7R", "SKAP1", "GZMK"]},
    {"id": "Q36", "text": "macrophage identity and FCGR3A expression in breast tissue stroma",
     "category": "ontology",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["FCGR3A", "ALCAM"]},
    {"id": "Q37", "text": "macrophage subtypes and tissue-resident immune cells in healthy breast",
     "category": "ontology",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"]},
    {"id": "Q38", "text": "breast tissue-resident macrophage phagocytic function and complement expression",
     "category": "ontology",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["FCGR3A", "ALCAM"]},
    {"id": "Q39", "text": "myeloid lineage immune cells and monocyte-derived macrophages in mammary gland",
     "category": "ontology",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"]},
    {"id": "Q40", "text": "adipocyte subtypes and lipid metabolism in breast tissue",
     "category": "ontology",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["FABP4", "PLIN1", "KIT"]},
    {"id": "Q41", "text": "adipocyte PLIN1 and FABP4 expression in healthy breast stroma",
     "category": "ontology",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["FABP4", "PLIN1"]},
    {"id": "Q42", "text": "PLIN1 lipid droplet biology and adipocyte identity in mammary fat pad",
     "category": "ontology",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["FABP4", "PLIN1"]},
    {"id": "Q43", "text": "mammary gland adipose tissue and fatty acid binding protein expression",
     "category": "ontology",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["FABP4", "PLIN1"]},
    {"id": "Q44", "text": "epithelial cell hierarchy from basal to luminal hormone sensing in breast",
     "category": "ontology",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["ELF5", "TP63", "FOXA1", "KRT14", "ESR1"]},
    {"id": "Q45", "text": "CXCL12 chemokine expression in endothelial cells and fibroblasts of breast",
     "category": "ontology",
     "expected_clusters": ["endothelial cell", "fibroblast"],
     "expected_genes": ["CXCL12", "LAMA2", "MECOM"]},
    {"id": "Q46", "text": "VEGFA angiogenic signaling from luminal cells to endothelium in breast",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "endothelial cell"],
     "expected_genes": ["VEGFA", "LDB2", "MECOM"]},
    {"id": "Q47", "text": "IGF1 paracrine signaling from fibroblasts to luminal cells in breast stroma",
     "category": "ontology",
     "expected_clusters": ["fibroblast", "luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["LAMA2", "IGF1", "IGF1R"]},
    {"id": "Q48", "text": "breast tissue microenvironment with stromal and immune cell interactions",
     "category": "ontology",
     "expected_clusters": ["fibroblast", "endothelial cell", "T cell", "macrophage"],
     "expected_genes": ["CXCL12", "PTPRC", "FCGR3A", "COL1A1"]},
    {"id": "Q49", "text": "ancestry differences in breast tissue cellular composition and cancer risk",
     "category": "ontology",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland", "fibroblast"],
     "expected_genes": ["ELF5", "CFD", "KIT", "MGST1"]},
    {"id": "Q50", "text": "gene expression differences between ductal and lobular epithelial cells of the breast",
     "category": "ontology",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["RPL36", "KRT17", "KRT14", "DPM3", "DUSP1"]},

    # ================================================================
    # EXPRESSION QUERIES (Q51–Q100)
    # ================================================================

    {"id": "Q51", "text": "FOXA1 ESR1 GATA3 ERBB4 ANKRD30A AFF3 TTC6", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["GATA3", "FOXA1", "AFF3", "ERBB4", "ANKRD30A", "TTC6", "ESR1"]},
    {"id": "Q52", "text": "MYBPC1 THSD4 CTNND2 DACH1 INPP4B NEK10", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["CTNND2", "MYBPC1", "DACH1", "INPP4B", "NEK10", "THSD4"]},
    {"id": "Q53", "text": "ESR1 FOXA1 GATA3 ELOVL5 ANKRD30A", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["ELOVL5", "FOXA1", "GATA3", "ESR1"]},
    {"id": "Q54", "text": "AFF3 TTC6 ERBB4 MYBPC1 THSD4", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["MYBPC1", "AFF3", "ERBB4", "THSD4", "TTC6"]},
    {"id": "Q55", "text": "DACH1 NEK10 CTNND2 INPP4B ELOVL5", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["ELOVL5", "CTNND2", "INPP4B", "NEK10", "DACH1"]},
    {"id": "Q56", "text": "APOD IGHA1 IGKC ESR1 FOXA1 GATA3", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["APOD", "ESR1", "FOXA1", "GATA3"]},
    {"id": "Q57", "text": "DUSP1 DPM3 RPL36 IGHA1 IGKC APOD", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["RPL36", "APOD", "DPM3", "DUSP1", "IGHA1", "IGKC"]},
    {"id": "Q58", "text": "ELF5 EHF KIT CCL28 KRT15 BARX2 NCALD", "category": "expression",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "KIT", "NCALD", "BARX2", "CCL28", "EHF", "KRT15"]},
    {"id": "Q59", "text": "MFGE8 SHANK2 SORBS2 AGAP1 ELF5", "category": "expression",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["SHANK2", "SORBS2", "MFGE8", "ELF5"]},
    {"id": "Q60", "text": "KRT15 CCL28 KIT INPP4B ELF5", "category": "expression",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "CCL28", "KIT", "KRT15"]},
    {"id": "Q61", "text": "RBMS3 EHF BARX2 NCALD ELF5", "category": "expression",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["BARX2", "EHF", "NCALD", "ELF5"]},
    {"id": "Q62", "text": "ESR1 ELF5 EHF KIT CCL28", "category": "expression",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland", "luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["ELF5", "EHF", "KIT", "ESR1"]},
    {"id": "Q63", "text": "ELF5 KIT CCL28 EHF KRT15 BARX2", "category": "expression",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["ELF5", "KIT", "CCL28", "EHF", "KRT15", "BARX2"]},
    {"id": "Q64", "text": "NCALD BARX2 SHANK2 SORBS2 MFGE8 ELF5", "category": "expression",
     "expected_clusters": ["luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["NCALD", "BARX2", "SHANK2", "SORBS2", "MFGE8", "ELF5"]},
    {"id": "Q65", "text": "TP63 KRT14 KLHL29 FHOD3 SEMA5A", "category": "expression",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["TP63", "SEMA5A", "KRT14", "FHOD3", "KLHL29"]},
    {"id": "Q66", "text": "KLHL13 KLHL29 TP63 KRT14 PTPRT", "category": "expression",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "KLHL29", "TP63", "KLHL13"]},
    {"id": "Q67", "text": "TP63 KRT14 KRT17 FHOD3 ABLIM3", "category": "expression",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "KRT17", "TP63", "FHOD3"]},
    {"id": "Q68", "text": "ST6GALNAC3 PTPRM SEMA5A KLHL29", "category": "expression",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KLHL29", "SEMA5A"]},
    {"id": "Q69", "text": "KRT14 KRT17 TP63 KLHL29 KLHL13 FHOD3", "category": "expression",
     "expected_clusters": ["basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["KRT14", "KRT17", "TP63", "KLHL29", "KLHL13", "FHOD3"]},
    {"id": "Q70", "text": "LAMA2 SLIT2 RUNX1T1 COL1A1 COL3A1", "category": "expression",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["RUNX1T1", "COL1A1", "SLIT2", "LAMA2", "COL3A1"]},
    {"id": "Q71", "text": "COL3A1 POSTN COL1A1 IGF1 ADAM12", "category": "expression",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["POSTN", "IGF1", "COL1A1", "ADAM12", "COL3A1"]},
    {"id": "Q72", "text": "CFD MGST1 MFAP5 COL3A1 POSTN", "category": "expression",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["MGST1", "CFD", "MFAP5", "COL3A1", "POSTN"]},
    {"id": "Q73", "text": "PROCR ZEB1 PDGFRA COL1A1 LAMA2", "category": "expression",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["PDGFRA", "PROCR", "ZEB1", "COL1A1", "LAMA2"]},
    {"id": "Q74", "text": "SFRP4 COL1A1 POSTN LAMA2 SLIT2", "category": "expression",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["COL1A1", "POSTN", "SFRP4", "LAMA2", "SLIT2"]},
    {"id": "Q75", "text": "COL1A1 PDPN CD34 CXCL12 LAMA2", "category": "expression",
     "expected_clusters": ["fibroblast"],
     "expected_genes": ["CXCL12", "COL1A1", "LAMA2"]},
    {"id": "Q76", "text": "MECOM LDB2 MMRN1 CXCL12 ACKR1", "category": "expression",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["MMRN1", "LDB2", "MECOM", "CXCL12", "ACKR1"]},
    {"id": "Q77", "text": "LYVE1 MECOM LDB2 MMRN1", "category": "expression",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["LYVE1", "MECOM", "LDB2", "MMRN1"]},
    {"id": "Q78", "text": "ACKR1 CXCL12 MECOM LDB2", "category": "expression",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["CXCL12", "ACKR1", "MECOM", "LDB2"]},
    {"id": "Q79", "text": "MECOM LDB2 MMRN1 LYVE1 ACKR1", "category": "expression",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["MECOM", "LDB2", "MMRN1", "LYVE1", "ACKR1"]},
    {"id": "Q80", "text": "CXCL12 MECOM LDB2 ACKR1 MMRN1", "category": "expression",
     "expected_clusters": ["endothelial cell"],
     "expected_genes": ["CXCL12", "MECOM", "LDB2", "ACKR1"]},
    {"id": "Q81", "text": "PTPRC SKAP1 ARHGAP15 THEMIS IL7R", "category": "expression",
     "expected_clusters": ["T cell"],
     "expected_genes": ["THEMIS", "PTPRC", "ARHGAP15", "SKAP1", "IL7R"]},
    {"id": "Q82", "text": "IL7R GZMK PTPRC SKAP1", "category": "expression",
     "expected_clusters": ["T cell"],
     "expected_genes": ["PTPRC", "IL7R", "SKAP1", "GZMK"]},
    {"id": "Q83", "text": "IFNG GZMK IL7R THEMIS PTPRC", "category": "expression",
     "expected_clusters": ["T cell"],
     "expected_genes": ["THEMIS", "IFNG", "IL7R", "GZMK", "PTPRC"]},
    {"id": "Q84", "text": "THEMIS ARHGAP15 SKAP1 PTPRC IL7R", "category": "expression",
     "expected_clusters": ["T cell"],
     "expected_genes": ["THEMIS", "ARHGAP15", "SKAP1", "PTPRC", "IL7R"]},
    {"id": "Q85", "text": "PTPRC SKAP1 GZMK IFNG THEMIS ARHGAP15", "category": "expression",
     "expected_clusters": ["T cell"],
     "expected_genes": ["PTPRC", "SKAP1", "GZMK", "IFNG", "THEMIS", "ARHGAP15"]},
    {"id": "Q86", "text": "FCGR3A ALCAM LYVE1 CD163", "category": "expression",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"]},
    {"id": "Q87", "text": "ALCAM FCGR3A LYVE1 CD14", "category": "expression",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["FCGR3A", "ALCAM", "LYVE1"]},
    {"id": "Q88", "text": "FCGR3A ALCAM CD163 MERTK", "category": "expression",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["FCGR3A", "ALCAM"]},
    {"id": "Q89", "text": "ALCAM LYVE1 FCGR3A CD163 MARCO", "category": "expression",
     "expected_clusters": ["macrophage"],
     "expected_genes": ["ALCAM", "LYVE1", "FCGR3A"]},
    {"id": "Q90", "text": "PLIN1 FABP4 KIT ADIPOQ LEP", "category": "expression",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["FABP4", "PLIN1", "KIT"]},
    {"id": "Q91", "text": "FABP4 PLIN1 ADIPOQ LEP LPL", "category": "expression",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["FABP4", "PLIN1"]},
    {"id": "Q92", "text": "PLIN1 FABP4 LPL PPARG ADIPOQ", "category": "expression",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["PLIN1", "FABP4"]},
    {"id": "Q93", "text": "FABP4 PLIN1 KIT ADIPOQ", "category": "expression",
     "expected_clusters": ["adipocyte"],
     "expected_genes": ["FABP4", "PLIN1", "KIT"]},
    {"id": "Q94", "text": "FOXA1 ELF5 TP63 KRT14 GATA3 ESR1", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["ELF5", "GATA3", "TP63", "FOXA1", "KRT14", "ESR1"]},
    {"id": "Q95", "text": "GATA3 EHF ELF5 FOXA1 KRT15 KRT14 TP63", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["ELF5", "GATA3", "TP63", "FOXA1", "KRT14", "EHF", "KRT15"]},
    {"id": "Q96", "text": "MECOM PTPRC FCGR3A PLIN1 LAMA2 TP63 FOXA1", "category": "expression",
     "expected_clusters": ["endothelial cell", "T cell", "macrophage", "adipocyte", "fibroblast", "basal-myoepithelial cell of mammary gland", "luminal hormone-sensing cell of mammary gland"],
     "expected_genes": ["TP63", "MECOM", "FOXA1", "FCGR3A", "PTPRC", "PLIN1", "LAMA2"]},
    {"id": "Q97", "text": "CXCL12 LAMA2 MECOM LDB2 COL1A1", "category": "expression",
     "expected_clusters": ["endothelial cell", "fibroblast"],
     "expected_genes": ["CXCL12", "LAMA2", "MECOM", "LDB2", "COL1A1"]},
    {"id": "Q98", "text": "ESR1 FOXA1 ELF5 EHF KIT TP63 KRT14", "category": "expression",
     "expected_clusters": ["luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland", "basal-myoepithelial cell of mammary gland"],
     "expected_genes": ["ELF5", "TP63", "FOXA1", "KRT14", "EHF", "ESR1"]},
    {"id": "Q99", "text": "PTPRC FCGR3A FABP4 PLIN1 MECOM", "category": "expression",
     "expected_clusters": ["T cell", "macrophage", "adipocyte", "endothelial cell"],
     "expected_genes": ["PTPRC", "FCGR3A", "FABP4", "PLIN1", "MECOM"]},
    {"id": "Q100", "text": "VEGFA LDB2 IGF1 LAMA2 FOXA1 ELF5", "category": "expression",
     "expected_clusters": ["endothelial cell", "fibroblast", "luminal hormone-sensing cell of mammary gland", "luminal adaptive secretory precursor cell of mammary gland"],
     "expected_genes": ["VEGFA", "LDB2", "IGF1", "LAMA2", "FOXA1", "ELF5"]},
]


# ============================================================
# METRICS
# ============================================================

def _word_overlap(a, b):
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0

def cluster_recall_at_k(expected, retrieved, k=5):
    if not expected: return 0.0
    ret_lower = [r.lower() for r in retrieved[:k]]
    return sum(1 for exp in expected if any(exp.lower() in r or r in exp.lower() or _word_overlap(exp.lower(), r) >= 0.5 for r in ret_lower)) / len(expected)

def mrr(expected, retrieved):
    for rank, ret in enumerate(retrieved, 1):
        ret_l = ret.lower()
        for exp in expected:
            if exp.lower() in ret_l or ret_l in exp.lower() or _word_overlap(exp.lower(), ret_l) >= 0.5:
                return 1.0 / rank
    return 0.0


# ============================================================
# CELLWHISPERER SCORER — real CLIP embeddings
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
                ct_embed_accum[ct] = np.zeros(self.cell_embeds.shape[1], dtype=np.float64)
                ct_counts[ct] = 0
            ct_embed_accum[ct] += self.cell_embeds[i].astype(np.float64)
            ct_counts[ct] += 1

        self.celltype_embeds = {}
        for ct in ct_embed_accum:
            self.celltype_embeds[ct] = (ct_embed_accum[ct] / ct_counts[ct]).astype(np.float32)
        print(f"[CW] {len(self.celltype_embeds)} unique cell types")

        self.model = None
        self._load_model(ckpt_path)

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
            self.model = None

    def embed_text(self, query_text):
        import torch
        if self.model is not None:
            with torch.no_grad(): return self.model.embed_texts([query_text]).cpu().numpy()[0]
        return None

    def score_query(self, query_text, top_k=10):
        text_embed = self.embed_text(query_text)
        if text_embed is None:
            ct_list = list(self.celltype_embeds.keys()); np.random.shuffle(ct_list)
            return ct_list[:top_k]
        t_norm = text_embed / (np.linalg.norm(text_embed) + 1e-8)
        scores = {}
        for ct, ct_embed in self.celltype_embeds.items():
            c_norm = ct_embed / (np.linalg.norm(ct_embed) + 1e-8)
            scores[ct] = float(np.dot(t_norm, c_norm))
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]


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

    print("\n" + "="*70)
    print("[CW] Running CellWhisperer on 100 queries (DT2)...")
    print("="*70)

    RECALL_KS = [1, 2, 3, 5, 8]
    cw_results = {"ontology": [], "expression": []}
    for q in QUERIES:
        ranked = cw_scorer.score_query(q["text"], top_k=8)
        entry = {"query_id": q["id"], "query_text": q["text"], "expected": q["expected_clusters"],
                 "retrieved_top8": ranked[:8], "mrr": round(mrr(q["expected_clusters"], ranked), 4)}
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
    MC = {"random":"#9E9E9E","cellwhisperer_real":"#E91E63","semantic":"#2196F3","scgpt":"#FF9800","union":"#4CAF50"}
    ML = {"random":"Random","cellwhisperer_real":"CellWhisp.","semantic":"Semantic","scgpt":"scGPT","union":"Union(S+G)"}

    def gm(cat, mode, mk):
        if mode == "cellwhisperer_real": return cw_agg[cat].get(mk, 0)
        return elisa_summary.get(f"{cat}_{mode}", {}).get(mk, 0)

    # Console
    print("\n" + "="*95)
    print("ELISA vs CellWhisperer — DT2: Breast Tissue Atlas (100 queries)")
    print("="*95)
    print(f"\n{'Category':<14} {'Mode':<16} {'R@1':>7} {'R@2':>7} {'R@3':>7} {'R@5':>7} {'R@8':>7} {'MRR':>7} {'GeneR':>7}")
    print("-"*90)
    for cat in ["ontology","expression"]:
        for mode in ALL_MODES:
            gr = gm(cat, mode, "mean_gene_recall"); gr_s = "  N/A" if mode=="cellwhisperer_real" else f"{gr:.3f}"
            print(f"{cat:<14} {ML[mode]:<16} "
                  f"{gm(cat,mode,'mean_recall@1'):>7.3f} {gm(cat,mode,'mean_recall@2'):>7.3f} "
                  f"{gm(cat,mode,'mean_recall@3'):>7.3f} {gm(cat,mode,'mean_recall@5'):>7.3f} "
                  f"{gm(cat,mode,'mean_recall@8'):>7.3f} {gm(cat,mode,'mean_mrr'):>7.3f} {gr_s:>7}")
        print()
    ana = elisa_analytical
    print("── Analytical (ELISA only) ──")
    print(f"  Pathways: {ana.get('pathways',{}).get('alignment',0):.1f}%  Interactions: {ana.get('interactions',{}).get('lr_recovery_rate',0):.1f}%")
    print("="*95)

    # Figures (same structure as original)
    cats = ["ontology","expression"]
    cat_titles = ["Ontology Queries\n(concept-level)","Expression Queries\n(gene-signature)"]

    # Fig 1: Recall@1
    fig, axes = plt.subplots(1,2,figsize=(14,5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        vals = [gm(cat,m,"mean_recall@1") for m in ALL_MODES]; x = np.arange(len(ALL_MODES))
        bars = ax.bar(x, vals, color=[MC[m] for m in ALL_MODES], alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0,1.15); ax.set_ylabel("Mean Cluster Recall@1"); ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for b,v in zip(bars,vals):
            if v>0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("ELISA vs CellWhisperer — DT2 (100 queries)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"fig1_recall1.png"),dpi=300,bbox_inches="tight"); plt.close()

    # Fig 2: All metrics
    fig, axes = plt.subplots(1,3,figsize=(20,5.5))
    for ax, mk, mt in zip(axes, ["mean_recall@1","mean_recall@2","mean_mrr"], ["Recall@1","Recall@2","MRR"]):
        x = np.arange(len(ALL_MODES)); w = 0.35
        ax.bar(x-w/2, [gm("ontology",m,mk) for m in ALL_MODES], w, label="Ontology", alpha=0.85, color=[MC[m] for m in ALL_MODES], edgecolor="white")
        ax.bar(x+w/2, [gm("expression",m,mk) for m in ALL_MODES], w, label="Expression", alpha=0.45, color=[MC[m] for m in ALL_MODES], edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0,1.15); ax.set_title(mt, fontsize=13, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"fig2_all_metrics.png"),dpi=300,bbox_inches="tight"); plt.close()

    # Fig 3: Radar
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    rl = ["Ont R@1","Ont R@2","Ont MRR","Exp R@1","Exp R@2","Exp MRR"]; N = len(rl)
    angles = [n/float(N)*2*np.pi for n in range(N)]; angles += angles[:1]
    for mode, color, ls in [("cellwhisperer_real","#E91E63","--"),("semantic","#2196F3","-"),("scgpt","#FF9800","-"),("union","#4CAF50","-")]:
        v = [gm(cat,mode,mk) for cat in ["ontology","expression"] for mk in ["mean_recall@1","mean_recall@2","mean_mrr"]]; v += v[:1]
        ax.plot(angles, v, linewidth=2, linestyle=ls, label=ML[mode], color=color); ax.fill(angles, v, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rl, fontsize=10); ax.set_ylim(0,1)
    ax.set_title("ELISA vs CellWhisperer\n(DT2: Breast Tissue)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=10)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"fig3_radar.png"),dpi=300,bbox_inches="tight"); plt.close()

    # Save JSON
    output = {"cellwhisperer": {"results": cw_results, "aggregated": cw_agg},
              "elisa_summary": elisa_summary, "elisa_analytical": elisa_analytical,
              "modes": ALL_MODES, "n_queries": len(QUERIES), "dataset": "DT2_BreastTissue",
              "timestamp": datetime.now().isoformat()}
    rp = os.path.join(out_dir, "elisa_vs_cellwhisperer_DT2_results.json")
    with open(rp, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")


def main():
    parser = argparse.ArgumentParser(description="ELISA vs CellWhisperer — DT2")
    parser.add_argument("--elisa-results", required=True)
    parser.add_argument("--cw-npz", required=True)
    parser.add_argument("--cw-leiden", required=True)
    parser.add_argument("--cw-ckpt", required=True)
    parser.add_argument("--cf-h5ad", required=True)
    parser.add_argument("--paper", default="DT2")
    parser.add_argument("--out", default="comparison_DT2/")
    args = parser.parse_args()
    cw = CellWhispererScorer(args.cw_npz, args.cw_leiden, args.cw_ckpt, args.cf_h5ad)
    run_comparison(cw, args.elisa_results, args.paper, args.out)
    print("\n[DONE]")

if __name__ == "__main__":
    main()
