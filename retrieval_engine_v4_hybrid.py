#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
elisa_engine_compat.py
======================
Compatibility layer between `retrieval_engine_v4_hybrid.RetrievalEngine`
(a thin v3-benchmark shim) and `elisa_chat_v4.py`.

Usage — in elisa_chat_v4.py:

    from elisa_engine_compat import RetrievalEngine

and, right after the h5ad is loaded:

    if adata is not None:
        engine.attach_adata(adata, cluster_key=cluster_key)


ALIGNMENT WITH THE MANUSCRIPT
-----------------------------
This version implements the algorithms exactly as described in the paper,
so that code and text agree:

* Hybrid retrieval is TRUE LATE FUSION via reciprocal rank fusion
  (Eq. 8, k = 60, per-pipeline weights). `lambda_sem` is the semantic
  weight: 0.0 = pure gene marker scoring, 1.0 = pure semantic,
  0.5 = balanced 1:1 fusion. `pre_k` controls the candidate depth of
  each pipeline before fusion and is genuinely used.
* `query_union()` implements the additive union benchmarking strategy
  (primary pipeline's full ranked list, then unique clusters from the
  secondary appended in original rank order).
* Ligand-receptor score = pct_in(ligand, source) * pct_in(receptor, target),
  with min ligand 0.10, min receptor 0.05, self-interactions excluded.
* Pathway score = mean pct_in over pathway genes detected in the cluster's
  DE profile, requiring >= 3 detected genes for a non-zero score, with
  coverage reported alongside.
* Gene evidence is ranked by |log2FC| x specificity x detection over
  ENRICHED genes only (pct_in >= 0.10, padj <= 0.05), which suppresses the
  junk (TCR segments, lincRNAs, olfactory receptors, unmapped ENSG) that
  plain |log2FC| sorting surfaces.

REMAINING CAVEATS (stated honestly in the payloads)
---------------------------------------------------
* `gene_stats` holds per-cluster DE/marker statistics, not a full expression
  matrix. Pathway scores and interactions are therefore *marker-enrichment
  proxies*, not per-cell measurements.
* `interactions()` predicts co-enrichment of a ligand in one cluster and its
  receptor in another. No null model / permutation test is applied.
* `compare()` requires per-cell condition labels (attached h5ad or cells_csv).
  Without them it returns {"error": ...} rather than fabricating a result.
"""

import math
from typing import List, Dict, Optional

import numpy as np

from retrieval_engine_v4_hybrid import (
    RetrievalEngine as _V3Wrapper,
    classify_query,
    extract_gene_names,
    gene_pipeline,
    semantic_pipeline,  # noqa: F401  (kept for API parity with the v3 module)
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
    "status", "group", "patient_group", "treatment", "genotype", "phenotype",
    "diagnosis", "sample_type", "stimulation", "timepoint", "age_group",
    "development_stage", "organoid_line", "line",
]

# Columns that look like conditions but are really identifiers
CONDITION_KEY_BLOCKLIST = {
    "donor_id", "sample_id", "cell_id", "barcode", "batch", "library_id",
    "suspension_type", "assay",
}

# Reciprocal rank fusion constant (manuscript Eq. 8)
RRF_K = 60


# ============================================================
# BUILT-IN PATHWAY GENE SETS
# 60+ curated sets across the five categories named in the manuscript.
# ============================================================

PATHWAYS: Dict[str, List[str]] = {

    # ---------- IMMUNE SIGNALING ----------
    "type_i_interferon": [
        "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "MX2", "OAS1", "OAS2",
        "OAS3", "IFI6", "IFI27", "IFI44L", "STAT1", "STAT2", "IRF7", "IRF9",
        "RSAD2", "BST2", "XAF1", "IFITM3",
    ],
    "interferon_gamma": [
        "IFNG", "IFNGR1", "IFNGR2", "STAT1", "IRF1", "JAK1", "JAK2", "GBP1",
        "GBP2", "GBP5", "CXCL9", "CXCL10", "CXCL11", "SOCS1", "CIITA",
    ],
    "tnf_nfkb": [
        "TNF", "TNFAIP3", "NFKB1", "NFKB2", "RELA", "RELB", "NFKBIA",
        "NFKBIZ", "TRAF1", "BIRC3", "CXCL2", "CCL20", "ICAM1", "SOD2",
    ],
    "jak_stat": [
        "JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3", "STAT4",
        "STAT5A", "STAT5B", "STAT6", "SOCS1", "SOCS3", "PIAS1",
    ],
    "complement": [
        "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C3", "C3AR1", "C5AR1",
        "CFB", "CFD", "CFH", "CD55", "CD59", "SERPING1",
    ],
    "tlr_signaling": [
        "TLR1", "TLR2", "TLR3", "TLR4", "TLR5", "TLR7", "TLR8", "TLR9",
        "MYD88", "TICAM1", "IRAK1", "IRAK4", "TRAF6", "CD14", "LY96",
    ],
    "chemokine_signaling": [
        "CXCL1", "CXCL2", "CXCL8", "CXCL9", "CXCL10", "CXCL12", "CCL2",
        "CCL3", "CCL4", "CCL5", "CCL19", "CCL21", "CXCR3", "CXCR4", "CCR7",
    ],
    "inflammatory_cytokines": [
        "IL1A", "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4",
        "CXCL1", "CXCL2", "PTGS2", "SOD2", "NFKB1", "NFKBIA",
    ],
    "antigen_presentation_mhc_i": [
        "HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "B2M", "TAP1", "TAP2",
        "TAPBP", "PSMB8", "PSMB9", "NLRC5", "CALR", "CANX",
    ],
    "antigen_presentation_mhc_ii": [
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
        "HLA-DMA", "HLA-DMB", "HLA-DOA", "CD74", "CIITA",
    ],
    "cytotoxicity": [
        "GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "GNLY", "NKG7", "KLRD1",
        "KLRK1", "KLRC1", "FASLG", "IFNG", "CTSW",
    ],
    "t_cell_activation": [
        "CD3D", "CD3E", "CD3G", "CD247", "LCK", "ZAP70", "LAT", "CD28",
        "ICOS", "TNFRSF9", "CD69", "IL2RA", "NFATC1", "ITK",
    ],
    "t_cell_exhaustion": [
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "TOX2", "ENTPD1",
        "BTLA", "CD244", "EOMES", "NR4A1",
    ],
    "b_cell_receptor": [
        "CD19", "MS4A1", "CD79A", "CD79B", "BLNK", "BTK", "SYK", "PAX5",
        "BANK1", "CR2", "FCRL1", "TNFRSF13C",
    ],
    "inflammasome": [
        "NLRP3", "PYCARD", "CASP1", "IL1B", "IL18", "GSDMD", "TXNIP",
        "NLRC4", "AIM2", "NEK7", "P2RX7",
    ],
    "immune_checkpoint": [
        "CD274", "PDCD1LG2", "PDCD1", "CTLA4", "CD80", "CD86", "LGALS9",
        "HAVCR2", "TIGIT", "NECTIN2", "PVR", "CD47", "SIRPA", "HLA-E",
    ],

    # ---------- CELL BIOLOGY ----------
    "cell_cycle": [
        "MKI67", "TOP2A", "CCNB1", "CCNA2", "CDK1", "PCNA", "TYMS", "BIRC5",
        "AURKB", "UBE2C", "RRM2", "TUBB4B", "STMN1", "HMGB2", "ASPM", "KIF11",
    ],
    "dna_replication": [
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "PCNA", "RFC2",
        "RFC4", "POLA1", "POLE", "PRIM1", "GINS2", "CDC45",
    ],
    "dna_damage_response": [
        "ATM", "ATR", "CHEK1", "CHEK2", "TP53", "TP53BP1", "H2AX", "BRCA1",
        "BRCA2", "RAD50", "RAD51", "MRE11", "ERCC6", "ERCC8", "XRCC1",
    ],
    "apoptosis": [
        "BAX", "BAK1", "BID", "CASP3", "CASP7", "CASP8", "CASP9", "BCL2",
        "BCL2L1", "MCL1", "TP53", "CDKN1A", "PMAIP1", "BBC3", "APAF1",
    ],
    "autophagy": [
        "MAP1LC3B", "GABARAP", "GABARAPL1", "SQSTM1", "ATG3", "ATG5", "ATG7",
        "ATG12", "ATG16L1", "BECN1", "ULK1", "WIPI2", "LAMP1", "TFEB",
    ],
    "senescence": [
        "CDKN1A", "CDKN2A", "TP53", "SERPINE1", "GLB1", "IL6", "CXCL8",
        "IGFBP3", "IGFBP7", "MMP3", "TNFRSF10C", "LMNB1",
    ],
    "mtor_signaling": [
        "MTOR", "RPTOR", "RICTOR", "AKT1", "RPS6", "RPS6KB1", "EIF4EBP1",
        "TSC1", "TSC2", "RHEB", "DDIT4", "LAMTOR1", "SLC7A5",
    ],
    "pi3k_akt": [
        "PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "PTEN", "PDK1",
        "GSK3B", "FOXO1", "FOXO3", "INPP4B", "MTOR", "BAD",
    ],
    "mapk_erk": [
        "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "RAF1", "BRAF", "KRAS", "HRAS",
        "NRAS", "DUSP1", "DUSP6", "SOS1", "SHC1", "GRB2",
    ],
    "wnt_signaling": [
        "WNT2", "WNT2B", "WNT5A", "WNT7B", "WNT11", "FZD1", "FZD2", "FZD4",
        "LRP5", "LRP6", "CTNNB1", "AXIN2", "LEF1", "TCF7L2", "RSPO2", "SFRP1",
    ],
    "notch_signaling": [
        "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1",
        "DLL3", "DLL4", "HES1", "HEY1", "HEYL", "RBPJ", "MAML1",
    ],
    "hippo_yap": [
        "YAP1", "WWTR1", "TEAD1", "TEAD2", "TEAD4", "LATS1", "LATS2", "STK3",
        "STK4", "SAV1", "MOB1A", "AMOTL2", "CTGF", "CYR61", "ANKRD1",
    ],
    "hedgehog_signaling": [
        "SHH", "IHH", "DHH", "PTCH1", "PTCH2", "SMO", "GLI1", "GLI2", "GLI3",
        "SUFU", "HHIP", "GAS1",
    ],
    "tgf_beta_bmp": [
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
        "SMAD4", "BMP2", "BMP4", "BMPR1A", "BMPR2", "ID1", "ID2", "ID3",
    ],
    "ubiquitin_proteasome": [
        "UBB", "UBC", "UBA1", "UBE2D3", "UBE2N", "UBE2I", "UBA2", "ITCH",
        "NEDD4", "NEDD4L", "TRIM21", "TRIM65", "PIAS1", "PSMA1", "PSMB5",
        "USP8", "UBAP1",
    ],
    "unfolded_protein_response": [
        "HSPA5", "DDIT3", "XBP1", "ATF4", "ATF6", "ERN1", "EIF2AK3", "PDIA3",
        "PDIA4", "HERPUD1", "SEL1L", "EDEM1", "CALR", "CANX",
    ],
    "epithelial_mesenchymal_transition": [
        "VIM", "FN1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2",
        "SPARC", "TAGLN", "ACTA2", "COL1A1", "COL1A2", "TGFB1", "SERPINE1",
    ],
    "hypoxia": [
        "HIF1A", "VEGFA", "SLC2A1", "LDHA", "PGK1", "ALDOA", "ENO1", "CA9",
        "ADM", "BNIP3", "NDRG1", "P4HA1", "EGLN3",
    ],

    # ---------- NEUROSCIENCE ----------
    "glutamatergic_synapse": [
        "SLC17A6", "SLC17A7", "GRIN1", "GRIN2B", "GRIA1", "GRIA2", "GRM5",
        "DLG4", "SHANK2", "SHANK3", "CAMK2A", "SYN1", "NEFM",
    ],
    "gabaergic_synapse": [
        "GAD1", "GAD2", "SLC32A1", "GABRA1", "GABRB2", "GABRG2", "GPHN",
        "DLX1", "DLX2", "DLX5", "LHX6", "SST", "PVALB", "VIP",
    ],
    "synaptic_vesicle_cycle": [
        "SNAP25", "SYT1", "STX1A", "VAMP2", "SYP", "SV2A", "RAB3A", "CPLX1",
        "NSF", "UNC13A", "SYN2", "DNM1",
    ],
    "neurogenesis": [
        "SOX2", "PAX6", "NES", "VIM", "HES1", "HES5", "FABP7", "DCX",
        "NEUROD1", "NEUROG2", "ASCL1", "TBR1", "EOMES", "RBFOX3",
    ],
    "myelination": [
        "MBP", "MOG", "PLP1", "MAG", "MPZ", "CNP", "SOX10", "OLIG1", "OLIG2",
        "PDGFRA", "CSPG4", "CDH19",
    ],
    "axon_guidance": [
        "ROBO1", "ROBO2", "SLIT1", "SLIT2", "DCC", "NTN1", "UNC5B", "EPHA4",
        "EPHB2", "EFNB2", "SEMA3A", "PLXNA2", "NRP1", "NRP2",
    ],
    "dopaminergic_neuron": [
        "TH", "DDC", "SLC6A3", "SLC18A2", "NR4A2", "LMX1A", "LMX1B", "FOXA2",
        "PITX3", "EN1", "KCNJ6", "CALB1",
    ],
    "neurodegeneration": [
        "APP", "APOE", "MAPT", "PSEN1", "SNCA", "TREM2", "GRN", "SORT1",
        "CLU", "BIN1", "PICALM", "TARDBP",
    ],

    # ---------- METABOLISM ----------
    "oxidative_phosphorylation": [
        "NDUFA4", "NDUFB2", "NDUFS1", "COX4I1", "COX5A", "COX6C", "COX7C",
        "ATP5F1A", "ATP5F1B", "ATP5MC2", "UQCRB", "UQCRQ", "SDHB", "CYC1",
    ],
    "glycolysis": [
        "HK1", "HK2", "GPI", "PFKL", "PFKP", "ALDOA", "GAPDH", "PGK1",
        "PGAM1", "ENO1", "PKM", "LDHA", "SLC2A1", "SLC16A3",
    ],
    "fatty_acid_oxidation": [
        "CPT1A", "CPT2", "ACADM", "ACADVL", "HADHA", "HADHB", "ACAA2",
        "ECHS1", "ACOX1", "PPARA", "SLC25A20", "ETFA",
    ],
    "lipid_metabolism": [
        "FASN", "SCD", "ACACA", "ACLY", "ELOVL5", "ELOVL6", "DGAT1", "DGAT2",
        "PLIN1", "PLIN2", "FABP4", "FABP5", "LPL", "APOE", "SOAT1",
    ],
    "cholesterol_biosynthesis": [
        "HMGCR", "HMGCS1", "SQLE", "LSS", "FDFT1", "IDI1", "MVD", "MVK",
        "DHCR7", "DHCR24", "SREBF2", "INSIG1", "LDLR",
    ],
    "amino_acid_metabolism": [
        "GLS", "GLUL", "ASNS", "PSAT1", "PHGDH", "SHMT2", "SLC7A11",
        "SLC7A5", "SLC1A5", "GOT1", "GOT2", "BCAT1",
    ],
    "one_carbon_metabolism": [
        "MTHFD1", "MTHFD2", "SHMT1", "SHMT2", "TYMS", "DHFR", "MTR", "MTRR",
        "ATIC", "GART", "FOLR1", "SLC19A1",
    ],
    "ros_oxidative_stress": [
        "SOD1", "SOD2", "CAT", "GPX1", "GPX4", "PRDX1", "PRDX2", "TXN",
        "TXNRD1", "NQO1", "HMOX1", "GCLC", "GCLM", "NFE2L2", "TXNIP",
    ],
    "heme_iron_metabolism": [
        "HMOX1", "FTL", "FTH1", "TFRC", "SLC40A1", "SLC11A2", "ALAS1",
        "ALAS2", "HBB", "HBA1", "CP", "HAMP",
    ],

    # ---------- TISSUE-SPECIFIC ----------
    "alveolar_surfactant": [
        "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "SFTPD", "SFTA3", "NAPSA",
        "LAMP3", "PGC", "SLC34A2", "ABCA3", "NKX2-1", "LPCAT1", "CTSH",
    ],
    "surfactant_processing_trafficking": [
        "ITCH", "NEDD4", "NEDD4L", "UBE2N", "HGS", "VPS28", "UBAP1", "USP8",
        "RABGEF1", "EEA1", "MICALL1", "LAMP3", "ABCA3", "CTSH", "CKAP4",
        "ZDHHC2",
    ],
    "alveolar_type_1": [
        "AGER", "PDPN", "CAV1", "AQP5", "HOPX", "CLIC5", "SPOCK2", "EMP2",
        "RTKN2", "MYL9", "TIMP3",
    ],
    "ciliogenesis": [
        "FOXJ1", "DNAH5", "DNAI1", "DNAI2", "RSPH1", "RSPH4A", "PIFO",
        "CAPS", "TPPP3", "SPAG6", "TEKT1", "CFAP43", "IFT81", "TUBA1A",
    ],
    "mucin_secretion": [
        "MUC5AC", "MUC5B", "MUC1", "MUC16", "SCGB1A1", "SCGB3A1", "SCGB3A2",
        "BPIFA1", "BPIFB1", "TFF3", "AGR2", "SPDEF", "CREB3L1",
    ],
    "epithelial_defense": [
        "LTF", "LYZ", "SLPI", "PIGR", "DEFB1", "S100A8", "S100A9", "CXCL17",
        "BPIFA1", "SCGB1A1", "WFDC2", "MUC1",
    ],
    "extracellular_matrix": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL6A2", "FN1", "LUM",
        "DCN", "ELN", "FBN1", "MMP2", "MMP9", "TIMP1", "SPARC", "POSTN",
    ],
    "fibrosis": [
        "COL1A1", "COL3A1", "ACTA2", "TAGLN", "POSTN", "FAP", "TGFB1",
        "SERPINE1", "CTGF", "LOX", "LOXL2", "PDGFRB", "THBS2",
    ],
    "angiogenesis_vegf": [
        "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "FLT4", "PGF", "NRP1",
        "NRP2", "ANGPT1", "ANGPT2", "TEK", "PECAM1", "CDH5", "ESM1",
    ],
    "endothelial_identity": [
        "PECAM1", "CDH5", "VWF", "CLDN5", "ERG", "ACKR1", "PLVAP", "LDB2",
        "MECOM", "EGFL7", "RAMP2", "SOX17",
    ],
    "smooth_muscle_contraction": [
        "ACTA2", "MYH11", "TAGLN", "CNN1", "MYL9", "DES", "LMOD1", "PDGFRB",
        "RGS5", "NOTCH3", "CALD1",
    ],
    "keratinization": [
        "KRT5", "KRT14", "KRT15", "KRT17", "TP63", "CSTA", "SFN", "DSP",
        "PKP1", "COL17A1", "LAMB3", "ITGB4",
    ],
    "neuroendocrine_program": [
        "ASCL1", "NEUROD1", "CHGA", "CHGB", "SYP", "GRP", "CALCA", "SCG2",
        "PCSK1", "INSM1", "SYT1", "PHOX2B",
    ],
    "adipogenesis": [
        "PPARG", "CEBPA", "CEBPB", "ADIPOQ", "LEP", "PLIN1", "FABP4", "LPL",
        "CIDEC", "CFD", "GPD1", "SLC2A4",
    ],
    "steroidogenesis": [
        "STAR", "CYP11A1", "CYP11B1", "CYP17A1", "CYP21A2", "HSD3B2",
        "NR5A1", "MC2R", "FDX1", "SCARB1",
    ],
    "melanocyte_pigmentation": [
        "MITF", "TYR", "TYRP1", "DCT", "PMEL", "MLANA", "SOX10", "GPR143",
        "OCA2", "SLC45A2",
    ],
    "hepatocyte_function": [
        "ALB", "APOB", "APOA1", "TTR", "SERPINA1", "HNF4A", "CYP3A4",
        "TF", "FGA", "FGB", "AHSG", "ASGR1",
    ],
}

PATHWAY_CATEGORIES: Dict[str, List[str]] = {
    "immune_signaling": [
        "type_i_interferon", "interferon_gamma", "tnf_nfkb", "jak_stat",
        "complement", "tlr_signaling", "chemokine_signaling",
        "inflammatory_cytokines", "antigen_presentation_mhc_i",
        "antigen_presentation_mhc_ii", "cytotoxicity", "t_cell_activation",
        "t_cell_exhaustion", "b_cell_receptor", "inflammasome",
        "immune_checkpoint",
    ],
    "cell_biology": [
        "cell_cycle", "dna_replication", "dna_damage_response", "apoptosis",
        "autophagy", "senescence", "mtor_signaling", "pi3k_akt", "mapk_erk",
        "wnt_signaling", "notch_signaling", "hippo_yap", "hedgehog_signaling",
        "tgf_beta_bmp", "ubiquitin_proteasome", "unfolded_protein_response",
        "epithelial_mesenchymal_transition", "hypoxia",
    ],
    "neuroscience": [
        "glutamatergic_synapse", "gabaergic_synapse", "synaptic_vesicle_cycle",
        "neurogenesis", "myelination", "axon_guidance", "dopaminergic_neuron",
        "neurodegeneration",
    ],
    "metabolism": [
        "oxidative_phosphorylation", "glycolysis", "fatty_acid_oxidation",
        "lipid_metabolism", "cholesterol_biosynthesis",
        "amino_acid_metabolism", "one_carbon_metabolism",
        "ros_oxidative_stress", "heme_iron_metabolism",
    ],
    "tissue_specific": [
        "alveolar_surfactant", "surfactant_processing_trafficking",
        "alveolar_type_1", "ciliogenesis", "mucin_secretion",
        "epithelial_defense", "extracellular_matrix", "fibrosis",
        "angiogenesis_vegf", "endothelial_identity",
        "smooth_muscle_contraction", "keratinization",
        "neuroendocrine_program", "adipogenesis", "steroidogenesis",
        "melanocyte_pigmentation", "hepatocyte_function",
    ],
}

PATHWAY_TO_CATEGORY = {p: cat for cat, ps in PATHWAY_CATEGORIES.items()
                       for p in ps}

# Names as they are actually written in papers -> canonical keys.
# Without these, "pathway: IFN-gamma signaling" fails to resolve.
PATHWAY_ALIASES = {
    "ifn_gamma": "interferon_gamma",
    "ifng": "interferon_gamma",
    "ifn_g": "interferon_gamma",
    "interferon_gamma_signaling": "interferon_gamma",
    "type_i_ifn": "type_i_interferon",
    "type_1_interferon": "type_i_interferon",
    "interferon_alpha": "type_i_interferon",
    "isg": "type_i_interferon",
    "nfkb": "tnf_nfkb",
    "nf_kb": "tnf_nfkb",
    "tnf_alpha": "tnf_nfkb",
    "mhc_i": "antigen_presentation_mhc_i",
    "mhc_class_i": "antigen_presentation_mhc_i",
    "mhc_ii": "antigen_presentation_mhc_ii",
    "mhc_class_ii": "antigen_presentation_mhc_ii",
    "checkpoint": "immune_checkpoint",
    "exhaustion": "t_cell_exhaustion",
    "surfactant": "alveolar_surfactant",
    "surfactant_metabolism": "alveolar_surfactant",
    "sftpc_trafficking": "surfactant_processing_trafficking",
    "at1": "alveolar_type_1",
    "at2": "alveolar_surfactant",
    "cilia": "ciliogenesis",
    "mucin": "mucin_secretion",
    "mucus": "mucin_secretion",
    "emt": "epithelial_mesenchymal_transition",
    "oxphos": "oxidative_phosphorylation",
    "wnt": "wnt_signaling",
    "notch": "notch_signaling",
    "hedgehog": "hedgehog_signaling",
    "shh": "hedgehog_signaling",
    "mtor": "mtor_signaling",
    "erbb": "mapk_erk",
    "erk": "mapk_erk",
    "akt": "pi3k_akt",
    "tgf_beta": "tgf_beta_bmp",
    "bmp": "tgf_beta_bmp",
    "vegf": "angiogenesis_vegf",
    "angiogenesis": "angiogenesis_vegf",
    "ecm": "extracellular_matrix",
    "glycolytic": "glycolysis",
    "ubiquitin": "ubiquitin_proteasome",
    "upr": "unfolded_protein_response",
    "er_stress": "unfolded_protein_response",
}

# Generic words that carry no discriminative signal when fuzzy-matching
PATHWAY_STOPWORDS = {
    "signaling", "signalling", "pathway", "pathways", "response",
    "responses", "activity", "program", "programs", "cell", "the", "of",
    "and", "in",
}


# ============================================================
# BUILT-IN LIGAND-RECEPTOR TABLE
# (ligand, receptor, pathway) — compiled from CellChat / CellPhoneDB /
# NicheNet style resources, plus context pairs for lung, neuro, and
# immune-checkpoint biology. The exact count is reported at runtime as
# `lr_database_size` in the interactions payload.
# ============================================================

LR_PAIRS = [
    # ---- WNT ----
    ("WNT1", "FZD1", "WNT"), ("WNT2", "FZD1", "WNT"),
    ("WNT2B", "FZD4", "WNT"), ("WNT3", "FZD1", "WNT"),
    ("WNT3A", "FZD1", "WNT"), ("WNT4", "FZD6", "WNT"),
    ("WNT5A", "FZD4", "WNT"), ("WNT5A", "ROR2", "WNT"),
    ("WNT5A", "RYK", "WNT"), ("WNT7A", "FZD5", "WNT"),
    ("WNT7B", "FZD1", "WNT"), ("WNT9A", "FZD5", "WNT"),
    ("WNT11", "FZD7", "WNT"), ("WNT16", "FZD4", "WNT"),
    ("RSPO1", "LGR4", "WNT"), ("RSPO2", "LGR5", "WNT"),
    ("RSPO3", "LGR5", "WNT"), ("DKK1", "LRP6", "WNT"),
    ("SFRP1", "FZD1", "WNT"),

    # ---- FGF ----
    ("FGF1", "FGFR1", "FGF"), ("FGF2", "FGFR1", "FGF"),
    ("FGF2", "FGFR2", "FGF"), ("FGF5", "FGFR1", "FGF"),
    ("FGF7", "FGFR2", "FGF"), ("FGF8", "FGFR2", "FGF"),
    ("FGF9", "FGFR3", "FGF"), ("FGF10", "FGFR1", "FGF"),
    ("FGF10", "FGFR2", "FGF"), ("FGF13", "FGFR1", "FGF"),
    ("FGF18", "FGFR3", "FGF"), ("FGF19", "FGFR4", "FGF"),
    ("FGF21", "FGFR1", "FGF"), ("FGF23", "FGFR1", "FGF"),

    # ---- TGFB / ACTIVIN ----
    ("TGFB1", "TGFBR1", "TGFB"), ("TGFB1", "TGFBR2", "TGFB"),
    ("TGFB2", "TGFBR2", "TGFB"), ("TGFB3", "TGFBR2", "TGFB"),
    ("TGFB1", "ACVRL1", "TGFB"), ("INHBA", "ACVR2A", "ACTIVIN"),
    ("INHBA", "ACVR1B", "ACTIVIN"), ("INHBB", "ACVR2B", "ACTIVIN"),
    ("NODAL", "ACVR2B", "ACTIVIN"), ("LEFTY1", "ACVR2B", "ACTIVIN"),
    ("GDF15", "GFRAL", "GDF"),

    # ---- BMP ----
    ("BMP2", "BMPR1A", "BMP"), ("BMP2", "BMPR2", "BMP"),
    ("BMP3", "ACVR2B", "BMP"), ("BMP4", "BMPR1A", "BMP"),
    ("BMP4", "BMPR2", "BMP"), ("BMP5", "BMPR1B", "BMP"),
    ("BMP6", "BMPR1A", "BMP"), ("BMP7", "BMPR2", "BMP"),
    ("BMP7", "ACVR1", "BMP"), ("BMP8B", "BMPR1A", "BMP"),
    ("GDF5", "BMPR1B", "BMP"), ("GREM1", "BMPR2", "BMP"),

    # ---- NOTCH ----
    ("JAG1", "NOTCH1", "NOTCH"), ("JAG1", "NOTCH2", "NOTCH"),
    ("JAG1", "NOTCH3", "NOTCH"), ("JAG1", "NOTCH4", "NOTCH"),
    ("JAG2", "NOTCH1", "NOTCH"), ("JAG2", "NOTCH2", "NOTCH"),
    ("DLL1", "NOTCH1", "NOTCH"), ("DLL1", "NOTCH2", "NOTCH"),
    ("DLL3", "NOTCH1", "NOTCH"), ("DLL4", "NOTCH1", "NOTCH"),
    ("DLL4", "NOTCH4", "NOTCH"), ("DLK1", "NOTCH1", "NOTCH"),

    # ---- VEGF / ANGIOPOIETIN ----
    ("VEGFA", "KDR", "VEGF"), ("VEGFA", "FLT1", "VEGF"),
    ("VEGFA", "NRP1", "VEGF"), ("VEGFB", "FLT1", "VEGF"),
    ("VEGFC", "FLT4", "VEGF"), ("VEGFC", "KDR", "VEGF"),
    ("PGF", "FLT1", "VEGF"), ("PGF", "NRP1", "VEGF"),
    ("ANGPT1", "TEK", "ANGIOPOIETIN"), ("ANGPT2", "TEK", "ANGIOPOIETIN"),
    ("ANGPTL4", "ITGB1", "ANGIOPOIETIN"), ("ESM1", "KDR", "VEGF"),
    ("EGFL7", "NOTCH1", "VEGF"), ("APLN", "APLNR", "APELIN"),

    # ---- PDGF ----
    ("PDGFA", "PDGFRA", "PDGF"), ("PDGFA", "PDGFRB", "PDGF"),
    ("PDGFB", "PDGFRA", "PDGF"), ("PDGFB", "PDGFRB", "PDGF"),
    ("PDGFC", "PDGFRA", "PDGF"), ("PDGFC", "PDGFRB", "PDGF"),
    ("PDGFD", "PDGFRA", "PDGF"), ("PDGFD", "PDGFRB", "PDGF"),

    # ---- HEDGEHOG ----
    ("SHH", "PTCH1", "HEDGEHOG"), ("SHH", "SMO", "HEDGEHOG"),
    ("SHH", "GAS1", "HEDGEHOG"), ("IHH", "PTCH1", "HEDGEHOG"),
    ("DHH", "PTCH1", "HEDGEHOG"),

    # ---- IGF / INSULIN ----
    ("IGF1", "IGF1R", "IGF"), ("IGF1", "INSR", "IGF"),
    ("IGF2", "IGF1R", "IGF"), ("IGF2", "IGF2R", "IGF"),
    ("IGF2", "INSR", "IGF"), ("INS", "INSR", "INSULIN"),
    ("IGFBP3", "IGF1R", "IGF"), ("IGFBP5", "IGF1R", "IGF"),

    # ---- HGF / TAM ----
    ("HGF", "MET", "HGF"), ("MST1", "MST1R", "HGF"),
    ("GAS6", "AXL", "TAM"), ("GAS6", "MERTK", "TAM"),
    ("PROS1", "AXL", "TAM"), ("PROS1", "MERTK", "TAM"),

    # ---- EGF / ERBB ----
    ("EGF", "EGFR", "EGF"), ("TGFA", "EGFR", "EGF"),
    ("HBEGF", "EGFR", "EGF"), ("HBEGF", "ERBB4", "EGF"),
    ("AREG", "EGFR", "EGF"), ("EREG", "EGFR", "EGF"),
    ("EREG", "ERBB4", "EGF"), ("BTC", "EGFR", "EGF"),
    ("BTC", "ERBB4", "EGF"), ("EPGN", "EGFR", "EGF"),
    ("NRG1", "ERBB3", "ERBB"), ("NRG1", "ERBB4", "ERBB"),
    ("NRG2", "ERBB4", "ERBB"), ("NRG3", "ERBB4", "ERBB"),
    ("NRG4", "ERBB4", "ERBB"),

    # ---- CHEMOKINES ----
    ("CXCL1", "CXCR2", "CHEMOKINE"), ("CXCL2", "CXCR2", "CHEMOKINE"),
    ("CXCL3", "CXCR2", "CHEMOKINE"), ("CXCL5", "CXCR2", "CHEMOKINE"),
    ("CXCL6", "CXCR1", "CHEMOKINE"), ("CXCL8", "CXCR1", "CHEMOKINE"),
    ("CXCL8", "CXCR2", "CHEMOKINE"), ("CXCL9", "CXCR3", "CHEMOKINE"),
    ("CXCL10", "CXCR3", "CHEMOKINE"), ("CXCL11", "CXCR3", "CHEMOKINE"),
    ("CXCL12", "CXCR4", "CHEMOKINE"), ("CXCL12", "ACKR3", "CHEMOKINE"),
    ("CXCL13", "CXCR5", "CHEMOKINE"), ("CXCL14", "CXCR4", "CHEMOKINE"),
    ("CXCL16", "CXCR6", "CHEMOKINE"), ("CXCL17", "GPR35", "CHEMOKINE"),
    ("CCL2", "CCR2", "CHEMOKINE"), ("CCL3", "CCR1", "CHEMOKINE"),
    ("CCL3", "CCR5", "CHEMOKINE"), ("CCL4", "CCR5", "CHEMOKINE"),
    ("CCL5", "CCR1", "CHEMOKINE"), ("CCL5", "CCR5", "CHEMOKINE"),
    ("CCL7", "CCR2", "CHEMOKINE"), ("CCL8", "CCR2", "CHEMOKINE"),
    ("CCL11", "CCR3", "CHEMOKINE"), ("CCL13", "CCR2", "CHEMOKINE"),
    ("CCL17", "CCR4", "CHEMOKINE"), ("CCL19", "CCR7", "CHEMOKINE"),
    ("CCL20", "CCR6", "CHEMOKINE"), ("CCL21", "CCR7", "CHEMOKINE"),
    ("CCL22", "CCR4", "CHEMOKINE"), ("CCL25", "CCR9", "CHEMOKINE"),
    ("CCL28", "CCR10", "CHEMOKINE"), ("CX3CL1", "CX3CR1", "CHEMOKINE"),
    ("XCL1", "XCR1", "CHEMOKINE"),

    # ---- INTERLEUKINS / CYTOKINES ----
    ("IL1A", "IL1R1", "IL1"), ("IL1B", "IL1R1", "IL1"),
    ("IL1RN", "IL1R1", "IL1"), ("IL18", "IL18R1", "IL1"),
    ("IL33", "IL1RL1", "IL1"), ("IL36G", "IL1RL2", "IL1"),
    ("IL2", "IL2RA", "IL2"), ("IL2", "IL2RB", "IL2"),
    ("IL15", "IL2RB", "IL2"), ("IL15", "IL15RA", "IL2"),
    ("IL7", "IL7R", "IL2"), ("IL9", "IL9R", "IL2"),
    ("IL21", "IL21R", "IL2"), ("IL4", "IL4R", "IL4"),
    ("IL13", "IL4R", "IL4"), ("IL13", "IL13RA1", "IL4"),
    ("IL5", "IL5RA", "IL5"), ("IL6", "IL6R", "IL6"),
    ("IL6", "IL6ST", "IL6"), ("IL11", "IL6ST", "IL6"),
    ("LIF", "LIFR", "IL6"), ("OSM", "OSMR", "IL6"),
    ("CNTF", "CNTFR", "IL6"), ("IL10", "IL10RA", "IL10"),
    ("IL22", "IL22RA1", "IL10"), ("IL24", "IL20RB", "IL10"),
    ("IL12A", "IL12RB1", "IL12"), ("IL23A", "IL12RB1", "IL12"),
    ("IL17A", "IL17RA", "IL17"), ("IL17F", "IL17RA", "IL17"),
    ("IL16", "CD4", "CYTOKINE"), ("IL34", "CSF1R", "CSF"),
    ("TSLP", "CRLF2", "CYTOKINE"), ("CSF1", "CSF1R", "CSF"),
    ("CSF2", "CSF2RA", "CSF"), ("CSF3", "CSF3R", "CSF"),
    ("KITLG", "KIT", "GROWTH_FACTOR"), ("FLT3LG", "FLT3", "GROWTH_FACTOR"),
    ("EPO", "EPOR", "GROWTH_FACTOR"), ("THPO", "MPL", "GROWTH_FACTOR"),

    # ---- INTERFERONS ----
    ("IFNG", "IFNGR1", "INTERFERON"), ("IFNG", "IFNGR2", "INTERFERON"),
    ("IFNA1", "IFNAR1", "INTERFERON"), ("IFNA2", "IFNAR2", "INTERFERON"),
    ("IFNB1", "IFNAR1", "INTERFERON"), ("IFNL1", "IFNLR1", "INTERFERON"),

    # ---- TNF SUPERFAMILY ----
    ("TNF", "TNFRSF1A", "TNF"), ("TNF", "TNFRSF1B", "TNF"),
    ("LTA", "TNFRSF1A", "TNF"), ("LTB", "LTBR", "TNF"),
    ("TNFSF10", "TNFRSF10A", "TNF"), ("TNFSF10", "TNFRSF10B", "TNF"),
    ("FASLG", "FAS", "TNF"), ("TNFSF11", "TNFRSF11A", "TNF"),
    ("TNFSF12", "TNFRSF12A", "TNF"), ("TNFSF13", "TNFRSF13B", "TNF"),
    ("TNFSF13B", "TNFRSF13C", "TNF"), ("TNFSF14", "TNFRSF14", "TNF"),
    ("TNFSF15", "TNFRSF25", "TNF"), ("CD40LG", "CD40", "TNF"),
    ("CD70", "CD27", "TNF"), ("TNFSF4", "TNFRSF4", "TNF"),
    ("TNFSF9", "TNFRSF9", "TNF"), ("TNFSF18", "TNFRSF18", "TNF"),

    # ---- CHECKPOINT / CO-STIMULATION / ANTIGEN ----
    ("CD274", "PDCD1", "IMMUNE_CHECKPOINT"),
    ("PDCD1LG2", "PDCD1", "IMMUNE_CHECKPOINT"),
    ("CD80", "CTLA4", "IMMUNE_CHECKPOINT"),
    ("CD86", "CTLA4", "IMMUNE_CHECKPOINT"),
    ("CD80", "CD28", "CO_STIMULATION"),
    ("CD86", "CD28", "CO_STIMULATION"),
    ("ICOSLG", "ICOS", "CO_STIMULATION"),
    ("HLA-E", "KLRC1", "IMMUNE_CHECKPOINT"),
    ("HLA-E", "KLRC2", "IMMUNE_CHECKPOINT"),
    ("HLA-E", "KLRD1", "IMMUNE_CHECKPOINT"),
    ("NECTIN2", "TIGIT", "IMMUNE_CHECKPOINT"),
    ("PVR", "TIGIT", "IMMUNE_CHECKPOINT"),
    ("PVR", "CD226", "IMMUNE_CHECKPOINT"),
    ("LGALS9", "HAVCR2", "IMMUNE_CHECKPOINT"),
    ("CD47", "SIRPA", "IMMUNE_CHECKPOINT"),
    ("CD24", "SIGLEC10", "IMMUNE_CHECKPOINT"),
    ("BTLA", "TNFRSF14", "IMMUNE_CHECKPOINT"),
    ("HLA-A", "CD8A", "ANTIGEN_PRESENTATION"),
    ("HLA-B", "CD8A", "ANTIGEN_PRESENTATION"),
    ("HLA-DRA", "CD4", "ANTIGEN_PRESENTATION"),
    ("HLA-DRB1", "CD4", "ANTIGEN_PRESENTATION"),
    ("HLA-DPA1", "CD4", "ANTIGEN_PRESENTATION"),

    # ---- ADHESION / ECM ----
    ("ICAM1", "ITGAL", "ADHESION"), ("ICAM1", "ITGAM", "ADHESION"),
    ("ICAM2", "ITGAL", "ADHESION"), ("VCAM1", "ITGA4", "ADHESION"),
    ("VCAM1", "ITGB1", "ADHESION"), ("SELE", "SELPLG", "ADHESION"),
    ("SELP", "SELPLG", "ADHESION"), ("SELL", "CD34", "ADHESION"),
    ("PECAM1", "PECAM1", "ADHESION"), ("CDH5", "CDH5", "ADHESION"),
    ("FN1", "ITGA5", "ECM"), ("FN1", "ITGAV", "ECM"),
    ("FN1", "ITGB1", "ECM"), ("FN1", "SDC4", "ECM"),
    ("COL1A1", "ITGB1", "ECM"), ("COL1A1", "DDR1", "ECM"),
    ("COL1A2", "ITGA2", "ECM"), ("COL3A1", "ITGB1", "ECM"),
    ("COL4A1", "ITGB1", "ECM"), ("COL4A1", "ITGA1", "ECM"),
    ("COL6A2", "ITGB1", "ECM"), ("LAMA3", "ITGA6", "ECM"),
    ("LAMA3", "ITGB4", "ECM"), ("LAMB3", "ITGA3", "ECM"),
    ("LAMC1", "ITGB1", "ECM"), ("NID1", "ITGB1", "ECM"),
    ("AGRN", "DAG1", "ECM"), ("RELN", "ITGB1", "ECM"),
    ("SPP1", "CD44", "ECM"), ("SPP1", "ITGAV", "ECM"),
    ("HAS2", "CD44", "ECM"), ("THBS1", "CD47", "ECM"),
    ("THBS1", "ITGB1", "ECM"), ("THBS2", "ITGB1", "ECM"),
    ("TNC", "ITGB1", "ECM"), ("VTN", "ITGAV", "ECM"),
    ("POSTN", "ITGAV", "ECM"), ("TGM2", "ITGB1", "ECM"),

    # ---- EPHRIN / SEMAPHORIN / GUIDANCE ----
    ("EFNA1", "EPHA2", "EPHRIN"), ("EFNA5", "EPHA4", "EPHRIN"),
    ("EFNB1", "EPHB2", "EPHRIN"), ("EFNB2", "EPHB4", "EPHRIN"),
    ("EFNB2", "EPHA4", "EPHRIN"), ("SEMA3A", "NRP1", "SEMAPHORIN"),
    ("SEMA3C", "NRP1", "SEMAPHORIN"), ("SEMA3E", "PLXND1", "SEMAPHORIN"),
    ("SEMA4D", "PLXNB1", "SEMAPHORIN"), ("SEMA6A", "PLXNA2", "SEMAPHORIN"),
    ("SEMA7A", "ITGB1", "SEMAPHORIN"), ("NTN1", "DCC", "NETRIN"),
    ("NTN1", "UNC5B", "NETRIN"), ("SLIT2", "ROBO1", "SLIT"),
    ("SLIT2", "ROBO2", "SLIT"), ("SLIT3", "ROBO1", "SLIT"),

    # ---- NEUROTROPHIN / GDNF ----
    ("NGF", "NTRK1", "NEUROTROPHIN"), ("NGF", "NGFR", "NEUROTROPHIN"),
    ("BDNF", "NTRK2", "NEUROTROPHIN"), ("NTF3", "NTRK3", "NEUROTROPHIN"),
    ("GDNF", "RET", "GDNF"), ("GDNF", "GFRA1", "GDNF"),
    ("NRTN", "GFRA2", "GDNF"), ("ARTN", "GFRA3", "GDNF"),

    # ---- COMPLEMENT / INNATE / MISC ----
    ("C3", "C3AR1", "COMPLEMENT"), ("C5", "C5AR1", "COMPLEMENT"),
    ("C1QA", "CD93", "COMPLEMENT"), ("C1QB", "CD93", "COMPLEMENT"),
    ("CALR", "LRP1", "PHAGOCYTOSIS"), ("CALR", "SCARF1", "PHAGOCYTOSIS"),
    ("APOE", "LRP1", "LIPID"), ("APOE", "TREM2", "LIPID"),
    ("APOE", "SORL1", "LIPID"), ("GRN", "SORT1", "GROWTH_FACTOR"),
    ("MDK", "LRP1", "MIDKINE"), ("MDK", "SDC1", "MIDKINE"),
    ("PTN", "PTPRZ1", "PLEIOTROPHIN"), ("PTN", "SDC3", "PLEIOTROPHIN"),
    ("APP", "CD74", "APP"), ("MIF", "CD74", "MIF"),
    ("MIF", "CXCR4", "MIF"), ("HMGB1", "TLR4", "DAMP"),
    ("HMGB1", "AGER", "DAMP"), ("S100A8", "TLR4", "DAMP"),
    ("S100A9", "AGER", "DAMP"), ("ANXA1", "FPR1", "ANNEXIN"),
    ("ANXA1", "FPR2", "ANNEXIN"), ("LGALS1", "PTPRC", "GALECTIN"),
    ("LGALS3", "LAG3", "GALECTIN"), ("TIMP1", "CD63", "MMP"),
    ("PLAU", "PLAUR", "PLASMINOGEN"), ("SERPINE1", "LRP1", "PLASMINOGEN"),
    ("EDN1", "EDNRA", "ENDOTHELIN"), ("EDN1", "EDNRB", "ENDOTHELIN"),
    ("EDN3", "EDNRB", "ENDOTHELIN"), ("ADM", "CALCRL", "ADRENOMEDULLIN"),
    ("CALCA", "CALCRL", "ADRENOMEDULLIN"), ("AGT", "AGTR1", "RAS"),
    ("AGT", "AGTR2", "RAS"), ("VIP", "VIPR1", "NEUROPEPTIDE"),
    ("ADCYAP1", "ADCYAP1R1", "NEUROPEPTIDE"),
    ("NPY", "NPY1R", "NEUROPEPTIDE"), ("GAL", "GALR1", "NEUROPEPTIDE"),
    ("GRP", "GRPR", "NEUROPEPTIDE"), ("POMC", "MC1R", "NEUROPEPTIDE"),
    ("NAMPT", "INSR", "METABOLIC"), ("LEP", "LEPR", "METABOLIC"),
    ("ADIPOQ", "ADIPOR1", "METABOLIC"), ("ADIPOQ", "ADIPOR2", "METABOLIC"),
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
            meta["top_markers"] = [e["gene"] for e in
                                   self._gene_evidence(key, 25)]
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
            "n_pathways": len(PATHWAYS),
            "pathway_categories": {k: len(v)
                                   for k, v in PATHWAY_CATEGORIES.items()},
            "available_pathways": sorted(PATHWAYS.keys()),
            "lr_database_size": len(LR_PAIRS),
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": RRF_K,
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
    # GENE EVIDENCE
    # --------------------------------------------------------

    def _gene_evidence(self, cid: str, top_n: int = 25,
                       min_pct_in: float = 0.10,
                       max_padj: float = 0.05) -> List[dict]:
        """
        Enriched markers only, ranked by |log2FC| x specificity x detection.

        Sorting by raw |log2FC| surfaces junk (TCR/BCR segments, lincRNAs,
        olfactory receptors, unmapped ENSG ids) that is nominally extreme but
        detected in a handful of cells. Requiring positive enrichment, a
        minimum detection rate and adjusted-p significance removes it.
        """
        stats = self.gene_stats.get(str(cid), {})
        if not stats:
            return []

        def _score(s):
            lf = _f(s.get("logfc"))
            pin = _pct(s.get("pct_in"))
            pout = _pct(s.get("pct_out"))
            spec = max(pin - pout, 0.0)
            return abs(lf) * (0.3 + spec) * pin

        kept = []
        for g, s in stats.items():
            lf = _f(s.get("logfc"))
            pin = _pct(s.get("pct_in"))
            padj = _f(s.get("pval_adj"), 1.0)
            if lf <= 0:
                continue
            if pin < min_pct_in:
                continue
            if padj > max_padj:
                continue
            kept.append((g, s, _score(s)))

        # Relax progressively rather than returning an empty evidence block
        if not kept:
            for g, s in stats.items():
                if _f(s.get("logfc")) > 0 and _pct(s.get("pct_in")) >= 0.05:
                    kept.append((g, s, _score(s)))
        if not kept:
            for g, s in stats.items():
                kept.append((g, s, _score(s)))

        kept.sort(key=lambda t: -t[2])
        out = []
        for g, s, sc in kept[:top_n]:
            pin = _pct(s.get("pct_in"))
            pout = _pct(s.get("pct_out"))
            out.append({
                "gene": g,
                "logfc": round(_f(s.get("logfc")), 4),
                "pct_in": round(pin, 4),
                "pct_out": round(pout, 4),
                "specificity": round(max(pin - pout, 0.0), 4),
                "pval_adj": _f(s.get("pval_adj"), 1.0),
                "marker_score": round(sc, 4),
            })
        return out

    # --------------------------------------------------------
    # RESULT FORMATTING
    # --------------------------------------------------------

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
        d = {str(cid): s for cid, s in ranked}
        arr = np.array([d.get(str(c), 0.0) for c in self.cluster_ids],
                       dtype=float)
        mx = arr.max() if arr.size else 0.0
        if mx > 0:
            arr = arr / mx
        return arr

    def _refresh_diagnostics(self, text: str):
        self._last_sem_sims = self._compute_all_sem_sims(text)
        self._last_expr_sims = self._compute_all_expr_scores(text)

    # --------------------------------------------------------
    # THE TWO PIPELINES (kept separate — no implicit fusion)
    # --------------------------------------------------------

    def _semantic_ranked(self, text: str, top_k: int):
        """BioBERT cosine similarity + name boost + synonym expansion."""
        return [(str(c), float(s)) for c, s in
                self._hybrid.query(text, top_k=top_k, force_mode="ontology")]

    def _gene_ranked(self, text: str, top_k: int):
        """Gene marker scoring against per-cluster DE profiles."""
        genes = extract_gene_names(text, self.all_genes)
        if not genes:
            return []
        ranked = gene_pipeline(genes, self.gene_stats,
                               [str(c) for c in self.cluster_ids],
                               scoring_mode=self._hybrid.gene_scoring_mode,
                               top_k=top_k)
        return [(str(c), float(s)) for c, s in ranked]

    @staticmethod
    def _rrf(ranked_lists, weights, k: int = RRF_K, top_k: int = 10):
        """
        Reciprocal rank fusion (manuscript Eq. 8):
            RRF(d) = sum_r  w_r / (k + rank_r(d) + 1),  rank 0-indexed.
        """
        scores: Dict[str, float] = {}
        for ranked, w in zip(ranked_lists, weights):
            if not ranked or w <= 0:
                continue
            for rank, (cid, _s) in enumerate(ranked):
                cid = str(cid)
                scores[cid] = scores.get(cid, 0.0) + float(w) / (k + rank + 1)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    def query_semantic(self, text: str, top_k: int = 10,
                       with_genes: bool = False, **_ignored) -> dict:
        self._refresh_diagnostics(text)
        ranked = self._semantic_ranked(text, top_k)
        payload = self._format_results(ranked, with_genes=with_genes,
                                       mode="semantic", query=text)
        payload["pipeline"] = ("semantic (BioBERT cosine similarity, "
                               "name boost alpha=0.15, "
                               "synonym boost beta=0.10)")
        return payload

    def query_hybrid(self, text: str, top_k: int = 10, lambda_sem: float = 0.5,
                     with_genes: bool = False, pre_k: int = 40,
                     gamma: float = None, **_ignored) -> dict:
        """
        Late fusion of the two independent pipelines via reciprocal rank
        fusion, as described in the manuscript.

        lambda_sem is the semantic pipeline weight:
            0.0 -> pure gene marker scoring
            0.5 -> balanced 1:1 fusion
            1.0 -> pure semantic
        pre_k is the candidate depth taken from each pipeline before fusion.
        """
        self._refresh_diagnostics(text)
        n_clusters = len(self.cluster_ids)
        pre_k = int(min(pre_k or n_clusters, n_clusters))

        w_sem = float(max(0.0, min(1.0, lambda_sem)))
        w_gene = 1.0 - w_sem

        sem_ranked = self._semantic_ranked(text, pre_k) if w_sem > 0 else []
        gene_ranked = self._gene_ranked(text, pre_k) if w_gene > 0 else []

        query_genes = extract_gene_names(text, self.all_genes)

        # Safety net: a gene-weighted query with no recognizable gene symbols
        # would otherwise return nothing. Fall back to the semantic pipeline.
        if not gene_ranked and not sem_ranked:
            sem_ranked = self._semantic_ranked(text, pre_k)
            w_sem, w_gene = 1.0, 0.0

        ranked = self._rrf([sem_ranked, gene_ranked], [w_sem, w_gene],
                           k=RRF_K, top_k=top_k)

        payload = self._format_results(ranked, with_genes=with_genes,
                                       mode="hybrid", query=text)
        payload.update({
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": RRF_K,
            "lambda_sem": w_sem,
            "weights": {"semantic": w_sem, "gene": w_gene},
            "pre_k": pre_k,
            "query_genes": query_genes,
            "gene_pipeline_active": bool(gene_ranked),
            "semantic_pipeline_active": bool(sem_ranked),
        })
        if gamma is not None:
            payload["gamma_note"] = (
                "gamma is a score-level reranking sharpness parameter and has "
                "no effect under rank-based RRF; it was not applied.")
        return payload

    # Tokens that indicate natural language rather than a gene signature.
    NL_HINT_TOKENS = {
        "cell", "cells", "activation", "signaling", "signalling", "response",
        "pathway", "expression", "infiltration", "differentiation", "marker",
        "markers", "in", "of", "and", "with", "the", "during", "after",
    }

    def _gene_token_fraction(self, text: str) -> float:
        """
        Fraction of query tokens that are recognised gene symbols.
        This is the same signal the query classifier uses, exposed as a
        number so the union can break 'mixed' ties without any ground truth.
        """
        toks = [t for t in text.replace(",", " ").replace(";", " ").split()
                if t.strip()]
        if not toks:
            return 0.0
        known = {g.upper() for g in self.all_genes}
        n_gene = sum(1 for t in toks if t.upper().strip(".,;:()") in known)
        return n_gene / len(toks)

    def query_union(self, text: str, top_k: int = 10,
                    with_genes: bool = False, pre_k: int = 40,
                    selection_mode: str = "classifier",
                    expected_clusters: Optional[List[str]] = None,
                    **_ignored) -> dict:
        """
        Additive union of the two pipelines.

        The primary pipeline supplies its full ranked list; clusters unique to
        the secondary are appended in their original rank order.

        selection_mode controls how the primary is chosen:

        * "classifier" (DEFAULT, and the only mode valid for reporting):
          the primary is chosen by the query classifier, exactly as at
          inference time. Gene-signature queries take the gene pipeline,
          natural-language queries take the semantic pipeline, and mixed
          queries are broken by the gene-token fraction. No ground truth is
          consulted, so the result is directly comparable to any
          single-system baseline.

        * "oracle_recall": the primary is the pipeline with the higher
          Recall@5 against `expected_clusters`. This CONSULTS THE ANSWER KEY.
          Because the primary's top-5 is also the union's top-5, Recall@5
          under this mode is algebraically identical to the per-query maximum
          of the two pipelines. It is therefore an oracle UPPER BOUND, not a
          system result, and must never be compared against baselines that
          receive no such privilege. Payloads from this mode are stamped with
          `oracle_assisted: True`.
        """
        self._refresh_diagnostics(text)
        n_clusters = len(self.cluster_ids)
        pre_k = int(min(pre_k or n_clusters, n_clusters))

        sem_ranked = self._semantic_ranked(text, pre_k)
        gene_ranked = self._gene_ranked(text, pre_k)
        auto_mode = classify_query(text, self.all_genes)
        gene_frac = self._gene_token_fraction(text)

        oracle_assisted = False

        if selection_mode == "oracle_recall":
            if not expected_clusters:
                return {"error": "selection_mode='oracle_recall' requires "
                                 "expected_clusters."}

            def _recall_at(ranked, expected, k=5):
                if not ranked:
                    return -1.0
                exp = {str(e).lower() for e in expected}
                got = [str(c).lower() for c, _ in ranked[:k]]
                hit = sum(1 for e in exp if any(e in g or g in e for g in got))
                return hit / len(exp)

            sem_primary = (_recall_at(sem_ranked, expected_clusters)
                           >= _recall_at(gene_ranked, expected_clusters))
            selection = "oracle Recall@5 against expected_clusters"
            oracle_assisted = True

        elif selection_mode == "classifier":
            if auto_mode == "gene_list":
                sem_primary = False
            elif auto_mode == "ontology":
                sem_primary = True
            else:  # mixed — break the tie on the gene-token fraction
                sem_primary = gene_frac < 0.5
            selection = f"query classifier ({auto_mode}, "
            selection += f"gene token fraction {gene_frac:.2f})"
        else:
            return {"error": f"Unknown selection_mode '{selection_mode}'. "
                             f"Use 'classifier' or 'oracle_recall'."}

        # A pipeline that returned nothing cannot be primary.
        if not gene_ranked:
            sem_primary = True
            selection += " [gene pipeline empty: no gene symbols in query]"
        elif not sem_ranked:
            sem_primary = False
            selection += " [semantic pipeline empty]"

        primary, secondary = ((sem_ranked, gene_ranked) if sem_primary
                              else (gene_ranked, sem_ranked))
        seen = set()
        merged = []
        for cid, s in list(primary) + list(secondary):
            if cid in seen:
                continue
            seen.add(cid)
            merged.append((cid, s))

        n_appended = len(merged) - len(primary)
        payload = self._format_results(merged[:max(top_k, 2 * top_k)],
                                       with_genes=with_genes,
                                       mode="union", query=text)
        payload.update({
            "strategy": "additive_union",
            "selection_mode": selection_mode,
            "primary_pipeline": "semantic" if sem_primary else "gene",
            "primary_selected_by": selection,
            "classified_as": auto_mode,
            "gene_token_fraction": round(gene_frac, 3),
            "pre_k": pre_k,
            "n_primary": len(primary),
            "n_appended_from_secondary": n_appended,
        })
        if oracle_assisted:
            payload["oracle_assisted"] = True
            payload["reporting_note"] = (
                "Primary pipeline chosen using the answer key. Recall@5 here "
                "equals the per-query maximum of the two pipelines by "
                "construction. Report as an oracle upper bound only; do not "
                "compare against single-system baselines.")
        if n_appended == 0 and pre_k >= n_clusters:
            payload["union_note"] = (
                f"The secondary pipeline added nothing: pre_k ({pre_k}) covers "
                f"all {n_clusters} clusters, so the primary list is already "
                f"exhaustive. Lower pre_k for the additive step to have any "
                f"effect.")
        return payload

    def query_annotation_only(self, text: str, top_k: int = 10,
                              with_genes: bool = False, **_ignored) -> dict:
        self._refresh_diagnostics(text)
        ranked = self._semantic_ranked(text, top_k)
        return self._format_results(ranked, with_genes=with_genes,
                                    mode="annotation", query=text)

    def discover(self, text: str, top_k: int = 5, lambda_sem: float = 0.5,
                 pre_k: int = 40, gamma: float = None,
                 with_modules: bool = True, **_ignored) -> dict:
        """
        Discovery mode.

        Per the manuscript, the language model receives the retrieved clusters
        with their gene-level evidence AND the outputs of the analytical
        modules (pathway scoring, ligand-receptor inference). Both are
        assembled here so the prompt carries the full evidence package.
        """
        self._refresh_diagnostics(text)
        auto_mode = classify_query(text, self.all_genes)

        # Route the fusion weight by query type, then fuse (never implicit).
        if auto_mode == "gene_list":
            lam = 0.0
        elif auto_mode == "ontology":
            lam = 1.0
        else:
            lam = float(lambda_sem)

        payload = self.query_hybrid(text, top_k=top_k, lambda_sem=lam,
                                    with_genes=True, pre_k=pre_k)
        payload["mode"] = "discovery"
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

        if with_modules:
            hits = [r["cluster_id"] for r in payload["results"]]
            try:
                pw = self.pathway()
                payload["pathway_context"] = [
                    p for p in pw.get("ranked", [])[:12]
                ]
                payload["pathway_caveat"] = pw.get("caveat")
            except Exception as e:
                payload["pathway_context_error"] = f"{type(e).__name__}: {e}"
            try:
                inter = self.interactions(max_results=40)
                rel = [it for it in inter.get("interactions", [])
                       if it["source"] in hits or it["target"] in hits][:15]
                payload["interaction_context"] = {
                    "n_total": inter.get("n_total"),
                    "top_involving_retrieved_clusters": rel,
                    "pathway_summary": inter.get("pathway_summary", [])[:8],
                }
                payload["interaction_caveat"] = inter.get("caveat")
            except Exception as e:
                payload["interaction_context_error"] = f"{type(e).__name__}: {e}"

        return payload

    def lambda_sweep(self, query: str, top_k: int = 5) -> dict:
        """
        Sweep the semantic/expression fusion weight and report, at each lambda,
        the fraction of query genes covered by the returned clusters' marker
        sets. Falls back to rank overlap when the query has no gene symbols.
        """
        lambdas = [round(x * 0.1, 1) for x in range(11)]
        query_genes = [g.upper() for g in extract_gene_names(query,
                                                             self.all_genes)]
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
                    coverages.append(len(baseline & set(cids))
                                     / max(len(baseline), 1))

        return {
            "query": query,
            "lambdas": lambdas,
            "coverages": coverages,
            "metric": ("query_gene_coverage" if query_genes
                       else "rank_overlap_vs_lambda0"),
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
                cond_props[cond] = {"total_cells": sub_total,
                                    "clusters": clusters}

            payload["condition_column"] = col
            payload["condition_proportions"] = cond_props

            if len(cond_props) == 2:
                a, b = list(cond_props.keys())
                fa = {c["cluster_id"]: c["fraction"]
                      for c in cond_props[a]["clusters"]}
                fb = {c["cluster_id"]: c["fraction"]
                      for c in cond_props[b]["clusters"]}
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
                             f"{caps['condition_values']}",
                    "available": caps["condition_values"]}

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
                "log2_fold_change": round(
                    math.log2((fa + 1e-6) / (fb + 1e-6)), 4),
                "condition_bias": (a if fa > 0.6 * (fa + fb) else
                                   (b if fb > 0.6 * (fa + fb) else "none")),
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
                upper_map = {}
                for i, v in enumerate(self._adata.var_names):
                    upper_map.setdefault(str(v).upper(), i)

                for g in genes:
                    idx = upper_map.get(g.upper())
                    if idx is None:
                        continue
                    vec = self._adata.X[:, idx]
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
            "condition_bias_threshold": 0.6,
            "caveat": "Cluster fold changes are compositional. "
                      "No formal differential-abundance test was run.",
            "gene_note": gene_note,
        }

    # --------------------------------------------------------
    # PATHWAY
    # --------------------------------------------------------

    METRIC_DESC = ("mean pct_in across pathway genes detected in the "
                   "cluster's DE profile; requires >= 3 detected genes for a "
                   "non-zero score; coverage = detected/pathway size")
    MIN_PATHWAY_GENES = 3

    def _score_pathway_in_cluster(self, cid: str, pathway_genes: List[str]):
        stats = self.gene_stats.get(str(cid), {})
        if not stats:
            return 0.0, [], 0.0
        upper = {g.upper(): s for g, s in stats.items()}
        hits = []
        for g in pathway_genes:
            s = upper.get(g.upper())
            if s is None:
                continue
            pin = _pct(s.get("pct_in"))
            pout = _pct(s.get("pct_out"))
            hits.append({
                "gene": g,
                "pct_in": round(pin, 4),
                "pct_out": round(pout, 4),
                "logfc": round(_f(s.get("logfc")), 4),
                "specificity": round(max(pin - pout, 0.0), 4),
            })
        coverage = len(hits) / max(len(pathway_genes), 1)
        if len(hits) < self.MIN_PATHWAY_GENES:
            return 0.0, hits, round(coverage, 4)
        score = sum(h["pct_in"] for h in hits) / len(hits)
        hits.sort(key=lambda h: -h["pct_in"])
        return score, hits, round(coverage, 4)

    def pathway(self, pathway_name: str = None) -> dict:
        caveat = ("Scores derive from per-cluster marker/DE statistics, not "
                  "per-cell expression. This is an enrichment proxy, not "
                  "AUCell or ssGSEA.")

        if pathway_name is None or str(pathway_name).lower() == "all":
            ranked = []
            for name, genes in PATHWAYS.items():
                per_cluster = []
                for cid in self.cluster_ids:
                    sc, hits, cov = self._score_pathway_in_cluster(str(cid),
                                                                   genes)
                    per_cluster.append((str(cid), sc, len(hits), cov))
                per_cluster.sort(key=lambda t: -t[1])
                top = per_cluster[0] if per_cluster else (None, 0.0, 0, 0.0)
                ranked.append({
                    "pathway": name,
                    "category": PATHWAY_TO_CATEGORY.get(name, "uncategorized"),
                    "top_cluster": top[0],
                    "top_score": round(top[1], 4),
                    "n_genes_detected": top[2],
                    "coverage": top[3],
                    "pathway_size": len(genes),
                })
            ranked.sort(key=lambda r: -r["top_score"])
            return {
                "query": "all pathways",
                "mode": "pathway",
                "metric": self.METRIC_DESC,
                "n_pathways": len(PATHWAYS),
                "categories": {k: len(v)
                               for k, v in PATHWAY_CATEGORIES.items()},
                "ranked": ranked,
                "caveat": caveat,
            }

        key = str(pathway_name).strip().lower().replace(" ", "_").replace("-", "_")
        key = PATHWAY_ALIASES.get(key, key)
        if key not in PATHWAYS:
            # word-overlap fuzzy matching, ignoring generic words
            toks = {t for t in key.split("_")
                    if t and t not in PATHWAY_STOPWORDS}
            scored = []
            for p in PATHWAYS:
                ptoks = {t for t in p.split("_")
                         if t and t not in PATHWAY_STOPWORDS}
                union = toks | ptoks
                overlap = len(toks & ptoks) / max(len(union), 1)
                if key in p or p in key:
                    overlap += 0.5
                if overlap > 0:
                    scored.append((overlap, p))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= 0.3:
                key = scored[0][1]
            else:
                return {
                    "error": f"Unknown pathway '{pathway_name}'",
                    "available": sorted(PATHWAYS.keys()),
                }

        genes = PATHWAYS[key]
        scores = []
        for cid in self.cluster_ids:
            sc, hits, cov = self._score_pathway_in_cluster(str(cid), genes)
            scores.append({
                "cluster_id": str(cid),
                "score": round(sc, 4),
                "n_genes_detected": len(hits),
                "coverage": cov,
                "top_genes": hits[:8],
            })
        scores.sort(key=lambda r: -r["score"])

        return {
            "query": key,
            "mode": "pathway",
            "pathway": key,
            "category": PATHWAY_TO_CATEGORY.get(key, "uncategorized"),
            "metric": self.METRIC_DESC,
            "min_genes_required": self.MIN_PATHWAY_GENES,
            "genes_in_pathway": genes,
            "scores": scores,
            "caveat": caveat,
        }

    # --------------------------------------------------------
    # INTERACTIONS
    # --------------------------------------------------------

    def _expressed_pct(self, cid: str) -> Dict[str, float]:
        """Gene -> pct_in for every gene present in the cluster's DE profile."""
        return {g.upper(): _pct(s.get("pct_in"))
                for g, s in self.gene_stats.get(str(cid), {}).items()}

    def interactions(self, source: str = None, target: str = None,
                     max_results: int = 200,
                     min_ligand_pct: float = 0.10,
                     min_receptor_pct: float = 0.05,
                     include_self: bool = False) -> dict:
        """
        Ligand-receptor scoring as described in the manuscript:

            s_ij = pct_in(ligand, c_i) * pct_in(receptor, c_j)

        filtered at ligand >= 0.10 and receptor >= 0.05, self-interactions
        excluded by default.
        """
        expressed = {str(c): self._expressed_pct(str(c))
                     for c in self.cluster_ids}

        src_list = ([self._cid(source)] if source
                    else [str(c) for c in self.cluster_ids])
        tgt_list = ([self._cid(target)] if target
                    else [str(c) for c in self.cluster_ids])

        found = []
        for lig, rec, pw in LR_PAIRS:
            for s in src_list:
                pl = expressed.get(s, {}).get(lig, 0.0)
                if pl < min_ligand_pct:
                    continue
                for t in tgt_list:
                    if t == s and not include_self:
                        continue
                    pr = expressed.get(t, {}).get(rec, 0.0)
                    if pr < min_receptor_pct:
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
            "score_definition": "pct_in(ligand, source) * pct_in(receptor, target)",
            "thresholds": {"min_ligand_pct": min_ligand_pct,
                           "min_receptor_pct": min_receptor_pct,
                           "self_interactions": include_self},
            "caveat": "Predicted from co-enrichment of ligand and receptor in "
                      "per-cluster marker statistics. Not a validated "
                      "interaction inference method; no null model or "
                      "permutation testing was performed.",
        }
        if n_total == 0:
            payload["note"] = ("No ligand-receptor pairs from the built-in "
                               "table passed the expression thresholds. Try "
                               "engine.interactions(min_ligand_pct=0.05, "
                               "min_receptor_pct=0.02) or include_self=True.")
        return payload
