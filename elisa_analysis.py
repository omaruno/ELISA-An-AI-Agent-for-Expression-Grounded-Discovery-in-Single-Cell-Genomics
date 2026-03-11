#!/usr/bin/env python
# ============================================================
# ELISA – Analysis Module 
# ============================================================

import os, json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# ============================================================
# BUILT-IN LIGAND-RECEPTOR DATABASE (expanded)
# ============================================================

LR_DATABASE = [
    # --- IFN-gamma axis (paper Fig 4E: CD8+ T cell IFNG -> multiple targets) ---
    ("IFNG", "IFNGR1", "IFN-gamma signaling"),
    ("IFNG", "IFNGR2", "IFN-gamma signaling"),
    ("CALR",   "LRP1",   "Phagocytosis / Calreticulin"),
    ("HLA-E",  "KLRC1",  "HLA-E / NKG2A checkpoint"),
    ("HLA-E",  "KLRD1",  "HLA-E / NKG2A checkpoint"),
    ("HLA-E",  "KLRC2",  "HLA-E / NKG2A checkpoint"),
    ("CXCL10", "CXCR3",  "Chemokine signaling"),
    ("F2",     "F2R",    "Coagulation signaling"),
    ("CCL5",   "CCR5",   "Chemokine signaling"),

    # --- Type I IFN ---
    ("IFNB1", "IFNAR1", "Type I IFN signaling"),
    ("IFNB1", "IFNAR2", "Type I IFN signaling"),
    ("IFNA1", "IFNAR1", "Type I IFN signaling"),
    ("IFNA1", "IFNAR2", "Type I IFN signaling"),

    # --- Calreticulin axis (paper Fig 4E: CALR on T cells -> LRP1 on macrophages) ---
    ("CALR", "LRP1", "Phagocytosis / Calreticulin"),

    # --- HLA-E / NKG2A checkpoint (paper Fig 4F: key finding) ---
    ("HLA-E", "KLRC1", "HLA-E / NKG2A checkpoint"),   # NKG2A (inhibitory)
    ("HLA-E", "KLRD1", "HLA-E / NKG2A checkpoint"),   # CD94
    ("HLA-E", "KLRC2", "HLA-E / NKG2A checkpoint"),   # NKG2C (activating)
    ("HLA-E", "KLRC3", "HLA-E / NKG2A checkpoint"),   # NKG2E
    ("HLA-E", "KLRK1", "HLA-E / NKG2A checkpoint"),   # NKG2D
    ("HLA-E", "CD8A",  "HLA-E / CD8 recognition"),
    ("HLA-E", "CD8B",  "HLA-E / CD8 recognition"),

    # --- GNAI2 interactions (paper Fig 4E-F: GNAI2 from lymphocytes
    #     binds multiple receptors on other cells) ---
    ("GNAI2", "CXCR3", "Chemokine / GNAI2 signaling"),
    ("GNAI2", "F2R",   "Chemokine / GNAI2 signaling"),
    ("GNAI2", "S1PR4", "Chemokine / GNAI2 signaling"),
    ("GNAI2", "EDNRA", "Chemokine / GNAI2 signaling"),
    ("GNAI2", "EGFR",  "Chemokine / GNAI2 signaling"),
    ("GNAI2", "F2R1",  "Chemokine / GNAI2 signaling"),

    # --- Chemokine signaling (paper: CXCL10-CXCR3 and CCL5-CCR5) ---
    ("CXCL10", "CXCR3", "Chemokine signaling"),
    ("CXCL9",  "CXCR3", "Chemokine signaling"),
    ("CXCL11", "CXCR3", "Chemokine signaling"),
    ("CCL5",   "CCR5",  "Chemokine signaling"),
    ("CCL5",   "CCR1",  "Chemokine signaling"),
    ("CCL3",   "CCR5",  "Chemokine signaling"),

    # --- Coagulation / thrombin axis (paper Fig 4F: F2-F2R interaction) ---
    ("F2",   "F2R",   "Coagulation signaling"),

    # --- VEGF signaling (paper: VEGFA elevated, endothelial remodeling) ---
    ("VEGFA", "KDR",   "VEGF signaling"),
    ("VEGFA", "FLT1",  "VEGF signaling"),
    ("VEGFA", "NRP1",  "VEGF signaling"),
    ("VEGFB", "FLT1",  "VEGF signaling"),
    ("VEGFC", "FLT4",  "VEGF signaling"),

    # --- PDGF signaling (paper: PDGFRB in B cells) ---
    ("PDGFA", "PDGFRA", "PDGF signaling"),
    ("PDGFB", "PDGFRB", "PDGF signaling"),
    ("PDGFD", "PDGFRB", "PDGF signaling"),

    # --- T cell co-stimulation / antigen presentation ---
    ("HLA-DPB1", "CD4",  "Antigen presentation"),
    ("HLA-DPA1", "CD4",  "Antigen presentation"),
    ("HLA-DRB1", "CD4",  "Antigen presentation"),
    ("CD80",     "CD28", "T cell co-stimulation"),
    ("CD86",     "CD28", "T cell co-stimulation"),
    ("CD80",     "CTLA4", "Immune checkpoint"),
    ("CD86",     "CTLA4", "Immune checkpoint"),

    # --- LTB signaling (paper: upregulated in B cells) ---
    ("LTB",  "TNFRSF14", "Lymphotoxin signaling"),
    ("LTB",  "LTBR",     "Lymphotoxin signaling"),

    # --- CD81 co-stimulation (paper: upregulated on T and B cells) ---
    ("CD81", "CD3G",  "T cell co-stimulation"),
    ("CD81", "CD19",  "B cell co-stimulation"),

    # --- VIM-CD44 (paper Fig 4E-F) ---
    ("VIM",  "CD44",  "Cell adhesion"),

    # --- FASLG-FAS (paper Fig 4E) ---
    ("FASLG", "FAS",  "Apoptosis signaling"),

    # --- IL-6 signaling (paper: IL-6 elevated in CF sputum/BAL,
    #     induces HLA-E expression) ---
    ("IL6",  "IL6R",   "IL-6 signaling"),
    ("IL6",  "IL6ST",  "IL-6 signaling"),

    # --- Ionocyte-related (paper: unique CF interactions from ionocytes) ---
    ("CFTR",  "SLC26A9", "CFTR / ion transport"),
    # --- Cytokine signaling ---
    ("IFNG", "IFNGR1", "IFN-gamma signaling"),
    ("IFNG", "IFNGR2", "IFN-gamma signaling"),
    ("IFNB1", "IFNAR1", "Type I IFN signaling"),
    ("IFNB1", "IFNAR2", "Type I IFN signaling"),
    ("IFNA1", "IFNAR1", "Type I IFN signaling"),
    ("IL1B", "IL1R1", "IL-1 signaling"),
    ("IL1B", "IL1R2", "IL-1 signaling"),
    ("IL6", "IL6R", "IL-6 signaling"),
    ("IL6", "IL6ST", "IL-6 signaling"),
    ("IL10", "IL10RA", "IL-10 signaling"),
    ("IL10", "IL10RB", "IL-10 signaling"),
    ("IL4", "IL4R", "IL-4/IL-13 signaling"),
    ("IL13", "IL4R", "IL-4/IL-13 signaling"),
    ("IL13", "IL13RA1", "IL-4/IL-13 signaling"),
    ("IL17A", "IL17RA", "IL-17 signaling"),
    ("IL17A", "IL17RC", "IL-17 signaling"),
    ("IL2", "IL2RA", "IL-2 signaling"),
    ("IL2", "IL2RB", "IL-2 signaling"),
    ("IL15", "IL15RA", "IL-15 signaling"),
    ("IL15", "IL2RB", "IL-15 signaling"),
    ("IL33", "IL1RL1", "IL-33/ST2 signaling"),
    ("TSLP", "CRLF2", "TSLP signaling"),
    ("IL18", "IL18R1", "IL-18 signaling"),
    ("IL12A", "IL12RB1", "IL-12 signaling"),
    ("IL12B", "IL12RB2", "IL-12 signaling"),
    ("IL21", "IL21R", "IL-21 signaling"),
    ("IL22", "IL22RA1", "IL-22 signaling"),
    ("OSM", "OSMR", "Oncostatin M signaling"),
    ("LIF", "LIFR", "LIF signaling"),

    # --- Chemokine signaling ---
    ("CXCL10", "CXCR3", "CXCL10-CXCR3 axis"),
    ("CXCL9", "CXCR3", "CXCL9-CXCR3 axis"),
    ("CXCL11", "CXCR3", "CXCL11-CXCR3 axis"),
    ("CXCL12", "CXCR4", "CXCL12-CXCR4 axis"),
    ("CCL2", "CCR2", "CCL2-CCR2 monocyte recruitment"),
    ("CCL5", "CCR5", "CCL5-CCR5 T cell recruitment"),
    ("CCL3", "CCR1", "CCL3-CCR1 signaling"),
    ("CCL4", "CCR5", "CCL4-CCR5 signaling"),
    ("CCL19", "CCR7", "Lymph node homing"),
    ("CCL21", "CCR7", "Lymph node homing"),
    ("CX3CL1", "CX3CR1", "Fractalkine signaling"),
    ("CCL20", "CCR6", "CCL20-CCR6 Th17 recruitment"),
    ("CXCL1", "CXCR2", "Neutrophil recruitment"),
    ("CXCL2", "CXCR2", "Neutrophil recruitment"),
    ("CXCL3", "CXCR2", "CXCL3-CXCR2 chemokine signaling"),
    ("CXCL8", "CXCR1", "Neutrophil recruitment"),
    ("CXCL8", "CXCR2", "Neutrophil recruitment"),
    ("CXCL13", "CXCR5", "B cell recruitment"),
    ("CXCL16", "CXCR6", "NKT/T cell recruitment"),

    # --- TNF superfamily ---
    ("TNF", "TNFRSF1A", "TNF signaling"),
    ("TNF", "TNFRSF1B", "TNF signaling"),
    ("TNFSF10", "TNFRSF10A", "TRAIL signaling"),
    ("TNFSF10", "TNFRSF10B", "TRAIL signaling"),
    ("FASLG", "FAS", "Fas/FasL apoptosis"),
    ("CD40LG", "CD40", "CD40 co-stimulation"),
    ("TNFSF4", "TNFRSF4", "OX40 co-stimulation"),
    ("TNFSF9", "TNFRSF9", "4-1BB co-stimulation"),
    ("BAFF", "TNFRSF13B", "B cell survival"),
    ("BAFF", "TNFRSF13C", "B cell survival"),
    ("RANKL", "RANK", "RANKL signaling"),
    ("LIGHT", "TNFRSF14", "LIGHT-HVEM signaling"),

    # --- Antigen presentation / immune checkpoint ---
    ("HLA-A", "CD8A", "MHC-I antigen presentation"),
    ("HLA-B", "CD8A", "MHC-I antigen presentation"),
    ("HLA-C", "CD8A", "MHC-I antigen presentation"),
    ("HLA-E", "KLRC1", "HLA-E/NKG2A immune checkpoint"),
    ("HLA-E", "KLRC2", "HLA-E/NKG2C activation"),
    ("HLA-E", "KLRD1", "HLA-E/CD94 immune checkpoint"),
    ("HLA-DRA", "CD4", "MHC-II antigen presentation"),
    ("HLA-DRB1", "CD4", "MHC-II antigen presentation"),
    ("HLA-DPA1", "CD4", "MHC-II antigen presentation"),
    ("HLA-DPB1", "CD4", "MHC-II antigen presentation"),
    ("B2M", "KLRD1", "MHC-I/CD94 recognition"),
    ("CD274", "PDCD1", "PD-L1/PD-1 checkpoint"),
    ("PDCD1LG2", "PDCD1", "PD-L2/PD-1 checkpoint"),
    ("CD80", "CD28", "B7-CD28 co-stimulation"),
    ("CD80", "CTLA4", "B7-CTLA4 inhibition"),
    ("CD86", "CD28", "B7-CD28 co-stimulation"),
    ("CD86", "CTLA4", "B7-CTLA4 inhibition"),
    ("LGALS9", "HAVCR2", "Galectin-9/TIM-3 checkpoint"),
    ("HMGB1", "HAVCR2", "HMGB1/TIM-3 signaling"),
    ("CD47", "SIRPA", "CD47/SIRPa don't eat me"),
    ("CALR", "LRP1", "Calreticulin/LRP1 eat me"),
    ("MIF", "CD74", "MIF-CD74 signaling"),

    # --- Growth factors ---
    ("VEGFA", "FLT1", "VEGF angiogenesis"),
    ("VEGFA", "KDR", "VEGF angiogenesis"),
    ("EGF", "EGFR", "EGF signaling"),
    ("TGFB1", "TGFBR1", "TGF-beta signaling"),
    ("TGFB1", "TGFBR2", "TGF-beta signaling"),
    ("TGFB2", "TGFBR1", "TGF-beta signaling"),
    ("TGFB3", "TGFBR2", "TGF-beta signaling"),
    ("PDGFA", "PDGFRA", "PDGF signaling"),
    ("PDGFB", "PDGFRB", "PDGF signaling"),
    ("FGF2", "FGFR1", "FGF signaling"),
    ("FGF7", "FGFR2", "FGF signaling (KGF)"),
    ("FGF10", "FGFR2", "FGF signaling (lung branching)"),
    ("HGF", "MET", "HGF/c-MET signaling"),
    ("IGF1", "IGF1R", "IGF signaling"),
    ("BMP2", "BMPR1A", "BMP signaling"),
    ("BMP4", "BMPR2", "BMP signaling"),

    # --- Wnt signaling ---
    ("WNT5A", "FZD5", "Wnt signaling"),
    ("WNT3A", "FZD1", "Wnt signaling"),
    ("WNT7B", "FZD4", "Wnt signaling"),
    ("WNT2", "FZD3", "Wnt signaling (mesenchymal)"),
    ("WNT5A", "ROR2", "Wnt/PCP signaling"),
    ("RSPO1", "LGR5", "R-spondin/Wnt amplification"),
    ("RSPO2", "LGR5", "R-spondin/Wnt amplification"),
    ("WIF1", "WNT5A", "Wnt inhibition (WIF1)"),

    # --- Notch signaling ---
    ("DLL1", "NOTCH1", "Notch signaling"),
    ("DLL4", "NOTCH1", "Notch signaling"),
    ("JAG1", "NOTCH2", "Notch signaling"),
    ("JAG2", "NOTCH1", "Notch signaling"),

    # --- Hedgehog ---
    ("SHH", "PTCH1", "Hedgehog signaling"),

    # --- Adhesion / migration ---
    ("ICAM1", "ITGAL", "LFA-1/ICAM adhesion"),
    ("VCAM1", "ITGA4", "VLA-4/VCAM adhesion"),
    ("CDH1", "CDH1", "E-cadherin homophilic"),
    ("FN1", "ITGB1", "Fibronectin/integrin adhesion"),
    ("COL1A1", "ITGA2", "Collagen/integrin adhesion"),
    ("SEMA3A", "NRP1", "Semaphorin guidance"),
    ("EFNA1", "EPHA2", "Ephrin signaling"),

    # --- Complement ---
    ("C3", "C3AR1", "Complement C3a signaling"),
    ("C5", "C5AR1", "Complement C5a signaling"),

    # --- Alarmin / damage ---
    ("HMGB1", "TLR4", "DAMP signaling"),
    ("HMGB1", "AGER", "HMGB1/RAGE signaling"),
    ("S100A8", "TLR4", "S100/TLR4 alarmin"),
    ("S100A9", "TLR4", "S100/TLR4 alarmin"),
    ("IL1A", "IL1R1", "IL-1 alarmin"),
    ("ANXA1", "FPR2", "Annexin/FPR2 resolution"),

    # --- Coagulation / thrombin ---
    ("F2", "F2R", "Thrombin/PAR1 signaling"),
    ("PROS1", "AXL", "Protein S/AXL signaling"),
    ("GAS6", "AXL", "Gas6/AXL signaling"),
    ("GAS6", "TYRO3", "Gas6/Tyro3 signaling"),

    # =========== Lung epithelial & organoid signals ===========
    ("SFTPD", "TLR4", "Surfactant innate immune"),
    ("SFTPA1", "TLR2", "SP-A innate immune"),
    ("SFTPA2", "TLR2", "SP-A innate immune"),
    ("BMP4", "BMPR1A", "BMP4 AT1 differentiation"),
    ("NRG1", "ERBB3", "Neuregulin/ErbB3 signaling"),
    ("NRG1", "ERBB4", "Neuregulin/ErbB4 signaling"),
    ("AREG", "EGFR", "Amphiregulin/EGFR signaling"),
    ("EREG", "EGFR", "Epiregulin/EGFR signaling"),
    ("RSPO3", "LGR4", "R-spondin3/LGR4 stem cell"),
    ("WNT3", "FZD7", "Wnt3/FZD7 stem maintenance"),
    ("NOGGIN", "BMPR1A", "Noggin BMP antagonism"),
    ("CHRD", "BMP4", "Chordin BMP antagonism"),
    ("DKK1", "LRP6", "Dickkopf Wnt inhibition"),
    ("DLL3", "NOTCH1", "DLL3/Notch NE specification"),
    ("DLL3", "NOTCH2", "DLL3/Notch NE specification"),
    ("GRP", "GRPR", "GRP/GRPR neuroendocrine"),
    ("NMB", "NMBR", "Neuromedin B signaling"),
    ("ATP", "P2RY2", "P2RY2 surfactant secretion"),
    ("UTP", "P2RY2", "P2RY2 surfactant secretion"),

    # =========== Neuroscience / FCD / epilepsy signals ===========
    ("C1QA", "C3AR1", "Complement microglia signaling"),
    ("C1QB", "C3AR1", "Complement microglia signaling"),
    ("C1QC", "C3AR1", "Complement microglia signaling"),
    ("C3", "ITGAM", "Complement/CR3 phagocytosis"),
    ("C1QA", "CR1", "C1q-CR1 synaptic tagging"),
    ("C1QB", "ITGAM", "C1q-CR3 microglial phagocytosis"),
    ("CX3CL1", "CX3CR1", "Fractalkine neuron-microglia"),
    ("TREM2", "TYROBP", "TREM2/DAP12 microglia activation"),
    ("CSF1", "CSF1R", "CSF1/CSF1R microglia survival"),
    ("IL34", "CSF1R", "IL34/CSF1R microglia maintenance"),
    ("CD74", "MIF", "CD74/MIF microglia signaling"),
    ("CD74", "APP", "CD74/APP neurodegeneration"),
    ("BDNF", "NTRK2", "BDNF/TrkB neurotrophic"),
    ("NTF3", "NTRK3", "NT3/TrkC neurotrophic"),
    ("GDNF", "GFRA1", "GDNF neuroprotection"),
    ("NRTN", "GFRA2", "Neurturin signaling"),
    ("TNFSF18", "TNFRSF18", "GITRL/GITR immune activation"),

    # =========== v3: Glutamate / GABA / OPC / Endothelial ===========
    ("SLC17A7", "GRIA1", "VGLUT1-AMPA glutamatergic synapse"),
    ("SLC17A7", "GRIA2", "VGLUT1-AMPA glutamatergic synapse"),
    ("SLC17A7", "GRIN1", "VGLUT1-NMDA glutamatergic synapse"),
    ("SLC17A7", "GRIN2A", "VGLUT1-NMDA glutamatergic synapse"),
    ("SLC17A6", "GRIA1", "VGLUT2-AMPA glutamatergic synapse"),
    ("GAD1", "GABRA1", "GAD67-GABAA GABAergic synapse"),
    ("GAD2", "GABRA1", "GAD65-GABAA GABAergic synapse"),
    ("GAD1", "GABRG2", "GAD67-GABAA gamma2 synapse"),
    ("GAD2", "GABRG2", "GAD65-GABAA gamma2 synapse"),
    ("GAD2", "GABRD", "GAD65-GABAA delta synapse"),
    ("PDGFA", "PDGFRA", "PDGF-A OPC proliferation"),
    ("MAG", "RTN4R", "MAG/NogoR myelination"),
    ("NES", "ITGB1", "Nestin progenitor adhesion"),
    ("POU5F1", "NANOG", "Oct4-Nanog pluripotency"),
    ("SOX2", "FGFR1", "SOX2-FGF progenitor maintenance"),
    ("PECAM1", "PECAM1", "PECAM1 homophilic endothelial"),
    ("CLDN5", "CLDN5", "Claudin-5 tight junction BBB"),
    ("FLT1", "VEGFA", "VEGFR1 endothelial signaling"),
    # =========== Neuroblastoma / ErbB / TME (Yu et al. 2025) ===========
    ("HBEGF", "ERBB4", "HB-EGF/ERBB4 neuroblastoma signaling"),
    ("HBEGF", "EGFR", "HB-EGF/EGFR signaling"),
    ("HBEGF", "CD9", "HB-EGF/CD9 signaling"),
    ("TGFA", "ERBB4", "TGFa/ERBB4 signaling"),
    ("TGFA", "EGFR", "TGFa/EGFR signaling"),
    ("EREG", "ERBB4", "Epiregulin/ERBB4 signaling"),
    ("EREG", "EGFR", "Epiregulin/EGFR signaling"),
    ("AREG", "ERBB4", "Amphiregulin/ERBB4 signaling"),
    ("ICAM1", "EGFR", "ICAM1/EGFR signaling"),
    ("DCN", "ERBB4", "Decorin/ERBB4 signaling"),
    ("VCAN", "EGFR", "Versican/EGFR adhesion-signaling"),
    ("VCAN", "ITGB1", "Versican/integrin adhesion"),
    ("THBS1", "CD47", "Thrombospondin/CD47 dont-eat-me"),
    ("THBS1", "ITGB1", "Thrombospondin/integrin adhesion"),
    ("THBS1", "ITGA3", "Thrombospondin/ITGA3 adhesion"),
    ("THBS1", "LRP5", "Thrombospondin/LRP5 signaling"),
    ("THBS1", "ITGA2B", "Thrombospondin/ITGA2B adhesion"),
    ("APOE", "LDLR", "ApoE/LDLR lipid transport"),
    ("APOE", "VLDLR", "ApoE/VLDLR lipid transport"),
    ("APOE", "SCARB1", "ApoE/SCARB1 lipid uptake"),
    ("LPL", "VLDLR", "Lipoprotein lipase/VLDLR metabolism"),
    ("TFPI", "VLDLR", "TFPI/VLDLR coagulation"),
    ("VEGFA", "GPC1", "VEGFA/Glypican-1 angiogenesis"),
    ("SEMA3A", "NRP1", "Semaphorin3A/NRP1 axon guidance"),
    ("SEMA3C", "PLXND1", "Semaphorin3C/PlexinD1 guidance"),
    ("SEMA3D", "NRP1", "Semaphorin3D/NRP1 guidance"),
    ("INHBA", "ACVR1B", "Inhibin/Activin signaling"),
    ("INHBA", "ACVR2B", "Inhibin/Activin signaling"),
    ("OSM", "OSMR", "Oncostatin M macrophage signaling"),
    ("NAMPT", "INSR", "Visfatin/insulin receptor signaling"),
    ("GAL", "GALR1", "Galanin neuropeptide signaling"),
    ("NPY", "NPFFR2", "NPY/NPFFR2 neuropeptide signaling"),
    ("CALCA", "RAMP1", "CGRP/RAMP1 neuropeptide signaling"),
    ("IL16", "KCND2", "IL16/KCND2 signaling"),
    ("SYTL3", "NRXN1", "Synaptotagmin/Neurexin synapse"),
    ("GNAI2", "ADCY1", "GNAI2/Adenylyl cyclase signaling"),
    ("GNAI2", "IGF1R", "GNAI2/IGF1R signaling"),
    ("GNAI2", "DRD2", "GNAI2/DRD2 dopamine signaling"),
    ("GNAI2", "OPRM1", "GNAI2/OPRM1 opioid signaling"),
    ("GZMB", "CHRM3", "GZMB/CHRM3 cytotoxicity"),
    ("F13A1", "ITGB1", "Factor XIII/integrin tissue repair"),
    ("COL1A1", "ITGA11", "Collagen/ITGA11 fibroblast adhesion"),
    ("COL1A1", "DDR2", "Collagen/DDR2 receptor signaling"),
    ("COL3A1", "DDR2", "Collagen III/DDR2 signaling"),
    ("COL1A1", "CD44", "Collagen/CD44 adhesion"),
    ("FN1", "ITGA8", "Fibronectin/ITGA8 adhesion"),
    ("FN1", "ITGA4", "Fibronectin/ITGA4 migration"),
    ("FN1", "ITGA9", "Fibronectin/ITGA9 adhesion"),
    ("FN1", "PLAUR", "Fibronectin/uPAR migration"),
    ("IGF1", "IGF1R", "IGF1/IGF1R neuroblastoma growth"),
    ("IL18", "IL18R1", "IL18 pro-inflammatory macrophage"),
    ("IL18", "IL1RAPL1", "IL18/IL1RAPL1 signaling"),
    ("CD274", "PDCD1", "PD-L1/PD-1 neuroblastoma immune evasion"),
    ("NECTIN2", "TIGIT", "NECTIN2/TIGIT immune checkpoint"),
        # =========== ICB / Immune Checkpoint (Gondal et al. 2025) ===========
    ("CD274", "PDCD1", "PD-L1/PD-1 immune checkpoint"),
    ("PDCD1LG2", "PDCD1", "PD-L2/PD-1 immune checkpoint"),
    ("NECTIN2", "TIGIT", "NECTIN2/TIGIT immune checkpoint"),
    ("NECTIN2", "CD226", "NECTIN2/DNAM-1 co-stimulation"),
    ("PVR", "TIGIT", "PVR/TIGIT immune checkpoint"),
    ("PVR", "CD226", "PVR/DNAM-1 co-stimulation"),
    ("LGALS9", "HAVCR2", "Galectin-9/TIM-3 checkpoint"),
    ("CXCL13", "CXCR5", "CXCL13/CXCR5 Tfh-B cell recruitment"),
    ("IL2", "IL2RA", "IL-2/CD25 Treg expansion"),
    ("IL2", "IL2RB", "IL-2/IL2RB T cell signaling"),
    ("IL21", "IL21R", "IL-21/IL21R B cell differentiation"),
    ("TNFSF4", "TNFRSF4", "OX40L/OX40 co-stimulation"),
    ("TNFSF9", "TNFRSF9", "4-1BBL/4-1BB co-stimulation"),
    ("CD40LG", "CD40", "CD40L/CD40 B-DC co-stimulation"),
    ("CXCL9", "CXCR3", "CXCL9/CXCR3 effector T recruitment"),
    ("CXCL10", "CXCR3", "CXCL10/CXCR3 effector T recruitment"),
    ("CCL5", "CCR5", "CCL5/CCR5 T cell recruitment ICB"),
    ("CCL4", "CCR5", "CCL4/CCR5 macrophage-T signaling"),
    ("IFNG", "IFNGR1", "IFNg/IFNGR1 anti-tumor immunity"),
    ("TNF", "TNFRSF1A", "TNF/TNFR1 anti-tumor signaling"),
    ("CD47", "SIRPA", "CD47/SIRPa dont-eat-me phagocytosis"),
    ("CALR", "LRP1", "Calreticulin/LRP1 eat-me phagocytosis"),
    ("MIF", "CD74", "MIF/CD74 macrophage signaling"),
    ("SPP1", "CD44", "Osteopontin/CD44 TAM signaling"),
    ("SPP1", "ITGAV", "Osteopontin/integrin TAM adhesion"),
    ("TGFB1", "TGFBR1", "TGFb1/TGFBR1 immunosuppression"),
    ("TGFB1", "TGFBR2", "TGFb1/TGFBR2 T cell exclusion"),
    ("IDO1", "AHR", "IDO1/AHR tryptophan immunosuppression"),
    ("VEGFA", "KDR", "VEGFA/VEGFR2 tumor angiogenesis ICB"),
    ("VEGFA", "NRP1", "VEGFA/NRP1 Treg signaling"),
    ("CSF1", "CSF1R", "CSF1/CSF1R macrophage recruitment"),
    ("IL34", "CSF1R", "IL34/CSF1R macrophage maintenance"),
    ("CCL2", "CCR2", "CCL2/CCR2 monocyte recruitment ICB"),
    ("CXCL12", "CXCR4", "CXCL12/CXCR4 T cell exclusion"),
    ("HLA-A", "CD8A", "MHC-I/CD8 antigen recognition"),
    ("HLA-B", "CD8A", "MHC-I/CD8 antigen recognition"),
    ("HLA-DRA", "CD4", "MHC-II/CD4 antigen presentation"),
    ("HLA-DRB1", "CD4", "MHC-II/CD4 antigen presentation"),
    ("EPCAM", "CD44", "EpCAM/CD44 cancer stem signaling"),
    ("FN1", "ITGB1", "Fibronectin/integrin CAF-tumor adhesion"),
    ("COL1A1", "ITGA2", "Collagen/integrin CAF-tumor adhesion"),
    ("COL1A1", "DDR2", "Collagen/DDR2 CAF signaling"),
]


# ============================================================
# BUILT-IN PATHWAY GENE SETS (expanded for lung + FCD biology)
# ============================================================

PATHWAY_GENESETS = {
        # --- EXISTING — merge these genes into your current entry ---

    "IFN-gamma signaling": [
        # Core axis (paper Fig 4E: IFNG->IFNGR1/2 from CD8+ T cells)
        "IFNG", "IFNGR1", "IFNGR2", "JAK1", "JAK2", "STAT1",
        "IRF1", "IRF2", "IRF8", "IRF9", "GBP1", "GBP2", "GBP4", "GBP5",
        "IDO1", "CIITA", "TAP1", "TAP2", "PSMB8", "PSMB9", "PSMB10",
        "HLA-A", "HLA-B", "HLA-C", "HLA-E", "B2M",
        "CXCL9", "CXCL10", "CXCL11",
        # Paper-specific additions (DEGs from CD8+ T cells)
        "GNAI2", "CD69", "CD81", "FOS", "JUND",
    ],

    "Type I IFN signaling": [
        # Paper: IFIT1, MX1, OAS2 upregulated in epithelial cells
        "IFNB1", "IFNAR1", "IFNAR2", "IFNA1", "IFNA2",
        "STAT1", "STAT2", "IRF3", "IRF7", "IRF9",
        "IFIT1", "IFIT2", "IFIT3", "IFIT5",
        "MX1", "MX2", "OAS1", "OAS2", "OAS3",
        "ISG15", "ISG20", "IFITM1", "IFITM2", "IFITM3",
        "IFI6", "IFI27", "IFI35", "IFI44", "IFI44L",
        "BST2", "RSAD2", "HERC5", "USP18",
    ],

    # --- NEW PATHWAYS — add these as new entries ---

    "T cell activation": [
        # Paper: CD8+ T cell DEGs (Fig 3C) and CD4+ T cell DEGs (Fig 3D)
        "CD69", "CD81", "CD3G", "CD3E", "CD3D",
        "CD8A", "CD8B", "CD4",
        "IFNG", "GZMB", "PRF1", "NKG7", "GNLY",
        "FOS", "FOSB", "JUN", "JUNB", "JUND",
        "GNAI2", "GNAS",
        "IL7R", "CD48", "KLF2", "ETS1",
        "TXNIP", "MAP2K2",
        # TCR signaling components
        "LCK", "ZAP70", "LAT", "SLP76", "ITK",
        "NFATC1", "NFATC2", "AP1",
        # Co-stimulation
        "CD28", "ICOS", "CTLA4", "PDCD1",
        # Calreticulin axis (paper Fig 4E)
        "CALR", "LRP1",
    ],

    "NK cell activity": [
        # Paper: NKG2A checkpoint, NK/ILC biology
        "GNLY", "NKG7", "PRF1", "GZMB", "GZMA", "GZMH", "GZMK",
        "KLRD1", "KLRC1", "KLRC2", "KLRC3", "KLRK1",
        "KLRB1", "KLRF1",
        "NCAM1", "FCGR3A",
        "HLA-E",  # ligand for NKG2A/C
        "IFNG", "TNF",
        # NK cell receptors
        "NCR1", "NCR2", "NCR3",
        "CD226", "TIGIT",
        # Paper-specific
        "CD94",  # alias for KLRD1
    ],

    "Antigen presentation": [
        # Paper: HLA-DPA1, HLA-DRB1 upregulated on ciliated cells and B cells
        # MHC class I
        "HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G",
        "B2M", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10",
        # MHC class II
        "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
        "HLA-DRA", "HLA-DRB1", "HLA-DRB5",
        "CD74", "CIITA",
        # Co-stimulation
        "CD80", "CD86", "CD40",
        # Antigen processing
        "CTSS", "CTSL", "CTSH", "LGMN",
        "PSME1", "PSME2",
    ],

    "Epithelial defense": [
        # Paper: basal cell DEGs, keratinization, DNA damage, IFN response
        # Keratinization (paper: CSTA, HSPB1 downregulated)
        "CSTA", "HSPB1", "KRT5", "KRT14", "KRT15", "TP63",
        # Barrier function
        "MUC5AC", "MUC5B", "MUC1", "MUC4",
        "SCGB1A1", "SCGB3A1",
        "PIGR", "SLPI", "WFDC2", "LYZ",
        # Antimicrobial peptides
        "DEFB1", "DEFB4A", "S100A7", "S100A8", "S100A9",
        # Ciliogenesis (paper: increased in CF)
        "FOXJ1", "DNAH5", "SYNE1", "SYNE2", "CAPS", "PIFO",
        # DNA damage repair (paper: upregulated in basal cells)
        "RAD50", "ERCC6", "ERCC8", "KDM1A", "KMT5A",
        # Interferon responsive (paper: upregulated across epithelial)
        "IFIT1", "MX1", "OAS2", "ISG15", "IFITM3",
    ],

    "Fibrosis": [
        # Paper: stromal changes, PDGFRB signaling in B cells
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1", "COL6A1",
        "FN1", "ELN", "LOX", "LOXL2",
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
        "SMAD2", "SMAD3", "SMAD4",
        "ACTA2", "VIM", "FAP",
        "PDGFA", "PDGFB", "PDGFRA", "PDGFRB",
        "CTGF", "TIMP1", "TIMP2", "TIMP3",
        "MMP2", "MMP9", "MMP14",
        "LUM", "DCN", "BGN", "SFRP2",
        # Paper-specific
        "HSPB1",
    ],

    "Angiogenesis": [
        # Paper: VEGFR signaling upregulated across cell types,
        # TXNIP/MAP2K2/ETS1 in lymphocytes, endothelial remodeling
        "VEGFA", "VEGFB", "VEGFC", "VEGFD",
        "FLT1", "KDR", "FLT4", "NRP1", "NRP2",
        "ANGPT1", "ANGPT2", "TEK", "TIE1",
        "PECAM1", "CDH5", "VWF", "ENG",
        "PLVAP", "ACKR1", "ERG",
        # Hypoxia response (paper: TXNIP, mucus plugs cause hypoxia)
        "HIF1A", "EPAS1", "ARNT",
        "TXNIP", "MAP2K2", "ETS1",
        # Downstream signaling
        "DLL4", "NOTCH1", "JAG1",
        "EFNB2", "EPHB4",
    ],

    # --- BONUS PATHWAYS (referenced in paper but not in ground truth) ---

    "NLRP3 inflammasome": [
        # Paper: TXNIP promotes NLRP3 assembly, proposed therapeutic target
        "NLRP3", "PYCARD", "CASP1", "IL1B", "IL18",
        "TXNIP", "NFKB1", "NFKB2", "RELA",
        "GSDMD", "HMGB1",
    ],

    "VEGFR signaling": [
        # Paper: VEGFA-VEGFR2 pathway enriched in CD8+ and CD4+ T cells
        "VEGFA", "KDR", "FLT1", "NRP1",
        "TXNIP", "MAP2K2", "ETS1",
        "SRC", "PIK3CA", "AKT1",
        "PLCG1", "PTK2",
    ],

    "BCR signaling": [
        # Paper: BCR downregulation in B cells (IGHG3, IGLC2 decreased)
        "CD79A", "CD79B", "SYK", "BTK", "BLNK",
        "PLCG2", "PIK3CD", "AKT1",
        "IGHG1", "IGHG3", "IGHD", "IGHA1",
        "IGLC1", "IGLC2", "IGLC3",
        "IGKC",
        "CSK", "CD81", "CD9", "CD19",
    ],

    "Chemokine signaling": [
        # Paper: GNAI2 upregulated, CXCR3/F2R/S1PR4 interactions
        "GNAI2", "GNAS",
        "CXCL9", "CXCL10", "CXCL11", "CXCR3",
        "CCL5", "CCR5",
        "CCL2", "CCR2",
        "CXCL12", "CXCR4",
        "F2R", "S1PR4", "S1PR1",
    ],
    # ===== EXISTING PATHWAYS (from v1) =====
    "IFN-gamma signaling": [
        "IFNG", "IFNGR1", "IFNGR2", "JAK1", "JAK2", "STAT1",
        "IRF1", "IRF2", "IRF8", "IRF9", "GBP1", "GBP2", "GBP4", "GBP5",
        "IDO1", "CIITA", "TAP1", "TAP2", "PSMB8", "PSMB9", "PSMB10",
        "HLA-A", "HLA-B", "HLA-C", "HLA-E", "B2M",
        "CXCL9", "CXCL10", "CXCL11",
    ],
    "Type I IFN signaling": [
        "IFNA1", "IFNB1", "IFNAR1", "IFNAR2", "JAK1", "TYK2",
        "STAT1", "STAT2", "IRF3", "IRF7", "IRF9",
        "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM3",
        "ISG15", "ISG20", "MX1", "MX2", "OAS1", "OAS2", "OAS3",
        "RSAD2", "BST2",
    ],
    "TNF/NF-kB signaling": [
        "TNF", "TNFRSF1A", "TNFRSF1B", "NFKB1", "NFKB2", "RELA", "RELB",
        "NFKBIA", "NFKBIB", "IKBKB", "IKBKG", "TRAF1", "TRAF2", "TRAF6",
        "BIRC2", "BIRC3", "BCL2L1", "TNFAIP3", "CXCL8", "CCL2", "CCL5",
        "ICAM1", "VCAM1", "SELE",
        "TNFSF18", "TNFAIP6",
    ],
    "TGF-beta signaling": [
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
        "SMAD2", "SMAD3", "SMAD4", "SMAD7",
        "COL1A1", "COL1A2", "COL3A1", "FN1", "ACTA2",
        "CTGF", "SERPINE1", "MMP2", "MMP9", "TIMP1",
    ],
    "T cell activation": [
        "CD3D", "CD3E", "CD3G", "CD28", "ICOS", "CD69", "CD44",
        "IL2RA", "IL2", "IFNG", "TNF", "GZMB", "GZMA", "PRF1",
        "CTLA4", "PDCD1", "LAG3", "HAVCR2", "TIGIT",
        "LCK", "ZAP70", "LAT", "NFATC1",
    ],
    "NK cell activity": [
        "KLRC1", "KLRD1", "KLRK1", "KLRB1",
        "NCR1", "NCR2", "NCR3", "FCGR3A",
        "GZMB", "GZMA", "GZMH", "PRF1", "GNLY",
        "IFNG", "TNF", "FASLG", "XCL1",
    ],
    "Epithelial defense": [
        "MUC5AC", "MUC5B", "MUC1", "MUC4",
        "SCGB1A1", "SCGB3A1", "SFTPB", "SFTPC", "SFTPD",
        "BPIFA1", "BPIFB1", "LTF", "LYZ", "SLPI", "WFDC2",
        "DEFB1", "S100A7", "S100A8", "S100A9",
        "CLDN1", "CLDN4", "OCLN", "TJP1",
    ],
    "Fibrosis": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL6A1",
        "FN1", "ACTA2", "FAP", "PDGFRA", "PDGFRB",
        "TGFB1", "CTGF", "SERPINE1", "LOX", "LOXL2",
        "MMP2", "MMP9", "MMP14", "TIMP1", "TIMP3",
        "VIM", "DES", "THY1",
        "DCN", "LUM",
    ],
    "Apoptosis": [
        "TP53", "BAX", "BAK1", "BCL2", "BCL2L1", "MCL1",
        "CASP3", "CASP7", "CASP8", "CASP9",
        "CYCS", "APAF1", "BID", "BIM",
        "FAS", "FASLG", "TNFRSF10A", "TNFRSF10B",
        "XIAP", "BIRC5",
    ],
    "Oxidative stress": [
        "SOD1", "SOD2", "CAT", "GPX1", "GPX4",
        "PRDX1", "PRDX2", "TXN", "TXNRD1", "TXNIP",
        "NFE2L2", "KEAP1", "HMOX1", "NQO1",
        "NOX2", "NOX4", "DUOX1", "DUOX2",
    ],
    "Glycolysis": [
        "HK1", "HK2", "GPI", "PFKFB3", "PFKL", "PFKM",
        "ALDOA", "TPI1", "GAPDH", "PGK1", "PGAM1",
        "ENO1", "PKM", "LDHA", "LDHB", "SLC2A1", "SLC2A3",
        "HIF1A",
    ],
    "Angiogenesis": [
        "VEGFA", "VEGFB", "VEGFC", "FLT1", "KDR", "FLT4",
        "NRP1", "NRP2", "ANGPT1", "ANGPT2", "TEK",
        "DLL4", "NOTCH1", "PECAM1", "CDH5", "VWF",
        "HIF1A", "EPAS1",
        "CLDN5",
    ],

    # ===== LUNG / ORGANOID PATHWAYS (from v2) =====
    "Surfactant metabolism": [
        "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "SFTPD", "SFTA3",
        "ABCA3", "LAMP3", "NAPSA", "NKX2-1",
        "LPCAT1", "SLC34A2", "DMBT1", "P2RY2",
        "CTSH", "CSF2", "GM-CSF",
        "FASN", "SCD", "PCYT1A", "CHKA",
    ],
    "Vesicle-mediated transport": [
        "RAB5A", "RAB7A", "RAB11A", "RAB27A", "RAB27B",
        "EEA1", "LAMP1", "LAMP2", "LAMP3",
        "SEC23A", "SEC24A", "SEC31A", "SAR1A",
        "COPA", "COPB1", "COPB2", "COPG1",
        "AP1S1", "AP2A1", "AP3B1",
        "VPS28", "VPS4A", "VPS4B", "TSG101",
        "SNX1", "SNX2", "SNX3", "SORT1",
        "MICALL1", "HPS6", "LAMTOR1",
    ],
    "Ubiquitin-mediated proteolysis": [
        "UBE2N", "UBE2I", "UBA2", "UBE2D1", "UBE2D3",
        "ITCH", "NEDD4", "NEDD4L", "SMURF1", "SMURF2",
        "TRIM21", "TRIM33", "TRIM65", "TRIM68",
        "CUL2", "CUL3", "CUL4A", "CUL5",
        "PIAS1", "PIAS2", "PIAS3",
        "USP7", "USP8", "USP14",
        "UBAP1", "OTUD6B", "BAP1",
        "RABGEF1", "SOCS3", "STAT5B",
        "UBA1", "UBA3", "UBC", "RBX1",
        "PSMA1", "PSMB8", "PSMD1",
    ],
    "Endosomal sorting": [
        "HRS", "STAM1", "STAM2",
        "TSG101", "VPS28", "VPS37A", "MVB12A",
        "VPS22", "VPS25", "VPS36",
        "VPS4A", "VPS4B", "CHMP2A", "CHMP3", "CHMP4B", "CHMP6",
        "ALIX",
        "EEA1", "RAB5A", "RAB7A", "RABGEF1",
        "MICALL1", "SNX1", "SNX3",
        "UBAP1", "USP8",
        "HPS6", "LAMTOR1",
        "NEDD4", "NEDD4L", "ITCH",
    ],
    "Lipid metabolism": [
        "FASN", "SCD", "ACACA", "ACACB",
        "LPCAT1", "LPCAT2",
        "ABCA3", "ABCA1", "ABCB8", "ABCC3", "ABCC6",
        "SLC27A1", "CD36", "FABP4", "FABP5",
        "PPARG", "PPARA", "RXRA",
        "FITM2", "DGAT1", "DGAT2", "LPIN1",
        "HMGCR", "HMGCS1", "FDFT1", "SQLE",
        "SLCO3A1", "SLC35F6",
    ],
    "Wnt signaling": [
        "WNT2", "WNT3A", "WNT5A", "WNT7B",
        "FZD1", "FZD3", "FZD4", "FZD5", "FZD7",
        "LRP5", "LRP6",
        "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B",
        "TCF7L2", "LEF1", "TCF7",
        "WIF1", "DKK1", "SFRP1", "SFRP2",
        "RSPO1", "RSPO2", "RSPO3", "LGR5",
        "DVL1", "DVL2", "DVL3",
        "ROR2", "RNF43", "ZNRF3",
    ],
    "Hippo signaling": [
        "YAP1", "WWTR1",
        "LATS1", "LATS2", "MST1", "MST2",
        "MOB1A", "MOB1B", "SAV1",
        "TEAD1", "TEAD2", "TEAD3", "TEAD4",
        "CYR61", "CTGF", "ANKRD1",
        "NF2", "AMOT", "AMOTL1", "AMOTL2",
        "CDH1", "FAT4", "DCHS1",
        "RASSF1", "KIBRA",
    ],
    "ER to Golgi vesicle-mediated transport": [
        "SEC23A", "SEC23B", "SEC24A", "SEC24B", "SEC24C", "SEC24D",
        "SEC31A", "SEC13", "SAR1A", "SAR1B",
        "COPA", "COPB1", "COPB2", "COPG1", "COPG2", "COPE",
        "ARF1", "ARF4",
        "ERGIC1", "ERGIC3", "LMAN1", "LMAN2",
        "TMED2", "TMED9", "TMED10",
        "SURF4", "CKAP4",
    ],
    "Antigen processing and presentation": [
        "HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G",
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
        "HLA-DQA1", "HLA-DQA2", "HLA-DQB1",
        "HLA-DMA", "HLA-DMB", "HLA-DOA", "HLA-DOB",
        "B2M", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10",
        "CD74", "CIITA", "CALR", "CANX",
        "CD86", "CD80",
    ],
    "ER stress response": [
        "XBP1", "ERN1", "ATF4", "ATF6", "EIF2AK3",
        "DDIT3", "HSPA5", "HSP90B1",
        "CALR", "CANX", "PDIA3", "PDIA4",
        "EDEM1", "DERL1", "SEL1L", "HRD1",
        "SOCS3", "GADD45A", "DNAJB9", "DNAJB11",
    ],
    "FGF signaling": [
        "FGF1", "FGF2", "FGF7", "FGF10", "FGF18",
        "FGFR1", "FGFR2", "FGFR3", "FGFR4",
        "FRS2", "GAB1", "GRB2", "SOS1",
        "HRAS", "KRAS", "NRAS",
        "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
        "ETV4", "ETV5", "SPRY1", "SPRY2", "SPRY4",
        "DUSP6",
    ],
    "Cytokine-cytokine receptor interaction": [
        "IL1A", "IL1B", "IL1R1", "IL1R2",
        "IL6", "IL6R", "IL6ST",
        "TNF", "TNFRSF1A", "TNFRSF1B",
        "IFNG", "IFNGR1", "IFNGR2",
        "CXCL1", "CXCL2", "CXCL3", "CXCL8", "CXCL9", "CXCL10",
        "CXCR2", "CXCR3", "CXCR4",
        "CCL2", "CCL3", "CCL4", "CCL5", "CCL20",
        "CCR1", "CCR2", "CCR5", "CCR6", "CCR7",
        "CX3CL1", "CX3CR1",
        "CSF1", "CSF1R", "CSF2", "CSF3", "CSF3R",
        "TGFB1", "TGFBR1", "TGFBR2",
        "CD86", "PIK3CG", "ITGB2",
        "CCL4L2", "TNFSF18",
    ],

    # =============================================================
    # NEW v3: 18 KEGG-named pathways for FCD paper coverage
    # =============================================================

    "mTOR signaling": [
        "MTOR", "RPTOR", "RICTOR", "MLST8",
        "AKT1", "AKT2", "AKT3",
        "TSC1", "TSC2", "RHEB",
        "RPS6KB1", "RPS6KB2", "RPS6",
        "EIF4EBP1", "EIF4E",
        "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2",
        "PTEN", "DEPDC5", "NPRL2", "NPRL3",
        "FKBP5", "DEPTOR", "SESN1", "SESN2",
        "SLC7A5", "SLC3A2", "SLC38A2",
        "HIF1A", "VEGFA", "LDHA", "STK11", "DDIT4",
    ],
    "PI3K-Akt signaling": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PIK3R2",
        "AKT1", "AKT2", "AKT3", "PTEN",
        "MTOR", "RPTOR", "RICTOR",
        "GSK3B", "FOXO1", "FOXO3",
        "MYC", "BCL2", "BAD",
        "TSC1", "TSC2", "DEPDC5",
        "RPS6KB1", "RPS6", "EIF4EBP1",
        "VEGFA", "IGF1R", "EGFR",
        "CDKN1A", "CDKN1B", "CCND1",
    ],
    "Parkinson disease": [
        "SNCA", "PARK7", "PINK1", "PRKN", "LRRK2",
        "UCHL1", "NEFL", "NEFM", "NEFH",
        "MAP2", "TUBB3", "RBFOX3",
        "COX5A", "COX6C", "COX7C",
        "NDUFA4", "ATP5PO", "ATP5F1C", "ATP5MC3",
        "CALM1", "CALM2", "CALM3",
        "GAPDH", "PDYN",
        "CASP3", "CASP9", "CYCS",
        "SLC18A2", "TH", "DDC",
    ],
    "Alzheimer disease": [
        "APP", "PSEN1", "PSEN2", "MAPT", "APOE", "TREM2",
        "NEFL", "NEFM", "NEFH",
        "MAP2", "RBFOX3", "TUBB3",
        "CALM1", "CALM2", "CALM3", "CALR",
        "GAPDH", "UCHL1",
        "COX5A", "COX6C", "COX7C",
        "NDUFA4", "ATP5PO", "ATP5F1C", "ATP5MC3",
        "CASP3", "CASP9",
        "GRIN1", "GRIN2A", "GRIN2B",
        "BACE1", "NCSTN", "APH1A",
    ],
    "Amyotrophic lateral sclerosis": [
        "SOD1", "TARDBP", "FUS", "C9orf72",
        "NEFL", "NEFM", "NEFH",
        "MAP2", "TUBB3", "RBFOX3",
        "UCHL1", "GAPDH",
        "CASP3", "CASP9",
        "GRIA1", "GRIA2", "GRIN1",
        "SLC1A2", "SLC1A3",
        "COX5A", "COX6C", "ATP5PO",
        "NDUFA4", "ATP5F1C", "ATP5MC3",
        "CYCS", "BAX", "BCL2",
    ],
    "Oxidative phosphorylation": [
        "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4",
        "NDUFB1", "NDUFB2",
        "COX5A", "COX5B", "COX6C", "COX7C",
        "ATP5F1A", "ATP5F1B", "ATP5F1C",
        "ATP5PO", "ATP5MC1", "ATP5MC3",
        "ATP6V0B", "ATP6V0C",
        "UQCRC1", "UQCRC2",
        "SDHB", "SDHC",
        "CYCS",
        "COX4I1", "COX6A1", "COX7A2",
    ],
    "Prion disease": [
        "PRNP", "NEFL", "NEFM", "NEFH",
        "MAP2", "TUBB3", "RBFOX3",
        "COX5A", "COX6C", "COX7C",
        "NDUFA4", "ATP5PO", "ATP5F1C", "ATP5MC3",
        "CALM1", "CALM2", "CALM3",
        "GAPDH", "UCHL1",
        "CASP3", "CASP8",
        "IL1B", "IL6",
        "C1QA", "C1QB", "C1QC",
    ],
    "Toll-like receptor signaling": [
        "TLR1", "TLR2", "TLR3", "TLR4", "TLR5",
        "TLR6", "TLR7", "TLR8", "TLR9",
        "MYD88", "TIRAP", "TRAF6", "IRAK1", "IRAK4",
        "NFKB1", "RELA",
        "IRF3", "IRF7", "IRF8",
        "IL1B", "IL6", "TNF", "CXCL8",
        "CCL2", "CCL3", "CCL4",
        "CD86", "CD80",
    ],
    "JAK-STAT signaling": [
        "JAK1", "JAK2", "JAK3", "TYK2",
        "STAT1", "STAT2", "STAT3", "STAT4",
        "STAT5A", "STAT5B", "STAT6",
        "IFNG", "IFNGR1", "IFNGR2",
        "IL6", "IL6R", "IL6ST",
        "IL10", "IL10RA", "IL10RB",
        "SOCS1", "SOCS3",
        "IRF1", "IRF8", "IRF9",
        "PIAS1", "PIAS3",
        "MYC", "BCL2",
    ],
    "Complement and coagulation": [
        "C1QA", "C1QB", "C1QC", "C1R", "C1S",
        "C2", "C3", "C4A", "C4B", "C5",
        "C3AR1", "C5AR1",
        "CFB", "CFD", "CFH", "CFI",
        "SERPING1", "CD55", "CD59",
        "CR1", "ITGAM", "ITGB2",
        "F2", "F13A1",
        "PROS1", "SERPINA3", "SERPINE1",
    ],
    "Chemokine signaling": [
        "CCL2", "CCL3", "CCL4", "CCL5", "CCL20",
        "CCL4L2",
        "CXCL8", "CXCL9", "CXCL10", "CXCL11",
        "CXCL12", "CXCL13",
        "CCR1", "CCR2", "CCR5", "CCR7",
        "CXCR3", "CXCR4", "CXCR5",
        "CX3CL1", "CX3CR1",
        "JAK2", "STAT1", "STAT3",
        "PIK3CA", "PIK3R1", "AKT1",
    ],
    "TNF signaling": [
        "TNF", "TNFRSF1A", "TNFRSF1B",
        "TNFSF18", "TNFAIP6",
        "NFKB1", "RELA", "NFKBIA",
        "TRAF2", "TRAF6",
        "CASP3", "CASP8",
        "CCL2", "CXCL8",
        "IL1B", "IL6",
        "ICAM1", "VCAM1",
        "MMP9", "PTGS2", "SOCS3",
    ],
    "Glutamatergic synapse": [
        "SLC17A7", "SLC17A6",
        "GRIA1", "GRIA2", "GRIA3", "GRIA4",
        "GRIN1", "GRIN2A", "GRIN2B",
        "GRM1", "GRM5",
        "SLC1A2", "SLC1A3",
        "GLRA3", "PTPN3", "LNX2",
        "RORB", "CUX2", "SATB2",
        "MAP2", "RBFOX3",
        "CAMK2A", "HOMER1", "DLG4", "SHANK2",
    ],
    "GABAergic synapse": [
        "GAD1", "GAD2", "SLC32A1",
        "GABRA1", "GABRB2", "GABRG2", "GABRD",
        "PVALB", "SST", "VIP",
        "LAMP5", "SNCG",
        "PAX6", "SAMD13", "PRRT4",
        "RPH3AL", "FAM20A", "SMOC1", "COL15A1",
        "MIR101-1",
    ],
    "Calcium signaling": [
        "CALM1", "CALM2", "CALM3",
        "CALR", "CAMK2A", "CAMK2B", "CAMK4",
        "ITPR1", "ITPR2", "ITPR3",
        "RYR1", "RYR2",
        "ATP2B1", "ATP2B2", "SLC8A1",
        "CACNA1A", "CACNA1B", "CACNA1C", "CACNA1E",
    ],
    "Synaptic vesicle cycle": [
        "SYN1", "SYP", "SNAP25", "STX1A", "VAMP2",
        "SYT1", "SLC17A7", "SLC17A6",
        "SLC32A1", "GAD1", "GAD2",
        "CPLX1", "CPLX2", "DNM1", "NSF",
        "STXBP1", "RAB3A", "RIMS1",
    ],
    "Necroptosis": [
        "RIPK1", "RIPK3", "MLKL",
        "FADD", "CASP8",
        "TNF", "TNFRSF1A", "TNFRSF1B",
        "TRAF2", "CFLAR",
        "BIRC2", "BIRC3", "NFKB1",
        "HMGB1", "TLR4",
        "IFNG", "IFNGR1",
        "JAK2", "STAT1",
        "TNFSF18", "TNFAIP6",
    ],
    "Leukocyte transendothelial migration": [
        "ICAM1", "VCAM1", "PECAM1",
        "CDH5", "CLDN5",
        "ITGAL", "ITGA4", "ITGB2",
        "SELL",
        "MMP2", "MMP9",
        "CCL2", "CXCL12",
        "FLT1", "ESAM",
        "JAM2", "JAM3",
    ],

    # =============================================================
    # EXISTING NEUROSCIENCE PATHWAYS (expanded with FCD genes)
    # =============================================================

    "mTORC1 signaling": [
        "MTOR", "RPTOR", "MLST8", "AKT1", "AKT2", "AKT3",
        "TSC1", "TSC2", "RHEB", "RPS6KB1", "RPS6KB2",
        "RPS6", "EIF4EBP1", "EIF4E",
        "PIK3CA", "PIK3CB", "PIK3R1", "PIK3R2",
        "PTEN", "STK11", "DDIT4", "FKBP5",
        "DEPTOR", "SESN1", "SESN2",
        "SLC7A5", "SLC3A2", "SLC38A2",
        "HIF1A", "VEGFA", "LDHA",
    ],
    "Neurodegeneration": [
        "NEFL", "NEFM", "NEFH", "MAPT", "APP", "PSEN1", "PSEN2",
        "SNCA", "PARK7", "PINK1", "PRKN", "LRRK2",
        "SOD1", "SOD2", "TARDBP", "FUS", "C9orf72",
        "UCHL1", "PDYN", "TUBA1A", "TUBB2A", "TUBB2B",
        "COX5A", "COX5B", "COX6C", "COX7C", "NDUFA4",
        "ATP5MC3", "ATP5MG", "ATP6V0B",
        "CALM1", "CALM2", "CALM3",
        "GAPDH",
        "MAP2", "TUBB3", "RBFOX3", "MAG", "OLIG1", "OLIG2",
    ],
    "Microglia activation": [
        "CD74", "CD83", "CD86",
        "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "HLA-DRB1", "HLA-DQA1",
        "C1QA", "C1QB", "C1QC",
        "IFNGR1", "JAK2", "B2M",
        "IL1B", "CCL2", "CCL3", "CCL4",
        "TNFSF18", "NR4A1",
        "IRF8", "STAT1", "STAT2",
        "ELF1", "IKZF1", "NFYC",
        "TREM2", "TYROBP", "CSF1R",
        "AIF1", "CX3CR1", "P2RY12",
        "TMEM119", "HEXB", "SALL1",
        "SC1N", "CCL4L2",
    ],
    "Reactive astrogliosis": [
        "GFAP", "VIM", "NES",
        "CHI3L1", "HSPB1", "SERPINA3",
        "CD44", "TNC", "VCAN",
        "SLC1A2", "SLC1A3", "GPC5",
        "AQP4", "S100B", "ALDH1L1",
        "LCN2", "STEAP4", "CXCL10",
        "C3", "SERPING1", "SRGN",
        "FKBP5",
        "TNFAIP6", "POU5F1", "SOX2", "MYC",
    ],
    "Cortical neuron identity": [
        "SLC17A7", "SATB2", "CUX1", "CUX2", "RORB",
        "THEMIS", "FEZF2", "TBR1", "BCL11B", "LDB2",
        "NRGN", "CAMK2A",
        "GAD1", "GAD2", "SLC32A1",
        "PVALB", "SST", "VIP", "LAMP5", "SNCG",
        "ADARB2",
        "DLX1", "DLX2", "DLX5", "DLX6",
        "RELN",
        "LINC00507", "GLRA3", "PTPN3", "LNX2",
        "LINC00343", "C9orf135-AS1", "SMYD1", "OTOGL", "KLK7",
        "TNFAIP6", "SAMD13", "PRRT4", "MIR101-1",
        "RPH3AL", "FAM20A", "SMOC1", "COL15A1",
        "MAP2", "RBFOX3", "NES",
    ],
    "Complement system": [
        "C1QA", "C1QB", "C1QC", "C1R", "C1S",
        "C2", "C3", "C4A", "C4B", "C5",
        "C3AR1", "C5AR1",
        "CFB", "CFD", "CFH", "CFI",
        "SERPING1", "CD55", "CD59",
        "CR1", "CR2", "ITGAM", "ITGB2",
    ],
    "Epilepsy / synaptic signaling": [
        "SCN1A", "SCN2A", "SCN8A", "KCNQ2", "KCNQ3",
        "GABRA1", "GABRB2", "GABRG2", "GABRD",
        "GRIA1", "GRIA2", "GRIN1", "GRIN2A", "GRIN2B",
        "SYN1", "SYP", "SNAP25", "STX1A", "VAMP2",
        "SLC17A7", "SLC17A6",
        "GAD1", "GAD2", "SLC32A1",
        "NPTX2", "NPTXR",
    ],

    # =============================================================
    # NEW v3: FCD progenitor & balloon cell markers
    # =============================================================
    "FCD progenitor and balloon cell markers": [
        "NES", "VIM", "GFAP",
        "SOX2", "MYC", "POU5F1",
        "CD44", "TNC", "VCAN",
        "CHI3L1", "HSPB1", "SERPINA3",
        "MAP2", "RBFOX3",
        "OLIG1", "OLIG2", "PDGFRA", "CSPG4",
        "MAG", "MBP", "MOG", "PLP1",
    ],
    # NEW v4: 14 Breast Tissue Atlas pathways (Bhat-Nakshatri 2024)
    # =============================================================

    "Estrogen signaling": [
        "ESR1", "ESR2", "FOXA1", "GATA3", "SP1",
        "NCOA1", "NCOA3", "NCOR1", "NCOR2",
        "CYP19A1", "HSD17B1", "SRD5A1",
        "PGR", "TFF1", "GREB1", "CCND1", "MYC",
        "ERBB4", "ERBB2", "IGF1R",
        "SHC1", "GRB2", "SOS1", "HRAS", "KRAS",
        "RAF1", "MAP2K1", "MAPK1", "MAPK3",
        "PIK3CA", "PIK3R1", "AKT1",
        "CREB1", "FOS", "JUN",
        "CALM1", "CALM2", "CALM3",
        "GNAI1", "ADCY7",
    ],
    "EGF signaling": [
        "EGF", "EGFR", "ERBB2", "ERBB3", "ERBB4",
        "TGFA", "HBEGF", "AREG", "BTC", "NRG1",
        "GRB2", "SOS1", "SHC1",
        "HRAS", "KRAS", "NRAS",
        "RAF1", "BRAF", "MAP2K1", "MAP2K2",
        "MAPK1", "MAPK3",
        "PIK3CA", "PIK3R1", "AKT1", "AKT2",
        "STAT3", "STAT5A", "STAT5B",
        "JAK1", "JAK2",
        "MYC", "FOS", "JUN", "EGR1",
        "PLCG1", "PRKCA",
    ],
    "MAPK signaling": [
        "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14",
        "MAP2K1", "MAP2K2", "MAP2K3", "MAP2K4", "MAP2K6", "MAP2K7",
        "MAP3K1", "MAP3K5", "MAP3K7", "MAP3K11",
        "RAF1", "BRAF", "ARAF",
        "HRAS", "KRAS", "NRAS",
        "GRB2", "SOS1", "SHC1",
        "EGFR", "FGFR1", "PDGFRA",
        "FOS", "JUN", "MYC", "EGR1",
        "DUSP1", "DUSP6", "SPRY2",
        "TP53", "CASP3",
        "TGFB1", "TGFBR1",
    ],
    "Protein kinase A signaling": [
        "PRKACA", "PRKACB", "PRKAR1A", "PRKAR1B", "PRKAR2A", "PRKAR2B",
        "ADCY1", "ADCY2", "ADCY3", "ADCY5", "ADCY7", "ADCY9",
        "GNAS", "GNAI1", "GNAI2",
        "CREB1", "CREB3", "ATF4",
        "MAPK1", "MAPK3", "RAF1",
        "PDE4A", "PDE4B", "PDE4D",
        "PTBP1", "RYR1", "RYR2",
        "CALM1", "CALM2", "CALM3",
        "DUSP1", "PPP1CA", "PPP2CA",
        "AKAP1", "AKAP5", "AKAP12",
        "PTCH1", "GLI1",
    ],
    "eIF2 signaling": [
        "EIF2S1", "EIF2S2", "EIF2S3",
        "EIF2AK1", "EIF2AK2", "EIF2AK3", "EIF2AK4",
        "EIF2B1", "EIF2B2", "EIF2B3", "EIF2B4", "EIF2B5",
        "ATF4", "ATF3", "DDIT3",
        "PPP1R15A", "PPP1R15B",
        "HSPA5", "ASNS", "TRIB3",
        "EIF4E", "EIF4G1", "EIF4A1",
        "RPS6", "RPS6KB1",
        "MTOR", "RPTOR",
        "EIF5", "EIF1", "EIF3A",
        "PTBP1",
    ],
    "Notch signaling": [
        "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4",
        "DLL1", "DLL3", "DLL4",
        "JAG1", "JAG2",
        "RBPJ", "MAML1", "MAML2", "MAML3",
        "HES1", "HES5", "HEY1", "HEY2", "HEYL",
        "ADAM10", "ADAM17",
        "PSEN1", "PSEN2", "NCSTN", "APH1A",
        "NUMB", "NUMBL",
        "DTX1", "DTX3L",
        "LFNG", "MFNG",
    ],
    "Cell adhesion molecules": [
        "CDH1", "CDH2", "CDH5",
        "NCAM1", "NCAM2", "L1CAM",
        "ICAM1", "ICAM2", "VCAM1",
        "PECAM1", "CD34",
        "ITGA1", "ITGA4", "ITGA6", "ITGAL",
        "ITGB1", "ITGB2", "ITGB4",
        "SELE", "SELL", "SELP",
        "CD44", "ALCAM", "EPCAM",
        "CLDN1", "CLDN3", "CLDN4", "CLDN5",
        "OCLN", "JAM2", "JAM3",
        "NECTIN1", "NECTIN2", "NECTIN3",
        "PTPRC", "CD96",
    ],
    "ECM-receptor interaction": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL6A1",
        "FN1", "LAMA2", "LAMA4", "LAMB1", "LAMC1",
        "ITGA1", "ITGA2", "ITGA3", "ITGA5", "ITGA6",
        "ITGB1", "ITGB3", "ITGB4",
        "CD44", "SDC1", "SDC4",
        "THBS1", "THBS2",
        "TNC", "VTN", "SPP1",
        "HSPG2", "AGRN",
        "COL25A1",
        "DAG1", "SV2A",
    ],
    "Focal adhesion": [
        "PTK2", "PXN", "VCL", "TLN1", "TLN2",
        "ITGA1", "ITGA5", "ITGA6", "ITGAV",
        "ITGB1", "ITGB3", "ITGB4",
        "COL1A1", "COL3A1", "FN1", "LAMA2",
        "ACTN1", "ACTN4", "FLNA", "FLNB",
        "RAC1", "CDC42", "RHOA",
        "PIK3CA", "PIK3R1", "AKT1",
        "MAPK1", "MAPK3", "SRC",
        "EGFR", "ERBB2", "IGF1R", "PDGFRA",
        "VEGFA", "KDR",
        "MYL9", "MYL12A",
        "PAK1", "PAK2",
    ],
    "Breast cancer": [
        "ESR1", "PGR", "ERBB2", "EGFR",
        "FOXA1", "GATA3",
        "PIK3CA", "AKT1", "PTEN", "MTOR",
        "TP53", "RB1", "CCND1", "CDK4", "CDK6",
        "BRCA1", "BRCA2",
        "MYC", "KRAS", "BRAF",
        "NOTCH1", "WNT1", "CTNNB1",
        "CDH1", "CDH2",
        "MMP2", "MMP9",
        "VEGFA", "KDR",
        "BAX", "BCL2", "CASP3",
        "DACH1", "INPP4B", "NEK10",
        "KRT14", "KRT17", "KRT15",
        "ELF5", "EHF",
    ],
    "Tight junction": [
        "TJP1", "TJP2", "TJP3",
        "CLDN1", "CLDN3", "CLDN4", "CLDN5", "CLDN7",
        "OCLN", "F11R",
        "JAM2", "JAM3",
        "MAGI1", "MAGI2", "MAGI3",
        "CGN", "CGNL1",
        "ACTN1", "ACTN4",
        "MYH11", "MYL9", "MYLK",
        "RHOA", "CDC42", "RAC1",
        "PARD3", "PARD6A",
        "PRKCZ", "PRKCI",
        "CDH5", "ESAM",
    ],
    "Fatty acid metabolism": [
        "FASN", "SCD", "ACACA", "ACACB",
        "ACSL1", "ACSL3", "ACSL4", "ACSL5",
        "CPT1A", "CPT1B", "CPT2",
        "ACOX1", "ACADM", "ACADL", "HADHA", "HADHB",
        "FABP4", "FABP5", "FABP7",
        "CD36", "SLC27A1", "SLC27A2",
        "PPARA", "PPARG", "RXRA",
        "PLIN1", "DGAT1", "DGAT2",
        "ADIRF", "ELOVL5", "ELOVL6",
        "HMGCR", "HMGCS1",
    ],
    "PPAR signaling": [
        "PPARA", "PPARD", "PPARG",
        "RXRA", "RXRB", "RXRG",
        "NCOA1", "NCOA3", "NCOR1",
        "FABP4", "FABP5", "FABP7",
        "CD36", "SLC27A1",
        "ACOX1", "CPT1A", "CPT2",
        "LPL", "PLIN1", "ADIPOQ", "LEP",
        "ACSL1", "ACSL3",
        "SCD", "FASN",
        "HMGCS1", "HMGCS2",
        "CYP7A1", "EHHADH",
        "ADIRF", "MGP",
        "SORBS1", "ANGPTL4",
    ],
    "Natural killer cell cytotoxicity": [
        "KLRC1", "KLRD1", "KLRK1", "KLRB1",
        "NCR1", "NCR2", "NCR3",
        "FCGR3A",
        "GZMB", "GZMA", "GZMH", "GZMK",
        "PRF1", "GNLY",
        "IFNG", "TNF", "FASLG",
        "XCL1", "CSF2",
        "CD244", "CD96",
        "ITGAL", "ITGB2", "ICAM1",
        "SH2D1A", "FCER1G", "TYROBP",
        "HLA-A", "HLA-B", "HLA-C", "HLA-E",
        "KIR2DL1", "KIR2DL3", "KIR3DL1",
        "SKAP1",
    ],
        # =========== Neuroblastoma pathways (Yu et al. Nat Genet 2025) ===========

    "ErbB signaling": [
        "EGFR", "ERBB2", "ERBB3", "ERBB4",
        "HBEGF", "TGFA", "EREG", "AREG", "BTC", "NRG1", "EGF",
        "GRB2", "SOS1", "SHC1",
        "HRAS", "KRAS", "NRAS",
        "RAF1", "BRAF", "MAP2K1", "MAP2K2",
        "MAPK1", "MAPK3",
        "PIK3CA", "PIK3R1", "AKT1", "AKT2",
        "STAT3", "STAT5A", "STAT5B",
        "PLCG1", "PRKCA",
        "MYC", "FOS", "JUN",
        "CD9", "ICAM1",
    ],

    "Cell cycle": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CDK2", "CDK4", "CDK6",
        "CCNA2", "CCNB1", "CCNB2", "CCND1", "CCNE1", "CCNE2",
        "E2F1", "E2F5", "RB1", "TP53",
        "CDC25C", "CDC45", "CDC20",
        "BUB1B", "MAD2L1", "BUB1",
        "BIRC5", "ASPM", "KIF11", "KIF2C", "KIF18B",
        "CENPU", "TACC3", "ANLN", "TPX2",
        "EZH2", "SMC4", "MELK",
        "AURKB", "PLK1", "CHEK1",
        "MCM2", "MCM4", "MCM6",
    ],

    "Axon guidance": [
        "SEMA3A", "SEMA3C", "SEMA3D", "SEMA5A",
        "NRP1", "NRP2", "PLXNA1", "PLXNA2", "PLXND1",
        "ROBO1", "ROBO2", "SLIT1", "SLIT2",
        "DCC", "NTN1", "UNC5C",
        "EPHB2", "EPHB3", "EPHA4",
        "EFNA1", "EFNB2",
        "NTRK1", "NTRK2", "BDNF", "NTF3",
        "L1CAM", "NCAM1",
        "PHOX2B", "PHOX2A", "ISL1", "HAND2",
    ],

    "Dopaminergic synapse": [
        "TH", "DDC", "SLC18A2", "SLC6A3",
        "DRD1", "DRD2", "DRD3", "DRD4", "DRD5",
        "COMT", "MAOA", "MAOB",
        "GNAI2", "GNAS", "ADCY5",
        "CREB1", "FOS", "MAPK1", "MAPK3",
        "PPP1R1B", "CALM1", "CALM2",
        "AGTR2", "ATP2A2",
    ],

    "Neuroblastoma identity": [
        "PHOX2B", "PHOX2A", "ISL1", "HAND2", "GATA3", "ASCL1",
        "TH", "DBH", "DDC", "SLC18A2",
        "CHGA", "CHGB", "SYP",
        "MYCN", "ALK", "RET", "NTRK1", "NTRK2",
        "NEFM", "NEFL",
        "GPC5", "CACNA1B", "SYN2", "KCNMA1", "KCNQ3",
    ],

    "Neuroblastoma MES state": [
        "YAP1", "FN1", "VIM", "COL1A1", "COL4A1", "COL4A2",
        "SERPINE1", "SPARC", "THBS2", "NNMT",
        "JUN", "FOS", "JUNB", "JUND", "FOSL2",
        "BACH1", "BACH2",
        "ZNF148", "ETS1", "ETV6", "ELF1", "KLF6", "KLF7",
        "RUNX1", "GLI3", "SP3", "PURA", "NFIC",
        "NECTIN2", "CD274",
        "SNAI1", "ZEB1", "TWIST1",
    ],

    "Macrophage polarization": [
        "CD68", "CD163", "CSF1R", "MRC1", "CD86",
        "IL18", "CCL4", "CCL3",
        "VCAN", "VEGFA",
        "C1QC", "SPP1", "APOE", "TREM2",
        "F13A1", "LYVE1",
        "HS3ST2", "CYP27A1",
        "THY1",
        "MKI67", "TOP2A",
        "HBEGF", "TGFA", "EREG", "AREG",
        "IL10", "TGFB1", "IDO1",
        "CD80", "HLA-DRA",
    ],

    "Catecholamine biosynthesis": [
        "TH", "DDC", "DBH", "PNMT",
        "SLC18A1", "SLC18A2", "SLC6A3",
        "AGTR2", "ATP2A2",
        "PHOX2B", "PHOX2A", "GATA3",
        "CHGA", "CHGB", "SYP",
        "SCG2", "VGF",
    ],
    "PD-1 signaling": [
        "PDCD1", "CD274", "PDCD1LG2",
        "SHP2", "PTPN11", "PTPN6",
        "LCK", "ZAP70", "LAT",
        "PIK3CA", "PIK3R1", "AKT1",
        "RAS", "RAF1", "MAP2K1", "MAPK1", "MAPK3",
        "NFATC1", "NFATC2",
        "IL2", "IL2RA", "IFNG",
        "GZMB", "PRF1", "TNF",
        "BCL2L1", "BCL2",
        "BATF", "TOX", "TOX2",
    ],

    "T cell exhaustion": [
        "TOX", "TOX2", "NR4A1", "NR4A2", "NR4A3",
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4",
        "ENTPD1", "BTLA",
        "BATF", "IRF4", "PRDM1",
        "TCF7", "LEF1",
        "GZMB", "GZMK", "PRF1", "IFNG",
        "EOMES", "TBX21",
        "CD8A", "CD8B",
        "CXCL13", "IL10",
    ],

    "T cell co-stimulation and co-inhibition": [
        "CD28", "CTLA4", "ICOS",
        "CD80", "CD86",
        "PDCD1", "CD274", "PDCD1LG2",
        "LAG3", "HAVCR2", "TIGIT",
        "TNFRSF9", "TNFRSF4", "TNFRSF18",
        "TNFSF9", "TNFSF4",
        "CD40", "CD40LG",
        "CD226", "NECTIN2", "PVR",
        "BTLA", "TNFRSF14",
        "HVEM", "LIGHT",
    ],

    "Tumor immune evasion": [
        "CD274", "PDCD1LG2", "CD47",
        "B2M", "HLA-A", "HLA-B", "HLA-C",
        "TAP1", "TAP2",
        "IDO1", "IDO2", "TDO2",
        "TGFB1", "TGFB2",
        "VEGFA", "VEGFB",
        "IL10", "IL6",
        "PTGS2", "ARG1",
        "FOXP3", "IL2RA",
        "MYC", "CCND1",
        "SNAI1", "ZEB1", "VIM",
        "CD44", "ALDH1A1",
    ],

    "Melanoma lineage": [
        "MITF", "SOX10", "PAX3",
        "TYR", "TYRP1", "DCT",
        "MLANA", "PMEL", "SILV",
        "MC1R", "EDNRB", "KIT",
        "CDH1", "CDH2",
        "AXL", "NGFR",
        "VIM", "ZEB1", "SNAI1",
        "CD274", "B2M",
    ],

    "Breast cancer subtypes": [
        "ESR1", "PGR", "ERBB2", "EGFR",
        "EPCAM", "KRT8", "KRT18", "KRT19",
        "MUC1", "CDH1",
        "GATA3", "FOXA1",
        "MKI67", "TOP2A",
        "BRCA1", "BRCA2",
        "TP53", "PIK3CA", "AKT1",
        "CD274", "PDCD1LG2",
        "TGFB1", "VEGFA",
    ],

    "Liver cancer markers": [
        "ALB", "AFP", "GPC3",
        "EPCAM", "KRT19", "KRT7",
        "HNF4A", "HNF1A",
        "APOB", "TTR",
        "CYP3A4", "CYP1A2",
        "MKI67", "TOP2A",
        "CD274", "VEGFA",
        "TERT", "CTNNB1", "TP53",
    ],

    "Renal cell carcinoma": [
        "CA9", "PAX8", "PAX2",
        "MME", "EPCAM",
        "VHL", "HIF1A", "EPAS1",
        "VEGFA", "KDR", "FLT1",
        "CD274", "PDCD1LG2",
        "TGFB1", "IL6",
        "MKI67", "TOP2A",
        "SLC2A1", "LDHA",
    ],

    "Basal cell carcinoma / Hedgehog": [
        "PTCH1", "PTCH2", "SMO",
        "GLI1", "GLI2", "GLI3",
        "SHH", "IHH", "DHH",
        "SUFU", "KIF7",
        "EPCAM", "KRT14", "KRT5",
        "TP63", "MYCN",
        "CD274", "B2M",
    ],

    "Tertiary lymphoid structures": [
        "CXCL13", "CXCR5",
        "CCL19", "CCL21", "CCR7",
        "MS4A1", "CD79A", "CD79B",
        "SDC1", "MZB1", "JCHAIN",
        "IGHG1", "IGKC",
        "BCL6", "ICOS", "PDCD1",
        "CD4", "CD3D",
        "CR2", "FCER2",
        "LAMP3", "CCR7",
    ],

    "Myeloid recruitment and polarization": [
        "CSF1", "CSF1R", "CSF2", "CSF3R",
        "CCL2", "CCR2", "CCL5", "CCR5",
        "CXCL12", "CXCR4",
        "CD68", "CD163", "MRC1", "MSR1",
        "C1QA", "C1QB", "APOE", "TREM2", "SPP1",
        "CD14", "FCGR3A", "S100A8", "S100A9",
        "ITGAM", "LYZ",
        "ARG1", "NOS2", "IL10", "TGFB1",
        "CD80", "CD86", "IL12A",
    ],
}


# ============================================================
# All functions below are IDENTICAL to v1/v2 — no changes
# ============================================================

def detect_capabilities(metadata, gene_stats):
    caps = {
        "has_conditions": False, "condition_column": None,
        "condition_values": [], "has_proportions": True,
        "has_interactions": True, "has_pathways": True,
        "n_clusters": len(metadata),
    }
    condition_keywords = [
        "patient_group", "case_control", "condition", "disease",
        "treatment", "status", "health_status", "tissue_status",
        "genotype", "phenotype", "diagnosis", "sample_type", "group",
    ]
    col_candidates = {}
    for cid, meta in metadata.items():
        fields = meta.get("fields", {})
        for col, dist in fields.items():
            if col not in col_candidates: col_candidates[col] = set()
            col_candidates[col].update(dist.keys())
    for kw in condition_keywords:
        for col, values in col_candidates.items():
            col_lower = col.lower().replace("_", " ").replace("-", " ")
            if kw in col_lower or col_lower in kw:
                vals = sorted(values)
                if 2 <= len(vals) <= 10:
                    caps["has_conditions"] = True
                    caps["condition_column"] = col
                    caps["condition_values"] = vals
                    return caps
    skip_patterns = ["batch", "lab", "donor", "sample", "plate", "lane", "seq",
                     "lib", "run", "class", "subclass", "level", "annotation",
                     "ontology", "type", "assay", "sex", "ethnicity",
                     "development", "suspension", "publication", "primary"]
    for col, values in col_candidates.items():
        if len(values) == 2:
            if not any(p in col.lower() for p in skip_patterns):
                caps["has_conditions"] = True
                caps["condition_column"] = col
                caps["condition_values"] = sorted(values)
                return caps
    return caps


def comparative_analysis(gene_stats, metadata, condition_col, group_a, group_b,
                         genes=None, min_pct=0.05, top_n=20):
    results = {"comparison": f"{group_a} vs {group_b}", "condition_column": condition_col,
               "clusters": {}, "summary": {}}
    all_diff_genes = defaultdict(list)
    for cid, stats in gene_stats.items():
        if not stats: continue
        meta = metadata.get(cid, {})
        fields = meta.get("fields", {})
        cond_dist = fields.get(condition_col, {})
        frac_a = cond_dist.get(group_a, 0.0)
        frac_b = cond_dist.get(group_b, 0.0)
        if frac_a < 0.01 and frac_b < 0.01: continue
        bias = group_a if frac_a > 0.6 else (group_b if frac_b > 0.6 else "mixed")
        cluster_genes = []
        gene_pool = genes if genes else list(stats.keys())
        for gene in gene_pool:
            if gene not in stats:
                matches = [g for g in stats if g.upper() == gene.upper()]
                if matches: gene = matches[0]
                else: continue
            g = stats[gene]
            pct_in = g.get("pct_in", 0); pct_out = g.get("pct_out", 0); logfc = g.get("logfc", 0)
            if pct_in < min_pct and not genes: continue
            cluster_genes.append({"gene": gene, "logfc": logfc, "pct_in": pct_in, "pct_out": pct_out,
                                  "cluster_bias": bias, "frac_condition_a": frac_a, "frac_condition_b": frac_b})
            all_diff_genes[gene.upper()].append({"cluster": cid, "logfc": logfc, "pct_in": pct_in, "bias": bias})
        cluster_genes.sort(key=lambda x: abs(x.get("logfc", 0)), reverse=True)
        if not genes: cluster_genes = cluster_genes[:top_n]
        if cluster_genes:
            results["clusters"][cid] = {"n_cells": meta.get("n_cells", "?"),
                "condition_distribution": {group_a: frac_a, group_b: frac_b},
                "bias": bias, "genes": cluster_genes}
    condition_enriched = {group_a: [], group_b: []}
    for gene, entries in all_diff_genes.items():
        a_cl = [e for e in entries if e["bias"] == group_a and e["logfc"] > 0]
        b_cl = [e for e in entries if e["bias"] == group_b and e["logfc"] > 0]
        if a_cl: condition_enriched[group_a].append({"gene": gene, "n_clusters": len(a_cl),
                 "max_logfc": max(e["logfc"] for e in a_cl)})
        if b_cl: condition_enriched[group_b].append({"gene": gene, "n_clusters": len(b_cl),
                 "max_logfc": max(e["logfc"] for e in b_cl)})
    for grp in condition_enriched:
        condition_enriched[grp].sort(key=lambda x: x["max_logfc"], reverse=True)
        condition_enriched[grp] = condition_enriched[grp][:30]
    results["summary"]["condition_enriched_genes"] = condition_enriched
    return results


def find_interactions(gene_stats, cluster_ids, source_clusters=None, target_clusters=None,
                      min_ligand_pct=0.1, min_receptor_pct=0.05, lr_database=None):
    if lr_database is None: lr_database = LR_DATABASE
    if source_clusters is None: source_clusters = cluster_ids
    if target_clusters is None: target_clusters = cluster_ids
    expr_lookup = defaultdict(dict)
    for cid in cluster_ids:
        stats = gene_stats.get(cid, {})
        for gene, vals in stats.items():
            expr_lookup[gene.upper()][cid] = vals.get("pct_in", 0)
    interactions = []; pathway_counts = defaultdict(int)
    for ligand, receptor, pathway in lr_database:
        for src in source_clusters:
            lig_pct = expr_lookup.get(ligand.upper(), {}).get(src, 0)
            if lig_pct < min_ligand_pct: continue
            for tgt in target_clusters:
                if src == tgt: continue
                rec_pct = expr_lookup.get(receptor.upper(), {}).get(tgt, 0)
                if rec_pct < min_receptor_pct: continue
                score = lig_pct * rec_pct
                interactions.append({"ligand": ligand, "receptor": receptor, "pathway": pathway,
                    "source": src, "target": tgt, "ligand_pct": round(lig_pct, 3),
                    "receptor_pct": round(rec_pct, 3), "score": round(score, 4)})
                pathway_counts[pathway] += 1
    interactions.sort(key=lambda x: x["score"], reverse=True)
    pathway_summary = sorted([{"pathway": p, "n_interactions": n} for p, n in pathway_counts.items()],
                             key=lambda x: x["n_interactions"], reverse=True)
    pair_counts = defaultdict(int)
    for ix in interactions: pair_counts[f"{ix['source']} → {ix['target']}"] += 1
    pair_summary = sorted([{"pair": p, "n_interactions": n} for p, n in pair_counts.items()],
                          key=lambda x: x["n_interactions"], reverse=True)[:20]
    return {"n_total": len(interactions), "interactions": interactions,
            "pathway_summary": pathway_summary, "pair_summary": pair_summary, "mode": "interactions"}


def proportion_analysis(metadata, condition_col=None):
    total_cells = sum(m.get("n_cells", 0) for m in metadata.values())
    proportions = []
    for cid, meta in metadata.items():
        n = meta.get("n_cells", 0); frac = n / total_cells if total_cells > 0 else 0
        proportions.append({"cluster": cid, "n_cells": n, "fraction": round(frac, 4), "percentage": round(frac * 100, 2)})
    proportions.sort(key=lambda x: x["n_cells"], reverse=True)
    result = {"total_cells": total_cells, "n_clusters": len(metadata), "proportions": proportions, "mode": "proportions"}
    if condition_col:
        cond_props = defaultdict(list); cond_totals = defaultdict(int)
        for cid, meta in metadata.items():
            n = meta.get("n_cells", 0); fields = meta.get("fields", {})
            cond_dist = fields.get(condition_col, {})
            for cv, frac in cond_dist.items():
                n_in = int(round(n * frac)); cond_totals[cv] += n_in
                cond_props[cv].append({"cluster": cid, "n_cells_approx": n_in, "fraction_of_cluster": round(frac, 3)})
        condition_proportions = {}
        for cv in sorted(cond_props.keys()):
            total_in = cond_totals[cv]; entries = cond_props[cv]
            for e in entries:
                e["fraction_of_condition"] = round(e["n_cells_approx"] / total_in, 4) if total_in > 0 else 0
                e["pct_of_condition"] = round(e["fraction_of_condition"] * 100, 2)
            entries.sort(key=lambda x: x["n_cells_approx"], reverse=True)
            condition_proportions[cv] = {"total_cells": total_in, "clusters": entries}
        result["condition_column"] = condition_col
        result["condition_proportions"] = condition_proportions
        cond_vals = sorted(cond_props.keys())
        if len(cond_vals) == 2:
            a, b = cond_vals; fc_list = []
            a_map = {e["cluster"]: e["fraction_of_condition"] for e in cond_props[a]}
            b_map = {e["cluster"]: e["fraction_of_condition"] for e in cond_props[b]}
            for cid in set(list(a_map) + list(b_map)):
                fa = a_map.get(cid, 0); fb = b_map.get(cid, 0)
                fc = fa / fb if fb > 0.001 else (float("inf") if fa > 0.001 else 1.0)
                fc_list.append({"cluster": cid, f"pct_{a}": round(fa * 100, 2), f"pct_{b}": round(fb * 100, 2),
                                f"fold_change_{a}_vs_{b}": round(fc, 3) if fc != float("inf") else "inf"})
            fc_list.sort(key=lambda x: abs(x.get(f"fold_change_{a}_vs_{b}", 1))
                         if x.get(f"fold_change_{a}_vs_{b}") != "inf" else 999, reverse=True)
            result["proportion_fold_changes"] = fc_list
    return result


def pathway_scoring(gene_stats, cluster_ids, pathways=None, metric="pct_in", min_genes=3):
    if pathways is None: pathways = PATHWAY_GENESETS
    expr_lookup = defaultdict(dict)
    for cid in cluster_ids:
        stats = gene_stats.get(cid, {})
        for gene, vals in stats.items():
            expr_lookup[gene.upper()][cid] = vals.get(metric, 0)
    pathway_results = {}
    for pw_name, pw_genes in pathways.items():
        pw_upper = [g.upper() for g in pw_genes]
        scores = []
        for cid in cluster_ids:
            vals = []; found = []
            for g in pw_upper:
                v = expr_lookup.get(g, {}).get(cid, None)
                if v is not None: vals.append(v); found.append(g)
            score = float(np.mean(vals)) if len(found) >= min_genes else 0.0
            cov = len(found) / len(pw_genes) if pw_genes else 0
            scores.append({"cluster": cid, "score": round(score, 4), "n_genes_found": len(found),
                           "n_genes_total": len(pw_genes), "coverage": round(cov, 3),
                           "top_genes": sorted([(g, expr_lookup.get(g, {}).get(cid, 0)) for g in found],
                                               key=lambda x: x[1], reverse=True)[:5]})
        scores.sort(key=lambda x: x["score"], reverse=True)
        pathway_results[pw_name] = {"scores": scores,
            "top_cluster": scores[0]["cluster"] if scores else None,
            "top_score": scores[0]["score"] if scores else 0}
    ranked = sorted([{"pathway": k, **v} for k, v in pathway_results.items()], key=lambda x: x["top_score"], reverse=True)
    return {"mode": "pathway_scoring", "metric": metric, "n_pathways": len(pathway_results),
            "pathways": pathway_results, "ranked": ranked}


def query_pathway(gene_stats, cluster_ids, pathway_name, pathways=None, metric="pct_in"):
    if pathways is None: pathways = PATHWAY_GENESETS
    query_lower = pathway_name.lower()
    best_match = None; best_score = 0
    for pw in pathways:
        pw_lower = pw.lower()
        q_words = set(query_lower.split()); p_words = set(pw_lower.replace("-", " ").replace("/", " ").split())
        overlap = len(q_words & p_words)
        if overlap > best_score: best_score = overlap; best_match = pw
        if query_lower in pw_lower or pw_lower in query_lower: best_match = pw; break
    if best_match is None:
        return {"error": f"Pathway '{pathway_name}' not found", "available": list(pathways.keys())}
    result = pathway_scoring(gene_stats, cluster_ids, pathways={best_match: pathways[best_match]}, metric=metric)
    pw_data = result["pathways"].get(best_match, {})
    return {"mode": "pathway_query", "pathway": best_match, "genes_in_pathway": pathways[best_match],
            "scores": pw_data.get("scores", [])}


def marker_specificity(gene_stats, cluster_ids, genes=None, top_n=20):
    gene_profiles = defaultdict(list)
    for cid in cluster_ids:
        stats = gene_stats.get(str(cid), {}) or {}
        for g, vals in stats.items():
            if genes and g.upper() not in {x.upper() for x in genes}: continue
            gene_profiles[g.upper()].append({"cluster": cid, "pct_in": vals.get("pct_in", 0),
                "pct_out": vals.get("pct_out", 0), "logfc": vals.get("logfc", 0)})
    per_cluster = defaultdict(list); global_spec = []
    for gene, entries in gene_profiles.items():
        for e in entries:
            denom = e["pct_in"] + e["pct_out"] + 1e-8
            spec = e["pct_in"] / denom; score = spec * abs(e.get("logfc", 0) or 0)
            entry = {"gene": gene, "cluster": e["cluster"], "specificity": round(spec, 4),
                     "weighted_score": round(score, 4), "pct_in": round(e["pct_in"], 3),
                     "pct_out": round(e["pct_out"], 3), "logfc": round(e.get("logfc", 0) or 0, 3)}
            per_cluster[e["cluster"]].append(entry); global_spec.append(entry)
    for cid in per_cluster:
        per_cluster[cid].sort(key=lambda x: x["weighted_score"], reverse=True)
        per_cluster[cid] = per_cluster[cid][:top_n]
    global_spec.sort(key=lambda x: x["weighted_score"], reverse=True)
    return {"mode": "marker_specificity", "n_genes_analyzed": len(gene_profiles),
            "per_cluster": dict(per_cluster), "top_global": global_spec[:50]}


def coexpression_analysis(gene_stats, cluster_ids, gene_a, gene_b=None, top_n=20):
    gene_vectors = {}
    for cid in cluster_ids:
        stats = gene_stats.get(str(cid), {}) or {}
        for g, vals in stats.items():
            g_up = g.upper()
            if g_up not in gene_vectors: gene_vectors[g_up] = {}
            gene_vectors[g_up][cid] = vals.get("pct_in", 0)
    ga_up = gene_a.upper()
    if ga_up not in gene_vectors: return {"error": f"Gene {gene_a} not found"}
    vec_a = np.array([gene_vectors[ga_up].get(cid, 0) for cid in cluster_ids])
    if gene_b:
        gb_up = gene_b.upper()
        if gb_up not in gene_vectors: return {"error": f"Gene {gene_b} not found"}
        vec_b = np.array([gene_vectors[gb_up].get(cid, 0) for cid in cluster_ids])
        corr = float(np.corrcoef(vec_a, vec_b)[0, 1]) if np.std(vec_a) > 0 and np.std(vec_b) > 0 else 0.0
        return {"mode": "coexpression", "gene_a": gene_a, "gene_b": gene_b, "correlation": round(corr, 4),
                "profile_a": {cid: round(gene_vectors[ga_up].get(cid, 0), 3) for cid in cluster_ids},
                "profile_b": {cid: round(gene_vectors[gb_up].get(cid, 0), 3) for cid in cluster_ids}}
    correlations = []
    for g_up, vec_dict in gene_vectors.items():
        if g_up == ga_up: continue
        vec_g = np.array([vec_dict.get(cid, 0) for cid in cluster_ids])
        if np.std(vec_g) < 1e-8: continue
        corr = float(np.corrcoef(vec_a, vec_g)[0, 1])
        if not np.isnan(corr): correlations.append({"gene": g_up, "correlation": round(corr, 4)})
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {"mode": "coexpression", "query_gene": gene_a,
            "top_positive": [c for c in correlations if c["correlation"] > 0][:top_n],
            "top_negative": [c for c in correlations if c["correlation"] < 0][:top_n],
            "n_genes_tested": len(correlations)}


CELL_CYCLE_GENES = {
    "S_phase": ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
        "CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP","RAD51AP1",
        "GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2",
        "CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1",
        "CHAF1B","BRIP1","E2F8"],
    "G2M_phase": ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
        "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L",
        "CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3",
        "HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8",
        "ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2",
        "G2E3","GAS2L3","CBX5","CENPA"],
}

def cell_cycle_scoring(gene_stats, cluster_ids, metric="pct_in"):
    expr_lookup = defaultdict(dict)
    for cid in cluster_ids:
        stats = gene_stats.get(str(cid), {}) or {}
        for gene, vals in stats.items():
            expr_lookup[gene.upper()][cid] = vals.get(metric, 0)
    results = []
    for cid in cluster_ids:
        s_vals = [expr_lookup.get(g.upper(), {}).get(cid) for g in CELL_CYCLE_GENES["S_phase"]]
        g2m_vals = [expr_lookup.get(g.upper(), {}).get(cid) for g in CELL_CYCLE_GENES["G2M_phase"]]
        s_vals = [v for v in s_vals if v is not None]; g2m_vals = [v for v in g2m_vals if v is not None]
        s_score = float(np.mean(s_vals)) if s_vals else 0.0
        g2m_score = float(np.mean(g2m_vals)) if g2m_vals else 0.0
        if s_score > 0.3 and g2m_score > 0.3: phase = "cycling"
        elif s_score > 0.2: phase = "S"
        elif g2m_score > 0.2: phase = "G2M"
        else: phase = "G1"
        results.append({"cluster": cid, "s_score": round(s_score, 4), "g2m_score": round(g2m_score, 4),
                        "phase": phase, "n_s_genes": len(s_vals), "n_g2m_genes": len(g2m_vals)})
    results.sort(key=lambda x: x["s_score"] + x["g2m_score"], reverse=True)
    return {"mode": "cell_cycle", "metric": metric, "clusters": results,
            "cycling_clusters": [r["cluster"] for r in results if r["phase"] == "cycling"],
            "quiescent_clusters": [r["cluster"] for r in results if r["phase"] == "G1"]}


HALLMARK_GENESETS = {
    "Inflammatory response": ["IL1B","IL6","TNF","CXCL8","CCL2","CCL3","CCL4","CCL5","NFKB1","NFKBIA",
        "PTGS2","SOCS3","ICAM1","VCAM1","SELE","CSF2","CSF3","IL1A","IL18","HMGB1"],
    "Epithelial mesenchymal transition": ["VIM","CDH2","FN1","SNAI1","SNAI2","TWIST1","ZEB1","ZEB2",
        "MMP2","MMP9","COL1A1","COL3A1","TGFB1","TGFB2","ACTA2","SPARC","LOX","POSTN","ITGB1","CTGF"],
    "Oxidative phosphorylation": ["NDUFA1","NDUFA2","NDUFB1","COX5A","COX5B","COX6A1","COX7A2",
        "ATP5F1A","ATP5F1B","ATP5MC1","UQCRC1","UQCRC2","SDHB","SDHC","CYCS","COX4I1","ATP5PO"],
    "Apoptosis": ["BAX","BAK1","BCL2","BCL2L1","CASP3","CASP8","CASP9","CYCS","DIABLO","BID","BAD",
        "PARP1","APAF1","FAS","FASLG","TNFRSF10A","TNFRSF10B","XIAP","BIRC2","TP53"],
    "Hypoxia": ["HIF1A","VEGFA","SLC2A1","LDHA","PGK1","ENO1","PKM","ALDOA","GAPDH","BNIP3","BNIP3L",
        "CA9","EPO","EDN1","HMOX1","ADM","EGLN1","EGLN3","PDK1"],
    "MYC targets": ["MYC","NPM1","NCL","LDHA","ENO1","PKM","HSPE1","HSPD1","SRM","ODC1","RPS2",
        "RPL3","RPS3","RPL4","EIF4E","CDK4"],
    "P53 pathway": ["TP53","MDM2","CDKN1A","BAX","BBC3","GADD45A","GADD45B","SESN1","SESN2","DDB2",
        "RRM2B","TIGAR","ZMAT3","FAS"],
    "Interferon alpha response": ["ISG15","MX1","MX2","IFIT1","IFIT2","IFIT3","OAS1","OAS2","OAS3",
        "STAT1","STAT2","IRF7","IRF9","BST2","IFI44","IFI44L","RSAD2","DDX58","IFIH1","XAF1"],
    "Interferon gamma response": ["IFNG","STAT1","IRF1","GBP1","GBP2","GBP4","GBP5","IDO1","TAP1",
        "TAP2","PSMB8","PSMB9","HLA-A","HLA-B","HLA-C","CIITA","CXCL9","CXCL10","CXCL11","CD274"],
    "TNF-alpha signaling via NF-kB": ["NFKB1","NFKB2","RELA","RELB","TNFAIP3","NFKBIA","BIRC2","BIRC3",
        "BCL2L1","CXCL8","CCL2","CCL5","ICAM1","VCAM1","SELE","MMP9","SOD2","PTGS2"],
}

def geneset_enrichment(gene_stats, cluster_ids, genesets=None, metric="pct_in", min_genes=3):
    if genesets is None: genesets = HALLMARK_GENESETS
    expr_lookup = defaultdict(dict)
    for cid in cluster_ids:
        stats = gene_stats.get(str(cid), {}) or {}
        for gene, vals in stats.items():
            expr_lookup[gene.upper()][cid] = vals.get(metric, 0)
    enrichment_matrix = {}
    for gs_name, gs_genes in genesets.items():
        gs_upper = [g.upper() for g in gs_genes]; cluster_scores = {}
        for cid in cluster_ids:
            found = [(g, expr_lookup.get(g, {}).get(cid, None)) for g in gs_upper]
            found = [(g, v) for g, v in found if v is not None]
            if len(found) >= min_genes:
                score = float(np.mean([v for _, v in found]))
                cluster_scores[cid] = {"score": round(score, 4), "n_found": len(found),
                    "n_total": len(gs_genes), "coverage": round(len(found)/len(gs_genes), 3),
                    "top_genes": sorted(found, key=lambda x: x[1], reverse=True)[:5]}
            else:
                cluster_scores[cid] = {"score": 0.0, "n_found": len(found), "n_total": len(gs_genes),
                    "coverage": round(len(found)/len(gs_genes), 3), "top_genes": []}
        enrichment_matrix[gs_name] = cluster_scores
    top_per_cluster = {cid: {"geneset": max(enrichment_matrix, key=lambda gs: enrichment_matrix[gs].get(cid, {}).get("score", 0)),
        "score": max(enrichment_matrix[gs].get(cid, {}).get("score", 0) for gs in enrichment_matrix)} for cid in cluster_ids}
    top_per_geneset = {gs: {"cluster": max(cluster_ids, key=lambda c: enrichment_matrix[gs].get(c, {}).get("score", 0)),
        "score": max(enrichment_matrix[gs].get(c, {}).get("score", 0) for c in cluster_ids)} for gs in enrichment_matrix}
    return {"mode": "geneset_enrichment", "metric": metric, "n_genesets": len(genesets),
            "enrichment": enrichment_matrix, "top_per_cluster": top_per_cluster, "top_per_geneset": top_per_geneset}
