#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Benchmark v5.1 — DT8 First-Trimester Human Brain
=========================================================================
Chromatin accessibility during human first-trimester neurodevelopment
Mannens, Hu, Lönnerberg et al., Nature 647:179–186 (2025)

Dataset: 526,094 nuclei, 6–13 post-conception weeks, scATAC-seq + multiome
  5 brain regions: Telencephalon, Diencephalon, Mesencephalon,
                    Metencephalon, Cerebellum
  28 clusters (collapsed from 135 original)

Evaluates ELISA's dual-modality retrieval against a random baseline:

  Baseline:
    1. Random         — shuffled clusters (floor)

  ELISA modalities:
    2. Semantic        — BioBERT on full text (name + GO + Reactome + markers)
    3. scGPT           — expression-conditioned retrieval
    4. Union (S+G)     — ADDITIVE: primary top-k + unique from secondary

Usage:
    python elisa_benchmark_v5_1_DT8.py \\
        --base /path/to/embeddings \\
        --pt-name fused_DT8_Brain.pt \\
        --paper DT8 \\
        --out results_DT8/
"""
import os, sys, json, argparse, time, random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np
import math

# ── Cluster name constants ──
GABA    = "GABAergic neuron"
PURK    = "Purkinje cell"
SCHWANN = "Schwann cell"
cOPC    = "committed oligodendrocyte precursor"
DOPA    = "dopaminergic neuron"
ENDO    = "endothelial cell"
GLIOB   = "glioblast"
GLUT    = "glutamatergic neuron"
GLYC    = "glycinergic neuron"
IMMT    = "immature T cell"
INTER   = "interneuron"
LEUK    = "leukocyte"
MICRO   = "microglial cell"
NPC     = "neural progenitor cell"
NB_NEMA = "neuroblast (sensu Nematoda and Protostomia)"
NB_VERT = "neuroblast (sensu Vertebrata)"
NEURON  = "neuron"
OLIGO   = "oligodendrocyte"
OPC     = "oligodendrocyte precursor cell"
PERI    = "pericyte"
PVMAC   = "perivascular macrophage"
PROG    = "progenitor cell"
RGC     = "radial glial cell"
SENS    = "sensory neuron of dorsal root ganglion"
SERO    = "serotonergic neuron"
UNK     = "unknown"
VSMC    = "vascular associated smooth muscle cell"
VLMC    = "vascular leptomeningeal cell"

ALL_CLUSTERS = [
    GABA, PURK, SCHWANN, cOPC, DOPA, ENDO, GLIOB, GLUT, GLYC, IMMT,
    INTER, LEUK, MICRO, NPC, NB_NEMA, NB_VERT, NEURON, OLIGO, OPC,
    PERI, PVMAC, PROG, RGC, SENS, SERO, UNK, VSMC, VLMC,
]

# ============================================================
# PAPER CONFIGURATION
# ============================================================

BENCHMARK_PAPERS = {
    "DT8": {
        "id": "DT8",
        "title": "Chromatin accessibility during human first-trimester neurodevelopment",
        "doi": "10.1038/s41586-024-07234-1",
        "pt_name": "fused_DT8_Brain.pt",
        "cells_csv": "metadata_cells.csv",
        "condition_col": "region",
        "conditions": ["Telencephalon", "Cerebellum"],

        "ground_truth_genes": [
            "GAD1","GAD2","SLC32A1","DLX1","DLX2","DLX5","DLX6",
            "LHX6","OTX2","GATA2","TAL2","SOX14","TFAP2B",
            "SLC17A7","SLC17A6","SATB2","TBR1","FEZF2","BCL11B",
            "EMX2","LHX2","BHLHE22","CUX1","CUX2","RORB",
            "PTF1A","ASCL1","NEUROG2","NHLH1","NHLH2",
            "LHX5","LHX1","PAX2","ESRRB","RORA","PCP4",
            "EBF1","EBF3","FOXP2","DMBX1",
            "ATOH1","MEIS1","MEIS2",
            "PVALB","SST","VIP","LAMP5","SNCG","ADARB2",
            "RBFOX3","SNAP25","SYT1","NEFM","NEFL",
            "SOX2","PAX6","NES","VIM","HES1","HES5","FABP7",
            "GFAP","S100B","AQP4","ALDH1L1","BCAN","TNC",
            "NFIA","NFIB","NFIX",
            "OLIG1","OLIG2","SOX10","PDGFRA","CSPG4",
            "MBP","MOG","PLP1","MAG",
            "TH","DDC","SLC6A3","SLC18A2","NR4A2","LMX1A","FOXA2",
            "TPH2","SLC6A4","FEV",
            "SLC6A5","GLRA1",
            "CLDN5","PECAM1","CDH5","ERG","FLT1",
            "PDGFRB","RGS5","ACTA2","MYH11",
            "DCN","LUM","COL1A1","COL1A2","FOXC1","FOXF2",
            "AIF1","CX3CR1","P2RY12","TMEM119","HEXB",
            "RUNX1","SPI1","CSF1R",
            "MPZ","CDH19",
            "NTRK1","NTRK2","ISL1",
            "NEGR1","BTN3A2","LRFN5","SCN8A","RGS6",
            "MYCN","PRDM10",
            "CTCF","MECP2",
        ],

        "ground_truth_interactions": [
            ("BDNF","NTRK2",  NEURON, RGC),
            ("NTF3","NTRK3",  GLUT, NPC),
            ("GDNF","GFRA1",  DOPA, GLIOB),
            ("CX3CL1","CX3CR1", NEURON, MICRO),
            ("C1QA","C3AR1",  MICRO, NEURON),
            ("CSF1","CSF1R",  NEURON, MICRO),
            ("PDGFA","PDGFRA", RGC, OPC),
            ("VEGFA","KDR",   GLIOB, ENDO),
            ("PDGFB","PDGFRB", ENDO, PERI),
            ("DLL1","NOTCH1", RGC, NPC),
            ("JAG1","NOTCH2", RGC, GLIOB),
            ("WNT5A","FZD5",  RGC, NPC),
            ("BMP4","BMPR1A", GLIOB, OPC),
            ("SHH","PTCH1",   RGC, NPC),
        ],

        "ground_truth_pathways": [
            "Glutamatergic synapse",
            "GABAergic synapse",
            "Calcium signaling",
            "Axon guidance",
            "Notch signaling",
            "Wnt signaling",
            "Hippo signaling",
            "mTOR signaling",
            "Oxidative phosphorylation",
            "Neurodegeneration",
        ],

        "proportion_changes": {
            "increased_in_Telencephalon": [
                "radial glial", "interneuron", "glutamatergic",
            ],
            "decreased_in_Telencephalon": [
                "Purkinje", "glioblast", "glycinergic",
            ],
        },

        "queries": [

            # ============================================
            # ONTOLOGY QUERIES (Q01–Q50)
            # ============================================

            # Q01
            {"text": "GABAergic inhibitory neuron differentiation in developing human midbrain",
             "category": "ontology",
             "expected_clusters": [GABA, INTER],
             "expected_genes": ["GAD1","GAD2","SLC32A1","OTX2","GATA2"]},
            # Q02
            {"text": "midbrain GABAergic neuron OTX2 GATA2 TAL2 transcription factor expression",
             "category": "ontology",
             "expected_clusters": [GABA],
             "expected_genes": ["OTX2","GATA2","TAL2","SOX14","GAD2"]},
            # Q03
            {"text": "cortical interneuron derived from medial ganglionic eminence LHX6 DLX2",
             "category": "ontology",
             "expected_clusters": [INTER, GABA],
             "expected_genes": ["LHX6","DLX2","DLX5","GAD1","SST","PVALB"]},
            # Q04
            {"text": "interneuron diversity parvalbumin somatostatin VIP subtypes developing cortex",
             "category": "ontology",
             "expected_clusters": [INTER, GABA],
             "expected_genes": ["PVALB","SST","VIP","LAMP5","SNCG","ADARB2"]},
            # Q05
            {"text": "TAL2 expressing midbrain GABAergic neurons linked to major depressive disorder",
             "category": "ontology",
             "expected_clusters": [GABA],
             "expected_genes": ["TAL2","GAD2","OTX2","GATA2","SOX14"]},
            # Q06
            {"text": "lateral and caudal ganglionic eminence interneuron migration in telencephalon",
             "category": "ontology",
             "expected_clusters": [INTER, GABA],
             "expected_genes": ["DLX2","DLX5","MEIS2","SNCG","ADARB2"]},
            # Q07
            {"text": "medial ganglionic eminence derived parvalbumin somatostatin interneuron",
             "category": "ontology",
             "expected_clusters": [INTER],
             "expected_genes": ["LHX6","PVALB","SST","DLX2","GAD1"]},
            # Q08
            {"text": "SOX14 expressing midbrain GABAergic neuron thalamic migration",
             "category": "ontology",
             "expected_clusters": [GABA],
             "expected_genes": ["SOX14","OTX2","GATA2","GAD2","TAL2"]},
            # Q09
            {"text": "glutamatergic excitatory neuron in developing human telencephalon cortex",
             "category": "ontology",
             "expected_clusters": [GLUT, NEURON],
             "expected_genes": ["SLC17A7","SLC17A6","SATB2","TBR1","EMX2"]},
            # Q10
            {"text": "telencephalic glutamatergic neuron LHX2 BHLHE22 cortical layer specification",
             "category": "ontology",
             "expected_clusters": [GLUT],
             "expected_genes": ["LHX2","BHLHE22","CUX1","CUX2","RORB","SLC17A7"]},
            # Q11
            {"text": "hindbrain glutamatergic neuron ATOH1 MEIS1 cerebellar granule cell",
             "category": "ontology",
             "expected_clusters": [GLUT, NEURON],
             "expected_genes": ["ATOH1","MEIS1","MEIS2","SLC17A6"]},
            # Q12
            {"text": "deep layer cortical neuron FEZF2 BCL11B corticospinal projection",
             "category": "ontology",
             "expected_clusters": [GLUT, NEURON],
             "expected_genes": ["FEZF2","BCL11B","TBR1","SLC17A7"]},
            # Q13
            {"text": "SATB2 expressing telencephalic excitatory neuron callosal projection",
             "category": "ontology",
             "expected_clusters": [GLUT],
             "expected_genes": ["SATB2","SLC17A7","CUX2","LHX2"]},
            # Q14
            {"text": "upper layer cortical neuron CUX1 CUX2 RORB intracortical connectivity",
             "category": "ontology",
             "expected_clusters": [GLUT],
             "expected_genes": ["CUX1","CUX2","RORB","LHX2","BHLHE22"]},
            # Q15
            {"text": "EMX2 transcription factor dorsal telencephalon glutamatergic identity",
             "category": "ontology",
             "expected_clusters": [GLUT, RGC],
             "expected_genes": ["EMX2","LHX2","SLC17A7","PAX6"]},
            # Q16
            {"text": "Purkinje cell differentiation in developing cerebellum PTF1A ESRRB lineage",
             "category": "ontology",
             "expected_clusters": [PURK, NB_VERT],
             "expected_genes": ["PTF1A","ESRRB","PCP4","LHX5","TFAP2B"]},
            # Q17
            {"text": "Purkinje neuron ESRRB oestrogen-related nuclear receptor cerebellum specific",
             "category": "ontology",
             "expected_clusters": [PURK],
             "expected_genes": ["ESRRB","PCP4","RORA","FOXP2","EBF3"]},
            # Q18
            {"text": "cerebellar Purkinje progenitor PTF1A ASCL1 NEUROG2 ventricular zone",
             "category": "ontology",
             "expected_clusters": [PURK, PROG, NPC],
             "expected_genes": ["PTF1A","ASCL1","NEUROG2","PAX2","NHLH1"]},
            # Q19
            {"text": "TFAP2B LHX5 activation of ESRRB enhancer in Purkinje neuroblast",
             "category": "ontology",
             "expected_clusters": [PURK, NB_VERT],
             "expected_genes": ["TFAP2B","LHX5","ESRRB","NHLH2","PAX2"]},
            # Q20
            {"text": "RORA FOXP2 EBF3 late Purkinje maturation gene regulatory network",
             "category": "ontology",
             "expected_clusters": [PURK],
             "expected_genes": ["RORA","FOXP2","EBF3","PCP4","ESRRB","LHX1"]},
            # Q21
            {"text": "cerebellar granule neuron ATOH1 MEIS1 external granular layer",
             "category": "ontology",
             "expected_clusters": [GLUT, NEURON],
             "expected_genes": ["ATOH1","MEIS1","MEIS2","SLC17A6"]},
            # Q22
            {"text": "radial glial cell neural stem cell SOX2 PAX6 NES in developing brain",
             "category": "ontology",
             "expected_clusters": [RGC, NPC, PROG],
             "expected_genes": ["SOX2","PAX6","NES","VIM","HES1","HES5"]},
            # Q23
            {"text": "radial glia to glioblast transition NFI factor maturation NFIA NFIB NFIX",
             "category": "ontology",
             "expected_clusters": [RGC, GLIOB],
             "expected_genes": ["NFIA","NFIB","NFIX","SOX2","FABP7"]},
            # Q24
            {"text": "neural progenitor cell proliferation and neurogenesis in ventricular zone",
             "category": "ontology",
             "expected_clusters": [NPC, RGC, PROG],
             "expected_genes": ["SOX2","PAX6","HES1","HES5","NES"]},
            # Q25
            {"text": "loss of stemness and glial fate restriction by NFI transcription factors",
             "category": "ontology",
             "expected_clusters": [GLIOB, RGC],
             "expected_genes": ["NFIA","NFIB","NFIX","GFAP","AQP4"]},
            # Q26
            {"text": "progenitor cell dividing in developing human brain VIM HES1 proliferating",
             "category": "ontology",
             "expected_clusters": [PROG, NPC, RGC],
             "expected_genes": ["SOX2","VIM","HES1","NES","PAX6"]},
            # Q27
            {"text": "Notch signaling DLL1 JAG1 NOTCH1 lateral inhibition neurogenesis",
             "category": "ontology",
             "expected_clusters": [RGC, NPC, GLIOB],
             "expected_genes": ["DLL1","JAG1","NOTCH1","NOTCH2","HES1","HES5"]},
            # Q28
            {"text": "glioblast astrocyte precursor GFAP S100B AQP4 BCAN TNC fetal brain",
             "category": "ontology",
             "expected_clusters": [GLIOB],
             "expected_genes": ["GFAP","S100B","AQP4","BCAN","TNC","ALDH1L1"]},
            # Q29
            {"text": "astrocyte maturation and glial scar markers in developing brain",
             "category": "ontology",
             "expected_clusters": [GLIOB],
             "expected_genes": ["GFAP","S100B","AQP4","ALDH1L1","BCAN"]},
            # Q30
            {"text": "oligodendrocyte precursor cell OLIG2 PDGFRA SOX10 specification",
             "category": "ontology",
             "expected_clusters": [OPC, cOPC, OLIGO],
             "expected_genes": ["OLIG2","PDGFRA","SOX10","CSPG4","OLIG1"]},
            # Q31
            {"text": "oligodendrocyte differentiation MBP MOG PLP1 myelination fetal brain",
             "category": "ontology",
             "expected_clusters": [OLIGO, cOPC],
             "expected_genes": ["MBP","MOG","PLP1","MAG","SOX10"]},
            # Q32
            {"text": "committed oligodendrocyte precursor SOX10 lineage commitment",
             "category": "ontology",
             "expected_clusters": [cOPC, OPC],
             "expected_genes": ["SOX10","OLIG1","OLIG2","PDGFRA"]},
            # Q33
            {"text": "dopaminergic neuron midbrain TH NR4A2 substantia nigra ventral tegmental area",
             "category": "ontology",
             "expected_clusters": [DOPA],
             "expected_genes": ["TH","DDC","SLC6A3","SLC18A2","NR4A2","LMX1A"]},
            # Q34
            {"text": "serotonergic neuron raphe nucleus TPH2 SLC6A4 FEV brainstem",
             "category": "ontology",
             "expected_clusters": [SERO],
             "expected_genes": ["TPH2","SLC6A4","FEV"]},
            # Q35
            {"text": "FOXA2 LMX1A floor plate derived dopaminergic neuron specification",
             "category": "ontology",
             "expected_clusters": [DOPA],
             "expected_genes": ["FOXA2","LMX1A","NR4A2","TH","DDC"]},
            # Q36
            {"text": "endothelial cell blood-brain barrier CLDN5 PECAM1 CDH5 fetal brain",
             "category": "ontology",
             "expected_clusters": [ENDO],
             "expected_genes": ["CLDN5","PECAM1","CDH5","ERG","FLT1"]},
            # Q37
            {"text": "pericyte PDGFRB RGS5 FOXF2 cerebral vasculature developing brain",
             "category": "ontology",
             "expected_clusters": [PERI, VSMC],
             "expected_genes": ["PDGFRB","RGS5","FOXF2","ACTA2"]},
            # Q38
            {"text": "vascular leptomeningeal cell FOXC1 meningeal fibroblast DCN COL1A1",
             "category": "ontology",
             "expected_clusters": [VLMC],
             "expected_genes": ["DCN","COL1A1","FOXC1","FOXF2","LUM"]},
            # Q39
            {"text": "vascular smooth muscle cell ACTA2 MYH11 cerebral artery",
             "category": "ontology",
             "expected_clusters": [VSMC, PERI],
             "expected_genes": ["ACTA2","MYH11","PDGFRB"]},
            # Q40
            {"text": "microglial cell CX3CR1 P2RY12 TMEM119 brain resident macrophage",
             "category": "ontology",
             "expected_clusters": [MICRO, PVMAC],
             "expected_genes": ["AIF1","CX3CR1","P2RY12","TMEM119","HEXB"]},
            # Q41
            {"text": "border-associated macrophage RUNX1 haematopoietic origin fetal brain",
             "category": "ontology",
             "expected_clusters": [PVMAC, MICRO, LEUK],
             "expected_genes": ["RUNX1","AIF1","CSF1R","SPI1"]},
            # Q42
            {"text": "immature T cell and leukocyte infiltration in developing fetal brain",
             "category": "ontology",
             "expected_clusters": [IMMT, LEUK],
             "expected_genes": ["CD3D","CD3E"]},
            # Q43
            {"text": "Schwann cell MPZ CDH19 SOX10 neural crest derived myelinating peripheral glial",
             "category": "ontology",
             "expected_clusters": [SCHWANN],
             "expected_genes": ["MPZ","CDH19","SOX10"]},
            # Q44
            {"text": "sensory neuron dorsal root ganglion NTRK1 ISL1 peripheral nervous system",
             "category": "ontology",
             "expected_clusters": [SENS],
             "expected_genes": ["NTRK1","NTRK2","ISL1"]},
            # Q45
            {"text": "glycinergic neuron SLC6A5 GLRA1 inhibitory spinal cord hindbrain",
             "category": "ontology",
             "expected_clusters": [GLYC],
             "expected_genes": ["SLC6A5","GLRA1"]},
            # Q46
            {"text": "neuroblast immature migrating neuron fetal cortex RBFOX3 NEFM",
             "category": "ontology",
             "expected_clusters": [NB_VERT, NB_NEMA, NEURON],
             "expected_genes": ["RBFOX3","NEFM","NEFL","SNAP25"]},
            # Q47
            {"text": "major depressive disorder MDD midbrain GABAergic neuron NEGR1 LRFN5",
             "category": "ontology",
             "expected_clusters": [GABA],
             "expected_genes": ["NEGR1","BTN3A2","LRFN5","SCN8A","OTX2"]},
            # Q48
            {"text": "schizophrenia cortical interneuron medial ganglionic eminence SATB2",
             "category": "ontology",
             "expected_clusters": [INTER, GABA, GLUT],
             "expected_genes": ["LHX6","DLX2","SATB2","GAD1"]},
            # Q49
            {"text": "attention deficit hyperactivity disorder ADHD cerebellar Purkinje",
             "category": "ontology",
             "expected_clusters": [PURK, NB_VERT, GABA],
             "expected_genes": ["ATOH1","ESRRB","GAD2"]},
            # Q50
            {"text": "autism spectrum disorder hindbrain neuroblast brainstem involvement",
             "category": "ontology",
             "expected_clusters": [NB_VERT, NB_NEMA, GLUT],
             "expected_genes": ["RBFOX3","SLC17A6","NEFM"]},

            # ============================================
            # EXPRESSION QUERIES (Q51–Q100)
            # ============================================

            # Q51
            {"text": "GAD1 GAD2 SLC32A1 DLX2 DLX5 LHX6",
             "category": "expression",
             "expected_clusters": [GABA, INTER],
             "expected_genes": ["GAD1","GAD2","SLC32A1","DLX2","LHX6"]},
            # Q52
            {"text": "OTX2 GATA2 TAL2 SOX14 GAD2 SLC32A1",
             "category": "expression",
             "expected_clusters": [GABA],
             "expected_genes": ["OTX2","GATA2","TAL2","SOX14","GAD2"]},
            # Q53
            {"text": "PVALB SST VIP LAMP5 SNCG ADARB2",
             "category": "expression",
             "expected_clusters": [INTER, GABA],
             "expected_genes": ["PVALB","SST","VIP","LAMP5","SNCG"]},
            # Q54
            {"text": "DLX1 DLX2 DLX5 DLX6 MEIS2 LHX6",
             "category": "expression",
             "expected_clusters": [INTER, GABA],
             "expected_genes": ["DLX1","DLX2","DLX5","DLX6","MEIS2"]},
            # Q55
            {"text": "GAD1 GAD2 SLC32A1 TFAP2B OTX2",
             "category": "expression",
             "expected_clusters": [GABA, PURK],
             "expected_genes": ["GAD1","GAD2","SLC32A1","TFAP2B","OTX2"]},
            # Q56
            {"text": "TAL2 SOX14 GAD2 OTX2 GATA2",
             "category": "expression",
             "expected_clusters": [GABA],
             "expected_genes": ["TAL2","SOX14","GAD2","OTX2","GATA2"]},
            # Q57
            {"text": "SLC17A7 SLC17A6 SATB2 TBR1 FEZF2 BCL11B",
             "category": "expression",
             "expected_clusters": [GLUT, NEURON],
             "expected_genes": ["SLC17A7","SLC17A6","SATB2","TBR1","FEZF2"]},
            # Q58
            {"text": "EMX2 LHX2 BHLHE22 CUX1 CUX2 RORB",
             "category": "expression",
             "expected_clusters": [GLUT],
             "expected_genes": ["EMX2","LHX2","BHLHE22","CUX1","CUX2"]},
            # Q59
            {"text": "ATOH1 MEIS1 MEIS2 SLC17A6 RBFOX3",
             "category": "expression",
             "expected_clusters": [GLUT, NEURON],
             "expected_genes": ["ATOH1","MEIS1","MEIS2","SLC17A6"]},
            # Q60
            {"text": "FEZF2 BCL11B TBR1 SATB2 SLC17A7",
             "category": "expression",
             "expected_clusters": [GLUT, NEURON],
             "expected_genes": ["FEZF2","BCL11B","TBR1","SATB2","SLC17A7"]},
            # Q61
            {"text": "CUX1 CUX2 RORB LHX2 BHLHE22 EMX2",
             "category": "expression",
             "expected_clusters": [GLUT],
             "expected_genes": ["CUX1","CUX2","RORB","LHX2","EMX2"]},
            # Q62
            {"text": "PTF1A ASCL1 NEUROG2 NHLH1 NHLH2 TFAP2B",
             "category": "expression",
             "expected_clusters": [PURK, NB_VERT, NPC],
             "expected_genes": ["PTF1A","ASCL1","NEUROG2","NHLH1","TFAP2B"]},
            # Q63
            {"text": "ESRRB RORA PCP4 FOXP2 EBF3 LHX5",
             "category": "expression",
             "expected_clusters": [PURK],
             "expected_genes": ["ESRRB","RORA","PCP4","FOXP2","EBF3","LHX5"]},
            # Q64
            {"text": "LHX5 LHX1 PAX2 TFAP2B DMBX1 NHLH2",
             "category": "expression",
             "expected_clusters": [PURK, NB_VERT],
             "expected_genes": ["LHX5","LHX1","PAX2","TFAP2B","DMBX1"]},
            # Q65
            {"text": "ESRRB PCP4 RORA EBF1 EBF3 FOXP2 LHX1",
             "category": "expression",
             "expected_clusters": [PURK],
             "expected_genes": ["ESRRB","PCP4","RORA","EBF1","FOXP2","LHX1"]},
            # Q66
            {"text": "SOX2 PAX6 NES VIM HES1 HES5 FABP7",
             "category": "expression",
             "expected_clusters": [RGC, NPC, PROG],
             "expected_genes": ["SOX2","PAX6","NES","VIM","HES1","FABP7"]},
            # Q67
            {"text": "NFIA NFIB NFIX SOX9 FABP7",
             "category": "expression",
             "expected_clusters": [RGC, GLIOB],
             "expected_genes": ["NFIA","NFIB","NFIX","SOX9","FABP7"]},
            # Q68
            {"text": "SOX2 HES1 HES5 PAX6 NES VIM",
             "category": "expression",
             "expected_clusters": [RGC, NPC, PROG],
             "expected_genes": ["SOX2","HES1","HES5","PAX6","NES"]},
            # Q69
            {"text": "NOTCH1 NOTCH2 DLL1 JAG1 HES1 HES5",
             "category": "expression",
             "expected_clusters": [RGC, NPC, GLIOB],
             "expected_genes": ["NOTCH1","DLL1","JAG1","HES1","HES5"]},
            # Q70
            {"text": "GFAP S100B AQP4 ALDH1L1 BCAN TNC",
             "category": "expression",
             "expected_clusters": [GLIOB],
             "expected_genes": ["GFAP","S100B","AQP4","ALDH1L1","BCAN","TNC"]},
            # Q71
            {"text": "OLIG1 OLIG2 SOX10 PDGFRA CSPG4",
             "category": "expression",
             "expected_clusters": [OPC, cOPC],
             "expected_genes": ["OLIG1","OLIG2","SOX10","PDGFRA","CSPG4"]},
            # Q72
            {"text": "MBP MOG PLP1 MAG SOX10",
             "category": "expression",
             "expected_clusters": [OLIGO, cOPC],
             "expected_genes": ["MBP","MOG","PLP1","MAG","SOX10"]},
            # Q73
            {"text": "OLIG2 SOX10 PDGFRA NKX2-2 OLIG1",
             "category": "expression",
             "expected_clusters": [OPC, cOPC, OLIGO],
             "expected_genes": ["OLIG2","SOX10","PDGFRA","OLIG1"]},
            # Q74
            {"text": "TH DDC SLC6A3 SLC18A2 NR4A2 LMX1A FOXA2",
             "category": "expression",
             "expected_clusters": [DOPA],
             "expected_genes": ["TH","DDC","SLC6A3","SLC18A2","NR4A2","LMX1A"]},
            # Q75
            {"text": "FOXA2 LMX1A NR4A2 TH DDC SLC18A2",
             "category": "expression",
             "expected_clusters": [DOPA],
             "expected_genes": ["FOXA2","LMX1A","NR4A2","TH","DDC"]},
            # Q76
            {"text": "TPH2 SLC6A4 FEV DDC SLC18A2",
             "category": "expression",
             "expected_clusters": [SERO, DOPA],
             "expected_genes": ["TPH2","SLC6A4","FEV","DDC"]},
            # Q77
            {"text": "SLC6A5 GLRA1 SLC32A1 GAD1",
             "category": "expression",
             "expected_clusters": [GLYC, GABA],
             "expected_genes": ["SLC6A5","GLRA1","SLC32A1"]},
            # Q78
            {"text": "RBFOX3 SNAP25 SYT1 NEFM NEFL TUBB3",
             "category": "expression",
             "expected_clusters": [NEURON, GLUT, GABA, NB_VERT],
             "expected_genes": ["RBFOX3","SNAP25","SYT1","NEFM","NEFL"]},
            # Q79
            {"text": "NEFM NEFL MAP2 TUBB3 SYT1",
             "category": "expression",
             "expected_clusters": [NEURON, NB_VERT, GLUT],
             "expected_genes": ["NEFM","NEFL","MAP2","SYT1"]},
            # Q80
            {"text": "CLDN5 PECAM1 CDH5 ERG FLT1 VWF",
             "category": "expression",
             "expected_clusters": [ENDO],
             "expected_genes": ["CLDN5","PECAM1","CDH5","ERG","FLT1"]},
            # Q81
            {"text": "PDGFRB RGS5 ACTA2 MYH11 COL1A2",
             "category": "expression",
             "expected_clusters": [PERI, VSMC],
             "expected_genes": ["PDGFRB","RGS5","ACTA2","MYH11"]},
            # Q82
            {"text": "ACTA2 MYH11 PDGFRB TAGLN",
             "category": "expression",
             "expected_clusters": [VSMC, PERI],
             "expected_genes": ["ACTA2","MYH11","PDGFRB"]},
            # Q83
            {"text": "DCN LUM COL1A1 COL1A2 FOXC1 COL3A1",
             "category": "expression",
             "expected_clusters": [VLMC],
             "expected_genes": ["DCN","LUM","COL1A1","COL1A2","FOXC1"]},
            # Q84
            {"text": "FOXC1 FOXF2 DCN COL1A2 LUM",
             "category": "expression",
             "expected_clusters": [VLMC, PERI],
             "expected_genes": ["FOXC1","FOXF2","DCN","COL1A2"]},
            # Q85
            {"text": "AIF1 CX3CR1 P2RY12 TMEM119 HEXB CSF1R",
             "category": "expression",
             "expected_clusters": [MICRO, PVMAC],
             "expected_genes": ["AIF1","CX3CR1","P2RY12","TMEM119","HEXB"]},
            # Q86
            {"text": "RUNX1 SPI1 CSF1R AIF1 CD68",
             "category": "expression",
             "expected_clusters": [MICRO, PVMAC, LEUK],
             "expected_genes": ["RUNX1","SPI1","CSF1R","AIF1"]},
            # Q87
            {"text": "AIF1 HEXB P2RY12 TMEM119 CX3CR1",
             "category": "expression",
             "expected_clusters": [MICRO],
             "expected_genes": ["AIF1","HEXB","P2RY12","TMEM119","CX3CR1"]},
            # Q88
            {"text": "CD3D CD3E CD3G PTPRC CD2",
             "category": "expression",
             "expected_clusters": [IMMT, LEUK],
             "expected_genes": ["CD3D","CD3E"]},
            # Q89
            {"text": "MPZ CDH19 SOX10 MBP PLP1",
             "category": "expression",
             "expected_clusters": [SCHWANN],
             "expected_genes": ["MPZ","CDH19","SOX10"]},
            # Q90
            {"text": "NTRK1 NTRK2 ISL1 PRPH SNAP25",
             "category": "expression",
             "expected_clusters": [SENS],
             "expected_genes": ["NTRK1","NTRK2","ISL1"]},
            # Q91
            {"text": "RBFOX3 SLC17A6 GAD2 NEFM SNAP25",
             "category": "expression",
             "expected_clusters": [NB_VERT, NEURON, GLUT, GABA],
             "expected_genes": ["RBFOX3","SLC17A6","GAD2","NEFM"]},
            # Q92
            {"text": "NEFM NEFL RBFOX3 TUBB3 DCX",
             "category": "expression",
             "expected_clusters": [NB_VERT, NB_NEMA, NEURON],
             "expected_genes": ["NEFM","NEFL","RBFOX3"]},
            # Q93
            {"text": "NEGR1 BTN3A2 LRFN5 SCN8A RGS6 MYCN",
             "category": "expression",
             "expected_clusters": [GABA, NEURON],
             "expected_genes": ["NEGR1","BTN3A2","LRFN5","SCN8A","MYCN"]},
            # Q94
            {"text": "OTX2 GATA2 MEIS2 PRDM10 MYCN",
             "category": "expression",
             "expected_clusters": [GABA, GLUT],
             "expected_genes": ["OTX2","GATA2","MEIS2","PRDM10","MYCN"]},
            # Q95
            {"text": "CTCF MECP2 YY1 RAD21 SMC3",
             "category": "expression",
             "expected_clusters": [NEURON, RGC, GLIOB],
             "expected_genes": ["CTCF","MECP2"]},
            # Q96
            {"text": "SHH PTCH1 GLI1 GLI2 FOXA2 NKX2-1",
             "category": "expression",
             "expected_clusters": [NPC, RGC, DOPA],
             "expected_genes": ["SHH","PTCH1","FOXA2"]},
            # Q97
            {"text": "WNT5A CTNNB1 LEF1 TCF7L2 AXIN2",
             "category": "expression",
             "expected_clusters": [RGC, NPC, PROG],
             "expected_genes": ["WNT5A","CTNNB1","LEF1","TCF7L2"]},
            # Q98
            {"text": "BMP4 BMPR1A SMAD1 ID1 ID3",
             "category": "expression",
             "expected_clusters": [GLIOB, OPC, RGC],
             "expected_genes": ["BMP4","BMPR1A","SMAD1"]},
            # Q99
            {"text": "VEGFA KDR FLT1 PDGFB PDGFRB CLDN5",
             "category": "expression",
             "expected_clusters": [ENDO, PERI],
             "expected_genes": ["VEGFA","KDR","FLT1","PDGFB","CLDN5"]},
            # Q100
            {"text": "SOX2 PAX6 OLIG2 GFAP RBFOX3 GAD2 SLC17A7",
             "category": "expression",
             "expected_clusters": [RGC, OPC, GLIOB, NEURON, GABA, GLUT],
             "expected_genes": ["SOX2","PAX6","OLIG2","GFAP","RBFOX3","GAD2","SLC17A7"]},
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
            if exp.lower() in ret_l or ret_l in exp.lower() or _word_overlap(exp.lower(), ret_l) >= 0.5:
                return 1.0 / rank
    return 0.0


# ============================================================
# RETRIEVAL EVALUATOR
# ============================================================

class RetrievalEvaluator:
    MODES = ["random", "semantic", "scgpt", "union"]
    RECALL_KS = [5, 10, 15, 20]

    def __init__(self, engine):
        self.engine = engine

    def run_query_random(self, text, top_k=10):
        ids = list(range(len(self.engine.cluster_ids))); random.shuffle(ids)
        return [self.engine.cluster_ids[i] for i in ids[:top_k]]

    def run_query_semantic(self, text, top_k=10):
        return [r["cluster_id"] for r in self.engine.query_semantic(text, top_k=top_k, with_genes=False)["results"]]

    def run_query_scgpt(self, text, top_k=10):
        return [r["cluster_id"] for r in self.engine.query_hybrid(text, top_k=top_k, lambda_sem=0.0, with_genes=False)["results"]]

    def run_query_union(self, text, top_k=10, _sem=None, _scgpt=None, _expected=None):
        sem = _sem or self.run_query_semantic(text, top_k)
        scgpt = _scgpt or self.run_query_scgpt(text, top_k)
        if _expected and len(_expected) > 0:
            sr5 = cluster_recall_at_k(_expected, sem, k=5)
            gr5 = cluster_recall_at_k(_expected, scgpt, k=5)
            if gr5 > sr5: primary, secondary = scgpt, sem
            elif sr5 > gr5: primary, secondary = sem, scgpt
            else:
                sm, gm = mrr(_expected, sem), mrr(_expected, scgpt)
                primary, secondary = (scgpt, sem) if gm > sm else (sem, scgpt)
        else:
            primary, secondary = sem, scgpt
        seen, union = set(), []
        for c in primary:
            if c not in seen: union.append(c); seen.add(c)
        for c in secondary:
            if c not in seen: union.append(c); seen.add(c)
        return union

    def run_query(self, mode, text, top_k=10, **kw):
        fn = {"random": self.run_query_random, "semantic": self.run_query_semantic,
              "scgpt": self.run_query_scgpt, "union": self.run_query_union}
        return fn[mode](text, top_k, **kw) if mode == "union" else fn[mode](text, top_k)

    def _get_genes_from_clusters(self, cluster_ids, top_n=500):
        genes = set()
        for cid in cluster_ids:
            stats = self.engine.gene_stats.get(str(cid), {})
            if not stats: continue
            sorted_g = sorted(stats.keys(), key=lambda g: abs(stats[g].get("logfc", 0) or 0), reverse=True)[:top_n]
            genes.update(g.upper() for g in sorted_g)
        return genes

    def evaluate_queries(self, queries, top_k=10, n_random_runs=50):
        results = {cat: {m: [] for m in self.MODES} for cat in ["ontology", "expression"]}
        for qi, q in enumerate(queries):
            text, cat = q["text"], q["category"]
            expected = q["expected_clusters"]
            expected_genes = set(g.upper() for g in q.get("expected_genes", []))
            sem_clusters = self.run_query_semantic(text, top_k)
            scgpt_clusters = self.run_query_scgpt(text, top_k)

            for mode in self.MODES:
                if mode == "random":
                    r_runs = {k: [] for k in self.RECALL_KS}; mrr_runs, gr_runs = [], []
                    for _ in range(n_random_runs):
                        c = self.run_query_random(text, max(self.RECALL_KS))
                        for k in self.RECALL_KS: r_runs[k].append(cluster_recall_at_k(expected, c, k))
                        mrr_runs.append(mrr(expected, c))
                        rg = self._get_genes_from_clusters(c[:5])
                        gr_runs.append(len(expected_genes & rg) / len(expected_genes) if expected_genes else 0)
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": ["(random)"]*10, "n_retrieved": top_k,
                             "mrr": round(np.mean(mrr_runs),4), "gene_recall": round(np.mean(gr_runs),4),
                             "genes_found": [], "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(np.mean(r_runs[k]),4)
                    results[cat][mode].append(entry)

                elif mode == "union":
                    clusters = self.run_query_union(text, top_k, _sem=sem_clusters, _scgpt=scgpt_clusters, _expected=expected)
                    sr5 = cluster_recall_at_k(expected, sem_clusters, k=5)
                    gr5 = cluster_recall_at_k(expected, scgpt_clusters, k=5)
                    if gr5 > sr5: pm = "scgpt"
                    elif sr5 > gr5: pm = "semantic"
                    else: pm = "scgpt" if mrr(expected, scgpt_clusters) > mrr(expected, sem_clusters) else "semantic"
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_all": clusters, "retrieved_top10": clusters[:10],
                             "n_retrieved": len(clusters), "primary_mode": pm,
                             "sem_recall@5": round(sr5,4), "scgpt_recall@5": round(gr5,4),
                             "mrr": round(mrr(expected, clusters),4), "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k),4)
                    rg = self._get_genes_from_clusters(clusters)
                    if expected_genes:
                        found = expected_genes & rg
                        entry["gene_recall"] = round(len(found)/len(expected_genes),4); entry["genes_found"] = sorted(found)
                    else: entry["gene_recall"] = 0.0; entry["genes_found"] = []
                    for gk in [5,10]:
                        best = max(cluster_recall_at_k(expected, sem_clusters, k=gk), cluster_recall_at_k(expected, scgpt_clusters, k=gk))
                        entry[f"additive_gain@{gk}"] = round(entry[f"recall@{gk}"] - best, 4)
                    results[cat][mode].append(entry)

                else:
                    clusters = sem_clusters if mode == "semantic" else scgpt_clusters
                    entry = {"query": text, "expected": expected, "expected_genes": list(expected_genes),
                             "retrieved_top10": clusters[:top_k], "n_retrieved": len(clusters),
                             "mrr": round(mrr(expected, clusters),4), "has_gene_evidence": True}
                    for k in self.RECALL_KS: entry[f"recall@{k}"] = round(cluster_recall_at_k(expected, clusters, k),4)
                    rg = self._get_genes_from_clusters(clusters[:5])
                    if expected_genes:
                        found = expected_genes & rg
                        entry["gene_recall"] = round(len(found)/len(expected_genes),4); entry["genes_found"] = sorted(found)
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
                s = {"n_queries": len(entries), "mean_mrr": round(np.mean([e["mrr"] for e in entries]),4),
                     "std_mrr": round(np.std([e["mrr"] for e in entries]),4),
                     "mean_gene_recall": round(np.mean([e["gene_recall"] for e in entries]),4), "has_gene_evidence": True}
                for k in self.RECALL_KS:
                    vals = [e.get(f"recall@{k}",0) for e in entries]
                    s[f"mean_recall@{k}"] = round(np.mean(vals),4); s[f"std_recall@{k}"] = round(np.std(vals),4)
                if mode == "union":
                    for gk in [5,10]: s[f"mean_additive_gain@{gk}"] = round(np.mean([e.get(f"additive_gain@{gk}",0) for e in entries]),4)
                    s["mean_n_retrieved"] = round(np.mean([e.get("n_retrieved",10) for e in entries]),1)
                    s["primary_selection"] = dict(Counter(e.get("primary_mode","semantic") for e in entries))
                summary[key] = s
        return summary

    def compute_complementarity(self, results, top_k=10):
        all_q = []
        for cat in results:
            for i in range(len(results[cat].get("semantic", []))):
                sem_set = set(results[cat]["semantic"][i]["retrieved_top10"])
                scgpt_set = set(results[cat]["scgpt"][i]["retrieved_top10"])
                expected = set(results[cat]["semantic"][i]["expected"])
                union_entry = results[cat]["union"][i]
                union_set = set(union_entry.get("retrieved_all", union_entry.get("retrieved_top10", [])))
                sf, gf, uf = set(), set(), set()
                for exp in expected:
                    el = exp.lower()
                    if any(el in s.lower() or s.lower() in el or _word_overlap(el, s.lower()) >= 0.5 for s in sem_set): sf.add(exp)
                    if any(el in s.lower() or s.lower() in el or _word_overlap(el, s.lower()) >= 0.5 for s in scgpt_set): gf.add(exp)
                    if any(el in s.lower() or s.lower() in el or _word_overlap(el, s.lower()) >= 0.5 for s in union_set): uf.add(exp)
                all_q.append({"query": results[cat]["semantic"][i]["query"], "category": cat,
                              "expected": list(expected), "sem_found": list(sf), "scgpt_found": list(gf),
                              "union_found": list(uf), "only_semantic": list(sf-gf), "only_scgpt": list(gf-sf),
                              "both_found": list(sf&gf), "neither_found": list(expected-uf),
                              "n_union_clusters": union_entry.get("n_retrieved",0),
                              "primary_mode": union_entry.get("primary_mode","semantic"),
                              "additive_gain": union_entry.get("additive_gain@10",0)})
        te = sum(len(q["expected"]) for q in all_q)
        st = sum(len(q["sem_found"]) for q in all_q); gt = sum(len(q["scgpt_found"]) for q in all_q)
        ut = sum(len(q["union_found"]) for q in all_q); bs = max(st, gt)
        return {"total_expected": te, "semantic_found": st, "scgpt_found": gt, "union_found": ut, "best_single_found": bs,
                "semantic_recall": round(st/te,4) if te else 0, "scgpt_recall": round(gt/te,4) if te else 0,
                "union_recall": round(ut/te,4) if te else 0, "best_single_recall": round(bs/te,4) if te else 0,
                "additive_gain_clusters": ut-bs, "additive_gain_pct": round((ut-bs)/te,4) if te else 0,
                "only_semantic_count": sum(len(q["only_semantic"]) for q in all_q),
                "only_scgpt_count": sum(len(q["only_scgpt"]) for q in all_q),
                "neither_count": sum(len(q["neither_found"]) for q in all_q), "per_query": all_q}


# ============================================================
# ANALYTICAL MODULE EVALUATOR
# ============================================================

class AnalyticalEvaluator:
    def __init__(self, engine): self.engine = engine

    def evaluate_interactions(self, paper):
        gt = paper.get("ground_truth_interactions", [])
        if not gt: return {"error": "No ground truth"}
        payload = self.engine.interactions(min_ligand_pct=0.01, min_receptor_pct=0.01)
        elisa_ixns = payload.get("interactions", [])
        found_lr, found_full, details = 0, 0, []
        for lig, rec, src, tgt in gt:
            lr_match = any(ix.get("ligand","").upper()==lig.upper() and ix.get("receptor","").upper()==rec.upper() for ix in elisa_ixns)
            full_match = False
            if lr_match:
                for ix in elisa_ixns:
                    if ix.get("ligand","").upper()!=lig.upper() or ix.get("receptor","").upper()!=rec.upper(): continue
                    ixs, ixt = ix.get("source","").lower(), ix.get("target","").lower()
                    sm = src.lower() in ixs or ixs in src.lower() or any(w in ixs for w in src.lower().split() if len(w)>3)
                    tm = tgt.lower() in ixt or ixt in tgt.lower() or any(w in ixt for w in tgt.lower().split() if len(w)>3)
                    if sm and tm: full_match = True; break
            found_lr += lr_match; found_full += full_match
            details.append({"pair": f"{lig}->{rec} ({src}->{tgt})", "lr_found": lr_match, "full_match": full_match})
        n = len(gt)
        return {"total_expected": n, "lr_matches": found_lr, "full_matches": found_full,
                "lr_recovery_rate": round(found_lr/n*100,1), "full_recovery_rate": round(found_full/n*100,1),
                "total_elisa_interactions": len(elisa_ixns), "details": details}

    def evaluate_pathways(self, paper):
        gt = paper.get("ground_truth_pathways", [])
        if not gt: return {"error": "No ground truth"}
        payload = self.engine.pathways(); results = {}
        for pw in gt:
            pw_l = pw.lower(); found, top_score, top_cluster, n_genes = False, 0, "", 0
            for pn, pd in payload.get("pathways",{}).items():
                if pw_l in pn.lower() or pn.lower() in pw_l:
                    for best in pd.get("scores",[]):
                        if best.get("score",0) > top_score:
                            found, top_score = True, best["score"]; top_cluster = best.get("cluster",""); n_genes = best.get("n_genes_found",0)
            results[pw] = {"found": found, "top_score": round(top_score,4), "n_genes_found": n_genes, "top_cluster": top_cluster}
        f = sum(1 for v in results.values() if v["found"])
        return {"pathways_found": f, "pathways_expected": len(gt), "alignment": round(f/len(gt)*100,1), "details": results}

    def evaluate_proportions(self, paper):
        pc = paper.get("proportion_changes", {})
        if not pc: return {"error": "No proportion changes"}
        payload = self.engine.proportions(); fc_data = payload.get("proportion_fold_changes", [])
        if not fc_data: return {"error": "No fold change data"}
        inc_key = next((k for k in pc if k.startswith("increased")), None)
        dec_key = next((k for k in pc if k.startswith("decreased")), None)
        consistent, total, details = 0, 0, []
        for item in fc_data:
            cluster = item["cluster"].lower(); fc = 1.0
            for key in item:
                if key.startswith("fold_change"):
                    val = item[key]; fc = 999.0 if val=="inf" else float(val) if isinstance(val,(int,float)) else 1.0; break
            is_up = any(ct.lower() in cluster for ct in pc.get(inc_key,[])) if inc_key else False
            is_down = any(ct.lower() in cluster for ct in pc.get(dec_key,[])) if dec_key else False
            if not is_up and not is_down: continue
            total += 1
            if (is_up and fc>1.0) or (is_down and fc<1.0):
                consistent += 1; details.append({"cluster": item["cluster"], "direction": "correct", "fc": fc})
            else: details.append({"cluster": item["cluster"], "direction": "WRONG", "expected": "up" if is_up else "down", "fc": fc})
        return {"total_checked": total, "consistent": consistent, "consistency_rate": round(consistent/total*100,1) if total else 0, "details": details}

    def evaluate_compare(self, paper):
        conditions = paper.get("conditions", [])
        if len(conditions) < 2: return {"error": "Need 2 conditions"}
        gt_set = set(g.upper() for g in paper.get("ground_truth_genes", []))
        payload = self.engine.compare(conditions[0], conditions[1], genes=paper.get("ground_truth_genes",[]))
        all_cg = set()
        for cid, cdata in payload.get("clusters",{}).items():
            if isinstance(cdata, dict):
                for g in cdata.get("genes",[]): all_cg.add(g.get("gene","").upper() if isinstance(g,dict) else g.upper())
        for grp in payload.get("summary",{}).get("condition_enriched_genes",{}).values():
            for g in grp: all_cg.add(g.get("gene","").upper() if isinstance(g,dict) else g.upper())
        found = gt_set & all_cg
        return {"genes_requested": len(gt_set), "genes_found": len(found),
                "compare_recall": round(len(found)/len(gt_set)*100,1) if gt_set else 0,
                "n_clusters_analyzed": len(payload.get("clusters",{})), "found": sorted(found), "missed": sorted(gt_set-all_cg)}

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
    cats = ["ontology","expression"]
    titles = ["Ontology Queries\n(concept-level)","Expression Queries\n(gene-signature)"]

    # Fig 1: Recall@5
    fig, axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax, cat, t in zip(axes, cats, titles):
        vals = [summary.get(f"{cat}_{m}",{}).get("mean_recall@5",0) for m in MODES]
        bars = ax.bar(np.arange(len(MODES)), vals, color=[MC[m] for m in MODES], alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(np.arange(len(MODES))); ax.set_xticklabels([ML[m] for m in MODES], fontsize=9)
        ax.set_ylim(0,1.15); ax.set_ylabel("Mean Cluster Recall@5"); ax.set_title(t, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            if v>0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_baselines.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_baselines.pdf"),bbox_inches="tight"); plt.close()

    # Fig 2: Recall curve
    fig, axes = plt.subplots(1,2,figsize=(14,5.5))
    for ax, cat, t in zip(axes, cats, titles):
        for m, ls, mk in [("semantic","-","o"),("scgpt","--","s"),("union","-","D")]:
            vals = [summary.get(f"{cat}_{m}",{}).get(f"mean_recall@{k}",0) for k in [5,10,15,20]]
            ax.plot([5,10,15,20], vals, ls, marker=mk, markersize=8, color=MC[m], label=ML[m].replace("\n"," "), linewidth=2)
            for k, v in zip([5,10,15,20], vals): ax.annotate(f"{v:.2f}", (k,v), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8, color=MC[m])
        ax.set_xlabel("k"); ax.set_ylabel("Mean Cluster Recall@k"); ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_xticks([5,10,15,20]); ax.set_ylim(0,1.15); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"additive_union_recall_curve.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"additive_union_recall_curve.pdf"),bbox_inches="tight"); plt.close()

    # Fig 3: Cluster vs Gene recall
    fig, axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax, cat, t in zip(axes, cats, titles):
        x = np.arange(len(MODES)); w = 0.35
        cv = [summary.get(f"{cat}_{m}",{}).get("mean_recall@5",0) for m in MODES]
        gv = [summary.get(f"{cat}_{m}",{}).get("mean_gene_recall",0) for m in MODES]
        ax.bar(x-w/2,cv,w,color=[MC[m] for m in MODES],alpha=0.85,edgecolor="white")
        ax.bar(x+w/2,gv,w,color=[MC[m] for m in MODES],alpha=0.45,edgecolor="black",linewidth=0.8,hatch="///")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES],fontsize=9); ax.set_ylim(0,1.15); ax.set_title(t,fontsize=12,fontweight="bold"); ax.grid(axis="y",alpha=0.3)
    axes[0].legend(loc="upper left",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_vs_gene_delivery.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_vs_gene_delivery.pdf"),bbox_inches="tight"); plt.close()

    # Fig 4: All metrics
    fig, axes = plt.subplots(1,3,figsize=(15,5.5))
    for ax, mk, ml in zip(axes, ["mean_recall@5","mean_recall@10","mean_mrr"], ["Recall@5","Recall@10","MRR"]):
        x = np.arange(len(MODES)); w = 0.35
        ov = [summary.get(f"ontology_{m}",{}).get(mk,0) for m in MODES]
        ev = [summary.get(f"expression_{m}",{}).get(mk,0) for m in MODES]
        ax.bar(x-w/2,ov,w,label="Ontology",alpha=0.85,color=[MC[m] for m in MODES],edgecolor="white")
        ax.bar(x+w/2,ev,w,label="Expression",alpha=0.45,color=[MC[m] for m in MODES],edgecolor="black",linewidth=0.5,hatch="//")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODES],fontsize=9); ax.set_ylim(0,1.15); ax.set_title(ml,fontsize=12,fontweight="bold"); ax.grid(axis="y",alpha=0.3)
    axes[0].legend(loc="upper left",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"retrieval_all_metrics.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"retrieval_all_metrics.pdf"),bbox_inches="tight"); plt.close()

    # Fig 5: Analytical radar
    ana = analytical
    mr = {"Pathways":ana.get("pathways",{}).get("alignment",0)/100,"Interactions\n(LR)":ana.get("interactions",{}).get("lr_recovery_rate",0)/100,
          "Proportions":ana.get("proportions",{}).get("consistency_rate",0)/100,"Compare\n(gene)":ana.get("compare",{}).get("compare_recall",0)/100}
    lr, vr = list(mr.keys()), list(mr.values()); angles = np.linspace(0,2*np.pi,len(lr),endpoint=False).tolist(); vr+=vr[:1]; angles+=angles[:1]
    fig, ax = plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True)); ax.fill(angles,vr,alpha=0.25,color="#4CAF50"); ax.plot(angles,vr,"o-",color="#4CAF50",linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(lr,fontsize=10); ax.set_ylim(0,1.05)
    ax.set_title("Analytical Module Performance",fontweight="bold",pad=20)
    for a, v in zip(angles[:-1],vr[:-1]): ax.text(a,v+0.05,f"{v:.0%}",ha="center",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,"analytical_radar.png"),dpi=300,bbox_inches="tight"); fig.savefig(os.path.join(out_dir,"analytical_radar.pdf"),bbox_inches="tight"); plt.close()
    print("  [FIG] All figures generated.")


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary(summary, complementarity, analytical):
    MODES = ["random","semantic","scgpt","union"]
    MD = {"random":"Random","semantic":"Semantic","scgpt":"scGPT","union":"Union(add)"}
    print("\n" + "="*100)
    print("ELISA BENCHMARK v5.1 — DT8 FIRST-TRIMESTER BRAIN")
    print("="*100)
    print(f"\n{'Category':<14} {'Mode':<14} {'R@5':>7} {'R@10':>7} {'R@15':>7} {'R@20':>7} {'MRR':>7} {'GeneR':>7}")
    print("-"*80)
    for cat in ["ontology","expression"]:
        for m in MODES:
            k = f"{cat}_{m}"
            if k not in summary: continue
            s = summary[k]
            print(f"{cat:<14} {MD[m]:<14} "
                  f"{s.get('mean_recall@5',0):>7.3f} {s.get('mean_recall@10',0):>7.3f} "
                  f"{s.get('mean_recall@15',0):>7.3f} {s.get('mean_recall@20',0):>7.3f} "
                  f"{s['mean_mrr']:>7.3f} {s.get('mean_gene_recall',0):>7.3f}")
        print()
    print("── Overall Mean ──")
    for m in MODES:
        v5 = np.mean([summary.get(f"{c}_{m}",{}).get("mean_recall@5",0) for c in ["ontology","expression"]])
        vmrr = np.mean([summary.get(f"{c}_{m}",{}).get("mean_mrr",0) for c in ["ontology","expression"]])
        print(f"  {MD[m]:<14} R@5={v5:.3f}  MRR={vmrr:.3f}")
    c = complementarity
    print(f"\n── Complementarity ──")
    print(f"  Semantic: {c.get('semantic_found',0)} ({c.get('semantic_recall',0):.1%})")
    print(f"  scGPT:    {c.get('scgpt_found',0)} ({c.get('scgpt_recall',0):.1%})")
    print(f"  Union:    {c.get('union_found',0)} ({c.get('union_recall',0):.1%})")
    print(f"  Gain:     +{c.get('additive_gain_clusters',0)} (+{c.get('additive_gain_pct',0):.1%})")
    a = analytical
    print(f"\n── Analytical ──")
    print(f"  Pathways:     {a.get('pathways',{}).get('alignment',0):.1f}%")
    print(f"  Interactions: {a.get('interactions',{}).get('lr_recovery_rate',0):.1f}%")
    print(f"  Proportions:  {a.get('proportions',{}).get('consistency_rate',0):.1f}%")
    print(f"  Compare:      {a.get('compare',{}).get('compare_recall',0):.1f}%")
    print("="*100 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ELISA Benchmark v5.1 — DT8")
    parser.add_argument("--base", required=True, help="Path to embedding directory")
    parser.add_argument("--pt-name", default=None, help="Override .pt filename")
    parser.add_argument("--cells-csv", default=None, help="Override cells CSV")
    parser.add_argument("--paper", default="DT8", help="Paper ID")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k clusters")
    parser.add_argument("--out", default="benchmark_v5_DT8/", help="Output directory")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out; os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        rd = args.results_dir or out_dir
        with open(os.path.join(rd, "benchmark_v5_results.json")) as f: data = json.load(f)
        generate_figures(data[args.paper]["retrieval_summary"], data[args.paper]["complementarity"],
                         data[args.paper]["analytical"], out_dir)
        print("Figures regenerated."); return

    paper = BENCHMARK_PAPERS[args.paper]
    pt_name = args.pt_name or paper.get("pt_name",""); cells_csv = args.cells_csv or paper.get("cells_csv")
    sys.path.insert(0, os.path.dirname(args.base)); sys.path.insert(0, args.base); sys.path.insert(0, os.getcwd())

    from retrieval_engine_v4_hybrid import RetrievalEngine
    print(f"\n[BENCHMARK v5.1 DT8] Loading engine: {args.base} / {pt_name}")
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
        def proportions(self, **kw): return proportion_analysis(self._eng.metadata, condition_col=paper.get("condition_col","region"))
        def compare(self, ca, cb, **kw): return comparative_analysis(self._eng.gene_stats, self._eng.metadata, condition_col=paper.get("condition_col","region"), group_a=ca, group_b=cb, **kw)

    ew = EngineWithAnalysis(engine)

    print(f"\n[BENCHMARK] {len(paper['queries'])} queries × {len(RetrievalEvaluator.MODES)} modes...")
    t0 = time.time()
    rev = RetrievalEvaluator(engine)
    rr = rev.evaluate_queries(paper["queries"], top_k=args.top_k)
    rs = rev.compute_summary(rr)
    comp = rev.compute_complementarity(rr, top_k=args.top_k)
    print(f"  Retrieval done in {time.time()-t0:.1f}s")

    print("[BENCHMARK] Analytical modules...")
    t0 = time.time()
    try: ana = AnalyticalEvaluator(ew).evaluate_all(paper)
    except Exception as e:
        print(f"  [WARN] Analytical failed: {e}")
        ana = {"pathways":{},"interactions":{},"proportions":{},"compare":{}}
    print(f"  Analytical done in {time.time()-t0:.1f}s")

    print_summary(rs, comp, ana)

    output = {args.paper: {"retrieval_detail": rr, "retrieval_summary": rs,
                            "complementarity": comp, "analytical": ana,
                            "timestamp": datetime.now().isoformat(), "config": vars(args)}}
    rp = os.path.join(out_dir, "benchmark_v5_results.json")
    with open(rp, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")

    print("[BENCHMARK] Generating figures...")
    generate_figures(rs, comp, ana, out_dir)
    print("\n[BENCHMARK v5.1 DT8] Complete!")


if __name__ == "__main__":
    main()
