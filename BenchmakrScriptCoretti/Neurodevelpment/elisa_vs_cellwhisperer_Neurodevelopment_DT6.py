#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA vs CellWhisperer — DT8 First-Trimester Human Brain (Mannens et al. Nature 2025)
======================================================================================
Head-to-Head comparison on first-trimester neurodevelopment scATAC+multiome data.

28 clusters:
   0: GABAergic neuron                 14: neuroblast (sensu Nematoda and Protostomia)
   1: Purkinje cell                    15: neuroblast (sensu Vertebrata)
   2: Schwann cell                     16: neuron
   3: committed oligodendrocyte prec.  17: oligodendrocyte
   4: dopaminergic neuron              18: oligodendrocyte precursor cell
   5: endothelial cell                 19: pericyte
   6: glioblast                        20: perivascular macrophage
   7: glutamatergic neuron             21: progenitor cell
   8: glycinergic neuron               22: radial glial cell
   9: immature T cell                  23: sensory neuron of dorsal root ganglion
  10: interneuron                      24: serotonergic neuron
  11: leukocyte                        25: unknown
  12: microglial cell                  26: vascular associated smooth muscle cell
  13: neural progenitor cell           27: vascular leptomeningeal cell

Step 1: Run ELISA benchmark first:
    python elisa_benchmark_v5_1_DT8.py \\
        --base /path/to/embeddings \\
        --pt-name fused_DT8_Brain.pt \\
        --paper DT8 \\
        --out results_DT8/

Step 2: Run this script (in cellwhisperer env):
    conda activate cellwhisperer
    python elisa_vs_cellwhisperer_DT8.py \\
        --elisa-results results_DT8/benchmark_v5_results.json \\
        --cw-npz /path/to/cellwhisperer/full_output.npz \\
        --cw-leiden /path/to/leiden_umap_embeddings.h5ad \\
        --cw-ckpt /path/to/cellwhisperer_clip_v1.ckpt \\
        --cf-h5ad /path/to/read_count_table.h5ad \\
        --out comparison_DT8/
"""

import os, sys, json, argparse
from datetime import datetime
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

ALL_CLUSTER_NAMES = [
    GABA, PURK, SCHWANN, cOPC, DOPA, ENDO, GLIOB, GLUT, GLYC, IMMT,
    INTER, LEUK, MICRO, NPC, NB_NEMA, NB_VERT, NEURON, OLIGO, OPC,
    PERI, PVMAC, PROG, RGC, SENS, SERO, UNK, VSMC, VLMC,
]


# ============================================================
# QUERIES — 50 ontology + 50 expression (same as benchmark)
# ============================================================

QUERIES = [
    # ══════════════════════════════════════════════
    # ONTOLOGY QUERIES (Q01–Q50)
    # ══════════════════════════════════════════════

    # --- GABAergic neurons (Q01–Q08) ---
    {"id":"Q01","text":"GABAergic inhibitory neuron differentiation in developing human midbrain",
     "category":"ontology","expected_clusters":[GABA,INTER],
     "expected_genes":["GAD1","GAD2","SLC32A1","OTX2","GATA2"]},
    {"id":"Q02","text":"midbrain GABAergic neuron OTX2 GATA2 TAL2 transcription factor expression",
     "category":"ontology","expected_clusters":[GABA],
     "expected_genes":["OTX2","GATA2","TAL2","SOX14","GAD2"]},
    {"id":"Q03","text":"cortical interneuron derived from medial ganglionic eminence LHX6 DLX2",
     "category":"ontology","expected_clusters":[INTER,GABA],
     "expected_genes":["LHX6","DLX2","DLX5","GAD1","SST","PVALB"]},
    {"id":"Q04","text":"interneuron diversity parvalbumin somatostatin VIP subtypes developing cortex",
     "category":"ontology","expected_clusters":[INTER,GABA],
     "expected_genes":["PVALB","SST","VIP","LAMP5","SNCG","ADARB2"]},
    {"id":"Q05","text":"TAL2 expressing midbrain GABAergic neurons linked to major depressive disorder",
     "category":"ontology","expected_clusters":[GABA],
     "expected_genes":["TAL2","GAD2","OTX2","GATA2","SOX14"]},
    {"id":"Q06","text":"lateral and caudal ganglionic eminence interneuron migration in telencephalon",
     "category":"ontology","expected_clusters":[INTER,GABA],
     "expected_genes":["DLX2","DLX5","MEIS2","SNCG","ADARB2"]},
    {"id":"Q07","text":"medial ganglionic eminence derived parvalbumin somatostatin interneuron",
     "category":"ontology","expected_clusters":[INTER],
     "expected_genes":["LHX6","PVALB","SST","DLX2","GAD1"]},
    {"id":"Q08","text":"SOX14 expressing midbrain GABAergic neuron thalamic migration",
     "category":"ontology","expected_clusters":[GABA],
     "expected_genes":["SOX14","OTX2","GATA2","GAD2","TAL2"]},

    # --- Glutamatergic neurons (Q09–Q15) ---
    {"id":"Q09","text":"glutamatergic excitatory neuron in developing human telencephalon cortex",
     "category":"ontology","expected_clusters":[GLUT,NEURON],
     "expected_genes":["SLC17A7","SLC17A6","SATB2","TBR1","EMX2"]},
    {"id":"Q10","text":"telencephalic glutamatergic neuron LHX2 BHLHE22 cortical layer specification",
     "category":"ontology","expected_clusters":[GLUT],
     "expected_genes":["LHX2","BHLHE22","CUX1","CUX2","RORB","SLC17A7"]},
    {"id":"Q11","text":"hindbrain glutamatergic neuron ATOH1 MEIS1 cerebellar granule cell",
     "category":"ontology","expected_clusters":[GLUT,NEURON],
     "expected_genes":["ATOH1","MEIS1","MEIS2","SLC17A6"]},
    {"id":"Q12","text":"deep layer cortical neuron FEZF2 BCL11B corticospinal projection",
     "category":"ontology","expected_clusters":[GLUT,NEURON],
     "expected_genes":["FEZF2","BCL11B","TBR1","SLC17A7"]},
    {"id":"Q13","text":"SATB2 expressing telencephalic excitatory neuron callosal projection",
     "category":"ontology","expected_clusters":[GLUT],
     "expected_genes":["SATB2","SLC17A7","CUX2","LHX2"]},
    {"id":"Q14","text":"upper layer cortical neuron CUX1 CUX2 RORB intracortical connectivity",
     "category":"ontology","expected_clusters":[GLUT],
     "expected_genes":["CUX1","CUX2","RORB","LHX2","BHLHE22"]},
    {"id":"Q15","text":"EMX2 transcription factor dorsal telencephalon glutamatergic identity",
     "category":"ontology","expected_clusters":[GLUT,RGC],
     "expected_genes":["EMX2","LHX2","SLC17A7","PAX6"]},

    # --- Purkinje lineage (Q16–Q21) ---
    {"id":"Q16","text":"Purkinje cell differentiation in developing cerebellum PTF1A ESRRB lineage",
     "category":"ontology","expected_clusters":[PURK,NB_VERT],
     "expected_genes":["PTF1A","ESRRB","PCP4","LHX5","TFAP2B"]},
    {"id":"Q17","text":"Purkinje neuron ESRRB oestrogen-related nuclear receptor cerebellum specific",
     "category":"ontology","expected_clusters":[PURK],
     "expected_genes":["ESRRB","PCP4","RORA","FOXP2","EBF3"]},
    {"id":"Q18","text":"cerebellar Purkinje progenitor PTF1A ASCL1 NEUROG2 ventricular zone",
     "category":"ontology","expected_clusters":[PURK,PROG,NPC],
     "expected_genes":["PTF1A","ASCL1","NEUROG2","PAX2","NHLH1"]},
    {"id":"Q19","text":"TFAP2B LHX5 activation of ESRRB enhancer in Purkinje neuroblast",
     "category":"ontology","expected_clusters":[PURK,NB_VERT],
     "expected_genes":["TFAP2B","LHX5","ESRRB","NHLH2","PAX2"]},
    {"id":"Q20","text":"RORA FOXP2 EBF3 late Purkinje maturation gene regulatory network",
     "category":"ontology","expected_clusters":[PURK],
     "expected_genes":["RORA","FOXP2","EBF3","PCP4","ESRRB","LHX1"]},
    {"id":"Q21","text":"cerebellar granule neuron ATOH1 MEIS1 external granular layer",
     "category":"ontology","expected_clusters":[GLUT,NEURON],
     "expected_genes":["ATOH1","MEIS1","MEIS2","SLC17A6"]},

    # --- Radial glia / progenitors (Q22–Q27) ---
    {"id":"Q22","text":"radial glial cell neural stem cell SOX2 PAX6 NES in developing brain",
     "category":"ontology","expected_clusters":[RGC,NPC,PROG],
     "expected_genes":["SOX2","PAX6","NES","VIM","HES1","HES5"]},
    {"id":"Q23","text":"radial glia to glioblast transition NFI factor maturation NFIA NFIB NFIX",
     "category":"ontology","expected_clusters":[RGC,GLIOB],
     "expected_genes":["NFIA","NFIB","NFIX","SOX2","FABP7"]},
    {"id":"Q24","text":"neural progenitor cell proliferation and neurogenesis in ventricular zone",
     "category":"ontology","expected_clusters":[NPC,RGC,PROG],
     "expected_genes":["SOX2","PAX6","HES1","HES5","NES"]},
    {"id":"Q25","text":"loss of stemness and glial fate restriction by NFI transcription factors",
     "category":"ontology","expected_clusters":[GLIOB,RGC],
     "expected_genes":["NFIA","NFIB","NFIX","GFAP","AQP4"]},
    {"id":"Q26","text":"progenitor cell dividing in developing human brain VIM HES1 proliferating",
     "category":"ontology","expected_clusters":[PROG,NPC,RGC],
     "expected_genes":["SOX2","VIM","HES1","NES","PAX6"]},
    {"id":"Q27","text":"Notch signaling DLL1 JAG1 NOTCH1 lateral inhibition neurogenesis",
     "category":"ontology","expected_clusters":[RGC,NPC,GLIOB],
     "expected_genes":["DLL1","JAG1","NOTCH1","NOTCH2","HES1","HES5"]},

    # --- Glioblast (Q28–Q29) ---
    {"id":"Q28","text":"glioblast astrocyte precursor GFAP S100B AQP4 BCAN TNC fetal brain",
     "category":"ontology","expected_clusters":[GLIOB],
     "expected_genes":["GFAP","S100B","AQP4","BCAN","TNC","ALDH1L1"]},
    {"id":"Q29","text":"astrocyte maturation and glial scar markers in developing brain",
     "category":"ontology","expected_clusters":[GLIOB],
     "expected_genes":["GFAP","S100B","AQP4","ALDH1L1","BCAN"]},

    # --- OPC / oligodendrocyte (Q30–Q32) ---
    {"id":"Q30","text":"oligodendrocyte precursor cell OLIG2 PDGFRA SOX10 specification",
     "category":"ontology","expected_clusters":[OPC,cOPC,OLIGO],
     "expected_genes":["OLIG2","PDGFRA","SOX10","CSPG4","OLIG1"]},
    {"id":"Q31","text":"oligodendrocyte differentiation MBP MOG PLP1 myelination fetal brain",
     "category":"ontology","expected_clusters":[OLIGO,cOPC],
     "expected_genes":["MBP","MOG","PLP1","MAG","SOX10"]},
    {"id":"Q32","text":"committed oligodendrocyte precursor SOX10 lineage commitment",
     "category":"ontology","expected_clusters":[cOPC,OPC],
     "expected_genes":["SOX10","OLIG1","OLIG2","PDGFRA"]},

    # --- Dopaminergic / serotonergic (Q33–Q35) ---
    {"id":"Q33","text":"dopaminergic neuron midbrain TH NR4A2 substantia nigra ventral tegmental area",
     "category":"ontology","expected_clusters":[DOPA],
     "expected_genes":["TH","DDC","SLC6A3","SLC18A2","NR4A2","LMX1A"]},
    {"id":"Q34","text":"serotonergic neuron raphe nucleus TPH2 SLC6A4 FEV brainstem",
     "category":"ontology","expected_clusters":[SERO],
     "expected_genes":["TPH2","SLC6A4","FEV"]},
    {"id":"Q35","text":"FOXA2 LMX1A floor plate derived dopaminergic neuron specification",
     "category":"ontology","expected_clusters":[DOPA],
     "expected_genes":["FOXA2","LMX1A","NR4A2","TH","DDC"]},

    # --- Vascular (Q36–Q39) ---
    {"id":"Q36","text":"endothelial cell blood-brain barrier CLDN5 PECAM1 CDH5 fetal brain",
     "category":"ontology","expected_clusters":[ENDO],
     "expected_genes":["CLDN5","PECAM1","CDH5","ERG","FLT1"]},
    {"id":"Q37","text":"pericyte PDGFRB RGS5 FOXF2 cerebral vasculature developing brain",
     "category":"ontology","expected_clusters":[PERI,VSMC],
     "expected_genes":["PDGFRB","RGS5","FOXF2","ACTA2"]},
    {"id":"Q38","text":"vascular leptomeningeal cell FOXC1 meningeal fibroblast DCN COL1A1",
     "category":"ontology","expected_clusters":[VLMC],
     "expected_genes":["DCN","COL1A1","FOXC1","FOXF2","LUM"]},
    {"id":"Q39","text":"vascular smooth muscle cell ACTA2 MYH11 cerebral artery",
     "category":"ontology","expected_clusters":[VSMC,PERI],
     "expected_genes":["ACTA2","MYH11","PDGFRB"]},

    # --- Immune (Q40–Q42) ---
    {"id":"Q40","text":"microglial cell CX3CR1 P2RY12 TMEM119 brain resident macrophage",
     "category":"ontology","expected_clusters":[MICRO,PVMAC],
     "expected_genes":["AIF1","CX3CR1","P2RY12","TMEM119","HEXB"]},
    {"id":"Q41","text":"border-associated macrophage RUNX1 haematopoietic origin fetal brain",
     "category":"ontology","expected_clusters":[PVMAC,MICRO,LEUK],
     "expected_genes":["RUNX1","AIF1","CSF1R","SPI1"]},
    {"id":"Q42","text":"immature T cell and leukocyte infiltration in developing fetal brain",
     "category":"ontology","expected_clusters":[IMMT,LEUK],
     "expected_genes":["CD3D","CD3E"]},

    # --- Other cell types (Q43–Q46) ---
    {"id":"Q43","text":"Schwann cell MPZ CDH19 SOX10 neural crest derived myelinating peripheral glial",
     "category":"ontology","expected_clusters":[SCHWANN],
     "expected_genes":["MPZ","CDH19","SOX10"]},
    {"id":"Q44","text":"sensory neuron dorsal root ganglion NTRK1 ISL1 peripheral nervous system",
     "category":"ontology","expected_clusters":[SENS],
     "expected_genes":["NTRK1","NTRK2","ISL1"]},
    {"id":"Q45","text":"glycinergic neuron SLC6A5 GLRA1 inhibitory spinal cord hindbrain",
     "category":"ontology","expected_clusters":[GLYC],
     "expected_genes":["SLC6A5","GLRA1"]},
    {"id":"Q46","text":"neuroblast immature migrating neuron fetal cortex RBFOX3 NEFM",
     "category":"ontology","expected_clusters":[NB_VERT,NB_NEMA,NEURON],
     "expected_genes":["RBFOX3","NEFM","NEFL","SNAP25"]},

    # --- GWAS / disease (Q47–Q50) ---
    {"id":"Q47","text":"major depressive disorder MDD midbrain GABAergic neuron NEGR1 LRFN5",
     "category":"ontology","expected_clusters":[GABA],
     "expected_genes":["NEGR1","BTN3A2","LRFN5","SCN8A","OTX2"]},
    {"id":"Q48","text":"schizophrenia cortical interneuron medial ganglionic eminence SATB2",
     "category":"ontology","expected_clusters":[INTER,GABA,GLUT],
     "expected_genes":["LHX6","DLX2","SATB2","GAD1"]},
    {"id":"Q49","text":"attention deficit hyperactivity disorder ADHD cerebellar Purkinje",
     "category":"ontology","expected_clusters":[PURK,NB_VERT,GABA],
     "expected_genes":["ATOH1","ESRRB","GAD2"]},
    {"id":"Q50","text":"autism spectrum disorder hindbrain neuroblast brainstem involvement",
     "category":"ontology","expected_clusters":[NB_VERT,NB_NEMA,GLUT],
     "expected_genes":["RBFOX3","SLC17A6","NEFM"]},

    # ══════════════════════════════════════════════
    # EXPRESSION QUERIES (Q51–Q100)
    # ══════════════════════════════════════════════

    # --- GABAergic signatures (Q51–Q56) ---
    {"id":"Q51","text":"GAD1 GAD2 SLC32A1 DLX2 DLX5 LHX6",
     "category":"expression","expected_clusters":[GABA,INTER],
     "expected_genes":["GAD1","GAD2","SLC32A1","DLX2","LHX6"]},
    {"id":"Q52","text":"OTX2 GATA2 TAL2 SOX14 GAD2 SLC32A1",
     "category":"expression","expected_clusters":[GABA],
     "expected_genes":["OTX2","GATA2","TAL2","SOX14","GAD2"]},
    {"id":"Q53","text":"PVALB SST VIP LAMP5 SNCG ADARB2",
     "category":"expression","expected_clusters":[INTER,GABA],
     "expected_genes":["PVALB","SST","VIP","LAMP5","SNCG"]},
    {"id":"Q54","text":"DLX1 DLX2 DLX5 DLX6 MEIS2 LHX6",
     "category":"expression","expected_clusters":[INTER,GABA],
     "expected_genes":["DLX1","DLX2","DLX5","DLX6","MEIS2"]},
    {"id":"Q55","text":"GAD1 GAD2 SLC32A1 TFAP2B OTX2",
     "category":"expression","expected_clusters":[GABA,PURK],
     "expected_genes":["GAD1","GAD2","SLC32A1","TFAP2B","OTX2"]},
    {"id":"Q56","text":"TAL2 SOX14 GAD2 OTX2 GATA2",
     "category":"expression","expected_clusters":[GABA],
     "expected_genes":["TAL2","SOX14","GAD2","OTX2","GATA2"]},

    # --- Glutamatergic signatures (Q57–Q61) ---
    {"id":"Q57","text":"SLC17A7 SLC17A6 SATB2 TBR1 FEZF2 BCL11B",
     "category":"expression","expected_clusters":[GLUT,NEURON],
     "expected_genes":["SLC17A7","SLC17A6","SATB2","TBR1","FEZF2"]},
    {"id":"Q58","text":"EMX2 LHX2 BHLHE22 CUX1 CUX2 RORB",
     "category":"expression","expected_clusters":[GLUT],
     "expected_genes":["EMX2","LHX2","BHLHE22","CUX1","CUX2"]},
    {"id":"Q59","text":"ATOH1 MEIS1 MEIS2 SLC17A6 RBFOX3",
     "category":"expression","expected_clusters":[GLUT,NEURON],
     "expected_genes":["ATOH1","MEIS1","MEIS2","SLC17A6"]},
    {"id":"Q60","text":"FEZF2 BCL11B TBR1 SATB2 SLC17A7",
     "category":"expression","expected_clusters":[GLUT,NEURON],
     "expected_genes":["FEZF2","BCL11B","TBR1","SATB2","SLC17A7"]},
    {"id":"Q61","text":"CUX1 CUX2 RORB LHX2 BHLHE22 EMX2",
     "category":"expression","expected_clusters":[GLUT],
     "expected_genes":["CUX1","CUX2","RORB","LHX2","EMX2"]},

    # --- Purkinje signatures (Q62–Q65) ---
    {"id":"Q62","text":"PTF1A ASCL1 NEUROG2 NHLH1 NHLH2 TFAP2B",
     "category":"expression","expected_clusters":[PURK,NB_VERT,NPC],
     "expected_genes":["PTF1A","ASCL1","NEUROG2","NHLH1","TFAP2B"]},
    {"id":"Q63","text":"ESRRB RORA PCP4 FOXP2 EBF3 LHX5",
     "category":"expression","expected_clusters":[PURK],
     "expected_genes":["ESRRB","RORA","PCP4","FOXP2","EBF3","LHX5"]},
    {"id":"Q64","text":"LHX5 LHX1 PAX2 TFAP2B DMBX1 NHLH2",
     "category":"expression","expected_clusters":[PURK,NB_VERT],
     "expected_genes":["LHX5","LHX1","PAX2","TFAP2B","DMBX1"]},
    {"id":"Q65","text":"ESRRB PCP4 RORA EBF1 EBF3 FOXP2 LHX1",
     "category":"expression","expected_clusters":[PURK],
     "expected_genes":["ESRRB","PCP4","RORA","EBF1","FOXP2","LHX1"]},

    # --- Radial glia / progenitor (Q66–Q69) ---
    {"id":"Q66","text":"SOX2 PAX6 NES VIM HES1 HES5 FABP7",
     "category":"expression","expected_clusters":[RGC,NPC,PROG],
     "expected_genes":["SOX2","PAX6","NES","VIM","HES1","FABP7"]},
    {"id":"Q67","text":"NFIA NFIB NFIX SOX9 FABP7",
     "category":"expression","expected_clusters":[RGC,GLIOB],
     "expected_genes":["NFIA","NFIB","NFIX","SOX9","FABP7"]},
    {"id":"Q68","text":"SOX2 HES1 HES5 PAX6 NES VIM",
     "category":"expression","expected_clusters":[RGC,NPC,PROG],
     "expected_genes":["SOX2","HES1","HES5","PAX6","NES"]},
    {"id":"Q69","text":"NOTCH1 NOTCH2 DLL1 JAG1 HES1 HES5",
     "category":"expression","expected_clusters":[RGC,NPC,GLIOB],
     "expected_genes":["NOTCH1","DLL1","JAG1","HES1","HES5"]},

    # --- Glioblast (Q70) ---
    {"id":"Q70","text":"GFAP S100B AQP4 ALDH1L1 BCAN TNC",
     "category":"expression","expected_clusters":[GLIOB],
     "expected_genes":["GFAP","S100B","AQP4","ALDH1L1","BCAN","TNC"]},

    # --- OPC / oligodendrocyte (Q71–Q73) ---
    {"id":"Q71","text":"OLIG1 OLIG2 SOX10 PDGFRA CSPG4",
     "category":"expression","expected_clusters":[OPC,cOPC],
     "expected_genes":["OLIG1","OLIG2","SOX10","PDGFRA","CSPG4"]},
    {"id":"Q72","text":"MBP MOG PLP1 MAG SOX10",
     "category":"expression","expected_clusters":[OLIGO,cOPC],
     "expected_genes":["MBP","MOG","PLP1","MAG","SOX10"]},
    {"id":"Q73","text":"OLIG2 SOX10 PDGFRA NKX2-2 OLIG1",
     "category":"expression","expected_clusters":[OPC,cOPC,OLIGO],
     "expected_genes":["OLIG2","SOX10","PDGFRA","OLIG1"]},

    # --- Dopaminergic (Q74–Q75) ---
    {"id":"Q74","text":"TH DDC SLC6A3 SLC18A2 NR4A2 LMX1A FOXA2",
     "category":"expression","expected_clusters":[DOPA],
     "expected_genes":["TH","DDC","SLC6A3","SLC18A2","NR4A2","LMX1A"]},
    {"id":"Q75","text":"FOXA2 LMX1A NR4A2 TH DDC SLC18A2",
     "category":"expression","expected_clusters":[DOPA],
     "expected_genes":["FOXA2","LMX1A","NR4A2","TH","DDC"]},

    # --- Serotonergic (Q76) ---
    {"id":"Q76","text":"TPH2 SLC6A4 FEV DDC SLC18A2",
     "category":"expression","expected_clusters":[SERO,DOPA],
     "expected_genes":["TPH2","SLC6A4","FEV","DDC"]},

    # --- Glycinergic (Q77) ---
    {"id":"Q77","text":"SLC6A5 GLRA1 SLC32A1 GAD1",
     "category":"expression","expected_clusters":[GLYC,GABA],
     "expected_genes":["SLC6A5","GLRA1","SLC32A1"]},

    # --- Pan-neuronal (Q78–Q79) ---
    {"id":"Q78","text":"RBFOX3 SNAP25 SYT1 NEFM NEFL TUBB3",
     "category":"expression","expected_clusters":[NEURON,GLUT,GABA,NB_VERT],
     "expected_genes":["RBFOX3","SNAP25","SYT1","NEFM","NEFL"]},
    {"id":"Q79","text":"NEFM NEFL MAP2 TUBB3 SYT1",
     "category":"expression","expected_clusters":[NEURON,NB_VERT,GLUT],
     "expected_genes":["NEFM","NEFL","MAP2","SYT1"]},

    # --- Endothelial (Q80) ---
    {"id":"Q80","text":"CLDN5 PECAM1 CDH5 ERG FLT1 VWF",
     "category":"expression","expected_clusters":[ENDO],
     "expected_genes":["CLDN5","PECAM1","CDH5","ERG","FLT1"]},

    # --- Pericyte / VSMC (Q81–Q82) ---
    {"id":"Q81","text":"PDGFRB RGS5 ACTA2 MYH11 COL1A2",
     "category":"expression","expected_clusters":[PERI,VSMC],
     "expected_genes":["PDGFRB","RGS5","ACTA2","MYH11"]},
    {"id":"Q82","text":"ACTA2 MYH11 PDGFRB TAGLN",
     "category":"expression","expected_clusters":[VSMC,PERI],
     "expected_genes":["ACTA2","MYH11","PDGFRB"]},

    # --- VLMC (Q83–Q84) ---
    {"id":"Q83","text":"DCN LUM COL1A1 COL1A2 FOXC1 COL3A1",
     "category":"expression","expected_clusters":[VLMC],
     "expected_genes":["DCN","LUM","COL1A1","COL1A2","FOXC1"]},
    {"id":"Q84","text":"FOXC1 FOXF2 DCN COL1A2 LUM",
     "category":"expression","expected_clusters":[VLMC,PERI],
     "expected_genes":["FOXC1","FOXF2","DCN","COL1A2"]},

    # --- Immune (Q85–Q88) ---
    {"id":"Q85","text":"AIF1 CX3CR1 P2RY12 TMEM119 HEXB CSF1R",
     "category":"expression","expected_clusters":[MICRO,PVMAC],
     "expected_genes":["AIF1","CX3CR1","P2RY12","TMEM119","HEXB"]},
    {"id":"Q86","text":"RUNX1 SPI1 CSF1R AIF1 CD68",
     "category":"expression","expected_clusters":[MICRO,PVMAC,LEUK],
     "expected_genes":["RUNX1","SPI1","CSF1R","AIF1"]},
    {"id":"Q87","text":"AIF1 HEXB P2RY12 TMEM119 CX3CR1",
     "category":"expression","expected_clusters":[MICRO],
     "expected_genes":["AIF1","HEXB","P2RY12","TMEM119","CX3CR1"]},
    {"id":"Q88","text":"CD3D CD3E CD3G PTPRC CD2",
     "category":"expression","expected_clusters":[IMMT,LEUK],
     "expected_genes":["CD3D","CD3E"]},

    # --- Schwann / sensory (Q89–Q90) ---
    {"id":"Q89","text":"MPZ CDH19 SOX10 MBP PLP1",
     "category":"expression","expected_clusters":[SCHWANN],
     "expected_genes":["MPZ","CDH19","SOX10"]},
    {"id":"Q90","text":"NTRK1 NTRK2 ISL1 PRPH SNAP25",
     "category":"expression","expected_clusters":[SENS],
     "expected_genes":["NTRK1","NTRK2","ISL1"]},

    # --- Neuroblast (Q91–Q92) ---
    {"id":"Q91","text":"RBFOX3 SLC17A6 GAD2 NEFM SNAP25",
     "category":"expression","expected_clusters":[NB_VERT,NEURON,GLUT,GABA],
     "expected_genes":["RBFOX3","SLC17A6","GAD2","NEFM"]},
    {"id":"Q92","text":"NEFM NEFL RBFOX3 TUBB3 DCX",
     "category":"expression","expected_clusters":[NB_VERT,NB_NEMA,NEURON],
     "expected_genes":["NEFM","NEFL","RBFOX3"]},

    # --- MDD / GWAS (Q93–Q95) ---
    {"id":"Q93","text":"NEGR1 BTN3A2 LRFN5 SCN8A RGS6 MYCN",
     "category":"expression","expected_clusters":[GABA,NEURON],
     "expected_genes":["NEGR1","BTN3A2","LRFN5","SCN8A","MYCN"]},
    {"id":"Q94","text":"OTX2 GATA2 MEIS2 PRDM10 MYCN",
     "category":"expression","expected_clusters":[GABA,GLUT],
     "expected_genes":["OTX2","GATA2","MEIS2","PRDM10","MYCN"]},
    {"id":"Q95","text":"CTCF MECP2 YY1 RAD21 SMC3",
     "category":"expression","expected_clusters":[NEURON,RGC,GLIOB],
     "expected_genes":["CTCF","MECP2"]},

    # --- Developmental signaling (Q96–Q98) ---
    {"id":"Q96","text":"SHH PTCH1 GLI1 GLI2 FOXA2 NKX2-1",
     "category":"expression","expected_clusters":[NPC,RGC,DOPA],
     "expected_genes":["SHH","PTCH1","FOXA2"]},
    {"id":"Q97","text":"WNT5A CTNNB1 LEF1 TCF7L2 AXIN2",
     "category":"expression","expected_clusters":[RGC,NPC,PROG],
     "expected_genes":["WNT5A","CTNNB1","LEF1","TCF7L2"]},
    {"id":"Q98","text":"BMP4 BMPR1A SMAD1 ID1 ID3",
     "category":"expression","expected_clusters":[GLIOB,OPC,RGC],
     "expected_genes":["BMP4","BMPR1A","SMAD1"]},

    # --- Mixed / broad (Q99–Q100) ---
    {"id":"Q99","text":"VEGFA KDR FLT1 PDGFB PDGFRB CLDN5",
     "category":"expression","expected_clusters":[ENDO,PERI],
     "expected_genes":["VEGFA","KDR","FLT1","PDGFB","CLDN5"]},
    {"id":"Q100","text":"SOX2 PAX6 OLIG2 GFAP RBFOX3 GAD2 SLC17A7",
     "category":"expression","expected_clusters":[RGC,OPC,GLIOB,NEURON,GABA,GLUT],
     "expected_genes":["SOX2","PAX6","OLIG2","GFAP","RBFOX3","GAD2","SLC17A7"]},
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
            from cellwhisperer.jointemb.cellwhisperer_lightning import TranscriptomeTextDualEncoderLightning
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

    def score_query(self, query_text, top_k=28):
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

    print(f"\n[ELISA] Loading results from {elisa_json_path}")
    with open(elisa_json_path) as f:
        elisa_all = json.load(f)
    paper = elisa_all[paper_id]
    elisa_detail = paper["retrieval_detail"]
    elisa_summary = paper["retrieval_summary"]
    elisa_analytical = paper.get("analytical", {})

    # 28 clusters → RECALL_KS
    RECALL_KS = [5, 10, 15, 20]

    print("\n" + "=" * 70)
    print("[CW] Running CellWhisperer on 100 queries (DT8)...")
    print("=" * 70)

    cw_results = {"ontology": [], "expression": []}
    for q in QUERIES:
        ranked = cw_scorer.score_query(q["text"], top_k=28)
        entry = {"query_id": q["id"], "query_text": q["text"],
                 "expected": q["expected_clusters"],
                 "retrieved_top20": ranked[:20],
                 "mrr": round(mrr(q["expected_clusters"], ranked), 4)}
        for k in RECALL_KS:
            entry[f"recall@{k}"] = round(cluster_recall_at_k(q["expected_clusters"], ranked, k), 4)
        cw_results[q["category"]].append(entry)
        print(f"  [{q['id']}] R@5={entry['recall@5']:.2f}  MRR={entry['mrr']:.2f}  Top3={ranked[:3]}")

    cw_agg = {}
    for cat in ["ontology", "expression"]:
        e = cw_results[cat]
        cw_agg[cat] = {"mean_mrr": round(np.mean([x["mrr"] for x in e]), 4),
                       "mean_gene_recall": 0.0}
        for k in RECALL_KS:
            cw_agg[cat][f"mean_recall@{k}"] = round(np.mean([x[f"recall@{k}"] for x in e]), 4)

    # Modes — no BM25, no annotation_only
    ALL_MODES = ["random", "cellwhisperer_real", "semantic", "scgpt", "union"]
    MC = {"random": "#9E9E9E", "cellwhisperer_real": "#E91E63",
          "semantic": "#2196F3", "scgpt": "#FF9800", "union": "#4CAF50"}
    ML = {"random": "Random", "cellwhisperer_real": "CellWhisp.",
          "semantic": "Semantic", "scgpt": "scGPT", "union": "Union(S+G)"}

    def gm(cat, mode, mk):
        if mode == "cellwhisperer_real":
            return cw_agg[cat].get(mk, 0)
        return elisa_summary.get(f"{cat}_{mode}", {}).get(mk, 0)

    # ── Console ──
    print("\n" + "=" * 90)
    print("ELISA vs CellWhisperer — DT8: First-Trimester Brain (100 queries)")
    print("=" * 90)
    print(f"\n{'Category':<14} {'Mode':<16} {'R@5':>7} {'R@10':>7} {'R@15':>7} {'R@20':>7} {'MRR':>7} {'GeneR':>7}")
    print("-" * 85)
    for cat in ["ontology", "expression"]:
        for mode in ALL_MODES:
            r5 = gm(cat, mode, "mean_recall@5")
            r10 = gm(cat, mode, "mean_recall@10")
            r15 = gm(cat, mode, "mean_recall@15")
            r20 = gm(cat, mode, "mean_recall@20")
            mmr = gm(cat, mode, "mean_mrr")
            gr = gm(cat, mode, "mean_gene_recall")
            gr_s = "  N/A" if mode == "cellwhisperer_real" else f"{gr:.3f}"
            print(f"{cat:<14} {ML.get(mode,mode):<16} {r5:>7.3f} {r10:>7.3f} {r15:>7.3f} {r20:>7.3f} {mmr:>7.3f} {gr_s:>7}")
        print()
    print("── Overall ──")
    for mode in ALL_MODES:
        r5 = np.mean([gm(c, mode, "mean_recall@5") for c in ["ontology", "expression"]])
        mmr = np.mean([gm(c, mode, "mean_mrr") for c in ["ontology", "expression"]])
        print(f"  {ML.get(mode,mode):<16} R@5={r5:.3f}  MRR={mmr:.3f}")
    ana = elisa_analytical
    print("\n── Analytical (ELISA only) ──")
    print(f"  Pathways:     {ana.get('pathways',{}).get('alignment',0):.1f}%")
    print(f"  Interactions: {ana.get('interactions',{}).get('lr_recovery_rate',0):.1f}%")
    print(f"  Proportions:  {ana.get('proportions',{}).get('consistency_rate',0):.1f}%")
    print(f"  Compare:      {ana.get('compare',{}).get('compare_recall',0):.1f}%")
    print("=" * 90)

    # ── FIGURES ──
    cats = ["ontology", "expression"]
    cat_titles = ["Ontology Queries\n(concept-level)", "Expression Queries\n(gene-signature)"]

    # Fig 1: Recall@5
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES))
        vals = [gm(cat, m, "mean_recall@5") for m in ALL_MODES]
        bars = ax.bar(x, vals, color=[MC[m] for m in ALL_MODES], alpha=0.85, edgecolor="white", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([ML[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15); ax.set_ylabel("Mean Cluster Recall@5")
        ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("ELISA vs CellWhisperer — DT8: First-Trimester Brain", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_recall5.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig1_recall5.pdf"), bbox_inches="tight")
    plt.close(); print("\n[FIG] fig1_recall5.png")

    # Fig 2: All metrics
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, mk, mt in zip(axes,
                           ["mean_recall@5", "mean_recall@10", "mean_mrr"],
                           ["Recall@5", "Recall@10", "MRR"]):
        x = np.arange(len(ALL_MODES)); w = 0.35
        ont = [gm("ontology", m, mk) for m in ALL_MODES]
        exp = [gm("expression", m, mk) for m in ALL_MODES]
        ax.bar(x - w/2, ont, w, label="Ontology", alpha=0.85,
               color=[MC[m] for m in ALL_MODES], edgecolor="white")
        ax.bar(x + w/2, exp, w, label="Expression", alpha=0.45,
               color=[MC[m] for m in ALL_MODES], edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([ML[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.15); ax.set_title(mt, fontsize=13, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_all_metrics.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_all_metrics.pdf"), bbox_inches="tight")
    plt.close(); print("[FIG] fig2_all_metrics.png")

    # Fig 3: Cluster vs Gene recall
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, cat, ttl in zip(axes, cats, cat_titles):
        x = np.arange(len(ALL_MODES)); w = 0.35
        cv = [gm(cat, m, "mean_recall@5") for m in ALL_MODES]
        gv = [gm(cat, m, "mean_gene_recall") for m in ALL_MODES]
        ax.bar(x - w/2, cv, w, label="Cluster R@5", color=[MC[m] for m in ALL_MODES], alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, gv, w, label="Gene Recall", color=[MC[m] for m in ALL_MODES], alpha=0.45,
               edgecolor="black", linewidth=0.8, hatch="///")
        ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in ALL_MODES], fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(list(zip(*[ax.patches]))[0] if False else [], cv):
            pass  # labels on bars handled below
    legend_elements = [Patch(facecolor="#888", alpha=0.85, label="Cluster R@5"),
                       Patch(facecolor="#888", alpha=0.45, hatch="///", edgecolor="black", label="Gene Recall")]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=9)
    cw_idx = ALL_MODES.index("cellwhisperer_real")
    for ax in axes:
        ax.text(cw_idx + 0.175, 0.05, "N/A", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#E91E63")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_cluster_vs_gene.pdf"), bbox_inches="tight")
    plt.close(); print("[FIG] fig3_cluster_vs_gene.png")

    # Fig 4: Radar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    rl = ["Ont R@5", "Ont R@10", "Ont MRR", "Exp R@5", "Exp R@10", "Exp MRR"]
    N = len(rl)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    for mode, color, ls in [("cellwhisperer_real", "#E91E63", "--"),
                             ("semantic", "#2196F3", "-"),
                             ("scgpt", "#FF9800", "-"),
                             ("union", "#4CAF50", "-")]:
        v = []
        for cat in ["ontology", "expression"]:
            for mk in ["mean_recall@5", "mean_recall@10", "mean_mrr"]:
                v.append(gm(cat, mode, mk))
        v += v[:1]
        ax.plot(angles, v, linewidth=2.5, linestyle=ls, label=ML.get(mode, mode),
                color=color, marker="o", markersize=5)
        ax.fill(angles, v, alpha=0.06, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rl, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.set_title("ELISA vs CellWhisperer\n(DT8: First-Trimester Brain)", fontsize=13, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11,
              frameon=True, framealpha=0.9, edgecolor="#ccc")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_radar.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig4_radar.pdf"), bbox_inches="tight")
    plt.close(); print("[FIG] fig4_radar.png")

    # Fig 5: Gene recall + Analytical radar
    fig, (ax1, ax2_placeholder) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(ALL_MODES)); w = 0.35
    og = [gm("ontology", m, "mean_gene_recall") for m in ALL_MODES]
    eg = [gm("expression", m, "mean_gene_recall") for m in ALL_MODES]
    ax1.bar(x - w/2, og, w, label="Ontology", alpha=0.85,
            color=[MC[m] for m in ALL_MODES], edgecolor="white")
    ax1.bar(x + w/2, eg, w, label="Expression", alpha=0.45,
            color=[MC[m] for m in ALL_MODES], edgecolor="black", linewidth=0.5, hatch="//")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ML[m] for m in ALL_MODES], fontsize=9, rotation=15, ha="right")
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("Gene Recall")
    ax1.set_title("Gene-Level Evidence Delivery", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    ax1.text(cw_idx, 0.05, "N/A", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#E91E63")

    ax2_placeholder.set_axis_off()
    ax2b = fig.add_axes([0.55, 0.1, 0.4, 0.75], polar=True)
    al = ["Pathways", "Interactions\n(LR)", "Proportions", "Compare"]
    av = [ana.get("pathways", {}).get("alignment", 0) / 100,
          ana.get("interactions", {}).get("lr_recovery_rate", 0) / 100,
          ana.get("proportions", {}).get("consistency_rate", 0) / 100,
          ana.get("compare", {}).get("compare_recall", 0) / 100]
    aa = np.linspace(0, 2 * np.pi, len(al), endpoint=False).tolist()
    av_c = av + av[:1]; aa_c = aa + aa[:1]
    ax2b.fill(aa_c, av_c, alpha=0.25, color="#4CAF50")
    ax2b.plot(aa_c, av_c, "o-", color="#4CAF50", linewidth=2, label="ELISA")
    ax2b.plot(aa_c, [0]*len(aa_c), "--", color="#E91E63", linewidth=1.5, label="CellWhisp. (N/A)")
    ax2b.set_xticks(aa); ax2b.set_xticklabels(al, fontsize=9); ax2b.set_ylim(0, 1.05)
    ax2b.set_title("Analytical Modules\n(ELISA only)", fontsize=11, fontweight="bold", pad=15)
    ax2b.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig5_gene_analytical.pdf"), bbox_inches="tight")
    plt.close(); print("[FIG] fig5_gene_analytical.png")

    # Fig 6: Per-query (first 25)
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for ax, cat in zip(axes, ["ontology", "expression"]):
        cw_e = cw_results[cat]
        qids = [e["query_id"] for e in cw_e]
        cw_r5 = [e["recall@5"] for e in cw_e]
        el_union = elisa_detail.get(cat, {}).get("union", [])
        el_r5 = [e.get("recall@5", 0) for e in el_union] if el_union else [0]*len(qids)
        n = min(25, len(qids), len(el_r5))
        y = np.arange(n); h = 0.35
        ax.barh(y - h/2, el_r5[:n], h, label="ELISA Union", color="#4CAF50", alpha=0.8)
        ax.barh(y + h/2, cw_r5[:n], h, label="CellWhisperer", color="#E91E63", alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(qids[:n], fontsize=8)
        ax.set_xlabel("Recall@5"); ax.set_xlim(0, 1.1)
        ax.set_title(f"{cat.capitalize()} Queries (first 25)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    plt.suptitle("Per-Query: ELISA Union vs CellWhisperer (DT8)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_perquery.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig6_perquery.pdf"), bbox_inches="tight")
    plt.close(); print("[FIG] fig6_perquery.png")

    # ── Save JSON ──
    comparison = {}
    for cat in cats:
        comparison[cat] = {}
        for mode in ALL_MODES:
            comparison[cat][mode] = {
                mk: gm(cat, mode, mk)
                for mk in [f"mean_recall@{k}" for k in RECALL_KS] + ["mean_mrr"]
            }

    output = {
        "cellwhisperer": {"results": cw_results, "aggregated": cw_agg},
        "elisa_summary": elisa_summary,
        "elisa_analytical": elisa_analytical,
        "comparison": comparison,
        "modes": ALL_MODES,
        "n_queries": len(QUERIES),
        "dataset": "DT8_FirstTrimester_Brain",
        "paper": "Mannens et al. Nature 2025",
        "n_clusters": 28,
        "cluster_names": ALL_CLUSTER_NAMES,
        "timestamp": datetime.now().isoformat(),
    }
    rp = os.path.join(out_dir, "elisa_vs_cellwhisperer_DT8_results.json")
    with open(rp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {rp}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ELISA vs CellWhisperer — DT8 First-Trimester Brain")
    parser.add_argument("--elisa-results", required=True,
                        help="ELISA benchmark_v5_results.json from DT8")
    parser.add_argument("--cw-npz", required=True,
                        help="CellWhisperer full_output.npz")
    parser.add_argument("--cw-leiden", required=True,
                        help="CellWhisperer leiden_umap_embeddings.h5ad")
    parser.add_argument("--cw-ckpt", required=True,
                        help="cellwhisperer_clip_v1.ckpt")
    parser.add_argument("--cf-h5ad", required=True,
                        help="Original read_count_table.h5ad")
    parser.add_argument("--paper", default="DT8")
    parser.add_argument("--out", default="comparison_DT8/")
    args = parser.parse_args()

    cw = CellWhispererScorer(args.cw_npz, args.cw_leiden, args.cw_ckpt, args.cf_h5ad)
    run_comparison(cw, args.elisa_results, args.paper, args.out)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
