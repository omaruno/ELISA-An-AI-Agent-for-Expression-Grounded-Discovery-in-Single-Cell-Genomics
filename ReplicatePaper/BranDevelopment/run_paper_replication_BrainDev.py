#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELISA Paper Replication — Automated Batch Runner
=================================================
Target Paper:  Mannens, Hu, Lönnerberg et al. (2025) Nature 647, 179–186
               "Chromatin accessibility during human first-trimester
                neurodevelopment"

NOTE: This replication uses ONLY the scRNA-seq–derived findings from the
paper (gene expression, cell type markers, TF expression, pseudotime
expression). All scATAC-seq–only results (chromatin accessibility,
cCREs, CNN enhancer models, pycisTopic, GWAS LD score regression)
are excluded.

Uses retrieval_engine_v4_hybrid + elisa_analysis.

Usage:
    python run_paper_replication_DT8_Brain.py \\
        --h5ad /path/to/DT8_Brain.h5ad \\
        --out-dir /path/to/elisa_replication_DT8/ \\
        --base /path/to/embeddings/ \\
        --pt-name fused_DT8_Brain.pt \\
        --cluster-key cell_type \\
        --dataset-name "First-Trimester Human Brain — scRNA-seq Atlas"

28 clusters:
    GABAergic neuron, Purkinje cell, Schwann cell, committed oligodendrocyte
    precursor, dopaminergic neuron, endothelial cell, glioblast,
    glutamatergic neuron, glycinergic neuron, immature T cell, interneuron,
    leukocyte, microglial cell, neural progenitor cell,
    neuroblast (sensu Nematoda and Protostomia),
    neuroblast (sensu Vertebrata), neuron, oligodendrocyte,
    oligodendrocyte precursor cell, pericyte, perivascular macrophage,
    progenitor cell, radial glial cell,
    sensory neuron of dorsal root ganglion, serotonergic neuron, unknown,
    vascular associated smooth muscle cell, vascular leptomeningeal cell

Condition column: region (Telencephalon, Cerebellum, etc.)
"""

import os, sys, json, time, argparse, functools, re
from datetime import datetime

print = functools.partial(print, flush=True)


def setup_args():
    parser = argparse.ArgumentParser(
        description="ELISA Paper Replication — First-Trimester Brain (scRNA-seq)")
    parser.add_argument("--h5ad", default=None)
    parser.add_argument("--cluster-key", default="cell_type")
    parser.add_argument("--out-dir", default="elisa_replication_DT8")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dataset-name",
                        default="First-Trimester Human Brain — scRNA-seq Atlas")
    parser.add_argument("--base", required=True)
    parser.add_argument("--pt-name", required=True)
    parser.add_argument("--cells-csv", default="metadata_cells.csv")
    return parser.parse_args()


def init_elisa(args):
    from retrieval_engine_v4_hybrid import RetrievalEngine
    from elisa_analysis import (find_interactions, pathway_scoring,
                                proportion_analysis, comparative_analysis,
                                query_pathway)
    from elisa_report import ReportBuilder
    import elisa_viz as viz

    print("[INIT] Loading retrieval engine (v4 hybrid)...")
    raw_engine = RetrievalEngine(
        base=args.base, pt_name=args.pt_name, cells_csv=args.cells_csv)
    print(f"[INIT] {len(raw_engine.cluster_ids)} clusters loaded")

    class EngineWrapper:
        def __init__(self, eng):
            self._eng = eng
            self.cluster_ids = eng.cluster_ids
            self.gene_stats = eng.gene_stats
            self.metadata = eng.metadata
            self.cluster_metadata = getattr(eng, 'cluster_metadata', {})
            self.n = len(eng.cluster_ids)

        def query_semantic(self, text, top_k=5, with_genes=False):
            return self._eng.query_semantic(text, top_k=top_k, with_genes=with_genes)

        def query_hybrid(self, text, top_k=5, lambda_sem=0.5, with_genes=False, **kw):
            return self._eng.query_hybrid(text, top_k=top_k, lambda_sem=lambda_sem, with_genes=with_genes)

        def query_annotation_only(self, text, top_k=5, with_genes=False):
            return self._eng.query_annotation_only(text, top_k=top_k, with_genes=with_genes)

        def discover(self, text, top_k=5, **kw):
            payload = self._eng.query_semantic(text, top_k=top_k, with_genes=True)
            payload["mode"] = "discovery"; payload["query"] = text
            return payload

        def interactions(self, source=None, target=None, **kw):
            src_cl = tgt_cl = None
            if source:
                sl = source.lower()
                src_cl = [c for c in self._eng.cluster_ids if sl in str(c).lower()] or None
            if target:
                tl = target.lower()
                tgt_cl = [c for c in self._eng.cluster_ids if tl in str(c).lower()] or None
            return find_interactions(self._eng.gene_stats, self._eng.cluster_ids,
                                     source_clusters=src_cl, target_clusters=tgt_cl, **kw)

        def pathway(self, pathway_name=None, **kw):
            if pathway_name:
                return query_pathway(self._eng.gene_stats, self._eng.cluster_ids,
                                     pathway_name=pathway_name, **kw)
            return pathway_scoring(self._eng.gene_stats, self._eng.cluster_ids, **kw)

        def proportions(self, **kw):
            return proportion_analysis(self._eng.metadata, **kw)

        def compare(self, group_a, group_b, genes=None, **kw):
            condition_col = None
            meta = self._eng.metadata
            if meta:
                for cid, m in meta.items():
                    if isinstance(m, dict):
                        for col, dist in m.get("fields", {}).items():
                            if isinstance(dist, dict):
                                dk = {k.lower() for k in dist}
                                if group_a.lower() in dk or group_b.lower() in dk:
                                    condition_col = col; break
                    if condition_col: break
            if not condition_col: condition_col = "region"
            return comparative_analysis(self._eng.gene_stats, self._eng.metadata,
                                        condition_col=condition_col,
                                        group_a=group_a, group_b=group_b, genes=genes, **kw)

        def detect_capabilities(self):
            caps = {"has_conditions": False, "condition_values": [],
                    "condition_column": None,
                    "n_clusters": len(self._eng.cluster_ids),
                    "cluster_ids": list(self._eng.cluster_ids)}
            meta = self._eng.metadata
            if not meta: return caps
            kws = ["region", "age", "patient_group", "condition", "timepoint",
                   "treatment", "status", "group", "sample_type"]
            col_cands = {}
            for cid, m in meta.items():
                if not isinstance(m, dict): continue
                for col, dist in m.get("fields", {}).items():
                    if isinstance(dist, dict):
                        if col not in col_cands: col_cands[col] = set()
                        col_cands[col].update(dist.keys())
            for kw in kws:
                for col, vals in col_cands.items():
                    if kw in col.lower().replace("_", " "):
                        if 2 <= len(vals) <= 10:
                            caps["has_conditions"] = True
                            caps["condition_column"] = col
                            caps["condition_values"] = sorted(vals)
                            return caps
            return caps

    engine = EngineWrapper(raw_engine)

    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY not set")
    llm = Groq(api_key=key)

    adata = None
    if args.h5ad:
        try:
            import scanpy as sc
            print(f"[INIT] Loading h5ad: {args.h5ad}")
            adata = sc.read_h5ad(args.h5ad)
            if adata.var_names[0].startswith("ENSG"):
                if "feature_name" in adata.var.columns:
                    adata.var["ensembl_id"] = adata.var_names.copy()
                    adata.var_names = adata.var["feature_name"].astype(str).values
                    adata.var_names_make_unique()
            print(f"[INIT] h5ad: {adata.shape[0]} cells, {adata.shape[1]} genes")
        except ImportError:
            print("[WARN] scanpy not installed")

    report = ReportBuilder(dataset_name=args.dataset_name)
    caps = engine.detect_capabilities()
    print(f"[INIT] Conditions: {caps.get('has_conditions')} → {caps.get('condition_values', [])}")
    return engine, llm, adata, report, viz, caps


def ask_llm(llm, system_prompt, user_prompt):
    res = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2)
    return res.choices[0].message.content.strip()


SYSTEM_PROMPT = (
    "You are ELISA, an expert assistant for single-cell RNA-seq biology and "
    "neurodevelopment. Never hallucinate. Ground claims strictly in provided data. "
    "Be concise and scientific. Focus on first-trimester human brain development: "
    "GABAergic/glutamatergic neuron specification (OTX2/GATA2/TAL2 vs LHX2/BHLHE22), "
    "Purkinje cell lineage (PTF1A→ASCL1→TFAP2B→LHX5→ESRRB→PCP4), "
    "radial glia to glioblast transition (NFI factors), "
    "oligodendrocyte differentiation (SOX10), "
    "regional brain patterning (telencephalon vs cerebellum vs midbrain), "
    "dopaminergic/serotonergic neuron identity, "
    "and neurodevelopmental disease cell-type vulnerability."
)

MAX_PROMPT_CHARS = 12000

def trim_payload(payload, max_chars=MAX_PROMPT_CHARS):
    trimmed = dict(payload)
    if "clusters" in trimmed and isinstance(trimmed["clusters"], dict):
        cl = trimmed["clusters"]
        if len(cl) > 10:
            trimmed["clusters"] = dict(sorted(cl.items(),
                key=lambda kv: len(kv[1].get("genes", [])), reverse=True)[:10])
    if "interactions" in trimmed and isinstance(trimmed["interactions"], list):
        if len(trimmed["interactions"]) > 30:
            trimmed["interactions"] = trimmed["interactions"][:30]
    if "scores" in trimmed and isinstance(trimmed["scores"], list):
        trimmed["scores"] = trimmed["scores"][:10]
    for r in trimmed.get("results", []):
        if "gene_evidence" in r and len(r["gene_evidence"]) > 5:
            r["gene_evidence"] = r["gene_evidence"][:5]
    ctx = json.dumps(trimmed, indent=1, default=str)
    return ctx[:max_chars] + "\n... [TRUNCATED]" if len(ctx) > max_chars else ctx

def build_prompt(mode, query, payload):
    ctx = trim_payload(payload)
    if mode == "discovery":
        return f"DISCOVERY mode.\n1. DATASET EVIDENCE 2. ESTABLISHED BIOLOGY 3. CONSISTENCY 4. HYPOTHESES\nQUESTION: {query}\nDATASET: {ctx}"
    elif mode == "compare":
        return f"COMPARATIVE analysis.\nCOMPARISON: {query}\nEVIDENCE: {ctx}\nIdentify region-biased clusters, DEGs."
    elif mode == "interactions":
        return f"CELL-CELL INTERACTIONS.\nQUERY: {query}\nINTERACTIONS: {ctx}\nGroup by pathway."
    elif mode == "proportions":
        return f"CELL TYPE PROPORTIONS.\nQUERY: {query}\nDATA: {ctx}\nRegional differences."
    elif mode in ("pathway_scoring", "pathway_query"):
        return f"PATHWAY ACTIVITY.\nQUERY: {query}\nSCORES: {ctx}\nTop cell types, genes."
    return f"MODE: {mode.upper()} | QUERY: {query}\nEVIDENCE: {ctx}"


# ══════════════════════════════════════════════════════════════
# QUERIES — scRNA-seq–derived only
# ══════════════════════════════════════════════════════════════

def get_queries(skip_plots=False):
    queries = [
        # ── Phase 1: Landscape ──
        ("info", "info"),
        ("proportions", "proportions"),

        # ── Phase 2: Major cell classes — expression markers (Fig 1f) ──
        ("semantic", "semantic: GABAergic inhibitory neuron GAD1 GAD2 SLC32A1 midbrain developing brain"),
        ("semantic", "semantic: glutamatergic excitatory neuron SLC17A6 SLC17A7 telencephalon hindbrain"),
        ("semantic", "semantic: Purkinje cell cerebellum PTF1A ESRRB PCP4 RORA large arborated neuron"),
        ("semantic", "semantic: interneuron DLX2 LHX6 PVALB SST VIP cortical inhibitory"),
        ("semantic", "semantic: radial glial cell SOX2 PAX6 NES VIM neural stem cell progenitor"),
        ("semantic", "semantic: glioblast astrocyte precursor GFAP S100B AQP4 BCAN TNC"),
        ("semantic", "semantic: oligodendrocyte precursor cell OLIG2 PDGFRA SOX10 CSPG4"),
        ("semantic", "semantic: dopaminergic neuron TH DDC SLC6A3 NR4A2 midbrain"),
        ("semantic", "semantic: serotonergic neuron TPH2 SLC6A4 FEV raphe nucleus brainstem"),
        ("semantic", "semantic: endothelial cell CLDN5 PECAM1 CDH5 blood-brain barrier"),
        ("semantic", "semantic: microglial cell CX3CR1 P2RY12 TMEM119 brain resident macrophage"),
        ("semantic", "semantic: pericyte PDGFRB RGS5 FOXF2 cerebral vasculature"),
        ("semantic", "semantic: vascular leptomeningeal cell DCN FOXC1 COL1A1 meningeal fibroblast"),
        ("semantic", "semantic: Schwann cell MPZ CDH19 SOX10 neural crest myelinating glial"),

        # ── Phase 3: Neuronal specification — TF expression (Fig 3c dot plot) ──
        ("discover", "discover: OTX2 GATA2 TAL2 SOX14 midbrain GABAergic neuron identity"),
        ("discover", "discover: LHX2 BHLHE22 CUX1 CUX2 telencephalic glutamatergic cortical layer"),
        ("discover", "discover: ATOH1 MEIS1 MEIS2 hindbrain glutamatergic cerebellar granule neuron"),
        ("discover", "discover: GATA2 TAL2 midbrain GABAergic OTX2 DMBX1 TF expression"),
        ("discover", "discover: LHX6 DLX2 DLX5 interneuron medial ganglionic eminence cortical"),
        ("discover", "discover: NFIA NFIB NFIX NFI maturation factor shared across neuronal clades"),
        ("discover", "discover: TFAP2B LHX5 LHX1 Purkinje and GABAergic midbrain shared TF expression"),
        ("discover", "discover: SATB2 TBR1 FEZF2 BCL11B cortical glutamatergic projection neuron"),
        ("discover", "discover: EBF1 early neuronal marker expression across cell types"),
        ("discover", "discover: EMX2 SOX9 SOX10 lineage-specific TF expression dot plot"),

        # ── Phase 4: Purkinje lineage — pseudotime expression (Fig 4d) ──
        ("discover", "discover: PTF1A ASCL1 NEUROG2 Purkinje progenitor ventricular zone cerebellum"),
        ("discover", "discover: NHLH1 NHLH2 TFAP2B Purkinje neuroblast early differentiation"),
        ("discover", "discover: LHX5 PAX2 EBF3 Purkinje neuroblast late differentiation"),
        ("discover", "discover: ESRRB oestrogen-related nuclear receptor Purkinje specific cerebellum"),
        ("discover", "discover: ESRRB RORA PCP4 FOXP2 EBF3 LHX1 mature Purkinje markers"),
        ("discover", "discover: TFAP2B expressed before ESRRB in Purkinje lineage pseudotime"),
        ("discover", "discover: LHX5 expression precedes ESRRB induction Purkinje two-step"),
        ("discover", "discover: DMBX1 low expression Purkinje progenitor same TF family as OTX2"),
        ("discover", "discover: PCP4 late Purkinje marker preceded by ESRRB expression"),

        # ── Phase 5: Radial glia → glioblast (gene expression) ──
        ("discover", "discover: SOX2 PAX6 NES VIM radial glia stemness anterior brain"),
        ("discover", "discover: NFIA NFIB NFIX NFI factor expression maturation glioblast transition"),
        ("discover", "discover: GFAP AQP4 ALDH1L1 BCAN TNC glioblast astrocyte restricted identity"),
        ("discover", "discover: radial glia posterior regions enriched for glioblasts earlier transition"),
        ("discover", "discover: SOX9 pan-glial marker expression radial glia glioblast"),

        # ── Phase 6: OPC / oligodendrocyte (gene expression) ──
        ("discover", "discover: OLIG1 OLIG2 SOX10 PDGFRA oligodendrocyte precursor specification"),
        ("discover", "discover: MBP MOG PLP1 MAG oligodendrocyte myelination differentiation"),
        ("discover", "discover: SOX10 oligodendrocyte lineage marker expression"),

        # ── Phase 7: Disease cell-type vulnerability (expression-derived) ──
        # Paper reports which cell types express MDD genes — this is scRNA-seq
        ("discover", "discover: NEGR1 BTN3A2 LRFN5 SCN8A MDD associated genes expressed broadly"),
        ("discover", "discover: midbrain GABAergic neuron vulnerability to psychiatric disorder"),
        ("discover", "discover: schizophrenia cortical interneuron MGE SATB2 excitatory neuron"),
        ("discover", "discover: ADHD cerebellar Purkinje neuroblast GABAergic neuron"),
        ("discover", "discover: anorexia nervosa LGE CGE interneuron GABAergic"),
        ("discover", "discover: autism spectrum disorder hindbrain neuroblast brainstem"),
        ("discover", "discover: insomnia TAL2 midbrain GABAergic reticular formation wakefulness"),
        ("discover", "discover: SOX14 midbrain GABAergic neuron thalamus pons migration"),

        # ── Phase 8: Non-neural cell types (Fig 1f expression) ──
        ("discover", "discover: RUNX1 immune cell microglia haematopoietic stem cell development"),
        ("discover", "discover: FOXF2 pericyte development FOXC1 meningeal fibroblast meninges"),
        ("discover", "discover: ERG endothelial cell vascular identity developing brain"),
        ("discover", "discover: glycinergic neuron SLC6A5 GLRA1 hindbrain spinal cord inhibitory"),
        ("discover", "discover: sensory neuron dorsal root ganglion NTRK1 ISL1 peripheral"),
        ("discover", "discover: Schwann cell MPZ CDH19 SOX10 neural crest myelinating"),
        ("discover", "discover: immature T cell leukocyte fetal brain immune infiltration"),

        # ── Phase 9: Regional comparisons ──
        ("compare", "compare: Telencephalon vs Cerebellum | SOX2, PAX6, NES, VIM, HES1, FABP7"),
        ("compare", "compare: Telencephalon vs Cerebellum | GAD1, GAD2, DLX2, LHX6, PVALB, SST"),
        ("compare", "compare: Telencephalon vs Cerebellum | SLC17A7, SLC17A6, SATB2, TBR1, EMX2, LHX2"),
        ("compare", "compare: Telencephalon vs Cerebellum | PTF1A, ESRRB, PCP4, ATOH1, MEIS1"),
        ("compare", "compare: Telencephalon vs Cerebellum | NFIA, NFIB, NFIX, GFAP, AQP4, BCAN"),
        ("compare", "compare: Telencephalon vs Cerebellum | OLIG1, OLIG2, SOX10, PDGFRA, MBP"),
        ("compare", "compare: Telencephalon vs Cerebellum | TH, DDC, SLC6A3, NR4A2, TPH2, SLC6A4"),
        ("compare", "compare: Telencephalon vs Cerebellum | CLDN5, PECAM1, PDGFRB, DCN, FOXC1"),
        ("compare", "compare: Telencephalon vs Cerebellum | AIF1, CX3CR1, P2RY12, RUNX1, CSF1R"),
        ("compare", "compare: Telencephalon vs Cerebellum | OTX2, GATA2, TAL2, BHLHE22, CUX1, RORB"),

        # ── Phase 10: Pathway analyses ──
        ("pathway", "pathway: Glutamatergic synapse"),
        ("pathway", "pathway: GABAergic synapse"),
        ("pathway", "pathway: Calcium signaling"),
        ("pathway", "pathway: Axon guidance"),
        ("pathway", "pathway: Notch signaling"),
        ("pathway", "pathway: Wnt signaling"),
        ("pathway", "pathway: Hippo signaling"),
        ("pathway", "pathway: mTOR signaling"),
        ("pathway", "pathway: Oxidative phosphorylation"),
        ("pathway", "pathway: Neurodegeneration"),
        ("pathway", "pathway: all"),

        # ── Phase 11: Cell-cell interactions ──
        ("interactions", "interactions:"),
        ("interactions", "interactions: neuron -> radial glial"),
        ("interactions", "interactions: radial glial -> neural progenitor"),
        ("interactions", "interactions: glioblast -> oligodendrocyte precursor"),
        ("interactions", "interactions: microglial -> neuron"),
        ("interactions", "interactions: endothelial -> pericyte"),
        ("interactions", "interactions: radial glial -> glioblast"),
        ("interactions", "interactions: dopaminergic -> glioblast"),

        # ── Phase 12: Gene recovery — targeted compare batches ──

        # Batch A: GABAergic TFs
        ("compare", "compare: Telencephalon vs Cerebellum | OTX2, GATA2, TAL2, SOX14, DLX1, DLX5, DLX6"),
        ("compare", "compare: Telencephalon vs Cerebellum | PVALB, SST, VIP, LAMP5, SNCG, ADARB2"),

        # Batch B: Glutamatergic
        ("compare", "compare: Telencephalon vs Cerebellum | FEZF2, BCL11B, CUX2, RORB, BHLHE22"),
        ("compare", "compare: Telencephalon vs Cerebellum | ATOH1, MEIS1, MEIS2, SLC17A6"),

        # Batch C: Purkinje lineage
        ("compare", "compare: Telencephalon vs Cerebellum | ASCL1, NEUROG2, NHLH1, NHLH2, TFAP2B"),
        ("compare", "compare: Telencephalon vs Cerebellum | LHX5, LHX1, PAX2, EBF1, EBF3, FOXP2"),
        ("compare", "compare: Telencephalon vs Cerebellum | ESRRB, RORA, PCP4, DMBX1"),

        # Batch D: NFI / glial
        ("compare", "compare: Telencephalon vs Cerebellum | NFIA, NFIB, NFIX, SOX9"),
        ("compare", "compare: Telencephalon vs Cerebellum | GFAP, S100B, AQP4, ALDH1L1, TNC"),

        # Batch E: Dopaminergic / serotonergic
        ("compare", "compare: Telencephalon vs Cerebellum | TH, DDC, SLC6A3, SLC18A2, NR4A2, LMX1A, FOXA2"),
        ("compare", "compare: Telencephalon vs Cerebellum | TPH2, SLC6A4, FEV"),

        # Batch F: Pan-neuronal
        ("compare", "compare: Telencephalon vs Cerebellum | RBFOX3, SNAP25, SYT1, NEFM, NEFL"),

        # Batch G: OPC / oligo
        ("compare", "compare: Telencephalon vs Cerebellum | CSPG4, MBP, MOG, PLP1, MAG"),

        # Batch H: Vascular / immune
        ("compare", "compare: Telencephalon vs Cerebellum | PDGFRB, RGS5, ACTA2, MYH11, FOXF2"),
        ("compare", "compare: Telencephalon vs Cerebellum | TMEM119, HEXB, SPI1, RUNX1"),
        ("compare", "compare: Telencephalon vs Cerebellum | DCN, LUM, COL1A1, COL1A2, FOXC1"),
        ("compare", "compare: Telencephalon vs Cerebellum | ERG, FLT1, CDH5"),

        # Batch I: Schwann / sensory / glycinergic
        ("compare", "compare: Telencephalon vs Cerebellum | MPZ, CDH19, S100B"),
        ("compare", "compare: Telencephalon vs Cerebellum | NTRK1, NTRK2, ISL1"),
        ("compare", "compare: Telencephalon vs Cerebellum | SLC6A5, GLRA1"),

        # Batch J: MDD genes (expressed in scRNA-seq, not GWAS-only)
        ("compare", "compare: Telencephalon vs Cerebellum | NEGR1, BTN3A2, LRFN5, SCN8A, RGS6"),
        ("compare", "compare: Telencephalon vs Cerebellum | MYCN, MEIS2"),

        # ── Phase 13: Semantic recovery ──
        ("semantic", "semantic: MYCN NEGR1 BTN3A2 LRFN5 SCN8A expressed in midbrain GABAergic neuron"),
        ("semantic", "semantic: OTX2 GATA2 MEIS2 midbrain GABAergic TF expression"),
        ("semantic", "semantic: RBFOX3 SNAP25 SYT1 NEFM NEFL pan-neuronal mature neuron"),
        ("semantic", "semantic: ACTA2 MYH11 PDGFRB vascular smooth muscle pericyte"),
        ("semantic", "semantic: ERG FLT1 CDH5 VWF endothelial cell vascular marker"),
        ("semantic", "semantic: HEXB TMEM119 P2RY12 CX3CR1 microglial homeostatic"),
        ("semantic", "semantic: FOXF2 FOXC1 pericyte meningeal fibroblast leptomeningeal"),
        ("semantic", "semantic: PVALB SST VIP LAMP5 SNCG interneuron subtype marker"),
        ("semantic", "semantic: FEZF2 BCL11B TBR1 deep layer cortical projection neuron"),
        ("semantic", "semantic: FOXA2 LMX1A NR4A2 floor plate dopaminergic specification"),
    ]

    if not skip_plots:
        plot_queries = [
            ("plot", "plot:umap"),
            # Neuronal markers
            ("plot", "plot:expr GAD1"), ("plot", "plot:expr GAD2"),
            ("plot", "plot:expr SLC17A6"), ("plot", "plot:expr SLC17A7"),
            ("plot", "plot:expr RBFOX3"),
            # GABAergic TFs
            ("plot", "plot:expr OTX2"), ("plot", "plot:expr GATA2"),
            ("plot", "plot:expr DLX2"), ("plot", "plot:expr LHX6"),
            ("plot", "plot:expr TAL2"),
            # Glutamatergic TFs
            ("plot", "plot:expr LHX2"), ("plot", "plot:expr BHLHE22"),
            ("plot", "plot:expr EMX2"), ("plot", "plot:expr SATB2"),
            ("plot", "plot:expr TBR1"),
            # Purkinje lineage
            ("plot", "plot:expr PTF1A"), ("plot", "plot:expr ESRRB"),
            ("plot", "plot:expr PCP4"), ("plot", "plot:expr TFAP2B"),
            ("plot", "plot:expr LHX5"),
            # Radial glia / glioblast
            ("plot", "plot:expr SOX2"), ("plot", "plot:expr PAX6"),
            ("plot", "plot:expr NFIA"), ("plot", "plot:expr GFAP"),
            ("plot", "plot:expr BCAN"), ("plot", "plot:expr TNC"),
            # OPC / oligo
            ("plot", "plot:expr OLIG2"), ("plot", "plot:expr SOX10"),
            ("plot", "plot:expr PDGFRA"), ("plot", "plot:expr MBP"),
            # Dopaminergic / serotonergic
            ("plot", "plot:expr TH"), ("plot", "plot:expr NR4A2"),
            ("plot", "plot:expr TPH2"),
            # Vascular / immune
            ("plot", "plot:expr CLDN5"), ("plot", "plot:expr PDGFRB"),
            ("plot", "plot:expr AIF1"), ("plot", "plot:expr CX3CR1"),
            # Non-neural
            ("plot", "plot:expr DCN"), ("plot", "plot:expr FOXC1"),
            ("plot", "plot:expr MPZ"),
            # Dotplots
            ("plot", "plot:dotplot GAD1, GAD2, SLC17A6, SLC17A7, RBFOX3, SNAP25, NEFM, NEFL"),
            ("plot", "plot:dotplot OTX2, GATA2, TAL2, LHX6, DLX2, MEIS2, PVALB, SST"),
            ("plot", "plot:dotplot LHX2, BHLHE22, SATB2, TBR1, FEZF2, BCL11B, CUX1, EMX2"),
            ("plot", "plot:dotplot PTF1A, ESRRB, PCP4, TFAP2B, LHX5, RORA, FOXP2, ATOH1"),
            ("plot", "plot:dotplot SOX2, PAX6, NES, NFIA, NFIB, GFAP, AQP4, BCAN"),
            ("plot", "plot:dotplot OLIG2, SOX10, PDGFRA, MBP, MOG, PLP1, CSPG4, MAG"),
            ("plot", "plot:dotplot TH, DDC, NR4A2, FOXA2, LMX1A, TPH2, SLC6A4, FEV"),
            ("plot", "plot:dotplot CLDN5, PECAM1, PDGFRB, RGS5, DCN, FOXC1, AIF1, P2RY12"),
            # Grids
            ("plot", "plot:grid GAD1, SLC17A7, RBFOX3, OTX2, LHX2, ESRRB"),
            ("plot", "plot:grid SOX2, NFIA, GFAP, OLIG2, TH, CLDN5"),
            ("plot", "plot:grid PTF1A, TFAP2B, LHX5, PCP4, ATOH1, MEIS1"),
        ]
        queries.extend(plot_queries)

    return queries


def get_queries_test():
    return [
        ("info", "info"),
        ("proportions", "proportions"),
        ("semantic", "semantic: GABAergic neuron GAD1 GAD2 OTX2 midbrain developing brain"),
        ("discover", "discover: PTF1A ESRRB Purkinje cerebellum lineage TFAP2B LHX5"),
        ("compare", "compare: Telencephalon vs Cerebellum | GAD1, SLC17A7, SOX2, ESRRB, OTX2"),
        ("interactions", "interactions:"),
        ("interactions", "interactions: neuron -> radial glial"),
        ("pathway", "pathway: Axon guidance"),
        ("pathway", "pathway: all"),
    ]


# ══════════════════════════════════════════════════════════════
# EVALUATION REFERENCE — scRNA-seq–derived only
# ══════════════════════════════════════════════════════════════

PAPER_GENES = {
    # GABAergic — Fig 1f expression, Fig 3c TF dot plot
    "GAD1", "GAD2", "SLC32A1",
    "OTX2", "GATA2", "TAL2", "SOX14",
    "DLX1", "DLX2", "DLX5", "DLX6", "LHX6", "MEIS2",
    "PVALB", "SST", "VIP", "LAMP5", "SNCG", "ADARB2",
    "TFAP2B",
    # Glutamatergic — Fig 1f, Fig 3c
    "SLC17A7", "SLC17A6",
    "SATB2", "TBR1", "FEZF2", "BCL11B",
    "EMX2", "LHX2", "BHLHE22", "CUX1", "CUX2", "RORB",
    "ATOH1", "MEIS1",
    # Purkinje lineage — Fig 4d pseudotime expression
    "PTF1A", "ASCL1", "NEUROG2", "NHLH1", "NHLH2",
    "LHX5", "LHX1", "PAX2",
    "ESRRB", "RORA", "PCP4",
    "EBF1", "EBF3", "FOXP2", "DMBX1",
    # Pan-neuronal — Fig 1f expression
    "RBFOX3", "SNAP25", "SYT1", "NEFM", "NEFL",
    # Radial glia — Fig 1f expression
    "SOX2", "PAX6", "NES", "VIM", "HES1", "HES5", "FABP7",
    # NFI maturation — Fig 2a,b TF expression
    "NFIA", "NFIB", "NFIX",
    # Glioblast — Fig 1f expression
    "GFAP", "S100B", "AQP4", "ALDH1L1", "BCAN", "TNC",
    # Pan-glial
    "SOX9",
    # OPC / oligodendrocyte — Fig 1f expression
    "OLIG1", "OLIG2", "SOX10", "PDGFRA", "CSPG4",
    "MBP", "MOG", "PLP1", "MAG",
    # Dopaminergic — Fig 1f, Fig 3c
    "TH", "DDC", "SLC6A3", "SLC18A2", "NR4A2", "LMX1A", "FOXA2",
    # Serotonergic
    "TPH2", "SLC6A4", "FEV",
    # Glycinergic
    "SLC6A5", "GLRA1",
    # Endothelial — Fig 1f
    "CLDN5", "PECAM1", "CDH5", "ERG", "FLT1",
    # Pericyte / VSMC — Fig 1f
    "PDGFRB", "RGS5", "ACTA2", "MYH11",
    # VLMC — Fig 1f
    "DCN", "LUM", "COL1A1", "COL1A2", "FOXC1", "FOXF2",
    # Immune — Fig 1f
    "AIF1", "CX3CR1", "P2RY12", "TMEM119", "HEXB",
    "RUNX1", "SPI1", "CSF1R",
    # Schwann
    "MPZ", "CDH19",
    # Sensory
    "NTRK1", "NTRK2", "ISL1",
    # MDD genes — expressed in scRNA-seq (Fig 5b shows expression)
    "NEGR1", "BTN3A2", "LRFN5", "SCN8A", "RGS6",
    # Broadly expressed TFs with scRNA-seq evidence
    "MYCN",
    # NOTE: PRDM10, CTCF, MECP2 excluded — only scATAC/motif evidence
}

PAPER_INTERACTIONS = [
    ("neuron", "radial glial", "BDNF/NTRK2 neurotrophic"),
    ("glutamatergic", "neural progenitor", "NTF3/NTRK3 signaling"),
    ("dopaminergic", "glioblast", "GDNF/GFRA1 trophic"),
    ("neuron", "microglial", "CX3CL1/CX3CR1 fractalkine"),
    ("microglial", "neuron", "C1QA/C3AR1 complement"),
    ("neuron", "microglial", "CSF1/CSF1R trophic"),
    ("radial glial", "oligodendrocyte precursor", "PDGFA/PDGFRA OPC"),
    ("glioblast", "endothelial", "VEGFA/KDR angiogenesis"),
    ("endothelial", "pericyte", "PDGFB/PDGFRB vascular"),
    ("radial glial", "neural progenitor", "DLL1/NOTCH1 Notch"),
    ("radial glial", "glioblast", "JAG1/NOTCH2 Notch"),
    ("radial glial", "neural progenitor", "WNT5A/FZD5 Wnt"),
    ("glioblast", "oligodendrocyte precursor", "BMP4/BMPR1A"),
    ("radial glial", "neural progenitor", "SHH/PTCH1 ventral"),
]

PAPER_PATHWAYS = [
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
]

PROPORTION_CHANGES = {
    "radial glial cell": "enriched in anterior (telencephalon)",
    "glioblast": "enriched in posterior (cerebellum, hindbrain)",
    "interneuron": "enriched in telencephalon",
    "glutamatergic neuron": "broad across regions",
    "Purkinje cell": "cerebellum-specific",
    "dopaminergic neuron": "midbrain-specific",
    "endothelial cell": "limited spatial identity",
    "microglial cell": "limited spatial identity",
}


# ── Execute query ────────────────────────────────────────────
def execute_query(cmd_type, cmd_str, engine, llm, adata, viz,
                  cluster_key, plot_dir):
    payload, answer, plots = None, None, []

    if cmd_type == "info":
        caps = engine.detect_capabilities()
        return {"mode": "info", "capabilities": caps}, json.dumps(caps, indent=2), []
    elif cmd_type == "proportions":
        payload = engine.proportions()
    elif cmd_type == "semantic":
        payload = engine.query_semantic(cmd_str.split(":", 1)[1].strip(), top_k=5, with_genes=True)
    elif cmd_type == "discover":
        payload = engine.discover(cmd_str.split(":", 1)[1].strip(), top_k=5)
    elif cmd_type == "compare":
        txt = cmd_str.split(":", 1)[1].strip()
        genes = None
        if "|" in txt:
            txt, gs = txt.split("|", 1)
            genes = [g.strip() for g in gs.split(",") if g.strip()]
            txt = txt.strip()
        parts = txt.lower().split(" vs ")
        if len(parts) == 2:
            ga, gb = parts[0].strip(), parts[1].strip()
            caps = engine.detect_capabilities()
            if caps["has_conditions"]:
                for cv in caps["condition_values"]:
                    if cv.lower() == ga: ga = cv
                    if cv.lower() == gb: gb = cv
            payload = engine.compare(ga, gb, genes=genes)
        else:
            return {"error": f"Bad compare: {txt}"}, "", []
    elif cmd_type == "interactions":
        txt = cmd_str.split(":", 1)[1].strip() if ":" in cmd_str else ""
        src = tgt = None
        if "->" in txt:
            p = txt.split("->")
            src = p[0].strip() or None
            tgt = p[1].strip() if len(p) > 1 else None
        elif txt: src = txt
        payload = engine.interactions(source=src, target=tgt)
    elif cmd_type == "pathway":
        txt = cmd_str.split(":", 1)[1].strip()
        payload = engine.pathway(pathway_name=txt if txt.lower() != "all" else None)
    elif cmd_type == "plot":
        if adata is None:
            print(f"  [SKIP] No h5ad"); return None, None, []
        os.makedirs(plot_dir, exist_ok=True)
        sc = cmd_str.split(None, 1); sub = sc[0]; a = sc[1].strip() if len(sc) > 1 else ""
        import matplotlib; matplotlib.use("Agg")
        try:
            if sub == "plot:umap":
                p = f"{plot_dir}/cell_umap.png"
                viz.plot_cell_umap(adata, cluster_key=cluster_key, save_path=p)
                viz.plt.close("all"); plots.append(p)
            elif sub == "plot:expr":
                p = f"{plot_dir}/expr_{a}.png"
                viz.plot_gene_expression_umap(adata, gene=a, save_path=p)
                viz.plt.close("all"); plots.append(p)
            elif sub == "plot:dotplot":
                genes = [g.strip() for g in a.split(",")]
                p = f"{plot_dir}/dotplot_{'_'.join(genes[:3])}.png"
                viz.plot_dotplot(adata, genes=genes, cluster_key=cluster_key, save_path=p)
                viz.plt.close("all"); plots.append(p)
            elif sub == "plot:grid":
                genes = [g.strip() for g in a.split(",")]
                p = f"{plot_dir}/grid_{genes[0]}.png"
                viz.plot_gene_expression_grid(adata, genes=genes, cluster_key=cluster_key, save_path=p)
                viz.plt.close("all"); plots.append(p)
        except Exception as e:
            print(f"  [PLOT ERROR] {e}")
        return None, None, plots

    if payload and "error" not in payload:
        mode = payload.get("mode", cmd_type)
        prompt = build_prompt(mode, payload.get("query", cmd_str), payload)
        try: answer = ask_llm(llm, SYSTEM_PROMPT, prompt)
        except Exception as e: answer = f"[LLM ERROR] {e}"
    return payload, answer, plots


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def full_evaluation(report_text, elisa_genes, elisa_interactions,
                    elisa_pathway_scores, elisa_proportions):
    sc = {}
    found = PAPER_GENES & elisa_genes
    gr = len(found) / len(PAPER_GENES) * 100 if PAPER_GENES else 0
    sc["gene_recall"] = f"{gr:.1f}% ({len(found)}/{len(PAPER_GENES)})"

    pf = 0
    for pp in PAPER_PATHWAYS:
        if any(pp.lower() in ek.lower() or ek.lower() in pp.lower() for ek in elisa_pathway_scores):
            pf += 1
    pc = pf / len(PAPER_PATHWAYS) * 100 if PAPER_PATHWAYS else 0
    sc["pathway_coverage"] = f"{pc:.1f}% ({pf}/{len(PAPER_PATHWAYS)})"

    ixf = 0
    for src, tgt, _ in PAPER_INTERACTIONS:
        sl, tl = src.lower(), tgt.lower()
        if any((sl in ei.get("source","").lower() or ei.get("source","").lower() in sl) and
               (tl in ei.get("target","").lower() or ei.get("target","").lower() in tl)
               for ei in elisa_interactions):
            ixf += 1
    ixr = ixf / len(PAPER_INTERACTIONS) * 100 if PAPER_INTERACTIONS else 0
    sc["interaction_recall"] = f"{ixr:.1f}% ({ixf}/{len(PAPER_INTERACTIONS)})"

    hp = bool(elisa_proportions)
    sc["proportions_available"] = "Yes" if hp else "No"
    sc["report_words"] = len(report_text.split())

    # Themes — scRNA-seq derived only
    themes = {
        "GABAergic midbrain identity": ["OTX2", "GATA2", "TAL2", "GABAergic"],
        "Glutamatergic specification": ["SLC17A7", "glutamatergic", "SATB2", "TBR1"],
        "Telencephalic vs posterior": ["telencephalon", "LHX2", "BHLHE22", "EMX2"],
        "Hindbrain granule neuron": ["ATOH1", "MEIS1", "granule", "hindbrain"],
        "Purkinje lineage": ["Purkinje", "ESRRB", "PTF1A", "cerebell"],
        "ESRRB activation sequence": ["ESRRB", "TFAP2B", "LHX5", "two-step"],
        "Purkinje TF cascade": ["ASCL1", "NEUROG2", "NHLH1", "NHLH2", "pseudotime"],
        "Radial glia stemness": ["radial glia", "SOX2", "PAX6", "NES", "stemness"],
        "NFI maturation": ["NFIA", "NFIB", "NFIX", "NFI", "maturation"],
        "Glioblast transition": ["glioblast", "astrocyte", "GFAP", "posterior"],
        "OPC differentiation": ["oligodendrocyte", "OLIG2", "SOX10", "myelination"],
        "Dopaminergic identity": ["TH", "DDC", "dopaminergic", "NR4A2", "FOXA2"],
        "Serotonergic identity": ["TPH2", "serotonergic", "raphe", "FEV"],
        "MDD midbrain vulnerability": ["MDD", "major depressive", "midbrain"],
        "Schizophrenia cortical": ["schizophrenia", "cortical interneuron"],
        "ADHD cerebellar": ["ADHD", "attention deficit", "cerebell"],
        "ASD hindbrain": ["autism", "ASD", "brainstem"],
        "Insomnia TAL2": ["insomnia", "TAL2", "wakefulness"],
        "Regional patterning": ["regional", "anterior", "posterior", "patterning"],
        "Microglia RUNX1": ["RUNX1", "microglia", "haematopoietic"],
        "Pericyte FOXF2": ["FOXF2", "pericyte", "FOXC1", "meningeal"],
        "Schwann neural crest": ["Schwann", "MPZ", "CDH19", "neural crest"],
    }
    tf = sum(1 for kws in themes.values()
             if any(kw.lower() in report_text.lower() for kw in kws))
    tc = tf / len(themes) * 100 if themes else 0
    sc["theme_coverage"] = f"{tc:.1f}% ({tf}/{len(themes)})"

    composite = gr * 0.30 + pc * 0.20 + ixr * 0.15 + tc * 0.25 + (10 if hp else 0)
    sc["composite_score"] = f"{composite:.1f}%"

    return {"scorecard": sc, "composite_score": round(composite, 1),
            "details": {"paper_genes": sorted(PAPER_GENES), "found_genes": sorted(found),
                        "missing_genes": sorted(PAPER_GENES - found),
                        "paper_pathways": PAPER_PATHWAYS,
                        "paper_interactions": [(s, t, d) for s, t, d in PAPER_INTERACTIONS]}}


# ── Main ─────────────────────────────────────────────────────
def main():
    args = setup_args()
    out_dir = args.out_dir
    plot_dir = os.path.join(out_dir, "elisa_plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    engine, llm, adata, report, viz, caps = init_elisa(args)
    queries = get_queries(skip_plots=args.skip_plots)
    session_log, all_payloads, all_interactions = [], [], []

    print(f"\n{'='*70}")
    print(f"ELISA PAPER REPLICATION: First-Trimester Human Brain (scRNA-seq)")
    print(f"Paper: Mannens et al. (2025) Nature 647, 179-186")
    print(f"RUNNING {len(queries)} QUERIES")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")

    for i, (ct, cs) in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {cs}")
        t0 = time.time()
        payload, answer, plots = execute_query(ct, cs, engine, llm, adata, viz,
                                               args.cluster_key, plot_dir)
        elapsed = time.time() - t0
        if payload is None and ct == "plot":
            print(f"  → {len(plots)} plots ({elapsed:.1f}s)")
            if plots and report.entries: report.entries[-1]["plots"].extend(plots)
            continue
        if payload and "error" in payload:
            print(f"  [ERROR] {payload['error']}"); continue
        if payload:
            mode = payload.get("mode", ct)
            if mode == "interactions" and "interactions" in payload:
                all_interactions.extend(payload["interactions"])
            report.add_entry(entry_type=ct, query=payload.get("query", cs),
                             payload=payload, answer=answer or "", plots=plots)
            all_payloads.append(payload)
            session_log.append({"index": i+1, "command": cs, "type": ct, "mode": mode,
                                "query": payload.get("query",""),
                                "answer": answer[:500] if answer else "",
                                "elapsed": round(elapsed, 2), "n_plots": len(plots)})
            if answer: print(f"  → {mode} | {elapsed:.1f}s\n  {answer[:150]}...")
            else: print(f"  → {mode} | {elapsed:.1f}s")
        print()

    # Report
    print(f"\n{'='*70}\nGENERATING REPORT\n{'='*70}")
    def llm_fn(p): return ask_llm(llm, SYSTEM_PROMPT, p)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = os.path.join(out_dir, f"elisa_report_Brain_{ts}.md")
    report.generate_markdown(md_path, llm_func=llm_fn)
    docx_path = md_path.replace(".md", ".docx")
    report.generate_docx(docx_path, llm_func=llm_fn)
    log_path = os.path.join(out_dir, "session_log.json")
    with open(log_path, "w") as f: json.dump(session_log, f, indent=2, default=str)

    # Evaluation
    print(f"\n{'='*70}\nEVALUATION\n{'='*70}")
    report_text = open(md_path).read() if os.path.exists(md_path) else ""
    elisa_genes = set()
    for pl in all_payloads:
        for r in pl.get("results", []):
            for g in r.get("gene_evidence", []):
                elisa_genes.add(g.get("gene","") if isinstance(g, dict) else g)
            for g in r.get("genes", []):
                elisa_genes.add(g.get("gene","") if isinstance(g, dict) else g)
        for cdata in pl.get("clusters", {}).values():
            if isinstance(cdata, dict):
                for g in cdata.get("genes", []):
                    elisa_genes.add(g.get("gene","") if isinstance(g, dict) else g)
    for pl in all_payloads:
        if pl.get("mode") in ("pathway_scoring", "pathway_query"):
            for pd in pl.get("pathways", {}).values():
                for gs in pd.get("gene_set", []): elisa_genes.add(gs)
                for cd in pd.get("scores", []):
                    for tg in cd.get("top_genes", []):
                        elisa_genes.add(tg.get("gene","") if isinstance(tg, dict) else tg)
        if pl.get("mode") == "pathway_query":
            for g in pl.get("genes_in_pathway", []): elisa_genes.add(g)
    for entry in session_log:
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,12}(?:-[A-Z0-9]+)?)\b', entry.get("answer","")):
            elisa_genes.add(m.group(1))
    elisa_genes.discard("")
    print(f"  Genes: {len(elisa_genes)}")

    pw_sc = {}
    for pl in all_payloads:
        if pl.get("mode") == "pathway_scoring": pw_sc.update(pl.get("pathways", {}))
        elif pl.get("mode") == "pathway_query":
            pn = pl.get("pathway","")
            if pn: pw_sc[pn] = {"scores": pl.get("scores",[]), "genes_in_pathway": pl.get("genes_in_pathway",[])}
    prop_data = next((pl for pl in all_payloads if pl.get("mode") == "proportions"), {})

    ev = full_evaluation(report_text, elisa_genes, all_interactions, pw_sc, prop_data)
    print(f"\n{'='*70}\nSCORECARD\n{'='*70}")
    for m, s in ev["scorecard"].items(): print(f"  {m}: {s}")
    print(f"\n  ★ COMPOSITE: {ev['composite_score']}%\n{'='*70}")

    eval_path = os.path.join(out_dir, "evaluation_scorecard.json")
    with open(eval_path, "w") as f: json.dump(ev, f, indent=2, default=str)

    print(f"\nOUTPUTS:\n  MD:   {md_path}\n  DOCX: {docx_path}\n  Log:  {log_path}")
    print(f"  Eval: {eval_path}\n  Plots: {plot_dir}/\n  Analyses: {len(report.entries)}")
    print("DONE")


if __name__ == "__main__":
    main()
