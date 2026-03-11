#!/usr/bin/env python
"""
ELISA Paper Replication — Automated Batch Runner
=================================================
Target Paper:  Yu, Biyik-Sit, Uzun, Chen et al. (2025) Nature Genetics 57, 1142–1154
               "Longitudinal single-cell multiomic atlas of high-risk
                neuroblastoma reveals chemotherapy-induced tumor
                microenvironment rewiring"

Runs ALL queries from the Neuroblastoma paper replication plan automatically,
generates the report, and computes evaluation metrics.

Updated for retrieval_engine_v4_hybrid + elisa_analysis.

Usage:
    python run_paper_replication_Neuroblastoma.py \\
        --h5ad /path/to/DT6_Neuroblastoma.h5ad \\
        --out-dir /path/to/elisa_replication_Neuroblastoma/ \\
        --base /path/to/embeddings/ \\
        --pt-name fused_DT6_NB.pt \\
        --cluster-key cell_type \\
        --dataset-name "High-Risk Neuroblastoma — Longitudinal Multiomic Atlas"

Output (all saved to --out-dir):
    - elisa_report_TIMESTAMP.md
    - elisa_report_TIMESTAMP.docx
    - elisa_plots/
    - evaluation_scorecard.json
    - session_log.json

11 clusters:
    B cell, Schwann cell, T cell, cortical cell of adrenal gland,
    dendritic cell, endothelial cell, fibroblast, hepatocyte,
    kidney cell, macrophage, neuroblast (sensu Vertebrata)

Condition column: timepoint  (DX = diagnosis, PTX = post-induction chemo)
"""

import os
import sys
import json
import time
import textwrap
import argparse
import functools
from datetime import datetime

print = functools.partial(print, flush=True)


# ── Setup paths ──────────────────────────────────────────────
def setup_args():
    parser = argparse.ArgumentParser(
        description="ELISA Paper Replication Runner — High-Risk Neuroblastoma"
    )
    parser.add_argument("--h5ad", default=None,
                        help="Path to .h5ad (for Nature plots)")
    parser.add_argument("--cluster-key", default="cell_type",
                        help="obs column for cell types")
    parser.add_argument("--out-dir", default="elisa_replication_Neuroblastoma",
                        help="Output directory for all results")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip Nature-style plots (faster)")
    parser.add_argument("--dataset-name",
                        default="High-Risk Neuroblastoma — Longitudinal Multiomic Atlas",
                        help="Name for report title")
    parser.add_argument("--base", required=True,
                        help="Base directory for .pt and cells CSV")
    parser.add_argument("--pt-name", required=True,
                        help="Name of the fused .pt file")
    parser.add_argument("--cells-csv", default="metadata_cells.csv",
                        help="Name of the cells metadata CSV")
    return parser.parse_args()


# ── Import ELISA components ──────────────────────────────────
def init_elisa(args):
    """Initialize engine, LLM, adata — uses v4 hybrid engine."""
    from retrieval_engine_v4_hybrid import RetrievalEngine
    from elisa_analysis import (find_interactions, pathway_scoring,
                                proportion_analysis, comparative_analysis,
                                query_pathway)
    from elisa_report import ReportBuilder
    import elisa_viz as viz

    print("[INIT] Loading retrieval engine (v4 hybrid)...")
    raw_engine = RetrievalEngine(
        base=args.base,
        pt_name=args.pt_name,
        cells_csv=args.cells_csv,
    )
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

        def query_hybrid(self, text, top_k=5, lambda_sem=0.5, with_genes=False, **kwargs):
            return self._eng.query_hybrid(text, top_k=top_k, lambda_sem=lambda_sem, with_genes=with_genes)

        def query_annotation_only(self, text, top_k=5, with_genes=False):
            return self._eng.query_annotation_only(text, top_k=top_k, with_genes=with_genes)

        def discover(self, text, top_k=5, **kwargs):
            payload = self._eng.query_semantic(text, top_k=top_k, with_genes=True)
            payload["mode"] = "discovery"
            payload["query"] = text
            return payload

        def interactions(self, source=None, target=None, **kwargs):
            src_clusters = None
            tgt_clusters = None
            if source:
                src_lower = source.lower()
                src_clusters = [cid for cid in self._eng.cluster_ids if src_lower in str(cid).lower()]
                if not src_clusters: src_clusters = None
            if target:
                tgt_lower = target.lower()
                tgt_clusters = [cid for cid in self._eng.cluster_ids if tgt_lower in str(cid).lower()]
                if not tgt_clusters: tgt_clusters = None
            return find_interactions(self._eng.gene_stats, self._eng.cluster_ids,
                                     source_clusters=src_clusters, target_clusters=tgt_clusters, **kwargs)

        def pathway(self, pathway_name=None, **kwargs):
            if pathway_name:
                return query_pathway(self._eng.gene_stats, self._eng.cluster_ids,
                                     pathway_name=pathway_name, **kwargs)
            else:
                return pathway_scoring(self._eng.gene_stats, self._eng.cluster_ids, **kwargs)

        def proportions(self, **kwargs):
            return proportion_analysis(self._eng.metadata, **kwargs)

        def compare(self, group_a, group_b, genes=None, **kwargs):
            condition_col = None
            meta = self._eng.metadata
            if meta:
                for cid, m in meta.items():
                    if isinstance(m, dict):
                        fields = m.get("fields", {})
                        for col, dist in fields.items():
                            if isinstance(dist, dict):
                                dist_keys_lower = {k.lower() for k in dist}
                                if group_a.lower() in dist_keys_lower or group_b.lower() in dist_keys_lower:
                                    condition_col = col
                                    break
                    if condition_col: break
            if not condition_col: condition_col = "timepoint"
            return comparative_analysis(self._eng.gene_stats, self._eng.metadata,
                                        condition_col=condition_col, group_a=group_a, group_b=group_b,
                                        genes=genes, **kwargs)

        def detect_capabilities(self):
            caps = {"has_conditions": False, "condition_values": [], "condition_column": None,
                    "n_clusters": len(self._eng.cluster_ids), "cluster_ids": list(self._eng.cluster_ids)}
            meta = self._eng.metadata
            if not meta: return caps
            condition_keywords = ["patient_group", "case_control", "condition", "disease",
                                  "treatment", "status", "timepoint", "pre_post", "group", "sample_type"]
            col_candidates = {}
            for cid, m in meta.items():
                if not isinstance(m, dict): continue
                fields = m.get("fields", {})
                for col, dist in fields.items():
                    if isinstance(dist, dict):
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
            skip = ["batch", "lab", "donor", "sample", "plate", "lane", "seq", "lib", "run",
                    "class", "subclass", "level", "annotation", "ontology", "type", "assay", "sex"]
            for col, values in col_candidates.items():
                if len(values) == 2:
                    if not any(p in col.lower() for p in skip):
                        caps["has_conditions"] = True
                        caps["condition_column"] = col
                        caps["condition_values"] = sorted(values)
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
            print("[WARN] scanpy not installed, skipping h5ad")

    report = ReportBuilder(dataset_name=args.dataset_name)
    caps = engine.detect_capabilities()
    print(f"[INIT] Conditions: {caps.get('has_conditions', False)} → {caps.get('condition_values', [])}")

    return engine, llm, adata, report, viz, caps


# ── LLM helper ───────────────────────────────────────────────
def ask_llm(llm, system_prompt, user_prompt):
    res = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    return res.choices[0].message.content.strip()


SYSTEM_PROMPT = (
    "You are ELISA, an expert assistant for single-cell biology and "
    "pediatric oncology. Never hallucinate. Always ground claims "
    "strictly in provided data. Be concise and scientific. "
    "Focus on high-risk neuroblastoma biology: neoplastic cell states "
    "(ADRN-calcium, ADRN-baseline, ADRN-proliferating, ADRN-dopaminergic, "
    "Interm-OXPHOS, MES), macrophage subtypes (IL18+, VCAN+, CCL4+, "
    "C1QC+SPP1+, F13A1+, HS3ST2+, THY1+, Proliferating), "
    "HB-EGF–ERBB4 paracrine signaling axis, chemotherapy-induced "
    "tumor microenvironment rewiring, ErbB/ERK/MAPK signaling, "
    "immune evasion, and longitudinal DX vs PTX changes."
)


MAX_PROMPT_CHARS = 12000

def trim_payload(payload, max_chars=MAX_PROMPT_CHARS):
    trimmed = dict(payload)
    if "clusters" in trimmed and isinstance(trimmed["clusters"], dict):
        clusters = trimmed["clusters"]
        if len(clusters) > 10:
            ranked = sorted(clusters.items(), key=lambda kv: len(kv[1].get("genes", [])), reverse=True)[:10]
            trimmed["clusters"] = dict(ranked)
    if "interactions" in trimmed and isinstance(trimmed["interactions"], list):
        if len(trimmed["interactions"]) > 30:
            trimmed["interactions"] = trimmed["interactions"][:30]
    if "scores" in trimmed and isinstance(trimmed["scores"], list):
        trimmed["scores"] = trimmed["scores"][:10]
    for r in trimmed.get("results", []):
        if "gene_evidence" in r and len(r["gene_evidence"]) > 5:
            r["gene_evidence"] = r["gene_evidence"][:5]
    ctx = json.dumps(trimmed, indent=1, default=str)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n... [TRUNCATED]"
    return ctx

def build_prompt(mode, query, payload):
    ctx = trim_payload(payload)
    if mode == "discovery":
        return f"""You are in DISCOVERY mode.
Separate into: 1. DATASET EVIDENCE 2. ESTABLISHED BIOLOGY
3. CONSISTENCY ANALYSIS 4. CANDIDATE NOVEL HYPOTHESES
QUESTION: {query}
DATASET CONTEXT: {ctx}"""
    elif mode == "compare":
        return f"""You are ELISA analyzing COMPARATIVE analysis.
COMPARISON: {query}
EVIDENCE: {ctx}
Identify condition-biased clusters, highlight differentially expressed genes."""
    elif mode == "interactions":
        return f"""You are ELISA analyzing CELL-CELL INTERACTIONS.
QUERY: {query}
INTERACTIONS: {ctx}
Focus on highest-scoring, group by pathway, note unexpected interactions."""
    elif mode == "proportions":
        return f"""You are ELISA analyzing CELL TYPE PROPORTIONS.
QUERY: {query}
DATA: {ctx}
Report major types, condition differences, biological implications."""
    elif mode in ("pathway_scoring", "pathway_query"):
        return f"""You are ELISA analyzing PATHWAY ACTIVITY.
QUERY: {query}
SCORES: {ctx}
Identify top cell types, contributing genes, biological relevance."""
    else:
        return f"""You are ELISA for single-cell analysis.
MODE: {mode.upper()} | QUERY: {query}
EVIDENCE: {ctx}
Use ONLY provided evidence. Be concise and scientific."""


# ══════════════════════════════════════════════════════════════
# PAPER-SPECIFIC QUERIES
# Yu et al. (2025) Nat Genet 57, 1142–1154
# ══════════════════════════════════════════════════════════════

def get_queries(skip_plots=False):
    queries = [
        # ── Phase 1: Landscape & composition (Fig 1) ──
        ("info", "info"),
        ("proportions", "proportions"),

        # ── Phase 2: Major cell types (Fig 1b–d) ──
        ("semantic", "semantic: neuroblast neoplastic cell PHOX2B ISL1 sympathetic nervous system neural crest"),
        ("semantic", "semantic: macrophage tumor associated CD68 CD163 CD86 CSF1R in neuroblastoma microenvironment"),
        ("semantic", "semantic: T cell lymphocyte CD247 CD96 infiltrating neuroblastoma tumor"),
        ("semantic", "semantic: B cell PAX5 MS4A1 CD79A humoral immunity in pediatric neuroblastoma"),
        ("semantic", "semantic: Schwann cell PLP1 CDH19 SOX10 myelinating glial cell neuroblastoma"),
        ("semantic", "semantic: fibroblast PDGFRB DCN COL1A1 stroma extracellular matrix neuroblastoma"),
        ("semantic", "semantic: endothelial cell PECAM1 PTPRB CDH5 tumor vasculature neuroblastoma"),
        ("semantic", "semantic: dendritic cell IRF8 FLT3 antigen presentation in neuroblastoma"),

        # ── Phase 3: Neoplastic cell states (Fig 2) ──
        ("discover", "discover: PHOX2B ISL1 HAND2 TH DBH CHGA adrenergic neuroblast identity markers"),
        ("discover", "discover: MYCN MKI67 TOP2A EZH2 SMC4 BIRC5 ADRN-proliferating neuroblastoma cell cycle"),
        ("discover", "discover: CACNA1B SYN2 KCNMA1 KCNQ3 CREB5 ADRN-calcium synaptic signaling neuroblast"),
        ("discover", "discover: SLC18A2 TH DDC AGTR2 ATP2A2 ADRN-dopaminergic catecholamine biosynthesis"),
        ("discover", "discover: RPL3 RPL4 RPS6 ribosomal genes oxidative phosphorylation Interm-OXPHOS state"),
        ("discover", "discover: YAP1 FN1 VIM COL1A1 SERPINE1 mesenchymal MES neuroblastoma cell state"),
        ("discover", "discover: JUN FOS JUNB JUND FOSL2 BACH1 BACH2 AP-1 transcription factors MES state"),
        ("discover", "discover: EZH2 PRC2 polycomb repressive complex proliferating state chromatin regulation"),
        ("discover", "discover: SMC4 condensin genome organization ADRN-proliferating neuroblastoma"),
        ("discover", "discover: NECTIN2 immune checkpoint MES state T cell dysfunction neuroblastoma"),
        ("discover", "discover: PHOX2A PHOX2B GATA3 adrenergic transcription factor motifs snATAC-seq"),
        ("discover", "discover: ZNF148 MAZ transcriptional regulators MES ADRN-proliferating states"),
        ("discover", "discover: CTCF MAZ SP1 enhancer EZH2 locus ADRN-proliferating state epigenetics"),

        # ── Phase 4: Macrophage subtypes (Fig 4) ──
        ("discover", "discover: IL18 pro-inflammatory macrophage subtype neuroblastoma reduced after therapy"),
        ("discover", "discover: VCAN VEGFA pro-angiogenic macrophage expanded after chemotherapy neuroblastoma"),
        ("discover", "discover: CCL4 CCL3 pro-angiogenic macrophage chemokine signaling neuroblastoma"),
        ("discover", "discover: C1QC SPP1 CD68 CD163 immunosuppressive macrophage complement neuroblastoma"),
        ("discover", "discover: F13A1 tissue resident macrophage phagocytosis expanded post-therapy"),
        ("discover", "discover: HS3ST2 CYP27A1 lipid-associated macrophage metabolic phenotype"),
        ("discover", "discover: THY1 MRC1 undefined macrophage phenotype neuroblastoma"),
        ("discover", "discover: MKI67 TOP2A proliferating macrophage expansion after therapy"),
        ("discover", "discover: HBEGF macrophage ligand ERBB4 receptor neuroblast paracrine signaling"),
        ("discover", "discover: TGFA EREG AREG ICAM1 ErbB receptor ligands macrophage neuroblast"),

        # ── Phase 5: HB-EGF–ERBB4 axis (Fig 4e, Fig 6) ──
        ("semantic", "semantic: HB-EGF ERBB4 paracrine signaling macrophage neuroblast ErbB pathway"),
        ("semantic", "semantic: ERK MAPK phosphorylation downstream ERBB4 activation neuroblastoma"),
        ("semantic", "semantic: CRM197 HB-EGF inhibitor afatinib ERBB tyrosine kinase inhibitor neuroblastoma"),
        ("discover", "discover: HBEGF ERBB4 EGFR MAPK1 MAPK3 ERK signaling axis macrophage neuroblast"),
        ("discover", "discover: TGFA ERBB4 ligand receptor interaction VCAN macrophage neuroblast"),
        ("discover", "discover: EREG ERBB4 epiregulin growth factor receptor neuroblastoma signaling"),
        ("discover", "discover: VCAN EGFR ITGB1 cell adhesion migration angiogenesis macrophage"),
        ("discover", "discover: THBS1 CD47 LRP5 ITGA3 ITGB1 dont eat me signal immune evasion"),
        ("discover", "discover: VEGFA GPC1 NRP1 angiogenesis macrophage endothelial signaling"),
        ("discover", "discover: APOE LDLR VLDLR LPL lipid metabolism macrophage neuroblast interaction"),

        # ── Phase 6: Therapy-induced shifts (Fig 1f, Fig 2e) ──
        ("semantic", "semantic: chemotherapy induced tumor microenvironment rewiring macrophage expansion"),
        ("semantic", "semantic: ADRN-proliferating ADRN-baseline decrease after induction chemotherapy"),
        ("semantic", "semantic: ADRN-calcium ADRN-dopaminergic Interm-OXPHOS increase post-therapy"),
        ("semantic", "semantic: mesenchymal MES state chemotherapy response prognosis neuroblastoma"),
        ("discover", "discover: macrophage Schwann cell fibroblast expansion post-therapy microenvironment"),
        ("discover", "discover: B cell proportion decrease after induction chemotherapy neuroblastoma"),
        ("discover", "discover: ALK mutation smaller decrease ADRN-baseline ADRN-proliferating after therapy"),
        ("discover", "discover: MYCN amplification neoplastic state enrichment advanced stage disease"),

        # ── Phase 7: Clinical outcome (Fig 2g–i) ──
        ("discover", "discover: ADRN-proliferating Interm-OXPHOS poor overall survival event-free survival"),
        ("discover", "discover: ADRN-calcium ADRN-dopaminergic better prognosis differentiated state"),
        ("discover", "discover: MES state high proportion worse chemotherapy response adverse events"),
        ("discover", "discover: proliferative metabolically active developmentally arrested poor outcome"),

        # ── Phase 8: Immune evasion & T cell biology ──
        ("discover", "discover: NECTIN2 CD274 B2M HLA-A immune evasion neuroblastoma"),
        ("discover", "discover: CD247 CD96 CD3D CD8A T cell infiltration neuroblastoma"),
        ("discover", "discover: GZMA GZMB PRF1 IFNG cytotoxic T cell tumor killing"),
        ("discover", "discover: IRF8 FLT3 CLEC9A dendritic cell antigen presentation priming"),

        # ── Phase 9: Schwann cell and fibroblast biology ──
        ("discover", "discover: PLP1 CDH19 SOX10 MBP Schwann cell precursor neural crest expanded therapy"),
        ("discover", "discover: PDGFRB DCN LUM COL1A1 COL1A2 fibroblast stroma cancer associated"),

        # ── Phase 10: Adjacent tissue ──
        ("discover", "discover: CYP11A1 CYP11B1 CYP17A1 STAR adrenal cortex steroidogenesis normal tissue"),
        ("discover", "discover: ALB DCDC2 HNF4A hepatocyte liver adjacent tissue neuroblastoma"),
        ("discover", "discover: PKHD1 PAX2 WT1 kidney cell renal adjacent tissue neuroblastoma"),

        # ── Phase 11: Comparisons — DX vs PTX ──
        ("compare", "compare: DX vs PTX | PHOX2B, ISL1, MYCN, MKI67, TOP2A, EZH2, SMC4"),
        ("compare", "compare: DX vs PTX | CD68, CD163, HBEGF, VCAN, IL18, SPP1, C1QC"),
        ("compare", "compare: DX vs PTX | PLP1, CDH19, SOX10, PDGFRB, DCN, COL1A1"),
        ("compare", "compare: DX vs PTX | CD247, CD96, PAX5, MS4A1, PECAM1, PTPRB"),
        ("compare", "compare: DX vs PTX | ERBB4, EGFR, MAPK1, MAPK3, HBEGF, TGFA, EREG"),
        ("compare", "compare: DX vs PTX | YAP1, FN1, VIM, SERPINE1, JUN, FOS, JUNB"),
        ("compare", "compare: DX vs PTX | NECTIN2, CD274, B2M, HLA-A, HLA-B, THBS1, CD47"),
        ("compare", "compare: DX vs PTX | CACNA1B, SYN2, KCNMA1, CREB5, SLC18A2, TH, DDC"),

        # ── Phase 12: Pathway analyses ──
        ("pathway", "pathway: EGF signaling"),
        ("pathway", "pathway: MAPK signaling"),
        ("pathway", "pathway: PI3K-Akt signaling"),
        ("pathway", "pathway: Calcium signaling"),
        ("pathway", "pathway: Oxidative phosphorylation"),
        ("pathway", "pathway: Cell cycle"),
        ("pathway", "pathway: Axon guidance"),
        ("pathway", "pathway: ECM-receptor interaction"),
        ("pathway", "pathway: Focal adhesion"),
        ("pathway", "pathway: Angiogenesis"),
        ("pathway", "pathway: Complement and coagulation"),
        ("pathway", "pathway: Antigen processing and presentation"),
        ("pathway", "pathway: Cytokine-cytokine receptor interaction"),
        ("pathway", "pathway: Chemokine signaling"),
        ("pathway", "pathway: TNF signaling"),
        ("pathway", "pathway: Apoptosis"),
        ("pathway", "pathway: all"),

        # ── Phase 13: Cell-cell interactions ──
        ("interactions", "interactions:"),
        ("interactions", "interactions: macrophage -> neuroblast"),
        ("interactions", "interactions: neuroblast -> macrophage"),
        ("interactions", "interactions: macrophage -> endothelial"),
        ("interactions", "interactions: fibroblast -> neuroblast"),
        ("interactions", "interactions: T cell -> neuroblast"),
        ("interactions", "interactions: dendritic -> T cell"),
        ("interactions", "interactions: endothelial -> neuroblast"),
        ("interactions", "interactions: Schwann -> neuroblast"),

        # ── Phase 14: Cancer biology discussion ──
        ("discover", "discover: EZH2 therapeutic target PRC2 ADRN-proliferating neuroblastoma treatment"),
        ("discover", "discover: afatinib pan-ERBB inhibitor CRM197 HB-EGF inhibitor therapeutic strategy"),
        ("discover", "discover: ADRN to MES phenotypic spectrum adrenergic mesenchymal transition"),
        ("discover", "discover: treatment resistance persistent proliferating adrenergic cells chemotherapy"),
        ("discover", "discover: colony formation assay macrophage co-culture neuroblastoma growth"),
        ("discover", "discover: ERK phosphorylation increased AKT unchanged HB-EGF ErbB signaling"),
        ("discover", "discover: GD2 immunotherapy target ADRN-like-2 ERBB4 high neuroblast CODEX"),
        ("discover", "discover: Xenium spatial transcriptomics TH-MYCN mouse model validation"),

        # ── Phase 15: Additional gene recovery queries ──
        ("discover", "discover: PHOX2B ISL1 HAND2 TH DBH DDC CHGA CHGB neural crest markers"),
        ("discover", "discover: CACNA2D1 GPC5 EML6 KCNQ3 calcium channel neuroblast"),
        ("discover", "discover: PAPPA FMN1 MEF2D DOCK3 NFE2L2 ADRN-calcium target genes"),
        ("discover", "discover: BIRC5 BUB1B ASPM KIF11 CENPU TOP2A cell cycle proliferation"),
        ("discover", "discover: COL4A1 COL4A2 THBS2 P3H2 NNMT MES state extracellular matrix"),
        ("discover", "discover: APOE SPP1 TREM2 C1QC CD68 CD163 macrophage polarization"),
        ("discover", "discover: ALK RET NTRK1 NTRK2 oncogenic drivers neuroblastoma"),
        ("discover", "discover: PECAM1 CDH5 VWF KDR FLT1 endothelial vascular markers"),
        ("discover", "discover: CYP11A1 CYP11B1 STAR NR5A1 adrenal steroidogenesis"),
        # ── Phase 16: Targeted gene recovery — missing genes ──

        # Batch A: Neuroblast identity & catecholamine
        ("compare", "compare: DX vs PTX | MYCN, GATA3, ASCL1, TH, DBH, DDC, CHGA, CHGB"),
        ("compare", "compare: DX vs PTX | SLC18A2, AGTR2, ATP2A2, CREB5, SYN2, KCNMA1, KCNQ3"),

        # Batch B: ADRN-calcium / dopaminergic targets
        ("compare", "compare: DX vs PTX | CACNA2D1, GPC5, EML6, KLHL29, FMN1, PAPPA, MEF2D, DOCK3"),
        ("compare", "compare: DX vs PTX | NFE2L2, MAMLD1, SMARCC1, NTM, MYOF, STARD13"),

        # Batch C: ADRN-proliferating TFs & cell cycle
        ("compare", "compare: DX vs PTX | MAZ, CTCF, SP1, PBX3, TCF3, TFDP1, POLQ, TOX"),

        # Batch D: MES state genes
        ("compare", "compare: DX vs PTX | YAP1, VIM, SPARC, NNMT, P3H2, CRISPLD2"),
        ("compare", "compare: DX vs PTX | JUNB, JUND, FOSL2, BACH1, BACH2, ZNF148"),

        # Batch E: MES regulators
        ("compare", "compare: DX vs PTX | ETS1, ETV6, ELF1, KLF6, KLF7, RUNX1, GLI3, ERG"),
        ("compare", "compare: DX vs PTX | SP3, PURA, NFIC, RREB1, STK3, GAS7, DOCK8, KANK1"),

        # Batch F: MES regulatory network targets
        ("compare", "compare: DX vs PTX | UACA, ARHGAP42, LTBP1, TNS1, TNS3, PTPRE, PRSS23"),
        ("compare", "compare: DX vs PTX | SLC2A3, IL1R1, C7, NECTIN2, CD274, CD47"),

        # Batch G: Macrophage subtypes
        ("compare", "compare: DX vs PTX | CD68, CD163, MRC1, IL18, VCAN, SPP1, APOE, TREM2"),
        ("compare", "compare: DX vs PTX | HS3ST2, CYP27A1, LYVE1, F13A1, THY1, IRF8"),

        # Batch H: Immune cells
        ("compare", "compare: DX vs PTX | CD247, CD96, CD3D, CD3E, CD4, CD8A, GZMA, GZMB, PRF1"),
        ("compare", "compare: DX vs PTX | PAX5, MS4A1, CD19, CD79A, FLT3, CLEC9A"),

        # Batch I: Schwann / fibroblast / endothelial
        ("compare", "compare: DX vs PTX | PLP1, CDH19, SOX10, MBP, MPZ, S100B"),
        ("compare", "compare: DX vs PTX | PDGFRB, DCN, LUM, FAP, ACTA2, PTPRB"),

        # Batch J: Adrenal / hepatocyte / kidney
        ("compare", "compare: DX vs PTX | CYP11A1, CYP11B1, CYP17A1, STAR, NR5A1"),
        ("compare", "compare: DX vs PTX | HNF4A, DCDC2, PKHD1, PAX2, WT1, ALB"),

        # Batch K: Oncogenes & signaling
        ("compare", "compare: DX vs PTX | ALK, RET, GPC1, LRP5, ICAM1, EREG"),

        # Batch L: Ribosomal / housekeeping
        ("compare", "compare: DX vs PTX | RPL3, RPL4, RPS3, RPS6"),

        # Batch M: Semantic recovery for stubborn genes
        ("semantic", "semantic: MYCN GATA3 ASCL1 TH DBH DDC CHGA neuroblast adrenergic identity"),
        ("semantic", "semantic: YAP1 VIM SPARC NNMT mesenchymal MES state neuroblastoma ECM"),
        ("semantic", "semantic: JUNB JUND FOSL2 BACH1 BACH2 AP-1 transcription factor MES state"),
        ("semantic", "semantic: CD68 CD163 MRC1 IL18 VCAN SPP1 APOE TREM2 macrophage subtypes"),
        ("semantic", "semantic: PLP1 CDH19 SOX10 MBP MPZ S100B Schwann cell myelinating glial"),
        ("semantic", "semantic: CD247 CD96 CD3D CD3E CD8A GZMA GZMB PRF1 T cell cytotoxic"),
        ("semantic", "semantic: PAX5 MS4A1 CD19 CD79A B cell lymphocyte humoral immunity"),
        ("semantic", "semantic: CYP11A1 CYP11B1 STAR NR5A1 adrenal cortex steroidogenesis"),
        ("semantic", "semantic: MAZ CTCF SP1 ZNF148 transcriptional regulator enhancer EZH2"),
        ("semantic", "semantic: NECTIN2 CD274 CD47 immune checkpoint evasion neuroblastoma"),
        ("semantic", "semantic: ALK RET NTRK1 MYCN oncogenic driver neuroblastoma kinase"),
        ("semantic", "semantic: HS3ST2 CYP27A1 LYVE1 lipid associated macrophage metabolic"),
        ("semantic", "semantic: ETS1 ETV6 ELF1 KLF6 RUNX1 MES state transcription factor regulator"),
        ("semantic", "semantic: CACNA2D1 SYN2 KCNMA1 KCNQ3 calcium channel synaptic neuroblast"),
        ("semantic", "semantic: FAP ACTA2 PDGFRB DCN LUM fibroblast stroma neuroblastoma"),
        # ── Phase 17: Second gene recovery push ──

        # Batch N: Neuroblast identity — still missing
        ("compare", "compare: DX vs PTX | MYCN, GATA3, ASCL1, TH, DBH, DDC, CHGA, CHGB"),
        ("semantic", "semantic: MYCN amplification GATA3 ASCL1 neuroblast transcription factor identity"),
        ("semantic", "semantic: TH DBH DDC catecholamine biosynthesis dopamine norepinephrine neuroblast"),
        ("semantic", "semantic: CHGA CHGB chromogranin neuroendocrine secretory granule neuroblast"),

        # Batch O: ADRN-calcium missing genes
        ("compare", "compare: DX vs PTX | CACNA2D1, GPC5, EML6, KLHL29, PAPPA, SYN2, KCNMA1, KCNQ3"),
        ("compare", "compare: DX vs PTX | FMN1, MEF2D, DOCK3, NFE2L2, CREB5, MAMLD1"),

        # Batch P: MES state — still stubborn
        ("compare", "compare: DX vs PTX | YAP1, VIM, SPARC, NNMT, P3H2, MYOF, CRISPLD2"),
        ("compare", "compare: DX vs PTX | JUNB, JUND, FOSL2, BACH1, BACH2, ZNF148, NTM"),
        ("semantic", "semantic: YAP1 VIM SPARC NNMT mesenchymal MES neuroblastoma extracellular matrix"),
        ("semantic", "semantic: JUNB JUND FOSL2 BACH1 BACH2 AP-1 transcription factor neuroblastoma"),

        # Batch Q: MES regulators — still missing
        ("compare", "compare: DX vs PTX | ETS1, ETV6, ELF1, KLF6, KLF7, RUNX1, ERG, GLI3"),
        ("compare", "compare: DX vs PTX | SP1, SP3, PURA, NFIC, RREB1, STK3, GAS7"),
        ("compare", "compare: DX vs PTX | DOCK8, KANK1, UACA, ARHGAP42, STARD13, TNS1, TNS3"),
        ("semantic", "semantic: ETS1 ETV6 ELF1 KLF6 KLF7 RUNX1 MES transcription factor regulator"),
        ("semantic", "semantic: GLI3 ERG SP3 PURA NFIC RREB1 regulatory network neuroblastoma"),

        # Batch R: ADRN-proliferating TFs
        ("compare", "compare: DX vs PTX | MAZ, CTCF, SP1, PBX3, TCF3, TFDP1, POLQ, TOX"),
        ("compare", "compare: DX vs PTX | SMARCC1, RET, ALK, PKHD1, PAX2, WT1"),
        ("semantic", "semantic: MAZ CTCF SP1 enhancer transcriptional regulator ADRN-proliferating"),

        # Batch S: Macrophage subtypes still missing
        ("compare", "compare: DX vs PTX | HS3ST2, CYP27A1, LYVE1, THY1, IRF8, PRSS23, PTPRE"),
        ("semantic", "semantic: HS3ST2 CYP27A1 lipid-associated macrophage metabolic neuroblastoma"),
        ("semantic", "semantic: LYVE1 lymphatic vessel endothelial macrophage tissue resident"),

        # Batch T: Schwann / fibroblast still missing
        ("compare", "compare: DX vs PTX | PLP1, CDH19, SOX10, MBP, MPZ, S100B"),
        ("compare", "compare: DX vs PTX | DCN, LUM, FAP, ACTA2, PDGFRB, PTPRB"),
        ("semantic", "semantic: PLP1 CDH19 SOX10 MBP MPZ S100B Schwann cell neural crest myelinating"),
        ("semantic", "semantic: DCN LUM FAP ACTA2 PDGFRB fibroblast stroma cancer associated"),

        # Batch U: B cell / dendritic / remaining immune
        ("compare", "compare: DX vs PTX | PAX5, MS4A1, CD19, CD79A, FLT3, CLEC9A"),
        ("semantic", "semantic: PAX5 MS4A1 CD19 CD79A B cell neuroblastoma lymphocyte"),
        ("semantic", "semantic: FLT3 CLEC9A IRF8 dendritic cell antigen presentation priming"),

        # Batch V: Adrenal / adjacent tissue
        ("compare", "compare: DX vs PTX | CYP11A1, CYP11B1, CYP17A1, STAR, NR5A1, DCDC2, HNF4A"),
        ("semantic", "semantic: CYP11A1 CYP11B1 CYP17A1 STAR adrenal cortex steroidogenesis"),

        # Batch W: Remaining network targets
        ("compare", "compare: DX vs PTX | LTBP1, LRP5, GPC1, FOSL2, RPL3, RPL4, RPS3"),
        ("compare", "compare: DX vs PTX | SLC18A2, AGTR2, ATP2A2, DCDC2, HNF4A"),
        ("semantic", "semantic: SLC18A2 AGTR2 ATP2A2 dopaminergic synapse catecholamine transport"),
        ("semantic", "semantic: RPL3 RPL4 RPS3 ribosomal protein oxidative phosphorylation Interm-OXPHOS"),
    ]

    if not skip_plots:
        plot_queries = [
            ("plot", "plot:umap"),
            # Neuroblast markers
            ("plot", "plot:expr PHOX2B"), ("plot", "plot:expr ISL1"), ("plot", "plot:expr MYCN"),
            ("plot", "plot:expr TH"), ("plot", "plot:expr DBH"), ("plot", "plot:expr HAND2"),
            # Proliferation / cell state
            ("plot", "plot:expr MKI67"), ("plot", "plot:expr TOP2A"), ("plot", "plot:expr EZH2"),
            ("plot", "plot:expr YAP1"), ("plot", "plot:expr VIM"), ("plot", "plot:expr FN1"),
            # HB-EGF axis
            ("plot", "plot:expr HBEGF"), ("plot", "plot:expr ERBB4"), ("plot", "plot:expr EGFR"),
            ("plot", "plot:expr TGFA"), ("plot", "plot:expr EREG"),
            # Macrophage
            ("plot", "plot:expr CD68"), ("plot", "plot:expr CD163"), ("plot", "plot:expr IL18"),
            ("plot", "plot:expr VCAN"), ("plot", "plot:expr SPP1"), ("plot", "plot:expr C1QC"),
            ("plot", "plot:expr F13A1"), ("plot", "plot:expr HS3ST2"), ("plot", "plot:expr CCL4"),
            # Immune
            ("plot", "plot:expr CD247"), ("plot", "plot:expr CD96"), ("plot", "plot:expr PAX5"),
            ("plot", "plot:expr IRF8"), ("plot", "plot:expr NECTIN2"),
            # Schwann / fibroblast / endothelial
            ("plot", "plot:expr PLP1"), ("plot", "plot:expr CDH19"), ("plot", "plot:expr SOX10"),
            ("plot", "plot:expr PDGFRB"), ("plot", "plot:expr DCN"), ("plot", "plot:expr PECAM1"),
            # Dotplots
            ("plot", "plot:dotplot PHOX2B, ISL1, HAND2, TH, DBH, MYCN, MKI67, EZH2"),
            ("plot", "plot:dotplot CD68, CD163, IL18, VCAN, SPP1, C1QC, F13A1, CCL4"),
            ("plot", "plot:dotplot HBEGF, ERBB4, EGFR, TGFA, EREG, MAPK1, MAPK3, AREG"),
            ("plot", "plot:dotplot YAP1, FN1, VIM, JUN, FOS, SERPINE1, NECTIN2, THBS1"),
            ("plot", "plot:dotplot PLP1, CDH19, SOX10, PDGFRB, DCN, COL1A1, PECAM1, PTPRB"),
            ("plot", "plot:dotplot CD247, CD96, PAX5, MS4A1, IRF8, FLT3, CD274, B2M"),
            ("plot", "plot:dotplot CACNA1B, SYN2, KCNMA1, SLC18A2, DDC, AGTR2, CREB5, GPC5"),
            # Grids
            ("plot", "plot:grid PHOX2B, MYCN, MKI67, YAP1, VIM, EZH2"),
            ("plot", "plot:grid HBEGF, ERBB4, CD68, VCAN, IL18, SPP1"),
            ("plot", "plot:grid PLP1, PDGFRB, PECAM1, CD247, PAX5, IRF8"),
        ]
        queries.extend(plot_queries)

    return queries


def get_queries_test():
    """Minimal test set for quick validation."""
    return [
        ("info", "info"),
        ("proportions", "proportions"),
        ("semantic", "semantic: neuroblast PHOX2B ISL1 neural crest neuroblastoma"),
        ("discover", "discover: HBEGF ERBB4 macrophage neuroblast paracrine signaling"),
        ("compare", "compare: DX vs PTX | PHOX2B, CD68, HBEGF, ERBB4, MKI67"),
        ("interactions", "interactions:"),
        ("interactions", "interactions: macrophage -> neuroblast"),
        ("pathway", "pathway: EGF signaling"),
        ("pathway", "pathway: all"),
    ]


# ══════════════════════════════════════════════════════════════
# PAPER-SPECIFIC EVALUATION REFERENCE DATA
# Yu et al. (2025) Nature Genetics 57, 1142–1154
# ══════════════════════════════════════════════════════════════

PAPER_GENES = {
    # Neuroblast identity (Fig 1d, Fig 2)
    "PHOX2B", "ISL1", "HAND2", "TH", "DBH", "DDC", "CHGA", "CHGB",
    "MYCN", "PHOX2A", "GATA3", "ASCL1",
    # ADRN-proliferating (Fig 2, Fig 3)
    "MKI67", "TOP2A", "EZH2", "SMC4", "BIRC5", "BUB1B", "ASPM", "KIF11",
    "CENPU", "KIF18B", "KIF2C", "TACC3", "ANLN", "CDC45", "CDC25C",
    "MAZ", "CTCF", "SP1", "PBX3", "TCF3", "TFDP1", "E2F5",
    # ADRN-calcium (Fig 2, Fig 3)
    "CACNA1B", "CACNA2D1", "SYN2", "KCNMA1", "KCNQ3", "CREB5", "GPC5",
    "KLHL29", "HS3ST2", "PAPPA", "FMN1", "MEF2D", "DOCK3", "NFE2L2",
    # ADRN-dopaminergic
    "SLC18A2", "AGTR2", "ATP2A2",
    # Interm-OXPHOS
    "RPL3", "RPL4", "RPS6", "RPS3",
    # MES state (Fig 2, Fig 3)
    "YAP1", "FN1", "VIM", "COL1A1", "COL4A1", "COL4A2", "SERPINE1",
    "SPARC", "THBS2", "NNMT", "P3H2",
    # AP-1 TFs
    "JUN", "FOS", "JUNB", "JUND", "FOSL2", "BACH1", "BACH2",
    # MES regulators
    "ZNF148", "ETS1", "ETV6", "ELF1", "KLF6", "KLF7", "RUNX1",
    "GLI3", "ERG", "SP3", "PURA", "NFIC", "RREB1", "STK3",
    # Macrophage markers (Fig 4)
    "CD68", "CD163", "CD86", "CSF1R", "MRC1",
    "IL18", "VCAN", "VEGFA", "CCL4", "CCL3",
    "C1QC", "SPP1", "APOE", "TREM2",
    "F13A1", "LYVE1",
    "HS3ST2", "CYP27A1",
    "THY1",
    # HB-EGF axis (Fig 4e, Fig 6)
    "HBEGF", "ERBB4", "EGFR", "TGFA", "EREG", "AREG", "ICAM1",
    "MAPK1", "MAPK3",
    # Immune evasion
    "NECTIN2", "CD274", "B2M", "HLA-A", "HLA-B", "HLA-C",
    # Cell adhesion / ECM / angiogenesis interactions
    "THBS1", "CD47", "ITGB1", "ITGA3", "LRP5",
    "VEGFA", "GPC1", "NRP1",
    "SEMA3A",
    # T cell
    "CD247", "CD96", "CD3D", "CD3E", "CD8A", "CD4",
    "GZMA", "GZMB", "PRF1", "IFNG",
    # B cell
    "PAX5", "MS4A1", "CD19", "CD79A", "HLA-DRA", "HLA-DRB1",
    # Dendritic cell
    "IRF8", "FLT3", "CLEC9A", "CD80",
    # Schwann cell
    "PLP1", "CDH19", "SOX10", "MPZ", "MBP", "S100B",
    # Fibroblast
    "PDGFRB", "DCN", "LUM", "COL1A2", "PDGFRA", "FAP", "ACTA2",
    # Endothelial
    "PECAM1", "PTPRB", "CDH5", "VWF", "KDR", "FLT1",
    # Adrenal cortex
    "CYP11A1", "CYP11B1", "CYP17A1", "STAR", "NR5A1",
    # Hepatocyte
    "ALB", "DCDC2", "HNF4A",
    # Kidney
    "PKHD1", "PAX2", "WT1",
    # Oncogenes
    "ALK", "RET", "NTRK1", "NTRK2",
    # Additional from regulatory networks (Fig 3)
    "SMARCC1", "MAMLD1", "EML6", "TOX", "POLQ", "MELK",
    "CRISPLD2", "LTBP1", "TNS1", "TNS3", "MYOF", "STARD13",
    "DOCK8", "KANK1", "UACA", "ARHGAP42", "GAS7",
    "PRSS23", "DUSP1", "SLC2A3", "PTPRE", "NTM",
    "IL1R1", "C7",
}

PAPER_INTERACTIONS = [
    ("macrophage", "neuroblast", "HB-EGF/ERBB4 signaling"),
    ("macrophage", "neuroblast", "TGFA/ERBB4 signaling"),
    ("macrophage", "neuroblast", "EREG/ERBB4 signaling"),
    ("macrophage", "neuroblast", "VCAN/EGFR signaling"),
    ("macrophage", "neuroblast", "THBS1/CD47 dont-eat-me"),
    ("macrophage", "neuroblast", "VEGFA/GPC1 angiogenesis"),
    ("macrophage", "neuroblast", "APOE/LDLR lipid metabolism"),
    ("neuroblast", "macrophage", "THBS1/ITGB1 adhesion"),
    ("macrophage", "endothelial", "VEGFA/KDR angiogenesis"),
    ("fibroblast", "neuroblast", "COL1A1 ECM signaling"),
]

PAPER_PATHWAYS = [
    "EGF signaling",
    "MAPK signaling",
    "PI3K-Akt signaling",
    "Calcium signaling",
    "Oxidative phosphorylation",
    "Axon guidance",
    "ECM-receptor interaction",
    "Focal adhesion",
    "Angiogenesis",
    "Complement and coagulation",
    "Antigen processing and presentation",
    "Cytokine-cytokine receptor interaction",
    "Chemokine signaling",
    "TNF signaling",
    "Apoptosis",
    "Cell cycle",
]

PROPORTION_CHANGES = {
    "macrophage": "significantly expanded after therapy (P=1.4e-3)",
    "Schwann cell": "expanded after therapy (P=2.1e-3)",
    "fibroblast": "modest expansion (P=0.088)",
    "B cell": "decreased after therapy (P=0.032)",
    "neuroblast": "slight decrease (P=0.064)",
    "T cell": "no significant change",
    "endothelial cell": "no significant change",
    "dendritic cell": "no significant change",
}


# ── Execute a single query ───────────────────────────────────
def execute_query(cmd_type, cmd_str, engine, llm, adata, viz,
                  cluster_key, plot_dir):
    payload = None
    answer = None
    plots = []

    if cmd_type == "info":
        caps = engine.detect_capabilities()
        return {"mode": "info", "capabilities": caps}, json.dumps(caps, indent=2), []

    elif cmd_type == "proportions":
        payload = engine.proportions()

    elif cmd_type == "semantic":
        txt = cmd_str.split(":", 1)[1].strip()
        payload = engine.query_semantic(txt, top_k=5, with_genes=True)

    elif cmd_type == "hybrid":
        txt = cmd_str.split(":", 1)[1].strip()
        payload = engine.query_hybrid(txt, top_k=5, lambda_sem=0.0, with_genes=True)

    elif cmd_type == "discover":
        txt = cmd_str.split(":", 1)[1].strip()
        payload = engine.discover(txt, top_k=5)

    elif cmd_type == "compare":
        txt = cmd_str.split(":", 1)[1].strip()
        genes = None
        if "|" in txt:
            txt, gene_str = txt.split("|", 1)
            genes = [g.strip() for g in gene_str.split(",") if g.strip()]
            txt = txt.strip()
        parts = txt.lower().split(" vs ")
        if len(parts) == 2:
            group_a, group_b = parts[0].strip(), parts[1].strip()
            caps = engine.detect_capabilities()
            if caps["has_conditions"]:
                for cv in caps["condition_values"]:
                    if cv.lower() == group_a: group_a = cv
                    if cv.lower() == group_b: group_b = cv
            payload = engine.compare(group_a, group_b, genes=genes)
        else:
            return {"error": f"Bad compare format: {txt}"}, "", []

    elif cmd_type == "interactions":
        txt = cmd_str.split(":", 1)[1].strip() if ":" in cmd_str else ""
        src, tgt = None, None
        if "->" in txt:
            parts = txt.split("->")
            src = parts[0].strip() or None
            tgt = parts[1].strip() if len(parts) > 1 else None
        elif txt:
            src = txt
        payload = engine.interactions(source=src, target=tgt)

    elif cmd_type == "pathway":
        txt = cmd_str.split(":", 1)[1].strip()
        if txt.lower() == "all":
            payload = engine.pathway()
        else:
            payload = engine.pathway(pathway_name=txt)

    elif cmd_type == "plot":
        if adata is None:
            print(f"  [SKIP] No h5ad loaded for: {cmd_str}")
            return None, None, []
        os.makedirs(plot_dir, exist_ok=True)
        subcmd = cmd_str.split(None, 1)
        sub = subcmd[0]
        args_str = subcmd[1].strip() if len(subcmd) > 1 else ""
        import matplotlib; matplotlib.use("Agg")
        try:
            if sub == "plot:umap":
                p = f"{plot_dir}/cell_umap.png"
                viz.plot_cell_umap(adata, cluster_key=cluster_key, save_path=p)
                viz.plt.close("all"); plots.append(p)
            elif sub == "plot:expr":
                gene = args_str
                p = f"{plot_dir}/expr_{gene}.png"
                viz.plot_gene_expression_umap(adata, gene=gene, save_path=p)
                viz.plt.close("all"); plots.append(p)
            elif sub == "plot:dotplot":
                genes = [g.strip() for g in args_str.split(",")]
                tag = "_".join(genes[:3])
                p = f"{plot_dir}/dotplot_{tag}.png"
                viz.plot_dotplot(adata, genes=genes, cluster_key=cluster_key, save_path=p)
                viz.plt.close("all"); plots.append(p)
            elif sub == "plot:grid":
                genes = [g.strip() for g in args_str.split(",")]
                p = f"{plot_dir}/grid_{genes[0]}.png"
                viz.plot_gene_expression_grid(adata, genes=genes, cluster_key=cluster_key, save_path=p)
                viz.plt.close("all"); plots.append(p)
        except Exception as e:
            print(f"  [PLOT ERROR] {e}")
        return None, None, plots

    if payload and "error" not in payload:
        mode = payload.get("mode", cmd_type)
        query = payload.get("query", cmd_str)
        prompt = build_prompt(mode, query, payload)
        try:
            answer = ask_llm(llm, SYSTEM_PROMPT, prompt)
        except Exception as e:
            answer = f"[LLM ERROR] {e}"
            print(f"  [LLM ERROR] {e}")

    return payload, answer, plots


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def full_evaluation(elisa_report_text, elisa_genes, elisa_interactions,
                    elisa_pathway_scores, elisa_proportions):
    scorecard = {}

    found_genes = PAPER_GENES.intersection(elisa_genes)
    gene_recall = len(found_genes) / len(PAPER_GENES) * 100 if PAPER_GENES else 0
    scorecard["gene_recall"] = f"{gene_recall:.1f}% ({len(found_genes)}/{len(PAPER_GENES)})"

    pathways_found = 0
    for pp in PAPER_PATHWAYS:
        pp_lower = pp.lower()
        for ep_key in elisa_pathway_scores:
            if pp_lower in ep_key.lower() or ep_key.lower() in pp_lower:
                pathways_found += 1; break
    pathway_coverage = pathways_found / len(PAPER_PATHWAYS) * 100 if PAPER_PATHWAYS else 0
    scorecard["pathway_coverage"] = f"{pathway_coverage:.1f}% ({pathways_found}/{len(PAPER_PATHWAYS)})"

    interactions_found = 0
    for src, tgt, _ in PAPER_INTERACTIONS:
        src_l, tgt_l = src.lower(), tgt.lower()
        for ei in elisa_interactions:
            ei_src, ei_tgt = ei.get("source", "").lower(), ei.get("target", "").lower()
            if (src_l in ei_src or ei_src in src_l) and (tgt_l in ei_tgt or ei_tgt in tgt_l):
                interactions_found += 1; break
    interaction_recall = interactions_found / len(PAPER_INTERACTIONS) * 100 if PAPER_INTERACTIONS else 0
    scorecard["interaction_recall"] = f"{interaction_recall:.1f}% ({interactions_found}/{len(PAPER_INTERACTIONS)})"

    has_proportions = bool(elisa_proportions)
    scorecard["proportions_available"] = "Yes" if has_proportions else "No"
    scorecard["report_words"] = len(elisa_report_text.split())

    themes = {
        "Neuroblast identity": ["PHOX2B", "ISL1", "neural crest", "neuroblast"],
        "ADRN-proliferating state": ["ADRN-proliferating", "MKI67", "TOP2A", "EZH2", "proliferating"],
        "ADRN-calcium state": ["ADRN-calcium", "calcium signaling", "synaptic", "CACNA"],
        "ADRN-dopaminergic state": ["ADRN-dopaminergic", "dopamine", "catecholamine", "SLC18A2"],
        "Interm-OXPHOS state": ["Interm-OXPHOS", "oxidative phosphorylation", "ribosomal"],
        "MES state": ["mesenchymal", "MES", "YAP1", "FN1", "VIM"],
        "AP-1 TFs": ["AP-1", "JUN", "FOS", "BACH", "FOSL2"],
        "EZH2 enhancer regulation": ["EZH2", "enhancer", "PRC2", "polycomb"],
        "MAZ/CTCF regulators": ["MAZ", "CTCF", "transcription factor", "regulatory"],
        "Macrophage subtypes": ["IL18", "VCAN", "CCL4", "C1QC", "SPP1", "F13A1", "HS3ST2"],
        "Macrophage expansion": ["macrophage expansion", "expanded after therapy", "pro-tumorigenic"],
        "HB-EGF/ERBB4 axis": ["HB-EGF", "HBEGF", "ERBB4", "paracrine", "ErbB"],
        "ERK signaling": ["ERK", "MAPK", "phosphorylation", "CRM197", "afatinib"],
        "Therapy-induced shifts": ["chemotherapy", "post-therapy", "PTX", "therapy-induced"],
        "NECTIN2 immune evasion": ["NECTIN2", "immune evasion", "immune checkpoint"],
        "THBS1/CD47 dont-eat-me": ["THBS1", "CD47", "dont eat me", "phagocytosis"],
        "Clinical outcome": ["survival", "prognosis", "clinical outcome", "event-free"],
        "ALK mutation effects": ["ALK mutation", "ALK mutated", "ALK wild type"],
        "MYCN amplification": ["MYCN amplification", "MYCN-amplified"],
        "Schwann expansion": ["Schwann", "PLP1", "CDH19", "SOX10"],
        "CODEX validation": ["CODEX", "spatial proteomics", "ERBB4"],
        "Mouse model validation": ["Xenium", "TH-MYCN", "mouse model"],
        "Colony formation": ["colony formation", "co-culture", "THP-1"],
        "Adrenal cortex tissue": ["adrenal cortex", "CYP11A1", "steroidogenesis"],
    }
    themes_found = 0
    for theme_name, keywords in themes.items():
        for kw in keywords:
            if kw.lower() in elisa_report_text.lower():
                themes_found += 1; break
    theme_coverage = themes_found / len(themes) * 100 if themes else 0
    scorecard["theme_coverage"] = f"{theme_coverage:.1f}% ({themes_found}/{len(themes)})"

    composite = (gene_recall * 0.30 + pathway_coverage * 0.20 +
                 interaction_recall * 0.15 + theme_coverage * 0.25 +
                 (10 if has_proportions else 0))
    scorecard["composite_score"] = f"{composite:.1f}%"

    return {
        "scorecard": scorecard,
        "composite_score": round(composite, 1),
        "details": {
            "paper_genes": sorted(PAPER_GENES),
            "found_genes": sorted(found_genes),
            "missing_genes": sorted(PAPER_GENES - found_genes),
            "paper_pathways": PAPER_PATHWAYS,
            "paper_interactions": [(s, t, d) for s, t, d in PAPER_INTERACTIONS],
        }
    }


# ── Main runner ──────────────────────────────────────────────
def main():
    args = setup_args()
    out_dir = args.out_dir
    plot_dir = os.path.join(out_dir, "elisa_plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    engine, llm, adata, report, viz, caps = init_elisa(args)
    queries = get_queries(skip_plots=args.skip_plots)
    session_log = []
    all_payloads = []
    all_interactions = []

    print(f"\n{'='*70}")
    print(f"ELISA PAPER REPLICATION: High-Risk Neuroblastoma")
    print(f"Paper: Yu et al. (2025) Nature Genetics 57, 1142-1154")
    print(f"RUNNING {len(queries)} QUERIES")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")

    for i, (cmd_type, cmd_str) in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {cmd_str}")
        t0 = time.time()
        payload, answer, plots = execute_query(
            cmd_type, cmd_str, engine, llm, adata, viz, args.cluster_key, plot_dir)
        elapsed = time.time() - t0

        if payload is None and cmd_type == "plot":
            print(f"  → {len(plots)} plots ({elapsed:.1f}s)")
            if plots and report.entries:
                report.entries[-1]["plots"].extend(plots)
            continue

        if payload and "error" in payload:
            print(f"  [ERROR] {payload['error']}")
            continue

        if payload:
            mode = payload.get("mode", cmd_type)
            if mode == "interactions" and "interactions" in payload:
                all_interactions.extend(payload["interactions"])
            report.add_entry(entry_type=cmd_type, query=payload.get("query", cmd_str),
                             payload=payload, answer=answer or "", plots=plots)
            all_payloads.append(payload)
            session_log.append({"index": i + 1, "command": cmd_str, "type": cmd_type,
                                "mode": mode, "query": payload.get("query", ""),
                                "answer": answer[:500] if answer else "",
                                "elapsed": round(elapsed, 2), "n_plots": len(plots)})
            if answer:
                print(f"  → {mode} | {elapsed:.1f}s")
                print(f"  {answer[:150]}...")
            else:
                print(f"  → {mode} | {elapsed:.1f}s (no LLM answer)")
        print()

    # ── Generate report ──
    print(f"\n{'='*70}\nGENERATING REPORT\n{'='*70}")
    def llm_fn(prompt): return ask_llm(llm, SYSTEM_PROMPT, prompt)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = os.path.join(out_dir, f"elisa_report_Neuroblastoma_{ts}.md")
    report.generate_markdown(md_path, llm_func=llm_fn)
    docx_path = md_path.replace(".md", ".docx")
    report.generate_docx(docx_path, llm_func=llm_fn)

    log_path = os.path.join(out_dir, "session_log.json")
    with open(log_path, "w") as f: json.dump(session_log, f, indent=2, default=str)
    print(f"[SAVED] Session log: {log_path}")

    # ── Evaluation ──
    print(f"\n{'='*70}\nRUNNING EVALUATION\n{'='*70}")
    import re
    report_text = ""
    if os.path.exists(md_path):
        with open(md_path) as f: report_text = f.read()

    elisa_genes = set()
    for payload in all_payloads:
        for r in payload.get("results", []):
            for g in r.get("gene_evidence", []):
                if isinstance(g, dict): elisa_genes.add(g.get("gene", ""))
                elif isinstance(g, str): elisa_genes.add(g)
            for g in r.get("genes", []):
                if isinstance(g, dict): elisa_genes.add(g.get("gene", ""))
                elif isinstance(g, str): elisa_genes.add(g)
        for cid, cdata in payload.get("clusters", {}).items():
            if isinstance(cdata, dict):
                for g in cdata.get("genes", []):
                    if isinstance(g, dict): elisa_genes.add(g.get("gene", ""))
                    elif isinstance(g, str): elisa_genes.add(g)

    for payload in all_payloads:
        if payload.get("mode") in ("pathway_scoring", "pathway_query"):
            for pw_name, pw_data in payload.get("pathways", {}).items():
                for gs in pw_data.get("gene_set", []): elisa_genes.add(gs)
                for cl_data in pw_data.get("scores", []):
                    for tg in cl_data.get("top_genes", []):
                        if isinstance(tg, dict): elisa_genes.add(tg.get("gene", ""))
                        elif isinstance(tg, str): elisa_genes.add(tg)
        if payload.get("mode") == "pathway_query":
            for g in payload.get("genes_in_pathway", []): elisa_genes.add(g)

    for entry in session_log:
        answer = entry.get("answer", "")
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,12}(?:-[A-Z0-9]+)?)\b', answer):
            elisa_genes.add(m.group(1))
    for m in re.finditer(r'\b([A-Z][A-Z0-9-]{1,15})\s*\(ENSG\d+', report_text):
        elisa_genes.add(m.group(1))

    elisa_genes.discard("")
    print(f"  Genes collected: {len(elisa_genes)}")

    pathway_scores = {}
    for payload in all_payloads:
        mode = payload.get("mode", "")
        if mode == "pathway_scoring": pathway_scores.update(payload.get("pathways", {}))
        elif mode == "pathway_query":
            pw_name = payload.get("pathway", "")
            if pw_name: pathway_scores[pw_name] = {"scores": payload.get("scores", []),
                                                     "genes_in_pathway": payload.get("genes_in_pathway", [])}
    print(f"  Pathways collected: {len(pathway_scores)}")
    print(f"  Interactions collected: {len(all_interactions)}")

    proportion_data = {}
    for payload in all_payloads:
        if payload.get("mode") == "proportions": proportion_data = payload; break
    print(f"  Proportion data: {'found' if proportion_data else 'missing'}")

    eval_result = full_evaluation(report_text, elisa_genes, all_interactions,
                                  pathway_scores, proportion_data)

    print(f"\n{'='*70}\nEVALUATION SCORECARD\n{'='*70}")
    for metric, score in eval_result["scorecard"].items():
        print(f"  {metric}: {score}")
    print(f"\n  ★ COMPOSITE SCORE: {eval_result['composite_score']}%")
    print(f"{'='*70}")

    eval_path = os.path.join(out_dir, "evaluation_scorecard.json")
    with open(eval_path, "w") as f: json.dump(eval_result, f, indent=2, default=str)
    print(f"[SAVED] Evaluation: {eval_path}")

    print(f"\n{'='*70}\nALL OUTPUTS:")
    print(f"  Report (md):   {md_path}")
    print(f"  Report (docx): {docx_path}")
    print(f"  Session log:   {log_path}")
    print(f"  Evaluation:    {eval_path}")
    print(f"  Plots:         {plot_dir}/")
    n_plots = sum(len(e.get("plots", [])) for e in report.entries)
    print(f"  Total plots:   {n_plots}")
    print(f"  Analyses:      {len(report.entries)}")
    print(f"{'='*70}\nDONE")


if __name__ == "__main__":
    main()
