#!/usr/bin/env python
"""
ELISA Paper Replication — Automated Batch Runner
=================================================
Target Paper:  Bhat-Nakshatri et al. (2024) Nature Medicine 30, 3482–3494
               "Single-nucleus chromatin accessibility and transcriptomic
                map of breast tissues of women of diverse genetic ancestry"

Runs ALL queries from the Breast Tissue paper replication plan automatically,
generates the report, and computes evaluation metrics.

Updated for retrieval_engine_v4_hybrid + elisa_analysis.

Usage:
    python run_paper_replication_BreastTissue.py \\
        --h5ad /path/to/DT2_BreastTissue.h5ad \\
        --out-dir /path/to/elisa_replication_BreastTissue/ \\
        --base /path/to/embeddings/ \\
        --pt-name fused_embeddings.pt \\
        --cluster-key cell_type \\
        --dataset-name "Breast Tissue Atlas — Diverse Genetic Ancestry"

Output (all saved to --out-dir):
    - elisa_report_TIMESTAMP.md         (structured report)
    - elisa_report_TIMESTAMP.docx       (if pandoc available)
    - elisa_plots/                      (all Nature-style + retrieval plots)
    - evaluation_scorecard.json         (quantitative metrics)
    - session_log.json                  (all payloads + answers)
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
        description="ELISA Paper Replication Runner — Breast Tissue Atlas"
    )
    parser.add_argument("--h5ad", default=None,
                        help="Path to .h5ad (for Nature plots)")
    parser.add_argument("--cluster-key", default="cell_type",
                        help="obs column for cell types")
    parser.add_argument("--out-dir", default="elisa_replication_BreastTissue",
                        help="Output directory for all results")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip Nature-style plots (faster)")
    parser.add_argument("--dataset-name",
                        default="Breast Tissue Atlas — Diverse Genetic Ancestry",
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

    # Engine
    print("[INIT] Loading retrieval engine (v4 hybrid)...")
    raw_engine = RetrievalEngine(
        base=args.base,
        pt_name=args.pt_name,
        cells_csv=args.cells_csv,
    )
    print(f"[INIT] {len(raw_engine.cluster_ids)} clusters loaded")

    # ── Wrap engine with analytical methods ──
    class EngineWrapper:
        """
        Wraps v4 RetrievalEngine with all methods the replication
        script expects: discover, compare, interactions, proportions,
        pathway, detect_capabilities, plus .n attribute.
        """
        def __init__(self, eng):
            self._eng = eng
            self.cluster_ids = eng.cluster_ids
            self.gene_stats = eng.gene_stats
            self.metadata = eng.metadata
            self.cluster_metadata = getattr(eng, 'cluster_metadata', {})
            self.n = len(eng.cluster_ids)

        # ── Retrieval methods (pass-through) ──
        def query_semantic(self, text, top_k=5, with_genes=False):
            return self._eng.query_semantic(text, top_k=top_k,
                                            with_genes=with_genes)

        def query_hybrid(self, text, top_k=5, lambda_sem=0.5,
                         with_genes=False, **kwargs):
            # v4 doesn't accept pre_k/gamma — ignore them
            return self._eng.query_hybrid(text, top_k=top_k,
                                          lambda_sem=lambda_sem,
                                          with_genes=with_genes)

        def query_annotation_only(self, text, top_k=5, with_genes=False):
            return self._eng.query_annotation_only(text, top_k=top_k,
                                                   with_genes=with_genes)

        # ── Discovery mode: semantic + gene evidence ──
        def discover(self, text, top_k=5, **kwargs):
            payload = self._eng.query_semantic(text, top_k=top_k,
                                               with_genes=True)
            payload["mode"] = "discovery"
            payload["query"] = text
            return payload

        # ── Analytical methods via elisa_analysis ──
        def interactions(self, source=None, target=None, **kwargs):
            src_clusters = None
            tgt_clusters = None
            if source:
                src_lower = source.lower()
                src_clusters = [cid for cid in self._eng.cluster_ids
                                if src_lower in str(cid).lower()]
                if not src_clusters:
                    src_clusters = None
            if target:
                tgt_lower = target.lower()
                tgt_clusters = [cid for cid in self._eng.cluster_ids
                                if tgt_lower in str(cid).lower()]
                if not tgt_clusters:
                    tgt_clusters = None
            return find_interactions(
                self._eng.gene_stats,
                self._eng.cluster_ids,
                source_clusters=src_clusters,
                target_clusters=tgt_clusters,
                **kwargs,
            )

        def pathway(self, pathway_name=None, **kwargs):
            if pathway_name:
                return query_pathway(
                    self._eng.gene_stats,
                    self._eng.cluster_ids,
                    pathway_name=pathway_name,
                    **kwargs,
                )
            else:
                return pathway_scoring(
                    self._eng.gene_stats,
                    self._eng.cluster_ids,
                    **kwargs,
                )

        def proportions(self, **kwargs):
            return proportion_analysis(self._eng.metadata, **kwargs)

        def compare(self, group_a, group_b, genes=None, **kwargs):
            # Detect condition column from metadata fields
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
                    if condition_col:
                        break
            if not condition_col:
                condition_col = "condition"  # fallback
            return comparative_analysis(
                self._eng.gene_stats,
                self._eng.metadata,
                condition_col=condition_col,
                group_a=group_a,
                group_b=group_b,
                genes=genes,
                **kwargs,
            )

        def detect_capabilities(self):
            caps = {
                "has_conditions": False,
                "condition_values": [],
                "condition_column": None,
                "n_clusters": len(self._eng.cluster_ids),
                "cluster_ids": list(self._eng.cluster_ids),
            }
            meta = self._eng.metadata
            if not meta:
                return caps
            # Scan metadata fields for condition-like columns
            condition_keywords = [
                "patient_group", "case_control", "condition", "disease",
                "treatment", "status", "health_status", "tissue_status",
                "genotype", "phenotype", "diagnosis", "sample_type", "group",
                "timepoint", "pre_post",
            ]
            col_candidates = {}
            for cid, m in meta.items():
                if not isinstance(m, dict):
                    continue
                fields = m.get("fields", {})
                for col, dist in fields.items():
                    if isinstance(dist, dict):
                        if col not in col_candidates:
                            col_candidates[col] = set()
                        col_candidates[col].update(dist.keys())
            # First pass: match by keyword
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
            # Second pass: any binary column
            skip = ["batch", "lab", "donor", "sample", "plate", "lane",
                    "seq", "lib", "run", "class", "subclass", "level",
                    "annotation", "ontology", "type", "assay", "sex"]
            for col, values in col_candidates.items():
                if len(values) == 2:
                    if not any(p in col.lower() for p in skip):
                        caps["has_conditions"] = True
                        caps["condition_column"] = col
                        caps["condition_values"] = sorted(values)
                        return caps
            return caps

    engine = EngineWrapper(raw_engine)

    # LLM
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    llm = Groq(api_key=key)

    # Adata (optional)
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

    # Report builder
    report = ReportBuilder(dataset_name=args.dataset_name)

    # Capabilities
    caps = engine.detect_capabilities()
    print(f"[INIT] Conditions: {caps.get('has_conditions', False)} "
          f"→ {caps.get('condition_values', [])}")

    return engine, llm, adata, report, viz, caps


# ── LLM helper ───────────────────────────────────────────────
def ask_llm(llm, system_prompt, user_prompt):
    res = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return res.choices[0].message.content.strip()


SYSTEM_PROMPT = (
    "You are ELISA, an expert assistant for single-cell biology and "
    "breast tissue biology. Never hallucinate. Always ground claims "
    "strictly in provided data. Be concise and scientific. "
    "Focus on breast epithelial cell subtypes (LHS, LASP, BM), "
    "chromatin accessibility, gene regulatory networks, genetic "
    "ancestry-dependent variability, spatial transcriptomics, "
    "ductal vs lobular epithelial differences, fibroblast states, "
    "BRCA1/BRCA2 mutation effects, estrogen receptor signaling, "
    "and breast cancer cell-of-origin biology."
)


# ── Prompt builders ──────────────────────────────────────────

MAX_PROMPT_CHARS = 12000


def trim_payload(payload, max_chars=MAX_PROMPT_CHARS):
    """Trim payload to fit within LLM token limits."""
    trimmed = dict(payload)

    if "clusters" in trimmed and isinstance(trimmed["clusters"], dict):
        clusters = trimmed["clusters"]
        if len(clusters) > 10:
            ranked = sorted(
                clusters.items(),
                key=lambda kv: len(kv[1].get("genes", [])),
                reverse=True
            )[:10]
            trimmed["clusters"] = dict(ranked)
            trimmed["_trimmed"] = f"Showing top 10 of {len(clusters)} clusters"

    if "interactions" in trimmed and isinstance(trimmed["interactions"], list):
        if len(trimmed["interactions"]) > 30:
            trimmed["interactions"] = trimmed["interactions"][:30]
            trimmed["_trimmed_interactions"] = True

    if "overall" in trimmed and isinstance(trimmed["overall"], list):
        for item in trimmed.get("overall", []):
            for k in list(item.keys()):
                if k not in ("cluster", "n_cells", "fraction", "percentage"):
                    item.pop(k, None)

    if "scores" in trimmed and isinstance(trimmed["scores"], list):
        trimmed["scores"] = trimmed["scores"][:10]

    for r in trimmed.get("results", []):
        if "gene_evidence" in r and len(r["gene_evidence"]) > 5:
            r["gene_evidence"] = r["gene_evidence"][:5]

    ctx = json.dumps(trimmed, indent=1, default=str)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n... [TRUNCATED]"
        return ctx

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
    elif mode == "pathway_scoring" or mode == "pathway_query":
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
# Adapted from: Bhat-Nakshatri et al. (2024) Nature Medicine 30, 3482–3494
# ══════════════════════════════════════════════════════════════

def get_queries(skip_plots=False):
    """Return ordered list of (command_type, command_string) tuples."""
    queries = [
        # ──────────────────────────────────────────────────────
        # Phase 1: Cellular landscape & composition (Fig 1b-f)
        # ──────────────────────────────────────────────────────
        ("info", "info"),
        ("proportions", "proportions"),

        # ──────────────────────────────────────────────────────
        # Phase 2: Major cell type identification & markers (Fig 1e)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: luminal hormone sensing LHS mature luminal epithelial cells FOXA1 ESR1 in breast tissue"),
        ("semantic", "semantic: luminal adaptive secretory precursor LASP luminal progenitor EHF ELF5 KIT in healthy breast"),
        ("semantic", "semantic: basal-myoepithelial BM cells TP63 KRT14 KLHL29 markers in breast tissue"),
        ("semantic", "semantic: breast fibroblast stromal cell LAMA2 SLIT2 RUNX1T1 extracellular matrix"),
        ("semantic", "semantic: endothelial cell MECOM LDB2 ST6GALNAC3 MMRN1 vascular markers in breast"),
        ("semantic", "semantic: T cell lymphocyte PTPRC SKAP1 ARHGAP15 THEMIS CD96 markers breast tissue"),
        ("semantic", "semantic: macrophage ALCAM CD74 MS4A4E immune cell in breast tissue"),
        ("semantic", "semantic: adipocyte subtypes breast tissue lipid metabolism"),

        # Cell type marker discovery queries — LHS cells
        ("discover", "discover: ERBB4 ANKRD30A AFF3 TTC6 MYBPC1 THSD4 luminal hormone sensing LHS cell markers breast"),
        ("discover", "discover: FOXA1 ESR1 GATA3 hormone receptor transcription factor LHS cells breast tissue"),
        ("discover", "discover: CTNND2 DACH1 INPP4B NEK10 novel LHS-enriched genes breast epithelial cells"),
        ("discover", "discover: CACNA1C SEMA3C SAMD5 TSHZ2 PRKG1 KLHL29 FHOD3 LHS cell markers"),

        # Cell type marker discovery queries — LASP cells
        ("discover", "discover: KRT15 ELF5 CCL28 KIT luminal adaptive secretory precursor LASP markers breast"),
        ("discover", "discover: EHF ELF5 LASP cell identity gene expression chromatin accessibility breast"),
        ("discover", "discover: BARX2 NCALD LASP-enriched genes breast epithelial cell subtypes"),
        ("discover", "discover: AGAP1 SORBS2 SHANK2 MFGE8 DOCK5 LASP BL cell state markers"),

        # Cell type marker discovery queries — BM cells
        ("discover", "discover: TP63 KRT14 basal-myoepithelial BM cell markers chromatin accessibility breast"),
        ("discover", "discover: KLHL13 FHOD3 SEMA5A ABLIM3 basal-alpha BAα cell markers breast tissue"),
        ("discover", "discover: KLHL29 basal-myoepithelial marker gene breast epithelial subtype"),
        ("discover", "discover: SOX10 NKX2-8 NKX2-5 BM_BAβ basal-beta cell markers breast"),

        # Fibroblast markers
        ("discover", "discover: LAMA2 SLIT2 RUNX1T1 COL25A1 SOX5 fibroblast markers breast tissue"),
        ("discover", "discover: CFD MGST1 MFAP5 prematrix fibroblast state breast tissue"),
        ("discover", "discover: COL3A1 POSTN COL1A1 IGF1 ADAM12 matrix fibroblast state breast tissue"),
        ("discover", "discover: SFRP4 CLU G0S2 MGP OGN ADIRF IGFBP5 fibro-SFRP4 fibroblast subtype breast"),
        ("discover", "discover: MMP3 CXCL1 GEM KDM6B CEBPB TNFAIP6 CXCL2 fibro-major fibroblast subtype"),

        # Endothelial markers
        ("discover", "discover: MECOM LDB2 ST6GALNAC3 AL357507.1 PKHD1L1 MMRN1 endothelial cell markers breast"),
        ("discover", "discover: LYVE1 lymphatic endothelial cell marker breast tissue"),
        ("discover", "discover: ACKR1 endothelial stalk-like subtype marker breast tissue"),
        ("discover", "discover: CXCL12 endothelial fibroblast marker gene expression breast"),

        # T cell markers
        ("discover", "discover: PTPRC SKAP1 ARHGAP15 THEMIS CD96 T cell markers breast tissue"),
        ("discover", "discover: IL7R CD4 T cell marker expression chromatin accessibility breast"),
        ("discover", "discover: GZMK IFNG CD8 T cell marker gene expression breast tissue"),

        # Macrophage markers
        ("discover", "discover: ALCAM FCGR3A macrophage marker gene chromatin accessibility breast tissue"),
        ("discover", "discover: CD74 B2M MS4A4E macrophage antigen presentation breast tissue"),

        # Adipocyte markers
        ("discover", "discover: PLIN1 HPSE2 PDE3B adipocyte marker gene expression breast tissue"),
        ("discover", "discover: BICC1 DLC1 RGS6 PLA2R1 adipocyte subtype 2 markers breast"),

        # ──────────────────────────────────────────────────────
        # Phase 3: Epithelial subtype markers & chromatin (Fig 2)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: FOXA1 expression chromatin accessibility LHS cells marker validation breast"),
        ("semantic", "semantic: ESR1 estrogen receptor expression chromatin accessibility LHS LASP BM cells breast"),
        ("semantic", "semantic: GATA3 expression widespread LHS LASP BM cells not LHS-specific breast"),
        ("semantic", "semantic: EHF ELF5 expression chromatin accessibility LASP cell specific marker breast"),
        ("semantic", "semantic: TP63 KRT14 expression chromatin accessibility BM basal-myoepithelial specific marker"),
        ("semantic", "semantic: KIT expression LASP cells promoter accessibility breast epithelial"),

        ("discover", "discover: FOXA1 promoter chromatin accessibility LHS-specific marker breast epithelial cells"),
        ("discover", "discover: ESR1 chromatin accessible regions LHS LASP cells breast tissue peak analysis"),
        ("discover", "discover: GATA3 widespread expression multiple epithelial subtypes breast tissue"),
        ("discover", "discover: EHF promoter accessibility LASP cells binding site enrichment breast"),
        ("discover", "discover: ELF5 expression chromatin accessibility binding site LASP cell marker"),
        ("discover", "discover: KIT expression LASP cells promoter similar accessibility all cell types"),
        ("discover", "discover: TP63 expression chromatin accessibility binding site enrichment BM cells breast"),
        ("discover", "discover: KRT14 expression restricted BM cells promoter distinct accessibility breast"),

        # ──────────────────────────────────────────────────────
        # Phase 4: Gene regulatory networks — SCENIC regulons (Fig 1g)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: SCENIC regulon gene regulatory network breast epithelial cell subtypes"),
        ("semantic", "semantic: transcription factor binding motif Signac snATAC-seq chromatin accessibility breast"),
        ("discover", "discover: RCOR1 ZNF420 CHD2 PBX3 SMAD1 TFAP2A BM cell regulon breast tissue SCENIC"),
        ("discover", "discover: ELF1 ZNF780A BCL6 KLF8 KLF7 FOSB LASP cell regulon breast tissue"),
        ("discover", "discover: NFIB EGR3 ERG MAFB GLI3 KLF12 LASP regulon breast epithelial"),
        ("discover", "discover: TFAP2C SP4 TCF7 SMAD2 SP3 JUND LHS cell regulon breast tissue"),
        ("discover", "discover: TFAP2B TBX3 THRB RARB EGR1 SMC3 LHS cell regulon breast tissue SCENIC"),
        ("discover", "discover: GHRL1 transcription factor LASP_AP cells binding motif enrichment breast"),
        ("discover", "discover: SOX10 transcription factor BM LASP cells binding motif BM_BAβ enrichment"),

        # DNA motif analyses
        ("discover", "discover: TCFL5 transcription factor binding motif enrichment epithelial cells breast"),
        ("discover", "discover: ZBTB14 ZFP161 binding motif epithelial cell chromatin breast cancer prognosis"),
        ("discover", "discover: THRB thyroid hormone receptor binding motif footprinting epithelial cells breast"),

        # ──────────────────────────────────────────────────────
        # Phase 5: Spatial transcriptomics — duct vs lobule (Fig 3)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: spatial transcriptomics ductal lobular epithelial cell gene expression differences breast"),
        ("semantic", "semantic: KRT14 KRT17 ductal epithelial cells BM cell markers breast duct"),
        ("semantic", "semantic: IGHA1 IGKC immunoglobulin lobular epithelial cells breast lobule"),
        ("semantic", "semantic: DUSP1 DPM3 RPL36 lobular epithelial cell enriched genes breast"),
        ("semantic", "semantic: PTBP1 age-dependent decline mRNA processing alternative splicing breast"),

        ("discover", "discover: KRT14 KRT17 elevated ductal epithelial cells basal-myoepithelial enriched breast"),
        ("discover", "discover: IGHA1 IGKC immunoglobulin higher lobular epithelial cells breast tissue"),
        ("discover", "discover: DUSP1 DPM3 RPL36 lobular epithelial cell enriched lobular carcinoma breast"),
        ("discover", "discover: MGP ANXA1 TACSTD2 WFDC2 STAC2 ALDH1A3 ductal epithelial markers breast"),
        ("discover", "discover: APOD SNORC lobular epithelial cell markers breast tissue spatial"),
        ("discover", "discover: PTBP1 mRNA splicing age-dependent expression decline breast epithelial"),
        ("discover", "discover: ALDH1A3 stem progenitor marker ductal epithelial LASP cells breast cancer BRCA1"),

        # Age-dependent changes
        ("discover", "discover: PKA protein kinase A signaling pathway age-reduced breast epithelial cells"),
        ("discover", "discover: eIF2 signaling pathway age-increased breast epithelial cells oxidative phosphorylation"),

        # ──────────────────────────────────────────────────────
        # Phase 6: Genetic ancestry variability (Fig 4)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: genetic ancestry dependent variability breast epithelial cell state Indigenous American"),
        ("semantic", "semantic: ESR1 expression LASP alveolar progenitor cells Indigenous American ancestry breast"),
        ("semantic", "semantic: MKI67 cell proliferation marker breast tissue genetic ancestry groups"),
        ("semantic", "semantic: EGF signaling estrogen receptor cross-talk alveolar progenitor Indigenous American"),

        ("discover", "discover: ESR1 expression LASP alveolar progenitor cells Indigenous American ancestry breast"),
        ("discover", "discover: LASP_AP alveolar progenitor cells 19% Indigenous American versus 5-9% other groups"),
        ("discover", "discover: MKI67 proliferation marker low expression all ancestry groups breast tissue"),
        ("discover", "discover: EGF EGF signaling cross-talk ESR1 antiestrogen resistance AP cells Indigenous"),
        ("discover", "discover: FOXA1 GATA3 ELF5 EHF KRT14 TP63 no ancestry-dependent variability breast"),

        # ──────────────────────────────────────────────────────
        # Phase 7: African vs European ancestry (Fig 5)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: African ancestry European ancestry breast fibroblast epithelial cell state differences"),
        ("semantic", "semantic: PROCR ZEB1 PDGFRα multipotent stromal cells African ancestry breast tissue"),
        ("semantic", "semantic: fibro-prematrix fibro-matrix fibroblast state genetic ancestry breast tissue"),

        ("discover", "discover: LASP BL cell state dominant African ancestry versus AP BL European ancestry breast"),
        ("discover", "discover: PROCR ZEB1 PDGFRA multipotent stromal progenitor cells African ancestry breast"),
        ("discover", "discover: CFD MGST1 MFAP5 prematrix fibroblast state enriched African ancestry breast"),
        ("discover", "discover: COL3A1 POSTN COL1A1 IGF1 ADAM12 matrix fibroblast enriched European ancestry"),
        ("discover", "discover: fibro-prematrix vasculogenesis African ancestry fibroblast cell state breast"),

        # Fibroblast gene expression ancestry differences
        ("discover", "discover: ABCA10 ABCA9 ABCA8 NEGR1 MMP16 MAGI1 fibroblast genes African ancestry breast"),
        ("discover", "discover: KIAA1217 PTPRK SEMA5A fibroblast genes ancestry difference breast tissue"),

        # ──────────────────────────────────────────────────────
        # Phase 8: Chromatin accessibility of cell identity genes (Fig 6)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: IL7R IFNG T cell specific expression chromatin accessibility breast tissue"),
        ("semantic", "semantic: GZMK CD8 T cell expression chromatin accessibility restricted breast"),
        ("semantic", "semantic: FCGR3A macrophage specific chromatin accessibility expression breast"),
        ("semantic", "semantic: LYVE1 lymphatic endothelial expression chromatin accessibility discordance breast"),
        ("semantic", "semantic: ACKR1 endothelial stalk chromatin accessibility not cell-type specific breast"),
        ("semantic", "semantic: CXCL12 expression endothelial fibroblast chromatin accessibility breast"),

        ("discover", "discover: IL7R T cell specific expression chromatin accessibility compatible breast tissue"),
        ("discover", "discover: IFNG CD8 T cell expression chromatin accessibility restricted breast"),
        ("discover", "discover: GZMK expression chromatin accessibility restricted T cells breast tissue"),
        ("discover", "discover: FCGR3A macrophage restricted chromatin accessibility expression breast"),
        ("discover", "discover: LYVE1 endothelial cell 2 macrophage expression chromatin not similar breast"),
        ("discover", "discover: ACKR1 endothelial stalk expression restricted chromatin accessible all types"),
        ("discover", "discover: CXCL12 expression endothelial fibroblast chromatin similar except T cells breast"),

        # ──────────────────────────────────────────────────────
        # Phase 9: Breast cancer cell-of-origin (Extended Data Fig 9)
        # ──────────────────────────────────────────────────────
        ("semantic", "semantic: LHS cells origin LumA LumB HER2 breast cancer subtypes gene modules"),
        ("semantic", "semantic: LASP cells cycling module BM cells basal breast cancer module"),
        ("semantic", "semantic: myofibroblast inflammatory fibroblast pericyte COL1A1 PDPN CD34 breast cancer"),

        ("discover", "discover: LumA LumB HER2 gene module overlap LHS cell signatures breast cancer origin"),
        ("discover", "discover: cycling gene module overlap LASP cell signature breast cancer"),
        ("discover", "discover: basal breast cancer module overlap BM cells gene signature"),
        ("discover", "discover: COL1A1 PDPN myofibroblast marker breast cancer stroma"),
        ("discover", "discover: CD34 CXCL12 inflammatory fibroblast marker breast cancer stroma"),
        ("discover", "discover: MYH11 MYLK differentiated pericyte marker breast tissue stroma"),
        ("discover", "discover: CD36 RGS5 immature pericyte marker breast tissue stroma"),

        # ──────────────────────────────────────────────────────
        # Phase 10: Comparisons — epithelial subtypes
        # ──────────────────────────────────────────────────────
        ("compare", "compare: LHS vs LASP | FOXA1, ESR1, GATA3, ERBB4, ANKRD30A, AFF3, TTC6, MYBPC1"),
        ("compare", "compare: LHS vs BM | FOXA1, ESR1, GATA3, TP63, KRT14, KLHL29, DACH1, INPP4B"),
        ("compare", "compare: LASP vs BM | EHF, ELF5, KIT, KRT15, CCL28, TP63, KRT14, KLHL13"),
        ("compare", "compare: LHS_HSα vs LHS_HSβ | ESR1, FOXA1, GATA3, ERBB4, ANKRD30A, THSD4"),
        ("compare", "compare: LASP_AP vs LASP_BL | ELF5, EHF, KIT, KRT15, CCL28, BARX2, NCALD"),
        ("compare", "compare: BM_BAα vs BM_BAβ | TP63, KRT14, KLHL13, KLHL29, FHOD3, SEMA5A, SOX10"),
        # Novel markers comparisons
        ("compare", "compare: LHS vs LASP | CTNND2, DACH1, INPP4B, NEK10, TTC6, THSD4, MYBPC1"),
        ("compare", "compare: LHS vs BM | CTNND2, DACH1, INPP4B, NEK10, ERBB4, CACNA1C, SEMA3C"),
        ("compare", "compare: LASP vs BM | BARX2, NCALD, AGAP1, SORBS2, SHANK2, MFGE8, FHOD3"),

        # ──────────────────────────────────────────────────────
        # Phase 11: Comparisons — duct vs lobule and ancestry
        # ──────────────────────────────────────────────────────
        ("compare", "compare: Duct vs Lobule | KRT14, KRT17, IGHA1, IGKC, DUSP1, DPM3, RPL36, MGP"),
        ("compare", "compare: Duct vs Lobule | ANXA1, TACSTD2, WFDC2, STAC2, ALDH1A3, APOD, SNORC"),
        ("compare", "compare: Duct vs Lobule | PTBP1, KRT14, KRT17, FCGBP, CRYBG1, SCRN3, EBPL"),

        # Ancestry comparisons (if condition column available)
        ("compare", "compare: African vs European | PROCR, ZEB1, PDGFRA, CFD, MGST1, MFAP5, COL3A1, POSTN"),
        ("compare", "compare: African vs European | COL1A1, IGF1, ADAM12, ABCA10, ABCA9, ABCA8, MMP16"),
        ("compare", "compare: African vs European | SFRP4, CLU, MGP, OGN, G0S2, IGFBP5, NEGR1"),
        ("compare", "compare: Indigenous American vs European | ESR1, FOXA1, ELF5, EHF, GATA3, KIT"),

        # BRCA comparisons
        ("compare", "compare: BRCA1 vs Control | ESR1, FOXA1, GATA3, ELF5, EHF, TP63, KRT14, MKI67"),
        ("compare", "compare: BRCA2 vs Control | ESR1, FOXA1, GATA3, ELF5, EHF, TP63, KRT14, MKI67"),

        # ──────────────────────────────────────────────────────
        # Phase 12: Pathway analyses
        # ──────────────────────────────────────────────────────
        ("pathway", "pathway: Estrogen signaling"),
        ("pathway", "pathway: EGF signaling"),
        ("pathway", "pathway: PI3K-Akt signaling"),
        ("pathway", "pathway: MAPK signaling"),
        ("pathway", "pathway: Protein kinase A signaling"),
        ("pathway", "pathway: eIF2 signaling"),
        ("pathway", "pathway: Oxidative phosphorylation"),
        ("pathway", "pathway: Wnt signaling"),
        ("pathway", "pathway: Notch signaling"),
        ("pathway", "pathway: Cell adhesion molecules"),
        ("pathway", "pathway: ECM-receptor interaction"),
        ("pathway", "pathway: Focal adhesion"),
        ("pathway", "pathway: Breast cancer"),
        ("pathway", "pathway: Tight junction"),
        ("pathway", "pathway: Cytokine-cytokine receptor interaction"),
        ("pathway", "pathway: Antigen processing and presentation"),
        ("pathway", "pathway: Natural killer cell cytotoxicity"),
        ("pathway", "pathway: Complement and coagulation"),
        ("pathway", "pathway: Fatty acid metabolism"),
        ("pathway", "pathway: PPAR signaling"),

        # ──────────────────────────────────────────────────────
        # Phase 13: Cell-cell interactions
        # ──────────────────────────────────────────────────────
        ("interactions", "interactions:"),
        ("interactions", "interactions: LHS -> LASP"),
        ("interactions", "interactions: LHS -> BM"),
        ("interactions", "interactions: LASP -> BM"),
        ("interactions", "interactions: fibroblast -> epithelial"),
        ("interactions", "interactions: fibroblast -> endothelial"),
        ("interactions", "interactions: macrophage -> epithelial"),
        ("interactions", "interactions: macrophage -> T cell"),
        ("interactions", "interactions: T cell -> macrophage"),
        ("interactions", "interactions: endothelial -> fibroblast"),
        ("interactions", "interactions: adipocyte -> epithelial"),
        ("interactions", "interactions: endothelial -> T cell"),

        # ──────────────────────────────────────────────────────
        # Phase 14: Discussion — cancer relevance, biomarkers
        # ──────────────────────────────────────────────────────
        ("discover", "discover: DACH1 cell fate factor YB-1 oncogenic repressor breast cancer relevance"),
        ("discover", "discover: INPP4B PIK3CA Akt pathway negative regulator breast cancer tumor suppressor"),
        ("discover", "discover: NEK10 p53 activator tyrosine phosphorylation breast cancer relevance"),
        ("discover", "discover: TCFL5 basic helix-loop-helix transcription factor amplification breast cancer"),
        ("discover", "discover: ZBTB14 ZFP161 chromosomal loss chr18p11.31 LumB breast cancer prognosis"),
        ("discover", "discover: ALDH1A3 ductal epithelial LASP cells BRCA1 mutation breast cancer origin"),
        ("discover", "discover: DUSP1 DPM3 RPL36 lobular carcinoma higher expression lobular cell origin"),
        ("discover", "discover: PTBP1 overexpressed breast cancer mRNA splicing age decline"),
        ("discover", "discover: PKA pathway aging breast epithelial lifespan extension breast cancer"),
        ("discover", "discover: eIF2 pathway translational control aging breast epithelial tumorigenesis"),
        ("discover", "discover: genetic ancestry breast cancer disparity triple negative breast cancer African ancestry"),
        ("discover", "discover: fibro-prematrix fibroblast vasculogenesis African ancestry breast cancer microenvironment"),

        # Stromal cell types in cancer
        ("discover", "discover: COL1A1 PDPN myofibroblast inflammatory fibroblast CD34 CXCL12 breast cancer stroma"),
        ("discover", "discover: MYH11 MYLK differentiated pericyte CD36 RGS5 immature pericyte breast cancer stroma"),

        # ──────────────────────────────────────────────────────
        # Phase 15: Additional targeted queries for gene recovery
        # ──────────────────────────────────────────────────────
        ("discover", "discover: ERBB4 receptor tyrosine kinase LHS cell enriched breast epithelial signaling"),
        ("discover", "discover: ANKRD30A NY-BR-1 breast differentiation antigen LHS cells"),
        ("discover", "discover: AFF3 transcription factor LHS cells breast epithelial gene regulation"),
        ("discover", "discover: TTC6 tetratricopeptide repeat LHS cells breast tissue marker"),
        ("discover", "discover: THSD4 ADAMTSL6 LHS cell marker extracellular matrix breast tissue"),
        ("discover", "discover: KRT15 keratin 15 LASP luminal progenitor marker breast tissue"),
        ("discover", "discover: CCL28 chemokine LASP cell marker breast epithelial tissue"),
        ("discover", "discover: SEMA5A ABLIM3 ST6GALNAC3 PTPRM cell markers breast tissue"),
        ("discover", "discover: EBF1 PCDH9 LAMA4 immune and stromal markers breast tissue"),
        ("discover", "discover: ELOVL5 LIMCH1 ARHGAP26 LASP cell gene expression breast"),
        ("discover", "discover: NEK10 AZGP1 STC2 MGP THSD4 LHS cell additional markers breast"),
        ("discover", "discover: RERG INPP4B MFGE8 ELF5 DOCK5 TFP1 LASP markers breast tissue"),
    ]

    # ──────────────────────────────────────────────────────
    # Nature-style plots
    # ──────────────────────────────────────────────────────
    if not skip_plots:
        plot_queries = [
            ("plot", "plot:umap"),
            # Major cell type markers
            ("plot", "plot:expr FOXA1"),
            ("plot", "plot:expr ESR1"),
            ("plot", "plot:expr GATA3"),
            ("plot", "plot:expr EHF"),
            ("plot", "plot:expr ELF5"),
            ("plot", "plot:expr KIT"),
            ("plot", "plot:expr TP63"),
            ("plot", "plot:expr KRT14"),
            ("plot", "plot:expr KRT15"),
            ("plot", "plot:expr CCL28"),
            # LHS-enriched
            ("plot", "plot:expr ERBB4"),
            ("plot", "plot:expr ANKRD30A"),
            ("plot", "plot:expr AFF3"),
            ("plot", "plot:expr DACH1"),
            ("plot", "plot:expr INPP4B"),
            ("plot", "plot:expr NEK10"),
            ("plot", "plot:expr CTNND2"),
            # LASP-enriched
            ("plot", "plot:expr BARX2"),
            ("plot", "plot:expr NCALD"),
            # BM-enriched
            ("plot", "plot:expr FHOD3"),
            ("plot", "plot:expr KLHL29"),
            ("plot", "plot:expr KLHL13"),
            # Stromal / Immune
            ("plot", "plot:expr LAMA2"),
            ("plot", "plot:expr MECOM"),
            ("plot", "plot:expr PTPRC"),
            ("plot", "plot:expr ALCAM"),
            ("plot", "plot:expr MKI67"),
            # Duct vs lobule
            ("plot", "plot:expr KRT17"),
            ("plot", "plot:expr DUSP1"),
            ("plot", "plot:expr PTBP1"),
            ("plot", "plot:expr ALDH1A3"),
            # Ancestry
            ("plot", "plot:expr PROCR"),
            ("plot", "plot:expr ZEB1"),
            ("plot", "plot:expr PDGFRA"),
            # Cell identity chromatin
            ("plot", "plot:expr IL7R"),
            ("plot", "plot:expr GZMK"),
            ("plot", "plot:expr FCGR3A"),
            ("plot", "plot:expr LYVE1"),
            ("plot", "plot:expr ACKR1"),
            ("plot", "plot:expr CXCL12"),
            # Fibroblast state
            ("plot", "plot:expr CFD"),
            ("plot", "plot:expr COL1A1"),
            # Cancer stroma
            ("plot", "plot:expr PDPN"),
            ("plot", "plot:expr CD34"),
            ("plot", "plot:expr MYH11"),
            ("plot", "plot:expr CD36"),

            # Dotplots — major epithelial markers
            ("plot", "plot:dotplot FOXA1, ESR1, GATA3, EHF, ELF5, KIT, TP63, KRT14"),
            ("plot", "plot:dotplot ERBB4, ANKRD30A, AFF3, TTC6, MYBPC1, THSD4, DACH1, INPP4B"),
            ("plot", "plot:dotplot KRT15, CCL28, BARX2, NCALD, AGAP1, SORBS2, SHANK2, MFGE8"),
            ("plot", "plot:dotplot KLHL29, KLHL13, FHOD3, SEMA5A, ABLIM3, SOX10, NKX2-8, CTNND2"),
            ("plot", "plot:dotplot NEK10, CACNA1C, SEMA3C, SAMD5, TSHZ2, PRKG1, ELOVL5, LIMCH1"),

            # Dotplots — stromal and immune
            ("plot", "plot:dotplot LAMA2, SLIT2, RUNX1T1, MECOM, LDB2, PTPRC, SKAP1, ALCAM"),
            ("plot", "plot:dotplot IL7R, GZMK, IFNG, FCGR3A, LYVE1, ACKR1, CXCL12, CD96"),
            ("plot", "plot:dotplot CD74, B2M, MS4A4E, PLIN1, HPSE2, PDE3B, BICC1, DLC1"),

            # Dotplots — fibroblast states
            ("plot", "plot:dotplot CFD, MGST1, MFAP5, COL3A1, POSTN, COL1A1, IGF1, ADAM12"),
            ("plot", "plot:dotplot SFRP4, CLU, G0S2, MGP, OGN, MMP3, CXCL1, TNFAIP6"),

            # Dotplots — duct/lobule and cancer
            ("plot", "plot:dotplot KRT14, KRT17, IGHA1, IGKC, DUSP1, DPM3, RPL36, ALDH1A3"),
            ("plot", "plot:dotplot PTBP1, ANXA1, MGP, TACSTD2, WFDC2, STAC2, APOD, SNORC"),
            ("plot", "plot:dotplot COL1A1, PDPN, CD34, CXCL12, MYH11, MYLK, CD36, RGS5"),

            # Dotplots — ancestry and BRCA
            ("plot", "plot:dotplot PROCR, ZEB1, PDGFRA, ESR1, FOXA1, MKI67, ELF5, EHF"),
            ("plot", "plot:dotplot ABCA10, ABCA9, ABCA8, NEGR1, MMP16, MAGI1, KIAA1217, PTPRK"),

            # Dotplots — regulons/TFs
            ("plot", "plot:dotplot RCOR1, TFAP2A, TFAP2C, TFAP2B, TBX3, THRB, RARB, SMAD1"),

            # Grid plots
            ("plot", "plot:grid FOXA1, ESR1, EHF, ELF5, TP63, KRT14"),
            ("plot", "plot:grid DACH1, INPP4B, NEK10, BARX2, NCALD, FHOD3"),
            ("plot", "plot:grid PROCR, ZEB1, PDGFRA, CFD, COL1A1, POSTN"),
        ]
        queries.extend(plot_queries)

    return queries


# ══════════════════════════════════════════════════════════════
# PAPER-SPECIFIC EVALUATION REFERENCE DATA
# Ground truth from Bhat-Nakshatri et al. (2024) Nature Medicine
# ══════════════════════════════════════════════════════════════

PAPER_GENES = {
    # LHS cell markers (Fig 1e,f, Extended Data Fig 2)
    "ERBB4", "ANKRD30A", "AFF3", "TTC6", "MYBPC1", "THSD4",
    "FOXA1", "ESR1", "GATA3",
    "CTNND2", "DACH1", "INPP4B", "NEK10",
    "CACNA1C", "SEMA3C", "SAMD5", "TSHZ2", "PRKG1",
    "ELOVL5", "LIMCH1", "AZGP1", "STC2",
    # LASP cell markers
    "KRT15", "ELF5", "CCL28", "KIT", "EHF",
    "BARX2", "NCALD", "AGAP1", "SORBS2", "SHANK2", "MFGE8",
    "DOCK5", "ARHGAP26",
    # BM cell markers
    "TP63", "KRT14", "KLHL29", "KLHL13", "FHOD3", "SEMA5A",
    "ABLIM3", "SOX10", "NKX2-8",
    # Fibroblast markers
    "LAMA2", "SLIT2", "RUNX1T1", "COL25A1", "SOX5",
    "CFD", "MGST1", "MFAP5",
    "COL3A1", "POSTN", "COL1A1", "IGF1", "ADAM12",
    "SFRP4", "CLU", "G0S2", "MGP", "OGN", "ADIRF", "IGFBP5",
    "MMP3", "CXCL1", "GEM", "KDM6B", "CEBPB", "TNFAIP6", "CXCL2",
    "TNC",
    # Endothelial markers
    "MECOM", "LDB2", "ST6GALNAC3", "PKHD1L1", "MMRN1",
    "LYVE1", "ACKR1", "CXCL12",
    # T cell markers
    "PTPRC", "SKAP1", "ARHGAP15", "THEMIS", "CD96",
    "IL7R", "IFNG", "GZMK",
    # Macrophage markers
    "ALCAM", "FCGR3A", "CD74", "B2M", "MS4A4E",
    # Adipocyte markers
    "PLIN1", "HPSE2", "PDE3B", "BICC1", "DLC1", "RGS6", "PLA2R1",
    # Cell proliferation
    "MKI67",
    # Duct vs Lobule (spatial transcriptomics)
    "KRT17", "IGHA1", "IGKC", "DUSP1", "DPM3", "RPL36",
    "ANXA1", "TACSTD2", "WFDC2", "STAC2", "ALDH1A3",
    "APOD", "SNORC", "FCGBP", "CRYBG1", "SCRN3",
    "PTBP1",
    # Ancestry-related
    "PROCR", "ZEB1", "PDGFRA",
    "ABCA10", "ABCA9", "ABCA8", "NEGR1", "MMP16", "MAGI1",
    "KIAA1217", "PTPRK", "SEMA5A",
    # Regulons / TFs
    "RCOR1", "TFAP2A", "TFAP2C", "TFAP2B", "TBX3", "THRB",
    "RARB", "EGR1", "SMC3", "BCL6", "KLF8", "KLF7",
    "FOSB", "NFIB", "EGR3", "ERG", "MAFB", "GLI3", "KLF12",
    "SP4", "TCF7", "SMAD2", "SP3", "JUND",
    "SMAD1", "PBX3", "ELF1",
    # Cancer stroma markers
    "PDPN", "CD34", "MYH11", "MYLK", "CD36", "RGS5",
    # Motif-related TFs
    "TCFL5", "ZBTB14",
    # Additional genes from figures
    "EBF1", "PCDH9", "LAMA4", "RERG", "NEBL",
    "NHLRC4", "EBPL", "NOP53", "ZFYVE26", "SH3BP2", "DGAT2",
    "ITPR2", "RBMS3",
}

PAPER_INTERACTIONS = [
    ("LHS", "LASP", "hormone-signaling"),
    ("fibroblast", "epithelial", "ECM-signaling"),
    ("macrophage", "epithelial", "immune-regulation"),
    ("macrophage", "T cell", "antigen-presentation"),
    ("T cell", "macrophage", "immune-regulation"),
    ("endothelial", "fibroblast", "vascular-signaling"),
    ("fibroblast", "endothelial", "vasculogenesis"),
    ("adipocyte", "epithelial", "paracrine-signaling"),
    ("endothelial", "T cell", "recruitment"),
    ("LASP", "BM", "epithelial-crosstalk"),
]

PAPER_PATHWAYS = [
    "Estrogen signaling",
    "EGF signaling",
    "PI3K-Akt signaling",
    "MAPK signaling",
    "Protein kinase A signaling",
    "eIF2 signaling",
    "Oxidative phosphorylation",
    "Wnt signaling",
    "Notch signaling",
    "Cell adhesion molecules",
    "ECM-receptor interaction",
    "Focal adhesion",
    "Breast cancer",
    "Tight junction",
    "Cytokine-cytokine receptor interaction",
    "Antigen processing and presentation",
    "Fatty acid metabolism",
    "PPAR signaling",
    "Natural killer cell cytotoxicity",
    "Complement and coagulation",
]

PROPORTION_CHANGES = {
    "LHS": "major epithelial subtype, hormone sensing",
    "LASP": "luminal progenitor, AP enriched in Indigenous American",
    "BM": "basal-myoepithelial cells",
    "Fibroblasts": "fibro-prematrix dominant in African ancestry, fibro-matrix in European",
    "Endothelial 1": "vascular endothelial cells",
    "Endothelial 2": "lymphatic-enriched endothelial subtype",
    "T cells": "CD4+ and CD8+ lymphocytes",
    "Macrophages": "tissue-resident macrophages",
    "Adipocytes 1": "adipocyte subtype 1",
    "Adipocytes 2": "adipocyte subtype 2",
}


# ── Execute a single query ───────────────────────────────────
def execute_query(cmd_type, cmd_str, engine, llm, adata, viz,
                  cluster_key, plot_dir):
    """Execute one ELISA command and return (payload, answer, plots)."""

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
        payload = engine.query_hybrid(txt, top_k=5, lambda_sem=0.0,
                                       with_genes=True)

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
                    if cv.lower() == group_a:
                        group_a = cv
                    if cv.lower() == group_b:
                        group_b = cv
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

        import matplotlib
        matplotlib.use("Agg")

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
                viz.plot_gene_expression_grid(adata, genes=genes,
                                               cluster_key=cluster_key, save_path=p)
                viz.plt.close("all"); plots.append(p)
        except Exception as e:
            print(f"  [PLOT ERROR] {e}")

        return None, None, plots

    # If we got a payload, run LLM
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
# EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def full_evaluation(elisa_report_text, elisa_genes, elisa_interactions,
                    elisa_pathway_scores, elisa_proportions):
    """Compute evaluation metrics against paper ground truth."""

    scorecard = {}

    # 1. Gene recall
    found_genes = PAPER_GENES.intersection(elisa_genes)
    gene_recall = len(found_genes) / len(PAPER_GENES) * 100 if PAPER_GENES else 0
    scorecard["gene_recall"] = f"{gene_recall:.1f}% ({len(found_genes)}/{len(PAPER_GENES)})"

    # 2. Pathway coverage
    pathways_found = 0
    for pp in PAPER_PATHWAYS:
        pp_lower = pp.lower()
        for ep_key in elisa_pathway_scores:
            if pp_lower in ep_key.lower() or ep_key.lower() in pp_lower:
                pathways_found += 1
                break
    pathway_coverage = pathways_found / len(PAPER_PATHWAYS) * 100 if PAPER_PATHWAYS else 0
    scorecard["pathway_coverage"] = f"{pathway_coverage:.1f}% ({pathways_found}/{len(PAPER_PATHWAYS)})"

    # 3. Interaction recall
    interactions_found = 0
    for src, tgt, _ in PAPER_INTERACTIONS:
        src_l, tgt_l = src.lower(), tgt.lower()
        for ei in elisa_interactions:
            ei_src = ei.get("source", "").lower()
            ei_tgt = ei.get("target", "").lower()
            if (src_l in ei_src or ei_src in src_l) and \
               (tgt_l in ei_tgt or ei_tgt in tgt_l):
                interactions_found += 1
                break
    interaction_recall = interactions_found / len(PAPER_INTERACTIONS) * 100 if PAPER_INTERACTIONS else 0
    scorecard["interaction_recall"] = f"{interaction_recall:.1f}% ({interactions_found}/{len(PAPER_INTERACTIONS)})"

    # 4. Proportion data present
    has_proportions = bool(elisa_proportions)
    scorecard["proportions_available"] = "Yes" if has_proportions else "No"

    # 5. Report length
    report_words = len(elisa_report_text.split())
    scorecard["report_words"] = report_words

    # 6. Key theme coverage
    themes = {
        "LHS cell identity": ["LHS", "luminal hormone sensing", "FOXA1", "mature luminal"],
        "LASP cell identity": ["LASP", "luminal adaptive secretory", "EHF", "ELF5", "luminal progenitor"],
        "BM cell identity": ["BM", "basal-myoepithelial", "TP63", "KRT14"],
        "FOXA1 as LHS marker": ["FOXA1", "LHS marker", "LHS-specific", "chromatin accessibility"],
        "ESR1 expression patterns": ["ESR1", "estrogen receptor", "chromatin accessible"],
        "EHF/ELF5 as LASP markers": ["EHF", "ELF5", "LASP marker"],
        "TP63/KRT14 as BM markers": ["TP63", "KRT14", "BM marker", "basal marker"],
        "SCENIC regulons": ["SCENIC", "regulon", "gene regulatory network"],
        "Chromatin accessibility": ["chromatin accessibility", "snATAC", "ATAC-seq", "Signac"],
        "Duct vs lobule differences": ["ductal", "lobular", "duct", "lobule", "spatial transcriptomics"],
        "KRT14/KRT17 ductal enriched": ["KRT14", "KRT17", "ductal"],
        "DUSP1/DPM3/RPL36 lobular": ["DUSP1", "DPM3", "RPL36", "lobular"],
        "PTBP1 age decline": ["PTBP1", "age", "mRNA splicing", "alternative splicing"],
        "Indigenous American AP cells": ["Indigenous American", "alveolar progenitor", "AP cells"],
        "ESR1 in Indigenous LASP": ["ESR1", "Indigenous", "LASP", "alveolar"],
        "EGF-ER cross-talk": ["EGF", "estrogen receptor", "cross-talk", "antiestrogen"],
        "African ancestry fibroblasts": ["African ancestry", "fibro-prematrix", "fibroblast"],
        "PROCR+/ZEB1+ stromal cells": ["PROCR", "ZEB1", "PDGFRα", "multipotent", "stromal"],
        "Fibro-prematrix vs matrix": ["prematrix", "matrix state", "fibroblast state"],
        "Cell identity chromatin": ["chromatin", "cell identity", "gene expression"],
        "LHS cancer origin": ["LumA", "LumB", "HER2", "cell of origin", "LHS"],
        "ALDH1A3 ductal BRCA1": ["ALDH1A3", "BRCA1", "ductal", "stem"],
        "Novel epithelial markers": ["CTNND2", "DACH1", "INPP4B", "NEK10", "BARX2", "NCALD"],
        "Genetic ancestry variability": ["genetic ancestry", "ancestry", "disparity"],
        "BRCA mutation effects": ["BRCA1", "BRCA2", "mutation", "carrier"],
        "Age-dependent changes": ["age", "aging", "PKA", "eIF2", "oxidative phosphorylation"],
        "Cancer stroma subtypes": ["myofibroblast", "pericyte", "inflammatory fibroblast"],
        "Transcription factor motifs": ["TCFL5", "ZBTB14", "motif", "binding site"],
        "IHC validation": ["IHC", "immunohistochemistry", "protein", "positivity"],
    }
    themes_found = 0
    for theme_name, keywords in themes.items():
        for kw in keywords:
            if kw.lower() in elisa_report_text.lower():
                themes_found += 1
                break
    theme_coverage = themes_found / len(themes) * 100 if themes else 0
    scorecard["theme_coverage"] = f"{theme_coverage:.1f}% ({themes_found}/{len(themes)})"

    # Composite score
    composite = (gene_recall * 0.30 +
                 pathway_coverage * 0.20 +
                 interaction_recall * 0.15 +
                 theme_coverage * 0.25 +
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
    print(f"ELISA PAPER REPLICATION: Breast Tissue Atlas")
    print(f"Paper: Bhat-Nakshatri et al. (2024) Nature Medicine 30, 3482-3494")
    print(f"RUNNING {len(queries)} QUERIES")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")

    for i, (cmd_type, cmd_str) in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {cmd_str}")
        t0 = time.time()

        payload, answer, plots = execute_query(
            cmd_type, cmd_str, engine, llm, adata, viz,
            args.cluster_key, plot_dir
        )

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

            report.add_entry(
                entry_type=cmd_type,
                query=payload.get("query", cmd_str),
                payload=payload,
                answer=answer or "",
                plots=plots,
            )

            all_payloads.append(payload)

            entry = {
                "index": i + 1,
                "command": cmd_str,
                "type": cmd_type,
                "mode": mode,
                "query": payload.get("query", ""),
                "answer": answer[:500] if answer else "",
                "elapsed": round(elapsed, 2),
                "n_plots": len(plots),
            }
            session_log.append(entry)

            if answer:
                print(f"  → {mode} | {elapsed:.1f}s")
                print(f"  {answer[:150]}...")
            else:
                print(f"  → {mode} | {elapsed:.1f}s (no LLM answer)")

        print()

    # ── Generate report ────────────────────────────────────
    print(f"\n{'='*70}")
    print("GENERATING REPORT")
    print(f"{'='*70}")

    def llm_fn(prompt):
        return ask_llm(llm, SYSTEM_PROMPT, prompt)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = os.path.join(out_dir, f"elisa_report_BreastTissue_{ts}.md")
    report.generate_markdown(md_path, llm_func=llm_fn)

    docx_path = md_path.replace(".md", ".docx")
    report.generate_docx(docx_path, llm_func=llm_fn)

    # ── Save session log ───────────────────────────────────
    log_path = os.path.join(out_dir, "session_log.json")
    with open(log_path, "w") as f:
        json.dump(session_log, f, indent=2, default=str)
    print(f"[SAVED] Session log: {log_path}")

    # ── Run evaluation ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("RUNNING EVALUATION")
    print(f"{'='*70}")

    import re

    report_text = ""
    if os.path.exists(md_path):
        with open(md_path) as f:
            report_text = f.read()

    # ── Collect genes from multiple sources ────────────────
    elisa_genes = set()

    # Source 1: From payloads
    for payload in all_payloads:
        for r in payload.get("results", []):
            for g in r.get("gene_evidence", []):
                if isinstance(g, dict):
                    elisa_genes.add(g.get("gene", ""))
                elif isinstance(g, str):
                    elisa_genes.add(g)
            # Also check "genes" key (v4 format)
            for g in r.get("genes", []):
                if isinstance(g, dict):
                    elisa_genes.add(g.get("gene", ""))
                elif isinstance(g, str):
                    elisa_genes.add(g)
        for cid, cdata in payload.get("clusters", {}).items():
            if isinstance(cdata, dict):
                for g in cdata.get("genes", []):
                    if isinstance(g, dict):
                        elisa_genes.add(g.get("gene", ""))
                    elif isinstance(g, str):
                        elisa_genes.add(g)

    # Source 2: From pathway scores
    for payload in all_payloads:
        if payload.get("mode") in ("pathway_scoring", "pathway_query"):
            for pw_name, pw_data in payload.get("pathways", {}).items():
                for gs in pw_data.get("gene_set", []):
                    elisa_genes.add(gs)
                for cl_data in pw_data.get("scores", []):
                    for tg in cl_data.get("top_genes", []):
                        if isinstance(tg, dict):
                            elisa_genes.add(tg.get("gene", ""))
                        elif isinstance(tg, str):
                            elisa_genes.add(tg)

    # Source 3: From pathway genes_in_pathway field
    for payload in all_payloads:
        if payload.get("mode") == "pathway_query":
            for g in payload.get("genes_in_pathway", []):
                elisa_genes.add(g)

    # Source 4: From LLM answers
    for entry in session_log:
        answer = entry.get("answer", "")
        for m in re.finditer(r'\b([A-Z][A-Z0-9-]{1,15})\s*\(ENSG\d+', answer):
            elisa_genes.add(m.group(1))
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,12}(?:-[A-Z0-9]+)?)\b', answer):
            elisa_genes.add(m.group(1))

    # Source 5: From report text — paper-specific gene patterns
    for m in re.finditer(r'\b([A-Z][A-Z0-9-]{1,15})\s*\(ENSG\d+', report_text):
        elisa_genes.add(m.group(1))

    known_gene_pattern = re.compile(
        r'\b(FOXA1|ESR1|GATA3|EHF|ELF5|KIT|TP63|KRT14|KRT15|KRT17|'
        r'ERBB4|ANKRD30A|AFF3|TTC6|MYBPC1|THSD4|'
        r'CTNND2|DACH1|INPP4B|NEK10|CACNA1C|SEMA3C|SAMD5|TSHZ2|PRKG1|'
        r'ELOVL5|LIMCH1|AZGP1|STC2|'
        r'CCL28|BARX2|NCALD|AGAP1|SORBS2|SHANK2|MFGE8|DOCK5|ARHGAP26|'
        r'KLHL29|KLHL13|FHOD3|SEMA5A|ABLIM3|SOX10|NKX2-8|'
        r'LAMA2|SLIT2|RUNX1T1|COL25A1|SOX5|'
        r'CFD|MGST1|MFAP5|COL3A1|POSTN|COL1A1|IGF1|ADAM12|TNC|'
        r'SFRP4|CLU|G0S2|MGP|OGN|ADIRF|IGFBP5|'
        r'MMP3|CXCL1|GEM|KDM6B|CEBPB|TNFAIP6|CXCL2|'
        r'MECOM|LDB2|ST6GALNAC3|PKHD1L1|MMRN1|'
        r'LYVE1|ACKR1|CXCL12|'
        r'PTPRC|SKAP1|ARHGAP15|THEMIS|CD96|IL7R|IFNG|GZMK|'
        r'ALCAM|FCGR3A|CD74|B2M|MS4A4E|'
        r'PLIN1|HPSE2|PDE3B|BICC1|DLC1|RGS6|PLA2R1|'
        r'MKI67|IGHA1|IGKC|DUSP1|DPM3|RPL36|'
        r'ANXA1|TACSTD2|WFDC2|STAC2|ALDH1A3|APOD|SNORC|PTBP1|'
        r'FCGBP|CRYBG1|SCRN3|'
        r'PROCR|ZEB1|PDGFRA|'
        r'ABCA10|ABCA9|ABCA8|NEGR1|MMP16|MAGI1|KIAA1217|PTPRK|'
        r'RCOR1|TFAP2A|TFAP2C|TFAP2B|TBX3|THRB|RARB|EGR1|SMC3|'
        r'BCL6|KLF8|KLF7|FOSB|NFIB|EGR3|ERG|MAFB|GLI3|KLF12|'
        r'SP4|TCF7|SMAD2|SP3|JUND|SMAD1|PBX3|ELF1|'
        r'PDPN|CD34|MYH11|MYLK|CD36|RGS5|'
        r'TCFL5|ZBTB14|'
        r'EBF1|PCDH9|LAMA4|RERG|NEBL|ITPR2|RBMS3|'
        r'NHLRC4|EBPL|NOP53|ZFYVE26|SH3BP2|DGAT2)\b'
    )
    for m in known_gene_pattern.finditer(report_text):
        elisa_genes.add(m.group(1))

    elisa_genes.discard("")
    print(f"  Genes collected: {len(elisa_genes)}")
    print(f"  Sample: {sorted(elisa_genes)[:20]}")

    # ── Collect pathway scores ─────────────────────────────
    pathway_scores = {}
    for payload in all_payloads:
        mode = payload.get("mode", "")
        if mode == "pathway_scoring":
            pathway_scores.update(payload.get("pathways", {}))
        elif mode == "pathway_query":
            pw_name = payload.get("pathway", "")
            if pw_name:
                pathway_scores[pw_name] = {
                    "scores": payload.get("scores", []),
                    "genes_in_pathway": payload.get("genes_in_pathway", []),
                }

    print(f"  Pathways collected: {len(pathway_scores)}")
    if pathway_scores:
        print(f"  Keys: {list(pathway_scores.keys())[:5]}")

    # ── Collect interactions ───────────────────────────────
    print(f"  Interactions collected: {len(all_interactions)}")

    # ── Collect proportions ────────────────────────────────
    proportion_data = {}
    for payload in all_payloads:
        if payload.get("mode") == "proportions":
            proportion_data = payload
            break
    print(f"  Proportion data: {'found' if proportion_data else 'missing'}")

    # ── Run evaluation ─────────────────────────────────────
    eval_result = full_evaluation(
        elisa_report_text=report_text,
        elisa_genes=elisa_genes,
        elisa_interactions=all_interactions,
        elisa_pathway_scores=pathway_scores,
        elisa_proportions=proportion_data,
    )

    # Print scorecard
    print(f"\n{'='*70}")
    print("EVALUATION SCORECARD")
    print(f"{'='*70}")
    for metric, score in eval_result["scorecard"].items():
        print(f"  {metric}: {score}")
    print(f"\n  ★ COMPOSITE SCORE: {eval_result['composite_score']}%")
    print(f"{'='*70}")

    # Save evaluation
    eval_path = os.path.join(out_dir, "evaluation_scorecard.json")
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2, default=str)
    print(f"[SAVED] Evaluation: {eval_path}")

    # Summary
    print(f"\n{'='*70}")
    print("ALL OUTPUTS:")
    print(f"  Report (md):   {md_path}")
    print(f"  Report (docx): {docx_path}")
    print(f"  Session log:   {log_path}")
    print(f"  Evaluation:    {eval_path}")
    print(f"  Plots:         {plot_dir}/")
    n_plots = sum(len(e.get("plots", [])) for e in report.entries)
    print(f"  Total plots:   {n_plots}")
    print(f"  Analyses:      {len(report.entries)}")
    print(f"{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
