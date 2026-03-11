#!/usr/bin/env python
"""
ELISA Paper Replication — Automated Batch Runner
=================================================
Target Paper:  Gondal, Cieslik & Chinnaiyan (2025) Scientific Data 12:139
               "Integrated cancer cell-specific single-cell RNA-seq datasets
                of immune checkpoint blockade-treated patients"

8 studies, 9 cancer types, 223 patients, 90,270 cancer cells, 265,671 non-malignant.
Cancer types: Melanoma, MBM, BCC, TNBC, HER2+, ER+, HCC, iCCA, ccRCC.

31 clusters (Cell Ontology):
    B cell, CD4+ T cell, CD8+CD28- reg T cell, CD8+ T cell, T cell,
    T follicular helper, Th17, activated CD8+ T cell, central memory CD8+ T cell,
    dendritic cell, effector CD8+ T cell, endothelial cell, epithelial cell of thymus,
    fibroblast, hematopoietic progenitor, lymphocyte, macrophage, malignant cell,
    mast cell, mature NK T cell, melanocyte, microglial cell, monocyte,
    myeloid cell, myofibroblast, naive T cell, naive CD8+ T cell,
    plasma cell, plasmacytoid DC, regulatory T cell, unknown

Condition: pre_post (Pre / Post)

Usage:
    python run_paper_replication_ICB.py \\
        --base /path/to/embeddings/ \\
        --pt-name fused_DT7_ICB.pt \\
        --out-dir elisa_replication_ICB/
"""

import os, sys, json, time, argparse, functools
from datetime import datetime

print = functools.partial(print, flush=True)


def setup_args():
    p = argparse.ArgumentParser(description="ELISA Paper Replication — ICB Multi-Cancer")
    p.add_argument("--h5ad", default=None)
    p.add_argument("--cluster-key", default="cell_type")
    p.add_argument("--out-dir", default="elisa_replication_ICB")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--dataset-name", default="ICB Multi-Cancer — Integrated scRNA-seq Atlas")
    p.add_argument("--base", required=True)
    p.add_argument("--pt-name", required=True)
    p.add_argument("--cells-csv", default="metadata_cells.csv")
    return p.parse_args()


def init_elisa(args):
    from retrieval_engine_v4_hybrid import RetrievalEngine
    from elisa_analysis import (find_interactions, pathway_scoring,
                                proportion_analysis, comparative_analysis, query_pathway)
    from elisa_report import ReportBuilder
    import elisa_viz as viz

    print("[INIT] Loading retrieval engine (v4 hybrid)...")
    raw_engine = RetrievalEngine(base=args.base, pt_name=args.pt_name, cells_csv=args.cells_csv)
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

        def discover(self, text, top_k=5, **kw):
            payload = self._eng.query_semantic(text, top_k=top_k, with_genes=True)
            payload["mode"] = "discovery"; payload["query"] = text
            return payload

        def interactions(self, source=None, target=None, **kw):
            sc = [c for c in self._eng.cluster_ids if source.lower() in str(c).lower()] if source else None
            tc = [c for c in self._eng.cluster_ids if target.lower() in str(c).lower()] if target else None
            if sc is not None and not sc: sc = None
            if tc is not None and not tc: tc = None
            return find_interactions(self._eng.gene_stats, self._eng.cluster_ids,
                                     source_clusters=sc, target_clusters=tc, **kw)

        def pathway(self, pathway_name=None, **kw):
            if pathway_name:
                return query_pathway(self._eng.gene_stats, self._eng.cluster_ids, pathway_name=pathway_name, **kw)
            return pathway_scoring(self._eng.gene_stats, self._eng.cluster_ids, **kw)

        def proportions(self, **kw):
            return proportion_analysis(self._eng.metadata, **kw)

        def compare(self, group_a, group_b, genes=None, **kw):
            condition_col = None
            for cid, m in (self._eng.metadata or {}).items():
                if isinstance(m, dict):
                    for col, dist in m.get("fields", {}).items():
                        if isinstance(dist, dict):
                            if group_a.lower() in {k.lower() for k in dist} or group_b.lower() in {k.lower() for k in dist}:
                                condition_col = col; break
                if condition_col: break
            if not condition_col: condition_col = "pre_post"
            return comparative_analysis(self._eng.gene_stats, self._eng.metadata,
                                        condition_col=condition_col, group_a=group_a, group_b=group_b, genes=genes, **kw)

        def detect_capabilities(self):
            caps = {"has_conditions": False, "condition_values": [], "condition_column": None,
                    "n_clusters": len(self._eng.cluster_ids), "cluster_ids": list(self._eng.cluster_ids)}
            kws = ["patient_group", "case_control", "condition", "disease", "treatment",
                   "status", "timepoint", "pre_post", "group", "sample_type"]
            col_cands = {}
            for cid, m in (self._eng.metadata or {}).items():
                if not isinstance(m, dict): continue
                for col, dist in m.get("fields", {}).items():
                    if isinstance(dist, dict):
                        col_cands.setdefault(col, set()).update(dist.keys())
            for kw in kws:
                for col, vals in col_cands.items():
                    if kw in col.lower().replace("_", " "):
                        if 2 <= len(vals) <= 10:
                            caps.update(has_conditions=True, condition_column=col, condition_values=sorted(vals))
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
            adata = sc.read_h5ad(args.h5ad)
            if adata.var_names[0].startswith("ENSG") and "feature_name" in adata.var.columns:
                adata.var_names = adata.var["feature_name"].astype(str).values; adata.var_names_make_unique()
            print(f"[INIT] h5ad: {adata.shape[0]} cells, {adata.shape[1]} genes")
        except ImportError: print("[WARN] scanpy not installed")

    report = ReportBuilder(dataset_name=args.dataset_name)
    caps = engine.detect_capabilities()
    print(f"[INIT] Conditions: {caps.get('has_conditions')} → {caps.get('condition_values', [])}")
    return engine, llm, adata, report, viz, caps


def ask_llm(llm, system_prompt, user_prompt):
    res = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.2)
    return res.choices[0].message.content.strip()

SYSTEM_PROMPT = (
    "You are ELISA, an expert assistant for single-cell biology and immuno-oncology. "
    "Never hallucinate. Always ground claims strictly in provided data. Be concise and scientific. "
    "Focus on immune checkpoint blockade (ICB) biology: PD-1/PD-L1 axis, T cell exhaustion, "
    "effector CD8 T cell cytotoxicity, regulatory T cells, tumor immune evasion, "
    "macrophage polarization, NK cell activity, B cell and plasma cell responses, "
    "melanoma/breast/liver/kidney cancer markers, cancer-associated fibroblasts, "
    "and pre vs post ICB treatment changes across multiple cancer types."
)

MAX_PROMPT_CHARS = 12000

def trim_payload(payload, max_chars=MAX_PROMPT_CHARS):
    t = dict(payload)
    if "clusters" in t and isinstance(t["clusters"], dict) and len(t["clusters"]) > 10:
        t["clusters"] = dict(sorted(t["clusters"].items(), key=lambda kv: len(kv[1].get("genes", [])), reverse=True)[:10])
    if "interactions" in t and isinstance(t["interactions"], list) and len(t["interactions"]) > 30:
        t["interactions"] = t["interactions"][:30]
    if "scores" in t and isinstance(t["scores"], list): t["scores"] = t["scores"][:10]
    for r in t.get("results", []):
        if "gene_evidence" in r and len(r["gene_evidence"]) > 5: r["gene_evidence"] = r["gene_evidence"][:5]
    ctx = json.dumps(t, indent=1, default=str)
    return ctx[:max_chars] + "\n... [TRUNCATED]" if len(ctx) > max_chars else ctx

def build_prompt(mode, query, payload):
    ctx = trim_payload(payload)
    templates = {
        "discovery": f"You are in DISCOVERY mode.\nSeparate into: 1. DATASET EVIDENCE 2. ESTABLISHED BIOLOGY\n3. CONSISTENCY ANALYSIS 4. CANDIDATE NOVEL HYPOTHESES\nQUESTION: {query}\nDATASET CONTEXT: {ctx}",
        "compare": f"You are ELISA analyzing COMPARATIVE analysis.\nCOMPARISON: {query}\nEVIDENCE: {ctx}\nIdentify condition-biased clusters, highlight differentially expressed genes.",
        "interactions": f"You are ELISA analyzing CELL-CELL INTERACTIONS.\nQUERY: {query}\nINTERACTIONS: {ctx}\nFocus on highest-scoring, group by pathway, note unexpected interactions.",
        "proportions": f"You are ELISA analyzing CELL TYPE PROPORTIONS.\nQUERY: {query}\nDATA: {ctx}\nReport major types, condition differences, biological implications.",
    }
    if mode in ("pathway_scoring", "pathway_query"):
        return f"You are ELISA analyzing PATHWAY ACTIVITY.\nQUERY: {query}\nSCORES: {ctx}\nIdentify top cell types, contributing genes, biological relevance."
    return templates.get(mode, f"You are ELISA for single-cell analysis.\nMODE: {mode.upper()} | QUERY: {query}\nEVIDENCE: {ctx}\nUse ONLY provided evidence. Be concise and scientific.")


# ══════════════════════════════════════════════════════════════
# QUERIES — Gondal et al. Sci Data 2025
# ══════════════════════════════════════════════════════════════

def get_queries(skip_plots=False):
    queries = [
        # ── Phase 1: Landscape ──
        ("info", "info"),
        ("proportions", "proportions"),

        # ── Phase 2: Major cell types ──
        ("semantic", "semantic: malignant cancer cell PD-L1 CD274 immune evasion checkpoint ligand"),
        ("semantic", "semantic: effector CD8 T cell cytotoxic granzyme perforin GZMB PRF1 anti-tumor"),
        ("semantic", "semantic: activated CD8 T cell IFNG TNF anti-tumor cytokines NKG7"),
        ("semantic", "semantic: CD8 T cell exhaustion PD-1 LAG3 TIM3 TIGIT TOX checkpoint receptors"),
        ("semantic", "semantic: regulatory T cell FOXP3 immunosuppressive function tumor"),
        ("semantic", "semantic: CD4 positive helper T cell TCR signaling cytokine production"),
        ("semantic", "semantic: T follicular helper cell CXCR5 BCL6 tertiary lymphoid structures"),
        ("semantic", "semantic: natural killer T cell NKT innate cytotoxicity KLRD1 NKG7"),
        ("semantic", "semantic: B cell CD19 MS4A1 antigen presentation humoral immunity"),
        ("semantic", "semantic: plasma cell antibody secreting SDC1 MZB1 immunoglobulin"),
        ("semantic", "semantic: macrophage M2 polarization CD163 MRC1 immunosuppressive tumor"),
        ("semantic", "semantic: monocyte CD14 LYZ infiltration checkpoint blockade"),
        ("semantic", "semantic: dendritic cell CD80 CD86 antigen presentation priming T cells"),
        ("semantic", "semantic: plasmacytoid dendritic cell IRF7 LILRA4 type I interferon"),
        ("semantic", "semantic: cancer associated fibroblast FAP ACTA2 COL1A1 extracellular matrix"),
        ("semantic", "semantic: endothelial cell PECAM1 CDH5 VWF tumor vasculature"),
        ("semantic", "semantic: melanocyte MITF TYR TYRP1 DCT pigmentation lineage"),
        ("semantic", "semantic: mast cell KIT TPSB2 CPA3 allergic inflammatory"),

        # ── Phase 3: PD-1/PD-L1 axis & immune checkpoints ──
        ("discover", "discover: CD274 PDCD1 PD-L1 PD-1 checkpoint blockade immune evasion tumor"),
        ("discover", "discover: PDCD1LG2 PD-L2 alternative checkpoint ligand macrophage"),
        ("discover", "discover: CTLA4 CD80 CD86 B7 co-stimulation inhibition regulatory T cell"),
        ("discover", "discover: LGALS9 HAVCR2 TIM-3 galectin-9 checkpoint exhaustion"),
        ("discover", "discover: CD47 SIRPA dont eat me signal macrophage phagocytosis evasion"),
        ("discover", "discover: TIGIT CD226 NECTIN2 checkpoint receptor T cell NK cell"),

        # ── Phase 4: T cell biology ──
        ("discover", "discover: PRF1 GZMA GZMB GZMK GNLY NKG7 cytotoxic effector CD8 T cell killing"),
        ("discover", "discover: TOX TOX2 PDCD1 LAG3 HAVCR2 TIGIT T cell exhaustion program"),
        ("discover", "discover: TCF7 IL7R CCR7 SELL LEF1 central memory CD8 T cell stem-like"),
        ("discover", "discover: FOXP3 IL2RA CTLA4 IKZF2 TNFRSF18 regulatory T cell suppression"),
        ("discover", "discover: CXCR5 BCL6 ICOS PDCD1 T follicular helper germinal center"),
        ("discover", "discover: RORC IL17A IL23R CCR6 Th17 inflammatory response"),
        ("discover", "discover: CD8A GZMB PRF1 LAG3 CTLA4 CD28-negative regulatory T cell"),
        ("discover", "discover: CCR7 SELL TCF7 naive T cell before antigen encounter"),
        ("discover", "discover: IFNG CD274 STAT1 IRF1 interferon gamma PD-L1 upregulation"),

        # ── Phase 5: Myeloid biology ──
        ("discover", "discover: C1QA C1QB APOE TREM2 CD68 complement macrophage tumor"),
        ("discover", "discover: CD14 FCGR3A S100A8 S100A9 LYZ classical monocyte infiltration"),
        ("discover", "discover: CD80 CD86 CD83 CCR7 dendritic cell activation maturation"),
        ("discover", "discover: LILRA4 IRF7 IRF8 CLEC4C plasmacytoid DC interferon"),
        ("discover", "discover: ITGAM CSF1R CD68 LYZ myeloid cell innate immune"),
        ("discover", "discover: P2RY12 TMEM119 CX3CR1 microglial cell brain resident melanoma"),
        ("discover", "discover: KIT TPSB2 TPSAB1 CPA3 HPGDS mast cell degranulation"),

        # ── Phase 6: B cell & humoral immunity ──
        ("discover", "discover: CD19 MS4A1 CD79A CD79B B cell antigen presentation tumor"),
        ("discover", "discover: SDC1 MZB1 JCHAIN IGHG1 IGKC plasma cell antibody secretion"),
        ("discover", "discover: MS4A1 CD79A SDC1 MZB1 CXCR5 tertiary lymphoid structure ICB"),

        # ── Phase 7: Stromal cells ──
        ("discover", "discover: FAP ACTA2 COL1A1 COL1A2 PDGFRA DCN cancer associated fibroblast"),
        ("discover", "discover: ACTA2 TAGLN MYH11 COL1A1 PDGFRB myofibroblast contractile"),
        ("discover", "discover: PECAM1 CDH5 VWF KDR FLT1 ENG endothelial vascular"),
        ("discover", "discover: CD34 KIT FLT3 PROM1 hematopoietic progenitor stem cell"),

        # ── Phase 8: Cancer type-specific markers ──
        ("discover", "discover: MITF MLANA PMEL TYR DCT SOX10 melanoma lineage markers"),
        ("discover", "discover: EPCAM KRT8 KRT18 KRT19 MUC1 CDH1 breast cancer epithelial"),
        ("discover", "discover: ALB AFP GPC3 hepatocellular carcinoma liver cancer markers"),
        ("discover", "discover: CA9 PAX8 MME clear cell renal carcinoma kidney cancer"),
        ("discover", "discover: PTCH1 GLI1 GLI2 Hedgehog pathway basal cell carcinoma"),
        ("discover", "discover: ERBB2 ESR1 EPCAM KRT8 MUC1 breast cancer subtypes HER2 ER"),
        ("discover", "discover: B2M HLA-A HLA-B HLA-C MHC class I tumor antigen presentation"),
        ("discover", "discover: CD274 TGFB1 VEGFA IDO1 CD47 tumor immune evasion resistance"),
        ("discover", "discover: VIM CDH2 SNAI1 ZEB1 CD44 epithelial mesenchymal transition EMT"),
        ("discover", "discover: MKI67 TOP2A PCNA CDK1 CCNB1 tumor proliferation cell cycle"),

        # ── Phase 9: NK cell biology ──
        ("discover", "discover: KLRD1 KLRK1 NKG7 GNLY PRF1 GZMB NCAM1 NK T cell cytotoxicity"),
        ("discover", "discover: NCAM1 NCR1 KLRB1 KLRC1 NK cell receptor tumor killing"),

        # ── Phase 10: Comparisons Pre vs Post ──
        ("compare", "compare: Pre vs Post | CD274, PDCD1, GZMB, PRF1, IFNG, NKG7, CD8A"),
        ("compare", "compare: Pre vs Post | FOXP3, IL2RA, CTLA4, TIGIT, TOX, LAG3, HAVCR2"),
        ("compare", "compare: Pre vs Post | CD68, CD163, C1QA, SPP1, TREM2, MRC1, APOE"),
        ("compare", "compare: Pre vs Post | MS4A1, CD79A, SDC1, MZB1, JCHAIN, IGHG1"),
        ("compare", "compare: Pre vs Post | FAP, ACTA2, COL1A1, PECAM1, CDH5, VWF"),
        ("compare", "compare: Pre vs Post | MITF, MLANA, EPCAM, KRT8, ALB, CA9, PTCH1"),
        ("compare", "compare: Pre vs Post | CD274, CD47, IDO1, TGFB1, VEGFA, B2M, HLA-A"),
        ("compare", "compare: Pre vs Post | MKI67, TOP2A, PCNA, VIM, CDH2, SNAI1, ZEB1"),

        # ── Phase 11: Pathways ──
        ("pathway", "pathway: T cell activation"),
        ("pathway", "pathway: NK cell activity"),
        ("pathway", "pathway: Antigen processing and presentation"),
        ("pathway", "pathway: IFN-gamma signaling"),
        ("pathway", "pathway: Type I IFN signaling"),
        ("pathway", "pathway: TNF signaling"),
        ("pathway", "pathway: Chemokine signaling"),
        ("pathway", "pathway: Cytokine-cytokine receptor interaction"),
        ("pathway", "pathway: PI3K-Akt signaling"),
        ("pathway", "pathway: MAPK signaling"),
        ("pathway", "pathway: EGF signaling"),
        ("pathway", "pathway: Angiogenesis"),
        ("pathway", "pathway: Fibrosis"),
        ("pathway", "pathway: Apoptosis"),
        ("pathway", "pathway: Complement and coagulation"),
        ("pathway", "pathway: BCR signaling"),
        ("pathway", "pathway: all"),

        # ── Phase 12: Cell-cell interactions ──
        ("interactions", "interactions:"),
        ("interactions", "interactions: malignant -> T cell"),
        ("interactions", "interactions: malignant -> macrophage"),
        ("interactions", "interactions: macrophage -> T cell"),
        ("interactions", "interactions: dendritic -> T cell"),
        ("interactions", "interactions: fibroblast -> malignant"),
        ("interactions", "interactions: endothelial -> T cell"),
        ("interactions", "interactions: B cell -> T cell"),
        ("interactions", "interactions: NK -> malignant"),

        # ── Phase 13: Discussion — resistance and response ──
        ("discover", "discover: TCF4 mesenchymal-like MES program resistance immunotherapy melanoma"),
        ("discover", "discover: T cell exclusion program tumor cells resisting checkpoint blockade"),
        ("discover", "discover: clonal replacement tumor-specific T cells PD-1 blockade BCC"),
        ("discover", "discover: tumor cell biodiversity microenvironmental reprogramming liver cancer"),
        ("discover", "discover: tumor immune reprogramming immunotherapy renal cell carcinoma"),
        ("discover", "discover: intratumoral changes anti-PD1 treatment breast cancer single-cell"),
        ("discover", "discover: melanoma brain metastases microenvironmental immune checkpoint response"),

        # ── Phase 14: Additional gene recovery ──
        ("discover", "discover: PDCD1 CD274 CTLA4 LAG3 HAVCR2 TIGIT immune checkpoint receptors"),
        ("discover", "discover: GZMB PRF1 IFNG TNF FASLG NKG7 cytotoxic effector molecules"),
        ("discover", "discover: HLA-DRA HLA-DRB1 HLA-DPA1 CD74 CIITA MHC class II presentation"),
        ("discover", "discover: S100A8 S100A9 LYZ CD14 FCGR3A myeloid alarmin innate"),
        ("discover", "discover: COL1A1 COL1A2 FN1 PDGFRA DCN LUM fibroblast ECM stroma"),
        # ── Phase 15: Targeted gene recovery — missing genes ──

        # Batch A: T cell identity & exhaustion
        ("compare", "compare: Pre vs Post | CD8A, CD8B, TCF7, LEF1, SELL, TOX, TOX2"),
        ("compare", "compare: Pre vs Post | ENTPD1, BTLA, NKG7, TRAC, TRBC1, PCNA"),
        ("compare", "compare: Pre vs Post | RORC, IL23R, IL7R, IL3RA, IKZF2, FOXP3"),

        # Batch B: Treg / co-stimulation
        ("compare", "compare: Pre vs Post | TNFRSF4, TNFRSF9, TNFRSF18, CD28, ICOS, PDCD1LG2"),

        # Batch C: NK / mast
        ("compare", "compare: Pre vs Post | NCAM1, NKG7, KIT, TPSB2, TPSAB1, CPA3, HPGDS"),

        # Batch D: Macrophage / myeloid
        ("compare", "compare: Pre vs Post | CD68, CD163, MRC1, MSR1, MARCO, SPP1, APOE, TREM2"),
        ("compare", "compare: Pre vs Post | CD14, S100A8, S100A9, AIF1, CD47, TMEM119"),

        # Batch E: DC subtypes
        ("compare", "compare: Pre vs Post | CLEC9A, XCR1, CD1C, FCER1A, LILRA4, FLT3, CD83"),

        # Batch F: B cell / plasma
        ("compare", "compare: Pre vs Post | MS4A1, SDC1, MZB1, JCHAIN, BCL6, PROM1"),

        # Batch G: Melanoma lineage
        ("compare", "compare: Pre vs Post | MITF, MLANA, PMEL, SOX10, TYRP1, DCT"),

        # Batch H: Breast cancer epithelial
        ("compare", "compare: Pre vs Post | EPCAM, KRT8, KRT18, KRT19, MUC1, ESR1, CDH1"),

        # Batch I: Liver / kidney / BCC cancer markers
        ("compare", "compare: Pre vs Post | ALB, AFP, GPC3, CA9, PAX8, MME"),
        ("compare", "compare: Pre vs Post | PTCH1, GLI1, GLI2, SNAI1, ZEB1, CDH2"),

        # Batch J: Stromal / endothelial
        ("compare", "compare: Pre vs Post | TAGLN, MYH11, ENG, CD34, PTPRC, ALDH1A1"),

        # Batch K: Proliferation / EMT
        ("compare", "compare: Pre vs Post | MKI67, TOP2A, PCNA, SNAI1, ZEB1, VIM"),

        # Batch L: Semantic recovery for remaining stubborn genes
        ("semantic", "semantic: CD68 CD163 MRC1 MSR1 MARCO tumor associated macrophage M2 polarization"),
        ("semantic", "semantic: MITF MLANA PMEL SOX10 TYRP1 melanoma cancer cell lineage markers"),
        ("semantic", "semantic: EPCAM KRT8 KRT18 KRT19 MUC1 epithelial breast cancer markers"),
        ("semantic", "semantic: NKG7 NCAM1 GZMK cytotoxic NK T cell effector molecules"),
        ("semantic", "semantic: SDC1 MZB1 JCHAIN plasma cell antibody secreting immunoglobulin"),
        ("semantic", "semantic: TOX TOX2 ENTPD1 BTLA T cell exhaustion terminal differentiation"),
        ("semantic", "semantic: S100A8 S100A9 CD14 myeloid monocyte alarmin innate immunity"),
        ("semantic", "semantic: TPSB2 TPSAB1 CPA3 HPGDS KIT mast cell tryptase histamine"),
        ("semantic", "semantic: CLEC9A XCR1 CD1C FCER1A dendritic cell cross-presentation"),
        ("semantic", "semantic: LILRA4 IL3RA plasmacytoid dendritic cell type I interferon"),
        ("semantic", "semantic: ALB AFP GPC3 hepatocellular carcinoma liver cancer cell markers"),
        ("semantic", "semantic: CA9 PAX8 MME clear cell renal carcinoma kidney cancer markers"),
        ("semantic", "semantic: PTCH1 GLI1 GLI2 Hedgehog signaling basal cell carcinoma"),
        ("semantic", "semantic: CD47 SIRPA dont eat me phagocytosis immune evasion tumor"),
        ("semantic", "semantic: FOXP3 IKZF2 IL2RA TNFRSF18 regulatory T cell immunosuppression"),
    ]

    if not skip_plots:
        plot_queries = [
            ("plot", "plot:umap"),
            ("plot", "plot:expr CD274"), ("plot", "plot:expr PDCD1"), ("plot", "plot:expr GZMB"),
            ("plot", "plot:expr PRF1"), ("plot", "plot:expr IFNG"), ("plot", "plot:expr FOXP3"),
            ("plot", "plot:expr CD68"), ("plot", "plot:expr CD163"), ("plot", "plot:expr MS4A1"),
            ("plot", "plot:expr MITF"), ("plot", "plot:expr EPCAM"), ("plot", "plot:expr FAP"),
            ("plot", "plot:expr PECAM1"), ("plot", "plot:expr KIT"), ("plot", "plot:expr TOX"),
            ("plot", "plot:expr MKI67"), ("plot", "plot:expr CD47"), ("plot", "plot:expr B2M"),
            ("plot", "plot:dotplot CD274, PDCD1, CTLA4, LAG3, HAVCR2, TIGIT, TOX, CD47"),
            ("plot", "plot:dotplot GZMB, PRF1, IFNG, NKG7, GNLY, GZMA, GZMK, TNF"),
            ("plot", "plot:dotplot CD68, CD163, C1QA, SPP1, TREM2, MRC1, APOE, CD14"),
            ("plot", "plot:dotplot MS4A1, CD79A, SDC1, MZB1, JCHAIN, IGHG1, IGKC, CD19"),
            ("plot", "plot:dotplot FAP, ACTA2, COL1A1, PECAM1, CDH5, VWF, KDR, FLT1"),
            ("plot", "plot:dotplot MITF, MLANA, EPCAM, KRT8, ALB, CA9, PTCH1, ERBB2"),
            ("plot", "plot:dotplot FOXP3, IL2RA, CXCR5, BCL6, RORC, CCR7, KLRD1, NCAM1"),
            ("plot", "plot:grid CD274, GZMB, FOXP3, CD68, MS4A1, FAP"),
            ("plot", "plot:grid MITF, EPCAM, ALB, CA9, PTCH1, MKI67"),
        ]
        queries.extend(plot_queries)
    return queries

def get_queries_test():
    return [
        ("info", "info"),
        ("proportions", "proportions"),
        ("semantic", "semantic: effector CD8 T cell GZMB PRF1 cytotoxic anti-tumor"),
        ("discover", "discover: CD274 PDCD1 PD-L1 PD-1 checkpoint immune evasion"),
        ("compare", "compare: Pre vs Post | CD274, PDCD1, GZMB, FOXP3, CD68"),
        ("interactions", "interactions:"),
        ("interactions", "interactions: malignant -> T cell"),
        ("pathway", "pathway: T cell activation"),
        ("pathway", "pathway: all"),
    ]


# ══════════════════════════════════════════════════════════════
# GROUND TRUTH — derived from Gondal et al. + constituent studies
# ══════════════════════════════════════════════════════════════

PAPER_GENES = {
    # PD-1/PD-L1 axis
    "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
    "TOX", "TOX2", "ENTPD1", "BTLA",
    # Effector molecules
    "PRF1", "GZMA", "GZMB", "GZMK", "GZMH", "GNLY", "NKG7", "IFNG", "FASLG", "TNF",
    # T cell activation / co-stimulation
    "CD69", "ICOS", "TNFRSF9", "TNFRSF4", "TNFRSF18", "IL2RA", "CD28",
    # TCR
    "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "TRAC", "TRBC1",
    # Treg
    "FOXP3", "IKZF2",
    # Memory / naive
    "CCR7", "SELL", "TCF7", "LEF1", "IL7R",
    # Tfh
    "CXCR5", "BCL6",
    # Th17
    "RORC", "IL17A", "IL23R", "CCR6",
    # NK
    "NCAM1", "KLRD1", "KLRK1", "NCR1", "KLRB1", "KLRC1",
    # B cell / plasma
    "CD19", "MS4A1", "CD79A", "CD79B", "SDC1", "MZB1", "JCHAIN", "IGHG1", "IGKC",
    # DC
    "CD80", "CD86", "CD83", "CLEC9A", "XCR1", "CD1C", "FCER1A",
    # pDC
    "LILRA4", "IRF7", "IRF8", "IL3RA", "NRP1",
    # Macrophage
    "CD68", "CD163", "MRC1", "MSR1", "MARCO", "SPP1", "C1QA", "C1QB", "APOE", "TREM2",
    # Monocyte
    "CD14", "FCGR3A", "S100A8", "S100A9", "LYZ",
    # Myeloid
    "ITGAM", "CSF1R",
    # Mast
    "KIT", "TPSB2", "TPSAB1", "CPA3", "HPGDS", "HDC",
    # Microglia
    "P2RY12", "TMEM119", "CX3CR1", "AIF1",
    # Fibroblast / myofibroblast
    "FAP", "ACTA2", "COL1A1", "COL1A2", "COL3A1", "PDGFRA", "PDGFRB", "DCN", "LUM", "VIM",
    "TAGLN", "MYH11",
    # Endothelial
    "PECAM1", "CDH5", "VWF", "KDR", "FLT1", "ENG",
    # HLA
    "HLA-A", "HLA-B", "HLA-C", "B2M",
    "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA",
    # Melanoma
    "MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10", "TYRP1",
    # Breast
    "EPCAM", "KRT8", "KRT18", "KRT19", "ESR1", "ERBB2", "MUC1", "CDH1",
    # Liver
    "ALB", "AFP", "GPC3",
    # Kidney
    "CA9", "PAX8", "MME",
    # BCC
    "PTCH1", "GLI1", "GLI2",
    # Immune evasion
    "CD47", "IDO1", "VEGFA", "TGFB1",
    # Proliferation
    "MKI67", "TOP2A", "PCNA",
    # EMT
    "CD44", "CDH2", "SNAI1", "ZEB1",
    # Stem
    "ALDH1A1", "PROM1", "CD34",
    # HPC
    "FLT3", "THY1", "PTPRC",
}

PAPER_INTERACTIONS = [
    ("malignant", "T cell", "PD-L1/PD-1 checkpoint"),
    ("malignant", "macrophage", "CD47/SIRPa dont-eat-me"),
    ("macrophage", "T cell", "antigen presentation"),
    ("dendritic", "T cell", "co-stimulation priming"),
    ("malignant", "NK", "HLA/KIR recognition"),
    ("fibroblast", "malignant", "ECM signaling"),
    ("endothelial", "T cell", "recruitment"),
    ("B cell", "T cell", "antigen presentation"),
    ("macrophage", "malignant", "complement/phagocytosis"),
    ("malignant", "effector", "IFNG response"),
]

PAPER_PATHWAYS = [
    "T cell activation", "NK cell activity",
    "Antigen processing and presentation",
    "IFN-gamma signaling", "Type I IFN signaling",
    "TNF signaling", "Chemokine signaling",
    "Cytokine-cytokine receptor interaction",
    "PI3K-Akt signaling", "MAPK signaling",
    "EGF signaling", "Angiogenesis",
    "Fibrosis", "Apoptosis",
    "Complement and coagulation", "BCR signaling",
]

PROPORTION_CHANGES = {
    "effector CD8 T cell": "expected increase post-ICB in responders",
    "activated CD8 T cell": "expected increase post-ICB",
    "plasma cell": "expected increase post-ICB in responders",
    "malignant cell": "expected decrease post-ICB in responders",
    "regulatory T cell": "may decrease post-ICB",
}


# ── Execute query ──
def execute_query(cmd_type, cmd_str, engine, llm, adata, viz, cluster_key, plot_dir):
    payload, answer, plots = None, None, []

    if cmd_type == "info":
        caps = engine.detect_capabilities()
        return {"mode": "info", "capabilities": caps}, json.dumps(caps, indent=2), []
    elif cmd_type == "proportions": payload = engine.proportions()
    elif cmd_type == "semantic":
        payload = engine.query_semantic(cmd_str.split(":", 1)[1].strip(), top_k=5, with_genes=True)
    elif cmd_type == "hybrid":
        payload = engine.query_hybrid(cmd_str.split(":", 1)[1].strip(), top_k=5, lambda_sem=0.0, with_genes=True)
    elif cmd_type == "discover": payload = engine.discover(cmd_str.split(":", 1)[1].strip(), top_k=5)
    elif cmd_type == "compare":
        txt = cmd_str.split(":", 1)[1].strip(); genes = None
        if "|" in txt: txt, gs = txt.split("|", 1); genes = [g.strip() for g in gs.split(",") if g.strip()]; txt = txt.strip()
        parts = txt.lower().split(" vs ")
        if len(parts) == 2:
            ga, gb = parts[0].strip(), parts[1].strip()
            caps = engine.detect_capabilities()
            if caps["has_conditions"]:
                for cv in caps["condition_values"]:
                    if cv.lower() == ga: ga = cv
                    if cv.lower() == gb: gb = cv
            payload = engine.compare(ga, gb, genes=genes)
        else: return {"error": f"Bad compare: {txt}"}, "", []
    elif cmd_type == "interactions":
        txt = cmd_str.split(":", 1)[1].strip() if ":" in cmd_str else ""; src, tgt = None, None
        if "->" in txt: p = txt.split("->"); src = p[0].strip() or None; tgt = p[1].strip() if len(p) > 1 else None
        elif txt: src = txt
        payload = engine.interactions(source=src, target=tgt)
    elif cmd_type == "pathway":
        txt = cmd_str.split(":", 1)[1].strip()
        payload = engine.pathway() if txt.lower() == "all" else engine.pathway(pathway_name=txt)
    elif cmd_type == "plot":
        if not adata: return None, None, []
        os.makedirs(plot_dir, exist_ok=True)
        sub, args_str = (cmd_str.split(None, 1) + [""])[:2]; args_str = args_str.strip()
        import matplotlib; matplotlib.use("Agg")
        try:
            if sub == "plot:umap": p = f"{plot_dir}/cell_umap.png"; viz.plot_cell_umap(adata, cluster_key=cluster_key, save_path=p); viz.plt.close("all"); plots.append(p)
            elif sub == "plot:expr": p = f"{plot_dir}/expr_{args_str}.png"; viz.plot_gene_expression_umap(adata, gene=args_str, save_path=p); viz.plt.close("all"); plots.append(p)
            elif sub == "plot:dotplot": genes = [g.strip() for g in args_str.split(",")]; p = f"{plot_dir}/dotplot_{'_'.join(genes[:3])}.png"; viz.plot_dotplot(adata, genes=genes, cluster_key=cluster_key, save_path=p); viz.plt.close("all"); plots.append(p)
            elif sub == "plot:grid": genes = [g.strip() for g in args_str.split(",")]; p = f"{plot_dir}/grid_{genes[0]}.png"; viz.plot_gene_expression_grid(adata, genes=genes, cluster_key=cluster_key, save_path=p); viz.plt.close("all"); plots.append(p)
        except Exception as e: print(f"  [PLOT ERROR] {e}")
        return None, None, plots

    if payload and "error" not in payload:
        mode = payload.get("mode", cmd_type)
        try: answer = ask_llm(llm, SYSTEM_PROMPT, build_prompt(mode, payload.get("query", cmd_str), payload))
        except Exception as e: answer = f"[LLM ERROR] {e}"
    return payload, answer, plots


# ── Evaluation ──
def full_evaluation(report_text, elisa_genes, elisa_interactions, elisa_pathway_scores, elisa_proportions):
    sc = {}
    found = PAPER_GENES & elisa_genes
    gr = len(found) / len(PAPER_GENES) * 100 if PAPER_GENES else 0
    sc["gene_recall"] = f"{gr:.1f}% ({len(found)}/{len(PAPER_GENES)})"

    pf = sum(1 for pp in PAPER_PATHWAYS if any(pp.lower() in k.lower() or k.lower() in pp.lower() for k in elisa_pathway_scores))
    pc = pf / len(PAPER_PATHWAYS) * 100 if PAPER_PATHWAYS else 0
    sc["pathway_coverage"] = f"{pc:.1f}% ({pf}/{len(PAPER_PATHWAYS)})"

    irf = 0
    for s, t, _ in PAPER_INTERACTIONS:
        for ei in elisa_interactions:
            if (s.lower() in ei.get("source", "").lower() or ei.get("source", "").lower() in s.lower()) and \
               (t.lower() in ei.get("target", "").lower() or ei.get("target", "").lower() in t.lower()):
                irf += 1; break
    ir = irf / len(PAPER_INTERACTIONS) * 100 if PAPER_INTERACTIONS else 0
    sc["interaction_recall"] = f"{ir:.1f}% ({irf}/{len(PAPER_INTERACTIONS)})"

    hp = bool(elisa_proportions); sc["proportions_available"] = "Yes" if hp else "No"
    sc["report_words"] = len(report_text.split())

    themes = {
        "PD-1/PD-L1 axis": ["PD-1", "PD-L1", "PDCD1", "CD274", "checkpoint"],
        "T cell exhaustion": ["exhaustion", "TOX", "LAG3", "HAVCR2", "TIGIT"],
        "Effector CD8 cytotoxicity": ["GZMB", "PRF1", "cytotoxic", "effector", "NKG7"],
        "Regulatory T cells": ["FOXP3", "Treg", "regulatory T", "immunosuppressive"],
        "Macrophage polarization": ["CD163", "MRC1", "M2", "macrophage", "TREM2"],
        "NK cell activity": ["NK cell", "KLRD1", "natural killer", "NKG7"],
        "B cell / plasma": ["B cell", "plasma cell", "SDC1", "antibody", "immunoglobulin"],
        "Dendritic cell priming": ["dendritic", "CD80", "CD86", "antigen presentation"],
        "Melanoma markers": ["MITF", "melanoma", "MLANA", "PMEL"],
        "Breast cancer markers": ["EPCAM", "breast cancer", "KRT8", "HER2"],
        "Liver cancer markers": ["ALB", "hepatocellular", "HCC", "GPC3"],
        "Kidney cancer markers": ["CA9", "renal", "ccRCC", "PAX8"],
        "BCC/Hedgehog": ["basal cell carcinoma", "PTCH1", "GLI1", "Hedgehog"],
        "CAF biology": ["fibroblast", "FAP", "ACTA2", "COL1A1"],
        "Immune evasion": ["immune evasion", "CD47", "IDO1", "TGFB1"],
        "Pre vs Post changes": ["pre-treatment", "post-treatment", "Pre", "Post", "ICB"],
        "T cell clonal replacement": ["clonal replacement", "T cell clone", "PD-1 blockade"],
        "TCF4 resistance": ["TCF4", "mesenchymal", "MES", "resistance"],
        "EMT": ["EMT", "epithelial-mesenchymal", "VIM", "SNAI1", "ZEB1"],
        "Tumor proliferation": ["MKI67", "proliferation", "TOP2A", "cell cycle"],
    }
    tf = sum(1 for kws in themes.values() if any(kw.lower() in report_text.lower() for kw in kws))
    tc = tf / len(themes) * 100 if themes else 0
    sc["theme_coverage"] = f"{tc:.1f}% ({tf}/{len(themes)})"
    comp = gr * 0.30 + pc * 0.20 + ir * 0.15 + tc * 0.25 + (10 if hp else 0)
    sc["composite_score"] = f"{comp:.1f}%"
    return {"scorecard": sc, "composite_score": round(comp, 1),
            "details": {"paper_genes": sorted(PAPER_GENES), "found_genes": sorted(found),
                        "missing_genes": sorted(PAPER_GENES - found)}}


# ── Main ──
def main():
    args = setup_args()
    out_dir = args.out_dir; plot_dir = os.path.join(out_dir, "elisa_plots")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)
    engine, llm, adata, report, viz, caps = init_elisa(args)
    queries = get_queries(skip_plots=args.skip_plots)
    session_log, all_payloads, all_interactions = [], [], []

    print(f"\n{'='*70}\nELISA PAPER REPLICATION: ICB Multi-Cancer\nPaper: Gondal et al. (2025) Sci Data 12:139\nRUNNING {len(queries)} QUERIES\n{'='*70}\n")

    for i, (ct, cs) in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {cs}"); t0 = time.time()
        payload, answer, plots = execute_query(ct, cs, engine, llm, adata, viz, args.cluster_key, plot_dir)
        elapsed = time.time() - t0
        if payload is None and ct == "plot":
            print(f"  → {len(plots)} plots ({elapsed:.1f}s)")
            if plots and report.entries: report.entries[-1]["plots"].extend(plots)
            continue
        if payload and "error" in payload: print(f"  [ERROR] {payload['error']}"); continue
        if payload:
            mode = payload.get("mode", ct)
            if mode == "interactions" and "interactions" in payload: all_interactions.extend(payload["interactions"])
            report.add_entry(entry_type=ct, query=payload.get("query", cs), payload=payload, answer=answer or "", plots=plots)
            all_payloads.append(payload)
            session_log.append({"index": i+1, "command": cs, "type": ct, "mode": mode,
                                "answer": answer[:500] if answer else "", "elapsed": round(elapsed, 2)})
            if answer: print(f"  → {mode} | {elapsed:.1f}s\n  {answer[:150]}...")
        print()

    print(f"\n{'='*70}\nGENERATING REPORT\n{'='*70}")
    def llm_fn(p): return ask_llm(llm, SYSTEM_PROMPT, p)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = os.path.join(out_dir, f"elisa_report_ICB_{ts}.md")
    report.generate_markdown(md_path, llm_func=llm_fn)
    docx_path = md_path.replace(".md", ".docx"); report.generate_docx(docx_path, llm_func=llm_fn)
    with open(os.path.join(out_dir, "session_log.json"), "w") as f: json.dump(session_log, f, indent=2, default=str)

    print(f"\n{'='*70}\nRUNNING EVALUATION\n{'='*70}")
    import re
    report_text = open(md_path).read() if os.path.exists(md_path) else ""
    elisa_genes = set()
    for p in all_payloads:
        for r in p.get("results", []):
            for g in r.get("gene_evidence", []) + r.get("genes", []):
                elisa_genes.add(g.get("gene", "") if isinstance(g, dict) else g)
        for cdata in p.get("clusters", {}).values():
            if isinstance(cdata, dict):
                for g in cdata.get("genes", []):
                    elisa_genes.add(g.get("gene", "") if isinstance(g, dict) else g)
        if p.get("mode") in ("pathway_scoring", "pathway_query"):
            for pw in p.get("pathways", {}).values():
                for s in pw.get("scores", []):
                    for tg in s.get("top_genes", []):
                        elisa_genes.add(tg.get("gene", "") if isinstance(tg, dict) else tg)
        if p.get("mode") == "pathway_query":
            for g in p.get("genes_in_pathway", []): elisa_genes.add(g)
    for e in session_log:
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,12}(?:-[A-Z0-9]+)?)\b', e.get("answer", "")):
            elisa_genes.add(m.group(1))
    elisa_genes.discard("")

    pathway_scores = {}
    for p in all_payloads:
        if p.get("mode") == "pathway_scoring": pathway_scores.update(p.get("pathways", {}))
        elif p.get("mode") == "pathway_query" and p.get("pathway"):
            pathway_scores[p["pathway"]] = {"scores": p.get("scores", []), "genes_in_pathway": p.get("genes_in_pathway", [])}

    proportion_data = next((p for p in all_payloads if p.get("mode") == "proportions"), {})

    ev = full_evaluation(report_text, elisa_genes, all_interactions, pathway_scores, proportion_data)
    print(f"\n{'='*70}\nEVALUATION SCORECARD\n{'='*70}")
    for k, v in ev["scorecard"].items(): print(f"  {k}: {v}")
    print(f"\n  ★ COMPOSITE SCORE: {ev['composite_score']}%\n{'='*70}")
    with open(os.path.join(out_dir, "evaluation_scorecard.json"), "w") as f: json.dump(ev, f, indent=2, default=str)
    print("DONE")

if __name__ == "__main__":
    main()
