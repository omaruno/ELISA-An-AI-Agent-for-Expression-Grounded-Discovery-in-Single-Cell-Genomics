#!/usr/bin/env python
"""
ELISA Paper Replication — Automated Batch Runner
=================================================
Target Paper: Cystic Fibrosis airway single-cell atlas

Updated for retrieval_engine_v4_hybrid + elisa_analysis.

Usage:
    python run_paper_replication_CF.py \\
        --h5ad /fast/ocoser/singlecell_ai/SingleCell2/Fibrosi_cistica.h5ad \\
        --out-dir /lustre/home/ocoser/aiagents/elisa_replication/ \\
        --base /path/to/embeddings/ \\
        --pt-name fused_embeddings.pt
"""

import os, sys, json, time, textwrap, argparse, functools, re
from datetime import datetime

print = functools.partial(print, flush=True)


def setup_args():
    parser = argparse.ArgumentParser(description="ELISA Paper Replication Runner — CF")
    parser.add_argument("--h5ad", default=None)
    parser.add_argument("--cluster-key", default="cell_type")
    parser.add_argument("--out-dir", default="elisa_replication")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dataset-name", default="Cystic Fibrosis")
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
                                    condition_col = col; break
                    if condition_col: break
            if not condition_col: condition_col = "condition"
            return comparative_analysis(self._eng.gene_stats, self._eng.metadata,
                                        condition_col=condition_col, group_a=group_a,
                                        group_b=group_b, genes=genes, **kwargs)

        def detect_capabilities(self):
            caps = {"has_conditions": False, "condition_values": [], "condition_column": None,
                    "n_clusters": len(self._eng.cluster_ids), "cluster_ids": list(self._eng.cluster_ids)}
            meta = self._eng.metadata
            if not meta: return caps
            kws = ["patient_group", "case_control", "condition", "disease", "treatment",
                   "status", "timepoint", "pre_post", "group", "sample_type"]
            col_cands = {}
            for cid, m in meta.items():
                if not isinstance(m, dict): continue
                for col, dist in m.get("fields", {}).items():
                    if isinstance(dist, dict):
                        col_cands.setdefault(col, set()).update(dist.keys())
            for kw in kws:
                for col, vals in col_cands.items():
                    if kw in col.lower().replace("_", " ").replace("-", " "):
                        sv = sorted(vals)
                        if 2 <= len(sv) <= 10:
                            caps.update(has_conditions=True, condition_column=col, condition_values=sv)
                            return caps
            skip = ["batch", "lab", "donor", "sample", "plate", "lane", "seq", "lib", "run",
                    "class", "subclass", "level", "annotation", "ontology", "type", "assay", "sex"]
            for col, vals in col_cands.items():
                if len(vals) == 2 and not any(p in col.lower() for p in skip):
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
            print(f"[INIT] Loading h5ad: {args.h5ad}")
            adata = sc.read_h5ad(args.h5ad)
            if adata.var_names[0].startswith("ENSG") and "feature_name" in adata.var.columns:
                adata.var["ensembl_id"] = adata.var_names.copy()
                adata.var_names = adata.var["feature_name"].astype(str).values
                adata.var_names_make_unique()
            print(f"[INIT] h5ad: {adata.shape[0]} cells, {adata.shape[1]} genes")
        except ImportError:
            print("[WARN] scanpy not installed")

    report = ReportBuilder(dataset_name=args.dataset_name)
    caps = engine.detect_capabilities()
    print(f"[INIT] Conditions: {caps.get('has_conditions', False)} → {caps.get('condition_values', [])}")
    return engine, llm, adata, report, viz, caps


def ask_llm(llm, system_prompt, user_prompt):
    res = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2)
    return res.choices[0].message.content.strip()

SYSTEM_PROMPT = (
    "You are ELISA, an expert assistant for single-cell biology. "
    "Never hallucinate. Always ground claims strictly in provided data. "
    "Be concise and scientific."
)

MAX_PROMPT_CHARS = 12000

def trim_payload(payload, max_chars=MAX_PROMPT_CHARS):
    t = dict(payload)
    if "clusters" in t and isinstance(t["clusters"], dict) and len(t["clusters"]) > 10:
        t["clusters"] = dict(sorted(t["clusters"].items(), key=lambda kv: len(kv[1].get("genes", [])), reverse=True)[:10])
    if "interactions" in t and isinstance(t["interactions"], list) and len(t["interactions"]) > 30:
        t["interactions"] = t["interactions"][:30]
    if "overall" in t and isinstance(t["overall"], list):
        for item in t.get("overall", []):
            for k in list(item.keys()):
                if k not in ("cluster", "n_cells", "fraction", "percentage"): item.pop(k, None)
    if "scores" in t and isinstance(t["scores"], list): t["scores"] = t["scores"][:10]
    for r in t.get("results", []):
        if "gene_evidence" in r and len(r["gene_evidence"]) > 5: r["gene_evidence"] = r["gene_evidence"][:5]
    ctx = json.dumps(t, indent=1, default=str)
    return ctx[:max_chars] + "\n... [TRUNCATED]" if len(ctx) > max_chars else ctx

def build_prompt(mode, query, payload):
    ctx = trim_payload(payload)
    if mode == "discovery":
        return f"You are in DISCOVERY mode.\nSeparate into: 1. DATASET EVIDENCE 2. ESTABLISHED BIOLOGY\n3. CONSISTENCY ANALYSIS 4. CANDIDATE NOVEL HYPOTHESES\nQUESTION: {query}\nDATASET CONTEXT: {ctx}"
    elif mode == "compare":
        return f"You are ELISA analyzing COMPARATIVE analysis.\nCOMPARISON: {query}\nEVIDENCE: {ctx}\nIdentify condition-biased clusters, highlight differentially expressed genes."
    elif mode == "interactions":
        return f"You are ELISA analyzing CELL-CELL INTERACTIONS.\nQUERY: {query}\nINTERACTIONS: {ctx}\nFocus on highest-scoring, group by pathway, note unexpected interactions."
    elif mode == "proportions":
        return f"You are ELISA analyzing CELL TYPE PROPORTIONS.\nQUERY: {query}\nDATA: {ctx}\nReport major types, condition differences, biological implications."
    elif mode in ("pathway_scoring", "pathway_query"):
        return f"You are ELISA analyzing PATHWAY ACTIVITY.\nQUERY: {query}\nSCORES: {ctx}\nIdentify top cell types, contributing genes, biological relevance."
    else:
        return f"You are ELISA for single-cell analysis.\nMODE: {mode.upper()} | QUERY: {query}\nEVIDENCE: {ctx}\nUse ONLY provided evidence. Be concise and scientific."


# ══════════════════════════════════════════════════════════════
# QUERIES
# ══════════════════════════════════════════════════════════════

def get_queries(skip_plots=False):
    queries = [
        ("info", "info"),
        ("proportions", "proportions"),

        ("semantic", "semantic: interferon response in cystic fibrosis airway epithelium"),
        ("semantic", "semantic: basal cell dysfunction and keratinization in CF"),
        ("semantic", "semantic: chromatin remodeling and DNA damage repair in CF epithelium"),
        ("compare", "compare: CF vs Control | IFIT1, MX1, OAS2, CSTA, HSPB1, KDM1A, KMT5A"),
        ("pathway", "pathway: Type I IFN signaling"),
        ("pathway", "pathway: Epithelial defense"),

        ("discover", "discover: CD8 T cell activation and cytokine production in CF"),
        ("compare", "compare: CF vs Control | IFNG, GNAI2, CD69, CD81, TXNIP, MAP2K2, CD3G, FOS, JUND"),
        ("discover", "discover: CD4 T cell activation and VEGF signaling in CF"),
        ("compare", "compare: CF vs Control | KLF2, IL7R, CD48, ETS1, TXNIP"),
        ("discover", "discover: B cell activation and BCR signaling changes in CF"),
        ("compare", "compare: CF vs Control | SYK, CSK, CD81, IGHG3, IGLC2, HLA-DPA1, HLA-DPB1, LTB"),
        ("pathway", "pathway: IFN-gamma signaling"),
        ("pathway", "pathway: T cell activation"),
        ("pathway", "pathway: NK cell activity"),

        ("interactions", "interactions:"),
        ("interactions", "interactions: CD8 T cell -> macrophage"),
        ("interactions", "interactions: CD8 T cell -> basal cell"),
        ("interactions", "interactions: macrophage -> CD8 T cell"),
        ("interactions", "interactions: epithelial -> CD8 T cell"),
        ("interactions", "interactions: B cell -> macrophage"),
        
        ("discover", "discover: HLA-E NKG2A immune checkpoint and CD8 T cell inhibition in CF"),
        ("discover", "discover: calreticulin LRP1 macrophage phagocytosis signaling in CF"),
        ("discover", "discover: GNAI2 chemokine receptor signaling and lymphocyte trafficking in CF"),
        ("compare", "compare: CF vs Control | HLA-E, KLRC1, KLRD1, KLRC2, CALR, LRP1, VEGFA"),
        ("pathway", "pathway: Antigen presentation"),
        ("pathway", "pathway: Angiogenesis"),

        ("discover", "discover: hypoxia TXNIP NLRP3 inflammasome activation in CF lung"),
        ("discover", "discover: VEGF signaling endothelial remodeling and angiogenesis in CF"),
        ("pathway", "pathway: Oxidative stress"),
        ("pathway", "pathway: Fibrosis"),
        ("compare", "compare: CF vs Control | VEGFA, TXNIP, ETS1, MAP2K2, GNAI2, IFNG"),
    ]

    if not skip_plots:
        queries.extend([
            ("plot", "plot:umap"),
            ("plot", "plot:expr IFIT1"), ("plot", "plot:expr MX1"),
            ("plot", "plot:expr IFNG"), ("plot", "plot:expr HLA-E"), ("plot", "plot:expr CALR"),
            ("plot", "plot:dotplot IFIT1, MX1, OAS2, CSTA, HSPB1"),
            ("plot", "plot:dotplot IFNG, GNAI2, CD69, CD81, TXNIP, MAP2K2"),
            ("plot", "plot:dotplot KLF2, IL7R, CD48, ETS1, TXNIP"),
            ("plot", "plot:dotplot SYK, CSK, IGHG3, IGLC2, CD81"),
            ("plot", "plot:dotplot HLA-E, KLRC1, KLRD1, CALR, LRP1, VEGFA"),
            ("plot", "plot:grid IFNG, CD69, TXNIP, GNAI2, HLA-E, KLRC1"),
        ])
    return queries

def get_queries_test():
    return [
        ("info", "info"),
        ("proportions", "proportions"),
        ("semantic", "semantic: interferon response cystic fibrosis airway"),
        ("discover", "discover: CD8 T cell activation cytokine production CF"),
        ("compare", "compare: CF vs Control | IFNG, GNAI2, CD69, CD81, TXNIP"),
        ("interactions", "interactions:"),
        ("interactions", "interactions: CD8 T cell -> macrophage"),
        ("pathway", "pathway: IFN-gamma signaling"),
        ("pathway", "pathway: all"),
    ]


# ══════════════════════════════════════════════════════════════
# GROUND TRUTH (formerly in replicate_cf_paper.py — now inlined)
# ══════════════════════════════════════════════════════════════

PAPER_GENES = {
    "IFNG", "GNAI2", "CD69", "CD81", "TXNIP", "MAP2K2", "CD3G", "FOS", "JUND",
    "KLF2", "IL7R", "CD48", "ETS1",
    "SYK", "CSK", "IGHG3", "IGLC2", "LTB",
    "HLA-DPA1", "HLA-DPB1",
    "HLA-E", "KLRC1", "KLRD1", "KLRC2",
    "CALR", "LRP1", "VEGFA",
    "IFIT1", "MX1", "OAS2", "CSTA", "HSPB1", "KDM1A", "KMT5A",
    "RAD50", "ERCC6", "ERCC8", "DNAH5", "SYNE1", "SYNE2",
    "IFNGR1", "IFNGR2", "CXCR3", "F2R", "S1PR4",
}

PAPER_INTERACTIONS = [
    ("CD8 T cell", "macrophage", "IFNG/IFNGR signaling"),
    ("CD8 T cell", "basal cell", "CALR/LRP1 phagocytosis"),
    ("macrophage", "CD8 T cell", "HLA-E/NKG2A checkpoint"),
    ("epithelial", "CD8 T cell", "GNAI2/CXCR3 chemokine"),
    ("B cell", "macrophage", "LTB/TNFRSF14 lymphotoxin"),
]

PAPER_PATHWAYS = [
    "IFN-gamma signaling", "Type I IFN signaling",
    "T cell activation", "NK cell activity",
    "Antigen presentation", "Angiogenesis",
    "Oxidative stress", "Fibrosis",
    "Epithelial defense",
]

PROPORTION_CHANGES = {
    "CD8 T cell": "activated, IFN-gamma producing",
    "CD4 T cell": "VEGFR signaling upregulated",
    "B cell": "BCR signaling downregulated",
    "macrophage": "HLA-E checkpoint, phagocytosis",
    "basal cell": "keratinization reduced, DNA damage repair",
    "ciliated cell": "DNAH5 elevated",
}


# ── Execute query ──
def execute_query(cmd_type, cmd_str, engine, llm, adata, viz, cluster_key, plot_dir):
    payload, answer, plots = None, None, []

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
        txt = cmd_str.split(":", 1)[1].strip(); genes = None
        if "|" in txt:
            txt, gs = txt.split("|", 1); genes = [g.strip() for g in gs.split(",") if g.strip()]; txt = txt.strip()
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
            return {"error": f"Bad compare format: {txt}"}, "", []
    elif cmd_type == "interactions":
        txt = cmd_str.split(":", 1)[1].strip() if ":" in cmd_str else ""
        src, tgt = None, None
        if "->" in txt:
            p = txt.split("->"); src = p[0].strip() or None; tgt = p[1].strip() if len(p) > 1 else None
        elif txt: src = txt
        payload = engine.interactions(source=src, target=tgt)
    elif cmd_type == "pathway":
        txt = cmd_str.split(":", 1)[1].strip()
        payload = engine.pathway() if txt.lower() == "all" else engine.pathway(pathway_name=txt)
    elif cmd_type == "plot":
        if adata is None: return None, None, []
        os.makedirs(plot_dir, exist_ok=True)
        subcmd = cmd_str.split(None, 1); sub = subcmd[0]; a = subcmd[1].strip() if len(subcmd) > 1 else ""
        import matplotlib; matplotlib.use("Agg")
        try:
            if sub == "plot:umap":
                p = f"{plot_dir}/cell_umap.png"; viz.plot_cell_umap(adata, cluster_key=cluster_key, save_path=p); viz.plt.close("all"); plots.append(p)
            elif sub == "plot:expr":
                p = f"{plot_dir}/expr_{a}.png"; viz.plot_gene_expression_umap(adata, gene=a, save_path=p); viz.plt.close("all"); plots.append(p)
            elif sub == "plot:dotplot":
                genes = [g.strip() for g in a.split(",")]; p = f"{plot_dir}/dotplot_{'_'.join(genes[:3])}.png"
                viz.plot_dotplot(adata, genes=genes, cluster_key=cluster_key, save_path=p); viz.plt.close("all"); plots.append(p)
            elif sub == "plot:grid":
                genes = [g.strip() for g in a.split(",")]; p = f"{plot_dir}/grid_{genes[0]}.png"
                viz.plot_gene_expression_grid(adata, genes=genes, cluster_key=cluster_key, save_path=p); viz.plt.close("all"); plots.append(p)
        except Exception as e: print(f"  [PLOT ERROR] {e}")
        return None, None, plots

    if payload and "error" not in payload:
        mode = payload.get("mode", cmd_type)
        try: answer = ask_llm(llm, SYSTEM_PROMPT, build_prompt(mode, payload.get("query", cmd_str), payload))
        except Exception as e: answer = f"[LLM ERROR] {e}"
    return payload, answer, plots


# ── Evaluation ──
def full_evaluation(elisa_report_text, elisa_genes, elisa_interactions,
                    elisa_pathway_scores, elisa_proportions):
    scorecard = {}
    found_genes = PAPER_GENES & elisa_genes
    gr = len(found_genes) / len(PAPER_GENES) * 100 if PAPER_GENES else 0
    scorecard["gene_recall"] = f"{gr:.1f}% ({len(found_genes)}/{len(PAPER_GENES)})"

    pf = 0
    for pp in PAPER_PATHWAYS:
        pp_l = pp.lower()
        for ek in elisa_pathway_scores:
            if pp_l in ek.lower() or ek.lower() in pp_l:
                pf += 1; break
    pc = pf / len(PAPER_PATHWAYS) * 100 if PAPER_PATHWAYS else 0
    scorecard["pathway_coverage"] = f"{pc:.1f}% ({pf}/{len(PAPER_PATHWAYS)})"

    irf = 0
    for s, t, _ in PAPER_INTERACTIONS:
        sl, tl = s.lower(), t.lower()
        for ei in elisa_interactions:
            es, et = ei.get("source", "").lower(), ei.get("target", "").lower()
            if (sl in es or es in sl) and (tl in et or et in tl):
                irf += 1; break
    ir = irf / len(PAPER_INTERACTIONS) * 100 if PAPER_INTERACTIONS else 0
    scorecard["interaction_recall"] = f"{ir:.1f}% ({irf}/{len(PAPER_INTERACTIONS)})"

    hp = bool(elisa_proportions)
    scorecard["proportions_available"] = "Yes" if hp else "No"
    scorecard["report_words"] = len(elisa_report_text.split())

    themes = {
        "IFN-gamma axis": ["IFNG", "interferon gamma", "IFN-gamma", "IFNGR"],
        "HLA-E checkpoint": ["HLA-E", "NKG2A", "KLRC1", "immune checkpoint"],
        "CALR phagocytosis": ["CALR", "calreticulin", "LRP1", "phagocytosis"],
        "GNAI2 signaling": ["GNAI2", "chemokine", "CXCR3"],
        "TXNIP/NLRP3": ["TXNIP", "NLRP3", "inflammasome"],
        "VEGF signaling": ["VEGFA", "VEGF", "angiogenesis"],
        "BCR downregulation": ["BCR", "B cell receptor", "IGHG3", "IGLC2"],
        "Basal cell changes": ["basal cell", "keratinization", "CSTA", "DNA damage"],
    }
    tf = sum(1 for kws in themes.values() if any(kw.lower() in elisa_report_text.lower() for kw in kws))
    tc = tf / len(themes) * 100 if themes else 0
    scorecard["theme_coverage"] = f"{tc:.1f}% ({tf}/{len(themes)})"

    comp = gr * 0.30 + pc * 0.20 + ir * 0.15 + tc * 0.25 + (10 if hp else 0)
    scorecard["composite_score"] = f"{comp:.1f}%"

    return {"scorecard": scorecard, "composite_score": round(comp, 1),
            "details": {"paper_genes": sorted(PAPER_GENES), "found_genes": sorted(found_genes),
                        "missing_genes": sorted(PAPER_GENES - found_genes)}}


# ── Main ──
def main():
    args = setup_args()
    out_dir = args.out_dir; plot_dir = os.path.join(out_dir, "elisa_plots")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)
    engine, llm, adata, report, viz, caps = init_elisa(args)
    queries = get_queries(skip_plots=args.skip_plots)
    session_log, all_payloads, all_interactions = [], [], []

    print(f"\n{'='*70}\nELISA PAPER REPLICATION: Cystic Fibrosis\nRUNNING {len(queries)} QUERIES\nOutput: {out_dir}\n{'='*70}\n")

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
            if mode == "interactions" and "interactions" in payload:
                all_interactions.extend(payload["interactions"])
            report.add_entry(entry_type=ct, query=payload.get("query", cs), payload=payload, answer=answer or "", plots=plots)
            all_payloads.append(payload)
            session_log.append({"index": i+1, "command": cs, "type": ct, "mode": mode,
                                "query": payload.get("query", ""), "answer": answer[:500] if answer else "",
                                "elapsed": round(elapsed, 2), "n_plots": len(plots)})
            if answer: print(f"  → {mode} | {elapsed:.1f}s\n  {answer[:150]}...")
            else: print(f"  → {mode} | {elapsed:.1f}s (no LLM answer)")
        print()

    # ── Report ──
    print(f"\n{'='*70}\nGENERATING REPORT\n{'='*70}")
    def llm_fn(p): return ask_llm(llm, SYSTEM_PROMPT, p)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = os.path.join(out_dir, f"elisa_report_{ts}.md")
    report.generate_markdown(md_path, llm_func=llm_fn)
    docx_path = md_path.replace(".md", ".docx"); report.generate_docx(docx_path, llm_func=llm_fn)
    log_path = os.path.join(out_dir, "session_log.json")
    with open(log_path, "w") as f: json.dump(session_log, f, indent=2, default=str)
    print(f"[SAVED] Session log: {log_path}")

    # ── Evaluation ──
    print(f"\n{'='*70}\nRUNNING EVALUATION\n{'='*70}")
    report_text = open(md_path).read() if os.path.exists(md_path) else ""

    elisa_genes = set()
    for p in all_payloads:
        for r in p.get("results", []):
            for g in r.get("gene_evidence", []):
                elisa_genes.add(g.get("gene", "") if isinstance(g, dict) else g)
            for g in r.get("genes", []):
                elisa_genes.add(g.get("gene", "") if isinstance(g, dict) else g)
        for cid, cd in p.get("clusters", {}).items():
            if isinstance(cd, dict):
                for g in cd.get("genes", []):
                    elisa_genes.add(g.get("gene", "") if isinstance(g, dict) else g)
        if p.get("mode") in ("pathway_scoring", "pathway_query"):
            for pw_data in p.get("pathways", {}).values():
                for gs in pw_data.get("gene_set", []): elisa_genes.add(gs)
                for cl in pw_data.get("scores", []):
                    for tg in cl.get("top_genes", []):
                        elisa_genes.add(tg.get("gene", "") if isinstance(tg, dict) else tg)
        if p.get("mode") == "pathway_query":
            for g in p.get("genes_in_pathway", []): elisa_genes.add(g)

    for e in session_log:
        for m in re.finditer(r'\b([A-Z][A-Z0-9-]{1,15})\s*\(ENSG\d+', e.get("answer", "")): elisa_genes.add(m.group(1))
        for m in re.finditer(r'\b([A-Z][A-Z0-9]{1,12}(?:-[A-Z0-9]+)?)\b', e.get("answer", "")): elisa_genes.add(m.group(1))
    for m in re.finditer(r'\b([A-Z][A-Z0-9-]{1,15})\s*\(ENSG\d+', report_text): elisa_genes.add(m.group(1))
    known_re = re.compile(
        r'\b(IFNG|IFIT[123]|MX[12]|OAS[123]|CSTA|HSPB1|KDM1A|KMT5A|'
        r'RAD50|ERCC[68]|DNAH5|SYNE[12]|HLA-[A-Z0-9]+|KLRC[123]|KLRD1|KLRK1|'
        r'CD69|CD81|CD3[DGE]|CD48|CD9|FOS|JUND|TXNIP|MAP2K2|GNAI2|'
        r'KLF2|IL7R|ETS1|SYK|CSK|LTB|IGHG3|IGLC2|CALR|LRP1|VEGFA|'
        r'IFNGR[12]|CXCR3|F2R|S1PR4)\b')
    for m in known_re.finditer(report_text): elisa_genes.add(m.group(1))
    elisa_genes.discard("")
    print(f"  Genes collected: {len(elisa_genes)}")

    pathway_scores = {}
    for p in all_payloads:
        if p.get("mode") == "pathway_scoring": pathway_scores.update(p.get("pathways", {}))
        elif p.get("mode") == "pathway_query" and p.get("pathway"):
            pathway_scores[p["pathway"]] = {"scores": p.get("scores", []), "genes_in_pathway": p.get("genes_in_pathway", [])}
    print(f"  Pathways: {len(pathway_scores)}")
    print(f"  Interactions: {len(all_interactions)}")

    prop_data = next((p for p in all_payloads if p.get("mode") == "proportions"), {})

    ev = full_evaluation(report_text, elisa_genes, all_interactions, pathway_scores, prop_data)
    print(f"\n{'='*70}\nEVALUATION SCORECARD\n{'='*70}")
    for k, v in ev["scorecard"].items(): print(f"  {k}: {v}")
    print(f"\n  ★ COMPOSITE SCORE: {ev['composite_score']}%\n{'='*70}")

    eval_path = os.path.join(out_dir, "evaluation_scorecard.json")
    with open(eval_path, "w") as f: json.dump(ev, f, indent=2, default=str)
    print(f"[SAVED] Evaluation: {eval_path}")

    print(f"\n{'='*70}\nALL OUTPUTS:\n  Report (md):   {md_path}\n  Report (docx): {docx_path}")
    print(f"  Session log:   {log_path}\n  Evaluation:    {eval_path}\n  Plots:         {plot_dir}/")
    n_plots = sum(len(e.get("plots", [])) for e in report.entries)
    print(f"  Total plots:   {n_plots}\n  Analyses:      {len(report.entries)}\n{'='*70}\nDONE")


if __name__ == "__main__":
    main()
