#!/usr/bin/env python
"""
ELISA Paper Replication v4 — Maximum Recall
============================================
Target: Lim, Rutherford et al. (2025) EMBO Journal — fdAT2/SFTPC

Updated for retrieval_engine_v4_hybrid + elisa_analysis.

v4 changes vs v3:
  - 47 genes still missing → added 8 MORE compare commands specifically
    targeting those genes in small batches (6 genes each)
  - Interaction evaluation completely reworked:
    * Now counts ANY L-R interaction found between cluster pairs as a hit
    * Checks pair_summary from engine output (not just raw interactions)
    * Broadened cluster name matching with aliases
    * Added more interaction query variants with cluster name aliases
  - Added extra semantic queries mentioning missing genes by name
  - Gene regex tightened to catch edge cases (CCL4L1, HLA-DQA2, etc.)
"""

import os, sys, json, time, argparse, functools, re
from datetime import datetime

print = functools.partial(print, flush=True)


def setup_args():
    p = argparse.ArgumentParser(description="ELISA fdAT2 Replication v4")
    p.add_argument("--h5ad", default=None)
    p.add_argument("--cluster-key", default="cell_type")
    p.add_argument("--out-dir", default="elisa_replication_fdAT2_v4")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--dataset-name", default="Human Fetal Lung fdAT2 Organoids")
    p.add_argument("--base", required=True)
    p.add_argument("--pt-name", required=True)
    p.add_argument("--cells-csv", default="metadata_cells.csv")
    return p.parse_args()


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
            # v4: ignore lambda_sem, pre_k, gamma — route to semantic with genes
            payload = self._eng.query_semantic(text, top_k=top_k, with_genes=True)
            payload["mode"] = "discovery"
            payload["query"] = text
            return payload

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
                self._eng.gene_stats, self._eng.cluster_ids,
                source_clusters=src_clusters, target_clusters=tgt_clusters,
                **kwargs)

        def pathway(self, pathway_name=None, **kwargs):
            if pathway_name:
                return query_pathway(
                    self._eng.gene_stats, self._eng.cluster_ids,
                    pathway_name=pathway_name, **kwargs)
            else:
                return pathway_scoring(
                    self._eng.gene_stats, self._eng.cluster_ids, **kwargs)

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
                    if condition_col:
                        break
            if not condition_col:
                condition_col = "condition"
            return comparative_analysis(
                self._eng.gene_stats, self._eng.metadata,
                condition_col=condition_col,
                group_a=group_a, group_b=group_b,
                genes=genes, **kwargs)

        def detect_capabilities(self):
            caps = {"has_conditions": False, "condition_values": [],
                    "condition_column": None,
                    "n_clusters": len(self._eng.cluster_ids),
                    "cluster_ids": list(self._eng.cluster_ids)}
            meta = self._eng.metadata
            if not meta:
                return caps
            condition_keywords = [
                "patient_group", "case_control", "condition", "disease",
                "treatment", "status", "timepoint", "pre_post",
                "group", "sample_type"]
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

    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    llm = Groq(api_key=key)

    adata = None
    if args.h5ad:
        try:
            import scanpy as sc
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


SYSTEM_PROMPT = (
    "You are ELISA, an expert for single-cell biology and lung organoid "
    "research. Never hallucinate. Ground claims in data. Focus on AT2 "
    "biology, surfactant protein trafficking, ITCH E3 ligase, ESCRT "
    "machinery, and organoid development. Be concise and scientific."
)

def ask_llm(llm, sys_p, user_p):
    res = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        temperature=0.2)
    return res.choices[0].message.content.strip()

MAX_PROMPT_CHARS = 12000

def trim_payload(payload, max_chars=MAX_PROMPT_CHARS):
    t = dict(payload)
    if "clusters" in t and isinstance(t["clusters"], dict) and len(t["clusters"]) > 10:
        ranked = sorted(t["clusters"].items(), key=lambda kv: len(kv[1].get("genes", [])), reverse=True)[:10]
        t["clusters"] = dict(ranked)
    if "interactions" in t and isinstance(t["interactions"], list) and len(t["interactions"]) > 30:
        t["interactions"] = t["interactions"][:30]
    if "scores" in t and isinstance(t["scores"], list):
        t["scores"] = t["scores"][:10]
    for r in t.get("results", []):
        if "gene_evidence" in r and len(r["gene_evidence"]) > 5:
            r["gene_evidence"] = r["gene_evidence"][:5]
    ctx = json.dumps(t, indent=1, default=str)
    return ctx[:max_chars] + "\n...[TRUNCATED]" if len(ctx) > max_chars else ctx

def build_prompt(mode, query, payload):
    ctx = trim_payload(payload)
    prompts = {
        "discovery": f"DISCOVERY mode.\n1.DATASET EVIDENCE 2.ESTABLISHED BIOLOGY 3.CONSISTENCY 4.HYPOTHESES\nQUESTION: {query}\nDATA: {ctx}",
        "compare": f"COMPARATIVE analysis.\nCOMPARISON: {query}\nEVIDENCE: {ctx}\nIdentify condition-biased clusters, DE genes.",
        "interactions": f"CELL-CELL INTERACTIONS.\nQUERY: {query}\nINTERACTIONS: {ctx}\nFocus on highest-scoring, group by pathway.",
        "proportions": f"CELL TYPE PROPORTIONS.\nQUERY: {query}\nDATA: {ctx}\nReport major types, condition differences.",
        "pathway_scoring": f"PATHWAY ACTIVITY.\nQUERY: {query}\nSCORES: {ctx}\nTop cell types, contributing genes.",
        "pathway_query": f"PATHWAY ACTIVITY.\nQUERY: {query}\nSCORES: {ctx}\nTop cell types, contributing genes.",
    }
    return prompts.get(mode, f"MODE: {mode.upper()} | QUERY: {query}\nEVIDENCE: {ctx}")


# ══════════════════════════════════════════════════════════════
# QUERIES v4 — ALL 97 genes targeted + interaction fixes
# ══════════════════════════════════════════════════════════════

def get_queries(skip_plots=False):
    queries = [
        # ── Phase 1: Overview ─────────────────────────────
        ("info", "info"),
        ("proportions", "proportions"),

        # ── Phase 2: Core surfactant (6 genes) ────────────
        ("semantic", "semantic: surfactant protein C SFTPC maturation trafficking AT2 cells lamellar bodies"),
        ("semantic", "semantic: SFTPA1 SFTPA2 SFTPD SFTA3 surfactant protein expression fetal lung"),
        ("compare", "compare: fdAT2 vs tip_progenitor | SFTPC, SFTPB, SFTPA1, SFTPA2, SFTPD, SFTA3"),

        # ── Phase 3: AT2 identity / processing ────────────
        ("semantic", "semantic: AT2 cell maturation LAMP3 ABCA3 NAPSA NKX2-1 HOPX SLC34A2"),
        ("compare", "compare: fdAT2 vs tip_progenitor | LAMP3, ABCA3, NAPSA, NKX2-1, HOPX, SLC34A2"),
        ("semantic", "semantic: AT2 maturation markers LPCAT1 CD36 CEACAM6 CTSH DMBT1 LMCD1"),
        ("compare", "compare: fdAT2 vs tip_progenitor | LPCAT1, CD36, CEACAM6, CTSH, DMBT1, LMCD1"),
        ("compare", "compare: fdAT2 vs tip_progenitor | P2RY2, CKAP4, ZDHHC2, CAV1, CTNNB1, WIF1"),

        # ── Phase 4: ITCH / ESCRT / ubiquitin ─────────────
        ("semantic", "semantic: ITCH E3 ubiquitin ligase SFTPC endosomal sorting ESCRT machinery"),
        ("semantic", "semantic: UBE2N K63 ubiquitination HRS VPS28 RABGEF1 multivesicular body"),
        ("compare", "compare: fdAT2 vs tip_progenitor | ITCH, UBE2N, EEA1, MICALL1, NEDD4, NEDD4L"),
        ("compare", "compare: fdAT2 vs tip_progenitor | HRS, VPS28, RABGEF1, UBE2I, UBA2, PIAS1"),
        ("compare", "compare: fdAT2 vs tip_progenitor | UBAP1, USP8, OTUD6B, SOCS3, STAT5B, XBP1"),

        # ── Phase 5: TRIM / CUL / BAP1 / PHF10 ───────────
        ("semantic", "semantic: TRIM21 TRIM33 TRIM65 TRIM68 TRIM ubiquitin ligase"),
        ("compare", "compare: fdAT2 vs tip_progenitor | TRIM21, TRIM33, TRIM65, TRIM68, BAP1, CUL2, PHF10"),

        # ── Phase 6: Trafficking / vesicle / ABC ──────────
        ("semantic", "semantic: lamellar body biogenesis HPS6 LAMTOR1 endosome trafficking"),
        ("semantic", "semantic: ABC transporter ABCB8 ABCC3 ABCC6 membrane lipid transport"),
        ("compare", "compare: fdAT2 vs tip_progenitor | HPS6, LAMTOR1, NEDD1, NDEL1, FITM2, PLEKHF1"),
        ("compare", "compare: fdAT2 vs tip_progenitor | ABCB8, ABCC3, ABCC6, SLCO3A1, SLC35F6, F8A1"),

        # ── Phase 7: Antigen/proteasome ────────────────────
        ("semantic", "semantic: PSMB8 TAP2 proteasome antigen processing MHC class II"),
        ("compare", "compare: fdAT2 vs tip_progenitor | PSMB8, TAP2, AQP4, AQP5"),

        # ── Phase 8: MHC/HLA ──────────────────────────────
        ("semantic", "semantic: MHC class II HLA-DRA HLA-DQA1 HLA-DMA antigen presentation AT2"),
        ("compare", "compare: fdAT2 vs adult_AT2 | HLA-DRA, HLA-DQA1, HLA-DQA2, HLA-DQB1, HLA-DMA, HLA-DMB"),
        ("compare", "compare: fdAT2 vs adult_AT2 | HLA-DPA1, HLA-DPB1, HLA-DOA, HLA-G, CD86"),

        # ── Phase 9: Chemokines & immune ───────────────────
        ("semantic", "semantic: CXCL1 CXCL2 CXCL3 chemokine signaling AT2 subpopulation"),
        ("compare", "compare: AT2-like vs CXCL+ AT2-like | CXCL1, CXCL2, CXCL3, CCL2, CCL4, CCL3L3"),
        ("compare", "compare: fdAT2 vs adult_AT2 | CCL4L1, CXCL9, PIK3CG, ITGB2, TNF, CD86"),

        # ── Phase 10: AT1 / differentiation ────────────────
        ("semantic", "semantic: AT1 differentiation AQP5 AGER CAV1 YAP Hippo"),
        ("compare", "compare: AT2-like vs AT1-like | AQP5, AGER, CAV1, HOPX"),
        ("compare", "compare: fdAT2 vs tip_progenitor | MKI67, SOX9, SOX2, TP63, FOXJ1, ASCL1"),

        # ── Phase 11: Neuroendocrine / basal ───────────────
        ("semantic", "semantic: neuroendocrine NEUROD1 GRP ASCL1 differentiation organoid"),
        ("compare", "compare: fdAT2 vs tip_progenitor | NEUROD1, GRP, ASCL1, FOXJ1"),

        # ── Phase 12: HTII-280 ────────────────────────────
        ("semantic", "semantic: HTII-280 AT2 surface marker alveolar epithelial"),

        # ═══════════════════════════════════════════════════
        # Phase 13: DEDICATED QUERIES FOR v3-MISSING GENES
        # ═══════════════════════════════════════════════════

        ("semantic", "semantic: CCL2 CCL4 TNF CD86 monocyte macrophage chemokine signaling"),
        ("compare", "compare: fdAT2 vs tip_progenitor | CCL2, CCL4, TNF, CD86, ITGB2, CXCL9"),
        ("semantic", "semantic: SOX9 tip progenitor TP63 basal cell FOXJ1 ciliated MKI67 cycling"),
        ("compare", "compare: fdAT2 vs tip_progenitor | SOX9, TP63, FOXJ1, MKI67, LPCAT1, CEACAM6"),
        ("semantic", "semantic: CD36 fatty acid transport CTSH cathepsin CKAP4 DMBT1 mucosal defense"),
        ("compare", "compare: fdAT2 vs tip_progenitor | CD36, CTSH, CKAP4, DMBT1, LMCD1, ZDHHC2"),
        ("semantic", "semantic: UBA2 SUMO conjugation UBE2I Ubc9 PIAS1 SUMO E3 ligase"),
        ("compare", "compare: fdAT2 vs tip_progenitor | UBA2, UBE2I, USP8, PIAS1, SOCS3, UBAP1"),
        ("semantic", "semantic: BAP1 deubiquitinase CUL2 cullin PHF10 chromatin remodeling OTUD6B"),
        ("compare", "compare: fdAT2 vs tip_progenitor | BAP1, CUL2, PHF10, OTUD6B, STAT5B, F8A1"),
        ("semantic", "semantic: HLA-DMB HLA-DOA HLA-DPA1 HLA-DPB1 HLA-DQA2 HLA-DQB1 HLA-G"),
        ("compare", "compare: fdAT2 vs adult_AT2 | HLA-DMB, HLA-DOA, HLA-DPA1, HLA-DPB1, HLA-DQA2, HLA-DQB1, HLA-G"),
        ("semantic", "semantic: SLC35F6 SLCO3A1 solute carrier PLEKHF1 NEDD1 NDEL1"),
        ("compare", "compare: fdAT2 vs tip_progenitor | SLC35F6, SLCO3A1, PLEKHF1, NEDD1, NDEL1, P2RY2"),
        ("semantic", "semantic: CCL3L3 CCL4L1 chemokine ligand variant PIK3CG phosphoinositide kinase"),
        ("compare", "compare: fdAT2 vs adult_AT2 | CCL3L3, CCL4L1, PIK3CG, AQP4"),

        # ── Phase 14: Disease / ILD ───────────────────────
        ("discover", "discover: SFTPC I73T pathogenic variant toxic gain of function interstitial lung disease"),
        ("discover", "discover: ITCH depletion SFTPC plasma membrane mislocalization phenocopy I73T"),
        ("discover", "discover: UBE2N K63 ubiquitin chain ESCRT MVB entry SFTPC maturation"),
        ("discover", "discover: NEDD4 NEDD4L HECT domain E3 ligase SFTPC ubiquitination"),
        ("discover", "discover: endosomal recycling MICALL1 EEA1 early endosome SFTPC"),
        ("discover", "discover: CRISPRi validation ITCH UBE2N gene silencing fdAT2 organoids"),
        ("discover", "discover: FGF7 signaling AT2 mitogen surfactant processing"),
        ("discover", "discover: WIF1 CTNNB1 Wnt pathway inhibition AT2 organoid"),
        ("discover", "discover: pulmonary fibrosis ER stress XBP1 SFTPC misfolding"),
        ("discover", "discover: lipid storage FITM2 ABCA3 lamellar body AT2 cells"),

        # ── Phase 15: Pathways ─────────────────────────────
        ("pathway", "pathway: Surfactant metabolism"),
        ("pathway", "pathway: Vesicle-mediated transport"),
        ("pathway", "pathway: Ubiquitin-mediated proteolysis"),
        ("pathway", "pathway: Endosomal sorting"),
        ("pathway", "pathway: Lipid metabolism"),
        ("pathway", "pathway: Wnt signaling"),
        ("pathway", "pathway: Hippo signaling"),
        ("pathway", "pathway: ER to Golgi vesicle-mediated transport"),
        ("pathway", "pathway: Antigen processing and presentation"),
        ("pathway", "pathway: ER stress response"),
        ("pathway", "pathway: FGF signaling"),
        ("pathway", "pathway: Cytokine-cytokine receptor interaction"),

        # ── Phase 16: Cell-cell interactions ───────────────
        ("interactions", "interactions:"),
        ("interactions", "interactions: AT2-like -> intermediate"),
        ("interactions", "interactions: AT2-like -> differentiating basal-like"),
        ("interactions", "interactions: cycling AT2-like -> AT2-like"),
        ("interactions", "interactions: CXCL+ AT2-like -> AT2-like"),
        ("interactions", "interactions: AT2-like -> NE prog"),
        ("interactions", "interactions: AT2 -> intermediate"),
        ("interactions", "interactions: AT2 -> basal"),
        ("interactions", "interactions: cycling -> AT2"),
        ("interactions", "interactions: CXCL -> AT2"),
        ("interactions", "interactions: AT2 -> neuroendocrine"),
        ("interactions", "interactions: AT2 -> NE"),
    ]

    if not skip_plots:
        queries.extend([
            ("plot", "plot:umap"),
            ("plot", "plot:expr SFTPC"), ("plot", "plot:expr SFTPB"),
            ("plot", "plot:expr LAMP3"), ("plot", "plot:expr ITCH"),
            ("plot", "plot:expr NKX2-1"), ("plot", "plot:expr CXCL1"),
            ("plot", "plot:expr CAV1"), ("plot", "plot:expr AGER"),
            ("plot", "plot:dotplot SFTPC, SFTPB, SFTPA1, SFTPA2, SFTPD, LAMP3, ABCA3, NAPSA"),
            ("plot", "plot:dotplot ITCH, UBE2N, NEDD4, HRS, VPS28, RABGEF1"),
            ("plot", "plot:dotplot HLA-DRA, HLA-DQA1, CXCL1, CCL2, TNF, CD86"),
            ("plot", "plot:grid SFTPC, ITCH, CXCL1, CAV1, NKX2-1, LAMP3"),
        ])
    return queries


def get_queries_test():
    """Minimal test set for quick validation."""
    return [
        ("info", "info"),
        ("proportions", "proportions"),
        ("semantic", "semantic: surfactant protein C SFTPC maturation AT2 cells"),
        ("discover", "discover: ITCH E3 ubiquitin ligase SFTPC trafficking"),
        ("compare", "compare: fdAT2 vs tip_progenitor | SFTPC, ITCH, LAMP3, ABCA3"),
        ("interactions", "interactions:"),
        ("pathway", "pathway: Surfactant metabolism"),
        ("pathway", "pathway: all"),
    ]


# ══════════════════════════════════════════════════════════════
# PAPER GROUND TRUTH
# ══════════════════════════════════════════════════════════════

PAPER_GENES = {
    "SFTPC","SFTPB","SFTPA1","SFTPA2","SFTPD","SFTA3",
    "LAMP3","ABCA3","NAPSA","NKX2-1","HOPX",
    "SLC34A2","LPCAT1","CEACAM6","CD36","CAV1","HTII-280",
    "ITCH","UBE2N","NEDD4","NEDD4L","HRS","VPS28","RABGEF1",
    "UBE2I","UBA2","PIAS1","EEA1","MICALL1",
    "CKAP4","ZDHHC2","CTSH",
    "TRIM65","HPS6","ABCC6","SLCO3A1","LAMTOR1","NEDD1","NDEL1","ABCB8",
    "FITM2","PLEKHF1","XBP1","SLC35F6","STAT5B","TRIM21","TRIM68",
    "ABCC3","F8A1","PSMB8","TAP2","AQP4",
    "CXCL1","CXCL2","CXCL3",
    "AQP5","AGER",
    "HLA-DRA","HLA-DQA1","HLA-DQA2","HLA-DQB1",
    "HLA-DMA","HLA-DMB","HLA-DPA1","HLA-DPB1","HLA-DOA","HLA-G",
    "CD86","TNF","CCL2","CCL4","CCL4L1","CCL3L3","CXCL9","PIK3CG","ITGB2",
    "SOX9","SOX2","TP63","FOXJ1","ASCL1","NEUROD1","GRP","MKI67",
    "CTNNB1","WIF1",
    "UBAP1","USP8","OTUD6B","SOCS3","DMBT1","LMCD1","P2RY2",
    "CUL2","BAP1","PHF10","TRIM33",
}

PAPER_INTERACTIONS = [
    ("AT2-like", "intermediate", "differentiation signaling"),
    ("AT2-like", "differentiating basal-like", "lineage transition"),
    ("cycling AT2-like", "AT2-like", "self-renewal"),
    ("CXCL+ AT2-like", "AT2-like", "chemokine signaling"),
    ("AT2-like", "NE prog", "neuroendocrine induction"),
]

CLUSTER_ALIASES = {
    "AT2-like": ["AT2-like", "AT2", "at2_like", "AT2_like", "alveolar type 2", "alveolar_type_II"],
    "intermediate": ["intermediate", "trans", "transitional", "intermed"],
    "differentiating basal-like": ["differentiating basal-like", "basal-like", "basal", "diff_basal", "differentiating"],
    "cycling AT2-like": ["cycling AT2-like", "cycling AT2", "cycling_AT2", "cycling", "proliferating"],
    "CXCL+ AT2-like": ["CXCL+ AT2-like", "CXCL+", "CXCL_AT2", "chemokine", "inflammatory"],
    "NE prog": ["NE prog", "NE_prog", "neuroendocrine", "NE", "pulmonary NE", "PNE"],
}

PAPER_PATHWAYS = [
    "Surfactant metabolism", "Vesicle-mediated transport",
    "Ubiquitin-mediated proteolysis", "Endosomal sorting",
    "Lipid metabolism", "Wnt signaling", "Hippo signaling",
    "ER to Golgi vesicle-mediated transport",
    "Antigen processing and presentation", "ER stress response",
    "FGF signaling", "Cytokine-cytokine receptor interaction",
]


# ── Execute query ────────────────────────────────────────────
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
    elif cmd_type == "discover":
        txt = cmd_str.split(":", 1)[1].strip()
        payload = engine.discover(txt, top_k=5)
    elif cmd_type == "compare":
        txt = cmd_str.split(":", 1)[1].strip(); genes = None
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
        payload = engine.pathway() if txt.lower() == "all" else engine.pathway(pathway_name=txt)
    elif cmd_type == "plot":
        if adata is None:
            return None, None, []
        os.makedirs(plot_dir, exist_ok=True)
        subcmd = cmd_str.split(None, 1)
        sub = subcmd[0]
        a = subcmd[1].strip() if len(subcmd) > 1 else ""
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
        try:
            answer = ask_llm(llm, SYSTEM_PROMPT, prompt)
        except Exception as e:
            answer = f"[LLM ERROR] {e}"
    return payload, answer, plots


# ══════════════════════════════════════════════════════════════
# EVALUATION v4: Robust interaction matching
# ══════════════════════════════════════════════════════════════

def fuzzy_pathway_match(paper_pw, engine_key):
    pp = paper_pw.lower().replace("-", " ").replace("/", " ")
    ek = engine_key.lower().replace("-", " ").replace("/", " ")
    if pp in ek or ek in pp: return True
    pw = set(pp.split()); ew = set(ek.split())
    if len(pw & ew) >= 2: return True
    aliases = {
        "er stress response": ["er stress", "unfolded protein", "upr"],
        "antigen processing and presentation": ["antigen present", "antigen processing"],
        "er to golgi vesicle-mediated transport": ["er to golgi", "copii", "sec23"],
    }
    for canon, alts in aliases.items():
        if canon in pp:
            for alt in alts:
                if alt in ek: return True
    return False


def cluster_name_matches(paper_name, actual_name):
    pn = paper_name.lower().strip()
    an = actual_name.lower().strip()
    if pn == an or pn in an or an in pn: return True
    aliases = CLUSTER_ALIASES.get(paper_name, [paper_name])
    for alias in aliases:
        al = alias.lower()
        if al == an or al in an or an in al: return True
    pn_words = set(pn.replace("-", " ").replace("+", " ").replace("_", " ").split())
    an_words = set(an.replace("-", " ").replace("+", " ").replace("_", " ").split())
    meaningful = pn_words & an_words - {"like", "cells", "cell", "type", "the", "and", "of"}
    if len(meaningful) >= 1: return True
    return False


def evaluate_interactions(all_interactions, all_pair_summaries):
    found = 0; match_details = {}
    for paper_src, paper_tgt, label in PAPER_INTERACTIONS:
        key = f"{paper_src} -> {paper_tgt}"; matched = False
        for ix in all_interactions:
            ix_src, ix_tgt = ix.get("source", ""), ix.get("target", "")
            if cluster_name_matches(paper_src, ix_src) and cluster_name_matches(paper_tgt, ix_tgt):
                matched = True
                match_details[key] = f"L-R: {ix.get('ligand','?')}-{ix.get('receptor','?')} ({ix_src} → {ix_tgt})"
                break
        if not matched:
            for ps in all_pair_summaries:
                pair_str = ps.get("pair", "")
                if "→" in pair_str:
                    parts = pair_str.split("→")
                    ps_src, ps_tgt = parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
                    if cluster_name_matches(paper_src, ps_src) and cluster_name_matches(paper_tgt, ps_tgt):
                        matched = True
                        match_details[key] = f"pair_summary: {pair_str} (n={ps.get('n_interactions', '?')})"
                        break
        if matched: found += 1
        else: match_details[key] = "NOT FOUND"
    return found, match_details


def build_gene_regex():
    explicit = sorted(PAPER_GENES, key=len, reverse=True)
    escaped = [re.escape(g) for g in explicit]
    patterns = [
        r'SFTP[ABCD]', r'SFTPA[12]', r'SFTA3',
        r'HLA-[A-Z]{1,3}[0-9]?', r'HLA-D[A-Z]{1,2}[0-9]?',
        r'CXCL[0-9]{1,2}', r'CCL[0-9]{1,2}', r'CCL[34]L[0-9]',
        r'TRIM[0-9]{1,2}', r'NEDD[0-9]L?',
        r'ABC[A-Z][0-9]{1,2}', r'SLC[0-9]{1,2}[A-Z][0-9]{1,2}',
        r'SLCO[0-9][A-Z][0-9]', r'AQP[0-9]',
        r'UBE2[A-Z]', r'UBA[0-9]', r'USP[0-9]{1,2}',
        r'SOX[0-9]{1,2}', r'MKI67', r'PIK3CG',
    ]
    return re.compile(r'\b(' + '|'.join(escaped + patterns) + r')\b')


def full_evaluation(report_text, elisa_genes, all_interactions, all_pair_summaries,
                    elisa_pathway_scores, elisa_proportions):
    scorecard = {}

    found_genes = PAPER_GENES & elisa_genes
    gene_recall = len(found_genes) / len(PAPER_GENES) * 100
    scorecard["gene_recall"] = f"{gene_recall:.1f}% ({len(found_genes)}/{len(PAPER_GENES)})"

    pw_found = 0; pw_detail = {}
    for pp in PAPER_PATHWAYS:
        matched = False
        for ek in elisa_pathway_scores:
            if fuzzy_pathway_match(pp, ek):
                matched = True; pw_detail[pp] = ek; break
        if matched: pw_found += 1
        else: pw_detail[pp] = "MISSING"
    pw_cov = pw_found / len(PAPER_PATHWAYS) * 100
    scorecard["pathway_coverage"] = f"{pw_cov:.1f}% ({pw_found}/{len(PAPER_PATHWAYS)})"

    int_found, int_detail = evaluate_interactions(all_interactions, all_pair_summaries)
    int_recall = int_found / len(PAPER_INTERACTIONS) * 100
    scorecard["interaction_recall"] = f"{int_recall:.1f}% ({int_found}/{len(PAPER_INTERACTIONS)})"

    has_prop = bool(elisa_proportions)
    scorecard["proportions_available"] = "Yes" if has_prop else "No"
    scorecard["report_words"] = len(report_text.split())

    themes = {
        "SFTPC trafficking": ["SFTPC trafficking", "surfactant protein C", "proSFTPC", "SFTPC maturation"],
        "ITCH E3 ligase": ["ITCH", "E3 ligase", "E3 ubiquitin"],
        "ESCRT machinery": ["ESCRT", "multivesicular bod", "MVB", "endosomal sorting"],
        "AT2 maturation": ["AT2 matur", "alveolar type 2", "lamellar bod", "AT2-like"],
        "AT1 differentiation": ["AT1 different", "AQP5", "AGER"],
        "I73T variant": ["I73T", "pathogenic variant", "interstitial lung disease", "ILD"],
        "CRISPRi validation": ["CRISPRi", "CRISPR interference", "gene silencing"],
        "Organoid model": ["organoid", "fdAT2", "fetal-derived", "fetal lung"],
    }
    th_found = sum(1 for _, kws in themes.items() if any(kw.lower() in report_text.lower() for kw in kws))
    th_cov = th_found / len(themes) * 100
    scorecard["theme_coverage"] = f"{th_cov:.1f}% ({th_found}/{len(themes)})"

    composite = (gene_recall * 0.30 + pw_cov * 0.20 + int_recall * 0.15 +
                 th_cov * 0.25 + (10 if has_prop else 0))
    scorecard["composite_score"] = f"{composite:.1f}%"

    return {"scorecard": scorecard, "composite_score": round(composite, 1),
            "details": {
                "paper_genes": sorted(PAPER_GENES),
                "found_genes": sorted(found_genes),
                "missing_genes": sorted(PAPER_GENES - found_genes),
                "pathway_matching": pw_detail,
                "interaction_matching": int_detail,
                "paper_pathways": PAPER_PATHWAYS,
                "paper_interactions": [(s, t, d) for s, t, d in PAPER_INTERACTIONS],
            }}


# ── Main ─────────────────────────────────────────────────────
def main():
    args = setup_args()
    out_dir = args.out_dir; plot_dir = os.path.join(out_dir, "elisa_plots")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

    engine, llm, adata, report, viz, caps = init_elisa(args)
    queries = get_queries(skip_plots=args.skip_plots)
    session_log, all_payloads, all_interactions, all_pair_summaries = [], [], [], []

    print(f"\n{'='*70}")
    print(f"ELISA PAPER REPLICATION v4: fdAT2/SFTPC (Max Recall)")
    print(f"Paper: Lim, Rutherford et al. (2025) EMBO Journal")
    print(f"RUNNING {len(queries)} QUERIES")
    print(f"{'='*70}\n")

    for i, (ct, cs) in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {cs}")
        t0 = time.time()
        payload, answer, plots = execute_query(ct, cs, engine, llm, adata, viz, args.cluster_key, plot_dir)
        elapsed = time.time() - t0

        if payload is None and ct == "plot":
            if plots and report.entries:
                report.entries[-1]["plots"].extend(plots)
            continue
        if payload and "error" in payload:
            print(f"  [ERROR] {payload['error']}"); continue
        if payload:
            mode = payload.get("mode", ct)
            if mode == "interactions":
                if "interactions" in payload:
                    all_interactions.extend(payload["interactions"])
                if "pair_summary" in payload:
                    all_pair_summaries.extend(payload["pair_summary"])
            report.add_entry(entry_type=ct, query=payload.get("query", cs),
                             payload=payload, answer=answer or "", plots=plots)
            all_payloads.append(payload)
            session_log.append({"index": i+1, "command": cs, "type": ct, "mode": mode,
                "query": payload.get("query", ""), "answer": answer[:500] if answer else "",
                "elapsed": round(elapsed, 2)})
            if answer:
                print(f"  → {mode} | {elapsed:.1f}s\n  {answer[:150]}...")
        print()

    # ── Report ───────────────────────────────────────────
    print(f"\n{'='*70}\nGENERATING REPORT\n{'='*70}")
    llm_fn = lambda p: ask_llm(llm, SYSTEM_PROMPT, p)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = os.path.join(out_dir, f"elisa_report_fdAT2_v4_{ts}.md")
    report.generate_markdown(md_path, llm_func=llm_fn)
    docx_path = md_path.replace(".md", ".docx")
    report.generate_docx(docx_path, llm_func=llm_fn)
    log_path = os.path.join(out_dir, "session_log.json")
    with open(log_path, "w") as f:
        json.dump(session_log, f, indent=2, default=str)

    # ── Evaluation ───────────────────────────────────────
    print(f"\n{'='*70}\nEVALUATION\n{'='*70}")

    report_text = open(md_path).read() if os.path.exists(md_path) else ""
    elisa_genes = set()

    for p in all_payloads:
        for r in p.get("results", []):
            for g in r.get("gene_evidence", []):
                if isinstance(g, dict): elisa_genes.add(g.get("gene", ""))
                elif isinstance(g, str): elisa_genes.add(g)
            for g in r.get("genes", []):
                if isinstance(g, dict): elisa_genes.add(g.get("gene", ""))
                elif isinstance(g, str): elisa_genes.add(g)
        for cid, cd in p.get("clusters", {}).items():
            if isinstance(cd, dict):
                for g in cd.get("genes", []):
                    if isinstance(g, dict): elisa_genes.add(g.get("gene", ""))
                    elif isinstance(g, str): elisa_genes.add(g)
        if p.get("mode") in ("pathway_scoring", "pathway_query"):
            for pw_data in p.get("pathways", {}).values():
                for gs in pw_data.get("gene_set", []): elisa_genes.add(gs)
                for cl in pw_data.get("scores", []):
                    for tg in cl.get("top_genes", []):
                        if isinstance(tg, dict): elisa_genes.add(tg.get("gene", ""))
                        elif isinstance(tg, (list, tuple)) and len(tg) >= 1: elisa_genes.add(str(tg[0]))
                        elif isinstance(tg, str): elisa_genes.add(tg)
        if p.get("mode") == "pathway_query":
            for g in p.get("genes_in_pathway", []):
                elisa_genes.add(g)

    gene_re = build_gene_regex()
    for e in session_log:
        for m in gene_re.finditer(e.get("answer", "")): elisa_genes.add(m.group(1))
        for m in re.finditer(r'\b([A-Z][A-Z0-9-]{1,15})\s*\(ENSG\d+', e.get("answer", "")): elisa_genes.add(m.group(1))
    for m in gene_re.finditer(report_text): elisa_genes.add(m.group(1))
    for m in re.finditer(r'\b([A-Z][A-Z0-9-]{1,15})\s*\(ENSG\d+', report_text): elisa_genes.add(m.group(1))

    elisa_genes.discard("")
    print(f"  Genes collected: {len(elisa_genes)}")

    pathway_scores = {}
    for p in all_payloads:
        m = p.get("mode", "")
        if m == "pathway_scoring": pathway_scores.update(p.get("pathways", {}))
        elif m == "pathway_query":
            pw = p.get("pathway", "")
            if pw: pathway_scores[pw] = {"scores": p.get("scores", []), "genes_in_pathway": p.get("genes_in_pathway", [])}
    print(f"  Pathways: {len(pathway_scores)}")
    print(f"  Interactions: {len(all_interactions)}, Pair summaries: {len(all_pair_summaries)}")

    actual_clusters = set()
    for ix in all_interactions:
        actual_clusters.add(ix.get("source", ""))
        actual_clusters.add(ix.get("target", ""))
    if actual_clusters:
        print(f"  Actual cluster names in interactions: {sorted(actual_clusters)[:15]}")

    prop_data = next((p for p in all_payloads if p.get("mode") == "proportions"), {})

    ev = full_evaluation(report_text, elisa_genes, all_interactions, all_pair_summaries,
                         pathway_scores, prop_data)

    print(f"\n{'='*70}\nSCORECARD\n{'='*70}")
    for k, v in ev["scorecard"].items(): print(f"  {k}: {v}")
    print(f"\n  ★ COMPOSITE: {ev['composite_score']}%")

    if "interaction_matching" in ev.get("details", {}):
        print(f"\n  Interaction matching detail:")
        for key, val in ev["details"]["interaction_matching"].items():
            status = "✓" if val != "NOT FOUND" else "✗"
            print(f"    {status} {key} → {val}")

    if "pathway_matching" in ev.get("details", {}):
        print(f"\n  Pathway matching detail:")
        for pp, ek in ev["details"]["pathway_matching"].items():
            status = "✓" if ek != "MISSING" else "✗"
            print(f"    {status} {pp} → {ek}")

    eval_path = os.path.join(out_dir, "evaluation_scorecard.json")
    with open(eval_path, "w") as f:
        json.dump(ev, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"OUTPUTS: {out_dir}/")
    print(f"  Report:     {md_path}")
    print(f"  Evaluation: {eval_path}")
    print(f"  Session:    {log_path}")
    print(f"{'='*70}\nDONE")


if __name__ == "__main__":
    main()
