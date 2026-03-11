#!/usr/bin/env python
# ============================================================
# ELISA – LLM Chat Interface v4
# ============================================================
# New in v4:
#   - Comparative analysis (compare:)
#   - Cell-cell interactions (interactions:)
#   - Proportion analysis (proportions)
#   - Pathway scoring (pathway:)
#   - Structured report generation (report)
#   - Dataset capability auto-detection (info)
#   - All analysis results auto-added to report builder
# ============================================================

import os
import sys
import json
import textwrap
import argparse
from datetime import datetime
from typing import List

from elisa_llm_provider import get_llm_client, get_model_name, ask_llm as _provider_ask_llm

from retrieval_engine_v4_hybrid import RetrievalEngine
from elisa_report import ReportBuilder
import elisa_viz as viz

# Optional: scanpy for h5ad-based plots
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


# ============================================================
# LLM
# ============================================================

def get_llm():
    return get_llm_client()


def ask_llm(client, system_prompt, user_prompt, model=None):
    """Delegate to elisa_llm_provider.ask_llm which handles all providers
    (Groq, Gemini, OpenAI, Claude) with spending cap and retry logic."""
    return _provider_ask_llm(client, system_prompt, user_prompt)


def make_llm_func(client, system_prompt):
    """Create a simple callable for report generation."""
    def fn(prompt):
        return ask_llm(client, system_prompt, prompt)
    return fn


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = (
    "You are ELISA, an expert assistant for single-cell biology. "
    "Never hallucinate. Always ground claims strictly in provided data. "
    "Be concise and scientific."
)


MAX_PROMPT_CHARS = 12000  # ~4000 tokens, safe for Groq free tier

def _trim_ctx(payload, max_chars=MAX_PROMPT_CHARS):
    """Trim payload JSON to fit within LLM token limits."""
    trimmed = dict(payload)
    # Trim compare clusters to top 10
    if "clusters" in trimmed and isinstance(trimmed["clusters"], dict):
        clusters = trimmed["clusters"]
        if len(clusters) > 10:
            ranked = sorted(
                clusters.items(),
                key=lambda kv: len(kv[1].get("genes", [])),
                reverse=True
            )[:10]
            trimmed["clusters"] = dict(ranked)
    # Trim retrieval gene evidence
    for r in trimmed.get("results", []):
        if "gene_evidence" in r and len(r["gene_evidence"]) > 5:
            r["gene_evidence"] = r["gene_evidence"][:5]
    # Trim pathway scores
    if "scores" in trimmed and isinstance(trimmed["scores"], list):
        trimmed["scores"] = trimmed["scores"][:10]
    ctx = json.dumps(trimmed, indent=1, default=str)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n... [TRUNCATED]"
    return ctx


def build_standard_prompt(mode, query, payload):
    ctx = _trim_ctx(payload)
    return f"""
You are ELISA, an expert assistant for single-cell RNA-seq analysis.

MODE: {mode.upper()}
QUERY: {query}

DATASET EVIDENCE:
{ctx}

Guidelines:
- Use ONLY the provided dataset evidence
- Explain cluster relevance biologically
- Mention genes ONLY if explicitly present in the evidence
- Do NOT introduce external literature
- Do NOT infer causality
- Be concise, cautious, and scientific
"""


def build_discovery_prompt(query, payload):
    ctx = _trim_ctx(payload)
    return f"""
You are in DISCOVERY mode.

You must strictly separate the answer into FOUR sections:

1. DATASET EVIDENCE
- List clusters involved, genes with expression evidence.
- Use ONLY the provided dataset context.

2. ESTABLISHED BIOLOGY (general knowledge)
- Briefly summarize what these genes are commonly known to do.

3. CONSISTENCY ANALYSIS
- What MATCHES established biology?
- What is UNEXPECTED or CONTEXT-SHIFTED?

4. CANDIDATE NOVEL HYPOTHESES
- Propose cautious hypotheses grounded in mismatches.
- Use probabilistic language. Do NOT claim causality.

BIOLOGICAL QUESTION: {query}

DATASET CONTEXT:
{ctx}
"""


def build_compare_prompt(query, payload):
    ctx = _trim_ctx(payload)
    return f"""
You are ELISA analyzing a COMPARATIVE analysis between two conditions.

COMPARISON: {query}

DATASET EVIDENCE:
{ctx}

Guidelines:
- Identify which cell types show the strongest condition bias
- Highlight genes differentially expressed between conditions
- Note genes upregulated in each condition
- Discuss biological implications of condition-specific patterns
- Be cautious: these are proxy estimates from metadata weights, not formal DE
- Be concise and scientific
"""


def build_interaction_prompt(query, payload):
    # Trim to top interactions and summaries only
    trimmed = {
        "n_total": payload.get("n_total"),
        "top_interactions": payload.get("interactions", [])[:20],
        "pathway_summary": payload.get("pathway_summary", [])[:10],
        "pair_summary": payload.get("pair_summary", [])[:10],
    }
    ctx = json.dumps(trimmed, indent=2, default=str)
    return f"""
You are ELISA analyzing predicted CELL-CELL INTERACTIONS.

QUERY: {query}

PREDICTED INTERACTIONS:
{ctx}

Guidelines:
- Focus on the highest-scoring interactions
- Group by pathway/biological process
- Explain the biological significance of key ligand-receptor pairs
- Note which cell type pairs communicate most
- Mention any unexpected interactions
- Be cautious: these are expression-based predictions, not confirmed interactions
- Be concise and scientific
"""


def build_proportion_prompt(query, payload):
    # Trim: only send top 15 clusters and fold changes
    trimmed = {
        "total_cells": payload.get("total_cells"),
        "n_clusters": payload.get("n_clusters"),
        "top_clusters": payload.get("proportions", [])[:15],
        "mode": "proportions",
    }
    if "proportion_fold_changes" in payload:
        trimmed["proportion_fold_changes"] = payload["proportion_fold_changes"][:15]
    if "condition_column" in payload:
        trimmed["condition_column"] = payload["condition_column"]
    # Condition proportions: just top 5 per condition
    if "condition_proportions" in payload:
        cp = {}
        for cond, data in payload["condition_proportions"].items():
            cp[cond] = {
                "total_cells": data.get("total_cells"),
                "top_clusters": data.get("clusters", [])[:8],
            }
        trimmed["condition_proportions"] = cp

    ctx = json.dumps(trimmed, indent=2, default=str)
    return f"""
You are ELISA analyzing CELL TYPE PROPORTIONS.

QUERY: {query}

PROPORTION DATA:
{ctx}

Guidelines:
- Report the major cell types by abundance
- If condition-specific data is present, highlight differences
- Note cell types that are enriched or depleted in each condition
- Discuss biological implications
- Be concise and scientific
"""


def build_pathway_prompt(query, payload):
    # Trim: for "all pathways" mode, only send ranked summary + top 3 per pathway
    trimmed = {"query": query, "metric": payload.get("metric")}

    if "ranked" in payload:
        # All-pathways mode: just send ranked list with top cluster per pathway
        trimmed["ranked_pathways"] = [
            {"pathway": p["pathway"], "top_cluster": p.get("top_cluster"),
             "top_score": p.get("top_score")}
            for p in payload.get("ranked", [])[:15]
        ]
    elif "scores" in payload:
        # Single pathway mode: send top 10 clusters
        trimmed["pathway"] = payload.get("pathway")
        trimmed["genes_in_pathway"] = payload.get("genes_in_pathway", [])
        trimmed["top_clusters"] = payload.get("scores", [])[:10]

    ctx = json.dumps(trimmed, indent=2, default=str)
    return f"""
You are ELISA analyzing PATHWAY ACTIVITY across cell types.

QUERY: {query}

PATHWAY SCORES:
{ctx}

Guidelines:
- Identify which cell types show highest pathway activity
- Report the top contributing genes
- Discuss biological relevance of pathway activation patterns
- Be concise and scientific
"""


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

PLOT_DIR = "elisa_plots"


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def handle_viz_command(cmd, engine, last_payload, last_answer):
    ensure_plot_dir()
    parts = cmd.split(None, 1)
    subcmd = parts[0] if parts else ""
    args = parts[1].strip() if len(parts) > 1 else ""
    saved = []

    if subcmd in ("plot:auto", "plot:all"):
        if not last_payload:
            print("[VIZ] No retrieval results. Run a query first.")
            return []
        saved = viz.auto_plot_retrieval(engine, last_payload,
                                         save_dir=PLOT_DIR, method="umap")

    elif subcmd == "plot:landscape":
        space = args.lower() if args else "semantic"
        emb = engine.semantic_emb if space.startswith("sem") else engine.scgpt_emb
        label = "Semantic" if space.startswith("sem") else "scGPT"
        hl = ([str(r["cluster_id"]) for r in last_payload.get("results", [])]
              if last_payload else None)
        p = f"{PLOT_DIR}/landscape_{label.lower()}.png"
        viz.plot_embedding_landscape(emb, engine.cluster_ids, method="umap",
                                      highlight_ids=hl, title=f"{label} Embedding Landscape",
                                      space_label=label, save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:dual":
        hl = ([str(r["cluster_id"]) for r in last_payload.get("results", [])]
              if last_payload else None)
        p = f"{PLOT_DIR}/dual_embedding.png"
        viz.plot_dual_embedding(engine.semantic_emb, engine.scgpt_emb,
                                 engine.cluster_ids, highlight_ids=hl, save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:heatmap":
        space = args.lower() if args else "semantic"
        emb = engine.semantic_emb if space.startswith("sem") else engine.scgpt_emb
        label = "Semantic" if space.startswith("sem") else "scGPT"
        hl = ([str(r["cluster_id"]) for r in last_payload.get("results", [])]
              if last_payload else None)
        p = f"{PLOT_DIR}/heatmap_{label.lower()}.png"
        viz.plot_similarity_heatmap(emb, engine.cluster_ids, highlight_ids=hl,
                                     title=f"{label} Similarity", save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:genes":
        if not last_payload: print("[VIZ] No results."); return []
        p = f"{PLOT_DIR}/gene_evidence.png"
        viz.plot_gene_evidence(last_payload.get("results", []),
                                title=f"Gene Evidence – {last_payload.get('query', '')[:50]}",
                                save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:gene_heatmap":
        if not last_payload: print("[VIZ] No results."); return []
        metric = args if args in ("pct_in", "pct_out", "logfc") else "pct_in"
        p = f"{PLOT_DIR}/gene_cluster_heatmap.png"
        viz.plot_gene_cluster_heatmap(last_payload.get("results", []), metric=metric,
                                       title=f"Gene × Cluster ({metric})", save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:radar":
        if not last_payload: print("[VIZ] No results."); return []
        p = f"{PLOT_DIR}/radar.png"
        viz.plot_cluster_radar(last_payload.get("results", []),
                                title=f"Profiles – {last_payload.get('query', '')[:50]}",
                                save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:waterfall":
        if not last_payload: print("[VIZ] No results."); return []
        mode = last_payload.get("mode", "semantic")
        key = "hybrid_similarity" if mode == "hybrid" else "semantic_similarity"
        p = f"{PLOT_DIR}/waterfall.png"
        viz.plot_similarity_waterfall(last_payload.get("results", []), sim_key=key,
                                       title=f"Ranking – {mode.title()}", save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:scatter":
        if not last_payload or last_payload.get("mode") not in ("hybrid", "discovery"):
            print("[VIZ] Need hybrid/discovery results."); return []
        p = f"{PLOT_DIR}/sem_vs_expr.png"
        viz.plot_sem_vs_expr_scatter(last_payload.get("results", []),
                                      engine.cluster_ids, engine._last_sem_sims,
                                      engine._last_expr_sims,
                                      title=f"Sem vs Expr – {last_payload.get('query', '')[:50]}",
                                      save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:lambda_sweep":
        query = args if args else (last_payload.get("query", "") if last_payload else "")
        if not query: print("[VIZ] Provide query."); return []
        sweep = engine.lambda_sweep(query)
        p = f"{PLOT_DIR}/lambda_sweep.png"
        viz.plot_lambda_sweep(sweep["lambdas"], sweep["coverages"],
                               title=f"λ Sweep – {query[:50]}", save_path=p)
        viz.plt.close(); saved.append(p)

    else:
        print(f"[VIZ] Unknown: {subcmd}")
        print("  plot:auto  plot:landscape  plot:dual  plot:heatmap")
        print("  plot:genes  plot:gene_heatmap  plot:radar  plot:waterfall")
        print("  plot:scatter  plot:lambda_sweep")
        print("  plot:umap  plot:expr <gene>  plot:dotplot <genes>  plot:grid <genes>")
        return []

    for p in saved:
        print(f"  → {p}")
    return saved


def handle_h5ad_viz(cmd: str, adata, cluster_key: str = "cell_type",
                    plot_dir: str = PLOT_DIR) -> List[str]:
    """
    Handle h5ad-backed Nature-style plot commands.
    Returns list of saved file paths.
    """
    os.makedirs(plot_dir, exist_ok=True)
    parts = cmd.split(None, 1)
    subcmd = parts[0] if parts else ""
    args_str = parts[1].strip() if len(parts) > 1 else ""
    saved = []

    if adata is None:
        print("[VIZ] No h5ad loaded. Use --h5ad flag when starting ELISA.")
        return []

    if subcmd == "plot:umap":
        # Cell-level UMAP, optionally highlight clusters
        highlight = [c.strip() for c in args_str.split(",")] if args_str else None
        p = f"{plot_dir}/cell_umap.png"
        viz.plot_cell_umap(adata, cluster_key=cluster_key,
                           highlight_clusters=highlight if highlight and highlight[0] else None,
                           save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:expr":
        # Single gene expression UMAP
        if not args_str:
            print("[VIZ] Usage: plot:expr IFNG")
            return []
        gene = args_str.strip()
        p = f"{plot_dir}/expr_{gene}.png"
        try:
            viz.plot_gene_expression_umap(adata, gene=gene,
                                           save_path=p)
            viz.plt.close(); saved.append(p)
        except ValueError as e:
            print(f"[VIZ] {e}")

    elif subcmd == "plot:dotplot":
        # Dot plot for gene list
        if not args_str:
            print("[VIZ] Usage: plot:dotplot IFNG, CD69, HLA-E, KLRC1")
            return []
        genes = [g.strip() for g in args_str.split(",") if g.strip()]
        p = f"{plot_dir}/dotplot.png"
        try:
            viz.plot_dotplot(adata, genes=genes, cluster_key=cluster_key,
                              save_path=p)
            viz.plt.close(); saved.append(p)
        except ValueError as e:
            print(f"[VIZ] {e}")

    elif subcmd == "plot:grid":
        # Multi-gene expression grid
        if not args_str:
            print("[VIZ] Usage: plot:grid IFNG, CD69, HLA-E, KLRC1, IFIT1, MX1")
            return []
        genes = [g.strip() for g in args_str.split(",") if g.strip()]
        p = f"{plot_dir}/expr_grid.png"
        try:
            viz.plot_gene_expression_grid(adata, genes=genes,
                                           cluster_key=cluster_key,
                                           save_path=p)
            viz.plt.close(); saved.append(p)
        except ValueError as e:
            print(f"[VIZ] {e}")

    else:
        print(f"[VIZ] Unknown h5ad command: {subcmd}")
        return []

    for p in saved:
        print(f"  → {p}")
    return saved


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    # ── ARGUMENT PARSING ───────────────────────────────────────
    parser = argparse.ArgumentParser(description="ELISA v4 Chat Interface")
    parser.add_argument("--h5ad", default=None,
                        help="Path to .h5ad file for Nature-style cell plots")
    parser.add_argument("--cluster-key", default="cell_type",
                        help="obs column for cell types (default: cell_type)")
    cli_args = parser.parse_args()

    print("[ELISA] Initializing...")
    engine = RetrievalEngine()
    llm = get_llm()

    # ── LOAD H5AD (optional, for Nature-style plots) ──────────
    adata = None
    cluster_key = cli_args.cluster_key
    if cli_args.h5ad and HAS_SCANPY:
        print(f"[ELISA] Loading h5ad: {cli_args.h5ad}")
        adata = sc.read_h5ad(cli_args.h5ad)
        # Remap ENSEMBL → gene symbols if needed
        if adata.var_names[0].startswith("ENSG"):
            if "feature_name" in adata.var.columns:
                adata.var["ensembl_id"] = adata.var_names.copy()
                adata.var_names = adata.var["feature_name"].astype(str).values
                adata.var_names_make_unique()
                print(f"[ELISA] Remapped ENSEMBL → gene symbols")
        print(f"[ELISA] h5ad loaded: {adata.shape[0]} cells, "
              f"{adata.shape[1]} genes, "
              f"cluster_key='{cluster_key}'")
        if "X_umap" not in adata.obsm:
            print("[WARN] No X_umap found — running sc.tl.umap()...")
            sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
            sc.tl.umap(adata)
            print("[ELISA] UMAP computed.")
    elif cli_args.h5ad and not HAS_SCANPY:
        print("[WARN] scanpy not installed — Nature plots disabled")
    else:
        print("[ELISA] No --h5ad provided. Nature-style plots disabled.")
        print("        Use: python elisa_chat_v4.py --h5ad /path/to/data.h5ad")

    # Detect dataset capabilities
    caps = engine.detect_capabilities()
    ds_name = engine.cluster_ids[0].split()[0] if engine.cluster_ids else "Dataset"
    report = ReportBuilder(dataset_name=ds_name)

    last_payload = None
    last_answer = None
    last_plots = []

    # Print capabilities
    print(f"\n[DATASET] {engine.n} clusters loaded")
    if caps["has_conditions"]:
        print(f"[DATASET] Condition column: '{caps['condition_column']}' "
              f"→ {caps['condition_values']}")
        print(f"[DATASET] Comparative analysis: AVAILABLE")
    else:
        print(f"[DATASET] Comparative analysis: NOT AVAILABLE (no condition column)")
    print(f"[DATASET] Interactions, proportions, pathways: AVAILABLE")

    print("""
╔══════════════════════════════════════════════════════════════╗
║                     ELISA v4 – Commands                     ║
╠══════════════════════════════════════════════════════════════╣
║  RETRIEVAL                                                  ║
║    semantic: <text>        Semantic-only retrieval           ║
║    hybrid: <text>          Hybrid (semantic + scGPT)         ║
║    discover: <question>    Discovery mode                    ║
║                                                              ║
║  ANALYSIS                                                    ║
║    compare: <A> vs <B>     Comparative (condition A vs B)    ║
║    compare: <A> vs <B> | <genes>  Compare specific genes    ║
║    interactions:           All cell-cell interactions         ║
║    interactions: <src> -> <tgt>  Directed interactions       ║
║    proportions             Cell type proportions             ║
║    pathway: <name>         Score a specific pathway          ║
║    pathway: all            Score all built-in pathways        ║
║                                                              ║
║  VISUALIZATION                                               ║
║    plot:auto               All plots for last query          ║
║    plot:landscape <s|e>    Embedding landscape               ║
║    plot:dual               Side-by-side embeddings           ║
║    plot:heatmap <s|e>      Similarity heatmap                ║
║    plot:genes              Gene evidence bars                ║
║    plot:gene_heatmap [m]   Gene × cluster heatmap            ║
║    plot:radar              Radar profiles                    ║
║    plot:waterfall          Similarity ranking                ║
║    plot:scatter            Sem vs expr scatter               ║
║    plot:lambda_sweep <q>   Lambda sweep                      ║
║                                                              ║
║  NATURE-STYLE PLOTS (requires --h5ad)                        ║
║    plot:umap               Cell-level UMAP (all clusters)    ║
║    plot:umap <c1, c2>      UMAP highlighting clusters        ║
║    plot:expr <gene>        Gene expression UMAP              ║
║    plot:dotplot <g1,g2,..> Dot plot for gene list            ║
║    plot:grid <g1,g2,..>    Multi-gene expression grid        ║
║                                                              ║
║  DATA                                                        ║
║    info                    Dataset capabilities              ║
║    genes                   List all genes                    ║
║    genes: <prefix>         Filter genes                      ║
║    metadata: <cluster>     Cluster metadata                  ║
║    cells: <cluster>        Cell IDs                          ║
║                                                              ║
║  REPORT                                                      ║
║    report                  Generate structured report (docx) ║
║    report: md              Generate Markdown report          ║
║    export                  Quick export (last result)        ║
║    quit                    Exit                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    while True:
        try:
            q = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("quit", "exit"):
            break

        payload = None
        answer = None
        prompt = None
        entry_type = None

        # ── VISUALIZATION ──────────────────────────────────────
        if q.startswith("plot:"):
            # Route Nature-style commands to h5ad handler
            subcmd = q.split(None, 1)[0]
            if subcmd in ("plot:umap", "plot:expr", "plot:dotplot", "plot:grid"):
                plots = handle_h5ad_viz(q, adata, cluster_key=cluster_key,
                                        plot_dir=PLOT_DIR)
            else:
                plots = handle_viz_command(q, engine, last_payload, last_answer)
            last_plots.extend(plots)
            # Attach plots to the most recent report entry
            if plots and report.entries:
                report.entries[-1]["plots"].extend(plots)
            continue

        # ── INFO ───────────────────────────────────────────────
        elif q == "info":
            caps = engine.detect_capabilities()
            print(json.dumps(caps, indent=2, default=str))
            print(f"\nClusters: {engine.cluster_ids}")
            continue

        # ── RETRIEVAL COMMANDS ─────────────────────────────────
        elif q.startswith("semantic:"):
            txt = q.split(":", 1)[1].strip()
            payload = engine.query_semantic(txt, top_k=5, with_genes=True)
            prompt = build_standard_prompt("semantic", txt, payload)
            entry_type = "semantic"

        elif q.startswith("hybrid:"):
            txt = q.split(":", 1)[1].strip()
            payload = engine.query_hybrid(txt, top_k=5, lambda_sem=0.0,
                                           pre_k=40, gamma=2.5, with_genes=True)
            prompt = build_standard_prompt("hybrid", txt, payload)
            entry_type = "hybrid"

        elif q.startswith("discover:"):
            txt = q.split(":", 1)[1].strip()
            payload = engine.discover(txt, top_k=5, lambda_sem=0.5,
                                       pre_k=40, gamma=2.5)
            prompt = build_discovery_prompt(txt, payload)
            entry_type = "discovery"

        # ── COMPARE ────────────────────────────────────────────
        elif q.startswith("compare:"):
            txt = q.split(":", 1)[1].strip()
            # Parse: "CF vs Control" or "CF vs Control | IFNG, CD69"
            genes = None
            if "|" in txt:
                txt, gene_str = txt.split("|", 1)
                genes = [g.strip() for g in gene_str.split(",") if g.strip()]
                txt = txt.strip()

            parts = txt.lower().split(" vs ")
            if len(parts) != 2:
                print("[ERROR] Format: compare: <A> vs <B>")
                print("        compare: <A> vs <B> | gene1, gene2, gene3")
                continue

            group_a = parts[0].strip()
            group_b = parts[1].strip()

            # Match against actual condition values (case-insensitive)
            caps = engine.detect_capabilities()
            if caps["has_conditions"]:
                cond_vals = caps["condition_values"]
                for cv in cond_vals:
                    if cv.lower() == group_a:
                        group_a = cv
                    if cv.lower() == group_b:
                        group_b = cv

            payload = engine.compare(group_a, group_b, genes=genes)
            if "error" in payload:
                print(f"[ERROR] {payload['error']}")
                continue
            prompt = build_compare_prompt(payload.get("query", txt), payload)
            entry_type = "compare"

        # ── INTERACTIONS ───────────────────────────────────────
        elif q.startswith("interactions"):
            txt = q.split(":", 1)[1].strip() if ":" in q else ""
            src = None
            tgt = None
            if "->" in txt:
                parts = txt.split("->")
                src = parts[0].strip() if parts[0].strip() else None
                tgt = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            elif txt:
                src = txt  # single source

            payload = engine.interactions(source=src, target=tgt)
            prompt = build_interaction_prompt(
                payload.get("query", "Cell-cell interactions"), payload)
            entry_type = "interactions"

        # ── PROPORTIONS ────────────────────────────────────────
        elif q.startswith("proportions"):
            payload = engine.proportions()
            prompt = build_proportion_prompt(
                payload.get("query", "Cell type proportions"), payload)
            entry_type = "proportions"

        # ── PATHWAY ────────────────────────────────────────────
        elif q.startswith("pathway:"):
            txt = q.split(":", 1)[1].strip()
            if txt.lower() == "all":
                payload = engine.pathway()
            else:
                payload = engine.pathway(pathway_name=txt)

            if "error" in payload:
                print(f"[ERROR] {payload['error']}")
                if "available" in payload:
                    print("Available pathways:")
                    for pw in payload["available"]:
                        print(f"  - {pw}")
                continue
            prompt = build_pathway_prompt(
                payload.get("query", txt), payload)
            entry_type = "pathway"

        # ── DATA COMMANDS ──────────────────────────────────────
        elif q.startswith("genes"):
            prefix = q.split(":", 1)[1].strip() if ":" in q else None
            all_genes = set()
            for cid, stats in engine.gene_stats.items():
                all_genes.update(stats.keys())
            if prefix:
                matched = sorted(g for g in all_genes if g.upper().startswith(prefix.upper()))
            else:
                matched = sorted(all_genes)
            print(f"{len(matched)} genes" + (f" matching '{prefix}'" if prefix else ""))
            print(", ".join(matched[:100]))
            if len(matched) > 100:
                print(f"  ... and {len(matched) - 100} more")
            continue

        elif q.startswith("metadata:"):
            cid = q.split(":", 1)[1].strip()
            print(json.dumps(engine.get_metadata(cid), indent=2))
            continue

        elif q.startswith("cells:"):
            cid = q.split(":", 1)[1].strip()
            cells = engine.get_cells(cid)
            print(f"{len(cells)} cells:", cells[:20])
            continue

        # ── REPORT ─────────────────────────────────────────────
        elif q.startswith("report"):
            fmt = "docx"
            if ":" in q:
                fmt = q.split(":", 1)[1].strip().lower()

            llm_fn = make_llm_func(llm, SYSTEM_PROMPT)

            if fmt == "md":
                path = report.generate_markdown(llm_func=llm_fn)
            else:
                path = report.generate_docx(llm_func=llm_fn)

            print(f"[REPORT] Generated: {path}")
            print(f"[REPORT] Contains {len(report.entries)} analysis entries")
            continue

        elif q.startswith("export"):
            if last_payload and last_answer:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"elisa_export_{ts}.json"
                with open(fn, "w") as f:
                    json.dump({
                        "payload": last_payload,
                        "answer": last_answer,
                        "plots": last_plots,
                    }, f, indent=2, default=str)
                print(f"[EXPORTED] {fn}")
            else:
                print("[EXPORT] No results to export.")
            continue

        else:
            print("[ERROR] Unknown command. See the command list above.")
            continue

        # ── LLM CALL ──────────────────────────────────────────
        if prompt and payload:
            answer = ask_llm(llm, SYSTEM_PROMPT, prompt)
            last_payload = payload
            last_answer = answer
            last_plots = []

            # Auto-generate plots for retrieval-type analyses
            if entry_type in ("semantic", "hybrid", "discovery") and payload.get("results"):
                ensure_plot_dir()
                try:
                    auto_plots = viz.auto_plot_retrieval(
                        engine, payload, save_dir=PLOT_DIR, method="umap")
                    last_plots.extend(auto_plots)
                    viz.plt.close("all")
                except Exception as e:
                    print(f"[VIZ] Auto-plot failed: {e}")

            # Add to report builder (with plots attached)
            report.add_entry(
                entry_type=entry_type,
                query=payload.get("query", q),
                payload=payload,
                answer=answer,
                plots=last_plots,
            )

            print("\n" + textwrap.fill(answer, width=100))
            print("-" * 100)

            if last_plots:
                print(f"[VIZ] {len(last_plots)} plots auto-generated in {PLOT_DIR}/")

            # Print report counter
            print(f"[SESSION] {len(report.entries)} analyses collected. "
                  f"Type 'report' to generate.")


if __name__ == "__main__":
    main()
