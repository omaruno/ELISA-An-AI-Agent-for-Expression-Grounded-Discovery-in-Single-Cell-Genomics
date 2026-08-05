#!/usr/bin/env python
# ============================================================
# ELISA – LLM Chat Interface v4.2
# ============================================================
# Changes vs v4.1:
#   - Imports the compat engine (the v3 shim alone lacks half the surface)
#   - hybrid: now performs TRUE late fusion (RRF) in the engine; the
#     --hybrid-lambda flag is the semantic fusion weight
#   - union: exposes the additive union strategy from the manuscript
#   - discover: dedicated system prompt, per-section budget, max_tokens,
#     and a completion pass if any of the four sections is missing
#   - Grounding guidelines moved BEFORE the evidence blob, so truncation
#     can never drop the instructions
#   - `traceback` imported (the crash-proof loop was raising NameError)
#   - Removed duplicated __main__ block
# ============================================================

import os
import re
import json
import textwrap
import traceback
import argparse
from datetime import datetime
from typing import List

from elisa_llm_provider import get_llm_client, get_model_name

# Optional: some builds of elisa_llm_provider export a character budget.
# It is not required — fall back to the local constant if absent.
try:
    from elisa_llm_provider import LLM_MAX_INPUT_CHARS as _PROVIDER_MAX_CHARS
except ImportError:
    _PROVIDER_MAX_CHARS = None

# IMPORTANT: the compat layer, not the bare v3 shim.
from elisa_engine_compat import RetrievalEngine
from elisa_report import ReportBuilder
import elisa_viz as viz

# Optional: scanpy for h5ad-based plots
try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


# ============================================================
# CONFIG
# ============================================================

PLOT_DIR = "elisa_plots"

# Context budget for the dataset evidence blob (~3,000 tokens).
MAX_PROMPT_CHARS = 12000
# Room for the instruction wrapper around that blob.
PROMPT_OVERHEAD_CHARS = 6000
# Hard ceiling applied to the fully assembled user prompt (~4,500 tokens).
MAX_LLM_CHARS = MAX_PROMPT_CHARS + PROMPT_OVERHEAD_CHARS
if _PROVIDER_MAX_CHARS:
    MAX_LLM_CHARS = min(MAX_LLM_CHARS, int(_PROVIDER_MAX_CHARS))

DEFAULT_MAX_TOKENS = 1024
DISCOVERY_MAX_TOKENS = 3072


# ============================================================
# LLM
# ============================================================

def get_llm():
    return get_llm_client()


def ask_llm(client, system_prompt, user_prompt, model=None,
            max_tokens=DEFAULT_MAX_TOKENS):
    if model is None:
        model = get_model_name()
    if len(user_prompt) > MAX_LLM_CHARS:
        user_prompt = (user_prompt[:MAX_LLM_CHARS]
                       + "\n\n[... truncated for length ...]")

    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content.strip()


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
    "Do not introduce external literature and do not infer causality. "
    "Be concise and scientific."
)

# Discovery mode deliberately ALLOWS general knowledge, but only inside
# section 2, and requires the four-section structure. Using SYSTEM_PROMPT
# here would forbid section 2 and cause the model to truncate 3 and 4.
DISCOVERY_SYSTEM_PROMPT = (
    "You are ELISA operating in DISCOVERY mode.\n"
    "You MUST always output exactly four numbered sections, in this order, "
    "using these exact headers, even if a section must be brief:\n"
    "1. DATASET EVIDENCE — clusters, genes and scores taken ONLY from the "
    "provided dataset context. Never invent a gene, cluster or number.\n"
    "2. ESTABLISHED BIOLOGY — what those genes and cell types are generally "
    "known to do, from your own knowledge. Do NOT cite the dataset here.\n"
    "3. CONSISTENCY ANALYSIS — state explicitly what MATCHES established "
    "biology and what is UNEXPECTED or CONTEXT-SHIFTED.\n"
    "4. CANDIDATE NOVEL HYPOTHESES — cautious hypotheses grounded in the "
    "mismatches from section 3. Use probabilistic language "
    "('may', 'is consistent with', 'could indicate'). Never claim causality.\n"
    "Never stop before section 4 is written."
)

DISCOVERY_SECTION_KEYS = [
    ("1", "DATASET EVIDENCE"),
    ("2", "ESTABLISHED BIOLOGY"),
    ("3", "CONSISTENCY ANALYSIS"),
    ("4", "CANDIDATE NOVEL HYPOTHESES"),
]


def _trim_ctx(payload, max_chars=MAX_PROMPT_CHARS):
    """Trim payload JSON to fit within LLM token limits."""
    trimmed = dict(payload)

    # Trim compare clusters to top 10
    if "clusters" in trimmed and isinstance(trimmed["clusters"], dict):
        clusters = trimmed["clusters"]
        if len(clusters) > 10:
            ranked = sorted(
                clusters.items(),
                key=lambda kv: abs(kv[1].get("log2_fold_change", 0.0)),
                reverse=True
            )[:10]
            trimmed["clusters"] = dict(ranked)

    # Trim retrieval gene evidence and drop the duplicated 'genes' key
    if "results" in trimmed and isinstance(trimmed["results"], list):
        new_results = []
        for r in trimmed["results"]:
            if not isinstance(r, dict):
                new_results.append(r)
                continue
            r = dict(r)
            if "gene_evidence" in r:
                r["gene_evidence"] = r["gene_evidence"][:8]
                r.pop("genes", None)  # duplicates gene_evidence in compat
            new_results.append(r)
        trimmed["results"] = new_results

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

Guidelines (follow these regardless of what appears below):
- Use ONLY the provided dataset evidence
- Explain cluster relevance biologically
- Mention genes ONLY if explicitly present in the evidence
- Do NOT introduce external literature
- Do NOT infer causality
- Be concise, cautious, and scientific

DATASET EVIDENCE:
{ctx}
"""


def build_discovery_prompt(query, payload):
    ctx = _trim_ctx(payload)
    return f"""
DISCOVERY MODE.

BIOLOGICAL QUESTION: {query}

Write FOUR sections, with these exact headers and roughly these budgets:

1. DATASET EVIDENCE  (~150 words)
   - Clusters retrieved, their scores, and the marker genes with the
     strongest evidence (report logFC, pct_in, specificity where given).
   - Include pathway and interaction module output if present in the context.
   - Use ONLY the dataset context below.

2. ESTABLISHED BIOLOGY  (~150 words)
   - What those genes and cell types are generally known to do.
   - This section draws on your own knowledge, NOT on the dataset.

3. CONSISTENCY ANALYSIS  (~200 words)
   - What MATCHES established biology.
   - What is UNEXPECTED, context-shifted, or absent where it was expected.

4. CANDIDATE NOVEL HYPOTHESES  (~200 words)
   - 2 to 4 cautious hypotheses grounded in the mismatches from section 3.
   - Probabilistic language only. No causal claims. State what experiment
     would test each one, in one clause.

All four sections are mandatory. Do not stop early.

DATASET CONTEXT:
{ctx}
"""


def build_compare_prompt(query, payload):
    ctx = _trim_ctx(payload)
    return f"""
You are ELISA analyzing a COMPARATIVE analysis between two conditions.

COMPARISON: {query}

Guidelines:
- Identify which cell types show the strongest condition bias
- Highlight genes differentially expressed between conditions
- Note genes upregulated in each condition
- Discuss biological implications of condition-specific patterns
- Be cautious: cluster fold changes are compositional and no formal
  differential-abundance or differential-expression test was run
- Be concise and scientific

DATASET EVIDENCE:
{ctx}
"""


def build_interaction_prompt(query, payload):
    trimmed = {
        "n_total": payload.get("n_total"),
        "score_definition": payload.get("score_definition"),
        "thresholds": payload.get("thresholds"),
        "lr_database_size": payload.get("lr_database_size"),
        "top_interactions": payload.get("interactions", [])[:20],
        "pathway_summary": payload.get("pathway_summary", [])[:10],
        "pair_summary": payload.get("pair_summary", [])[:10],
        "caveat": payload.get("caveat"),
        "note": payload.get("note"),
    }
    ctx = json.dumps(trimmed, indent=2, default=str)
    return f"""
You are ELISA analyzing predicted CELL-CELL INTERACTIONS.

QUERY: {query}

Guidelines:
- Focus on the highest-scoring interactions
- Group by pathway/biological process
- Explain the biological significance of key ligand-receptor pairs
- Note which cell type pairs communicate most
- Mention any unexpected interactions
- Be cautious: these are co-enrichment predictions from marker statistics,
  not confirmed interactions, and no null model was applied
- Be concise and scientific

PREDICTED INTERACTIONS:
{ctx}
"""


def build_proportion_prompt(query, payload):
    trimmed = {
        "total_cells": payload.get("total_cells"),
        "n_clusters": payload.get("n_clusters"),
        "top_clusters": payload.get("proportions", [])[:15],
        "source": payload.get("source"),
        "mode": "proportions",
    }
    if "proportion_fold_changes" in payload:
        trimmed["proportion_fold_changes"] = payload["proportion_fold_changes"][:15]
    if "condition_column" in payload:
        trimmed["condition_column"] = payload["condition_column"]
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

Guidelines:
- Report the major cell types by abundance
- If condition-specific data is present, highlight differences
- Note cell types that are enriched or depleted in each condition
- Discuss biological implications
- Be concise and scientific

PROPORTION DATA:
{ctx}
"""


def build_pathway_prompt(query, payload):
    trimmed = {"query": query, "metric": payload.get("metric"),
               "caveat": payload.get("caveat")}

    if "ranked" in payload:
        trimmed["n_pathways_scored"] = payload.get("n_pathways")
        trimmed["ranked_pathways"] = [
            {"pathway": p["pathway"], "category": p.get("category"),
             "top_cluster": p.get("top_cluster"),
             "top_score": p.get("top_score"),
             "coverage": p.get("coverage")}
            for p in payload.get("ranked", [])[:15]
        ]
    elif "scores" in payload:
        trimmed["pathway"] = payload.get("pathway")
        trimmed["category"] = payload.get("category")
        trimmed["genes_in_pathway"] = payload.get("genes_in_pathway", [])
        trimmed["top_clusters"] = payload.get("scores", [])[:10]

    ctx = json.dumps(trimmed, indent=2, default=str)
    return f"""
You are ELISA analyzing PATHWAY ACTIVITY across cell types.

QUERY: {query}

Guidelines:
- Identify which cell types show highest pathway activity
- Report the top contributing genes and the coverage of each gene set
- Discuss biological relevance of pathway activation patterns
- Note that scores derive from per-cluster marker statistics, not per-cell
  expression, and are therefore an enrichment proxy
- Be concise and scientific

PATHWAY SCORES:
{ctx}
"""


# ============================================================
# DISCOVERY SECTION VALIDATION
# ============================================================

def missing_discovery_sections(answer: str) -> List[str]:
    """Return the headers of any of the four sections that are absent."""
    up = (answer or "").upper()
    missing = []
    for num, title in DISCOVERY_SECTION_KEYS:
        head = title.split()[0]
        pattern = rf"(?:^|\n)\s*\**\s*{num}[\.\):]?\s*\**\s*{head}"
        if re.search(pattern, up) or title in up:
            continue
        missing.append(f"{num}. {title}")
    return missing


def run_discovery(client, prompt, payload):
    """One call, plus one completion pass if a section is missing."""
    answer = ask_llm(client, DISCOVERY_SYSTEM_PROMPT, prompt,
                     max_tokens=DISCOVERY_MAX_TOKENS)

    missing = missing_discovery_sections(answer)
    if not missing:
        return answer

    print(f"[DISCOVERY] Missing section(s): {', '.join(missing)} — completing.")
    follow = (
        "Your previous answer was incomplete. Below is what you already "
        "wrote. Continue it by writing ONLY the missing sections, with their "
        "exact headers, in order. Do not repeat the sections already present.\n\n"
        f"MISSING SECTIONS: {', '.join(missing)}\n\n"
        "--- YOUR PREVIOUS ANSWER ---\n"
        f"{answer[-6000:]}\n"
        "--- END ---\n\n"
        "--- DATASET CONTEXT (unchanged) ---\n"
        f"{_trim_ctx(payload, max_chars=6000)}\n"
    )
    try:
        tail = ask_llm(client, DISCOVERY_SYSTEM_PROMPT, follow,
                       max_tokens=DISCOVERY_MAX_TOKENS)
        answer = answer.rstrip() + "\n\n" + tail.strip()
    except Exception as e:
        print(f"[DISCOVERY] Completion pass failed: {type(e).__name__}: {e}")

    still = missing_discovery_sections(answer)
    if still:
        print(f"[DISCOVERY] WARNING — still missing: {', '.join(still)}")
    return answer


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

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
        if emb is None:
            print(f"[VIZ] No {label} embeddings in this .pt file.")
            return []
        hl = ([str(r["cluster_id"]) for r in last_payload.get("results", [])]
              if last_payload else None)
        p = f"{PLOT_DIR}/landscape_{label.lower()}.png"
        viz.plot_embedding_landscape(emb, engine.cluster_ids, method="umap",
                                     highlight_ids=hl,
                                     title=f"{label} Embedding Landscape",
                                     space_label=label, save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:dual":
        if engine.semantic_emb is None or engine.scgpt_emb is None:
            print("[VIZ] Need both semantic and scGPT embeddings.")
            return []
        hl = ([str(r["cluster_id"]) for r in last_payload.get("results", [])]
              if last_payload else None)
        p = f"{PLOT_DIR}/dual_embedding.png"
        viz.plot_dual_embedding(engine.semantic_emb, engine.scgpt_emb,
                                engine.cluster_ids, highlight_ids=hl,
                                save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:heatmap":
        space = args.lower() if args else "semantic"
        emb = engine.semantic_emb if space.startswith("sem") else engine.scgpt_emb
        label = "Semantic" if space.startswith("sem") else "scGPT"
        if emb is None:
            print(f"[VIZ] No {label} embeddings in this .pt file.")
            return []
        hl = ([str(r["cluster_id"]) for r in last_payload.get("results", [])]
              if last_payload else None)
        p = f"{PLOT_DIR}/heatmap_{label.lower()}.png"
        viz.plot_similarity_heatmap(emb, engine.cluster_ids, highlight_ids=hl,
                                    title=f"{label} Similarity", save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:genes":
        if not last_payload:
            print("[VIZ] No results."); return []
        p = f"{PLOT_DIR}/gene_evidence.png"
        viz.plot_gene_evidence(
            last_payload.get("results", []),
            title=f"Gene Evidence – {last_payload.get('query', '')[:50]}",
            save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:gene_heatmap":
        if not last_payload:
            print("[VIZ] No results."); return []
        metric = args if args in ("pct_in", "pct_out", "logfc") else "pct_in"
        p = f"{PLOT_DIR}/gene_cluster_heatmap.png"
        viz.plot_gene_cluster_heatmap(last_payload.get("results", []),
                                      metric=metric,
                                      title=f"Gene × Cluster ({metric})",
                                      save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:radar":
        if not last_payload:
            print("[VIZ] No results."); return []
        p = f"{PLOT_DIR}/radar.png"
        viz.plot_cluster_radar(
            last_payload.get("results", []),
            title=f"Profiles – {last_payload.get('query', '')[:50]}",
            save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:waterfall":
        if not last_payload:
            print("[VIZ] No results."); return []
        mode = last_payload.get("mode", "semantic")
        key = ("hybrid_similarity" if mode in ("hybrid", "discovery", "union")
               else "semantic_similarity")
        p = f"{PLOT_DIR}/waterfall.png"
        viz.plot_similarity_waterfall(last_payload.get("results", []),
                                      sim_key=key,
                                      title=f"Ranking – {mode.title()}",
                                      save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:scatter":
        if not last_payload or last_payload.get("mode") not in (
                "hybrid", "discovery", "union"):
            print("[VIZ] Need hybrid/discovery/union results."); return []
        if engine._last_sem_sims is None or engine._last_expr_sims is None:
            print("[VIZ] No similarity diagnostics available for that query "
                  "(needs both a semantic space and gene symbols in the query).")
            return []
        p = f"{PLOT_DIR}/sem_vs_expr.png"
        viz.plot_sem_vs_expr_scatter(
            last_payload.get("results", []),
            engine.cluster_ids, engine._last_sem_sims,
            engine._last_expr_sims,
            title=f"Sem vs Expr – {last_payload.get('query', '')[:50]}",
            save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:lambda_sweep":
        query = args if args else (last_payload.get("query", "")
                                   if last_payload else "")
        if not query:
            print("[VIZ] Provide query."); return []
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
        print("  plot:umap  plot:expr <gene>  plot:dotplot <genes>  "
              "plot:grid <genes>")
        return []

    for p in saved:
        print(f"  → {p}")
    return saved


def handle_h5ad_viz(cmd: str, adata, cluster_key: str = "cell_type",
                    plot_dir: str = PLOT_DIR) -> List[str]:
    """Handle h5ad-backed Nature-style plot commands."""
    os.makedirs(plot_dir, exist_ok=True)
    parts = cmd.split(None, 1)
    subcmd = parts[0] if parts else ""
    args_str = parts[1].strip() if len(parts) > 1 else ""
    saved = []

    if adata is None:
        print("[VIZ] No h5ad loaded. Use --h5ad flag when starting ELISA.")
        return []

    if subcmd == "plot:umap":
        highlight = [c.strip() for c in args_str.split(",")] if args_str else None
        p = f"{plot_dir}/cell_umap.png"
        viz.plot_cell_umap(
            adata, cluster_key=cluster_key,
            highlight_clusters=highlight if highlight and highlight[0] else None,
            save_path=p)
        viz.plt.close(); saved.append(p)

    elif subcmd == "plot:expr":
        if not args_str:
            print("[VIZ] Usage: plot:expr SFTPC")
            return []
        gene = args_str.strip()
        p = f"{plot_dir}/expr_{gene}.png"
        try:
            viz.plot_gene_expression_umap(adata, gene=gene, save_path=p)
            viz.plt.close(); saved.append(p)
        except ValueError as e:
            print(f"[VIZ] {e}")

    elif subcmd == "plot:dotplot":
        if not args_str:
            print("[VIZ] Usage: plot:dotplot SFTPC, SFTPB, LAMP3, ABCA3")
            return []
        genes = [g.strip() for g in args_str.replace(",", " ").split() if g.strip()]
        p = f"{plot_dir}/dotplot.png"
        try:
            viz.plot_dotplot(adata, genes=genes, cluster_key=cluster_key,
                             save_path=p)
            viz.plt.close(); saved.append(p)
        except ValueError as e:
            print(f"[VIZ] {e}")

    elif subcmd == "plot:grid":
        if not args_str:
            print("[VIZ] Usage: plot:grid SFTPC, SFTPB, LAMP3, ABCA3, NAPSA")
            return []
        genes = [g.strip() for g in args_str.replace(",", " ").split() if g.strip()]
        p = f"{plot_dir}/expr_grid.png"
        try:
            viz.plot_gene_expression_grid(adata, genes=genes,
                                          cluster_key=cluster_key, save_path=p)
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
# BANNER
# ============================================================

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                    ELISA v4.2 – Commands                     ║
╠══════════════════════════════════════════════════════════════╣
║  RETRIEVAL                                                   ║
║    semantic: <text>        Semantic-only (BioBERT)           ║
║    hybrid: <text>          Late fusion (RRF, k=60)           ║
║    union: <text>           Additive union (benchmark mode)   ║
║    discover: <question>    Discovery mode (4 sections)       ║
║                                                              ║
║  ANALYSIS                                                    ║
║    compare: <A> vs <B>     Comparative (condition A vs B)    ║
║    compare: <A> vs <B> | <genes>   Compare specific genes    ║
║    interactions            All cell-cell interactions        ║
║    interactions: <s> -> <t>  Directed interactions           ║
║    proportions             Cell type proportions             ║
║    pathway: <name>         Score a specific pathway          ║
║    pathway: all            Score all built-in pathways       ║
║    pathways                List pathway names by category    ║
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
║    autoplot [on|off]       Auto-plot after queries (off)     ║
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
║    clusters                List cluster names                ║
║                                                              ║
║  REPORT                                                      ║
║    report                  Generate structured report (docx) ║
║    report: md              Generate Markdown report          ║
║    export                  Quick export (last result)        ║
║    quit                    Exit                              ║
╚══════════════════════════════════════════════════════════════╝
"""


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    # ── ARGUMENT PARSING ───────────────────────────────────────
    parser = argparse.ArgumentParser(description="ELISA v4.2 Chat Interface")
    parser.add_argument("--h5ad", default=None,
                        help="Path to .h5ad file for cell-level plots and "
                             "proportion/comparative analysis")
    parser.add_argument("--cluster-key", default="cell_type",
                        help="obs column for cell types (default: cell_type)")
    parser.add_argument("--base", required=True,
                        help="Embedding directory containing the .pt model")
    parser.add_argument("--pt-name", required=True,
                        help="Filename of the fused/hybrid .pt model")
    parser.add_argument("--cells-csv", default=None,
                        help="Optional cell metadata CSV (alternative to "
                             "--h5ad for proportions/conditions)")
    parser.add_argument("--hybrid-lambda", type=float, default=0.5,
                        help="Semantic weight in the RRF fusion for 'hybrid:' "
                             "queries. 0.0 = pure gene marker scoring, "
                             "1.0 = pure semantic, 0.5 = balanced (default)")
    parser.add_argument("--pre-k", type=int, default=40,
                        help="Candidate depth per pipeline before fusion "
                             "(default: 40, capped at the cluster count)")
    parser.add_argument("--autoplot", action="store_true",
                        help="Generate plots automatically after every "
                             "retrieval query. Off by default — use the plot: "
                             "commands, or toggle with 'autoplot on'.")
    cli_args = parser.parse_args()

    print("[ELISA] Initializing...")
    engine = RetrievalEngine(base=cli_args.base, pt_name=cli_args.pt_name,
                             cells_csv=cli_args.cells_csv)
    llm = get_llm()

    # ── LOAD H5AD ──────────────────────────────────────────────
    adata = None
    cluster_key = cli_args.cluster_key
    if cli_args.h5ad and HAS_SCANPY:
        print(f"[ELISA] Loading h5ad: {cli_args.h5ad}")
        adata = sc.read_h5ad(cli_args.h5ad)
        # Remap ENSEMBL → gene symbols if needed
        if str(adata.var_names[0]).startswith("ENSG"):
            if "feature_name" in adata.var.columns:
                adata.var["ensembl_id"] = adata.var_names.copy()
                adata.var_names = adata.var["feature_name"].astype(str).values
                adata.var_names_make_unique()
                print("[ELISA] Remapped ENSEMBL → gene symbols")
        print(f"[ELISA] h5ad loaded: {adata.shape[0]} cells, "
              f"{adata.shape[1]} genes, cluster_key='{cluster_key}'")
        if "X_umap" not in adata.obsm:
            print("[WARN] No X_umap found — running sc.tl.umap()...")
            sc.pp.neighbors(adata,
                            use_rep="X_pca" if "X_pca" in adata.obsm else None)
            sc.tl.umap(adata)
            print("[ELISA] UMAP computed.")
    elif cli_args.h5ad and not HAS_SCANPY:
        print("[WARN] scanpy not installed — Nature plots disabled")
    else:
        print("[ELISA] No --h5ad provided. Nature-style plots disabled.")
        print("        Use: python elisa_chat_v4.py --h5ad /path/to/data.h5ad")

    # ── ATTACH h5ad TO ENGINE ──────────────────────────────────
    # This is what makes proportions / compare / cells: use real per-cell data
    # instead of whatever happens to be in metadata_per_cluster.
    if adata is not None and hasattr(engine, "attach_adata"):
        engine.attach_adata(adata, cluster_key=cluster_key)

    # ── CAPABILITIES ───────────────────────────────────────────
    caps = engine.detect_capabilities()
    ds_name = os.path.splitext(os.path.basename(cli_args.pt_name))[0]
    if not ds_name and engine.cluster_ids:
        ds_name = str(engine.cluster_ids[0]).split()[0]
    report = ReportBuilder(dataset_name=ds_name)

    last_payload = None
    last_answer = None
    last_plots = []
    pending_plots = []   # plots made before any analysis exists to attach them
    autoplot = cli_args.autoplot   # off unless --autoplot or 'autoplot on'

    print(f"\n[DATASET] {engine.n} clusters loaded")
    if caps.get("has_conditions"):
        print(f"[DATASET] Condition column: '{caps['condition_column']}' "
              f"→ {caps['condition_values']} "
              f"(source: {caps.get('condition_source')})")
        print("[DATASET] Comparative analysis: AVAILABLE")
    else:
        print("[DATASET] Comparative analysis: NOT AVAILABLE "
              "(no condition column)")
    if caps.get("has_proportions"):
        print(f"[DATASET] Proportions: AVAILABLE "
              f"({caps.get('total_cells')} cells)")
    else:
        print("[DATASET] Proportions: NOT AVAILABLE (no per-cell counts)")
    print(f"[DATASET] Pathways: {caps.get('n_pathways')} gene sets "
          f"across {len(caps.get('pathway_categories', {}))} categories "
          f"(marker-based proxy)")
    print(f"[DATASET] Ligand-receptor pairs: {caps.get('lr_database_size')}")
    print(f"[DATASET] hybrid: RRF late fusion, k={caps.get('rrf_k')}, "
          f"lambda_sem = {cli_args.hybrid_lambda}, pre_k = {cli_args.pre_k}")
    print(f"[DATASET] Auto-plot: {'ON' if autoplot else 'OFF'} "
          f"(use plot: commands, or 'autoplot on')")

    print(BANNER)

    while True:
        try:
            q = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("quit", "exit"):
            if report.entries:
                print(f"[ELISA] {len(report.entries)} analyses in this session "
                      f"were never written to a report.")
            break

        payload = None
        answer = None
        prompt = None
        entry_type = None

        # Everything below runs inside a guard so that one failed command
        # does not end the session and discard accumulated report entries.
        try:
            # ── VISUALIZATION ──────────────────────────────────
            if q.startswith("plot:"):
                subcmd = q.split(None, 1)[0]
                if subcmd in ("plot:umap", "plot:expr", "plot:dotplot",
                              "plot:grid"):
                    plots = handle_h5ad_viz(q, adata, cluster_key=cluster_key,
                                            plot_dir=PLOT_DIR)
                else:
                    plots = handle_viz_command(q, engine, last_payload,
                                               last_answer)
                last_plots.extend(plots)
                if plots:
                    if report.entries:
                        report.entries[-1]["plots"].extend(plots)
                    else:
                        # Nothing to attach to yet — hold for the next analysis
                        pending_plots.extend(plots)
                continue

            # ── INFO ───────────────────────────────────────────
            elif q == "info":
                print(json.dumps(engine.detect_capabilities(), indent=2,
                                 default=str))
                continue

            elif q == "clusters":
                for i, cid in enumerate(engine.cluster_ids, 1):
                    print(f"  {i}. {cid}")
                continue

            elif q == "pathways":
                from elisa_engine_compat import PATHWAY_CATEGORIES
                for cat, names in PATHWAY_CATEGORIES.items():
                    print(f"\n  [{cat}] ({len(names)})")
                    for nm in names:
                        print(f"    - {nm}")
                continue

            elif q.startswith("autoplot"):
                arg = q.split(None, 1)[1].strip().lower() if " " in q else ""
                if arg in ("on", "off"):
                    autoplot = (arg == "on")
                elif arg:
                    print("[ERROR] Usage: autoplot [on|off]")
                    continue
                print(f"[ELISA] Auto-plot is {'ON' if autoplot else 'OFF'}")
                continue

            # ── RETRIEVAL COMMANDS ─────────────────────────────
            elif q.startswith("semantic:"):
                txt = q.split(":", 1)[1].strip()
                if not txt:
                    print("[ERROR] Usage: semantic: <text>")
                    continue
                payload = engine.query_semantic(txt, top_k=5, with_genes=True)
                prompt = build_standard_prompt("semantic", txt, payload)
                entry_type = "semantic"

            elif q.startswith("hybrid:"):
                txt = q.split(":", 1)[1].strip()
                if not txt:
                    print("[ERROR] Usage: hybrid: <text>")
                    continue
                payload = engine.query_hybrid(txt, top_k=5,
                                              lambda_sem=cli_args.hybrid_lambda,
                                              pre_k=cli_args.pre_k,
                                              with_genes=True)
                print(f"[HYBRID] RRF k={payload.get('rrf_k')} | "
                      f"weights {payload.get('weights')} | "
                      f"gene pipeline "
                      f"{'active' if payload.get('gene_pipeline_active') else 'inactive'}")
                prompt = build_standard_prompt("hybrid", txt, payload)
                entry_type = "hybrid"

            elif q.startswith("union:"):
                txt = q.split(":", 1)[1].strip()
                if not txt:
                    print("[ERROR] Usage: union: <text>")
                    continue
                payload = engine.query_union(txt, top_k=5,
                                             pre_k=cli_args.pre_k,
                                             with_genes=True)
                print(f"[UNION] primary = {payload.get('primary_pipeline')} "
                      f"(by {payload.get('primary_selected_by')}), "
                      f"+{payload.get('n_appended_from_secondary')} unique "
                      f"from secondary")
                prompt = build_standard_prompt("union", txt, payload)
                entry_type = "union"

            elif q.startswith("discover:"):
                txt = q.split(":", 1)[1].strip()
                if not txt:
                    print("[ERROR] Usage: discover: <question>")
                    continue
                payload = engine.discover(txt, top_k=5,
                                          lambda_sem=cli_args.hybrid_lambda,
                                          pre_k=cli_args.pre_k)
                prompt = build_discovery_prompt(txt, payload)
                entry_type = "discovery"

            # ── COMPARE ────────────────────────────────────────
            elif q.startswith("compare:"):
                txt = q.split(":", 1)[1].strip()
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

                caps = engine.detect_capabilities()
                if caps.get("has_conditions"):
                    for cv in caps["condition_values"]:
                        if str(cv).lower() == group_a:
                            group_a = cv
                        if str(cv).lower() == group_b:
                            group_b = cv

                payload = engine.compare(group_a, group_b, genes=genes)
                prompt = build_compare_prompt(payload.get("query", txt), payload)
                entry_type = "compare"

            # ── INTERACTIONS ───────────────────────────────────
            elif q.startswith("interactions"):
                txt = q.split(":", 1)[1].strip() if ":" in q else ""
                src = None
                tgt = None
                if "->" in txt:
                    parts = txt.split("->")
                    src = parts[0].strip() or None
                    tgt = (parts[1].strip()
                           if len(parts) > 1 and parts[1].strip() else None)
                elif txt:
                    src = txt

                payload = engine.interactions(source=src, target=tgt)
                if payload.get("n_total") == 0:
                    print(f"[INFO] {payload.get('note', 'No interactions found.')}")
                prompt = build_interaction_prompt(
                    payload.get("query", "Cell-cell interactions"), payload)
                entry_type = "interactions"

            # ── PROPORTIONS ────────────────────────────────────
            elif q.startswith("proportions"):
                payload = engine.proportions()
                prompt = build_proportion_prompt(
                    payload.get("query", "Cell type proportions"), payload)
                entry_type = "proportions"

            # ── PATHWAY ────────────────────────────────────────
            elif q.startswith("pathway:"):
                txt = q.split(":", 1)[1].strip()
                if txt.lower() in ("all", ""):
                    payload = engine.pathway()
                else:
                    payload = engine.pathway(pathway_name=txt)
                prompt = build_pathway_prompt(payload.get("query", txt), payload)
                entry_type = "pathway"

            # ── DATA COMMANDS ──────────────────────────────────
            elif q.startswith("genes"):
                prefix = q.split(":", 1)[1].strip() if ":" in q else None
                all_genes = set()
                for _cid, stats in engine.gene_stats.items():
                    all_genes.update(stats.keys())
                if prefix:
                    matched = sorted(g for g in all_genes
                                     if g.upper().startswith(prefix.upper()))
                else:
                    matched = sorted(all_genes)
                print(f"{len(matched)} genes"
                      + (f" matching '{prefix}'" if prefix else ""))
                print(", ".join(matched[:100]))
                if len(matched) > 100:
                    print(f"  ... and {len(matched) - 100} more")
                continue

            elif q.startswith("metadata:"):
                cid = q.split(":", 1)[1].strip()
                print(json.dumps(engine.get_metadata(cid), indent=2,
                                 default=str))
                continue

            elif q.startswith("cells:"):
                cid = q.split(":", 1)[1].strip()
                cells = engine.get_cells(cid)
                if not cells:
                    print(f"[INFO] No cell IDs available for '{cid}'. "
                          f"Start ELISA with --h5ad or --cells-csv.")
                else:
                    print(f"{len(cells)} cells:", cells[:20])
                continue

            # ── REPORT ─────────────────────────────────────────
            elif q.startswith("report"):
                if not report.entries:
                    print("[REPORT] No analyses collected yet.")
                    continue
                fmt = q.split(":", 1)[1].strip().lower() if ":" in q else "docx"
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

            # ── PAYLOAD ERROR CHECK ────────────────────────────
            # Applied uniformly, so an engine-level error never reaches the LLM
            # dressed up as evidence.
            if isinstance(payload, dict) and "error" in payload:
                print(f"[ERROR] {payload['error']}")
                if "available" in payload:
                    print("Available:")
                    for item in payload["available"]:
                        print(f"  - {item}")
                continue

            # ── LLM CALL ───────────────────────────────────────
            if prompt and payload:
                if entry_type == "discovery":
                    answer = run_discovery(llm, prompt, payload)
                else:
                    answer = ask_llm(llm, SYSTEM_PROMPT, prompt)

                last_payload = payload
                last_answer = answer
                last_plots = []

                if (autoplot
                        and entry_type in ("semantic", "hybrid", "union",
                                           "discovery")
                        and payload.get("results")):
                    ensure_plot_dir()
                    try:
                        auto_plots = viz.auto_plot_retrieval(
                            engine, payload, save_dir=PLOT_DIR, method="umap")
                        last_plots.extend(auto_plots)
                        viz.plt.close("all")
                    except Exception as e:
                        print(f"[VIZ] Auto-plot failed: {e}")

                attach = last_plots + pending_plots
                pending_plots = []

                report.add_entry(
                    entry_type=entry_type,
                    query=payload.get("query", q),
                    payload=payload,
                    answer=answer,
                    plots=attach,
                )

                print()
                for para in answer.split("\n"):
                    print(textwrap.fill(para, width=100) if para.strip() else "")
                print("-" * 100)

                if attach:
                    print(f"[VIZ] {len(attach)} plots attached in {PLOT_DIR}/")

                print(f"[SESSION] {len(report.entries)} analyses collected. "
                      f"Type 'report' to generate.")

        except KeyboardInterrupt:
            print("\n[ELISA] Command interrupted. Session and collected "
                  "analyses are preserved.")
            continue
        except Exception as e:
            print(f"\n[ERROR] Command failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"[ELISA] Session preserved — {len(report.entries)} analyses "
                  f"still collected.")
            continue


if __name__ == "__main__":
    main()
