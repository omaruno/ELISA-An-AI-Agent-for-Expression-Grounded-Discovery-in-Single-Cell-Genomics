#!/usr/bin/env python
# ============================================================
# ELISA – Visualization Module
# ============================================================
#
# Provides publication-quality plots for:
#   1. Embedding landscapes (UMAP/t-SNE of semantic & scGPT spaces)
#   2. Cluster-level heatmaps (gene × cluster)
#   3. Retrieval result overlays (highlight retrieved clusters)
#   4. Radar / polar charts (multi-metric cluster profiles)
#   5. Gene evidence bar charts
#   6. Hybrid fusion weight diagnostics
#   7. Coverage comparison across retrieval modes
#   8. Discovery mode: consistency / mismatch visualization
#
# All functions accept data structures returned by the
# RetrievalEngine and produce matplotlib figures that can be
# saved or displayed interactively.
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; switch to "TkAgg" for live display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from typing import Optional, List, Dict, Any, Tuple

# ── Optional heavy imports (graceful fallback) ─────────────────
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Colour palette ─────────────────────────────────────────────
ELISA_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E07B54", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#999999", "#E69F00", "#88CCEE",
]

ELISA_CMAP = LinearSegmentedColormap.from_list(
    "elisa", ["#f7fbff", "#4C72B0", "#1a3a5c"], N=256
)


def _set_style():
    """Apply a clean publication style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })


_set_style()


# ================================================================
# 1. EMBEDDING LANDSCAPE (UMAP / t-SNE)
# ================================================================

def plot_embedding_landscape(
    embeddings: np.ndarray,
    cluster_ids: List[str],
    method: str = "umap",
    highlight_ids: Optional[List[str]] = None,
    title: str = "Cluster Embedding Landscape",
    point_size: int = 120,
    label_clusters: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    space_label: str = "Embedding",
) -> plt.Figure:
    """
    Project high-dimensional cluster embeddings to 2-D and plot.

    Parameters
    ----------
    embeddings : (n_clusters, d) array
    cluster_ids : list of cluster ID strings
    method : 'umap' or 'tsne'
    highlight_ids : cluster IDs to highlight (e.g. retrieved clusters)
    title : plot title
    save_path : if given, save figure to this path
    space_label : label for axes (e.g. "Semantic" or "scGPT")
    """
    n = len(cluster_ids)
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.3,
                            metric="cosine", random_state=42)
        coords = reducer.fit_transform(embeddings)
    elif method == "tsne" and HAS_TSNE:
        perp = min(30, max(2, n - 1))
        coords = TSNE(n_components=2, perplexity=perp, metric="cosine",
                      random_state=42, init="pca").fit_transform(embeddings)
    else:
        # Fallback: PCA 2-D
        from numpy.linalg import svd
        emb_c = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, _, Vt = svd(emb_c, full_matrices=False)
        coords = emb_c @ Vt[:2].T
        method = "pca"

    fig, ax = plt.subplots(figsize=figsize)

    # Base scatter
    colors = [ELISA_PALETTE[i % len(ELISA_PALETTE)] for i in range(n)]
    alphas = [0.35] * n
    sizes = [point_size * 0.7] * n

    if highlight_ids:
        hl_set = set(str(h) for h in highlight_ids)
        for i, cid in enumerate(cluster_ids):
            if str(cid) in hl_set:
                alphas[i] = 1.0
                sizes[i] = point_size * 1.5

    for i in range(n):
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=colors[i], s=sizes[i], alpha=alphas[i],
                   edgecolors="white", linewidths=0.5, zorder=3)

    if label_clusters:
        for i, cid in enumerate(cluster_ids):
            fontweight = "bold" if highlight_ids and str(cid) in set(
                str(h) for h in (highlight_ids or [])) else "normal"
            ax.annotate(str(cid),
                        (coords[i, 0], coords[i, 1]),
                        fontsize=8, fontweight=fontweight,
                        ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(f"{space_label} {method.upper()} 1")
    ax.set_ylabel(f"{space_label} {method.upper()} 2")
    ax.grid(True, alpha=0.15)

    if highlight_ids:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#C44E52', markersize=10,
                   label='Retrieved'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#8C8C8C', markersize=8,
                   alpha=0.4, label='Other'),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 2. DUAL EMBEDDING LANDSCAPE (semantic + scGPT side by side)
# ================================================================

def plot_dual_embedding(
    semantic_emb: np.ndarray,
    scgpt_emb: np.ndarray,
    cluster_ids: List[str],
    highlight_ids: Optional[List[str]] = None,
    method: str = "umap",
    title: str = "Semantic vs scGPT Embedding Space",
    figsize: Tuple[int, int] = (18, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side embedding plots for semantic and scGPT spaces."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, emb, label in zip(axes,
                               [semantic_emb, scgpt_emb],
                               ["Semantic (BioBERT)", "Expression (scGPT)"]):
        n = len(cluster_ids)
        # Reduce
        if method == "umap" and HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.3,
                                metric="cosine", random_state=42)
            coords = reducer.fit_transform(emb)
        elif method == "tsne" and HAS_TSNE:
            perp = min(30, max(2, n - 1))
            coords = TSNE(n_components=2, perplexity=perp, metric="cosine",
                          random_state=42, init="pca").fit_transform(emb)
        else:
            emb_c = emb - emb.mean(axis=0, keepdims=True)
            from numpy.linalg import svd
            _, _, Vt = svd(emb_c, full_matrices=False)
            coords = emb_c @ Vt[:2].T

        hl_set = set(str(h) for h in (highlight_ids or []))
        for i, cid in enumerate(cluster_ids):
            is_hl = str(cid) in hl_set
            ax.scatter(coords[i, 0], coords[i, 1],
                       c=ELISA_PALETTE[i % len(ELISA_PALETTE)],
                       s=160 if is_hl else 80,
                       alpha=1.0 if is_hl else 0.35,
                       edgecolors="white", linewidths=0.5, zorder=3)
            ax.annotate(str(cid), (coords[i, 0], coords[i, 1]),
                        fontsize=8, fontweight="bold" if is_hl else "normal",
                        ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points")

        ax.set_title(label, fontweight="bold")
        ax.grid(True, alpha=0.15)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 3. GENE EVIDENCE BAR CHART
# ================================================================

def plot_gene_evidence(
    results: List[Dict],
    top_n_genes: int = 15,
    metric: str = "logfc",
    title: str = "Top Gene Evidence (Retrieval Results)",
    figsize: Tuple[int, int] = (10, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top genes from retrieval results.

    Parameters
    ----------
    results : list of result dicts from engine.query_*()["results"]
    metric : 'logfc' or 'pct_in'
    """
    # Aggregate genes across retrieved clusters
    gene_pool: Dict[str, Dict] = {}
    for r in results:
        cid = r.get("cluster_id", "?")
        for g in r.get("gene_evidence", []):
            name = g["gene"]
            val = g.get(metric, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            key = name
            if key not in gene_pool or abs(float(val)) > abs(float(gene_pool[key]["val"])):
                gene_pool[key] = {"val": float(val), "cluster": cid,
                                  "pct_in": g.get("pct_in", None)}

    if not gene_pool:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No gene evidence with valid metric found",
                ha="center", va="center", fontsize=12, color="#999")
        ax.axis("off")
        return fig

    sorted_genes = sorted(gene_pool.items(),
                          key=lambda x: abs(x[1]["val"]), reverse=True)[:top_n_genes]

    names = [g[0] for g in sorted_genes][::-1]
    vals = [g[1]["val"] for g in sorted_genes][::-1]
    clusters = [str(g[1]["cluster"]) for g in sorted_genes][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in vals]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("log₂ Fold Change" if metric == "logfc" else "Fraction Expressing (pct_in)")
    ax.set_title(title, fontweight="bold")
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.15)

    # Annotate cluster source
    for i, (bar, cid) in enumerate(zip(bars, clusters)):
        ax.text(bar.get_width() + 0.02 * max(abs(v) for v in vals),
                bar.get_y() + bar.get_height() / 2,
                f"c{cid}", fontsize=7, va="center", color="#666")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 4. CLUSTER SIMILARITY HEATMAP
# ================================================================

def plot_similarity_heatmap(
    embeddings: np.ndarray,
    cluster_ids: List[str],
    highlight_ids: Optional[List[str]] = None,
    title: str = "Inter-cluster Similarity",
    figsize: Tuple[int, int] = (10, 9),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Cosine similarity heatmap among clusters."""
    sim = embeddings @ embeddings.T  # already L2-normed
    n = len(cluster_ids)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim, cmap=ELISA_CMAP, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(cluster_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(cluster_ids, fontsize=8)

    # Highlight retrieved clusters
    if highlight_ids:
        hl_set = set(str(h) for h in highlight_ids)
        for i, cid in enumerate(cluster_ids):
            if str(cid) in hl_set:
                ax.get_xticklabels()[i].set_fontweight("bold")
                ax.get_xticklabels()[i].set_color("#C44E52")
                ax.get_yticklabels()[i].set_fontweight("bold")
                ax.get_yticklabels()[i].set_color("#C44E52")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 5. RADAR CHART (multi-metric cluster profile)
# ================================================================

def plot_cluster_radar(
    results: List[Dict],
    metrics: Optional[List[str]] = None,
    title: str = "Cluster Retrieval Profile",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Radar/polar chart showing multiple retrieval metrics per cluster.
    Works best with hybrid results that have semantic + expression sims.
    """
    if metrics is None:
        metrics = ["semantic_similarity", "expression_similarity", "hybrid_similarity"]

    # Filter to metrics that actually exist
    available = []
    for m in metrics:
        if any(m in r for r in results):
            available.append(m)
    if not available:
        available = ["semantic_similarity"]

    labels = [m.replace("_", " ").title() for m in available]
    n_metrics = len(available)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, r in enumerate(results):
        vals = [float(r.get(m, 0)) for m in available]
        vals += vals[:1]
        color = ELISA_PALETTE[i % len(ELISA_PALETTE)]
        ax.plot(angles, vals, 'o-', linewidth=2, color=color,
                label=f"Cluster {r.get('cluster_id', '?')}")
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 6. RETRIEVAL MODE COVERAGE COMPARISON
# ================================================================

def plot_coverage_comparison(
    query_labels: List[str],
    semantic_cov: List[float],
    hybrid_cov: List[float],
    scgpt_cov: Optional[List[float]] = None,
    title: str = "Retrieval Coverage Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart comparing coverage across retrieval modes."""
    n = len(query_labels)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, semantic_cov, width, label="Semantic (λ=1)",
           color="#4C72B0", edgecolor="white")
    ax.bar(x, hybrid_cov, width, label="Hybrid (λ=0.5)",
           color="#55A868", edgecolor="white")
    if scgpt_cov is not None:
        ax.bar(x + width, scgpt_cov, width, label="scGPT (λ=0)",
               color="#DD8452", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(query_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Coverage (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 7. GENE × CLUSTER HEATMAP
# ================================================================

def plot_gene_cluster_heatmap(
    results: List[Dict],
    metric: str = "pct_in",
    top_n_genes: int = 20,
    title: str = "Gene Expression Across Retrieved Clusters",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of gene metric (pct_in or logfc) across retrieved clusters.
    """
    # Collect all genes and cluster data
    cluster_ids = []
    gene_data: Dict[str, Dict[str, float]] = {}  # gene -> {cid: val}
    for r in results:
        cid = str(r.get("cluster_id", "?"))
        cluster_ids.append(cid)
        for g in r.get("gene_evidence", []):
            name = g["gene"]
            val = g.get(metric, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                gene_data.setdefault(name, {})[cid] = float(val)

    if not gene_data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No gene data available", ha="center", va="center")
        ax.axis("off")
        return fig

    # Rank genes by max value across clusters
    gene_rank = sorted(gene_data.keys(),
                       key=lambda g: max(gene_data[g].values()), reverse=True)
    gene_rank = gene_rank[:top_n_genes]

    # Build matrix
    matrix = np.full((len(gene_rank), len(cluster_ids)), np.nan)
    for i, gene in enumerate(gene_rank):
        for j, cid in enumerate(cluster_ids):
            matrix[i, j] = gene_data.get(gene, {}).get(cid, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=ELISA_CMAP, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f"c{c}" for c in cluster_ids], fontsize=9, rotation=0)
    ax.set_yticks(range(len(gene_rank)))
    ax.set_yticklabels(gene_rank, fontsize=9)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Gene")
    ax.set_title(title, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric.replace("_", " ").title())

    # Annotate values
    for i in range(len(gene_rank)):
        for j in range(len(cluster_ids)):
            if not np.isnan(matrix[i, j]):
                txt = f"{matrix[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > 0.6 else "#333")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 8. RETRIEVAL SIMILARITY WATERFALL
# ================================================================

def plot_similarity_waterfall(
    results: List[Dict],
    sim_key: str = "hybrid_similarity",
    title: str = "Retrieval Similarity Ranking",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Waterfall / lollipop chart of similarity scores for retrieved clusters."""
    cids = [str(r.get("cluster_id", "?")) for r in results]
    sims = [float(r.get(sim_key, r.get("semantic_similarity", 0))) for r in results]

    fig, ax = plt.subplots(figsize=figsize)
    colors = [ELISA_PALETTE[i % len(ELISA_PALETTE)] for i in range(len(cids))]

    ax.barh(range(len(cids)), sims, color=colors, edgecolor="white",
            linewidth=0.5, height=0.6)

    ax.set_yticks(range(len(cids)))
    ax.set_yticklabels([f"Cluster {c}" for c in cids], fontsize=10)
    ax.set_xlabel("Similarity Score")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    ax.invert_yaxis()

    for i, v in enumerate(sims):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9, color="#555")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 9. SEMANTIC vs EXPRESSION SCATTER (per-query diagnostic)
# ================================================================

def plot_sem_vs_expr_scatter(
    results_hybrid: List[Dict],
    all_cluster_ids: List[str],
    all_sem_sims: np.ndarray,
    all_expr_sims: np.ndarray,
    title: str = "Semantic vs Expression Similarity",
    figsize: Tuple[int, int] = (9, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter all clusters by semantic vs expression similarity,
    highlighting the retrieved ones.
    """
    hl_set = set(str(r["cluster_id"]) for r in results_hybrid)

    fig, ax = plt.subplots(figsize=figsize)

    for i, cid in enumerate(all_cluster_ids):
        is_hl = str(cid) in hl_set
        ax.scatter(all_sem_sims[i], all_expr_sims[i],
                   c="#C44E52" if is_hl else "#8C8C8C",
                   s=140 if is_hl else 50,
                   alpha=1.0 if is_hl else 0.3,
                   edgecolors="white", linewidths=0.5, zorder=3 if is_hl else 2)
        if is_hl:
            ax.annotate(str(cid), (all_sem_sims[i], all_expr_sims[i]),
                        fontsize=9, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Semantic Similarity")
    ax.set_ylabel("Expression Similarity (scGPT)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.15)

    # Diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, '--', color="#999", alpha=0.5, linewidth=1)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#C44E52',
               markersize=10, label='Retrieved'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8C8C8C',
               markersize=7, alpha=0.4, label='Other'),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 10. DISCOVERY MODE – EXPECTED vs UNEXPECTED GENE SPLIT
# ================================================================

def plot_discovery_split(
    expected_genes: List[str],
    unexpected_genes: List[str],
    title: str = "Discovery Mode: Expected vs Context-Shifted Genes",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visual summary of genes classified as expected vs context-shifted."""
    fig, ax = plt.subplots(figsize=figsize)

    all_genes = expected_genes + unexpected_genes
    colors = (["#55A868"] * len(expected_genes) +
              ["#C44E52"] * len(unexpected_genes))
    labels_cat = (["Expected"] * len(expected_genes) +
                  ["Context-shifted"] * len(unexpected_genes))

    y = range(len(all_genes))
    ax.barh(y, [1] * len(all_genes), color=colors, edgecolor="white",
            linewidth=0.5, height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(all_genes, fontsize=9)
    ax.set_xticks([])
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#55A868',
               markersize=10, label='Matches Established Biology'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#C44E52',
               markersize=10, label='Context-Shifted / Unexpected'),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 11. HYBRID WEIGHT DIAGNOSTIC (lambda sweep)
# ================================================================

def plot_lambda_sweep(
    lambdas: List[float],
    coverages: List[float],
    title: str = "Hybrid λ Sweep – Coverage vs Semantic Weight",
    figsize: Tuple[int, int] = (9, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot coverage as a function of λ for hybrid retrieval."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lambdas, coverages, 'o-', color="#4C72B0", linewidth=2, markersize=8)
    ax.fill_between(lambdas, coverages, alpha=0.1, color="#4C72B0")
    ax.set_xlabel("λ (Semantic Weight)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)

    # Mark key points
    best_idx = int(np.argmax(coverages))
    ax.annotate(f"Best: λ={lambdas[best_idx]:.2f}\n({coverages[best_idx]:.0f}%)",
                (lambdas[best_idx], coverages[best_idx]),
                fontsize=10, fontweight="bold",
                xytext=(15, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#333"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# CONVENIENCE: Quick plot from engine + payload
# ================================================================

def auto_plot_retrieval(engine, payload: Dict, save_dir: str = ".",
                        method: str = "umap") -> List[str]:
    """
    Given an engine instance and a retrieval payload, produce a standard
    set of plots and return a list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    mode = payload.get("mode", "unknown")
    query = payload.get("query", "query")
    results = payload.get("results", [])
    retrieved_ids = [str(r["cluster_id"]) for r in results]

    prefix = f"{save_dir}/elisa_{mode}"

    # 1. Embedding landscape (semantic)
    p = f"{prefix}_semantic_landscape.png"
    plot_embedding_landscape(
        engine.semantic_emb, engine.cluster_ids,
        method=method, highlight_ids=retrieved_ids,
        title=f"Semantic Space – {query[:50]}",
        space_label="Semantic", save_path=p)
    saved.append(p)
    plt.close()

    # 2. Embedding landscape (scGPT)
    p = f"{prefix}_scgpt_landscape.png"
    plot_embedding_landscape(
        engine.scgpt_emb, engine.cluster_ids,
        method=method, highlight_ids=retrieved_ids,
        title=f"scGPT Space – {query[:50]}",
        space_label="scGPT", save_path=p)
    saved.append(p)
    plt.close()

    # 3. Gene evidence
    p = f"{prefix}_gene_evidence.png"
    plot_gene_evidence(results, title=f"Gene Evidence – {query[:50]}", save_path=p)
    saved.append(p)
    plt.close()

    # 4. Similarity waterfall
    sim_key = ("hybrid_similarity" if mode == "hybrid"
               else "semantic_similarity" if mode == "semantic"
               else "hybrid_similarity")
    p = f"{prefix}_similarity.png"
    plot_similarity_waterfall(results, sim_key=sim_key,
                              title=f"Similarity Ranking – {mode.title()}",
                              save_path=p)
    saved.append(p)
    plt.close()

    # 5. Gene × cluster heatmap
    p = f"{prefix}_gene_cluster_heatmap.png"
    plot_gene_cluster_heatmap(results,
                              title=f"Gene × Cluster – {query[:50]}",
                              save_path=p)
    saved.append(p)
    plt.close()

    # 6. Radar chart (hybrid only)
    if mode in ("hybrid", "discovery"):
        p = f"{prefix}_radar.png"
        plot_cluster_radar(results, title=f"Cluster Profile – {query[:50]}",
                           save_path=p)
        saved.append(p)
        plt.close()

    print(f"[ELISA-VIZ] Generated {len(saved)} plots in {save_dir}/")
    return saved


# ================================================================
# 12. NATURE-STYLE CELL-LEVEL UMAP (like Nat Commun panel a)
# ================================================================

# Extended palette for up to 40 clusters — distinct, colorblind-friendly
NATURE_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC9A6C", "#7E6148", "#B09C85",
    "#E377C2", "#BCBD22", "#17BECF", "#AEC7E8", "#FFBB78",
    "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2",
    "#DBDB8D", "#9EDAE5", "#FF7F0E", "#2CA02C", "#D62728",
    "#9467BD", "#8C564B", "#E7969C", "#7F7F7F", "#CEDB9C",
    "#5254A3", "#6B6ECF", "#9C9EDE", "#637939", "#8CA252",
    "#B5CF6B", "#BD9E39", "#E7BA52", "#E7CB94", "#843C39",
]


def plot_cell_umap(
    adata,
    cluster_key: str = "cell_type",
    highlight_clusters: Optional[List[str]] = None,
    title: str = "",
    point_size: float = 0.3,
    label_fontsize: int = 10,
    label_style: str = "offset",  # "offset" or "centroid"
    palette: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    show_legend: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Publication-quality cell-level UMAP, matching Nature/Cell style.

    Reads X_umap from adata.obsm (pre-computed). Each cell is a tiny dot,
    colored by cluster. Cluster labels are placed at centroids with a
    clean offset or direct annotation style.

    Parameters
    ----------
    adata : AnnData with obsm['X_umap'] and obs[cluster_key]
    cluster_key : obs column for coloring
    highlight_clusters : if set, grey out all other clusters
    title : plot title (Nature style = no title or minimal)
    point_size : dot size per cell (0.1–1.0 for large datasets)
    label_fontsize : cluster label font size
    label_style : 'offset' = label with leader line, 'centroid' = label on top
    palette : custom color list; defaults to NATURE_PALETTE
    figsize : figure dimensions
    show_legend : add a color legend (usually False for Nature style)
    save_path : save to file
    dpi : resolution
    """
    # --- Get UMAP coordinates ---
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] not found. Run sc.tl.umap first.")

    coords = np.array(adata.obsm["X_umap"])
    clusters = adata.obs[cluster_key].astype(str).values
    unique_clusters = sorted(set(clusters))
    n_clusters = len(unique_clusters)

    if palette is None:
        palette = NATURE_PALETTE
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}

    # --- Highlight mode: grey out non-highlighted clusters ---
    if highlight_clusters:
        hl_set = set(str(h) for h in highlight_clusters)
        cell_colors = []
        cell_alphas = []
        cell_order = []
        for i, c in enumerate(clusters):
            if c in hl_set:
                cell_colors.append(color_map[c])
                cell_alphas.append(0.8)
                cell_order.append(1)
            else:
                cell_colors.append("#E0E0E0")
                cell_alphas.append(0.15)
                cell_order.append(0)
        # Draw grey cells first, colored on top
        order = np.argsort(cell_order)
    else:
        cell_colors = [color_map[c] for c in clusters]
        cell_alphas = [0.7] * len(clusters)
        order = np.random.permutation(len(clusters))  # shuffle to avoid overlap bias

    # --- Figure setup (Nature style: clean, minimal) ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # --- Plot cells ---
    ax.scatter(
        coords[order, 0], coords[order, 1],
        c=[cell_colors[i] for i in order],
        s=point_size,
        alpha=[cell_alphas[i] for i in order],
        edgecolors="none",
        rasterized=True,  # rasterize for smaller file size
    )

    # --- Compute centroids and place labels ---
    centroids = {}
    for c in unique_clusters:
        mask = clusters == c
        cx = coords[mask, 0].mean()
        cy = coords[mask, 1].mean()
        centroids[c] = (cx, cy)

    # Determine which clusters to label
    if highlight_clusters:
        label_clusters = [c for c in unique_clusters if c in hl_set]
    else:
        label_clusters = unique_clusters

    if label_style == "offset":
        _place_labels_offset(ax, centroids, label_clusters, color_map,
                             label_fontsize, coords)
    else:
        # Direct centroid labels
        for c in label_clusters:
            cx, cy = centroids[c]
            # Shorten long CellOntology names for display
            display_name = _shorten_cell_type(c)
            ax.annotate(
                display_name,
                (cx, cy),
                fontsize=label_fontsize,
                fontweight="bold",
                color=color_map[c],
                ha="center", va="center",
                path_effects=[
                    matplotlib.patheffects.withStroke(
                        linewidth=2.5, foreground="white"
                    )
                ],
            )

    # --- Nature style: remove axes, ticks, grid ---
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # --- Optional legend ---
    if show_legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=color_map[c], label=_shorten_cell_type(c))
                   for c in unique_clusters]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False, ncol=1 if n_clusters <= 20 else 2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


def _shorten_cell_type(name: str, max_len: int = 25) -> str:
    """Shorten Cell Ontology names for plot labels."""
    # Common replacements
    replacements = {
        "CD8-positive, alpha-beta T cell": "CD8+ T",
        "CD4-positive helper T cell": "CD4+ T helper",
        "CD4-positive, alpha-beta T cell": "CD4+ T",
        "respiratory tract multiciliated cell": "multiciliated",
        "respiratory tract suprabasal cell": "suprabasal",
        "respiratory tract goblet cell": "resp. goblet",
        "mucus secreting cell of bronchus submucosal gland": "submucosal gland",
        "nasal mucosa goblet cell": "nasal goblet",
        "alveolar adventitial fibroblast": "adventitial fib.",
        "alveolar type 1 fibroblast cell": "AT1 fibroblast",
        "bronchial goblet cell": "bronchial goblet",
        "pulmonary neuroendocrine cell": "neuroendocrine",
        "non-classical monocyte": "non-class. mono.",
        "epithelial cell of lung": "lung epithelial",
        "fibroblast of lung": "lung fibroblast",
        "dendritic cell, human": "dendritic",
        "innate lymphoid cell": "ILC",
        "natural killer cell": "NK",
        "endocardial cell": "endocardial",
        "secretory cell": "secretory",
        "stromal cell": "stromal",
        "mature T cell": "mature T",
        "cytotoxic T cell": "cytotoxic T",
        "plasma cell": "plasma",
        "mast cell": "mast",
        "club cell": "club",
        "basal cell": "basal",
        "macrophage": "macrophage",
        "monocyte": "monocyte",
        "pericyte": "pericyte",
        "ionocyte": "ionocyte",
        "B cell": "B cell",
    }

    if name in replacements:
        return replacements[name]

    # Generic shortening
    if len(name) > max_len:
        return name[:max_len - 1] + "."
    return name


def _place_labels_offset(ax, centroids, label_clusters, color_map,
                          fontsize, coords):
    """
    Place labels with leader lines pointing to centroids,
    avoiding overlap using a simple repulsion algorithm.
    Mimics the Nature figure panel (a) label placement.
    """
    import matplotlib.patheffects as pe

    # Get plot bounds
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    xrange = xmax - xmin
    yrange = ymax - ymin

    # Initial label positions: offset from centroid
    label_pos = {}
    for c in label_clusters:
        cx, cy = centroids[c]
        # Push label outward from center of mass
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        dx = (cx - center_x) / xrange * 0.15 * xrange
        dy = (cy - center_y) / yrange * 0.15 * yrange
        label_pos[c] = (cx + dx, cy + dy)

    # Simple repulsion to reduce overlaps (5 iterations)
    positions = list(label_pos.values())
    keys = list(label_pos.keys())
    for _ in range(8):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                xi, yi = positions[i]
                xj, yj = positions[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_dist = 0.06 * max(xrange, yrange)
                if dist < min_dist and dist > 0:
                    # Push apart
                    force = (min_dist - dist) / 2
                    dx = (xi - xj) / dist * force
                    dy = (yi - yj) / dist * force
                    positions[i] = (xi + dx, yi + dy)
                    positions[j] = (xj - dx, yj - dy)

    # Draw labels with leader lines
    for i, c in enumerate(keys):
        cx, cy = centroids[c]
        lx, ly = positions[i]
        display_name = _shorten_cell_type(c)

        # Leader line
        ax.annotate(
            "",
            xy=(cx, cy),
            xytext=(lx, ly),
            arrowprops=dict(
                arrowstyle="-",
                color="#555555",
                linewidth=0.6,
                shrinkA=3,
                shrinkB=3,
            ),
        )

        # Label text
        ax.text(
            lx, ly, display_name,
            fontsize=fontsize,
            fontweight="bold",
            color=color_map[c],
            ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )


# ================================================================
# 13. GENE EXPRESSION UMAP (like Nature panel d)
# ================================================================

def plot_gene_expression_umap(
    adata,
    gene: str,
    cluster_key: str = "cell_type",
    title: Optional[str] = None,
    cmap: str = "Purples",
    point_size: float = 0.3,
    figsize: Tuple[int, int] = (8, 8),
    vmax_percentile: float = 98,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    UMAP colored by expression of a single gene.
    Non-expressing cells shown in light grey, expressing cells
    on a purple gradient (matching Nature Fig 2d style).

    Parameters
    ----------
    adata : AnnData with obsm['X_umap']
    gene : gene symbol to plot
    cmap : colormap (default 'Purples' to match the figure)
    vmax_percentile : cap color scale at this percentile of nonzero values
    """
    from scipy import sparse as sp

    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] not found.")

    if gene not in adata.var_names:
        # Try case-insensitive search
        matches = [g for g in adata.var_names if g.upper() == gene.upper()]
        if matches:
            gene = matches[0]
        else:
            raise ValueError(f"Gene '{gene}' not found in adata.var_names")

    coords = np.array(adata.obsm["X_umap"])
    g_idx = list(adata.var_names).index(gene)
    X = adata.X
    if sp.issparse(X):
        expr = np.asarray(X[:, g_idx].todense()).flatten()
    else:
        expr = X[:, g_idx].flatten()

    if title is None:
        title = gene

    # --- Color: grey for zero, purple gradient for expressing ---
    nonzero_mask = expr > 0
    vmax = np.percentile(expr[nonzero_mask], vmax_percentile) if nonzero_mask.sum() > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # Plot non-expressing cells first (grey)
    ax.scatter(
        coords[~nonzero_mask, 0], coords[~nonzero_mask, 1],
        c="#E8E8E8", s=point_size, alpha=0.3,
        edgecolors="none", rasterized=True,
    )

    # Plot expressing cells on top
    sc_plot = ax.scatter(
        coords[nonzero_mask, 0], coords[nonzero_mask, 1],
        c=expr[nonzero_mask],
        cmap=cmap,
        vmin=0, vmax=vmax,
        s=point_size * 1.5,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
    )

    # Colorbar
    cbar = fig.colorbar(sc_plot, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=9)

    # Nature style: minimal
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 14. MULTI-GENE EXPRESSION UMAP GRID (Nature Fig 2d style)
# ================================================================

def plot_gene_expression_grid(
    adata,
    genes: List[str],
    cluster_key: str = "cell_type",
    ncols: int = 3,
    cmap: str = "Purples",
    point_size: float = 0.15,
    figsize_per_panel: Tuple[float, float] = (4, 4),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Grid of gene expression UMAPs, like Nature Fig 2d.
    Each panel shows expression of one gene.
    """
    from scipy import sparse as sp

    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] not found.")

    coords = np.array(adata.obsm["X_umap"])
    X = adata.X

    # Filter to valid genes
    valid_genes = []
    for g in genes:
        if g in adata.var_names:
            valid_genes.append(g)
        else:
            matches = [v for v in adata.var_names if v.upper() == g.upper()]
            if matches:
                valid_genes.append(matches[0])
            else:
                print(f"[WARN] Gene '{g}' not found, skipping")

    if not valid_genes:
        raise ValueError("No valid genes found")

    nrows = int(np.ceil(len(valid_genes) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, gene in enumerate(valid_genes):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        g_idx = list(adata.var_names).index(gene)
        if sp.issparse(X):
            expr = np.asarray(X[:, g_idx].todense()).flatten()
        else:
            expr = X[:, g_idx].flatten()

        nonzero_mask = expr > 0
        vmax = (np.percentile(expr[nonzero_mask], 98)
                if nonzero_mask.sum() > 10 else 1.0)

        # Grey background cells
        ax.scatter(coords[~nonzero_mask, 0], coords[~nonzero_mask, 1],
                   c="#E8E8E8", s=point_size, alpha=0.3,
                   edgecolors="none", rasterized=True)

        # Expressing cells
        sc_plot = ax.scatter(
            coords[nonzero_mask, 0], coords[nonzero_mask, 1],
            c=expr[nonzero_mask], cmap=cmap, vmin=0, vmax=vmax,
            s=point_size * 1.5, alpha=0.8,
            edgecolors="none", rasterized=True)

        # Colorbar
        cbar = fig.colorbar(sc_plot, ax=ax, shrink=0.6, pad=0.02, aspect=20)
        cbar.ax.tick_params(labelsize=7)

        # Clean axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(gene, fontsize=13, fontweight="bold")

    # Hide empty panels
    for idx in range(len(valid_genes), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig


# ================================================================
# 15. DOT PLOT (like Nature panel c)
# ================================================================

def plot_dotplot(
    adata,
    genes: List[str],
    cluster_key: str = "cell_type",
    group_order: Optional[List[str]] = None,
    title: str = "",
    cmap: str = "Blues",
    max_dot_size: float = 200,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Dot plot of gene expression across clusters.
    Dot size = percent expressing, dot color = mean expression.
    Matches Nature Fig 2c style.

    Parameters
    ----------
    adata : AnnData
    genes : list of gene names (shown as rows)
    cluster_key : obs column for groups (shown as columns)
    group_order : optional custom ordering of clusters
    """
    from scipy import sparse as sp

    clusters = adata.obs[cluster_key].astype(str).values
    if group_order is None:
        group_order = sorted(set(clusters))

    X = adata.X
    var_names = list(adata.var_names)

    # Filter valid genes
    valid_genes = []
    for g in genes:
        if g in var_names:
            valid_genes.append(g)
        else:
            matches = [v for v in var_names if v.upper() == g.upper()]
            if matches:
                valid_genes.append(matches[0])

    if not valid_genes:
        raise ValueError("No valid genes found")

    # Compute mean expression and percent expressing
    n_genes = len(valid_genes)
    n_groups = len(group_order)
    mean_expr = np.zeros((n_genes, n_groups))
    pct_expr = np.zeros((n_genes, n_groups))

    for j, cl in enumerate(group_order):
        mask = clusters == cl
        if mask.sum() == 0:
            continue
        for i, gene in enumerate(valid_genes):
            g_idx = var_names.index(gene)
            if sp.issparse(X):
                col = np.asarray(X[mask, g_idx].todense()).flatten()
            else:
                col = X[mask, g_idx].flatten()
            pct_expr[i, j] = (col > 0).mean() * 100
            mean_expr[i, j] = col.mean()

    # Z-score mean expression per gene (for color scaling like the Nature fig)
    for i in range(n_genes):
        row = mean_expr[i, :]
        if row.std() > 0:
            mean_expr[i, :] = (row - row.mean()) / row.std()
        else:
            mean_expr[i, :] = 0

    # --- Figure ---
    if figsize is None:
        figsize = (max(4, n_groups * 0.9 + 2), max(4, n_genes * 0.4 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    # Dot grid
    norm = Normalize(vmin=mean_expr.min(), vmax=mean_expr.max())
    size_norm = pct_expr.max() if pct_expr.max() > 0 else 1.0

    for i in range(n_genes):
        for j in range(n_groups):
            size = (pct_expr[i, j] / size_norm) * max_dot_size
            color = plt.get_cmap(cmap)(norm(mean_expr[i, j]))
            ax.scatter(j, i, s=size, c=[color], edgecolors="#333333",
                       linewidths=0.3, zorder=3)

    # Labels
    short_groups = [_shorten_cell_type(c) for c in group_order]
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(short_groups, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(valid_genes, fontsize=10)
    ax.invert_yaxis()

    # Grid lines (light, Nature style)
    for i in range(n_genes):
        ax.axhline(i, color="#EEEEEE", linewidth=0.5, zorder=1)
    for j in range(n_groups):
        ax.axvline(j, color="#EEEEEE", linewidth=0.5, zorder=1)

    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(n_genes - 0.5, -0.5)

    # Remove box
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Size legend
    size_legend_pcts = [25, 50, 75]
    size_handles = []
    for pct in size_legend_pcts:
        s = (pct / size_norm) * max_dot_size
        size_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
                   markersize=np.sqrt(s), label=f"{pct}%")
        )
    leg1 = ax.legend(handles=size_handles, title="% Expressing",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)

    # Color legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.15, aspect=10)
    cbar.set_label("Avg Expression\n(z-scored)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        print(f"[ELISA-VIZ] Saved → {save_path}")
    return fig
