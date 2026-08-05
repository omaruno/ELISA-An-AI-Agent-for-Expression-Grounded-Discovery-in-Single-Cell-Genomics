# ELISA — Embedding-Linked Interactive Single-cell Agent

An interpretable hybrid generative AI agent for expression-grounded discovery in single-cell genomics.

ELISA unifies scGPT expression embeddings with BioBERT-based semantic retrieval and LLM-mediated interpretation for interactive single-cell atlas interrogation. An automatic query classifier routes inputs to gene marker scoring, semantic matching, or reciprocal rank fusion pipelines, while integrated modules perform pathway scoring, ligand–receptor interaction prediction, comparative analysis, and proportion estimation.

Iam Currently looking for Job Opportunities, Post-Doc or Industry, feel free do drop me a message, omarcoser10@gmail.com
> **Paper Submitted (Currently Under Review):** Coser O. — *ELISA: An Interpretable Hybrid Generative AI Agent for Expression-Grounded Discovery in Single-Cell Genomics.* .https://arxiv.org/abs/2603.11872
---

## Quick Start (3-Step Pipeline)

```
.h5ad  ──▶  Step 1: scGPT embeddings  ──▶  Step 2: ELISA .pt file  ──▶  Step 3: Interactive chat
```

### Step 1: Generate scGPT Embeddings

Generate per-cell CLS embeddings from the scGPT foundation model, then aggregate them into per-cluster centroids.

```bash
python generate_scgpt_embeddings.py \
    --h5ad /path/to/dataset.h5ad \
    --cluster-key cell_type \
    --model-dir /path/to/scgpt_model/ \
    --name MyDataset \
    --out /path/to/output/ \
    --batch-size 64
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--h5ad` | Input `.h5ad` file | *required* |
| `--cluster-key` | Column in `adata.obs` with cell type labels | auto-detected |
| `--model-dir` | Directory with scGPT model files | `/lustre/home/ocoser/aiagents/scgptHuman` |
| `--batch-size` | Cells per inference batch | `64` |
| `--max-tokens` | Maximum gene tokens per cell | `3000` |
| `--save-cell-emb` | Also save per-cell embeddings as `.npy` | `False` |

**Output:**
```
output/
├── scgpt_cluster_embeddings_by_cell_type_MyDataset.pt   # ← needed for Step 2
└── metadata_cells_MyDataset.csv
```

> **Note:** If your `.h5ad` uses ENSEMBL IDs (e.g., `ENSG00000...`), the script will automatically remap them to gene symbols using `gene_info.csv` from the model directory or `adata.var["feature_name"]`.

---

### Step 2: Create ELISA Embedding File

Build the unified `.pt` file containing BioBERT semantic embeddings, scGPT expression embeddings, gene statistics, GO/Reactome enrichment, and metadata.

```bash
python create_embeddings_v3.py \
    --h5ad /path/to/dataset.h5ad \
    --cluster-key cell_type \
    --scgpt-pt /path/to/output/scgpt_cluster_embeddings_by_cell_type_MyDataset.pt \
    --name MyDataset \
    --out /path/to/output/
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--h5ad` | Input `.h5ad` file (same as Step 1) | *required* |
| `--cluster-key` | Cell type annotation column | auto-detected |
| `--scgpt-pt` | scGPT cluster embeddings from Step 1 | `None` (uses PCA fallback) |
| `--alpha` | Identity vs. context weight for semantic embeddings | `0.6` |
| `--n-top-genes-text` | Marker genes in cluster text summaries | `400` |
| `--n-top-genes-stats` | Maximum genes stored per cluster | `10000` |
| `--skip-enrichment` | Skip GO/Reactome enrichment (faster) | `False` |

**Output:**
```
output/
├── hybrid_v3_MyDataset.pt    # ← the unified ELISA embedding file
└── metadata_cells_MyDataset.csv
```

**What's inside the `.pt` file:**
- Cluster IDs and text descriptions
- BioBERT semantic embeddings (768-d, L2-normalized)
- scGPT expression embeddings (512-d, L2-normalized)
- Per-cluster gene statistics (log₂FC, pct_in, pct_out, padj)
- Inverted gene index for O(1) lookups
- GO and Reactome enrichment terms
- Cluster metadata and synonym mappings

---

### Step 3: Launch the Interactive Chat

Start the ELISA chat interface for interactive querying, analysis, and report generation.
```bash
# Set your LLM API key (Groq is free)
export GROQ_API_KEY=your-key-here

# Launch ELISA (default settings — good for Groq free tier)
python elisa_chat_v4.py \
  --base   /path/to/embeddings/Fibrosi_Cistica \
  --pt-name hybrid_v3_DT1_FibrosiCistica.pt \
  --h5ad   /path/to/Fibrosi_cistica.h5ad \
  --cluster-key cell_type \
  --hybrid-lambda 0.5 \
  --pre-k 40

# For larger models (GPT-4o, Claude) — increase token limits:
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key
export LLM_MAX_OUTPUT_TOKENS=4096
export LLM_MAX_INPUT_CHARS=60000
python elisa_chat_v4.py --h5ad /path/to/dataset.h5ad --cluster-key cell_type

# For local/open-source models (Ollama, vLLM) — reduce limits:
export LLM_PROVIDER=openai
export OPENAI_API_KEY=dummy
export LLM_MODEL=llama3:8b
export LLM_MAX_OUTPUT_TOKENS=512
export LLM_MAX_INPUT_CHARS=8000
python elisa_chat_v4.py --h5ad /path/to/dataset.h5ad --cluster-key cell_type
```

> **Important:** The `.pt` file from Step 2 must be in the current directory, or configure the `RetrievalEngine` path accordingly.

| Argument | Description | Default |
|----------|-------------|---------|
| `--h5ad` | Original `.h5ad` for Nature-style cell plots | `None` (plots disabled) |
| `--cluster-key` | Cell type column (must match Steps 1–2) | `cell_type` |

**Token and generation settings** (environment variables):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MAX_OUTPUT_TOKENS` | Max tokens in LLM response | `1024` |
| `LLM_MAX_INPUT_CHARS` | Max chars in user prompt (~4 chars ≈ 1 token) | `18000` |
| `LLM_MAX_CONTEXT_CHARS` | Context payload limit (auto: 2/3 of input) | `12000` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.2` |

**Example session:**

```
# ─── ispezione dataset ───────────────────────────────────
info
clusters
pathways
genes: KLR

# ─── retrieval ───────────────────────────────────────────
semantic: macrophage infiltration in CF airways
semantic: HLA-E CD94 NKG2A immune checkpoint inhibiting CD8 T cell activity
hybrid: CD8A CD8B GZMB PRF1 IFNG NKG7
hybrid: MARCO FABP4 APOC1 C1QB C1QC MSR1
union: HLA-E NKG2A immune checkpoint and CD8 T cell inhibition in CF
discover: HLA-E NKG2A immune checkpoint in cystic fibrosis

# ─── moduli analitici ────────────────────────────────────
proportions
compare: cystic fibrosis vs normal
compare: cystic fibrosis vs normal | IFNG, CD69, HLA-E, KLRC1, KLRD1
interactions
interactions: macrophage -> CD8-positive, alpha-beta T cell
pathway: IFN-gamma signaling
pathway: cytotoxicity
pathway: all

# ─── visualizzazione ─────────────────────────────────────
plot:umap
plot:umap CD8-positive, alpha-beta T cell, natural killer cell
plot:expr HLA-E
plot:dotplot IFNG, GZMB, PRF1, NKG7, HLA-E, KLRC1
plot:genes
plot:waterfall
plot:scatter
plot:lambda_sweep CD8A GZMB PRF1 NKG7

# ─── report ──────────────────────────────────────────────
report: md
report
quit
```

---

## LLM Provider Configuration

ELISA supports four LLM providers. Set via environment variables:

```bash
# Groq (default, free tier — 500K tokens/day)
export LLM_PROVIDER=groq
export GROQ_API_KEY=your-key

# Google Gemini
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your-key

# OpenAI (ChatGPT)
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key

# Anthropic (Claude)
export LLM_PROVIDER=claude
export ANTHROPIC_API_KEY=your-key
export LLM_MAX_SPEND_EUR=5.0   # raise cap for Claude's higher pricing
```

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider name | `groq` |
| `LLM_MODEL` | Override default model name | provider-specific |
| `LLM_MAX_SPEND_EUR` | Hard spending cap in EUR | `1.0` |

A built-in spending tracker with overestimated costs refuses calls once the limit is reached, independent of any cloud-side budgets.

---

## Citation

If you use ELISA in your research, please cite the arXiv or the preliminary version published at ICLR Workshop Generative AI in Genomics:

```bibtex
@article{coser2026elisa,
  title={ELISA: An Interpretable Hybrid Generative AI Agent for Expression-Grounded Discovery in Single-Cell Genomics},
  author={Coser, Omar},
  journal={arXiv preprint arXiv:2603.11872},
  year={2026}
}



