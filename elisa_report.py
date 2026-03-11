#!/usr/bin/env python
# ============================================================
# ELISA – Structured Report Generator
# ============================================================
#
# Generates publication-style reports (docx) from ELISA analysis
# sessions. Collects all queries, results, LLM interpretations,
# and plots into a structured document with:
#   - Title page
#   - Abstract (auto-generated)
#   - Results (organized by analysis type)
#   - Discussion & Future Perspectives
#   - Methods
#   - Supplementary Tables
#   - Figures with captions
#
# Uses docx-js via Node.js (per SKILL.md best practices).
# Falls back to python-docx if Node.js unavailable.
# ============================================================

import os
import json
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any


class ReportBuilder:
    """
    Accumulates analysis results throughout a session,
    then generates a structured report.
    """

    def __init__(self, dataset_name: str = "Dataset"):
        self.dataset_name = dataset_name
        self.entries = []       # list of {type, query, payload, answer, plots}
        self.timestamp = datetime.now()

    def add_entry(self, entry_type: str, query: str,
                  payload: Dict, answer: str,
                  plots: Optional[List[str]] = None):
        """
        Add an analysis result to the report.

        entry_type: 'semantic', 'hybrid', 'discovery', 'compare',
                    'interactions', 'proportions', 'pathway'
        """
        self.entries.append({
            "type": entry_type,
            "query": query,
            "payload": payload,
            "answer": answer,
            "plots": plots or [],
            "timestamp": datetime.now().isoformat(),
        })

    def generate_markdown(self, output_path: str = None,
                          llm_func=None) -> str:
        """
        Generate a Markdown report. Can be converted to docx via pandoc.
        If llm_func is provided, uses it to generate abstract and discussion.
        """
        if output_path is None:
            ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = f"elisa_report_{ts}.md"

        sections = []

        # --- Title ---
        sections.append(f"# ELISA Analysis Report: {self.dataset_name}\n")
        sections.append(f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M')}\n")
        sections.append(f"**Analyses performed:** {len(self.entries)}\n")
        sections.append("---\n")

        # --- Abstract ---
        abstract = self._build_abstract(llm_func)
        sections.append("## Abstract\n")
        sections.append(abstract + "\n")

        # --- Results ---
        sections.append("## Results\n")
        for i, entry in enumerate(self.entries):
            sections.append(self._format_result_section(i + 1, entry))

        # --- Discussion ---
        discussion = self._build_discussion(llm_func)
        sections.append("## Discussion\n")
        sections.append(discussion + "\n")

        # --- Future Perspectives ---
        perspectives = self._build_perspectives(llm_func)
        sections.append("## Future Perspectives\n")
        sections.append(perspectives + "\n")

        # --- Methods ---
        sections.append("## Methods\n")
        sections.append(self._build_methods() + "\n")

        # --- Supplementary Tables ---
        sections.append("## Supplementary Tables\n")
        for i, entry in enumerate(self.entries):
            supp = self._format_supplementary(i + 1, entry)
            if supp:
                sections.append(supp)

        # --- Figures ---
        all_plots = []
        for entry in self.entries:
            all_plots.extend(entry.get("plots", []))
        if all_plots:
            sections.append("## Figures\n")
            for i, p in enumerate(all_plots):
                fname = os.path.basename(p)
                sections.append(f"**Figure {i+1}.** {fname}\n")
                sections.append(f"![{fname}]({p})\n")

        report = "\n".join(sections)

        with open(output_path, "w") as f:
            f.write(report)

        print(f"[REPORT] Saved Markdown: {output_path}")
        return output_path

    def generate_docx(self, output_path: str = None,
                      llm_func=None) -> str:
        """
        Generate a structured .docx report.
        Strategy: generate Markdown first, then convert with pandoc.
        Falls back to plain Markdown if pandoc unavailable.
        """
        if output_path is None:
            ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = f"elisa_report_{ts}.docx"

        # Generate markdown first
        md_path = output_path.replace(".docx", ".md")
        self.generate_markdown(md_path, llm_func=llm_func)

        # Try pandoc conversion
        try:
            cmd = [
                "pandoc", md_path,
                "-o", output_path,
                "--from=markdown",
                "--to=docx",
                "--reference-doc=/dev/null",  # use default template
                "--toc",
                "--toc-depth=2",
            ]
            # Remove --reference-doc if it causes issues
            cmd = ["pandoc", md_path, "-o", output_path,
                   "--from=markdown", "--to=docx"]

            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[REPORT] Saved DOCX: {output_path}")
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"[WARN] pandoc not available, using Markdown only: {md_path}")
            return md_path

    # ────────────────────────────────────────────────────────
    # Internal builders
    # ────────────────────────────────────────────────────────

    def _build_abstract(self, llm_func=None) -> str:
        """Generate abstract from collected results."""
        # Collect key findings
        summary_parts = []
        summary_parts.append(
            f"We performed computational analysis of the {self.dataset_name} "
            f"single-cell RNA-seq dataset using ELISA (Embedding-based LLM-Integrated "
            f"Single-cell Analysis). "
        )

        n_types = {e["type"] for e in self.entries}
        if "compare" in n_types:
            summary_parts.append(
                "Comparative analysis was performed between conditions. ")
        if "interactions" in n_types:
            summary_parts.append(
                "Cell-cell communication was inferred using ligand-receptor analysis. ")
        if "proportions" in n_types:
            summary_parts.append(
                "Cell type composition was quantified. ")
        if "pathway" in n_types:
            summary_parts.append(
                "Pathway activity scoring was performed across cell types. ")

        summary_parts.append(
            f"A total of {len(self.entries)} analyses were conducted, "
            f"revealing key biological insights described below."
        )

        base = "".join(summary_parts)

        if llm_func:
            # Ask LLM to polish the abstract
            all_answers = "\n\n".join(
                f"[{e['type']}] {e['query']}: {e['answer'][:500]}"
                for e in self.entries
            )
            prompt = (
                f"Based on these analysis results, write a concise scientific "
                f"abstract (150-250 words) for a report on {self.dataset_name}:\n\n"
                f"{all_answers}\n\n"
                f"Write ONLY the abstract text, no headers."
            )
            try:
                return llm_func(prompt)
            except Exception:
                pass

        return base

    def _build_discussion(self, llm_func=None) -> str:
        """Generate discussion section."""
        if llm_func and self.entries:
            findings = "\n".join(
                f"- [{e['type']}] {e['query']}: {e['answer'][:300]}"
                for e in self.entries
            )
            prompt = (
                f"Based on these findings from {self.dataset_name}, write a "
                f"scientific discussion (300-500 words). Compare findings to known "
                f"biology, note unexpected results, discuss limitations. "
                f"Use cautious scientific language.\n\n"
                f"FINDINGS:\n{findings}\n\n"
                f"Write ONLY the discussion text."
            )
            try:
                return llm_func(prompt)
            except Exception:
                pass

        return (
            "The analyses presented here provide a computational exploration "
            f"of the {self.dataset_name} dataset using hybrid semantic and "
            "expression-based retrieval. Findings should be validated experimentally. "
            "The comparative analyses rely on metadata-weighted gene statistics, "
            "which serve as a proxy for condition-specific differential expression "
            "but may not capture subtle transcriptional changes. "
            "Cell-cell interaction predictions are based on co-expression of "
            "ligand-receptor pairs and do not confirm physical interactions. "
            "Pathway scores reflect average expression of pathway gene sets "
            "and may not capture post-translational regulation."
        )

    def _build_perspectives(self, llm_func=None) -> str:
        """Generate future perspectives."""
        if llm_func and self.entries:
            prompt = (
                f"Based on the ELISA analysis of {self.dataset_name}, "
                f"write 3-5 future perspectives (200-300 words). "
                f"Suggest experiments, additional analyses, or follow-up questions.\n\n"
                f"Analyses performed: {[e['type'] for e in self.entries]}\n"
                f"Queries: {[e['query'] for e in self.entries]}\n\n"
                f"Write ONLY the perspectives text."
            )
            try:
                return llm_func(prompt)
            except Exception:
                pass

        return (
            "Future work should include: "
            "(1) Experimental validation of predicted cell-cell interactions "
            "using co-culture assays or spatial transcriptomics. "
            "(2) Condition-specific differential expression analysis using "
            "pseudobulk approaches for rigorous statistical testing. "
            "(3) Integration with additional datasets to assess reproducibility. "
            "(4) Trajectory analysis to understand cell state transitions. "
            "(5) Protein-level validation of key findings using flow cytometry "
            "or immunohistochemistry."
        )

    def _build_methods(self) -> str:
        """Generate methods section."""
        methods = []
        methods.append(
            f"### Dataset\n"
            f"The {self.dataset_name} single-cell RNA-seq dataset was analyzed "
            f"using ELISA (Embedding-based LLM-Integrated Single-cell Analysis).\n"
        )

        methods.append(
            "### Embedding Generation\n"
            "Cluster-level embeddings were generated using a dual approach: "
            "(1) Semantic embeddings via BioBERT "
            "(pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb) encoding "
            "natural language cluster summaries including marker genes, GO terms, "
            "and Reactome pathways; "
            "(2) Expression embeddings via scGPT whole-human pre-trained model "
            "or PCA-based centroids as fallback.\n"
        )

        methods.append(
            "### Differential Expression\n"
            "Wilcoxon rank-sum tests were performed on all genes (no HVG filtering) "
            "to identify cluster markers. Per-gene statistics (logFC, fraction "
            "expressing in/out of cluster) were stored for the top 2000 genes "
            "per cluster.\n"
        )

        analysis_types = {e["type"] for e in self.entries}

        if "compare" in analysis_types:
            methods.append(
                "### Comparative Analysis\n"
                "Condition-specific gene expression differences were assessed "
                "by weighting cluster-level gene statistics by the proportion "
                "of cells from each condition within each cluster.\n"
            )

        if "interactions" in analysis_types:
            methods.append(
                "### Cell-Cell Interaction Inference\n"
                "Ligand-receptor interactions were predicted using a curated "
                "database derived from CellPhoneDB, CellChat, and KEGG. "
                "Interactions were scored by the product of ligand expression "
                "fraction in the source cluster and receptor expression fraction "
                "in the target cluster.\n"
            )

        if "proportions" in analysis_types:
            methods.append(
                "### Cell Type Proportion Analysis\n"
                "Cell type proportions were computed from cluster cell counts. "
                "Condition-specific proportions were estimated using metadata "
                "distribution weights.\n"
            )

        if "pathway" in analysis_types:
            methods.append(
                "### Pathway Activity Scoring\n"
                "Pathway activity was scored per cluster as the mean expression "
                "fraction (pct_in) of pathway member genes. Gene sets were "
                "derived from KEGG and Reactome databases.\n"
            )

        methods.append(
            "### LLM Interpretation\n"
            "Retrieval results were interpreted using Llama-3.1-8B-Instant "
            "via Groq API, with strict grounding instructions to prevent "
            "hallucination.\n"
        )

        return "\n".join(methods)

    def _format_result_section(self, idx: int, entry: Dict) -> str:
        """Format a single analysis result as a report section."""
        type_labels = {
            "semantic": "Semantic Retrieval",
            "hybrid": "Hybrid Retrieval",
            "discovery": "Discovery Analysis",
            "compare": "Comparative Analysis",
            "interactions": "Cell-Cell Interaction Analysis",
            "proportions": "Cell Type Proportion Analysis",
            "pathway": "Pathway Activity Analysis",
        }

        label = type_labels.get(entry["type"], entry["type"].title())
        section = f"### {idx}. {label}: {entry['query']}\n\n"
        section += entry["answer"] + "\n\n"

        # Add figure references
        if entry.get("plots"):
            section += "**Associated figures:** "
            section += ", ".join(os.path.basename(p) for p in entry["plots"])
            section += "\n\n"

        return section

    def _format_supplementary(self, idx: int, entry: Dict) -> str:
        """Format supplementary tables for an entry."""
        payload = entry.get("payload", {})
        results = payload.get("results", [])
        if not results:
            return ""

        section = f"### Supplementary Table {idx}: {entry['query']}\n\n"

        # Gene evidence tables
        if entry["type"] in ("semantic", "hybrid", "discovery"):
            section += "| Cluster | Gene | logFC | pct_in | pct_out |\n"
            section += "|---------|------|-------|--------|--------|\n"
            for r in results:
                cid = r.get("cluster_id", "?")
                for g in r.get("gene_evidence", [])[:5]:
                    lfc = g.get("logfc", "")
                    if isinstance(lfc, float):
                        lfc = f"{lfc:.3f}"
                    pi = g.get("pct_in", "")
                    if isinstance(pi, float):
                        pi = f"{pi:.3f}"
                    po = g.get("pct_out", "")
                    if isinstance(po, float):
                        po = f"{po:.3f}"
                    section += f"| {cid} | {g['gene']} | {lfc} | {pi} | {po} |\n"
            section += "\n"

        elif entry["type"] == "interactions":
            interactions = payload.get("interactions", [])[:20]
            if interactions:
                section += "| Ligand | Receptor | Source | Target | Pathway | Score |\n"
                section += "|--------|----------|--------|--------|---------|-------|\n"
                for ix in interactions:
                    section += (
                        f"| {ix['ligand']} | {ix['receptor']} | "
                        f"{ix['source'][:20]} | {ix['target'][:20]} | "
                        f"{ix['pathway']} | {ix['score']:.4f} |\n"
                    )
                section += "\n"

        elif entry["type"] == "proportions":
            props = payload.get("proportions", [])
            if props:
                section += "| Cluster | N cells | Percentage |\n"
                section += "|---------|---------|------------|\n"
                for p in props:
                    section += f"| {p['cluster']} | {p['n_cells']} | {p['percentage']}% |\n"
                section += "\n"

        return section

    def get_session_summary(self) -> Dict:
        """Return a JSON-serializable summary of the session."""
        return {
            "dataset": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "n_analyses": len(self.entries),
            "types": [e["type"] for e in self.entries],
            "queries": [e["query"] for e in self.entries],
        }
