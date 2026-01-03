"""
Markdown explainability report generator.

Creates human-readable reports with variance analysis and actionable insights.
"""

from datetime import datetime
from typing import Any
from collections import Counter

import numpy as np

from ruvrics.core.models import StabilityResult, RunResult


def cluster_outputs(runs: list[RunResult], embeddings: np.ndarray, threshold: float = 0.85) -> list[dict[str, Any]]:
    """
    Cluster outputs by semantic similarity.

    Args:
        runs: List of run results
        embeddings: Semantic embeddings for each run
        threshold: Similarity threshold for clustering (default: 0.85)

    Returns:
        List of clusters with representative outputs
    """
    if len(runs) == 0:
        return []

    clusters = []
    assigned = [False] * len(runs)

    # Simple clustering: for each unassigned run, create cluster with similar runs
    for i, run in enumerate(runs):
        if assigned[i]:
            continue

        # Start new cluster
        cluster_runs = [i]
        cluster_embedding = embeddings[i]
        assigned[i] = True

        # Find similar runs
        for j in range(i + 1, len(runs)):
            if assigned[j]:
                continue

            # Calculate cosine similarity
            similarity = np.dot(embeddings[j], cluster_embedding) / (
                np.linalg.norm(embeddings[j]) * np.linalg.norm(cluster_embedding)
            )

            if similarity >= threshold:
                cluster_runs.append(j)
                assigned[j] = True

        # Store cluster info
        clusters.append({
            "run_indices": cluster_runs,
            "size": len(cluster_runs),
            "percentage": (len(cluster_runs) / len(runs)) * 100,
            "representative_idx": cluster_runs[0],  # First run as representative
            "representative_output": runs[cluster_runs[0]].output_text,
            "avg_length": np.mean([runs[idx].output_length_tokens for idx in cluster_runs]),
            "structure": runs[cluster_runs[0]].output_structure,
            "tool_usage": len(runs[cluster_runs[0]].tool_calls) > 0,
        })

    # Sort by size (largest first)
    clusters.sort(key=lambda c: c["size"], reverse=True)

    return clusters


def analyze_tool_correlation(clusters: list[dict[str, Any]], runs: list[RunResult]) -> dict[str, Any]:
    """
    Analyze correlation between tool usage and semantic clusters.

    Args:
        clusters: Output clusters
        runs: All run results

    Returns:
        Correlation analysis dict
    """
    tool_used_count = sum(1 for r in runs if len(r.tool_calls) > 0)
    tool_not_used_count = len(runs) - tool_used_count

    # Analyze tool usage per cluster
    cluster_tool_analysis = []
    for idx, cluster in enumerate(clusters):
        cluster_runs = [runs[i] for i in cluster["run_indices"]]
        tools_in_cluster = sum(1 for r in cluster_runs if len(r.tool_calls) > 0)

        cluster_tool_analysis.append({
            "cluster_idx": idx,
            "cluster_label": chr(65 + idx),  # A, B, C, ...
            "total_runs": cluster["size"],
            "tool_used": tools_in_cluster,
            "tool_percentage": (tools_in_cluster / cluster["size"]) * 100 if cluster["size"] > 0 else 0,
        })

    return {
        "total_tool_used": tool_used_count,
        "total_tool_not_used": tool_not_used_count,
        "tool_percentage": (tool_used_count / len(runs)) * 100,
        "cluster_analysis": cluster_tool_analysis,
    }


def extract_divergent_phrases(runs: list[RunResult], top_n: int = 5) -> list[tuple[str, int]]:
    """
    Extract key phrases that appear inconsistently across runs.

    Args:
        runs: List of run results
        top_n: Number of top phrases to return

    Returns:
        List of (phrase, count) tuples
    """
    # Simple approach: find unique first sentences
    first_sentences = []
    for run in runs:
        text = run.output_text.strip()
        # Get first sentence (split on . ! ? or newline)
        first_sent = text.split('.')[0] if '.' in text else text.split('\n')[0]
        first_sentences.append(first_sent.strip()[:100])  # Limit length

    # Count occurrences
    phrase_counts = Counter(first_sentences)

    # Return top phrases
    return phrase_counts.most_common(top_n)


def generate_guidance(
    clusters: list[dict[str, Any]],
    tool_correlation: dict[str, Any],
    result: StabilityResult,
) -> str:
    """
    Generate actionable "Where to Look" guidance.

    Args:
        clusters: Output clusters
        tool_correlation: Tool usage correlation data
        result: Stability result

    Returns:
        Markdown formatted guidance text
    """
    guidance = []

    # Guidance based on number of clusters
    if len(clusters) > 1:
        guidance.append(
            f"1. **Compare Cluster A vs Cluster {chr(65 + len(clusters) - 1)}** - "
            f"See the difference between dominant behavior ({clusters[0]['percentage']:.0f}%) "
            f"and outliers ({clusters[-1]['percentage']:.0f}%)"
        )

    # Guidance based on tool usage
    if tool_correlation["total_tool_not_used"] > 0 and result.tool_consistency_score < 95:
        tool_pct = tool_correlation["tool_percentage"]
        guidance.append(
            f"2. **Notice:** Tools used in only {tool_pct:.0f}% of runs - "
            "this inconsistency drives semantic variance"
        )

    # Guidance based on root causes
    if result.root_causes:
        primary_cause = result.root_causes[0]
        if primary_cause.type == "NONDETERMINISTIC_TOOL_ROUTING":
            guidance.append(
                "3. **Fix priority:** Add explicit tool usage instructions to system prompt"
            )
        elif primary_cause.type == "UNDERSPECIFIED_PROMPT":
            guidance.append(
                "3. **Fix priority:** Add 3-5 example outputs to reduce ambiguity"
            )
        elif primary_cause.type == "FORMAT_INCONSISTENCY":
            guidance.append(
                "3. **Fix priority:** Enforce strict output format schema"
            )

    # Expected improvement
    if result.stability_score < 90:
        target_score = 90 if result.stability_score < 70 else 95
        guidance.append(
            f"4. **Expected improvement:** Should reach {target_score}%+ stability "
            "after applying Priority 1 fix"
        )

    return "\n".join(guidance) if guidance else "System is stable. No immediate action needed."


def generate_markdown_report(result: StabilityResult, runs: list[RunResult], embeddings: np.ndarray) -> str:
    """
    Generate complete Markdown explainability report.

    Args:
        result: Stability analysis result
        runs: All run results
        embeddings: Semantic embeddings for clustering

    Returns:
        Markdown formatted report
    """
    # Perform clustering
    clusters = cluster_outputs(runs, embeddings)

    # Analyze tool correlation
    tool_correlation = analyze_tool_correlation(clusters, runs)

    # Extract divergent phrases
    divergent_phrases = extract_divergent_phrases(runs)

    # Generate guidance
    guidance = generate_guidance(clusters, tool_correlation, result)

    # Build report
    report = []

    # Header
    report.append("# Ruvrics Stability Report\n")
    report.append(f"**Model:** {result.model}")
    report.append(f"**Timestamp:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    report.append(f"**Runs:** {result.successful_runs}/{result.total_runs} successful")
    report.append(f"**Duration:** {result.duration_seconds:.1f}s\n")

    # Overall score
    risk_emoji = {"SAFE": "✅", "RISKY": "⚠️", "DO_NOT_SHIP": "❌"}
    report.append(
        f"## Overall Score: {result.stability_score:.1f}% - "
        f"{risk_emoji.get(result.risk_classification, '')} {result.risk_classification}\n"
    )
    report.append("---\n")

    # Variance Summary Table
    report.append("## Variance Summary\n")
    report.append("| Dimension        | Score  | Variance | Evidence                              |")
    report.append("|------------------|--------|----------|---------------------------------------|")

    # Semantic
    cluster_evidence = f"{len(clusters)} distinct answer cluster{'s' if len(clusters) != 1 else ''} detected"
    report.append(
        f"| Semantic Meaning | {result.semantic_consistency_score:.1f}%  | "
        f"{result.semantic_drift:<8} | {cluster_evidence:<37} |"
    )

    # Tool
    tool_evidence = f"Tool used in {tool_correlation['total_tool_used']}/{result.total_runs} runs"
    report.append(
        f"| Tool Usage       | {result.tool_consistency_score:.1f}%  | "
        f"{result.tool_variance:<8} | {tool_evidence:<37} |"
    )

    # Structural
    structure_counts = Counter(run.output_structure for run in runs)
    structure_evidence = f"All outputs are {structure_counts.most_common(1)[0][0]} format" if len(structure_counts) == 1 else f"{len(structure_counts)} different formats"
    report.append(
        f"| Output Structure | {result.structural_consistency_score:.1f}%  | "
        f"{result.structural_variance:<8} | {structure_evidence:<37} |"
    )

    # Length
    lengths = [run.output_length_tokens for run in runs]
    length_evidence = f"Range: {min(lengths)}–{max(lengths)} tokens"
    report.append(
        f"| Length           | {result.length_consistency_score:.1f}%  | "
        f"{result.length_variance:<8} | {length_evidence:<37} |"
    )

    report.append("")

    # Quick diagnosis
    if result.root_causes:
        primary_cause = result.root_causes[0]
        report.append(f"**Quick Diagnosis:** {primary_cause.description}\n")

    report.append("---\n")

    # Semantic Clusters
    report.append(f"## Semantic Clusters ({result.total_runs} runs → {len(clusters)} cluster{'s' if len(clusters) != 1 else ''})\n")
    report.append("Your system exhibits " + (f"{len(clusters)} distinct behaviors:" if len(clusters) > 1 else "consistent behavior:") + "\n")

    for idx, cluster in enumerate(clusters):
        cluster_label = chr(65 + idx)  # A, B, C, ...
        is_dominant = idx == 0
        is_outlier = idx == len(clusters) - 1 and len(clusters) > 2

        label_suffix = " ← Dominant behavior" if is_dominant else (" ← Outlier" if is_outlier else "")

        report.append(f"### Cluster {cluster_label} ({cluster['size']} runs - {cluster['percentage']:.0f}%){label_suffix}")
        report.append(f"**Runs:** {', '.join(str(i+1) for i in cluster['run_indices'])}\n")

        report.append("**Representative Output:**")
        report.append("```")
        # Limit output length for readability
        output_preview = cluster['representative_output'][:500]
        if len(cluster['representative_output']) > 500:
            output_preview += "..."
        report.append(output_preview)
        report.append("```\n")

        report.append("**Characteristics:**")
        report.append(f"- Uses tools: {'YES' if cluster['tool_usage'] else 'NO'}")
        report.append(f"- Average length: {cluster['avg_length']:.0f} tokens")
        report.append(f"- Structure: {cluster['structure']}\n")

        # Show difference from dominant cluster
        if idx > 0:
            diff_text = []
            if cluster['tool_usage'] != clusters[0]['tool_usage']:
                diff_text.append("Skips tool" if not cluster['tool_usage'] else "Uses tool")

            length_diff = ((cluster['avg_length'] - clusters[0]['avg_length']) / clusters[0]['avg_length']) * 100
            if abs(length_diff) > 20:
                diff_text.append(f"{abs(length_diff):.0f}% {'shorter' if length_diff < 0 else 'longer'}")

            if diff_text:
                report.append(f"**Key Difference from Cluster A:** {', '.join(diff_text)}\n")

        report.append("---\n")

    # Tool-Usage Correlation
    if tool_correlation['total_tool_used'] > 0:
        report.append("## Tool-Usage Correlation\n")
        report.append("**Observed Pattern:**")

        for cluster_analysis in tool_correlation['cluster_analysis']:
            label = cluster_analysis['cluster_label']
            tool_pct = cluster_analysis['tool_percentage']
            tool_count = cluster_analysis['tool_used']
            total_count = cluster_analysis['total_runs']

            if tool_pct == 100:
                report.append(f"- Tool USED → All {total_count} runs in Cluster {label}")
            elif tool_pct == 0:
                report.append(f"- Tool NOT USED → All {total_count} runs in Cluster {label}")
            else:
                report.append(f"- Mixed usage in Cluster {label}: {tool_count}/{total_count} runs used tools")

        report.append("")

        # Conclusion
        if len(clusters) > 1 and result.tool_variance in ["MEDIUM", "HIGH"]:
            report.append("**Conclusion:**")
            report.append(
                "Semantic instability is correlated with tool routing decisions. "
                "When the model skips tools, output behavior changes significantly.\n"
            )

        report.append("---\n")

    # Key Divergent Phrases
    if len(divergent_phrases) > 1:
        report.append("## Key Divergent Phrases\n")
        report.append("The following phrases appear inconsistently:\n")

        for phrase, count in divergent_phrases[:5]:
            percentage = (count / len(runs)) * 100
            report.append(f"- \"{phrase}\" ({count} runs - {percentage:.0f}%)")

        report.append("\n---\n")

    # Where to Look
    report.append("## Where to Look\n")
    report.append("To understand this instability quickly:\n")
    report.append(guidance)
    report.append("\n---\n")

    # Root Causes
    if result.root_causes:
        report.append("## Root Causes\n")

        for idx, cause in enumerate(result.root_causes, 1):
            report.append(f"### {idx}. {cause.type} ({cause.severity} severity)")
            report.append(f"**Description:** {cause.description}")
            if cause.details:
                report.append(f"**Impact:** {cause.details}")
            report.append("")

        report.append("---\n")

    # Recommended Fixes
    if result.recommendations:
        report.append("## Recommended Fixes\n")

        for idx, rec in enumerate(result.recommendations, 1):
            report.append(f"### Priority {idx}: {rec.title}")
            report.append(f"**Category:** {rec.category}")
            report.append(f"**Action:** {rec.description}\n")

            if rec.example:
                report.append("```")
                report.append(rec.example)
                report.append("```\n")

        report.append("---\n")

    # Appendix: All Outputs
    report.append("## Appendix: All Outputs\n")

    for run in runs:
        cluster_label = None
        for idx, cluster in enumerate(clusters):
            if run.run_id - 1 in cluster['run_indices']:
                cluster_label = chr(65 + idx)
                break

        report.append(f"### Run {run.run_id} (Cluster {cluster_label})")
        report.append("```")
        # Limit output for very long responses
        output_text = run.output_text[:1000]
        if len(run.output_text) > 1000:
            output_text += "\n... (truncated)"
        report.append(output_text)
        report.append("```")

        tools_used = ", ".join(tc.name for tc in run.tool_calls) if run.tool_calls else "none"
        report.append(f"**Tools:** {tools_used} | **Length:** {run.output_length_tokens} tokens | **Structure:** {run.output_structure}\n")

    report.append("---\n")
    report.append("## Raw Data\n")
    report.append("See `ruvrics_report.json` for machine-readable format with all outputs.\n")

    return "\n".join(report)
