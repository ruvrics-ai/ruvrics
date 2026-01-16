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


def _variance_to_status(variance: str) -> str:
    """
    Convert technical variance classification to business-friendly status.

    Args:
        variance: Technical variance (LOW/MEDIUM/HIGH/N/A)

    Returns:
        Business-friendly status with emoji
    """
    status_map = {
        "LOW": "✅ EXCELLENT",
        "MEDIUM": "⚠️ GOOD",
        "HIGH": "❌ NEEDS ATTENTION",
        "N/A": "— N/A"
    }
    return status_map.get(variance, variance)


def _get_root_cause_business_title(cause_type: str) -> str:
    """
    Map technical root cause types to business-friendly titles.

    Args:
        cause_type: Technical root cause identifier

    Returns:
        Business-friendly title
    """
    business_titles = {
        "NONDETERMINISTIC_TOOL_ROUTING": "Inconsistent Tool Usage",
        "ARGUMENT_DRIFT": "Inconsistent Tool Parameters",
        "CHAIN_VARIANCE": "Unpredictable Tool Sequence",
        "UNDERSPECIFIED_PROMPT": "Ambiguous Instructions",
        "FORMAT_INCONSISTENCY": "Unpredictable Output Format",
        "HIGH_LENGTH_VARIANCE": "Inconsistent Response Detail",
        "CLAIM_INSTABILITY": "Inconsistent Safety Guardrails",
        "COMBINED_INSTABILITY": "Multiple Stability Issues",
    }
    return business_titles.get(cause_type, cause_type.replace("_", " ").title())


def _get_root_cause_analogy(cause_type: str) -> str:
    """
    Provide business analogy for root causes.

    Args:
        cause_type: Technical root cause identifier

    Returns:
        Business analogy or empty string
    """
    analogies = {
        "NONDETERMINISTIC_TOOL_ROUTING": "**Analogy:** Like a customer service agent who sometimes checks the database and sometimes doesn't - customers get inconsistent service quality.",
        "ARGUMENT_DRIFT": "**Analogy:** Like a search engine where 'recent news' sometimes means last 24 hours, sometimes last week, sometimes last month - users get unpredictable results.",
        "CHAIN_VARIANCE": "**Analogy:** Like a travel agent who sometimes books flight first then hotel, sometimes hotel first - the final package may differ and coordination breaks.",
        "UNDERSPECIFIED_PROMPT": "**Analogy:** Like giving an employee vague instructions - they'll interpret it differently each time.",
        "FORMAT_INCONSISTENCY": "**Analogy:** Like receiving reports in different formats (Excel one day, PDF the next) - downstream systems can't reliably process them.",
        "HIGH_LENGTH_VARIANCE": "**Analogy:** Like asking 10 people the same question and getting answers ranging from one sentence to three paragraphs - quality feels inconsistent.",
    }
    return analogies.get(cause_type, "")


def _get_consistency_meaning(score: float, metric_type: str, context: dict[str, Any] = None) -> str:
    """
    Generate business-friendly explanation for consistency scores.

    Args:
        score: Consistency score (0-100)
        metric_type: Type of metric (semantic, tool, structural, length)
        context: Additional context for specific explanations

    Returns:
        Human-readable explanation
    """
    context = context or {}

    if metric_type == "semantic":
        if score >= 95:
            return "AI gives nearly identical answers every time"
        elif score >= 85:
            return "AI responses are similar but have minor variations"
        elif score >= 70:
            return "AI responses vary noticeably in meaning"
        else:
            return "AI gives significantly different answers to same question"

    elif metric_type == "tool":
        if score >= 95:
            return "AI consistently performs the same actions"
        elif score >= 80:
            return "AI mostly performs same actions with some variation"
        elif score >= 60:
            return "AI is inconsistent about which tools to use"
        else:
            tool_pct = context.get('tool_percentage', 0) if context else 0
            return f"AI highly inconsistent - tools used only {tool_pct:.0f}% of time"

    elif metric_type == "structural":
        count = context.get('count', 1) if context else 1
        if score >= 95:
            return "Output format is consistent and parseable"
        elif score >= 80:
            return f"Output has {count} different formats - may cause parsing issues"
        else:
            return f"Output format highly unpredictable ({count} variations)"

    elif metric_type == "length":
        if context:
            length_range = context.get('range', 0)
            min_len = context.get('min', 0)
            max_len = context.get('max', 0)
        else:
            length_range = 0
            min_len = max_len = 0

        if score >= 95:
            return f"Response length very consistent ({min_len}-{max_len} tokens)"
        elif score >= 85:
            return f"Response length fairly consistent (±{length_range} token range)"
        else:
            return f"Response length varies widely ({min_len}-{max_len} tokens)"

    elif metric_type == "argument":
        variations = context.get('variations', 0) if context else 0
        tools = context.get('tools_with_variation', 0) if context else 0
        if score >= 95:
            return "Tool arguments are consistent across runs"
        elif score >= 80:
            return f"Some argument variation detected ({tools} tool(s))"
        else:
            return f"High argument drift - {tools} tool(s) with {variations}+ variations"

    elif metric_type == "chain":
        sequences = context.get('unique_sequences', 1) if context else 1
        avg_len = context.get('avg_chain_length', 0) if context else 0
        if score >= 95:
            return f"Tool sequence is consistent (avg {avg_len:.1f} tools/run)"
        elif score >= 80:
            return f"{sequences} different sequences detected"
        else:
            return f"High sequence variance - {sequences} different tool orderings"

    return "Consistency analysis"


def generate_executive_summary(
    result: StabilityResult,
    tool_correlation: dict[str, Any],
    runs: list[RunResult],
) -> str:
    """
    Generate executive summary for decision makers.

    Args:
        result: Stability analysis result
        tool_correlation: Tool usage correlation data
        runs: All run results

    Returns:
        Markdown formatted executive summary
    """
    summary = ["## Executive Summary\n"]

    # Verdict with clear status
    verdict_text = {
        "SAFE": "✅ **SAFE TO SHIP** - Your AI system is stable and production-ready",
        "RISKY": "⚠️ **REVIEW REQUIRED** - Stability issues detected that should be addressed before launch",
        "DO_NOT_SHIP": "❌ **DO NOT SHIP** - Critical instability issues must be fixed before deployment"
    }
    summary.append(verdict_text.get(result.risk_classification, "Status: Unknown"))
    summary.append(f"**Overall Stability Score:** {result.stability_score:.1f}%\n")

    # Main issue (if any)
    if result.root_causes and result.risk_classification != "SAFE":
        primary = result.root_causes[0]
        summary.append("**Main Issue:**")
        summary.append(f"{primary.description}\n")

        # User impact based on root cause type
        summary.append("**User Impact:**")
        if primary.type == "NONDETERMINISTIC_TOOL_ROUTING":
            tool_pct = tool_correlation.get('tool_percentage', 0)
            summary.append(f"- AI uses tools in only {tool_pct:.0f}% of identical requests")
            summary.append(f"- {100-tool_pct:.0f}% of users don't get the full functionality")
            summary.append("- Inconsistent experience damages user trust")
        elif primary.type == "UNDERSPECIFIED_PROMPT":
            summary.append("- AI interprets the same question in multiple ways")
            summary.append("- Users get unpredictable responses")
            summary.append("- Quality varies significantly between requests")
        elif primary.type == "FORMAT_INCONSISTENCY":
            summary.append("- Output format is unpredictable")
            summary.append("- Downstream systems may fail to parse responses")
            summary.append("- Integration breakage risk is high")
        elif primary.type == "HIGH_LENGTH_VARIANCE":
            lengths = [r.output_length_tokens for r in runs]
            min_len, max_len = min(lengths), max(lengths)
            summary.append(f"- Response length varies from {min_len} to {max_len} tokens")
            summary.append("- Some users get detailed answers, others get brief ones")
            summary.append("- Inconsistent perceived quality")
        else:
            summary.append("- System behavior is unpredictable")
            summary.append("- User experience quality varies")

        summary.append("")

        # Quick recommendation
        if result.recommendations:
            rec = result.recommendations[0]
            summary.append(f"**Quick Fix:** {rec.description}")

            # Expected improvement estimate
            if result.stability_score < 70:
                target = "90%+"
            elif result.stability_score < 85:
                target = "95%+"
            else:
                target = "98%+"
            summary.append(f"**Expected Improvement:** Applying this fix should bring stability to {target}\n")
    elif result.risk_classification == "SAFE":
        summary.append("**Status:** No critical issues detected. Your system performs consistently.\n")

    summary.append("---\n")
    return "\n".join(summary)


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


def generate_markdown_report(result: StabilityResult, runs: list[RunResult], embeddings: np.ndarray, input_query: str = None) -> str:
    """
    Generate complete Markdown explainability report.

    Args:
        result: Stability analysis result
        runs: All run results
        embeddings: Semantic embeddings for clustering
        input_query: Optional user input query to display in report

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

    # Test information
    if input_query:
        report.append(f"**Test Query:** {input_query}\n")

    report.append(f"**Model:** {result.model}")
    report.append(f"**Timestamp:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    report.append(f"**Runs:** {result.successful_runs}/{result.total_runs} successful")
    report.append(f"**Duration:** {result.duration_seconds:.1f}s\n")

    # Executive Summary
    report.append(generate_executive_summary(result, tool_correlation, runs))

    # Overall score (now secondary to executive summary)
    risk_emoji = {"SAFE": "✅", "RISKY": "⚠️", "DO_NOT_SHIP": "❌"}
    report.append(
        f"## Detailed Analysis\n"
    )
    report.append(
        f"**Overall Stability Score:** {result.stability_score:.1f}% - "
        f"{risk_emoji.get(result.risk_classification, '')} {result.risk_classification}\n"
    )
    report.append("---\n")

    # Variance Summary Table with Business-Only Terminology
    report.append("## Consistency Breakdown\n")
    report.append("| Dimension | Score | Status | What This Means |")
    report.append("|-----------|-------|--------|-----------------|")

    # Semantic - Response Consistency
    semantic_status = _variance_to_status(result.semantic_drift)
    semantic_meaning = _get_consistency_meaning(result.semantic_consistency_score, "semantic")
    report.append(
        f"| **Response Consistency** | "
        f"{result.semantic_consistency_score:.1f}% | "
        f"{semantic_status} | "
        f"{semantic_meaning} |"
    )

    # Tool - Action Reliability
    tool_status = _variance_to_status(result.tool_variance)
    tool_meaning = _get_consistency_meaning(result.tool_consistency_score, "tool", tool_correlation)
    report.append(
        f"| **Action Reliability** | "
        f"{result.tool_consistency_score:.1f}% | "
        f"{tool_status} | "
        f"{tool_meaning} |"
    )

    # Structural - Format Stability
    structure_counts = Counter(run.output_structure for run in runs)
    structural_status = _variance_to_status(result.structural_variance)
    structural_meaning = _get_consistency_meaning(result.structural_consistency_score, "structural", {"count": len(structure_counts)})
    report.append(
        f"| **Format Stability** | "
        f"{result.structural_consistency_score:.1f}% | "
        f"{structural_status} | "
        f"{structural_meaning} |"
    )

    # Length - Response Length
    lengths = [run.output_length_tokens for run in runs]
    length_range = max(lengths) - min(lengths)
    length_status = _variance_to_status(result.length_variance)
    length_meaning = _get_consistency_meaning(result.length_consistency_score, "length", {"range": length_range, "min": min(lengths), "max": max(lengths)})
    report.append(
        f"| **Response Length** | "
        f"{result.length_consistency_score:.1f}% | "
        f"{length_status} | "
        f"{length_meaning} |"
    )

    # Argument Consistency (v0.2.2) - only show if tools were used
    if result.argument_variance != "N/A":
        argument_status = _variance_to_status(result.argument_variance)
        arg_details = result.argument_details or {}
        tools_with_variation = arg_details.get("tools_with_variation", 0)
        tool_analysis = arg_details.get("tool_argument_analysis", {})
        total_variations = sum(
            t.get("argument_variations", 1)
            for t in tool_analysis.values()
        )
        argument_meaning = _get_consistency_meaning(
            result.argument_consistency_score, "argument",
            {"tools_with_variation": tools_with_variation, "variations": total_variations}
        )
        report.append(
            f"| **Tool Arguments** | "
            f"{result.argument_consistency_score:.1f}% | "
            f"{argument_status} | "
            f"{argument_meaning} |"
        )

    # Chain Consistency (v0.2.2) - only show if multi-tool runs detected
    if result.chain_variance != "N/A":
        chain_status = _variance_to_status(result.chain_variance)
        chain_details = result.chain_details or {}
        chain_meaning = _get_consistency_meaning(
            result.chain_consistency_score, "chain",
            {
                "unique_sequences": chain_details.get("unique_sequences", 0),
                "avg_chain_length": chain_details.get("avg_chain_length", 0),
            }
        )
        report.append(
            f"| **Tool Sequence** | "
            f"{result.chain_consistency_score:.1f}% | "
            f"{chain_status} | "
            f"{chain_meaning} |"
        )

    report.append("")

    # Quick diagnosis
    if result.root_causes:
        primary_cause = result.root_causes[0]
        report.append(f"**Quick Diagnosis:** {primary_cause.description}\n")

    report.append("---\n")

    # Semantic Clusters with Business-Friendly Labels
    report.append(f"## Behavior Patterns ({result.total_runs} runs → {len(clusters)} pattern{'s' if len(clusters) != 1 else ''})\n")
    report.append("Your AI exhibits " + (f"**{len(clusters)} distinct behaviors** when given identical input:" if len(clusters) > 1 else "**consistent behavior** across all tests:") + "\n")

    for idx, cluster in enumerate(clusters):
        cluster_label = chr(65 + idx)  # A, B, C, ... (keep for technical reference)

        # Business-friendly labels
        if idx == 0:
            behavior_label = "⭐ Primary Behavior"
            percentage_context = f"({cluster['size']} out of {result.total_runs} times - {cluster['percentage']:.0f}%)"
        elif idx == len(clusters) - 1 and len(clusters) > 2:
            behavior_label = f"🔴 Edge Case Behavior #{idx}"
            percentage_context = f"({cluster['size']} times - {cluster['percentage']:.0f}%)"
        else:
            behavior_label = f"⚠️ Alternative Behavior #{idx}"
            percentage_context = f"({cluster['size']} times - {cluster['percentage']:.0f}%)"

        label_suffix = ""

        report.append(f"### {behavior_label} {percentage_context}")
        report.append(f"**Example Runs:** {', '.join(str(i+1) for i in cluster['run_indices'][:10])}" + ("..." if len(cluster['run_indices']) > 10 else "") + "\n")

        # Get representative run
        rep_idx = cluster['representative_idx']
        rep_run = runs[rep_idx]

        # Handle different output scenarios
        if not cluster['representative_output'] and cluster['tool_usage']:
            # Tool-only response
            report.append("**Representative Output:** *(Model called tools without text response)*")
            report.append("**Actions:**")
            for tc in rep_run.tool_calls:
                report.append(f"  {tc.call_sequence}. `{tc.to_display_string()}`")
            report.append("")
        elif not cluster['representative_output']:
            # Empty output with no tools
            report.append("**Representative Output:** ⚠️ *(Empty output)*")
            report.append("```")
            report.append("")
            report.append("```\n")
        else:
            # Normal text output
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

        # Show difference from primary behavior
        if idx > 0:
            report.append("**What Makes This Different:**")
            diff_points = []

            # Tool usage difference
            if cluster['tool_usage'] != clusters[0]['tool_usage']:
                if not cluster['tool_usage']:
                    diff_points.append("- **Skips tools** that Primary Behavior uses → Users don't get full functionality")
                else:
                    diff_points.append("- **Uses tools** unlike Primary Behavior → More complete but different response")

            # Length difference
            length_diff = ((cluster['avg_length'] - clusters[0]['avg_length']) / clusters[0]['avg_length']) * 100
            if abs(length_diff) > 20:
                if length_diff < 0:
                    diff_points.append(f"- **{abs(length_diff):.0f}% shorter** responses → Less detailed information for users")
                else:
                    diff_points.append(f"- **{abs(length_diff):.0f}% longer** responses → More verbose but inconsistent experience")

            # Structure difference
            if cluster['structure'] != clusters[0]['structure']:
                diff_points.append(f"- **Different format** ({cluster['structure']} vs {clusters[0]['structure']}) → May break parsing")

            if diff_points:
                report.extend(diff_points)
                report.append("")

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

    # Argument Drift Analysis (v0.2.2)
    if result.argument_variance in ["HIGH", "MEDIUM"]:
        report.append("## Argument Drift Analysis\n")
        report.append("The model calls the same tools but with inconsistent arguments:\n")

        arg_details = result.argument_details or {}
        tool_analysis = arg_details.get("tool_argument_analysis", {})

        for tool_name, analysis in tool_analysis.items():
            variations = analysis.get("argument_variations", 1)
            if variations > 1:
                call_count = analysis.get("call_count", 0)
                consistency = analysis.get("consistency_score", 100)
                sample_args = analysis.get("sample_arguments", [])

                report.append(f"### Tool: `{tool_name}`")
                report.append(f"- **Calls:** {call_count} times across all runs")
                report.append(f"- **Unique argument patterns:** {variations}")
                report.append(f"- **Consistency:** {consistency:.0f}%\n")

                if sample_args:
                    report.append("**Sample argument variations:**")
                    for idx, args in enumerate(sample_args[:3], 1):
                        if args:
                            # Format arguments nicely
                            args_str = ", ".join(f"{k}={repr(v)}" for k, v in (args or {}).items())
                            report.append(f"  {idx}. `{args_str}`")
                        else:
                            report.append(f"  {idx}. *(no arguments)*")
                    report.append("")

        report.append("**Impact:** Different arguments may lead to different tool results, causing downstream variance in responses.\n")
        report.append("---\n")

    # Tool Chain Analysis (v0.2.2)
    if result.chain_variance in ["HIGH", "MEDIUM"]:
        report.append("## Tool Chain Analysis\n")
        report.append("The model executes tools in different sequences across runs:\n")

        chain_details = result.chain_details or {}
        seq_dist = chain_details.get("sequence_distribution", {})
        dominant = chain_details.get("dominant_sequence", "(no tools)")
        multi_turn = chain_details.get("multi_turn_runs", 0)

        report.append(f"**Dominant sequence:** `{dominant}`\n")

        if seq_dist:
            report.append("**All observed sequences:**")
            for seq, count in sorted(seq_dist.items(), key=lambda x: -x[1]):
                pct = (count / len(runs)) * 100
                report.append(f"- `{seq}`: {count} runs ({pct:.0f}%)")
            report.append("")

        if multi_turn > 0:
            report.append(f"**Multi-turn runs:** {multi_turn} runs required multiple tool iterations\n")

        report.append("**Impact:** Varying tool sequences can lead to different intermediate states and final outcomes.\n")
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

    # Root Causes with Business Context
    if result.root_causes:
        report.append("## Why Is This Happening?\n")

        for idx, cause in enumerate(result.root_causes, 1):
            business_title = _get_root_cause_business_title(cause.type)
            severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
            emoji = severity_emoji.get(cause.severity, "")

            # Show severity level in business terms
            severity_text = {"CRITICAL": "Critical", "HIGH": "High Priority", "MEDIUM": "Medium Priority", "LOW": "Low Priority"}
            severity_display = severity_text.get(cause.severity, cause.severity)

            report.append(f"### {emoji} Root Cause #{idx}: {business_title} ({severity_display})")
            report.append("")

            report.append(f"**What's Happening:** {cause.description}")

            if cause.details:
                report.append(f"**Why It Matters:** {cause.details}")

            # Add analogy if available
            analogy = _get_root_cause_analogy(cause.type)
            if analogy:
                report.append(f"\n{analogy}")

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

        # Handle different output scenarios
        if not run.output_text and len(run.tool_calls) > 0:
            # Tool-only response (no text output)
            report.append("**Response:** *(Model called tools without text response)*\n")
            report.append("**Actions:**")
            for tc in run.tool_calls:
                report.append(f"  {tc.call_sequence}. `{tc.to_display_string()}`")
            report.append("")
        elif not run.output_text:
            # Empty output with no tools (potential issue)
            report.append("**Response:** ⚠️ *(Empty output - possible error)*")
            report.append("```")
            report.append("")
            report.append("```\n")
        else:
            # Normal text response
            report.append("**Response:**")
            report.append("```")
            # Limit output for very long responses
            output_text = run.output_text[:1000]
            if len(run.output_text) > 1000:
                output_text += "\n... (truncated)"
            report.append(output_text)
            report.append("```\n")

        # Display token usage with context
        if len(run.tool_calls) > 0 and not run.output_text:
            token_info = f"**Token usage:** {run.output_length_tokens} tokens (tool call overhead)"
        elif len(run.tool_calls) > 0:
            token_info = f"**Token usage:** {run.output_length_tokens} tokens (text + {len(run.tool_calls)} tool call{'s' if len(run.tool_calls) > 1 else ''})"
        else:
            token_info = f"**Token usage:** {run.output_length_tokens} tokens"

        report.append(f"{token_info} | **Structure:** {run.output_structure}\n")

    report.append("---\n")
    report.append("## Raw Data\n")
    report.append("See the corresponding JSON file in `reports/` for machine-readable format with all outputs.\n")

    return "\n".join(report)
