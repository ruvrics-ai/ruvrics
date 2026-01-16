"""
Tool usage consistency metric calculation.

Measures whether the model consistently decides to use tools.
From spec Section 3.2 - Tool Usage Consistency.

Extended in v0.2.2 to include argument consistency analysis.
"""

import json
from collections import Counter
from typing import Any

from ruvrics.config import Config, get_config
from ruvrics.core.models import MetricResult, RunResult, ToolCall


def normalize_tool_calls(tool_calls: list) -> frozenset[str]:
    """
    Extract unique tool names from tool calls (order-independent).

    From spec Section 3.2:
    - Deduplicates tool names
    - Ignores call order
    - Ignores call frequency
    - Example: [search, search, book, search] → {book, search}

    Args:
        tool_calls: List of ToolCall objects

    Returns:
        Frozen set of unique tool names
    """
    return frozenset(call.name for call in tool_calls)


def calculate_tool_consistency(
    runs: list[RunResult],
    tools_available: bool,
    config: Config | None = None,
) -> MetricResult:
    """
    Calculate tool usage consistency.

    From spec Section 3.2:
    1. Extract unique tool names from each run (order-independent)
    2. Identify most common pattern (mode)
    3. Score = fraction of runs matching mode

    Args:
        runs: List of RunResult objects
        tools_available: Whether tools were provided in input
        config: Optional configuration

    Returns:
        MetricResult with score (0-100) and variance classification
    """
    cfg = config or get_config()

    # Special case: No tools available (from spec Section 3.2)
    if not tools_available:
        return MetricResult(
            score=100.0,
            variance="N/A",
            details={
                "reason": "No tools available in input",
                "pattern_distribution": {},
            },
        )

    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to calculate consistency")

    # Normalize each run's tool calls to sets
    tool_patterns = [normalize_tool_calls(run.tool_calls) for run in runs]

    # Find most common pattern (mode)
    pattern_counts = Counter(tool_patterns)
    most_common_pattern, mode_count = pattern_counts.most_common(1)[0]

    # Calculate consistency (from spec Section 3.2)
    tool_consistency_score = (mode_count / len(runs)) * 100

    # Check for inconsistent tool presence (from spec Section 3.2)
    tools_used_count = sum(1 for p in tool_patterns if len(p) > 0)

    # Special case: Some runs use tools, some don't (always HIGH variance)
    if 0 < tools_used_count < len(runs):
        variance = "HIGH"
        tool_usage_percentage = (tools_used_count / len(runs)) * 100
    else:
        # Classify variance using thresholds from spec Section 4
        variance = _classify_tool_variance(tool_consistency_score, cfg)
        tool_usage_percentage = 100.0 if tools_used_count > 0 else 0.0

    # Convert patterns to readable format for display
    pattern_dist = {
        f"{{{', '.join(sorted(p)) if p else 'none'}}}": count
        for p, count in pattern_counts.items()
    }

    return MetricResult(
        score=tool_consistency_score,
        variance=variance,
        details={
            "most_common_pattern": (
                set(most_common_pattern) if most_common_pattern else set()
            ),
            "pattern_distribution": pattern_dist,
            "tool_usage_percentage": tool_usage_percentage,
            "unique_patterns": len(pattern_counts),
        },
    )


def _classify_tool_variance(score: float, config: Config) -> str:
    """
    Classify tool usage variance based on score.

    From spec Section 4:
    - LOW: score >= 95
    - MEDIUM: 80 <= score < 95
    - HIGH: score < 80

    Args:
        score: Tool consistency score (0-100)
        config: Configuration with thresholds

    Returns:
        Variance classification: "LOW", "MEDIUM", or "HIGH"
    """
    if score >= config.tool_low_threshold:  # 95
        return "LOW"
    elif score >= config.tool_medium_threshold:  # 80
        return "MEDIUM"
    else:
        return "HIGH"


def _normalize_argument_value(value: Any) -> str:
    """
    Normalize argument value for comparison.

    Handles type coercion and case normalization for string comparisons.

    Args:
        value: Argument value of any type

    Returns:
        Normalized string representation
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower().strip()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _calculate_argument_similarity(args1: dict | None, args2: dict | None) -> float:
    """
    Calculate similarity between two argument dictionaries.

    Uses Jaccard-like similarity for keys and exact match for values.

    Args:
        args1: First argument dict
        args2: Second argument dict

    Returns:
        Similarity score (0.0 to 1.0)
    """
    if args1 is None and args2 is None:
        return 1.0
    if args1 is None or args2 is None:
        return 0.0
    if not args1 and not args2:
        return 1.0

    all_keys = set(args1.keys()) | set(args2.keys())
    if not all_keys:
        return 1.0

    matching_score = 0.0
    for key in all_keys:
        if key in args1 and key in args2:
            # Both have this key - compare values
            val1 = _normalize_argument_value(args1[key])
            val2 = _normalize_argument_value(args2[key])
            if val1 == val2:
                matching_score += 1.0
            else:
                # Partial credit for having same key with different value
                matching_score += 0.25
        else:
            # Key missing in one - no match for this key
            pass

    return matching_score / len(all_keys)


def calculate_argument_consistency(
    runs: list[RunResult],
    config: Config | None = None,
) -> MetricResult:
    """
    Calculate tool argument consistency across runs.

    For each tool that appears across runs, compare the arguments passed.
    This detects "argument drift" where the same tool is called but with
    varying parameters (e.g., different date ranges, filters, limits).

    Args:
        runs: List of RunResult objects
        config: Optional configuration

    Returns:
        MetricResult with score (0-100) and variance classification
    """
    cfg = config or get_config()

    # Collect tool calls by name across all runs
    tool_calls_by_name: dict[str, list[tuple[int, ToolCall]]] = {}

    for run in runs:
        if run.error is not None:
            continue
        for tc in run.tool_calls:
            if tc.name not in tool_calls_by_name:
                tool_calls_by_name[tc.name] = []
            tool_calls_by_name[tc.name].append((run.run_id, tc))

    # No tools called at all
    if not tool_calls_by_name:
        return MetricResult(
            score=100.0,
            variance="N/A",
            details={
                "reason": "No tool calls to analyze",
                "tool_argument_analysis": {},
            },
        )

    # Analyze argument consistency per tool
    tool_argument_analysis: dict[str, dict[str, Any]] = {}
    tool_scores: list[float] = []

    for tool_name, calls in tool_calls_by_name.items():
        if len(calls) < 2:
            # Tool only called once - can't measure consistency
            tool_argument_analysis[tool_name] = {
                "call_count": len(calls),
                "argument_variations": 1,
                "consistency_score": 100.0,
                "sample_arguments": [calls[0][1].arguments] if calls else [],
            }
            tool_scores.append(100.0)
            continue

        # Compare all pairs of argument sets
        args_list = [tc.arguments for _, tc in calls]

        # Find unique argument signatures
        unique_signatures: list[dict | None] = []
        signature_counts: list[int] = []

        for args in args_list:
            # Check if this signature matches any existing one
            found = False
            for idx, sig in enumerate(unique_signatures):
                if _calculate_argument_similarity(args, sig) >= 0.9:
                    signature_counts[idx] += 1
                    found = True
                    break

            if not found:
                unique_signatures.append(args)
                signature_counts.append(1)

        # Calculate consistency: mode count / total calls
        max_count = max(signature_counts) if signature_counts else 0
        total_calls = len(calls)
        consistency = (max_count / total_calls) * 100 if total_calls > 0 else 100.0

        # Store analysis
        tool_argument_analysis[tool_name] = {
            "call_count": total_calls,
            "argument_variations": len(unique_signatures),
            "consistency_score": consistency,
            "dominant_pattern_count": max_count,
            "sample_arguments": unique_signatures[:5],  # Top 5 variations
            "variation_distribution": dict(zip(
                [f"pattern_{i+1}" for i in range(len(signature_counts))],
                signature_counts
            )),
        }
        tool_scores.append(consistency)

    # Overall score: weighted average by call frequency
    total_calls = sum(len(calls) for calls in tool_calls_by_name.values())
    if total_calls > 0:
        weighted_score = sum(
            tool_argument_analysis[name]["consistency_score"] * len(tool_calls_by_name[name])
            for name in tool_calls_by_name
        ) / total_calls
    else:
        weighted_score = 100.0

    # Classify variance
    if weighted_score >= cfg.tool_low_threshold:  # 95
        variance = "LOW"
    elif weighted_score >= cfg.tool_medium_threshold:  # 80
        variance = "MEDIUM"
    else:
        variance = "HIGH"

    return MetricResult(
        score=weighted_score,
        variance=variance,
        details={
            "tool_argument_analysis": tool_argument_analysis,
            "total_tool_calls": total_calls,
            "tools_with_variation": sum(
                1 for t in tool_argument_analysis.values()
                if t["argument_variations"] > 1
            ),
        },
    )


def calculate_tool_chain_consistency(
    runs: list[RunResult],
    config: Config | None = None,
) -> MetricResult:
    """
    Calculate tool chain (sequence) consistency across runs.

    For multi-step tool workflows, measures whether the model calls tools
    in the same order across runs. This detects "chain variance" where
    Tool A → Tool B in some runs but Tool B → Tool A in others.

    Args:
        runs: List of RunResult objects
        config: Optional configuration

    Returns:
        MetricResult with score (0-100) and variance classification
    """
    cfg = config or get_config()

    # Extract tool sequences from each run
    sequences: list[tuple[str, ...]] = []

    for run in runs:
        if run.error is not None:
            continue
        if not run.tool_calls:
            sequences.append(())  # Empty sequence
        else:
            # Sort by call_sequence to get ordered list
            sorted_calls = sorted(run.tool_calls, key=lambda tc: tc.call_sequence)
            seq = tuple(tc.name for tc in sorted_calls)
            sequences.append(seq)

    # No tool calls at all
    if not sequences or all(len(s) == 0 for s in sequences):
        return MetricResult(
            score=100.0,
            variance="N/A",
            details={
                "reason": "No tool chains to analyze",
                "unique_sequences": 0,
                "sequence_distribution": {},
            },
        )

    # Count sequence patterns
    sequence_counts = Counter(sequences)
    most_common_seq, mode_count = sequence_counts.most_common(1)[0]

    # Calculate consistency: mode count / total runs
    total_runs = len(sequences)
    consistency_score = (mode_count / total_runs) * 100

    # Build readable sequence distribution
    seq_dist = {}
    for seq, count in sequence_counts.items():
        if not seq:
            key = "(no tools)"
        else:
            key = " → ".join(seq)
        seq_dist[key] = count

    # Classify variance
    if consistency_score >= cfg.tool_low_threshold:  # 95
        variance = "LOW"
    elif consistency_score >= cfg.tool_medium_threshold:  # 80
        variance = "MEDIUM"
    else:
        variance = "HIGH"

    # Check for multi-iteration runs
    multi_turn_runs = sum(1 for run in runs if run.error is None and run.tool_iterations > 1)

    return MetricResult(
        score=consistency_score,
        variance=variance,
        details={
            "unique_sequences": len(sequence_counts),
            "sequence_distribution": seq_dist,
            "dominant_sequence": " → ".join(most_common_seq) if most_common_seq else "(no tools)",
            "dominant_count": mode_count,
            "multi_turn_runs": multi_turn_runs,
            "avg_chain_length": sum(len(s) for s in sequences) / len(sequences) if sequences else 0,
        },
    )
