"""
Tool usage consistency metric calculation.

Measures whether the model consistently decides to use tools.
From spec Section 3.2 - Tool Usage Consistency.
"""

from collections import Counter

from ruvrics.config import Config, get_config
from ruvrics.core.models import MetricResult, RunResult


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
