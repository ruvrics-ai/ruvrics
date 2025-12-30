"""
Structural consistency metric calculation.

Measures whether output format is stable (JSON, Markdown, Text).
From spec Section 3.3 - Structural Consistency.
"""

from collections import Counter

from ruvrics.config import Config, get_config
from ruvrics.core.models import MetricResult, RunResult


def calculate_structural_consistency(
    runs: list[RunResult], config: Config | None = None
) -> MetricResult:
    """
    Calculate structural consistency.

    From spec Section 3.3:
    - Classify each output's structure
    - Find dominant pattern
    - Score = % matching dominant structure

    Structure detection is done in executor (output_structure field).

    Args:
        runs: List of RunResult objects with output_structure
        config: Optional configuration

    Returns:
        MetricResult with score (0-100) and variance classification
    """
    cfg = config or get_config()

    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to calculate consistency")

    # Get structure types from runs (already detected by executor)
    structure_types = [run.output_structure for run in runs]

    # Count structure types
    type_counts = Counter(structure_types)

    # Most common structure (dominant pattern)
    dominant_structure, dominant_count = type_counts.most_common(1)[0]

    # Score = % matching dominant (from spec Section 3.3)
    structural_consistency_score = (dominant_count / len(runs)) * 100

    # Classify variance using thresholds from spec Section 4
    variance = _classify_structural_variance(structural_consistency_score, cfg)

    # Prepare distribution for details
    distribution = dict(type_counts)

    return MetricResult(
        score=structural_consistency_score,
        variance=variance,
        details={
            "dominant_structure": dominant_structure,
            "dominant_count": dominant_count,
            "structure_distribution": distribution,
            "unique_structures": len(type_counts),
        },
    )


def _classify_structural_variance(score: float, config: Config) -> str:
    """
    Classify structural variance based on score.

    From spec Section 4:
    - LOW: score >= 95
    - MEDIUM: 85 <= score < 95
    - HIGH: score < 85

    Args:
        score: Structural consistency score (0-100)
        config: Configuration with thresholds

    Returns:
        Variance classification: "LOW", "MEDIUM", or "HIGH"
    """
    if score >= config.structural_low_threshold:  # 95
        return "LOW"
    elif score >= config.structural_medium_threshold:  # 85
        return "MEDIUM"
    else:
        return "HIGH"
