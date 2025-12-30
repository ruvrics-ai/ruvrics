"""
Length consistency metric calculation.

Measures whether response lengths vary significantly.
From spec Section 3.4 - Length Consistency.
"""

import numpy as np

from ruvrics.config import Config, get_config
from ruvrics.core.models import MetricResult, RunResult


def calculate_length_consistency(
    runs: list[RunResult], config: Config | None = None
) -> MetricResult:
    """
    Calculate length consistency using Coefficient of Variation.

    From spec Section 3.4:
    - Use token counts (more reliable than characters)
    - Calculate CV = std / mean
    - Map CV to 0-100 score

    Args:
        runs: List of RunResult objects
        config: Optional configuration

    Returns:
        MetricResult with score (0-100) and variance classification
    """
    cfg = config or get_config()

    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to calculate consistency")

    # Extract token lengths
    lengths = [run.output_length_tokens for run in runs]

    # Calculate statistics
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    min_length = int(np.min(lengths))
    max_length = int(np.max(lengths))

    # Edge case: very short outputs (from spec Section 3.4)
    if mean_length < 5:
        return MetricResult(
            score=100.0,
            variance="LOW",
            details={
                "reason": "Outputs too short to measure variance meaningfully",
                "mean_length": mean_length,
                "std_length": std_length,
                "min_length": min_length,
                "max_length": max_length,
                "coefficient_of_variation": 0.0,
            },
        )

    # Calculate Coefficient of Variation (from spec Section 3.4)
    cv = std_length / mean_length

    # Map to 0-100 score (CV of 0.4 → score of 0)
    # Formula from spec: max(0, (1 - CV/0.4) * 100)
    length_consistency_score = max(0.0, (1 - cv / 0.4) * 100)

    # Classify variance using CV thresholds from spec Section 4
    variance = _classify_length_variance(cv, cfg)

    return MetricResult(
        score=length_consistency_score,
        variance=variance,
        details={
            "mean_length": mean_length,
            "std_length": std_length,
            "min_length": min_length,
            "max_length": max_length,
            "coefficient_of_variation": cv,
            "length_range": f"{min_length}-{max_length}",
        },
    )


def _classify_length_variance(cv: float, config: Config) -> str:
    """
    Classify length variance based on Coefficient of Variation.

    From spec Section 4:
    - LOW: CV < 0.15 (~15% variation)
    - MEDIUM: 0.15 <= CV < 0.30 (~30% variation)
    - HIGH: CV >= 0.30 (30%+ variation)

    Args:
        cv: Coefficient of Variation (std/mean)
        config: Configuration with thresholds

    Returns:
        Variance classification: "LOW", "MEDIUM", or "HIGH"
    """
    if cv < config.length_cv_low_threshold:  # 0.15
        return "LOW"
    elif cv < config.length_cv_medium_threshold:  # 0.30
        return "MEDIUM"
    else:
        return "HIGH"
