"""
Overall stability score calculation.

Combines all metric scores with weighted average and classifies risk.
From spec Section 3 - Stability Score Formula.
"""

import time
from datetime import datetime

from ruvrics.config import Config, get_config
from ruvrics.core.models import RunResult, StabilityResult, InputConfig
from ruvrics.metrics.semantic import calculate_semantic_consistency
from ruvrics.metrics.tool import calculate_tool_consistency
from ruvrics.metrics.structural import calculate_structural_consistency
from ruvrics.metrics.length import calculate_length_consistency
from ruvrics.metrics.claims import analyze_claim_instability


def calculate_stability(
    runs: list[RunResult],
    input_config: InputConfig,
    model: str,
    duration_seconds: float,
    config: Config | None = None,
) -> StabilityResult:
    """
    Calculate overall stability score from all runs.

    From spec Section 3:
    - Semantic: 40% weight
    - Tool: 25% weight
    - Structural: 20% weight
    - Length: 15% weight

    Args:
        runs: List of successful RunResult objects
        input_config: Original input configuration
        model: Model identifier
        duration_seconds: Total execution time
        config: Optional configuration

    Returns:
        StabilityResult with complete analysis
    """
    cfg = config or get_config()

    # Extract successful runs only
    successful_runs = [r for r in runs if r.error is None]

    if len(successful_runs) < 2:
        raise ValueError("Need at least 2 successful runs for stability analysis")

    # Calculate all metrics
    outputs = [r.output_text for r in successful_runs]

    # Semantic consistency (40% weight)
    semantic_result = calculate_semantic_consistency(outputs, config=cfg)

    # Tool consistency (25% weight)
    tools_available = input_config.has_tools()
    tool_result = calculate_tool_consistency(
        successful_runs, tools_available=tools_available, config=cfg
    )

    # Structural consistency (20% weight)
    structural_result = calculate_structural_consistency(successful_runs, config=cfg)

    # Length consistency (15% weight)
    length_result = calculate_length_consistency(successful_runs, config=cfg)

    # Claim instability (does NOT affect score per spec Section 5)
    claim_result = analyze_claim_instability(successful_runs, config=cfg)

    # Calculate weighted average (from spec Section 3)
    stability_score = (
        cfg.semantic_weight * semantic_result.score
        + cfg.tool_weight * tool_result.score
        + cfg.structural_weight * structural_result.score
        + cfg.length_weight * length_result.score
    )

    # Apply claim penalty if HIGH instability (not in original spec but makes sense)
    # Actually, re-reading spec Section 5: "Does NOT affect overall stability score"
    # So we won't apply penalty, just report it separately

    # Classify risk (from spec Section 3)
    if stability_score >= cfg.safe_threshold:  # 90
        risk_classification = "SAFE"
    elif stability_score >= cfg.risky_threshold:  # 70
        risk_classification = "RISKY"
    else:
        risk_classification = "DO_NOT_SHIP"

    # Identify root causes (will implement in fingerprint.py)
    from ruvrics.analysis.fingerprint import identify_root_causes

    root_causes = identify_root_causes(
        semantic_drift=semantic_result.variance,
        semantic_score=semantic_result.score,
        tool_variance=tool_result.variance,
        tool_score=tool_result.score,
        tool_details=tool_result.details,
        structural_variance=structural_result.variance,
        structural_score=structural_result.score,
        length_variance=length_result.variance,
        length_score=length_result.score,
        claim_variance=claim_result.claim_variance,
        claim_percentage=claim_result.risky_percentage,
        config=cfg,
    )

    # Generate recommendations (will implement in recommender.py)
    from ruvrics.analysis.recommender import generate_recommendations

    recommendations = generate_recommendations(root_causes, config=cfg)

    return StabilityResult(
        # Overall metrics
        stability_score=stability_score,
        risk_classification=risk_classification,
        # Component scores
        semantic_consistency_score=semantic_result.score,
        semantic_drift=semantic_result.variance,
        tool_consistency_score=tool_result.score,
        tool_variance=tool_result.variance,
        structural_consistency_score=structural_result.score,
        structural_variance=structural_result.variance,
        length_consistency_score=length_result.score,
        length_variance=length_result.variance,
        # Claim analysis
        claim_analysis=claim_result,
        # Root causes and recommendations
        root_causes=root_causes,
        recommendations=recommendations,
        # Metadata
        model=model,
        total_runs=len(runs),
        successful_runs=len(successful_runs),
        duration_seconds=duration_seconds,
        timestamp=datetime.now(),
        runs=runs,
    )
