"""
Baseline comparison functionality for Ruvrics.

Enables saving baselines and detecting behavioral drift over time.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel

from ruvrics.core.models import StabilityResult


# Default baselines directory
BASELINES_DIR = Path(".ruvrics/baselines")


class BaselineData(BaseModel):
    """Stored baseline data for comparison."""

    name: str
    created_at: str
    model: str
    stability_score: float
    risk_classification: str
    semantic_consistency_score: float
    tool_consistency_score: float
    structural_consistency_score: float
    length_consistency_score: float
    total_runs: int
    input_hash: Optional[str] = None  # Hash of input for validation

    # Additional metadata
    semantic_drift: str
    tool_variance: str
    structural_variance: str
    length_variance: str


class ComparisonResult(BaseModel):
    """Result of comparing current run to baseline."""

    baseline_name: str
    baseline_score: float
    current_score: float
    score_delta: float  # current - baseline (negative = regression)

    # Component deltas
    semantic_delta: float
    tool_delta: float
    structural_delta: float
    length_delta: float

    # Classification
    status: str  # "IMPROVED", "STABLE", "MINOR_REGRESSION", "MAJOR_REGRESSION"

    # Details
    changes: list[str]  # Human-readable change descriptions


def get_baselines_dir() -> Path:
    """Get or create baselines directory."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINES_DIR


def save_baseline(result: StabilityResult, name: str, input_hash: Optional[str] = None) -> Path:
    """
    Save stability result as a named baseline.

    Args:
        result: StabilityResult to save
        name: Name for the baseline (e.g., "v1.0", "prod", "pre-refactor")
        input_hash: Optional hash of input for validation

    Returns:
        Path to saved baseline file
    """
    baselines_dir = get_baselines_dir()

    baseline = BaselineData(
        name=name,
        created_at=datetime.now().isoformat(),
        model=result.model,
        stability_score=result.stability_score,
        risk_classification=result.risk_classification,
        semantic_consistency_score=result.semantic_consistency_score,
        tool_consistency_score=result.tool_consistency_score,
        structural_consistency_score=result.structural_consistency_score,
        length_consistency_score=result.length_consistency_score,
        total_runs=result.total_runs,
        input_hash=input_hash,
        semantic_drift=result.semantic_drift,
        tool_variance=result.tool_variance,
        structural_variance=result.structural_variance,
        length_variance=result.length_variance,
    )

    # Save to file
    filepath = baselines_dir / f"{name}.json"
    with open(filepath, "w") as f:
        json.dump(baseline.model_dump(), f, indent=2)

    return filepath


def load_baseline(name: str) -> Optional[BaselineData]:
    """
    Load a saved baseline by name.

    Args:
        name: Baseline name

    Returns:
        BaselineData or None if not found
    """
    baselines_dir = get_baselines_dir()
    filepath = baselines_dir / f"{name}.json"

    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    return BaselineData(**data)


def list_baselines() -> list[str]:
    """List all saved baseline names."""
    baselines_dir = get_baselines_dir()

    if not baselines_dir.exists():
        return []

    return [f.stem for f in baselines_dir.glob("*.json")]


def compare_to_baseline(result: StabilityResult, baseline_name: str) -> Optional[ComparisonResult]:
    """
    Compare current result to a saved baseline.

    Args:
        result: Current StabilityResult
        baseline_name: Name of baseline to compare against

    Returns:
        ComparisonResult or None if baseline not found
    """
    baseline = load_baseline(baseline_name)
    if baseline is None:
        return None

    # Calculate deltas
    score_delta = result.stability_score - baseline.stability_score
    semantic_delta = result.semantic_consistency_score - baseline.semantic_consistency_score
    tool_delta = result.tool_consistency_score - baseline.tool_consistency_score
    structural_delta = result.structural_consistency_score - baseline.structural_consistency_score
    length_delta = result.length_consistency_score - baseline.length_consistency_score

    # Determine status
    if score_delta >= 2.0:
        status = "IMPROVED"
    elif score_delta >= -2.0:
        status = "STABLE"
    elif score_delta >= -10.0:
        status = "MINOR_REGRESSION"
    else:
        status = "MAJOR_REGRESSION"

    # Build change descriptions
    changes = []

    if abs(semantic_delta) >= 2.0:
        direction = "improved" if semantic_delta > 0 else "degraded"
        changes.append(f"Semantic consistency {direction}: {baseline.semantic_consistency_score:.1f}% → {result.semantic_consistency_score:.1f}%")

    if abs(tool_delta) >= 2.0:
        direction = "improved" if tool_delta > 0 else "degraded"
        changes.append(f"Tool consistency {direction}: {baseline.tool_consistency_score:.1f}% → {result.tool_consistency_score:.1f}%")

    if abs(structural_delta) >= 2.0:
        direction = "improved" if structural_delta > 0 else "degraded"
        changes.append(f"Structural consistency {direction}: {baseline.structural_consistency_score:.1f}% → {result.structural_consistency_score:.1f}%")

    if abs(length_delta) >= 2.0:
        direction = "improved" if length_delta > 0 else "degraded"
        changes.append(f"Length consistency {direction}: {baseline.length_consistency_score:.1f}% → {result.length_consistency_score:.1f}%")

    if not changes:
        changes.append("No significant changes detected")

    return ComparisonResult(
        baseline_name=baseline_name,
        baseline_score=baseline.stability_score,
        current_score=result.stability_score,
        score_delta=score_delta,
        semantic_delta=semantic_delta,
        tool_delta=tool_delta,
        structural_delta=structural_delta,
        length_delta=length_delta,
        status=status,
        changes=changes,
    )


def compare_models(
    result1: StabilityResult,
    result2: StabilityResult,
    model1_name: str,
    model2_name: str,
) -> dict[str, Any]:
    """
    Compare stability results between two models.

    Args:
        result1: StabilityResult from first model
        result2: StabilityResult from second model
        model1_name: Name of first model
        model2_name: Name of second model

    Returns:
        Comparison dictionary with detailed analysis
    """
    score_delta = result2.stability_score - result1.stability_score

    # Determine which model is more stable
    if abs(score_delta) < 2.0:
        winner = "TIE"
        summary = f"Both models have similar stability (~{result1.stability_score:.1f}%)"
    elif score_delta > 0:
        winner = model2_name
        summary = f"{model2_name} is more stable ({result2.stability_score:.1f}% vs {result1.stability_score:.1f}%)"
    else:
        winner = model1_name
        summary = f"{model1_name} is more stable ({result1.stability_score:.1f}% vs {result2.stability_score:.1f}%)"

    return {
        "model1": {
            "name": model1_name,
            "stability_score": result1.stability_score,
            "risk_classification": result1.risk_classification,
            "semantic": result1.semantic_consistency_score,
            "tool": result1.tool_consistency_score,
            "structural": result1.structural_consistency_score,
            "length": result1.length_consistency_score,
        },
        "model2": {
            "name": model2_name,
            "stability_score": result2.stability_score,
            "risk_classification": result2.risk_classification,
            "semantic": result2.semantic_consistency_score,
            "tool": result2.tool_consistency_score,
            "structural": result2.structural_consistency_score,
            "length": result2.length_consistency_score,
        },
        "comparison": {
            "score_delta": score_delta,
            "winner": winner,
            "summary": summary,
            "semantic_delta": result2.semantic_consistency_score - result1.semantic_consistency_score,
            "tool_delta": result2.tool_consistency_score - result1.tool_consistency_score,
            "structural_delta": result2.structural_consistency_score - result1.structural_consistency_score,
            "length_delta": result2.length_consistency_score - result1.length_consistency_score,
        },
    }
