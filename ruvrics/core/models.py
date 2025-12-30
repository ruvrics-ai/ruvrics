"""
Data models for Ruvrics AI Stability Engine.

Defines all data structures used throughout the application,
following the specification in AI_STABILITY_FINAL_SPEC.md Section 2.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """
    Represents a single tool/function call made by the model.

    Attributes:
        name: Tool/function name
        call_sequence: Order in which this call was made (1st, 2nd, etc.)
    """

    name: str
    call_sequence: int


class RunResult(BaseModel):
    """
    Results from a single execution run.

    This matches the exact schema defined in spec Section 2.
    """

    run_id: int  # 1 to N
    timestamp: datetime  # ISO 8601 format
    output_text: str  # Final text response
    tool_calls: list[ToolCall]  # List of tool calls made
    output_length_tokens: int  # Token count from API
    output_length_chars: int  # Character count
    output_structure: str  # "json", "markdown", "text", etc.
    api_latency_ms: int  # Response time in milliseconds
    model_used: str  # Actual model identifier from API
    error: Optional[str] = None  # Error message if failed

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class InputConfig(BaseModel):
    """
    Input configuration supporting three formats from spec Section 1.

    Format A (Simple):
        system_prompt + user_input

    Format B (Messages):
        messages list (OpenAI/Anthropic compatible)

    Format C (Tool-enabled):
        messages + tools + tool_choice
    """

    # Format A fields
    system_prompt: Optional[str] = None
    user_input: Optional[str] = None

    # Format B & C fields
    messages: Optional[list[dict[str, str]]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: str = "auto"

    # Model parameters (can be overridden)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def to_messages(self) -> list[dict[str, str]]:
        """
        Convert any input format to messages format.

        Returns:
            List of message dicts with role and content

        Raises:
            ValueError: If input format is invalid
        """
        if self.messages is not None:
            return self.messages

        if self.system_prompt and self.user_input:
            # Format A: Convert to messages
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_input},
            ]

        if self.user_input:
            # Just user input, no system prompt
            return [{"role": "user", "content": self.user_input}]

        raise ValueError(
            "Invalid input format. Must provide either:\n"
            "- system_prompt + user_input (Format A)\n"
            "- messages (Format B)\n"
            "- messages + tools (Format C)"
        )

    def has_tools(self) -> bool:
        """Check if tools are provided in the configuration."""
        return self.tools is not None and len(self.tools) > 0


class RootCause(BaseModel):
    """
    Identified root cause of instability.

    From spec Section 4 - Instability Fingerprint.
    """

    type: str  # e.g., "NONDETERMINISTIC_TOOL_ROUTING"
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    description: str  # Human-readable explanation
    details: Optional[str] = None  # Additional context


class Recommendation(BaseModel):
    """
    Actionable recommendation to fix instability.

    From spec Section 6 - Recommendation Decision Tree.
    """

    title: str  # Short description
    category: str  # "prompt", "code", "config"
    priority: int  # 1 = highest priority
    description: str  # Detailed explanation
    example: Optional[str] = None  # Code or prompt example


class MetricResult(BaseModel):
    """
    Results for a single metric calculation.

    Used internally to pass metric results between components.
    """

    score: float  # 0-100 score
    variance: str  # "LOW", "MEDIUM", "HIGH", or "N/A"
    details: dict[str, Any] = Field(default_factory=dict)  # Additional data


class ClaimAnalysis(BaseModel):
    """
    Analysis of risky claims across runs.

    From spec Section 5 - Claim / Safety Instability.
    """

    claim_variance: str  # "NONE", "LOW", "HIGH"
    risky_percentage: float  # Percentage of runs with risky claims
    risky_runs: list[dict[str, Any]]  # Runs that contained risky claims
    examples: list[str] = Field(default_factory=list)  # Example risky claims


class StabilityResult(BaseModel):
    """
    Complete stability analysis results.

    This is the main output of the stability analysis process.
    From spec Section 7 - Output Contract.
    """

    # Overall metrics
    stability_score: float  # 0-100 overall score
    risk_classification: str  # "SAFE", "RISKY", "DO_NOT_SHIP"

    # Component scores and classifications
    semantic_consistency_score: float
    semantic_drift: str  # "LOW", "MEDIUM", "HIGH"

    tool_consistency_score: float
    tool_variance: str  # "LOW", "MEDIUM", "HIGH", "N/A"

    structural_consistency_score: float
    structural_variance: str  # "LOW", "MEDIUM", "HIGH"

    length_consistency_score: float
    length_variance: str  # "LOW", "MEDIUM", "HIGH"

    # Claim analysis (optional, only if risky patterns detected)
    claim_analysis: Optional[ClaimAnalysis] = None

    # Root cause and recommendations
    root_causes: list[RootCause]
    recommendations: list[Recommendation]

    # Execution metadata
    model: str
    total_runs: int
    successful_runs: int
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.now)

    # Raw run results
    runs: list[RunResult]

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return self.model_dump()

    def save_to_file(self, filepath: str) -> None:
        """
        Save results to JSON file.

        Args:
            filepath: Path where to save the results
        """
        import json
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
