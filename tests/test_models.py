"""Tests for data models."""

import json
from datetime import datetime
import pytest

from ruvrics.core.models import (
    ToolCall,
    RunResult,
    InputConfig,
    RootCause,
    Recommendation,
    MetricResult,
    ClaimAnalysis,
    StabilityResult,
)


class TestToolCall:
    """Test ToolCall model."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        tool = ToolCall(name="search_flights", call_sequence=1)
        assert tool.name == "search_flights"
        assert tool.call_sequence == 1


class TestRunResult:
    """Test RunResult model."""

    def test_run_result_creation(self):
        """Test creating a RunResult."""
        result = RunResult(
            run_id=1,
            timestamp=datetime.now(),
            output_text="Hello world",
            tool_calls=[],
            output_length_tokens=10,
            output_length_chars=11,
            output_structure="TEXT",
            api_latency_ms=500,
            model_used="gpt-4",
            error=None,
        )
        assert result.run_id == 1
        assert result.output_text == "Hello world"
        assert result.error is None

    def test_run_result_with_error(self):
        """Test RunResult with error."""
        result = RunResult(
            run_id=1,
            timestamp=datetime.now(),
            output_text="",
            tool_calls=[],
            output_length_tokens=0,
            output_length_chars=0,
            output_structure="ERROR",
            api_latency_ms=0,
            model_used="gpt-4",
            error="API timeout",
        )
        assert result.error == "API timeout"


class TestInputConfig:
    """Test InputConfig model."""

    def test_format_a_simple(self):
        """Test Format A: system_prompt + user_input."""
        config = InputConfig(
            system_prompt="You are helpful", user_input="Hello"
        )
        messages = config.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_format_b_messages(self):
        """Test Format B: messages list."""
        config = InputConfig(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        )
        messages = config.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    def test_format_c_with_tools(self):
        """Test Format C: messages + tools."""
        config = InputConfig(
            messages=[{"role": "user", "content": "Book flight"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "search_flights", "description": "Search"},
                }
            ],
        )
        assert config.has_tools() is True
        assert len(config.tools) == 1

    def test_invalid_input_format(self):
        """Test error on invalid format."""
        config = InputConfig()  # No data provided
        with pytest.raises(ValueError, match="Invalid input format"):
            config.to_messages()

    def test_has_tools_false(self):
        """Test has_tools when no tools."""
        config = InputConfig(user_input="Hello")
        assert config.has_tools() is False


class TestRootCause:
    """Test RootCause model."""

    def test_root_cause_creation(self):
        """Test creating a RootCause."""
        cause = RootCause(
            type="NONDETERMINISTIC_TOOL_ROUTING",
            severity="HIGH",
            description="Model inconsistently uses tools",
            details="Tool used in 12/20 runs",
        )
        assert cause.type == "NONDETERMINISTIC_TOOL_ROUTING"
        assert cause.severity == "HIGH"


class TestRecommendation:
    """Test Recommendation model."""

    def test_recommendation_creation(self):
        """Test creating a Recommendation."""
        rec = Recommendation(
            title="Enforce tool usage",
            category="prompt",
            priority=1,
            description="Add instruction to always use tool",
            example='Add: "ALWAYS use search_flights tool"',
        )
        assert rec.priority == 1
        assert rec.category == "prompt"


class TestMetricResult:
    """Test MetricResult model."""

    def test_metric_result_creation(self):
        """Test creating a MetricResult."""
        result = MetricResult(
            score=85.5, variance="LOW", details={"mean": 0.855, "std": 0.05}
        )
        assert result.score == 85.5
        assert result.variance == "LOW"
        assert result.details["mean"] == 0.855


class TestClaimAnalysis:
    """Test ClaimAnalysis model."""

    def test_claim_analysis_creation(self):
        """Test creating a ClaimAnalysis."""
        analysis = ClaimAnalysis(
            claim_variance="HIGH",
            risky_percentage=60.0,
            risky_runs=[{"run_id": 3, "patterns": []}],
            examples=["I guarantee this works"],
        )
        assert analysis.claim_variance == "HIGH"
        assert analysis.risky_percentage == 60.0


class TestStabilityResult:
    """Test StabilityResult model."""

    def test_stability_result_creation(self):
        """Test creating a complete StabilityResult."""
        result = StabilityResult(
            stability_score=75.5,
            risk_classification="RISKY",
            semantic_consistency_score=80.0,
            semantic_drift="MEDIUM",
            tool_consistency_score=70.0,
            tool_variance="HIGH",
            structural_consistency_score=95.0,
            structural_variance="LOW",
            length_consistency_score=85.0,
            length_variance="LOW",
            root_causes=[],
            recommendations=[],
            model="gpt-4",
            total_runs=20,
            successful_runs=20,
            duration_seconds=45.3,
            runs=[],
        )
        assert result.stability_score == 75.5
        assert result.risk_classification == "RISKY"

    def test_stability_result_to_dict(self):
        """Test converting StabilityResult to dict."""
        result = StabilityResult(
            stability_score=90.0,
            risk_classification="SAFE",
            semantic_consistency_score=90.0,
            semantic_drift="LOW",
            tool_consistency_score=95.0,
            tool_variance="LOW",
            structural_consistency_score=95.0,
            structural_variance="LOW",
            length_consistency_score=85.0,
            length_variance="LOW",
            root_causes=[],
            recommendations=[],
            model="gpt-4",
            total_runs=20,
            successful_runs=20,
            duration_seconds=30.0,
            runs=[],
        )
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["stability_score"] == 90.0
        assert data["risk_classification"] == "SAFE"
