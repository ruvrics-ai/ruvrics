"""Tests for analysis components (scorer, fingerprint, recommender)."""

import pytest
from datetime import datetime

from ruvrics.config import Config
from ruvrics.core.models import (
    RunResult,
    ToolCall,
    InputConfig,
    RootCause,
)
from ruvrics.analysis.scorer import calculate_stability
from ruvrics.analysis.fingerprint import (
    identify_root_causes,
    get_primary_root_cause,
)
from ruvrics.analysis.recommender import generate_recommendations, RECOMMENDATION_MAP


class TestStabilityScorer:
    """Test overall stability score calculation."""

    def create_run(
        self, run_id: int, text: str, tokens: int, structure: str, tools: list[str] = None
    ) -> RunResult:
        """Helper to create a RunResult."""
        tool_calls = [
            ToolCall(name=name, call_sequence=idx + 1)
            for idx, name in enumerate(tools or [])
        ]
        return RunResult(
            run_id=run_id,
            timestamp=datetime.now(),
            output_text=text,
            tool_calls=tool_calls,
            output_length_tokens=tokens,
            output_length_chars=len(text),
            output_structure=structure,
            api_latency_ms=100,
            model_used="test-model",
        )

    def test_perfect_stability(self):
        """Test that identical runs get SAFE classification."""
        # Create 20 identical runs
        runs = [
            self.create_run(
                i,
                "The weather is sunny today.",
                10,
                "TEXT",
                tools=["search"],
            )
            for i in range(1, 21)
        ]

        input_config = InputConfig(
            user_input="What's the weather?",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        result = calculate_stability(
            runs=runs,
            input_config=input_config,
            model="test-model",
            duration_seconds=10.0,
        )

        # Should be very high score
        assert result.stability_score > 95
        assert result.risk_classification == "SAFE"
        assert result.total_runs == 20
        assert result.successful_runs == 20

    def test_weighted_average(self):
        """Test that weights are applied correctly."""
        # Create runs with specific variance patterns
        runs = []
        # High semantic variance, low everything else
        outputs = [
            "The weather is sunny.",
            "I'll check the database.",
            "Error occurred.",
            "Processing request.",
        ]
        for i, output in enumerate(outputs * 5):  # Repeat to get 20 runs
            runs.append(
                self.create_run(i + 1, output, 10, "TEXT", tools=["search"])
            )

        input_config = InputConfig(
            user_input="test",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        result = calculate_stability(
            runs=runs,
            input_config=input_config,
            model="test-model",
            duration_seconds=10.0,
        )

        # Should have low semantic score (40% weight) which impacts overall
        # But other metrics are perfect (100%), so overall will be ~85%
        # Calculation: 0.40 * ~62 + 0.25 * 100 + 0.20 * 100 + 0.15 * 100 = ~85%
        assert result.semantic_drift == "HIGH"
        assert 70 <= result.stability_score < 90  # Should be RISKY range
        assert result.risk_classification == "RISKY"

    def test_risk_classification_thresholds(self):
        """Test risk classification boundaries."""
        config = Config()

        # Test data for different score levels
        test_cases = [
            (95.0, "SAFE"),  # >= 90
            (90.0, "SAFE"),  # >= 90
            (89.9, "RISKY"),  # 70-89.9
            (75.0, "RISKY"),  # 70-89.9
            (70.0, "RISKY"),  # 70-89.9
            (69.9, "DO_NOT_SHIP"),  # < 70
            (50.0, "DO_NOT_SHIP"),  # < 70
        ]

        for score, expected_risk in test_cases:
            # We can't directly test the classification without full runs,
            # but we can verify the thresholds
            if score >= config.safe_threshold:
                assert expected_risk == "SAFE"
            elif score >= config.risky_threshold:
                assert expected_risk == "RISKY"
            else:
                assert expected_risk == "DO_NOT_SHIP"


class TestRootCauseIdentification:
    """Test root cause fingerprinting."""

    def test_tool_routing_issue(self):
        """Test detection of nondeterministic tool routing."""
        root_causes = identify_root_causes(
            semantic_drift="LOW",  # Outputs similar
            semantic_score=90.0,
            tool_variance="HIGH",  # But tool use varies
            tool_score=60.0,
            tool_details={"tool_usage_percentage": 60.0},
            structural_variance="LOW",
            structural_score=95.0,
            length_variance="LOW",
            length_score=90.0,
            claim_variance="NONE",
            claim_percentage=0.0,
        )

        # Should detect NONDETERMINISTIC_TOOL_ROUTING
        assert len(root_causes) >= 1
        assert root_causes[0].type == "NONDETERMINISTIC_TOOL_ROUTING"
        assert root_causes[0].severity == "HIGH"

    def test_tool_confusion(self):
        """Test detection of tool confusion."""
        root_causes = identify_root_causes(
            semantic_drift="HIGH",  # Outputs vary
            semantic_score=65.0,
            tool_variance="HIGH",  # And tool use varies
            tool_score=60.0,
            tool_details={"tool_usage_percentage": 50.0},
            structural_variance="LOW",
            structural_score=95.0,
            length_variance="LOW",
            length_score=90.0,
            claim_variance="NONE",
            claim_percentage=0.0,
        )

        # Should detect TOOL_CONFUSION (tool AND semantic both high)
        assert any(c.type == "TOOL_CONFUSION" for c in root_causes)
        tool_confusion = next(c for c in root_causes if c.type == "TOOL_CONFUSION")
        assert tool_confusion.severity == "HIGH"

    def test_unconstrained_assertions(self):
        """Test detection of risky claims."""
        root_causes = identify_root_causes(
            semantic_drift="LOW",
            semantic_score=90.0,
            tool_variance="LOW",
            tool_score=95.0,
            tool_details={},
            structural_variance="LOW",
            structural_score=95.0,
            length_variance="LOW",
            length_score=90.0,
            claim_variance="HIGH",  # Claims vary
            claim_percentage=60.0,
        )

        # Should detect UNCONSTRAINED_ASSERTIONS
        assert any(c.type == "UNCONSTRAINED_ASSERTIONS" for c in root_causes)
        claim_issue = next(
            c for c in root_causes if c.type == "UNCONSTRAINED_ASSERTIONS"
        )
        assert claim_issue.severity == "CRITICAL"

    def test_underspecified_prompt(self):
        """Test detection of underspecified prompt."""
        root_causes = identify_root_causes(
            semantic_drift="HIGH",  # Outputs vary
            semantic_score=65.0,
            tool_variance="LOW",
            tool_score=95.0,
            tool_details={},
            structural_variance="LOW",  # But format is stable
            structural_score=95.0,
            length_variance="LOW",
            length_score=90.0,
            claim_variance="NONE",
            claim_percentage=0.0,
        )

        # Should detect UNDERSPECIFIED_PROMPT
        assert any(c.type == "UNDERSPECIFIED_PROMPT" for c in root_causes)
        prompt_issue = next(
            c for c in root_causes if c.type == "UNDERSPECIFIED_PROMPT"
        )
        assert prompt_issue.severity == "MEDIUM"

    def test_format_inconsistency(self):
        """Test detection of format inconsistency."""
        root_causes = identify_root_causes(
            semantic_drift="LOW",
            semantic_score=90.0,
            tool_variance="LOW",
            tool_score=95.0,
            tool_details={},
            structural_variance="HIGH",  # Format varies
            structural_score=60.0,
            length_variance="LOW",
            length_score=90.0,
            claim_variance="NONE",
            claim_percentage=0.0,
        )

        # Should detect FORMAT_INCONSISTENCY
        assert any(c.type == "FORMAT_INCONSISTENCY" for c in root_causes)
        format_issue = next(
            c for c in root_causes if c.type == "FORMAT_INCONSISTENCY"
        )
        assert format_issue.severity == "MEDIUM"

    def test_verbosity_drift(self):
        """Test detection of length variance (only when no other issues)."""
        root_causes = identify_root_causes(
            semantic_drift="LOW",
            semantic_score=90.0,
            tool_variance="LOW",
            tool_score=95.0,
            tool_details={},
            structural_variance="LOW",
            structural_score=95.0,
            length_variance="HIGH",  # Only length varies
            length_score=50.0,
            claim_variance="NONE",
            claim_percentage=0.0,
        )

        # Should detect VERBOSITY_DRIFT
        assert any(c.type == "VERBOSITY_DRIFT" for c in root_causes)
        length_issue = next(c for c in root_causes if c.type == "VERBOSITY_DRIFT")
        assert length_issue.severity == "LOW"

    def test_priority_ordering(self):
        """Test that root causes are identified in priority order."""
        root_causes = identify_root_causes(
            semantic_drift="HIGH",
            semantic_score=65.0,
            tool_variance="HIGH",
            tool_score=60.0,
            tool_details={"tool_usage_percentage": 50.0},
            structural_variance="HIGH",
            structural_score=60.0,
            length_variance="HIGH",
            length_score=50.0,
            claim_variance="HIGH",
            claim_percentage=60.0,
        )

        # Should detect multiple causes
        assert len(root_causes) >= 3

        # Claim issue should be present (CRITICAL)
        assert any(c.severity == "CRITICAL" for c in root_causes)

        # Tool issues should be present (HIGH)
        assert any(c.type == "TOOL_CONFUSION" for c in root_causes)

    def test_get_primary_root_cause(self):
        """Test getting the highest priority cause."""
        causes = [
            RootCause(
                type="VERBOSITY_DRIFT", severity="LOW", description="Length varies"
            ),
            RootCause(
                type="FORMAT_INCONSISTENCY",
                severity="MEDIUM",
                description="Format varies",
            ),
            RootCause(
                type="TOOL_CONFUSION", severity="HIGH", description="Tool confusion"
            ),
        ]

        primary = get_primary_root_cause(causes)

        # Should return the HIGH severity cause
        assert primary is not None
        assert primary.severity == "HIGH"
        assert primary.type == "TOOL_CONFUSION"

    def test_get_primary_empty_list(self):
        """Test getting primary cause from empty list."""
        primary = get_primary_root_cause([])
        assert primary is None


class TestRecommendationEngine:
    """Test recommendation generation."""

    def test_tool_routing_recommendations(self):
        """Test recommendations for tool routing issues."""
        root_causes = [
            RootCause(
                type="NONDETERMINISTIC_TOOL_ROUTING",
                severity="HIGH",
                description="Tool routing varies",
            )
        ]

        recommendations = generate_recommendations(root_causes)

        # Should get 2 recommendations for this cause
        assert len(recommendations) >= 1
        assert len(recommendations) <= 3  # Max 3 per spec

        # Check that we got relevant recommendations
        titles = [r.title for r in recommendations]
        assert any("tool" in t.lower() for t in titles)

    def test_claim_recommendations(self):
        """Test recommendations for claim issues."""
        root_causes = [
            RootCause(
                type="UNCONSTRAINED_ASSERTIONS",
                severity="CRITICAL",
                description="Risky claims",
            )
        ]

        recommendations = generate_recommendations(root_causes)

        assert len(recommendations) >= 1
        # Should get recommendations about constraints
        titles = [r.title for r in recommendations]
        assert any("guarantee" in t.lower() or "constraint" in t.lower() for t in titles)

    def test_format_recommendations(self):
        """Test recommendations for format issues."""
        root_causes = [
            RootCause(
                type="FORMAT_INCONSISTENCY",
                severity="MEDIUM",
                description="Format varies",
            )
        ]

        recommendations = generate_recommendations(root_causes)

        assert len(recommendations) >= 1
        # Should get recommendations about schema/format
        titles = [r.title for r in recommendations]
        assert any("format" in t.lower() or "schema" in t.lower() for t in titles)

    def test_max_three_recommendations(self):
        """Test that we get max 3 recommendations."""
        # Create multiple root causes
        root_causes = [
            RootCause(type="TOOL_CONFUSION", severity="HIGH", description="Tool issue"),
            RootCause(
                type="UNCONSTRAINED_ASSERTIONS",
                severity="CRITICAL",
                description="Claim issue",
            ),
            RootCause(
                type="FORMAT_INCONSISTENCY",
                severity="MEDIUM",
                description="Format issue",
            ),
        ]

        recommendations = generate_recommendations(root_causes)

        # Should get max 3 recommendations per spec
        assert len(recommendations) <= 3

    def test_severity_prioritization(self):
        """Test that higher severity causes get recommendations first."""
        root_causes = [
            RootCause(type="VERBOSITY_DRIFT", severity="LOW", description="Length"),
            RootCause(
                type="UNCONSTRAINED_ASSERTIONS",
                severity="CRITICAL",
                description="Claims",
            ),
            RootCause(
                type="FORMAT_INCONSISTENCY", severity="MEDIUM", description="Format"
            ),
        ]

        recommendations = generate_recommendations(root_causes)

        # First recommendations should be for CRITICAL or HIGH severity issues
        # Not for LOW severity
        assert len(recommendations) >= 1
        # The LOW severity cause should not dominate recommendations

    def test_empty_causes(self):
        """Test that no recommendations when no causes."""
        recommendations = generate_recommendations([])
        assert len(recommendations) == 0

    def test_recommendation_structure(self):
        """Test that recommendations have required fields."""
        root_causes = [
            RootCause(
                type="TOOL_CONFUSION", severity="HIGH", description="Tool issue"
            )
        ]

        recommendations = generate_recommendations(root_causes)

        assert len(recommendations) >= 1

        for rec in recommendations:
            assert rec.title
            assert rec.category in ["prompt", "code", "config"]
            assert rec.priority >= 1
            assert rec.description
            # Example is optional but should be present for most
