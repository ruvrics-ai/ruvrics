"""Tests for all metric calculators."""

import pytest
import numpy as np
from datetime import datetime

from ruvrics.config import Config
from ruvrics.core.models import RunResult, ToolCall
from ruvrics.metrics.semantic import SemanticAnalyzer, calculate_semantic_consistency
from ruvrics.metrics.tool import (
    normalize_tool_calls,
    calculate_tool_consistency,
)
from ruvrics.metrics.structural import calculate_structural_consistency
from ruvrics.metrics.length import calculate_length_consistency
from ruvrics.metrics.claims import (
    detect_risky_claims,
    analyze_claim_instability,
    RISKY_CLAIM_PATTERNS,
)
from ruvrics.utils.errors import EmbeddingError


class TestSemanticConsistency:
    """Test semantic consistency metric."""

    def test_identical_outputs(self):
        """Test that identical outputs get perfect score."""
        outputs = ["The weather is sunny today."] * 10

        result = calculate_semantic_consistency(outputs)

        # Should be very close to 100
        assert result.score > 99.0
        assert result.variance == "LOW"

    def test_similar_paraphrases(self):
        """Test similar meanings (paraphrases)."""
        outputs = [
            "The weather is sunny.",
            "It's sunny today.",
            "Today is a sunny day.",
            "The sun is shining.",
        ]

        result = calculate_semantic_consistency(outputs)

        # Should be high but not perfect
        assert 85 <= result.score <= 100
        assert result.variance in ["LOW", "MEDIUM"]

    def test_different_meanings(self):
        """Test semantically different outputs."""
        outputs = [
            "The weather is sunny.",
            "I'll check the database.",
            "Error occurred.",
            "Processing request.",
        ]

        result = calculate_semantic_consistency(outputs)

        # Should be low
        assert result.score < 70
        assert result.variance == "HIGH"

    def test_variance_classification(self):
        """Test variance thresholds."""
        config = Config()

        analyzer = SemanticAnalyzer(config=config)

        # Test LOW threshold (>= 85)
        assert analyzer._classify_drift(90.0) == "LOW"
        assert analyzer._classify_drift(85.0) == "LOW"

        # Test MEDIUM threshold (70-84)
        assert analyzer._classify_drift(84.9) == "MEDIUM"
        assert analyzer._classify_drift(70.0) == "MEDIUM"

        # Test HIGH threshold (< 70)
        assert analyzer._classify_drift(69.9) == "HIGH"
        assert analyzer._classify_drift(50.0) == "HIGH"

    def test_insufficient_outputs(self):
        """Test error with too few outputs."""
        with pytest.raises(ValueError, match="at least 2 outputs"):
            calculate_semantic_consistency(["single output"])


class TestToolConsistency:
    """Test tool consistency metric."""

    def setup_method(self):
        """Set up test data."""
        self.timestamp = datetime.now()

    def create_run(self, run_id: int, tool_names: list[str]) -> RunResult:
        """Helper to create a RunResult with specific tools."""
        tool_calls = [
            ToolCall(name=name, call_sequence=idx + 1)
            for idx, name in enumerate(tool_names)
        ]
        return RunResult(
            run_id=run_id,
            timestamp=self.timestamp,
            output_text="test",
            tool_calls=tool_calls,
            output_length_tokens=10,
            output_length_chars=10,
            output_structure="TEXT",
            api_latency_ms=100,
            model_used="test-model",
        )

    def test_normalize_tool_calls(self):
        """Test tool call normalization."""
        tools = [
            ToolCall(name="search", call_sequence=1),
            ToolCall(name="search", call_sequence=2),
            ToolCall(name="book", call_sequence=3),
            ToolCall(name="search", call_sequence=4),
        ]

        normalized = normalize_tool_calls(tools)

        # Should be {book, search} (deduplicated, order-independent)
        assert normalized == frozenset(["book", "search"])

    def test_perfect_consistency(self):
        """Test 100% tool consistency."""
        runs = [
            self.create_run(i, ["search"]) for i in range(1, 21)
        ]  # All use "search"

        result = calculate_tool_consistency(runs, tools_available=True)

        assert result.score == 100.0
        assert result.variance == "LOW"
        assert result.details["most_common_pattern"] == {"search"}

    def test_some_variance(self):
        """Test moderate tool variance."""
        runs = []
        # 17 runs use "search", 3 use "search" and "book"
        for i in range(1, 18):
            runs.append(self.create_run(i, ["search"]))
        for i in range(18, 21):
            runs.append(self.create_run(i, ["search", "book"]))

        result = calculate_tool_consistency(runs, tools_available=True)

        # 17/20 = 85%
        assert result.score == 85.0
        assert result.variance == "MEDIUM"

    def test_high_variance(self):
        """Test high tool variance."""
        runs = []
        # 12 use "search", 8 use nothing
        for i in range(1, 13):
            runs.append(self.create_run(i, ["search"]))
        for i in range(13, 21):
            runs.append(self.create_run(i, []))

        result = calculate_tool_consistency(runs, tools_available=True)

        # 12/20 = 60%
        assert result.score == 60.0
        assert result.variance == "HIGH"

    def test_no_tools_available(self):
        """Test N/A when no tools provided."""
        runs = [self.create_run(i, []) for i in range(1, 21)]

        result = calculate_tool_consistency(runs, tools_available=False)

        assert result.score == 100.0
        assert result.variance == "N/A"

    def test_inconsistent_tool_presence(self):
        """Test that inconsistent tool use is always HIGH."""
        runs = []
        # Some runs use tools, some don't (12 yes, 8 no)
        for i in range(1, 13):
            runs.append(self.create_run(i, ["search"]))
        for i in range(13, 21):
            runs.append(self.create_run(i, []))

        result = calculate_tool_consistency(runs, tools_available=True)

        # Should be HIGH regardless of score
        assert result.variance == "HIGH"
        assert result.details["tool_usage_percentage"] == 60.0


class TestStructuralConsistency:
    """Test structural consistency metric."""

    def setup_method(self):
        """Set up test data."""
        self.timestamp = datetime.now()

    def create_run(self, run_id: int, structure: str) -> RunResult:
        """Helper to create a RunResult with specific structure."""
        return RunResult(
            run_id=run_id,
            timestamp=self.timestamp,
            output_text="test",
            tool_calls=[],
            output_length_tokens=10,
            output_length_chars=10,
            output_structure=structure,
            api_latency_ms=100,
            model_used="test-model",
        )

    def test_perfect_consistency(self):
        """Test 100% structural consistency."""
        runs = [
            self.create_run(i, "JSON:DICT:data,status") for i in range(1, 21)
        ]

        result = calculate_structural_consistency(runs)

        assert result.score == 100.0
        assert result.variance == "LOW"
        assert result.details["dominant_structure"] == "JSON:DICT:data,status"

    def test_some_variance(self):
        """Test moderate structural variance."""
        runs = []
        # 18 JSON, 2 TEXT
        for i in range(1, 19):
            runs.append(self.create_run(i, "JSON:DICT:data,status"))
        for i in range(19, 21):
            runs.append(self.create_run(i, "TEXT"))

        result = calculate_structural_consistency(runs)

        # 18/20 = 90%
        assert result.score == 90.0
        assert result.variance == "MEDIUM"

    def test_high_variance(self):
        """Test high structural variance."""
        runs = []
        # 12 JSON, 5 MARKDOWN, 3 TEXT
        for i in range(1, 13):
            runs.append(self.create_run(i, "JSON:DICT:data"))
        for i in range(13, 18):
            runs.append(self.create_run(i, "MARKDOWN"))
        for i in range(18, 21):
            runs.append(self.create_run(i, "TEXT"))

        result = calculate_structural_consistency(runs)

        # 12/20 = 60%
        assert result.score == 60.0
        assert result.variance == "HIGH"


class TestLengthConsistency:
    """Test length consistency metric."""

    def setup_method(self):
        """Set up test data."""
        self.timestamp = datetime.now()

    def create_run(self, run_id: int, tokens: int) -> RunResult:
        """Helper to create a RunResult with specific token count."""
        return RunResult(
            run_id=run_id,
            timestamp=self.timestamp,
            output_text="test",
            tool_calls=[],
            output_length_tokens=tokens,
            output_length_chars=tokens * 4,  # Rough estimate
            output_structure="TEXT",
            api_latency_ms=100,
            model_used="test-model",
        )

    def test_low_variance(self):
        """Test low length variance (CV < 0.15)."""
        # Mean=100, std=10, CV=0.10
        runs = [self.create_run(i, length) for i, length in enumerate(range(90, 110))]

        result = calculate_length_consistency(runs)

        assert result.variance == "LOW"
        assert result.details["coefficient_of_variation"] < 0.15

    def test_medium_variance(self):
        """Test medium length variance (0.15 <= CV < 0.30)."""
        # Create data with CV ~ 0.20 (mean=100, std=20)
        # Use normal distribution to get the right spread
        np.random.seed(42)  # For reproducibility
        lengths = np.random.normal(loc=100, scale=20, size=20).astype(int)
        lengths = np.clip(lengths, 50, 150)  # Ensure positive values

        runs = [self.create_run(i, int(length)) for i, length in enumerate(lengths)]

        result = calculate_length_consistency(runs)

        cv = result.details["coefficient_of_variation"]
        # Should be MEDIUM (CV between 0.15 and 0.30)
        assert 0.15 <= cv < 0.30
        assert result.variance == "MEDIUM"

    def test_high_variance(self):
        """Test high length variance (CV >= 0.30)."""
        # Create data with CV ~ 0.40 (mean=100, std=40)
        np.random.seed(123)  # For reproducibility
        lengths = np.random.normal(loc=100, scale=40, size=20).astype(int)
        lengths = np.clip(lengths, 20, 200)  # Ensure positive values

        runs = [self.create_run(i, int(length)) for i, length in enumerate(lengths)]

        result = calculate_length_consistency(runs)

        cv = result.details["coefficient_of_variation"]
        # Should be HIGH (CV >= 0.30)
        assert cv >= 0.30
        assert result.variance == "HIGH"

    def test_very_short_outputs(self):
        """Test edge case: very short outputs."""
        # Mean < 5 tokens
        runs = [self.create_run(i, 2) for i in range(1, 21)]

        result = calculate_length_consistency(runs)

        # Should return 100 score per spec
        assert result.score == 100.0
        assert result.variance == "LOW"
        assert "too short" in result.details["reason"]


class TestClaimDetection:
    """Test claim pattern detection."""

    def test_detect_guarantee(self):
        """Test detecting guarantee claims."""
        output = "I guarantee this will work perfectly for your use case."

        result = detect_risky_claims(output)

        assert result["has_risky_claims"] is True
        assert any(
            p["pattern"] == "guarantee" for p in result["patterns_found"]
        )

    def test_detect_false_authority(self):
        """Test detecting false authority claims."""
        output = "This is certified by the FDA and approved by experts."

        result = detect_risky_claims(output)

        assert result["has_risky_claims"] is True

    def test_detect_false_capability(self):
        """Test detecting false capability claims."""
        output = "I have access to your database and can verify the information."

        result = detect_risky_claims(output)

        assert result["has_risky_claims"] is True
        assert any(
            p["pattern"] == "false_capability" for p in result["patterns_found"]
        )

    def test_no_risky_claims(self):
        """Test output without risky claims."""
        output = "This typically works well. It may help with your use case."

        result = detect_risky_claims(output)

        assert result["has_risky_claims"] is False
        assert len(result["patterns_found"]) == 0

    def test_analyze_claim_instability_none(self):
        """Test no claims across runs."""
        runs = []
        for i in range(1, 21):
            runs.append(
                RunResult(
                    run_id=i,
                    timestamp=datetime.now(),
                    output_text="This may help with your task.",
                    tool_calls=[],
                    output_length_tokens=10,
                    output_length_chars=30,
                    output_structure="TEXT",
                    api_latency_ms=100,
                    model_used="test-model",
                )
            )

        result = analyze_claim_instability(runs)

        assert result.claim_variance == "NONE"
        assert result.risky_percentage == 0.0

    def test_analyze_claim_instability_high(self):
        """Test inconsistent claims (HIGH variance)."""
        runs = []
        # 12 runs with risky claims, 8 without (60%)
        for i in range(1, 13):
            runs.append(
                RunResult(
                    run_id=i,
                    timestamp=datetime.now(),
                    output_text="I guarantee this is 100% safe.",
                    tool_calls=[],
                    output_length_tokens=10,
                    output_length_chars=30,
                    output_structure="TEXT",
                    api_latency_ms=100,
                    model_used="test-model",
                )
            )
        for i in range(13, 21):
            runs.append(
                RunResult(
                    run_id=i,
                    timestamp=datetime.now(),
                    output_text="This may be helpful.",
                    tool_calls=[],
                    output_length_tokens=10,
                    output_length_chars=20,
                    output_structure="TEXT",
                    api_latency_ms=100,
                    model_used="test-model",
                )
            )

        result = analyze_claim_instability(runs)

        assert result.claim_variance == "HIGH"  # 60% is between 20-80%
        assert result.risky_percentage == 60.0
        assert len(result.risky_runs) == 12

    def test_analyze_claim_instability_low(self):
        """Test consistent claims (LOW variance)."""
        runs = []
        # Only 2 runs with risky claims (10%)
        for i in range(1, 3):
            runs.append(
                RunResult(
                    run_id=i,
                    timestamp=datetime.now(),
                    output_text="I guarantee this works.",
                    tool_calls=[],
                    output_length_tokens=10,
                    output_length_chars=25,
                    output_structure="TEXT",
                    api_latency_ms=100,
                    model_used="test-model",
                )
            )
        for i in range(3, 21):
            runs.append(
                RunResult(
                    run_id=i,
                    timestamp=datetime.now(),
                    output_text="This may help.",
                    tool_calls=[],
                    output_length_tokens=10,
                    output_length_chars=15,
                    output_structure="TEXT",
                    api_latency_ms=100,
                    model_used="test-model",
                )
            )

        result = analyze_claim_instability(runs)

        assert result.claim_variance == "LOW"  # 10% < 20%
        assert result.risky_percentage == 10.0
