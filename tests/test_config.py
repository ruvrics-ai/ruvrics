"""Tests for configuration module."""

import os
import pytest
from ruvrics.config import (
    Config,
    ModelConfig,
    get_config,
    get_model_config,
    reset_config,
    SUPPORTED_MODELS,
)
from ruvrics.utils.errors import APIKeyMissingError, ModelNotSupportedError


class TestModelConfig:
    """Test ModelConfig data class."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        config = ModelConfig(
            name="gpt-4", provider="openai", max_tokens=4096, temperature=0.0
        )
        assert config.name == "gpt-4"
        assert config.provider == "openai"
        assert config.supports_tools is True

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig(name="test-model", provider="openai")
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.supports_tools is True


class TestConfig:
    """Test main Config class."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_config_creation(self):
        """Test creating Config with defaults."""
        config = Config()
        assert config.default_runs == 20
        assert config.min_successful_runs == 15
        assert config.max_retries == 3
        assert config.semantic_weight == 0.40
        assert config.tool_weight == 0.25
        assert config.structural_weight == 0.20
        assert config.length_weight == 0.15

    def test_config_thresholds(self):
        """Test that all thresholds match spec."""
        config = Config()
        # Semantic
        assert config.semantic_low_threshold == 85.0
        assert config.semantic_medium_threshold == 70.0
        # Tool
        assert config.tool_low_threshold == 95.0
        assert config.tool_medium_threshold == 80.0
        # Structural
        assert config.structural_low_threshold == 95.0
        assert config.structural_medium_threshold == 85.0
        # Risk classification
        assert config.safe_threshold == 90.0
        assert config.risky_threshold == 70.0

    def test_get_api_key_openai_missing(self):
        """Test error when OpenAI API key missing."""
        config = Config(openai_api_key=None)
        with pytest.raises(APIKeyMissingError) as exc_info:
            config.get_api_key("openai")
        assert exc_info.value.provider == "openai"

    def test_get_api_key_anthropic_missing(self):
        """Test error when Anthropic API key missing."""
        config = Config(anthropic_api_key=None)
        with pytest.raises(APIKeyMissingError) as exc_info:
            config.get_api_key("anthropic")
        assert exc_info.value.provider == "anthropic"

    def test_get_api_key_invalid_provider(self):
        """Test error for invalid provider."""
        config = Config()
        with pytest.raises(ValueError, match="Unknown provider"):
            config.get_api_key("invalid")

    def test_get_api_key_success(self):
        """Test successfully getting API keys."""
        config = Config(openai_api_key="test-key-123", anthropic_api_key="test-key-456")
        assert config.get_api_key("openai") == "test-key-123"
        assert config.get_api_key("anthropic") == "test-key-456"

    def test_validate_runs_range(self):
        """Test that runs must be within 10-50."""
        with pytest.raises(ValueError, match="must be between 10 and 50"):
            Config(default_runs=5)

        with pytest.raises(ValueError, match="must be between 10 and 50"):
            Config(default_runs=100)

        # Valid values should work
        config = Config(default_runs=20)
        assert config.default_runs == 20


class TestModelRegistry:
    """Test supported models registry."""

    def test_supported_models_exist(self):
        """Test that supported models are registered."""
        assert "gpt-4-turbo" in SUPPORTED_MODELS
        assert "gpt-4" in SUPPORTED_MODELS
        assert "gpt-4o" in SUPPORTED_MODELS
        assert "claude-sonnet-4" in SUPPORTED_MODELS

    def test_get_model_config_success(self):
        """Test getting a supported model config."""
        config = get_model_config("gpt-4-turbo")
        assert config.provider == "openai"
        assert config.supports_tools is True

    def test_get_model_config_not_found(self):
        """Test error when model not supported."""
        with pytest.raises(ModelNotSupportedError) as exc_info:
            get_model_config("unsupported-model-xyz")
        assert exc_info.value.model == "unsupported-model-xyz"
        assert len(exc_info.value.supported_models) > 0


class TestConfigSingleton:
    """Test singleton pattern for global config."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_get_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self):
        """Test that reset_config creates new instance."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2
