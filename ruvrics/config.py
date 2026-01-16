"""
Configuration management for Ruvrics AI Stability CLI.

Handles:
- API keys from environment variables
- Model configurations and registry
- Default parameters and thresholds
- Validation of settings
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from ruvrics.utils.errors import APIKeyMissingError, ModelNotSupportedError

# Load environment variables from .env file if present
load_dotenv()


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    name: str
    provider: str  # "openai" or "anthropic"
    max_tokens: int = 4096
    temperature: float = 0.0  # Default to deterministic
    supports_tools: bool = True


class Config(BaseModel):
    """Main configuration class with all settings."""

    # API Keys
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # Default execution settings
    default_runs: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_RUNS", "20"))
    )
    min_successful_runs: int = Field(
        default_factory=lambda: int(os.getenv("MIN_SUCCESSFUL_RUNS", "15"))
    )
    max_successful_runs: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SUCCESSFUL_RUNS", "50"))
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    retry_backoff_multiplier: float = 2.0
    api_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("API_TIMEOUT", "60.0"))
    )

    # Multi-turn execution (v0.2.2 - Scenario 6 support)
    max_tool_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_ITERATIONS", "5"))
    )

    # Embedding model (from spec Appendix C)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Thresholds from spec Section 4 - Semantic Consistency
    semantic_low_threshold: float = 85.0
    semantic_medium_threshold: float = 70.0

    # Tool Consistency thresholds
    tool_low_threshold: float = 95.0
    tool_medium_threshold: float = 80.0

    # Structural Consistency thresholds
    structural_low_threshold: float = 95.0
    structural_medium_threshold: float = 85.0

    # Length Consistency thresholds (CV-based)
    length_cv_low_threshold: float = 0.15  # CV < 0.15 = LOW
    length_cv_medium_threshold: float = 0.30  # CV < 0.30 = MEDIUM
    length_score_low_threshold: float = 80.0  # Score >= 80 = LOW
    length_score_medium_threshold: float = 60.0  # Score >= 60 = MEDIUM

    # Stability score weights (from spec Section 3)
    semantic_weight: float = 0.40
    tool_weight: float = 0.25
    structural_weight: float = 0.20
    length_weight: float = 0.15

    # Risk classification thresholds (from spec Section 3)
    safe_threshold: float = 90.0
    risky_threshold: float = 70.0

    # Claim instability thresholds (from spec Section 5)
    claim_low_threshold: float = 20.0  # < 20% or > 80% = LOW
    claim_high_threshold: float = 80.0  # Between 20-80% = HIGH

    # Results storage
    results_dir: Path = Field(default_factory=lambda: Path.home() / ".ruvrics")

    @field_validator("default_runs")
    @classmethod
    def validate_runs(cls, v: int) -> int:
        """Validate that runs are within acceptable range."""
        if not 10 <= v <= 50:
            raise ValueError("default_runs must be between 10 and 50")
        return v

    @field_validator("min_successful_runs")
    @classmethod
    def validate_min_successful(cls, v: int) -> int:
        """Validate minimum successful runs requirement."""
        if v < 10:
            raise ValueError("min_successful_runs must be at least 10")
        return v

    def get_api_key(self, provider: str) -> str:
        """
        Get API key for specified provider.

        Args:
            provider: Either "openai" or "anthropic"

        Returns:
            API key string

        Raises:
            APIKeyMissingError: If API key not found for the provider
        """
        if provider == "openai":
            if not self.openai_api_key:
                raise APIKeyMissingError("openai")
            return self.openai_api_key
        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise APIKeyMissingError("anthropic")
            return self.anthropic_api_key
        else:
            raise ValueError(f"Unknown provider: {provider}. Must be 'openai' or 'anthropic'.")

    def ensure_results_dir(self) -> None:
        """Create results directory if it doesn't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Supported models registry (from Configuration Guide + spec examples)
SUPPORTED_MODELS: dict[str, ModelConfig] = {
    # OpenAI models
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo-preview", provider="openai", supports_tools=True
    ),
    "gpt-4": ModelConfig(name="gpt-4", provider="openai", supports_tools=True),
    "gpt-4o": ModelConfig(name="gpt-4o", provider="openai", supports_tools=True),
    "gpt-4o-mini": ModelConfig(name="gpt-4o-mini", provider="openai", supports_tools=True),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo", provider="openai", supports_tools=True
    ),
    # Anthropic models
    "claude-opus-4": ModelConfig(
        name="claude-opus-4-20250514", provider="anthropic", supports_tools=True
    ),
    "claude-sonnet-4": ModelConfig(
        name="claude-sonnet-4-20250514", provider="anthropic", supports_tools=True
    ),
    "claude-sonnet-3.5": ModelConfig(
        name="claude-3-5-sonnet-20241022", provider="anthropic", supports_tools=True
    ),
    "claude-haiku-4": ModelConfig(
        name="claude-haiku-4-20250514", provider="anthropic", supports_tools=True
    ),
}


def get_model_config(model_identifier: str) -> ModelConfig:
    """
    Get configuration for a model by identifier.

    Args:
        model_identifier: Short model name (e.g., "gpt-4-turbo")

    Returns:
        ModelConfig with provider and settings

    Raises:
        ModelNotSupportedError: If model not supported
    """
    if model_identifier in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_identifier]

    # Provide helpful error message with supported models
    raise ModelNotSupportedError(model_identifier, list(SUPPORTED_MODELS.keys()))


# Global config instance (singleton pattern)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global config instance (singleton).

    Returns:
        Config instance with all settings loaded
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset config singleton (useful for testing)."""
    global _config
    _config = None
