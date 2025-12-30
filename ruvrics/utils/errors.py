"""
Custom exceptions for Ruvrics AI Stability Engine.

Provides clear, actionable error messages for users.
From spec Section 8 - Error Handling.
"""

from typing import Optional


class RuvricsError(Exception):
    """Base exception for all Ruvrics errors."""

    pass


class ConfigurationError(RuvricsError):
    """Raised when configuration is invalid or missing."""

    pass


class APIError(RuvricsError):
    """Raised when LLM API calls fail."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limit is hit."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        """
        Initialize rate limit error.

        Args:
            message: Error description
            retry_after: Seconds to wait before retrying (if provided by API)
        """
        super().__init__(message)
        self.retry_after = retry_after


class TimeoutError(APIError):
    """Raised when API request times out."""

    pass


class InsufficientDataError(RuvricsError):
    """
    Raised when not enough successful runs for reliable analysis.

    From spec Section 8: Require minimum 15/20 successful runs.
    """

    def __init__(self, successful: int, total: int, minimum: int = 15):
        """
        Initialize insufficient data error.

        Args:
            successful: Number of successful runs
            total: Total number of runs attempted
            minimum: Minimum required successful runs
        """
        message = (
            f"Only {successful}/{total} runs succeeded. "
            f"Need at least {minimum} successful runs for reliable analysis.\n\n"
            f"Possible causes:\n"
            f"- API connectivity issues\n"
            f"- Rate limiting\n"
            f"- Invalid request format\n"
            f"- Model availability issues\n\n"
            f"Check the error logs for failed runs."
        )
        super().__init__(message)
        self.successful = successful
        self.total = total
        self.minimum = minimum


class EmbeddingError(RuvricsError):
    """
    Raised when embedding model fails.

    From spec Section 8: Critical failure that should abort immediately.
    """

    def __init__(self, message: str, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding error.

        Args:
            message: Error description
            model: Embedding model that failed
        """
        full_message = (
            f"Failed to embed outputs using sentence-transformers/{model}.\n\n"
            f"Error: {message}\n\n"
            f"Solution:\n"
            f"1. Ensure sentence-transformers is installed:\n"
            f"   pip install sentence-transformers\n"
            f"2. Check your internet connection (first-time model download)\n"
            f"3. Verify sufficient disk space (~80MB for model)"
        )
        super().__init__(full_message)
        self.model = model


class InvalidInputError(RuvricsError):
    """Raised when input format is invalid."""

    pass


class ModelNotSupportedError(RuvricsError):
    """Raised when specified model is not supported."""

    def __init__(self, model: str, supported_models: list[str]):
        """
        Initialize model not supported error.

        Args:
            model: Model identifier that was requested
            supported_models: List of supported model identifiers
        """
        models_list = "\n  - ".join(supported_models)
        message = (
            f"Model '{model}' is not supported.\n\n"
            f"Supported models:\n  - {models_list}\n\n"
            f"To add a new model, update SUPPORTED_MODELS in ruvrics/config.py"
        )
        super().__init__(message)
        self.model = model
        self.supported_models = supported_models
