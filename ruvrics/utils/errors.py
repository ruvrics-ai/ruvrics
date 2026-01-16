"""
Custom exceptions for Ruvrics AI Stability Engine.

Provides clear, actionable error messages for users.
From spec Section 8 - Error Handling.

Design Philosophy:
- Every error should explain WHAT went wrong
- Every error should explain WHY it matters
- Every error should explain HOW to fix it
"""

from typing import Optional


class RuvricsError(Exception):
    """Base exception for all Ruvrics errors."""

    pass


class ConfigurationError(RuvricsError):
    """Raised when configuration is invalid or missing."""

    pass


class APIKeyMissingError(ConfigurationError):
    """
    Raised when required API key is not found.

    Provides detailed instructions on multiple ways to set the key.
    """

    def __init__(self, provider: str):
        """
        Initialize API key missing error.

        Args:
            provider: Either "openai" or "anthropic"
        """
        self.provider = provider

        if provider == "openai":
            env_var = "OPENAI_API_KEY"
            key_url = "https://platform.openai.com/api-keys"
            key_prefix = "sk-..."
        elif provider == "anthropic":
            env_var = "ANTHROPIC_API_KEY"
            key_url = "https://console.anthropic.com/settings/keys"
            key_prefix = "sk-ant-..."
        else:
            env_var = f"{provider.upper()}_API_KEY"
            key_url = "your provider's dashboard"
            key_prefix = "your-api-key"

        message = (
            f"{provider.title()} API key not found.\n\n"
            f"To use {provider.title()} models, you need to set {env_var}.\n\n"
            f"OPTION 1: Export in terminal (temporary, current session only)\n"
            f"  export {env_var}=\"{key_prefix}\"\n\n"
            f"OPTION 2: Add to shell profile (permanent)\n"
            f"  # Add this line to ~/.bashrc, ~/.zshrc, or ~/.profile:\n"
            f"  export {env_var}=\"{key_prefix}\"\n"
            f"  # Then reload: source ~/.bashrc\n\n"
            f"OPTION 3: Create a .env file (recommended for projects)\n"
            f"  # Create .env in your project directory:\n"
            f"  echo '{env_var}={key_prefix}' >> .env\n"
            f"  # Ruvrics automatically loads .env files\n\n"
            f"GET YOUR API KEY:\n"
            f"  {key_url}\n\n"
            f"SECURITY TIP:\n"
            f"  Never commit API keys to git. Add .env to .gitignore."
        )
        super().__init__(message)


class InvalidAPIKeyError(ConfigurationError):
    """
    Raised when API key is present but invalid/rejected by provider.
    """

    def __init__(self, provider: str, original_error: str = ""):
        """
        Initialize invalid API key error.

        Args:
            provider: Either "openai" or "anthropic"
            original_error: Original error message from API
        """
        self.provider = provider

        if provider == "openai":
            env_var = "OPENAI_API_KEY"
            key_url = "https://platform.openai.com/api-keys"
        elif provider == "anthropic":
            env_var = "ANTHROPIC_API_KEY"
            key_url = "https://console.anthropic.com/settings/keys"
        else:
            env_var = f"{provider.upper()}_API_KEY"
            key_url = "your provider's dashboard"

        message = (
            f"{provider.title()} API key is invalid or expired.\n\n"
            f"WHAT HAPPENED:\n"
            f"  The API rejected your key: {original_error or 'Authentication failed'}\n\n"
            f"HOW TO FIX:\n"
            f"  1. Check if your API key is correct (no extra spaces or characters)\n"
            f"  2. Verify the key hasn't expired or been revoked\n"
            f"  3. Ensure you have billing/credits set up with {provider.title()}\n"
            f"  4. Generate a new key if needed: {key_url}\n\n"
            f"CHECK YOUR CURRENT KEY:\n"
            f"  echo ${env_var}\n\n"
            f"COMMON ISSUES:\n"
            f"  - Key was copy-pasted with extra whitespace\n"
            f"  - Using a key from wrong account/organization\n"
            f"  - Account billing not set up or exceeded limits"
        )
        super().__init__(message)


class ModelNotAvailableError(ConfigurationError):
    """
    Raised when model exists but user doesn't have access.
    """

    def __init__(self, model: str, provider: str, original_error: str = ""):
        """
        Initialize model not available error.

        Args:
            model: Model identifier
            provider: Provider name
            original_error: Original error from API
        """
        self.model = model
        self.provider = provider

        message = (
            f"Model '{model}' is not available for your account.\n\n"
            f"WHAT HAPPENED:\n"
            f"  {original_error or 'Access denied to this model'}\n\n"
            f"POSSIBLE REASONS:\n"
            f"  - Model requires special access/waitlist approval\n"
            f"  - Your API tier doesn't include this model\n"
            f"  - Model is deprecated or renamed\n\n"
            f"HOW TO FIX:\n"
            f"  1. Check if you have access to this model in your {provider.title()} dashboard\n"
            f"  2. Try a different model (e.g., gpt-4o-mini, claude-sonnet-4)\n"
            f"  3. Upgrade your API plan if needed\n\n"
            f"AVAILABLE ALTERNATIVES:\n"
        )

        if provider == "openai":
            message += (
                f"  - gpt-4o-mini (fast, cheap, good for testing)\n"
                f"  - gpt-4o (balanced performance)\n"
                f"  - gpt-4-turbo (high capability)"
            )
        elif provider == "anthropic":
            message += (
                f"  - claude-haiku-4 (fast, cheap, good for testing)\n"
                f"  - claude-sonnet-4 (balanced performance)\n"
                f"  - claude-opus-4 (highest capability)"
            )

        super().__init__(message)


class APIError(RuvricsError):
    """Raised when LLM API calls fail."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limit is hit."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        """
        Initialize rate limit error.

        Args:
            provider: API provider name
            retry_after: Seconds to wait before retrying (if provided by API)
        """
        self.retry_after = retry_after
        self.provider = provider

        wait_msg = f" Wait {retry_after} seconds and try again." if retry_after else ""

        message = (
            f"{provider.title()} API rate limit exceeded.{wait_msg}\n\n"
            f"WHAT HAPPENED:\n"
            f"  You've made too many requests in a short period.\n\n"
            f"HOW TO FIX:\n"
            f"  1. Wait a few minutes and try again\n"
            f"  2. Reduce --runs to a smaller number (e.g., --runs 15)\n"
            f"  3. Upgrade your API plan for higher rate limits\n\n"
            f"FOR PRODUCTION USE:\n"
            f"  - OpenAI: Request higher rate limits at https://platform.openai.com/account/limits\n"
            f"  - Anthropic: Contact support for enterprise limits\n\n"
            f"TIP: Ruvrics uses exponential backoff automatically, but sustained\n"
            f"high-volume testing may require a higher-tier API plan."
        )
        super().__init__(message)


class TimeoutError(APIError):
    """Raised when API request times out."""

    def __init__(self, provider: str, timeout_seconds: float = 60.0):
        """
        Initialize timeout error.

        Args:
            provider: API provider name
            timeout_seconds: Timeout value that was exceeded
        """
        self.provider = provider
        self.timeout_seconds = timeout_seconds

        message = (
            f"{provider.title()} API request timed out after {timeout_seconds:.0f} seconds.\n\n"
            f"WHAT HAPPENED:\n"
            f"  The API took too long to respond.\n\n"
            f"POSSIBLE CAUSES:\n"
            f"  - API servers are overloaded\n"
            f"  - Network connectivity issues\n"
            f"  - Very long/complex prompt\n"
            f"  - Model cold start (first request of day)\n\n"
            f"HOW TO FIX:\n"
            f"  1. Try again - this is often temporary\n"
            f"  2. Check your internet connection\n"
            f"  3. Check API status: https://status.openai.com or https://status.anthropic.com\n"
            f"  4. Increase timeout: export API_TIMEOUT=120\n\n"
            f"TIP: Ruvrics will retry failed requests automatically with backoff."
        )
        super().__init__(message)


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

    def __init__(self, file_path: str, reason: str = ""):
        """
        Initialize invalid input error.

        Args:
            file_path: Path to the invalid input file
            reason: Specific reason the input is invalid
        """
        self.file_path = file_path

        message = (
            f"Invalid input file: {file_path}\n\n"
            f"WHAT HAPPENED:\n"
            f"  {reason or 'The input file format is not valid.'}\n\n"
            f"EXPECTED FORMAT:\n"
            f"  Your input JSON must contain at least one of:\n"
            f"  - \"user_input\": \"Your question here\"\n"
            f"  - \"messages\": [{'{'}\"role\": \"user\", \"content\": \"Your question\"{'}'}]\n\n"
            f"SIMPLE EXAMPLE (query.json):\n"
            f"  {{\n"
            f"    \"user_input\": \"What is the capital of France?\"\n"
            f"  }}\n\n"
            f"WITH SYSTEM PROMPT:\n"
            f"  {{\n"
            f"    \"system_prompt\": \"You are a helpful assistant.\",\n"
            f"    \"user_input\": \"What is the capital of France?\"\n"
            f"  }}\n\n"
            f"MESSAGES FORMAT (for complex conversations):\n"
            f"  {{\n"
            f"    \"messages\": [\n"
            f"      {{\"role\": \"system\", \"content\": \"You are helpful.\"}},\n"
            f"      {{\"role\": \"user\", \"content\": \"Hello!\"}}\n"
            f"    ]\n"
            f"  }}"
        )
        super().__init__(message)


class JSONParseError(RuvricsError):
    """Raised when JSON file cannot be parsed."""

    def __init__(self, file_path: str, error_detail: str, line: int = None, column: int = None):
        """
        Initialize JSON parse error.

        Args:
            file_path: Path to the file that failed to parse
            error_detail: Specific parse error message
            line: Line number where error occurred (if available)
            column: Column number where error occurred (if available)
        """
        self.file_path = file_path
        self.line = line
        self.column = column

        location = ""
        if line is not None:
            location = f" at line {line}"
            if column is not None:
                location += f", column {column}"

        message = (
            f"Failed to parse JSON file: {file_path}{location}\n\n"
            f"PARSE ERROR:\n"
            f"  {error_detail}\n\n"
            f"COMMON JSON MISTAKES:\n"
            f"  - Missing comma between items\n"
            f"  - Trailing comma after last item (not allowed in JSON)\n"
            f"  - Using single quotes instead of double quotes\n"
            f"  - Unescaped special characters in strings\n"
            f"  - Missing closing brackets/braces\n\n"
            f"HOW TO FIX:\n"
            f"  1. Validate your JSON at https://jsonlint.com\n"
            f"  2. Use a code editor with JSON syntax highlighting\n"
            f"  3. Check the line number mentioned above\n\n"
            f"EXAMPLE VALID JSON:\n"
            f"  {{\n"
            f"    \"user_input\": \"What is Python?\",\n"
            f"    \"system_prompt\": \"Be concise.\"\n"
            f"  }}"
        )
        super().__init__(message)


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


class ToolMockRequiredError(RuvricsError):
    """
    Raised when tools are provided but tool mocks are missing.

    Tool mocks are required for complete stability testing of tool-enabled AI systems.
    """

    def __init__(self, tool_names: list[str]):
        """
        Initialize tool mock required error.

        Args:
            tool_names: List of tool names that need mocks
        """
        tools_list = ", ".join(tool_names)
        message = (
            f"Tool mocks required for complete stability testing.\n\n"
            f"Your query uses tools: {tools_list}\n\n"
            f"WHY MOCKS ARE NEEDED:\n"
            f"When an LLM calls a tool, it waits for the tool's response before generating\n"
            f"the final answer. Ruvrics doesn't have access to your actual tool implementations\n"
            f"(APIs, databases, etc.), so it needs mock responses to complete the conversation.\n\n"
            f"Without mocks, Ruvrics can only test 'tool routing consistency' (did the LLM\n"
            f"call the right tools?), but cannot test 'response consistency' (did the LLM\n"
            f"give consistent final answers?).\n\n"
            f"HOW TO PROVIDE MOCKS:\n"
            f"Create a JSON file with mock responses for each tool:\n\n"
            f"  {{\n"
            f'    "{tool_names[0] if tool_names else "your_tool"}": {{\n'
            f'      "result": "sample data",\n'
            f'      "status": "success"\n'
            f"    }}\n"
            f"  }}\n\n"
            f"Then run:\n"
            f"  ruvrics stability --input query.json --tools tools.json --tool-mocks mocks.json\n\n"
            f"COST-EFFECTIVE:\n"
            f"Mock responses are reused for all N runs, so you only define them once.\n"
            f"This ensures consistent tool results across runs, isolating the LLM's behavior\n"
            f"for accurate stability measurement.\n\n"
            f"NO TOOLS? If your query doesn't need tools, remove --tools from the command."
        )
        super().__init__(message)
        self.tool_names = tool_names
