"""
Stability test executor for Ruvrics.

Orchestrates running N identical LLM requests and collecting results.
From spec Section 2 - Execution Strategy.
"""

import json
import re
import time
from datetime import datetime
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ruvrics.config import Config, ModelConfig, get_config, get_model_config
from ruvrics.core.adapters import LLMAdapter, create_adapter
from ruvrics.core.models import InputConfig, RunResult, ToolCall
from ruvrics.utils.errors import APIError, InsufficientDataError, RateLimitError


class StabilityExecutor:
    """
    Executes N identical LLM requests for stability testing.

    From spec Section 2:
    - Default N=20 (configurable 10-50)
    - Fixed parameters (temperature, etc.)
    - Parallel execution when possible
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        model: str,
        runs: int | None = None,
        config: Config | None = None,
    ):
        """
        Initialize executor.

        Args:
            model: Model identifier (e.g., "gpt-4-turbo")
            runs: Number of runs (default from config)
            config: Configuration (uses global if None)
        """
        self.config = config or get_config()
        self.model_config: ModelConfig = get_model_config(model)
        self.adapter: LLMAdapter = create_adapter(self.model_config, self.config)
        self.runs = runs or self.config.default_runs

        # Validate runs parameter
        if not 10 <= self.runs <= 50:
            raise ValueError(f"Runs must be between 10 and 50, got {self.runs}")

    def run(self, input_config: InputConfig) -> list[RunResult]:
        """
        Execute N identical runs and return results.

        Args:
            input_config: Input configuration (prompt, messages, tools)

        Returns:
            List of RunResult objects

        Raises:
            InsufficientDataError: If too many runs fail
        """
        # Convert input to messages format
        messages = input_config.to_messages()
        tools = input_config.tools if input_config.has_tools() else None

        # Execute runs with progress bar
        results: list[RunResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Running {self.runs} stability tests...",
                total=self.runs,
            )

            for run_id in range(1, self.runs + 1):
                try:
                    result = self._execute_single_run(
                        run_id=run_id,
                        messages=messages,
                        tools=tools,
                    )
                    results.append(result)

                except Exception as e:
                    # Log error but continue with other runs
                    error_result = self._create_error_result(run_id, str(e))
                    results.append(error_result)

                progress.update(task, advance=1)

        # Check minimum successful runs (from spec Section 8)
        successful = [r for r in results if r.error is None]
        if len(successful) < self.config.min_successful_runs:
            raise InsufficientDataError(
                successful=len(successful),
                total=self.runs,
                minimum=self.config.min_successful_runs,
            )

        return results

    def _execute_single_run(
        self,
        run_id: int,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
    ) -> RunResult:
        """
        Execute a single LLM call with retry logic.

        From spec Section 8:
        - MAX_RETRIES = 3
        - Exponential backoff with multiplier = 2
        - Handle rate limits gracefully

        Args:
            run_id: Run identifier (1 to N)
            messages: Messages list
            tools: Optional tools list

        Returns:
            RunResult with response data

        Raises:
            Exception: After all retries exhausted
        """
        for attempt in range(self.config.max_retries):
            try:
                # Call LLM adapter
                response = self.adapter.call(messages=messages, tools=tools)

                # Detect output structure
                structure = self._detect_structure(response["text"])

                # Convert tool calls to ToolCall objects
                tool_calls = [
                    ToolCall(
                        name=tc["name"],
                        call_sequence=tc["call_sequence"],
                    )
                    for tc in response["tool_calls"]
                ]

                # Create successful result
                return RunResult(
                    run_id=run_id,
                    timestamp=datetime.now(),
                    output_text=response["text"],
                    tool_calls=tool_calls,
                    output_length_tokens=response["tokens"],
                    output_length_chars=len(response["text"]),
                    output_structure=structure,
                    api_latency_ms=response["latency_ms"],
                    model_used=response["model"],
                    error=None,
                )

            except RateLimitError as e:
                # Handle rate limiting (from spec Section 8)
                if attempt == self.config.max_retries - 1:
                    raise  # Final attempt, give up

                wait_time = e.retry_after or (
                    self.config.retry_backoff_multiplier ** attempt
                )
                time.sleep(wait_time)

            except (APIError, Exception) as e:
                # Other errors: exponential backoff
                if attempt == self.config.max_retries - 1:
                    raise  # Final attempt, give up

                wait_time = self.config.retry_backoff_multiplier ** attempt
                time.sleep(wait_time)

        # Should never reach here, but just in case
        raise APIError("Failed after all retries")

    def _detect_structure(self, output: str) -> str:
        """
        Classify output structure type.

        From spec Section 3.3 - Structural Consistency.

        Returns:
            Structure type: "JSON:DICT:{keys}", "JSON:ARRAY", "MARKDOWN", "TEXT", "EMPTY"
        """
        if not output or not output.strip():
            return "EMPTY"

        # Try JSON parsing first
        try:
            obj = json.loads(output.strip())
            if isinstance(obj, dict):
                # Extract top-level keys, sorted
                keys = sorted(obj.keys())
                return f"JSON:DICT:{','.join(keys)}"
            elif isinstance(obj, list):
                return "JSON:ARRAY"
            else:
                return "JSON:PRIMITIVE"
        except (json.JSONDecodeError, ValueError):
            pass

        # Check for markdown patterns
        markdown_patterns = [
            r"^#{1,6}\s",  # Headers
            r"^\s*[\*\-\+]\s",  # Unordered lists
            r"^\s*\d+\.\s",  # Ordered lists
            r"```",  # Code blocks
            r"\[.+\]\(.+\)",  # Links
        ]

        if any(re.search(p, output, re.MULTILINE) for p in markdown_patterns):
            return "MARKDOWN"

        # Default to plain text
        return "TEXT"

    def _create_error_result(self, run_id: int, error_message: str) -> RunResult:
        """
        Create a RunResult for a failed run.

        Args:
            run_id: Run identifier
            error_message: Error description

        Returns:
            RunResult with error field populated
        """
        return RunResult(
            run_id=run_id,
            timestamp=datetime.now(),
            output_text="",
            tool_calls=[],
            output_length_tokens=0,
            output_length_chars=0,
            output_structure="ERROR",
            api_latency_ms=0,
            model_used=self.model_config.name,
            error=error_message,
        )
