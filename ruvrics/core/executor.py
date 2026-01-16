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
from ruvrics.utils.errors import APIError, InsufficientDataError, RateLimitError, ToolMockRequiredError


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
            ToolMockRequiredError: If tools provided but no mocks
        """
        # Convert input to messages format
        messages = input_config.to_messages()
        tools = input_config.tools if input_config.has_tools() else None
        tool_mocks = input_config.tool_mock_responses

        # Validate: if tools are provided, tool mocks must also be provided
        if tools and not tool_mocks:
            tool_names = [t["function"]["name"] for t in tools if "function" in t]
            raise ToolMockRequiredError(tool_names)

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
                        tool_mocks=tool_mocks,
                    )
                    results.append(result)

                # except Exception as e:
                #     # Log error but continue with other runs
                #     error_result = self._create_error_result(run_id, str(e))
                #     results.append(error_result)
                except Exception as e:
                    print(f"\n[DEBUG] Run {run_id} failed with error:")
                    print(type(e).__name__, str(e))

                    error_result = self._create_error_result(
                        run_id,
                        f"{type(e).__name__}: {str(e)}"
                    )
                    results.append(error_result)

                progress.update(task, advance=1)

        # Check minimum successful runs (from spec Section 8)
        successful = [r for r in results if r.error is None]
        if len(successful) < self.config.min_successful_runs:
            failed = self.runs - len(successful)
            print(
                f"\n[red]Execution completed, but only "
                f"{len(successful)}/{self.runs} runs succeeded "
                f"({failed} failures).[/red]"
            )
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
        tool_mocks: dict[str, Any] | None = None,
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
            tool_mocks: Optional mock responses for tool execution

        Returns:
            RunResult with response data

        Raises:
            Exception: After all retries exhausted
        """
        for attempt in range(self.config.max_retries):
            try:
                # Call LLM adapter (initial call)
                response = self.adapter.call(messages=messages, tools=tools)

                # Convert tool calls to ToolCall objects
                all_tool_calls = []
                call_sequence_offset = 0

                initial_tool_calls = [
                    ToolCall(
                        name=tc["name"],
                        call_sequence=tc["call_sequence"],
                        arguments=tc.get("arguments"),  # Include arguments from adapter
                        tool_call_id=tc.get("tool_call_id"),  # API-provided unique ID
                    )
                    for tc in response.get("tool_calls", []) or []
                ]
                all_tool_calls.extend(initial_tool_calls)

                # Store initial output and tool calls
                output_text = response["text"]
                total_tokens = response["tokens"]
                total_latency = response["latency_ms"]

                # Multi-turn agentic loop (v0.2.2 - supports Scenario 6)
                # Loop until no more tool calls or max iterations reached
                current_messages = messages
                current_response = response
                iteration = 0

                while initial_tool_calls and tool_mocks and iteration < self.config.max_tool_iterations:
                    iteration += 1
                    call_sequence_offset = len(all_tool_calls)

                    # Build messages with tool results
                    updated_messages = self._build_messages_with_tool_results(
                        messages=current_messages,
                        response=current_response,
                        tool_calls=initial_tool_calls,
                        tool_mocks=tool_mocks,
                    )

                    # Make next API call
                    next_response = self.adapter.call(
                        messages=updated_messages,
                        tools=tools,  # Keep tools available for chaining
                    )

                    # Accumulate tokens and latency
                    total_tokens += next_response["tokens"]
                    total_latency += next_response["latency_ms"]

                    # Check for new tool calls
                    new_tool_calls = [
                        ToolCall(
                            name=tc["name"],
                            call_sequence=call_sequence_offset + tc["call_sequence"],
                            arguments=tc.get("arguments"),
                            tool_call_id=tc.get("tool_call_id"),  # API-provided unique ID
                        )
                        for tc in next_response.get("tool_calls", []) or []
                    ]

                    if new_tool_calls:
                        # More tools called - continue loop
                        all_tool_calls.extend(new_tool_calls)
                        current_messages = updated_messages
                        current_response = next_response
                        initial_tool_calls = new_tool_calls
                    else:
                        # No more tool calls - we have final response
                        output_text = next_response["text"]
                        break
                else:
                    # Loop ended (either no initial tools or max iterations)
                    if iteration >= self.config.max_tool_iterations and initial_tool_calls:
                        # Hit max iterations with pending tool calls
                        output_text = current_response.get("text", "") or "(Max tool iterations reached)"

                # Use all collected tool calls
                tool_calls = all_tool_calls

                # Detect output structure
                structure = self._detect_structure(output_text)

                # Create successful result
                return RunResult(
                    run_id=run_id,
                    timestamp=datetime.now(),
                    output_text=output_text,
                    tool_calls=tool_calls,
                    output_length_tokens=total_tokens,
                    output_length_chars=len(output_text),
                    output_structure=structure,
                    api_latency_ms=total_latency,
                    model_used=response["model"],
                    error=None,
                    tool_iterations=max(1, iteration),  # Track multi-turn iterations
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

    def _build_messages_with_tool_results(
        self,
        messages: list[dict[str, str]],
        response: dict[str, Any],
        tool_calls: list[ToolCall],
        tool_mocks: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Build message history with tool results for second API call.

        Handles provider-specific formats for OpenAI vs Anthropic.

        Args:
            messages: Original messages
            response: Initial API response with tool calls
            tool_calls: List of ToolCall objects
            tool_mocks: Mock responses mapping tool names to outputs

        Returns:
            Updated messages list with tool results
        """
        # Start with original messages
        updated_messages = messages.copy()

        # Provider-specific formatting
        if self.model_config.provider == "openai":
            # OpenAI format: assistant message + tool messages
            # Build assistant message with tool_calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response["text"] or "",
            }

            # Add tool_calls in OpenAI format if present
            if response.get("tool_calls"):
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["tool_call_id"],  # Use actual ID from API response
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in response["tool_calls"]
                ]

            updated_messages.append(assistant_msg)

            # Add tool result messages - use actual tool_call_id
            for tc in tool_calls:
                tool_output = tool_mocks.get(tc.name, {"error": "Tool not mocked"})
                updated_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.tool_call_id,  # Use actual ID from ToolCall
                        "content": json.dumps(tool_output),
                    }
                )

        elif self.model_config.provider == "anthropic":
            # Anthropic format: assistant message with content blocks + user message with tool_result blocks
            # Assistant message with tool_use blocks
            assistant_content = []

            # Add text content if present
            if response["text"]:
                assistant_content.append({"type": "text", "text": response["text"]})

            # Add tool_use blocks - use actual ID from API
            for tc in tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.tool_call_id,  # Use actual ID from ToolCall
                        "name": tc.name,
                        "input": tc.arguments or {},
                    }
                )

            updated_messages.append({"role": "assistant", "content": assistant_content})

            # User message with tool_result blocks - use actual ID from API
            tool_results_content = []
            for tc in tool_calls:
                tool_output = tool_mocks.get(tc.name, {"error": "Tool not mocked"})
                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.tool_call_id,  # Use actual ID from ToolCall
                        "content": json.dumps(tool_output),
                    }
                )

            updated_messages.append({"role": "user", "content": tool_results_content})

        return updated_messages

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
