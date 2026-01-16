"""
LLM API adapters for Ruvrics.

Provides unified interface for calling OpenAI and Anthropic APIs.
Handles response standardization and error mapping.
From spec Section 2 and Configuration Guide.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import anthropic
import openai

from ruvrics.config import Config, ModelConfig
from ruvrics.core.models import RunResult, ToolCall
from ruvrics.utils.errors import APIError, RateLimitError, TimeoutError


class LLMAdapter(ABC):
    """Abstract base class for LLM API adapters."""

    def __init__(self, model_config: ModelConfig, config: Config):
        """
        Initialize adapter.

        Args:
            model_config: Configuration for the specific model
            config: Global configuration with API keys and settings
        """
        self.model_config = model_config
        self.config = config

    @abstractmethod
    def call(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Call the LLM API and return standardized response.

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions

        Returns:
            Standardized response dict with:
                - text: str (final output text)
                - tool_calls: list[dict] (tool calls made)
                - tokens: int (token count)
                - latency_ms: int (response time)
                - model: str (actual model used)

        Raises:
            APIError: On API failures
            RateLimitError: On rate limit
            TimeoutError: On timeout
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API (GPT models)."""

    def __init__(self, model_config: ModelConfig, config: Config):
        """Initialize OpenAI adapter with API key."""
        super().__init__(model_config, config)
        api_key = config.get_api_key("openai")
        self.client = openai.OpenAI(
            api_key=api_key,
            timeout=config.api_timeout_seconds,
        )

    def call(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Call OpenAI API.

        Uses temperature=0 for deterministic output (as much as possible).
        """
        try:
            start_time = time.time()

            # Build request parameters
            params: dict[str, Any] = {
                "model": self.model_config.name,
                "messages": messages,
                "temperature": self.model_config.temperature,
                "max_tokens": self.model_config.max_tokens,
            }

            # Add tools if provided
            if tools and len(tools) > 0:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            # Make API call
            response = self.client.chat.completions.create(**params)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Standardize response
            return self._standardize_response(response, latency_ms)

        except openai.RateLimitError as e:
            # Extract retry_after if available
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    retry_after = int(retry_after)
            raise RateLimitError(str(e), retry_after=retry_after)

        except openai.APITimeoutError as e:
            raise TimeoutError(f"OpenAI API timeout: {e}")

        except (openai.APIError, openai.APIConnectionError) as e:
            raise APIError(f"OpenAI API error: {e}")

    def _standardize_response(
        self, response: Any, latency_ms: int
    ) -> dict[str, Any]:
        """
        Convert OpenAI response to standardized format.

        Args:
            response: OpenAI ChatCompletion object
            latency_ms: Request latency in milliseconds

        Returns:
            Standardized response dict
        """
        choice = response.choices[0]
        message = choice.message

        # Extract text content
        text = message.content or ""

        # Extract tool calls (Format C from spec)
        tool_calls_list = []
        if message.tool_calls:
            for idx, tool_call in enumerate(message.tool_calls, start=1):
                # Parse arguments from JSON string
                arguments = None
                if tool_call.function.arguments:
                    try:
                        import json
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        # If arguments can't be parsed, store as None
                        arguments = None

                tool_calls_list.append(
                    {
                        "name": tool_call.function.name,
                        "call_sequence": idx,
                        "arguments": arguments,
                        "tool_call_id": tool_call.id,  # Capture OpenAI's unique ID
                    }
                )

        # Get token usage
        tokens = response.usage.total_tokens if response.usage else 0

        return {
            "text": text,
            "tool_calls": tool_calls_list,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "model": response.model,
        }


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic API (Claude models)."""

    def __init__(self, model_config: ModelConfig, config: Config):
        """Initialize Anthropic adapter with API key."""
        super().__init__(model_config, config)
        api_key = config.get_api_key("anthropic")
        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=config.api_timeout_seconds,
        )

    def call(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Call Anthropic API.

        Anthropic has different message format - need to separate system message.
        """
        try:
            start_time = time.time()

            # Separate system message (Anthropic requires it as separate param)
            system_message = None
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            # Build request parameters
            params: dict[str, Any] = {
                "model": self.model_config.name,
                "messages": user_messages,
                "temperature": self.model_config.temperature,
                "max_tokens": self.model_config.max_tokens,
            }

            if system_message:
                params["system"] = system_message

            # Add tools if provided
            if tools and len(tools) > 0:
                params["tools"] = self._convert_tools_to_anthropic(tools)

            # Make API call
            response = self.client.messages.create(**params)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Standardize response
            return self._standardize_response(response, latency_ms)

        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e))

        except anthropic.APITimeoutError as e:
            raise TimeoutError(f"Anthropic API timeout: {e}")

        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            raise APIError(f"Anthropic API error: {e}")

    def _convert_tools_to_anthropic(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert OpenAI tool format to Anthropic format.

        OpenAI uses nested structure, Anthropic is flatter.
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        return anthropic_tools

    def _standardize_response(
        self, response: Any, latency_ms: int
    ) -> dict[str, Any]:
        """
        Convert Anthropic response to standardized format.

        Args:
            response: Anthropic Message object
            latency_ms: Request latency in milliseconds

        Returns:
            Standardized response dict
        """
        # Extract text content from content blocks
        text_parts = []
        tool_calls_list = []
        tool_call_idx = 1

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Anthropic provides arguments as a dict in block.input
                tool_calls_list.append(
                    {
                        "name": block.name,
                        "call_sequence": tool_call_idx,
                        "arguments": block.input if hasattr(block, 'input') else None,
                        "tool_call_id": block.id,  # Capture Anthropic's unique ID
                    }
                )
                tool_call_idx += 1

        text = " ".join(text_parts)

        # Get token usage
        tokens = (
            response.usage.input_tokens + response.usage.output_tokens
            if response.usage
            else 0
        )

        return {
            "text": text,
            "tool_calls": tool_calls_list,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "model": response.model,
        }


def create_adapter(model_config: ModelConfig, config: Config) -> LLMAdapter:
    """
    Factory function to create appropriate adapter.

    Args:
        model_config: Model configuration
        config: Global configuration

    Returns:
        LLMAdapter instance (OpenAI or Anthropic)

    Raises:
        ValueError: If provider is not supported
    """
    if model_config.provider == "openai":
        return OpenAIAdapter(model_config, config)
    elif model_config.provider == "anthropic":
        return AnthropicAdapter(model_config, config)
    else:
        raise ValueError(
            f"Unsupported provider: {model_config.provider}. "
            "Must be 'openai' or 'anthropic'."
        )
