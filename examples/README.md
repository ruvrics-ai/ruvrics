# Ruvrics Examples

Example files for testing LLM behavioral stability.

## Quick Start Examples

### 1. Simple Query (No Tools)

```bash
ruvrics stability \
  --input examples/query_simple.json \
  --model gpt-4o-mini \
  --runs 20
```

### 2. With Tools (Agentic Testing)

```bash
ruvrics stability \
  --input examples/query_with_tools.json \
  --tools examples/tools.json \
  --tool-mocks examples/tool_mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

### 3. With System Prompt

```bash
ruvrics stability \
  --prompt examples/system_prompt.txt \
  --input examples/query_with_tools.json \
  --tools examples/tools.json \
  --tool-mocks examples/tool_mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

## Files in This Directory

| File | Purpose |
|------|---------|
| `query_simple.json` | Basic query without tools |
| `query_with_tools.json` | Query that triggers tool usage |
| `query_messages.json` | OpenAI messages format |
| `query.json` | Legacy format with embedded tools |
| `tools.json` | Tool definitions (OpenAI format) |
| `tool_mocks.json` | Mock responses for tool execution |
| `system_prompt.txt` | Example system prompt |

## Input Formats

### Simple Format
```json
{"user_input": "What is Python?"}
```

### With System Prompt
```json
{
  "system_prompt": "You are a helpful assistant.",
  "user_input": "What is Python?"
}
```

### Messages Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
  ]
}
```

## Tool Mock Format

When testing tool-enabled agents, provide mock responses:

```json
{
  "search_flights": {
    "flights": [
      {"flight_number": "UA875", "price": 850}
    ]
  }
}
```

**Why mocks?** LLMs stop and wait for tool results. Mocks let Ruvrics test the complete workflow: prompt → tool call → response → final answer.

## Advanced Scenarios

See `scenarios/` for 6 advanced test cases covering edge cases like multi-tool ambiguity, argument drift, and tool failure handling.
