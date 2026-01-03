# Ruvrics Examples

This directory contains example files for testing the Ruvrics stability analysis tool.

## Example Files

### 1. Simple Query (query_simple.json)

Basic query without tool usage. Tests semantic and structural stability of straightforward responses.

Usage:
```bash
ruvrics stability \
  --input examples/query_simple.json \
  --model gpt-4-turbo \
  --runs 20
```

### 2. Query with Tools (Modular Approach - Recommended)

Separate query and tool definitions for better reusability. Tests tool-routing stability.

Files:
- query_with_tools.json - Just the user query
- tools.json - Reusable tool definitions
- system_prompt.txt - System instructions

Usage:
```bash
ruvrics stability \
  --prompt examples/system_prompt.txt \
  --input examples/query_with_tools.json \
  --tools examples/tools.json \
  --model gpt-4-turbo \
  --runs 20
```

### 3. Query with Tools (All-in-One Format)

Legacy format with tools embedded in the query file. Still supported but less modular.

Usage:
```bash
ruvrics stability \
  --prompt examples/system_prompt.txt \
  --input examples/query.json \
  --model gpt-4-turbo \
  --runs 20 \
  --output results.json
```

### 4. Messages Format (query_messages.json)

Uses the OpenAI messages format with system and user messages. Tests stability with conversational context.

Usage:
```bash
ruvrics stability \
  --input examples/query_messages.json \
  --model gpt-4o \
  --runs 15
```

## Input Format Types

Ruvrics supports three input formats:

### Format A: Simple (system_prompt + user_input)

```json
{
  "system_prompt": "You are a helpful assistant.",
  "user_input": "What is Python?"
}
```

### Format B: Messages

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
  ]
}
```

### Format C: Tool-enabled

```json
{
  "user_input": "Find flights to Tokyo",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_flights",
        "description": "Search for flights",
        "parameters": {
          "type": "object",
          "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"}
          }
        }
      }
    }
  ]
}
```

You can also combine system prompt from a file with tools in JSON:

```bash
ruvrics stability \
  --prompt system_prompt.txt \
  --input query_with_tools.json \
  --tools tools.json \
  --model gpt-4o \
  --runs 20
```

## Expected API Keys

Make sure you have the appropriate API keys set in your environment:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a .env file in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Interpreting Results

- SAFE (score >= 90%): System is stable and ready to ship
- RISKY (70-89%): Review recommended fixes before shipping
- DO_NOT_SHIP (< 70%): Critical stability issues detected

See the generated report for:
- Component breakdown (semantic, tool, structural, length)
- Root cause analysis
- Actionable recommendations
- Next steps
