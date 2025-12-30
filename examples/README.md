# Ruvrics Examples

This directory contains example files for testing the Ruvrics stability analysis tool.

## Example Files

### 1. Simple Query (query_simple.json)
Basic query without tool usage. Tests semantic and structural stability of straightforward responses.

**Usage:**
```bash
ruvrics stability \
  --input examples/query_simple.json \
  --model gpt-4-turbo \
  --runs 20
```

### 2. Query with Tools (query.json)
Query that includes tool definitions. Tests tool-routing stability and whether the model consistently decides to use tools.

**Usage:**
```bash
ruvrics stability \
  --prompt examples/system_prompt.txt \
  --input examples/query.json \
  --model gpt-4-turbo \
  --runs 20 \
  --output results.json
```

### 3. Messages Format (query_messages.json)
Uses the OpenAI messages format with system and user messages. Tests stability with conversational context.

**Usage:**
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
        "parameters": {...}
      }
    }
  ]
}
```

## Expected API Keys

Make sure you have the appropriate API keys set in your environment:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Interpreting Results

- **SAFE (≥90%)**: System is stable and ready to ship
- **RISKY (70-89%)**: Review recommended fixes before shipping
- **DO_NOT_SHIP (<70%)**: Critical stability issues detected

See the generated report for:
- Component breakdown (semantic, tool, structural, length)
- Root cause analysis
- Actionable recommendations
- Next steps
