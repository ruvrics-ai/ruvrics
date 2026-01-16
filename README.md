# Ruvrics

**AI Behavioral Stability & Consistency Analyzer (Alpha)**

Ruvrics tells you whether your LLM behaves consistently under real-world uncertainty.

## The Problem

LLMs are unpredictable. The same prompt can produce different outputs, call different tools, or format responses differently each time. This makes AI systems unreliable in production.

**You need to know:** Will your AI behave the same way for every user?

## The Solution

Ruvrics runs your prompt N times (default: 20) and measures consistency across 4 dimensions:

| Metric | What It Measures |
|--------|------------------|
| **Semantic** (40%) | Do outputs mean the same thing? |
| **Tool Usage** (25%) | Does the AI use tools consistently? |
| **Structure** (20%) | Is the output format stable? |
| **Length** (15%) | Are responses similarly verbose? |

You get a **stability score** (0-100%) and a **verdict**:
- **SAFE** (90%+): Ship it
- **RISKY** (70-89%): Review the recommendations
- **DO NOT SHIP** (<70%): Fix the issues first

## Quick Start

**1. Install**
```bash
pip install ruvrics
```

**2. Set API key**
```bash
export OPENAI_API_KEY="sk-..."
```

**3. Run**
```bash
echo '{"user_input": "What is Python?"}' > query.json
ruvrics stability --input query.json --model gpt-4o-mini --runs 20
```

## Example Output

```
======================================================================
                         AI STABILITY REPORT
======================================================================
Tested: gpt-4o-mini | Runs: 20/20 | Duration: 45.2s

Overall Stability Score: 94.7%  SAFE

======================================================================
COMPONENT BREAKDOWN
======================================================================

Semantic Consistency:       96.2% | LOW variance
Tool-Call Consistency:     100.0% | LOW variance
Structural Consistency:    100.0% | LOW variance
Length Consistency:         82.5% | MEDIUM variance

======================================================================
NEXT STEPS
======================================================================

 This system is safe to ship!

======================================================================
```

Reports are saved to `reports/` with full details and recommendations.

## Testing Tool-Enabled Agents

If your AI uses tools, you need to provide mock responses so Ruvrics can test the complete workflow:

**query.json**
```json
{"user_input": "Find flights from NYC to London"}
```

**tools.json**
```json
[{
  "type": "function",
  "function": {
    "name": "search_flights",
    "description": "Search for flights",
    "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}}}
  }
}]
```

**mocks.json** (what your tool would return)
```json
{
  "search_flights": {"flights": [{"id": "UA123", "price": 450}, {"id": "BA456", "price": 520}]}
}
```

**Run:**
```bash
ruvrics stability \
  --input query.json \
  --tools tools.json \
  --tool-mocks mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

This tests the full loop: prompt → tool call → tool response → final answer.

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic | claude-opus-4, claude-sonnet-4, claude-sonnet-3.5, claude-haiku-4 |

## Installation Options

**Standard** (works on CPU and GPU):
```bash
pip install ruvrics
```

**Lightweight** (CPU-only, ~200MB instead of ~4GB):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ruvrics
```

**From source:**
```bash
git clone https://github.com/ruvrics-ai/ruvrics.git
cd ruvrics
pip install -e .
```

## Configuration

Set API keys via environment variables or `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## CLI Reference

```bash
ruvrics stability \
  --input query.json \        # Required: user query
  --model gpt-4o-mini \       # Required: model to test
  --runs 20 \                 # Optional: number of runs (10-50, default: 20)
  --prompt system.txt \       # Optional: system prompt file
  --tools tools.json \        # Optional: tool definitions
  --tool-mocks mocks.json \   # Required if using tools
  --output results.json       # Optional: custom output path
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | SAFE - stable, ready to ship |
| 1 | RISKY - review recommendations |
| 2 | DO NOT SHIP - critical issues |

## How It Works

1. **Execute**: Run the same prompt N times with temperature=0
2. **Measure**: Calculate semantic similarity (embeddings), tool patterns, structure, length
3. **Score**: Weighted average of all metrics
4. **Diagnose**: Identify root causes of any instability
5. **Recommend**: Provide actionable fixes

## Advanced Testing Scenarios

See `examples/scenarios/` for 6 advanced test cases:
1. Multi-tool ambiguity (tool selection varies)
2. Optional tool usage (unnecessary tool calls)
3. Argument drift (same tool, different arguments)
4. Tool→answer mismatch (same data, different conclusions)
5. Tool failure handling (error responses)
6. Multi-step chains (sequence stability)

## Documentation

- `docs/AI_STABILITY_FINAL_SPEC.md` - Technical specification
- `docs/CONFIGURATION_GUIDE.md` - Configuration reference
- `docs/FAQ.md` - Frequently asked questions

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black ruvrics/ tests/

# Type check
mypy ruvrics/
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `pytest tests/ -v`
5. Submit a pull request

## Support

Open an issue on [GitHub](https://github.com/ruvrics-ai/ruvrics/issues).
