# Ruvrics

**Catch behavioral regressions in your LLM systems before users do.**

Ruvrics detects when your AI system's behavior silently changes — whether from prompt edits, tool updates, model upgrades, or configuration drift.

## The Problem

Modern LLMs are surprisingly consistent in isolation. But LLM **systems** drift over time:

- Someone tweaks the prompt "for clarity"
- A tool schema gets renamed
- You upgrade from gpt-4o-mini to gpt-4.1
- RAG chunk ordering changes
- Retry logic gets added

No crash. No exception. No obvious error.
Just **different decisions** — and users notice before you do.

## The Solution

Ruvrics runs your prompt N times and measures whether your AI system behaves the same way every time. More importantly, it lets you **save baselines** and **detect drift** over time.

> **Note:** Ruvrics measures consistency, not correctness. It won't tell you if answers are *right* — it tells you if they're *changing*.

```bash
# Day 1: Establish baseline
ruvrics stability --input query.json --model gpt-4o-mini --save-baseline v1.0

# Day 14: After "minor" prompt changes
ruvrics stability --input query.json --model gpt-4o-mini --compare v1.0

# Output: "⚠️ REGRESSION: Stability dropped 98% → 84%"
```

## Quick Start

**1. Install**
```bash
pip install ruvrics
```

**2. Set API key**
```bash
export OPENAI_API_KEY="sk-..."
```

**3. Run your first test**
```bash
echo '{"user_input": "What is Python?"}' > query.json
ruvrics stability --input query.json --model gpt-4o-mini --runs 20
# Runs the same input multiple times to measure stability
```

**4. Save as baseline**
```bash
ruvrics stability --input query.json --model gpt-4o-mini --save-baseline v1.0
```

**5. Later: Check for drift**
```bash
ruvrics stability --input query.json --model gpt-4o-mini --compare v1.0
```

## What Gets Measured

| Check | Weight | What It Detects |
|-------|--------|-----------------|
| **Response Consistency** | 40% | Are answers saying the same thing? |
| **Tool Consistency** | 25% | Are the same tools being called? |
| **Format Consistency** | 20% | Is output format stable (JSON, text, etc.)? |
| **Length Consistency** | 15% | Is response length stable? |

**Verdicts:**
- **SAFE** (90%+): Behavior is consistent
- **RISKY** (70-89%): Review recommended
- **DO NOT SHIP** (<70%): Significant instability

## Key Features

### Baseline Comparison (Drift Detection)

Save a baseline when behavior is good, compare later to catch regressions:

```bash
# Save baseline after validation
ruvrics stability --input query.json --model gpt-4o-mini --save-baseline prod-v1

# After changes, compare to baseline
ruvrics stability --input query.json --model gpt-4o-mini --compare prod-v1
```

### Model Comparison

Compare behavior across model versions or providers:

```bash
# Compare two models on the same prompt
ruvrics stability --input query.json --model gpt-4o-mini --compare-model gpt-4o
```

### Tool-Enabled Agent Testing

Test complete agentic workflows with mock tool responses:

```bash
ruvrics stability \
  --input query.json \
  --tools tools.json \
  --tool-mocks mocks.json \
  --model gpt-4o-mini
```

## Example Output

```
======================================================================
                         STABILITY REPORT
======================================================================
Tested: gpt-4o-mini | Runs: 20/20 | Duration: 45.2s

Overall Stability Score: 94.7%  ✅ SAFE

======================================================================
CONSISTENCY BREAKDOWN
======================================================================

Response Consistency:       96.2%  ✅ Excellent
Tool Consistency:          100.0%  ✅ Excellent
Format Consistency:        100.0%  ✅ Excellent
Length Consistency:         82.5%  ⚠️ Good

======================================================================
COMPARISON TO BASELINE (prod-v1)
======================================================================

Baseline Score: 98.2% → Current: 94.7%  (-3.5%)
Status: ⚠️ MINOR REGRESSION

Changes Detected:
- Responses slightly more varied (99.1% → 96.2%)
- Verbosity increased (91.0% → 82.5%)

======================================================================
```

## When to Use Ruvrics

| Scenario | Command |
|----------|---------|
| Before deploying new prompt | `ruvrics stability --save-baseline prod` |
| After prompt changes | `ruvrics stability --compare prod` |
| Upgrading models | `ruvrics stability --compare-model gpt-4.1` |
| CI/CD gate | `ruvrics stability --compare prod --fail-on-regression` |

## CLI Reference

```bash
ruvrics stability \
  --input query.json \           # Required: user query
  --model gpt-4o-mini \          # Required: model to test
  --runs 20 \                    # Optional: runs (10-50, default: 20)
  --temperature 0.0 \            # Optional: LLM temperature (default: 0.0)
  --prompt system.txt \          # Optional: system prompt
  --tools tools.json \           # Optional: tool definitions
  --tool-mocks mocks.json \      # Required if using tools
  --save-baseline <name> \       # Save results as named baseline
  --compare <name> \             # Compare to saved baseline
  --compare-model <model> \      # Compare to different model
  --output results.json          # Custom output path
```

### Temperature Setting

By default, ruvrics uses `temperature=0.0` for maximum determinism. Use higher values to test real-world variance:

```bash
# Test with production-like temperature
ruvrics stability --input query.json --model gpt-4o-mini --temperature 0.5
```

- `0.0` - Maximum determinism (default, recommended for baseline)
- `0.3-0.7` - Typical production settings, shows natural variance
- `1.0` - Maximum variance

## Input File Examples

See [`examples/`](examples/) for ready-to-run query files, tool definitions, and mock responses.

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| Anthropic | claude-opus-4.5, claude-sonnet-4.5, claude-haiku-4.5, claude-opus-4, claude-sonnet-4, claude-sonnet-3.7, claude-haiku-3 |

## Installation Options

```bash
# Standard (CPU + GPU support)
pip install ruvrics

# Lightweight (CPU-only, ~200MB vs ~4GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ruvrics

# From source
git clone https://github.com/ruvrics-ai/ruvrics.git && cd ruvrics && pip install -e .
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | SAFE or no regression |
| 1 | RISKY or minor regression |
| 2 | DO NOT SHIP or major regression |

## Advanced Scenarios

See `examples/scenarios/` for edge case tests:
- Multi-tool ambiguity
- Argument drift
- Tool failure handling
- Multi-step chains

## Documentation

- [FAQ](docs/FAQ.md) - Frequently asked questions and troubleshooting

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

1. Fork → 2. Branch → 3. Test (`pytest tests/ -v`) → 4. PR

## Support

[Open an issue](https://github.com/ruvrics-ai/ruvrics/issues) on GitHub.
