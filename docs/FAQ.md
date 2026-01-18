# Ruvrics - Frequently Asked Questions

Common questions and troubleshooting for Ruvrics users.

---

## General

### What does Ruvrics measure?

Ruvrics measures **behavioral consistency**, not correctness. It runs the same prompt N times and checks if your LLM system behaves the same way every time. It won't tell you if answers are *right* — it tells you if they're *changing*.

### What's the difference between stability and correctness?

- **Stability**: Does the system give consistent responses to identical inputs?
- **Correctness**: Are the responses factually accurate?

Ruvrics focuses on stability. A system can be consistently wrong (stable but incorrect) or inconsistently right (unstable but sometimes correct). Ruvrics catches the instability.

### Why 20 runs by default?

20 runs provides a good balance between statistical reliability and cost. With 20 samples, you can detect patterns of instability without excessive API spend. You can adjust with `--runs 10` (minimum) to `--runs 50` (maximum).

---

## Installation

### PyTorch download is huge (~4GB). Is there a lighter option?

Yes. Pre-install CPU-only PyTorch before installing Ruvrics:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ruvrics
```

This reduces the download to ~200MB.

### First run is slow - what's happening?

On first run, Ruvrics downloads the sentence-transformers embedding model (~80MB). This is a one-time download and will be cached for future runs.

---

## Usage

### How do I test tool-enabled agents?

When your LLM uses tools, you must provide mock responses so Ruvrics can complete the workflow:

```bash
ruvrics stability \
  --input query.json \
  --tools tools.json \
  --tool-mocks mocks.json \
  --model gpt-4o-mini
```

Without mocks, Ruvrics can only measure tool *routing* consistency, not the final response consistency.

### What does "tool mocks required" error mean?

When you provide `--tools`, Ruvrics requires `--tool-mocks` because:
1. LLMs stop and wait for tool results before generating final answers
2. Ruvrics doesn't have access to your actual tool implementations
3. Mocks let Ruvrics test the complete workflow

Create a simple JSON file with mock responses:

```json
{
  "your_tool_name": {
    "result": "sample response"
  }
}
```

### How do I save and compare baselines?

```bash
# Save baseline when behavior is good
ruvrics stability --input query.json --model gpt-4o-mini --save-baseline prod-v1

# Later, compare after changes
ruvrics stability --input query.json --model gpt-4o-mini --compare prod-v1
```

Baselines are stored in `~/.ruvrics/baselines/`.

### How do I compare two different models?

```bash
ruvrics stability --input query.json --model gpt-4o-mini --compare-model gpt-4o
```

This runs stability tests on both models and shows the difference.

---

## Interpreting Results

### What do the verdicts mean?

| Verdict | Score | Meaning |
|---------|-------|---------|
| **SAFE** | 90%+ | Behavior is consistent, safe to deploy |
| **RISKY** | 70-89% | Some inconsistency, review before deploying |
| **DO NOT SHIP** | <70% | Significant instability, fix before deploying |

### What are the component scores?

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Semantic | 40% | Do responses mean the same thing? |
| Tool Usage | 25% | Are the same tools called consistently? |
| Structure | 20% | Is the output format consistent (JSON, markdown, etc.)? |
| Length | 15% | Is response length consistent? |

### My score is 100% - is that normal?

Yes, for simple queries with modern models (GPT-4o, Claude Sonnet 4), 100% stability is common. This is a good sign - your prompt is clear and deterministic. The value of Ruvrics is catching *regressions* when you make changes.

---

## Troubleshooting

### "API key not found" error

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Model not supported" error

Check supported models:
- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo
- **Anthropic**: claude-opus-4, claude-sonnet-4, claude-sonnet-3.5, claude-haiku-4

### "Insufficient successful runs" error

This means too many API calls failed (fewer than 15 out of 20 succeeded). Possible causes:
- Rate limiting - wait and retry
- Invalid API key - check credentials
- Network issues - check connectivity

Try increasing runs: `--runs 30`

### Progress bar stuck at 0%

The first API call may take longer due to cold start. Wait up to 60 seconds. If it times out, check your network and API key.

---

## More Questions?

[Open an issue](https://github.com/ruvrics-ai/ruvrics/issues) on GitHub.
