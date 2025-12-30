# Ruvrics - AI Behavioral Stability & Reliability Engine

**Version:** 0.1.0 (MVP)
**Status:** 🚧 In Development

> Measure whether an LLM-based system behaves consistently under identical conditions, explain *why* it is unstable, and suggest concrete fixes **before deployment**.

---

## Overview

Ruvrics helps developers answer the critical question: **"Is my LLM system stable enough to ship?"**

By running identical requests multiple times, Ruvrics measures:
- **Semantic Consistency** - Do outputs mean the same thing?
- **Tool Usage Consistency** - Does the model reliably call the right tools?
- **Structural Consistency** - Is the output format stable?
- **Length Consistency** - Does response length vary significantly?

## Features

✅ **Multi-Provider Support** - Works with OpenAI and Anthropic APIs
✅ **Comprehensive Metrics** - 4 core stability measurements
✅ **Root Cause Analysis** - Identifies WHY your system is unstable
✅ **Actionable Recommendations** - Get concrete fixes, not just scores
✅ **Beautiful CLI** - Rich terminal output with progress indicators
✅ **Pattern-Based Safety Detection** - Flags risky claims and guarantees

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ruvrics

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

1. **Set up your API keys**

```bash
cp .env.example .env
# Edit .env and add your OpenAI or Anthropic API key
```

2. **Run a stability test**

```bash
ruvrics stability \
  --prompt examples/system_prompt.txt \
  --input examples/query.json \
  --model gpt-4-turbo \
  --runs 20
```

## Requirements

- Python 3.10+
- OpenAI API key (for OpenAI models)
- Anthropic API key (for Anthropic models)

## Project Structure

```
ruvrics/
├── ruvrics/           # Main package
│   ├── core/          # Execution engine
│   ├── metrics/       # Stability metrics
│   ├── analysis/      # Root cause & recommendations
│   ├── output/        # Formatting & reporting
│   └── utils/         # Utilities
├── tests/             # Test suite
├── examples/          # Example prompts and queries
└── docs/              # Documentation
```

## Development Status

This is an MVP implementation. See the full specification in `AI_STABILITY_FINAL_SPEC.md`.

**Current Phase:** Core Infrastructure (Phase 1)

## License

MIT License - See LICENSE file for details

---

**Documentation:** [Full Specification](AI_STABILITY_FINAL_SPEC.md) | [Implementation Guide](IMPLEMENTATION_INSTRUCTIONS.md)
