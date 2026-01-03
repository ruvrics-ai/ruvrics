# Ruvrics

AI Behavioral Stability and Reliability Engine

Ruvrics measures whether an LLM system behaves consistently under identical conditions. It runs N identical requests and analyzes variance across semantic meaning, tool usage, output structure, and length.

## What is Stability?

Large Language Models are nondeterministic by nature. Even with temperature set to 0, the same input can produce different outputs. Ruvrics quantifies this instability and identifies its root causes.

The stability score is a weighted average of 4 core metrics:
- Semantic Consistency (40%): Do outputs mean the same thing?
- Tool Consistency (25%): Does the model use tools consistently?
- Structural Consistency (20%): Is the output format stable?
- Length Consistency (15%): Are responses similarly verbose?

## Risk Classifications

- SAFE (score >= 90%): System is stable and ready for production
- RISKY (70-89%): Review recommended fixes before shipping
- DO_NOT_SHIP (< 70%): Critical instability issues detected

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)

### Install from PyPI

```bash
pip install ruvrics
```

### Install from Source

```bash
git clone https://github.com/YOUR_USERNAME/ruvrics.git
cd ruvrics
pip install -e .
```

### Configure API Keys

Create a .env file in your project directory:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Or export them as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Quick Start

### Basic Usage

```bash
ruvrics stability \
  --input query.json \
  --model gpt-4o-mini \
  --runs 20
```

### With System Prompt

```bash
ruvrics stability \
  --prompt system_prompt.txt \
  --input query.json \
  --model gpt-4-turbo \
  --runs 20
```

### With Tools (Modular Approach)

```bash
ruvrics stability \
  --prompt system_prompt.txt \
  --input query.json \
  --tools tools.json \
  --model gpt-4o \
  --runs 20
```

### Save Results

```bash
ruvrics stability \
  --input query.json \
  --model claude-sonnet-4 \
  --runs 20 \
  --output results.json
```

## Input Formats

Ruvrics supports three input formats:

### Simple Format

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

### With Tools (Modular)

**query.json:**
```json
{
  "user_input": "Find flights to Tokyo"
}
```

**tools.json:**
```json
[
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
```

You can also combine formats by using separate files:

```bash
ruvrics stability \
  --prompt system_prompt.txt \
  --input query.json \
  --tools tools.json \
  --model gpt-4o \
  --runs 20
```

## Supported Models

### OpenAI
- gpt-4-turbo
- gpt-4
- gpt-4o
- gpt-4o-mini
- gpt-3.5-turbo

### Anthropic
- claude-opus-4
- claude-sonnet-4
- claude-sonnet-3.5
- claude-haiku-4

## Understanding the Report

### Component Breakdown

Each metric receives a score (0-100%) and a variance classification:

- LOW variance: Metric is stable
- MEDIUM variance: Some inconsistency detected
- HIGH variance: Significant instability

### Instability Fingerprint

When issues are detected, Ruvrics identifies the root cause:

1. NONDETERMINISTIC_TOOL_ROUTING: Model inconsistently decides whether to use tools
2. TOOL_CONFUSION: Tool usage affects output unpredictably
3. UNCONSTRAINED_ASSERTIONS: Model makes risky claims inconsistently
4. UNDERSPECIFIED_PROMPT: Prompt allows too much interpretation freedom
5. FORMAT_INCONSISTENCY: Output format not reliably enforced
6. VERBOSITY_DRIFT: Response length varies significantly
7. GENERAL_INSTABILITY: Multiple sources of variation

### Recommendations

Based on root causes, Ruvrics provides actionable fixes:

- Prompt improvements: Add constraints, examples, or negative instructions
- Code changes: Move decision logic outside the LLM
- Config adjustments: Lower temperature, enforce schemas

## Exit Codes

- 0: SAFE - System is stable
- 1: RISKY - Review recommended
- 2: DO_NOT_SHIP - Critical issues
- 3: Configuration error
- 4: Insufficient successful runs
- 5: Embedding error
- 6: Invalid JSON input
- 7: General Ruvrics error
- 8: Unexpected error

## Examples

See the examples/ directory for sample input files:

- query_simple.json: Basic query without tools
- query_with_tools.json: Simple query for use with separate tools file
- tools.json: Reusable tool definitions
- query.json: Query with embedded tools (legacy format)
- query_messages.json: Messages format example
- system_prompt.txt: Sample system prompt

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=ruvrics --cov-report=html
```

### Code Quality

```bash
# Format code
black ruvrics/ tests/

# Lint code
ruff check ruvrics/ tests/

# Type checking
mypy ruvrics/
```

## Project Structure

```
ruvrics/
├── ruvrics/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration management
│   ├── analysis/            # Analysis components
│   │   ├── fingerprint.py   # Root cause identification
│   │   ├── recommender.py   # Recommendation engine
│   │   └── scorer.py        # Overall stability scorer
│   ├── core/                # Core functionality
│   │   ├── adapters.py      # LLM provider adapters
│   │   ├── executor.py      # Execution orchestration
│   │   └── models.py        # Data models
│   ├── metrics/             # Stability metrics
│   │   ├── claims.py        # Claim pattern detection
│   │   ├── length.py        # Length consistency
│   │   ├── semantic.py      # Semantic consistency
│   │   ├── structural.py    # Structural consistency
│   │   └── tool.py          # Tool usage consistency
│   ├── output/              # Output formatting
│   │   └── formatter.py     # Terminal report formatter
│   └── utils/               # Utilities
│       └── errors.py        # Custom exceptions
├── tests/                   # Test suite
├── examples/                # Example input files
├── docs/                    # Documentation
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Implementation Details

### Semantic Consistency

Uses sentence-transformers (all-MiniLM-L6-v2) to compute embeddings and measures cosine similarity to the centroid. This approach is O(N) instead of O(N²) pairwise comparison.

### Tool Consistency

Normalizes tool call patterns using order-independent frozensets and measures how often the most common pattern appears.

### Structural Consistency

Detects output structure (JSON, Markdown, Text, etc.) and measures how often the most common structure appears.

### Length Consistency

Uses Coefficient of Variation (CV = std/mean) to measure length variance. Lower CV indicates more consistent length.

### Claim Detection

Pattern-based detection of risky claims:
- Guarantees and absolute promises
- False authority claims
- Hallucinated specifics
- Overconfident assertions

## Thresholds

All thresholds are configurable in ruvrics/config.py:

**Risk Classification:**
- SAFE: score >= 90%
- RISKY: 70% <= score < 90%
- DO_NOT_SHIP: score < 70%

**Semantic Variance:**
- LOW: similarity >= 85%
- MEDIUM: 70% <= similarity < 85%
- HIGH: similarity < 70%

**Tool Variance:**
- LOW: consistency >= 95%
- MEDIUM: 80% <= consistency < 95%
- HIGH: consistency < 80%

**Structural Variance:**
- LOW: consistency >= 95%
- MEDIUM: 85% <= consistency < 95%
- HIGH: consistency < 85%

**Length Variance (CV):**
- LOW: CV < 0.15
- MEDIUM: 0.15 <= CV < 0.30
- HIGH: CV >= 0.30

## Documentation

For detailed technical documentation, see:
- docs/AI_STABILITY_FINAL_SPEC.md - Complete technical specification
- docs/IMPLEMENTATION_INSTRUCTIONS.md - Implementation guide
- docs/CONFIGURATION_GUIDE.md - Configuration patterns

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: pytest tests/ -v
5. Format code: black ruvrics/ tests/
6. Submit a pull request

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
