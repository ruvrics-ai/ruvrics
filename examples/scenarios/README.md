# Advanced Test Scenarios for Ruvrics

This directory contains 6 advanced scenarios for testing LLM behavioral stability.

## Implementation Readiness Summary (v0.2.2)

| Scenario | Ready? | What's Measured | Notes |
|----------|--------|-----------------|-------|
| 1. Multi-tool ambiguity | ✅ Yes | Tool NAME + ARGUMENT variance | Full support |
| 2. Optional tool | ✅ Yes | Tool usage vs direct answer | Full support |
| 3. Argument drift | ✅ Yes | Per-tool argument consistency | NEW in v0.2.2 |
| 4. Tool→answer mismatch | ✅ Yes | Semantic consistency of final answer | Full support |
| 5. Tool failure | ✅ Yes | Response consistency under errors | Full support |
| 6. Multi-step chains | ✅ Yes | Tool sequence consistency | NEW in v0.2.2 (up to 5 iterations) |

---

## Scenario 1: Multi-Tool Ambiguity

**Goal**: Detect when tool selection varies across runs

**Files**:
- `scenario_1_multi_tool_ambiguity.json` - User prompt
- `scenario_1_tools.json` - 3 overlapping tools (search, check_availability, make_reservation)
- `scenario_1_mocks.json` - Mock responses

**Run Command**:
```bash
ruvrics stability \
  --input examples/scenarios/scenario_1_multi_tool_ambiguity.json \
  --tools examples/scenarios/scenario_1_tools.json \
  --tool-mocks examples/scenarios/scenario_1_mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

**What Ruvrics Should Reveal**:
- `tool_variance`: HIGH (if model chooses different tools)
- `argument_variance`: HIGH/MEDIUM (if same tool called with different arguments)
- `pattern_distribution`: Shows which tool combinations were selected
- Example: `{search_restaurants}: 12, {search_restaurants, check_availability}: 5, {search_restaurants, make_reservation}: 3`

**v0.2.2 Update**:
Arguments ARE now compared! If model calls `search_restaurants` with different arguments (e.g., `cuisine="Italian"` vs `cuisine="italian restaurant"`), this WILL be flagged as `ARGUMENT_DRIFT` root cause.

---

## Scenario 2: Optional Tool

**Goal**: Detect unnecessary tool calls (tool can help but isn't strictly needed)

**Files**:
- `scenario_2_optional_tool.json` - Factual question (capital of France)
- `scenario_2_tools.json` - web_search tool
- `scenario_2_mocks.json` - Search results

**Run Command**:
```bash
ruvrics stability \
  --input examples/scenarios/scenario_2_optional_tool.json \
  --tools examples/scenarios/scenario_2_tools.json \
  --tool-mocks examples/scenarios/scenario_2_mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

**What Ruvrics Should Reveal**:
- `tool_variance`: HIGH (if some runs call tool, some don't)
- `tool_usage_percentage`: Shows % of runs that used the tool
- `pattern_distribution`: `{none}: 8, {web_search}: 12`

**This is a HUGE real-world pain point** - models often call tools when they don't need to.

---

## Scenario 3: Argument Drift (Partial Tool Correctness)

**Goal**: Detect when same tool is called with varying arguments

**Files**:
- `scenario_3_argument_drift.json` - Ambiguous prompt ("recent" is vague)
- `scenario_3_tools.json` - search_news with many optional params
- `scenario_3_mocks.json` - News results

**Run Command**:
```bash
ruvrics stability \
  --input examples/scenarios/scenario_3_argument_drift.json \
  --tools examples/scenarios/scenario_3_tools.json \
  --tool-mocks examples/scenarios/scenario_3_mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

**What Ruvrics Should Reveal (v0.2.2)**:
- `tool_variance`: LOW (always calls `search_news`)
- `argument_variance`: HIGH/MEDIUM (arguments vary)
- Report shows: "Argument Drift Analysis" section with per-tool breakdown

**Example argument variations detected**:
  - Run 1: `{query: "tech news", limit: 5}`
  - Run 2: `{query: "technology news", limit: 10, sort_by: "date"}`
  - Run 3: `{query: "latest tech", category: "technology"}`

**v0.2.2 Implementation**: `calculate_argument_consistency()` metric now analyzes argument patterns and flags `ARGUMENT_DRIFT` root cause.

---

## Scenario 4: Tool → Answer Mismatch

**Goal**: Catch when tool output is the same but LLM's interpretation varies

**Files**:
- `scenario_4_answer_mismatch.json` - "Best laptop for video editing"
- `scenario_4_tools.json` - Product search tool
- `scenario_4_mocks.json` - 3 similar laptops (MacBook, Dell XPS, ASUS ProArt)

**Run Command**:
```bash
ruvrics stability \
  --input examples/scenarios/scenario_4_answer_mismatch.json \
  --tools examples/scenarios/scenario_4_tools.json \
  --tool-mocks examples/scenarios/scenario_4_mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

**What Ruvrics Should Reveal**:
- `tool_variance`: LOW (same tool always called)
- `semantic_drift`: MEDIUM/HIGH (different recommendations)
- Example: "Run 1 recommends MacBook, Run 2 recommends Dell XPS"

**This is where Ruvrics becomes decision-relevant** - same data, different conclusions.

---

## Scenario 5: Tool Failure / Empty Response

**Goal**: Test robustness under failure conditions

**Files**:
- `scenario_5_tool_failure.json` - Account balance request
- `scenario_5_tools.json` - Banking tools
- `scenario_5_mocks_error.json` - SERVICE_UNAVAILABLE errors
- `scenario_5_mocks_empty.json` - Empty/null results

**Run Commands**:
```bash
# Test with errors
ruvrics stability \
  --input examples/scenarios/scenario_5_tool_failure.json \
  --tools examples/scenarios/scenario_5_tools.json \
  --tool-mocks examples/scenarios/scenario_5_mocks_error.json \
  --model gpt-4o-mini \
  --runs 20

# Test with empty results
ruvrics stability \
  --input examples/scenarios/scenario_5_tool_failure.json \
  --tools examples/scenarios/scenario_5_tools.json \
  --tool-mocks examples/scenarios/scenario_5_mocks_empty.json \
  --model gpt-4o-mini \
  --runs 20
```

**What Ruvrics Should Reveal**:
- How consistently does the model handle errors?
- Does it:
  - Apologize and suggest retry?
  - Hallucinate data?
  - Gracefully degrade?
- `semantic_drift`: Shows response consistency under failures

**This scenario separates toys from tools.**

---

## Scenario 6: Multi-Step Tool Chains

**Goal**: Test sequence stability for chained tool calls

**Files**:
- `scenario_6_multi_step.json` - Book flight AND hotel
- `scenario_6_tools.json` - search_flights → book_flight → search_hotels → book_hotel
- `scenario_6_mocks.json` - Results for all 4 tools

**Run Command**:
```bash
ruvrics stability \
  --input examples/scenarios/scenario_6_multi_step.json \
  --tools examples/scenarios/scenario_6_tools.json \
  --tool-mocks examples/scenarios/scenario_6_mocks.json \
  --model gpt-4o-mini \
  --runs 20
```

**v0.2.2 Implementation**:
Ruvrics now executes multi-turn tool chains (up to 5 iterations by default):
1. User query → LLM → search_flights
2. Results → LLM → book_flight
3. Results → LLM → search_hotels
4. Results → LLM → book_hotel
5. Results → LLM → final response

**What Ruvrics Should Reveal**:
- `chain_variance`: HIGH/MEDIUM (if tool sequences differ)
- `chain_details`: Shows sequence distribution (e.g., "search_flights → book_flight → search_hotels": 15 runs)
- Report shows: "Tool Chain Analysis" section with sequence breakdown

**Configuration**: Set `MAX_TOOL_ITERATIONS` env var to change max iterations (default: 5).

---

## v0.2.2 New Features

### Feature 1: Argument Consistency Metric ✅

**Implementation** (`ruvrics/metrics/tool.py`):
```python
def calculate_argument_consistency(runs: list[RunResult]) -> MetricResult:
    """Compare tool arguments across runs."""
    # For each tool called, track argument variations
    # Score based on similarity of argument values
    # Flag when same tool called with different args
```

**New Root Cause**: `ARGUMENT_DRIFT` - detected when same tools called with inconsistent arguments.

### Feature 2: Multi-Turn Execution ✅

**Implementation** (`ruvrics/core/executor.py`):
```python
# Loop until no more tool calls or max iterations reached
while tool_calls and tool_mocks and iteration < config.max_tool_iterations:
    # Execute tools with mock responses
    # Get next response
    # Check for more tool calls
    iteration += 1
```

**Configuration**: `MAX_TOOL_ITERATIONS` env var (default: 5)

### Feature 3: Tool Chain Consistency ✅

**Implementation** (`ruvrics/metrics/tool.py`):
```python
def calculate_tool_chain_consistency(runs: list[RunResult]) -> MetricResult:
    """Track tool execution sequences across runs."""
    # Extract tool sequences from each run
    # Compare sequences for consistency
    # Track multi-turn runs
```

**New Root Cause**: `CHAIN_VARIANCE` - detected when tool execution order varies across runs.

---

## Quick Test Commands

```bash
# Run all supported scenarios
for i in 1 2 4 5; do
  echo "=== Scenario $i ==="
  ruvrics stability \
    --input examples/scenarios/scenario_${i}*.json \
    --tools examples/scenarios/scenario_${i}_tools.json \
    --tool-mocks examples/scenarios/scenario_${i}_mocks*.json \
    --model gpt-4o-mini \
    --runs 10
done
```

## Expected Report Insights (v0.2.2)

| Scenario | Tool Variance | Argument Variance | Chain Variance | Semantic Drift | Primary Root Cause |
|----------|--------------|-------------------|----------------|----------------|-------------------|
| 1 | HIGH | MEDIUM | N/A | MEDIUM | NONDETERMINISTIC_TOOL_ROUTING |
| 2 | HIGH | LOW | N/A | LOW | NONDETERMINISTIC_TOOL_ROUTING |
| 3 | LOW | HIGH | N/A | LOW | ARGUMENT_DRIFT |
| 4 | LOW | LOW | N/A | MEDIUM/HIGH | UNDERSPECIFIED_PROMPT |
| 5 | LOW | LOW | N/A | MEDIUM | GENERAL_INSTABILITY |
| 6 | VARIES | VARIES | HIGH/MEDIUM | VARIES | CHAIN_VARIANCE |

**New in v0.2.2**: Scenarios 3 and 6 are now fully supported with proper root cause detection.
