# Test Scenarios

Ready-to-run test scenarios for Ruvrics. Use these to validate stability testing on different use cases.

## Quick Start

```bash
# Run all scenarios with default settings
./examples/run_tests.sh

# Or run individual scenarios manually (see below)
```

---

## Scenario 1: Simple Query (Baseline)

**Purpose:** Establish baseline with a simple, deterministic query.

```bash
ruvrics stability \
  --input examples/demo_consistent.json \
  --model gpt-4o-mini \
  --runs 15
```

**Expected:** 95-100% stability, SAFE verdict.

---

## Scenario 2: Customer Support Agent (Multi-Tool)

**Purpose:** Test tool routing consistency with 4 tools and realistic customer context.

**Tools:** `search_knowledge_base`, `get_customer_info`, `create_ticket`, `process_refund`

```bash
ruvrics stability \
  --input examples/scenario_support_agent/query.json \
  --tools examples/scenario_support_agent/tools.json \
  --tool-mocks examples/scenario_support_agent/tool_mocks.json \
  --model gpt-4o-mini \
  --runs 15
```

**Expected:** Tests tool selection consistency and response stability.

---

## Scenario 3: Ambiguous Tool Selection

**Purpose:** Test behavior when multiple tools could apply.

**Tools:** `get_weather`, `get_news`, `get_calendar`

```bash
ruvrics stability \
  --input examples/demo_tool_ambiguous.json \
  --tools examples/demo_tools.json \
  --tool-mocks examples/demo_tool_mocks.json \
  --model gpt-4o-mini \
  --runs 15
```

**Expected:** May show tool routing variance.

---

## Scenario 4: Long Context Analysis

**Purpose:** Test stability with large context (executive briefing).

```bash
ruvrics stability \
  --input examples/query_executive_briefing.json \
  --model gpt-4o-mini \
  --runs 15
```

**Expected:** Tests semantic consistency on complex analysis tasks.

---

## Scenario 5: Temperature Variance Testing

**Purpose:** Test how temperature affects response consistency.

```bash
# Deterministic (temp=0)
ruvrics stability \
  --input examples/query_executive_briefing.json \
  --model gpt-4o-mini \
  --temperature 0.0 \
  --runs 15

# Production-like (temp=0.5)
ruvrics stability \
  --input examples/query_executive_briefing.json \
  --model gpt-4o-mini \
  --temperature 0.5 \
  --runs 15

# High variance (temp=0.7)
ruvrics stability \
  --input examples/query_executive_briefing.json \
  --model gpt-4o-mini \
  --temperature 0.7 \
  --runs 15
```

**Expected:** Higher temperature = more variance in responses.

---

## Scenario 6: Baseline Comparison (Drift Detection)

**Purpose:** Save baseline, then compare after changes.

```bash
# Step 1: Save baseline
ruvrics stability \
  --input examples/query_executive_v1.json \
  --model gpt-4o-mini \
  --runs 15 \
  --save-baseline exec-v1

# Step 2: Compare modified version
ruvrics stability \
  --input examples/query_executive_v2.json \
  --model gpt-4o-mini \
  --runs 15 \
  --compare exec-v1
```

**Expected:** Shows stability comparison between configurations.

---

## Scenario 7: Model Comparison

**Purpose:** Compare stability across different models.

```bash
ruvrics stability \
  --input examples/demo_consistent.json \
  --model gpt-4o-mini \
  --compare-model gpt-4o \
  --runs 15
```

**Expected:** Side-by-side model stability comparison.

---

## Adding New Scenarios

1. Create input JSON in `examples/` or `examples/scenario_name/`
2. If using tools, add `tools.json` and `tool_mocks.json`
3. Add documentation to this file
4. Update `run_tests.sh` if needed

---

## File Structure

```
examples/
├── README.md                  # Examples overview
├── TEST_SCENARIOS.md          # This file
├── run_tests.sh               # Automated test runner
│
├── demo_consistent.json       # Simple baseline query
├── demo_tool_ambiguous.json   # Ambiguous tool scenario
├── demo_tools.json            # Tool definitions (weather/news/calendar)
├── demo_tool_mocks.json       # Mock responses for demo tools
│
├── query.json                 # Travel booking example
├── query_simple.json          # Simple query with system prompt
├── query_messages.json        # Multi-turn conversation format
├── query_with_tools.json      # Query with embedded tools
├── query_executive_*.json     # Executive briefing scenarios (v1, v2, briefing)
├── tools.json                 # Flight search tool
├── tool_mocks.json            # Flight search mock
├── system_prompt.txt          # Example system prompt file
│
├── scenario_support_agent/    # Multi-tool customer support agent
│   ├── query.json
│   ├── tools.json
│   └── tool_mocks.json
│
├── scenarios/                 # Advanced tool scenarios (6 scenarios)
│   └── scenario_*             # Multi-tool ambiguity, optional tools, etc.
│
└── edge_cases/                # Edge case testing (6 cases)
    └── edge_*                 # Subjective, creative, ambiguous, etc.
```
