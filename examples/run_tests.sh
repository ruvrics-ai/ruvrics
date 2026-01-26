#!/bin/bash

# =============================================================================
# Ruvrics Test Runner
# =============================================================================
# Runs all test scenarios for stability testing.
# Modify variables below to customize your test run.
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Model to test
MODEL="gpt-4o-mini"

# Number of runs per test
RUNS=15

# Temperature (0.0 = deterministic, 0.5-0.7 = production-like)
TEMPERATURE=0.0

# Examples directory (relative to where script is run from)
EXAMPLES_DIR="examples"

# Pause between tests for screenshots (set to "true" or "false")
PAUSE_BETWEEN_TESTS="true"

# =============================================================================
# PATHS - Derived from EXAMPLES_DIR
# =============================================================================

SIMPLE_QUERY="${EXAMPLES_DIR}/demo_consistent.json"
TOOL_AMBIGUOUS="${EXAMPLES_DIR}/demo_tool_ambiguous.json"
GENERIC_TOOLS="${EXAMPLES_DIR}/demo_tools.json"
GENERIC_MOCKS="${EXAMPLES_DIR}/demo_tool_mocks.json"
EXEC_BRIEFING="${EXAMPLES_DIR}/query_executive_briefing.json"
EXEC_V1="${EXAMPLES_DIR}/query_executive_v1.json"
EXEC_V2="${EXAMPLES_DIR}/query_executive_v2.json"
SUPPORT_QUERY="${EXAMPLES_DIR}/scenario_support_agent/query.json"
SUPPORT_TOOLS="${EXAMPLES_DIR}/scenario_support_agent/tools.json"
SUPPORT_MOCKS="${EXAMPLES_DIR}/scenario_support_agent/tool_mocks.json"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
    echo ""
}

pause_if_enabled() {
    if [ "$PAUSE_BETWEEN_TESTS" = "true" ]; then
        read -p "Press Enter for next test..."
    fi
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

echo ""
echo "========================================"
echo "  RUVRICS TEST RUNNER"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Model:       $MODEL"
echo "  Runs:        $RUNS"
echo "  Temperature: $TEMPERATURE"
echo "  Examples:    $EXAMPLES_DIR"
echo ""

if [ "$PAUSE_BETWEEN_TESTS" = "true" ]; then
    echo "Pausing between tests for screenshots."
    echo "Press Enter to start..."
    read
fi

# -----------------------------------------------------------------------------
# Test 1: Simple Query (Baseline)
# -----------------------------------------------------------------------------
print_header "TEST 1: SIMPLE QUERY (BASELINE)"

ruvrics stability \
    --input "$SIMPLE_QUERY" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --runs "$RUNS"

pause_if_enabled

# -----------------------------------------------------------------------------
# Test 2: Customer Support Agent (Multi-Tool)
# -----------------------------------------------------------------------------
print_header "TEST 2: CUSTOMER SUPPORT AGENT (4 TOOLS)"

if [ -f "$SUPPORT_QUERY" ]; then
    ruvrics stability \
        --input "$SUPPORT_QUERY" \
        --tools "$SUPPORT_TOOLS" \
        --tool-mocks "$SUPPORT_MOCKS" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --runs "$RUNS"
else
    echo "Skipped: Support agent scenario not found"
fi

pause_if_enabled

# -----------------------------------------------------------------------------
# Test 3: Ambiguous Tool Selection
# -----------------------------------------------------------------------------
print_header "TEST 3: AMBIGUOUS TOOL SELECTION"

if [ -f "$TOOL_AMBIGUOUS" ]; then
    ruvrics stability \
        --input "$TOOL_AMBIGUOUS" \
        --tools "$GENERIC_TOOLS" \
        --tool-mocks "$GENERIC_MOCKS" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --runs "$RUNS"
else
    echo "Skipped: Tool ambiguous scenario not found"
fi

pause_if_enabled

# -----------------------------------------------------------------------------
# Test 4: Long Context Analysis
# -----------------------------------------------------------------------------
print_header "TEST 4: LONG CONTEXT ANALYSIS"

if [ -f "$EXEC_BRIEFING" ]; then
    ruvrics stability \
        --input "$EXEC_BRIEFING" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --runs "$RUNS"
else
    echo "Skipped: Executive briefing scenario not found"
fi

pause_if_enabled

# -----------------------------------------------------------------------------
# Test 5: Baseline Comparison
# -----------------------------------------------------------------------------
print_header "TEST 5: BASELINE COMPARISON"

if [ -f "$EXEC_V1" ] && [ -f "$EXEC_V2" ]; then
    echo "Step 1: Saving baseline..."
    ruvrics stability \
        --input "$EXEC_V1" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --runs "$RUNS" \
        --save-baseline test-baseline

    echo ""
    echo "Step 2: Comparing to baseline..."
    ruvrics stability \
        --input "$EXEC_V2" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --runs "$RUNS" \
        --compare test-baseline
else
    echo "Skipped: Executive v1/v2 scenarios not found"
fi

pause_if_enabled

# -----------------------------------------------------------------------------
# Test 6: Model Comparison (Optional)
# -----------------------------------------------------------------------------
print_header "TEST 6: MODEL COMPARISON"

COMPARE_MODEL="gpt-4o"

echo "Comparing $MODEL vs $COMPARE_MODEL"
echo "(This will run tests on both models)"
echo ""

ruvrics stability \
    --input "$SIMPLE_QUERY" \
    --model "$MODEL" \
    --compare-model "$COMPARE_MODEL" \
    --temperature "$TEMPERATURE" \
    --runs "$RUNS"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  ALL TESTS COMPLETE"
echo "========================================"
echo ""
echo "Reports saved to: reports/"
echo ""
