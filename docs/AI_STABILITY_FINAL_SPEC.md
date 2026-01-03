# CORE ENGINE LOGIC — MVP vs V2 SPEC

**AI Behavioral Stability & Reliability Engine**

**Version:** 1.0 Final  
**Last Updated:** December 30, 2025  
**Status:** ✅ Production Ready

> **Purpose:**
> Measure whether an LLM-based system behaves consistently under identical conditions, explain *why* it is unstable, and suggest concrete fixes **before deployment**.

---

## Table of Contents

1. [Supported Input Formats](#1-supported-input-formats)
2. [Execution Strategy](#2-execution-strategy)
3. [Stability Score Formula](#3-stability-score-formula)
4. [Instability Fingerprint](#4-instability-fingerprint)
5. [Claim / Safety Instability](#5-claim--safety-instability)
6. [Recommendation Decision Tree](#6-recommendation-decision-tree)
7. [Output Contract (CLI)](#7-output-contract-cli)
8. [Error Handling](#8-error-handling)
9. [Explicit Non-Goals](#9-explicit-non-goals-mvp)
10. [Design Principles](#10-design-principles)

---

## 1. Supported Input Formats

### MVP ✅ (Keep)

Supports three explicit request formats:

#### Format A: Simple
```json
{
  "system_prompt": "You are a helpful assistant",
  "user_input": "What is the weather in SF?"
}
```

#### Format B: Messages (OpenAI/Anthropic Compatible)
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the weather in SF?"}
  ]
}
```

#### Format C: Tool-enabled
```json
{
  "messages": [
    {"role": "user", "content": "Book a flight to NYC"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_flights",
        "description": "Search for available flights"
      }
    }
  ],
  "tool_choice": "auto"
}
```

**Why MVP:**
These cover 90% of real-world LLM systems with minimal abstraction and align directly with OpenAI / Anthropic APIs.

---

### V2 ⏭️ (Defer)

* Custom adapters (LangGraph, CrewAI, AutoGen)
* Multi-agent orchestration inputs
* Streaming / partial outputs
* RAG pipeline inputs

**Why V2:**
Adds complexity without improving early signal. Focus on core LLM behavior first.

---

## 2. Execution Strategy

### MVP ✅

**Process:**
* Run identical request **N times** (default: 20, configurable 10-50)
* Fixed parameters (temperature, top_p, etc. as specified)
* Parallel execution when possible (respect rate limits)

**Captured Data Per Run:**

```python
{
  "run_id": int,                    # 1 to N
  "timestamp": str,                 # ISO 8601 format
  "output_text": str,               # Final text response
  "tool_calls": [                   # List of tool calls made
    {
      "name": str,                  # Tool/function name
      "call_sequence": int          # 1st call, 2nd call, etc.
    }
  ],
  "output_length_tokens": int,      # Token count from API
  "output_length_chars": int,       # Character count
  "output_structure": str,          # "json", "markdown", "text"
  "api_latency_ms": int,            # Response time
  "model_used": str,                # Actual model identifier
  "error": str | None               # Error message if failed
}
```

**Tool Call Handling (MVP):**
- Captures: **unique set of tool names** used per run
- Ignores: call frequency, argument values, results
- Order: NOT considered for matching
- Example: `[search, search, book, search]` → `{book, search}`

**Why This Approach:**
Focuses on "did the model choose to use tools" vs "exact execution path". Sufficient for detecting nondeterministic tool routing.

**Why MVP:**
Stability emerges from repetition. No production telemetry required.

---

### V2 ⏭️

* Cross-version comparisons (GPT-4 vs GPT-4o)
* Prompt/model diffs (what changed between versions)
* Longitudinal trend storage (track stability over time)
* Tool argument consistency analysis
* Tool call ordering sensitivity

---

## 3. Stability Score Formula

### MVP ✅ (Locked)

**Overall Score:** ∈ [0, 100]

```python
stability_score = (
    0.40 * semantic_consistency_score +
    0.25 * tool_consistency_score +
    0.20 * structural_consistency_score +
    0.15 * length_consistency_score
)
```

**Risk Classification:**

```python
if stability_score >= 90:
    risk = "SAFE"           # ✅ Ship with confidence
elif stability_score >= 70:
    risk = "RISKY"          # ⚠️ Review required
else:
    risk = "DO_NOT_SHIP"    # ❌ Critical issues
```

**Rationale:**
- **40% Semantic**: Meaning changes are most critical for correctness
- **25% Tool**: Tool failures cause functional breakage
- **20% Structural**: Format breaks cause integration failures
- **15% Length**: Cosmetic variance, least impact on correctness

---

### 3.1 Semantic Consistency

#### MVP ✅ (Implementation)

**Method:**
1. Embed all N outputs using `sentence-transformers/all-MiniLM-L6-v2`
2. Compute centroid embedding (mean of all embeddings)
3. Calculate mean cosine similarity to centroid

**Formula:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all outputs
embeddings = [model.encode(output) for output in outputs]
centroid = np.mean(embeddings, axis=0)

# Calculate similarities
similarities = [
    np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
    for emb in embeddings
]

# Mean similarity (range: typically 0.7-1.0 for stable outputs)
semantic_raw_score = np.mean(similarities)

# Convert to 0-100 scale
semantic_consistency_score = semantic_raw_score * 100
```

**Thresholds:**

```python
if semantic_consistency_score >= 85:
    semantic_drift = "LOW"      # Outputs nearly identical in meaning
elif semantic_consistency_score >= 70:
    semantic_drift = "MEDIUM"   # Noticeable semantic variance
else:
    semantic_drift = "HIGH"     # Significantly different meanings
```

**Why Centroid vs Pairwise:**
- Centroid: O(N) complexity, stable, intuitive
- Pairwise: O(N²) complexity, sensitive to outliers
- For N=20, centroid is sufficient and 10x faster

**Example Scores:**
- 95-100: "The weather is sunny" vs "It's sunny today" (paraphrases)
- 80-90: "It's sunny" vs "The forecast shows clear skies" (related)
- <70: "It's sunny" vs "I'll check the database" (different intent)

---

#### V2 ⏭️

* Pairwise similarity matrix with outlier detection
* Standard deviation analysis
* Semantic cluster identification
* Context-aware embedding models

**Why V2:**
Higher precision, but harder to reason about for early adopters.

---

### 3.2 Tool Usage Consistency

#### MVP ✅ (Implementation)

**Method:**
1. Extract unique tool names from each run (order-independent)
2. Identify most common pattern (mode)
3. Score = fraction of runs matching mode

**Formula:**

```python
from collections import Counter

# Normalize tool calls to sets (deduplicates, ignores order)
def normalize_tools(run):
    return frozenset(call["name"] for call in run["tool_calls"])

tool_patterns = [normalize_tools(run) for run in runs]

# Find most common pattern
pattern_counts = Counter(tool_patterns)
most_common_pattern, mode_count = pattern_counts.most_common(1)[0]

# Calculate consistency
tool_consistency_score = (mode_count / len(runs)) * 100
```

**Special Cases:**

```python
# No tools available in input
if "tools" not in input_config:
    tool_consistency_score = 100  # N/A, don't penalize
    tool_variance = "N/A"

# Some runs use tools, some don't (dangerous!)
tools_used_count = sum(1 for p in tool_patterns if len(p) > 0)
if 0 < tools_used_count < len(runs):
    tool_variance = "HIGH"  # Always HIGH if inconsistent presence
```

**Thresholds:**

```python
if tool_consistency_score >= 95:
    tool_variance = "LOW"       # Nearly deterministic
elif tool_consistency_score >= 80:
    tool_variance = "MEDIUM"    # Some variance
else:
    tool_variance = "HIGH"      # Unpredictable tool routing
```

**Why MVP:**
Tool nondeterminism is the #1 real-world failure mode in production systems.

**Examples:**
- LOW: 19/20 runs use `{search}` → 95% consistency
- MEDIUM: 17/20 runs use `{search}`, 3/20 use `{search, book}` → 85%
- HIGH: 12/20 use `{search}`, 8/20 use nothing → 60%

---

#### V2 ⏭️

* Tool call frequency sensitivity (used once vs three times)
* Tool ordering consistency
* Tool argument similarity analysis
* Tool result impact on output

---

### 3.3 Structural Consistency

#### MVP ✅ (Implementation)

**Method:**
Classify each output's structure, find dominant pattern, measure adherence.

**Structure Detection:**

```python
import json
import re

def detect_structure(output: str) -> str:
    """Classify output structure type"""
    
    # Try JSON parsing first
    try:
        obj = json.loads(output.strip())
        if isinstance(obj, dict):
            # Extract top-level keys, sorted
            keys = sorted(obj.keys())
            return f"JSON:DICT:{','.join(keys)}"
        elif isinstance(obj, list):
            return "JSON:ARRAY"
        else:
            return "JSON:PRIMITIVE"
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Check for markdown
    markdown_patterns = [
        r'^#{1,6}\s',           # Headers
        r'^\s*[\*\-\+]\s',      # Unordered lists
        r'^\s*\d+\.\s',         # Ordered lists
        r'```',                 # Code blocks
        r'\[.+\]\(.+\)',        # Links
    ]
    
    if any(re.search(p, output, re.MULTILINE) for p in markdown_patterns):
        return "MARKDOWN"
    
    # Default to plain text
    return "TEXT"
```

**Scoring:**

```python
structure_types = [detect_structure(out) for out in outputs]
type_counts = Counter(structure_types)

# Most common structure
dominant_structure, dominant_count = type_counts.most_common(1)[0]

# Score = % matching dominant
structural_consistency_score = (dominant_count / len(outputs)) * 100
```

**Thresholds:**

```python
if structural_consistency_score >= 95:
    structural_variance = "LOW"     # Stable format
elif structural_consistency_score >= 85:
    structural_variance = "MEDIUM"  # Occasional breaks
else:
    structural_variance = "HIGH"    # Unreliable formatting
```

**Examples:**
- LOW: 20/20 outputs are valid JSON with keys `{status, data}`
- MEDIUM: 18/20 are JSON, 2/20 are plain text error messages
- HIGH: 12/20 JSON, 5/20 markdown, 3/20 plain text

---

#### V2 ⏭️

* Schema validation depth (nested structure consistency)
* Type consistency within JSON fields
* Markdown structure hierarchy checking

---

### 3.4 Length Consistency

#### MVP ✅ (Implementation)

**Method:**
Use Coefficient of Variation (CV) to measure relative length variance.

**Formula:**

```python
import numpy as np

lengths = [run["output_length_tokens"] for run in runs]
mean_length = np.mean(lengths)
std_length = np.std(lengths)

# Handle edge case: very short outputs
if mean_length < 5:
    # Outputs too short to measure variance meaningfully
    length_consistency_score = 100
    length_variance = "LOW"
else:
    # Coefficient of Variation
    CV = std_length / mean_length
    
    # Map to 0-100 score (CV of 0.4 → score of 0)
    length_consistency_score = max(0, (1 - CV/0.4) * 100)
```

**Thresholds:**

```python
CV = std_length / mean_length

if CV < 0.15:
    length_variance = "LOW"      # ~15% variation
elif CV < 0.30:
    length_variance = "MEDIUM"   # ~30% variation
else:
    length_variance = "HIGH"     # 30%+ variation
```

**Why Coefficient of Variation:**
Normalizes for output scale. 100-token variance matters more for 200-token outputs than 2000-token outputs.

**Examples:**
- LOW: Outputs range 95-105 tokens (mean=100, std=5, CV=0.05)
- MEDIUM: Outputs range 80-120 tokens (mean=100, std=20, CV=0.20)
- HIGH: Outputs range 50-150 tokens (mean=100, std=50, CV=0.50)

---

#### V2 ⏭️

* Percentile-based outlier detection
* Section-level length consistency (intro vs body vs conclusion)

---

## 4. Instability Fingerprint

### MVP ✅ (Required)

**Purpose:** Explain WHY the system is unstable, not just that it is.

**Output Format:**

Each category emits: `LOW | MEDIUM | HIGH | N/A`

| Category            | LOW Score     | MEDIUM Score  | HIGH Score   |
|---------------------|---------------|---------------|--------------|
| Semantic Drift      | score >= 85   | 70 ≤ x < 85   | score < 70   |
| Tool Variance       | score >= 95   | 80 ≤ x < 95   | score < 80   |
| Structural Variance | score >= 95   | 85 ≤ x < 95   | score < 85   |
| Length Variance     | score >= 80   | 60 ≤ x < 80   | score < 60   |

**Root Cause Decision Tree:**

```python
def identify_root_cause(metrics: dict) -> dict:
    """
    Identify primary instability root cause.
    Priority: Most specific → Most general
    """
    
    root_causes = []
    
    # 1. Check tool-use instability (highest priority)
    if metrics["tool_variance"] == "HIGH":
        if metrics["semantic_drift"] == "LOW":
            root_causes.append({
                "type": "NONDETERMINISTIC_TOOL_ROUTING",
                "severity": "HIGH",
                "description": "Model inconsistently decides whether to use tools",
                "details": f"Tool used in {metrics['tool_usage_percentage']}% of runs"
            })
        else:
            root_causes.append({
                "type": "TOOL_CONFUSION",
                "severity": "HIGH",
                "description": "Tool usage affects semantic output unpredictably"
            })
    
    # 2. Check claim instability (if applicable)
    if metrics.get("claim_variance") == "HIGH":
        root_causes.append({
            "type": "UNCONSTRAINED_ASSERTIONS",
            "severity": "CRITICAL",
            "description": "Model makes risky claims inconsistently",
            "details": f"Risky claims in {metrics['risky_claim_percentage']}% of runs"
        })
    
    # 3. Check semantic drift
    if metrics["semantic_drift"] == "HIGH":
        if metrics["structural_variance"] == "LOW":
            root_causes.append({
                "type": "UNDERSPECIFIED_PROMPT",
                "severity": "MEDIUM",
                "description": "Prompt allows too much interpretation freedom"
            })
        else:
            root_causes.append({
                "type": "GENERAL_INSTABILITY",
                "severity": "HIGH",
                "description": "Multiple sources of variation"
            })
    
    # 4. Check structural variance
    if metrics["structural_variance"] == "HIGH":
        root_causes.append({
            "type": "FORMAT_INCONSISTENCY",
            "severity": "MEDIUM",
            "description": "Output format not reliably enforced"
        })
    
    # 5. Check length variance (lowest priority)
    if metrics["length_variance"] == "HIGH" and len(root_causes) == 0:
        root_causes.append({
            "type": "VERBOSITY_DRIFT",
            "severity": "LOW",
            "description": "Response length varies significantly"
        })
    
    return {
        "primary_cause": root_causes[0] if root_causes else None,
        "all_causes": root_causes
    }
```

**Visual Example:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTABILITY FINGERPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Primary Root Cause: NONDETERMINISTIC_TOOL_ROUTING
Severity: HIGH

Observed Patterns:
├─ Tool usage variance:     HIGH (tool used in 12/20 runs = 60%)
├─ Semantic drift:          LOW  (outputs similar when tool not used)
├─ Claim instability:       MEDIUM (guarantees in 6/20 runs)
└─ Structural variance:     LOW

Real Examples:
• Run 3: "Let me search for that..." → [calls search_tool]
• Run 7: "Based on typical patterns, you should..." → [no tools]
• Run 15: "Let me search for that..." → [calls search_tool]

Analysis: Model cannot reliably decide when to use tools vs. answer directly.
```

**Why MVP:**
Explains root cause in language developers understand, with concrete examples.

---

### V2 ⏭️

* Confidence-weighted root causes (probabilistic ranking)
* Correlation analysis (which metrics co-vary)
* Temporal instability clustering (early vs late runs)
* Interactive drill-down into specific failure runs

---

## 5. Claim / Safety Instability

### MVP ⚠️ (Minimal, Binary Only)

**Purpose:** Flag when model *inconsistently* makes risky claims.

**Detection Method:** Pattern-based (fast, deterministic)

**Implementation:**

```python
import re

RISKY_CLAIM_PATTERNS = [
    # Absolute guarantees
    r'\b(guarantee|guaranteed|100% (?:safe|effective|accurate))\b',
    r'\b(always will|never (?:fails|breaks))\b',
    r'\b(certain(?:ly)? (?:will|to))\b',
    
    # False authority
    r'\b(legally (?:guaranteed|required|mandated|binding))\b',
    r'\b(certified|licensed|approved) by (?!the user|you|your)\b',
    r'\bI (?:have access to|can verify|checked)\b',
    
    # Hallucinated specifics (basic patterns only)
    r'\baccording to (?:our|my) (?:study|research|data)\b',
    r'\b(?:published|released) (?:on|in) [A-Z][a-z]+ \d{1,2},? \d{4}\b',
]

def detect_risky_claims(output: str) -> dict:
    """Check for risky claim patterns"""
    detected = []
    
    for pattern in RISKY_CLAIM_PATTERNS:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            detected.append({
                "pattern": pattern,
                "examples": matches[:2]  # First 2 matches
            })
    
    return {
        "has_risky_claims": len(detected) > 0,
        "patterns_found": detected
    }

# Analysis across all runs
def analyze_claim_instability(runs: List[dict]) -> dict:
    risky_runs = []
    
    for run in runs:
        result = detect_risky_claims(run["output_text"])
        if result["has_risky_claims"]:
            risky_runs.append({
                "run_id": run["run_id"],
                "patterns": result["patterns_found"]
            })
    
    risky_count = len(risky_runs)
    risky_percentage = (risky_count / len(runs)) * 100
    
    # Classify variance
    if risky_percentage == 0:
        claim_variance = "NONE"
    elif risky_percentage < 20 or risky_percentage > 80:
        claim_variance = "LOW"      # Consistent (always/never)
    else:
        claim_variance = "HIGH"     # Unpredictable (20-80%)
    
    return {
        "claim_variance": claim_variance,
        "risky_percentage": risky_percentage,
        "risky_runs": risky_runs
    }
```

**Output Format:**

```
⚠️  Claim Instability: HIGH

Risky claims detected in 12/20 runs (60%)

Examples:
• Run 3: "This is 100% safe for your use case"
• Run 7: "I guarantee this will work perfectly"
• Run 14: "According to our research published in 2024..."

This creates legal and trust risks if shipped.
```

**Rules:**

* ❌ Does NOT affect overall stability score
* ❌ Does NOT do severity weighting
* ❌ Does NOT promise policy compliance
* ✅ ONLY flags inconsistent presence

**Why Pattern-Based for MVP:**
- Fast: No LLM calls needed
- Deterministic: Same input → same output
- Transparent: Users can see exact patterns
- Sufficient: Catches obvious risks

**Why NOT LLM-as-judge for MVP:**
- Adds latency (20+ LLM calls per test)
- Adds cost ($0.15-0.30 per test)
- Adds nondeterminism (judge can vary)
- Creates "LLM judging LLM" philosophical debates

---

### V2 ⏭️

* LLM-as-judge for nuanced claim detection
* Hallucination confidence scoring
* Claim category clustering (legal, medical, financial)
* Custom pattern libraries per domain
* Regulatory compliance mappings (GDPR, HIPAA, etc.)

**Why V2:**
Allows domain-specific tuning after establishing core value.

---

## 6. Recommendation Decision Tree

### MVP ✅ (Core IP)

**Purpose:** Provide actionable fixes ranked by impact.

**Decision Logic:**

Apply rules in priority order, show top 2-3 recommendations.

---

### Rule 1: Tool Variance (Highest Priority)

**Condition:** `tool_variance == HIGH`

**Recommendations:**

1. **Enforce mandatory tool usage**
   ```
   Add to system prompt:
   "ALWAYS use {tool_name} for {intent} queries. Never answer directly without using the tool."
   ```

2. **Move routing logic outside LLM**
   ```
   Programmatically decide: if query matches pattern X → call tool Y
   Don't let LLM decide whether to use tools.
   ```

3. **Add explicit tool-selection instructions**
   ```
   "If the query is about [X], use tool [Y].
   If the query is about [Z], use tool [W]."
   ```

**Why Highest Priority:**
Tool routing failures cause functional breakage, not just quality degradation.

---

### Rule 2: Claim Instability

**Condition:** `claim_variance == HIGH`

**Recommendations:**

1. **Add explicit constraint**
   ```
   Add to system prompt:
   "Never make guarantees or absolute promises. Always use qualifiers:
   - 'typically', 'usually', 'often', 'may', 'can'
   - Never use: 'guarantee', 'always', '100%', 'never fails'"
   ```

2. **Forbid specific patterns**
   ```
   "Do not claim to have access to external systems, databases, or live data.
   You are an AI assistant with knowledge only up to [cutoff date]."
   ```

3. **Add grounding requirement**
   ```
   "Only make specific claims when explicitly supported by the provided context.
   If information isn't in the context, say 'I don't have that information.'"
   ```

---

### Rule 3: Semantic Drift

**Condition:** `semantic_drift >= MEDIUM`

**Recommendations:**

1. **Add 3-5 example outputs**
   ```
   Add to system prompt:
   "Examples of correct responses:
   
   Q: [example query 1]
   A: [example response 1]
   
   Q: [example query 2]
   A: [example response 2]"
   ```

2. **Add reasoning template**
   ```
   "Structure your response as:
   1. Acknowledge the question
   2. State relevant constraints
   3. Provide answer
   4. Include confidence level if uncertain"
   ```

3. **Add negative constraints**
   ```
   "Do NOT:
   - Provide medical/legal advice
   - Speculate beyond available information
   - Make comparisons to competitors"
   ```

---

### Rule 4: Structural Variance

**Condition:** `structural_variance >= MEDIUM`

**Recommendations:**

1. **Enforce JSON schema**
   ```
   Add to system prompt:
   "Always respond with valid JSON matching this exact schema:
   {
     "status": "success" | "error",
     "data": { ... },
     "message": "string"
   }"
   
   Or use API-level schema enforcement if supported (e.g., OpenAI's JSON mode).
   ```

2. **Add format validation example**
   ```
   "Example of correct format:
   {json_example}
   
   Do not deviate from this structure."
   ```

3. **Add retry logic**
   ```python
   # In your code
   if not validate_structure(response):
       response = retry_with_instruction("Your last response had invalid format...")
   ```

---

### Rule 5: Length Variance

**Condition:** `length_variance >= MEDIUM`

**Recommendations:**

1. **Specify length constraints**
   ```
   "Keep responses to 2-3 sentences."
   "Respond in under 100 words."
   "Provide a brief summary (50-75 tokens)."
   ```

2. **Add conciseness instruction**
   ```
   "Be concise and direct. Avoid unnecessary explanations."
   ```

3. **Provide length-constrained examples**
   ```
   Show examples of appropriately-sized responses.
   ```

---

**Recommendation Selection Rules:**

```python
def generate_recommendations(root_causes: List[dict]) -> List[str]:
    """
    Generate top 2-3 actionable fixes
    Priority: CRITICAL > HIGH > MEDIUM > LOW
    """
    
    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_causes = sorted(
        root_causes,
        key=lambda x: severity_order[x["severity"]]
    )
    
    recommendations = []
    
    # Take top 2 recommendations per cause
    for cause in sorted_causes[:2]:  # Max 2 root causes
        cause_type = cause["type"]
        recs = RECOMMENDATION_MAP.get(cause_type, [])
        recommendations.extend(recs[:2])  # Top 2 fixes per cause
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for rec in recommendations:
        key = rec["title"]
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    
    return unique[:3]  # Max 3 recommendations total
```

---

### V2 ⏭️

* Context-aware recommendations (analyze prompt to suggest specific edits)
* Historical fix effectiveness tracking (learn which fixes work)
* Auto-generate prompt patches (programmatic prompt modification)
* A/B test recommendations (try multiple fixes, measure impact)

---

## 7. Output Contract (CLI)

### MVP ✅

**Terminal Output Format:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI STABILITY REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tested: gpt-4-turbo | Runs: 20/20 ✓ | Duration: 45.3s

Overall Stability Score: 62.3% ❌ HIGH RISK

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Semantic Consistency:    72.1% │ ████████████░░░░░░░░ │ MEDIUM drift
Structural Consistency:  85.0% │ █████████████████░░░ │ LOW variance  
Tool-Call Consistency:   45.0% │ █████████░░░░░░░░░░░ │ HIGH variance
Length Consistency:      90.5% │ ██████████████████░░ │ LOW variance

⚠️  Claim Instability: HIGH
    Risky claims detected in 12/20 runs (60%)
    Examples: "guarantee", "100% safe", "always works"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTABILITY FINGERPRINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Primary Issue: NONDETERMINISTIC_TOOL_ROUTING (Severity: HIGH)

Root Cause:
  Model inconsistently decides whether to use tools
  
Observed Behavior:
  • Tool "search_flights" used in only 9/20 runs (45%)
  • When tool skipped, model makes direct claims
  • Semantic output similar regardless of tool use
  
Real Output Examples:
  → Run 3: "Let me search for flights..." [uses tool]
  → Run 7: "I can help you find flights..." [no tool, makes claims]
  → Run 12: "Searching for options..." [uses tool]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED FIXES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Priority 1: Enforce Mandatory Tool Usage
  → Add to system prompt:
    "ALWAYS use search_flights tool for flight queries.
     Never answer directly without using the tool."

Priority 2: Prevent Guarantee Language  
  → Add constraint:
    "Never guarantee availability or prices.
     Use: 'I can search for...', 'typically', 'often'"

Priority 3: Move Tool Routing Outside LLM
  → Code change:
    if query_intent == "flight_search":
        force_tool_use(search_flights)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Apply recommended fixes to your prompt
2. Re-run stability test:
   $ airel stability --prompt updated.txt --input query.json --runs 20
3. Compare results to validate improvement

💡 Tip: Focus on Priority 1 fix first - it addresses 55% of instability

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Before/After Comparison (if previous run exists):**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPROVEMENT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stability Score: 62.3% → 91.5% ✅ (+29.2 points)

What Changed:
├─ Tool Consistency:   45% → 100% ✅ (HIGH → NONE)
├─ Claim Instability:  HIGH → LOW ✅  
├─ Semantic Drift:     MEDIUM → LOW ✅
└─ Structural Variance: LOW → LOW (unchanged)

Risk Classification: HIGH RISK → SAFE ✅

🎉 Your fixes worked! This system is now safe to ship.

Previous test: 2025-12-30 09:15:32 UTC
Current test:  2025-12-30 10:42:18 UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### V2 ⏭️

* HTML reports with interactive charts
* JSON export for CI/CD integration
* PDF compliance reports
* Team dashboards with historical trends
* Slack/email notifications

---

## 8. Error Handling

### MVP ✅ (Required)

**API Call Failures:**

```python
# Retry logic
MAX_RETRIES = 3
BACKOFF_MULTIPLIER = 2  # Exponential backoff

for attempt in range(MAX_RETRIES):
    try:
        response = call_llm_api(...)
        break
    except (APIError, RateLimitError, TimeoutError) as e:
        if attempt == MAX_RETRIES - 1:
            raise  # Final attempt failed
        
        wait_time = BACKOFF_MULTIPLIER ** attempt
        time.sleep(wait_time)
```

**Partial Run Failures:**

```python
# Require minimum successful runs
MIN_SUCCESSFUL_RUNS = 15  # For N=20 default

successful_runs = [r for r in runs if r["error"] is None]

if len(successful_runs) < MIN_SUCCESSFUL_RUNS:
    raise InsufficientDataError(
        f"Only {len(successful_runs)}/{len(runs)} runs succeeded. "
        f"Need at least {MIN_SUCCESSFUL_RUNS} for reliable analysis."
    )
```

**Empty or Invalid Outputs:**

```python
# Count as structural variance
if not output.strip():
    output_structure = "EMPTY"
    # Still include in analysis, counts against structural consistency
```

**Embedding Model Failures:**

```python
# Critical failure - abort immediately
try:
    embeddings = embed_outputs(outputs)
except EmbeddingError as e:
    print(f"❌ Fatal error: Could not embed outputs")
    print(f"   Reason: {e}")
    print(f"   Solution: Check embedding model installation")
    sys.exit(1)
```

**Rate Limiting:**

```python
# Respect rate limits with smart pacing
if rate_limit_hit:
    wait_time = extract_retry_after_header(response)
    print(f"⏸️  Rate limit reached. Waiting {wait_time}s...")
    time.sleep(wait_time)
    # Resume from where we left off
```

**User-Friendly Error Messages:**

```python
# Bad
raise Exception("Embedding failed")

# Good
raise EmbeddingError(
    "Failed to embed outputs using sentence-transformers/all-MiniLM-L6-v2. "
    "Please install: pip install sentence-transformers"
)
```

---

### V2 ⏭️

* Automatic fallback to alternative embedding models
* Partial result recovery (analyze whatever succeeded)
* Detailed error telemetry
* Suggested fixes for common errors

---

## 9. Explicit Non-Goals (MVP)

Items explicitly OUT OF SCOPE for MVP:

❌ **Web dashboards** - CLI only  
❌ **Persistent storage** - Results stored locally only  
❌ **Enterprise auth/SSO** - Single-user tool  
❌ **CI/CD plugins** - Manual invocation only  
❌ **Custom scoring DSLs** - Fixed formula  
❌ **Large evaluation datasets** - Single query at a time  
❌ **Multi-user collaboration** - Individual developer tool  
❌ **Historical trend analysis** - Before/after comparison only  
❌ **Automated remediation** - Suggestions only, no auto-fix  
❌ **Model fine-tuning recommendations** - Prompt-level fixes only

**Why These Are Out:**

Each adds 2-4 weeks of development time without validating core value proposition. MVP must prove: "Can we detect instability and explain it?" Everything else is optimization.

---

## 10. Design Principles

### Core Beliefs (Non-Negotiable)

1. **Instability Before Correctness**
   > Measure consistency first, accuracy second. A reliably wrong system can be fixed. An unpredictably right system cannot.

2. **Developer Experience > Completeness**
   > 10-minute setup with 80% coverage beats 2-hour setup with 100% coverage.

3. **Explain, Don't Just Score**
   > "62% stable" is useless. "Tool routing nondeterministic" is actionable.

4. **CLI Before GUI**
   > Developers live in terminals. Meet them where they are.

5. **Open Source Core, Commercial Extensions**
   > MVP is OSS. Governance, compliance, team features are paid.

### Success Metrics (MVP)

**Technical:**
- Run 20 iterations in < 60 seconds
- Detect tool variance with 95%+ accuracy
- Zero false positives on claim detection

**User:**
- Developer runs tool in < 10 minutes from install
- Flags instability they didn't expect
- Provides actionable fix they can apply immediately
- Re-run shows measurable improvement

**Business:**
- 1000+ GitHub stars in first month
- 10+ "This saved me from a prod bug" testimonials
- 100+ weekly active users

---

## 11. Implementation Phases

### Phase 1: Core Pipeline (Week 1)
- CLI argument parsing
- LLM API adapters (OpenAI, Anthropic)
- Repeated execution engine
- Basic progress display

### Phase 2: Metrics (Week 2)
- Semantic similarity (sentence-transformers)
- Tool consistency measurement
- Structural consistency detection
- Length variance calculation
- Overall stability score

### Phase 3: Analysis (Week 2)
- Claim pattern detection
- Instability fingerprinting
- Root cause identification

### Phase 4: Recommendations (Week 3)
- Decision tree implementation
- Recommendation generation
- Output formatting

### Phase 5: Polish (Week 3-4)
- Error handling
- Before/after comparison
- Terminal UX improvements
- Documentation

### Phase 6: Launch (Week 4)
- Package for PyPI
- Write README with examples
- Create demo video
- GitHub/HN launch

---

## Appendix A: Complete Example

### Input Files

**system_prompt.txt:**
```
You are a flight booking assistant.
Help users find and book flights.
Use the search_flights tool when available.
```

**query.json:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Find me a flight to NYC tomorrow"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_flights",
        "description": "Search for available flights"
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Command

```bash
airel stability \
  --prompt system_prompt.txt \
  --input query.json \
  --model gpt-4-turbo \
  --runs 20
```

### Output

*(See Section 7 for full output example)*

---

## Appendix B: Thresholds Summary

Quick reference for all classification thresholds:

| Metric | LOW | MEDIUM | HIGH |
|--------|-----|--------|------|
| **Semantic** | ≥85% | 70-84% | <70% |
| **Tool** | ≥95% | 80-94% | <80% |
| **Structural** | ≥95% | 85-94% | <85% |
| **Length** | ≥80% | 60-79% | <60% |
| **Claim Variance** | 0-19% or 81-100% | N/A | 20-80% |

**Overall Risk:**
- **SAFE**: Stability ≥90%
- **RISKY**: Stability 70-89%
- **DO NOT SHIP**: Stability <70%

---

## Appendix C: Embedding Model Details

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Specs:**
- Dimensions: 384
- Max sequence length: 256 tokens
- Performance: ~3000 sentences/second on CPU
- Size: 80MB download

**Installation:**
```bash
pip install sentence-transformers
```

**Why This Model:**
- Fast enough for CLI (no GPU required)
- Good general-purpose performance
- Wide adoption (trusted)
- Small download size

**Alternative for V2:**
- `all-mpnet-base-v2` (better quality, slower)
- Domain-specific models (code, medical, legal)

---

## Document Changelog

**v1.0 (2025-12-30) - Production Ready**
- Added concrete thresholds for all metrics
- Specified pattern-based claim detection
- Added comprehensive error handling
- Included complete code examples
- Added before/after comparison logic
- Clarified tool call matching rules
- Fixed semantic score scaling
- Added length variance edge cases

---

**✅ This specification is now LOCKED and ready for implementation.**

Questions? Create an issue or ping the team.
