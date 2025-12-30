# Implementation Instructions for AI Stability CLI

**For:** Claude Code / AI Coding Assistant  
**Project:** AI Behavioral Stability & Reliability Engine  
**Language:** Python 3.9+  
**Timeline:** 4 weeks  

---

## 🎯 Project Overview

Build a CLI tool that stress-tests LLM systems by running identical requests multiple times, measures output consistency, identifies root causes of instability, and provides actionable recommendations.

**Core Value:** Help developers decide if their LLM system is safe to ship.

---

## 📋 Before You Start

**Read These Documents First (in order):**

1. **AI_STABILITY_FINAL_SPEC.md** - This is the authoritative specification
   - Read the entire document before writing any code
   - All formulas, thresholds, and logic are defined here
   - Follow it exactly - do not deviate without asking

2. **Key Principles:**
   - Build for MVP first (ignore all V2 features)
   - CLI-only (no web UI)
   - Developer experience > completeness
   - Make it work, then make it fast

---

## 🏗️ Project Structure

Create this exact directory structure:

```
airel/
├── pyproject.toml              # Project metadata, dependencies
├── README.md                   # User-facing documentation
├── .gitignore                  # Git ignore file
├── airel/
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # Entry point for `python -m airel`
│   ├── cli.py                 # CLI argument parsing (Click)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── executor.py        # Run N identical LLM requests
│   │   ├── models.py          # Data classes for runs, results
│   │   └── adapters.py        # OpenAI/Anthropic API adapters
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── semantic.py        # Semantic similarity measurement
│   │   ├── tool.py            # Tool consistency measurement
│   │   ├── structural.py      # Structural consistency
│   │   ├── length.py          # Length variance
│   │   └── claims.py          # Claim pattern detection
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── scorer.py          # Overall stability score calculation
│   │   ├── fingerprint.py     # Root cause identification
│   │   └── recommender.py     # Generate actionable recommendations
│   ├── output/
│   │   ├── __init__.py
│   │   ├── formatter.py       # Terminal output formatting
│   │   └── comparison.py      # Before/after comparison
│   └── utils/
│       ├── __init__.py
│       ├── errors.py          # Custom exceptions
│       └── persistence.py     # Save/load previous results
├── tests/
│   ├── __init__.py
│   ├── test_executor.py
│   ├── test_metrics.py
│   ├── test_analysis.py
│   └── fixtures/              # Test data
└── examples/
    ├── system_prompt.txt
    └── query.json
```

---

## 📦 Dependencies (pyproject.toml)

Use these exact dependencies for MVP:

```toml
[project]
name = "airel"
version = "0.1.0"
description = "AI Behavioral Stability & Reliability Engine"
requires-python = ">=3.9"
dependencies = [
    "click>=8.1.0",              # CLI framework
    "openai>=1.0.0",             # OpenAI API
    "anthropic>=0.18.0",         # Anthropic API
    "sentence-transformers>=2.2.0",  # Embeddings
    "numpy>=1.24.0",             # Numerical operations
    "rich>=13.0.0",              # Terminal formatting
    "pydantic>=2.0.0",           # Data validation
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
airel = "airel.cli:main"

[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"
```

---

## 🔧 Implementation Phases

### Phase 1: Core Infrastructure (Days 1-3)

**Goal:** Get basic pipeline working end-to-end.

#### Step 1.1: Project Setup
```bash
# Create structure
mkdir -p airel/airel/{core,metrics,analysis,output,utils}
mkdir -p tests examples

# Initialize git
git init
```

#### Step 1.2: Data Models (airel/core/models.py)

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ToolCall(BaseModel):
    name: str
    call_sequence: int

class RunResult(BaseModel):
    run_id: int
    timestamp: datetime
    output_text: str
    tool_calls: List[ToolCall]
    output_length_tokens: int
    output_length_chars: int
    output_structure: str  # "json", "markdown", "text"
    api_latency_ms: int
    model_used: str
    error: Optional[str] = None

class InputConfig(BaseModel):
    system_prompt: Optional[str] = None
    user_input: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: str = "auto"
    
class StabilityResult(BaseModel):
    stability_score: float
    risk_classification: str
    semantic_consistency_score: float
    semantic_drift: str
    tool_consistency_score: float
    tool_variance: str
    structural_consistency_score: float
    structural_variance: str
    length_consistency_score: float
    length_variance: str
    claim_variance: Optional[str] = None
    risky_claim_percentage: Optional[float] = None
    root_causes: List[Dict[str, Any]]
    recommendations: List[str]
    runs: List[RunResult]
```

#### Step 1.2b Before implementing adapters, create airel/config.py with:

Config class (Pydantic BaseModel)
Model registry (SUPPORTED_MODELS dict)
Singleton pattern (get_config function)
Environment variable loading (.env support)

#### Step 1.3: CLI Interface (airel/cli.py)

```python
import click
from pathlib import Path
from airel.core.executor import StabilityExecutor

@click.command()
@click.option('--prompt', type=click.Path(exists=True), required=True,
              help='Path to system prompt file')
@click.option('--input', type=click.Path(exists=True), required=True,
              help='Path to input JSON file')
@click.option('--model', type=str, required=True,
              help='Model identifier (e.g., gpt-4-turbo)')
@click.option('--runs', type=int, default=20,
              help='Number of identical runs (default: 20)')
@click.option('--provider', type=click.Choice(['openai', 'anthropic']),
              default='openai', help='LLM provider')
def stability(prompt, input, model, runs, provider):
    """Run stability analysis on an LLM system."""
    
    click.echo("🔍 AI Stability Analysis")
    click.echo(f"Model: {model} | Runs: {runs}")
    click.echo()
    
    # Load files
    with open(prompt) as f:
        prompt_text = f.read()
    
    with open(input) as f:
        input_data = json.load(f)
    
    # Execute
    executor = StabilityExecutor(
        model=model,
        provider=provider,
        runs=runs
    )
    
    result = executor.run(prompt_text, input_data)
    
    # Format and display
    from airel.output.formatter import format_report
    report = format_report(result)
    click.echo(report)

def main():
    stability()

if __name__ == '__main__':
    main()
```

#### Step 1.4: LLM API Adapters (airel/core/adapters.py)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import openai
import anthropic

class LLMAdapter(ABC):
    @abstractmethod
    def call(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM and return standardized response"""
        pass

class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str):
        self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
        self.model = model
    
    def call(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Convert config to OpenAI format
        messages = config.get("messages", [])
        tools = config.get("tools")
        
        params = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = config.get("tool_choice", "auto")
        
        response = self.client.chat.completions.create(**params)
        
        # Standardize response
        return self._standardize_response(response)
    
    def _standardize_response(self, response):
        # Extract tool calls, text, tokens, etc.
        # Return in our RunResult format
        pass

class AnthropicAdapter(LLMAdapter):
    def __init__(self, model: str):
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY
        self.model = model
    
    def call(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Similar to OpenAI but for Anthropic
        pass
```

#### Step 1.5: Executor (airel/core/executor.py)

```python
import time
from typing import List
from airel.core.models import RunResult, StabilityResult
from airel.core.adapters import OpenAIAdapter, AnthropicAdapter
from rich.progress import Progress

class StabilityExecutor:
    def __init__(self, model: str, provider: str, runs: int):
        self.model = model
        self.runs = runs
        
        if provider == "openai":
            self.adapter = OpenAIAdapter(model)
        else:
            self.adapter = AnthropicAdapter(model)
    
    def run(self, prompt: str, input_data: dict) -> StabilityResult:
        """Execute N identical runs and analyze results"""
        
        # Prepare config
        config = self._prepare_config(prompt, input_data)
        
        # Execute runs with progress bar
        results = []
        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Running stability test...", 
                total=self.runs
            )
            
            for i in range(self.runs):
                try:
                    result = self._execute_single_run(i + 1, config)
                    results.append(result)
                except Exception as e:
                    # Log error but continue
                    results.append(RunResult(
                        run_id=i + 1,
                        error=str(e),
                        # ... fill other required fields
                    ))
                
                progress.update(task, advance=1)
        
        # Check minimum successful runs
        successful = [r for r in results if r.error is None]
        if len(successful) < 15:
            raise InsufficientDataError(
                f"Only {len(successful)}/{self.runs} runs succeeded"
            )
        
        # Analyze results
        from airel.analysis.scorer import calculate_stability
        return calculate_stability(successful)
    
    def _execute_single_run(self, run_id: int, config: dict) -> RunResult:
        """Execute a single LLM call with retry logic"""
        
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                start = time.time()
                response = self.adapter.call(config)
                latency = int((time.time() - start) * 1000)
                
                return RunResult(
                    run_id=run_id,
                    timestamp=datetime.now(),
                    output_text=response["text"],
                    tool_calls=response["tool_calls"],
                    output_length_tokens=response["tokens"],
                    # ... etc
                )
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
```

---

### Phase 2: Metrics Implementation (Days 4-7)

Implement each metric file according to Section 3 of the spec.

#### airel/metrics/semantic.py

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class SemanticAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_consistency(self, outputs: List[str]) -> dict:
        """
        Calculate semantic consistency using centroid method.
        See spec Section 3.1 for exact formula.
        """
        
        # Embed all outputs
        embeddings = self.model.encode(outputs)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate similarities to centroid
        similarities = [
            np.dot(emb, centroid) / (
                np.linalg.norm(emb) * np.linalg.norm(centroid)
            )
            for emb in embeddings
        ]
        
        # Mean similarity (raw score)
        semantic_raw_score = np.mean(similarities)
        
        # Convert to 0-100 scale
        semantic_consistency_score = semantic_raw_score * 100
        
        # Classify drift
        if semantic_consistency_score >= 85:
            drift = "LOW"
        elif semantic_consistency_score >= 70:
            drift = "MEDIUM"
        else:
            drift = "HIGH"
        
        return {
            "semantic_consistency_score": semantic_consistency_score,
            "semantic_drift": drift,
            "raw_similarities": similarities
        }
```

#### airel/metrics/tool.py

```python
from collections import Counter
from typing import List, FrozenSet

def normalize_tool_calls(tool_calls: List[dict]) -> FrozenSet[str]:
    """Extract unique tool names, order-independent"""
    return frozenset(call["name"] for call in tool_calls)

def calculate_tool_consistency(runs: List[RunResult]) -> dict:
    """
    Calculate tool usage consistency.
    See spec Section 3.2 for exact formula.
    """
    
    # Check if tools were available
    # (this info should be in the run context)
    
    # Normalize each run's tool calls
    tool_patterns = [normalize_tool_calls(run.tool_calls) for run in runs]
    
    # Find most common pattern
    pattern_counts = Counter(tool_patterns)
    most_common_pattern, mode_count = pattern_counts.most_common(1)[0]
    
    # Calculate consistency
    tool_consistency_score = (mode_count / len(runs)) * 100
    
    # Classify variance
    if tool_consistency_score >= 95:
        variance = "LOW"
    elif tool_consistency_score >= 80:
        variance = "MEDIUM"
    else:
        variance = "HIGH"
    
    return {
        "tool_consistency_score": tool_consistency_score,
        "tool_variance": variance,
        "most_common_pattern": most_common_pattern,
        "pattern_distribution": dict(pattern_counts)
    }
```

**Continue similarly for:**
- `airel/metrics/structural.py` (Section 3.3)
- `airel/metrics/length.py` (Section 3.4)
- `airel/metrics/claims.py` (Section 5)

---

### Phase 3: Analysis & Recommendations (Days 8-10)

#### airel/analysis/scorer.py

```python
from airel.metrics import semantic, tool, structural, length, claims

def calculate_stability(runs: List[RunResult]) -> StabilityResult:
    """
    Calculate overall stability score.
    See spec Section 3 for exact formula.
    """
    
    # Calculate all metrics
    outputs = [r.output_text for r in runs]
    
    semantic_result = semantic.calculate_consistency(outputs)
    tool_result = tool.calculate_tool_consistency(runs)
    structural_result = structural.calculate_structural_consistency(runs)
    length_result = length.calculate_length_consistency(runs)
    claim_result = claims.analyze_claim_instability(runs)
    
    # Overall score (weighted average)
    stability_score = (
        0.40 * semantic_result["semantic_consistency_score"] +
        0.25 * tool_result["tool_consistency_score"] +
        0.20 * structural_result["structural_consistency_score"] +
        0.15 * length_result["length_consistency_score"]
    )
    
    # Apply claim penalty if HIGH
    if claim_result.get("claim_variance") == "HIGH":
        stability_score *= 0.80
    
    # Risk classification
    if stability_score >= 90:
        risk = "SAFE"
    elif stability_score >= 70:
        risk = "RISKY"
    else:
        risk = "DO_NOT_SHIP"
    
    # Identify root causes
    from airel.analysis.fingerprint import identify_root_cause
    root_causes = identify_root_cause({
        **semantic_result,
        **tool_result,
        **structural_result,
        **length_result,
        **claim_result
    })
    
    # Generate recommendations
    from airel.analysis.recommender import generate_recommendations
    recommendations = generate_recommendations(root_causes)
    
    return StabilityResult(
        stability_score=stability_score,
        risk_classification=risk,
        # ... all other fields
    )
```

#### airel/analysis/fingerprint.py

Implement the root cause decision tree from Section 4 of the spec.

#### airel/analysis/recommender.py

Implement the recommendation decision tree from Section 6 of the spec.

---

### Phase 4: Output Formatting (Days 11-12)

#### airel/output/formatter.py

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from airel.core.models import StabilityResult

def format_report(result: StabilityResult) -> str:
    """
    Format stability results for terminal output.
    See spec Section 7 for exact format.
    """
    
    console = Console()
    
    # Header
    console.print("\n" + "━" * 70)
    console.print("AI STABILITY REPORT", style="bold cyan")
    console.print("━" * 70)
    
    # Overall score
    score_color = "green" if result.risk_classification == "SAFE" else "red"
    console.print(f"\nOverall Stability Score: {result.stability_score:.1f}%", 
                  style=f"bold {score_color}")
    console.print(f"Risk: {result.risk_classification}\n")
    
    # Component breakdown table
    table = Table(show_header=True)
    table.add_column("Component")
    table.add_column("Score")
    table.add_column("Status")
    
    table.add_row(
        "Semantic Consistency",
        f"{result.semantic_consistency_score:.1f}%",
        result.semantic_drift
    )
    # ... add other rows
    
    console.print(table)
    
    # Recommendations
    if result.recommendations:
        console.print("\n" + "━" * 70)
        console.print("RECOMMENDED FIXES", style="bold yellow")
        console.print("━" * 70)
        
        for i, rec in enumerate(result.recommendations, 1):
            console.print(f"\n{i}. {rec}")
    
    return console.export_text()
```

---

### Phase 5: Testing & Polish (Days 13-15)

#### tests/test_executor.py

```python
import pytest
from airel.core.executor import StabilityExecutor

def test_executor_basic():
    """Test basic execution flow"""
    executor = StabilityExecutor(
        model="gpt-3.5-turbo",
        provider="openai",
        runs=5  # Small number for testing
    )
    
    result = executor.run(
        prompt="You are helpful",
        input_data={"user_input": "Say hello"}
    )
    
    assert result.stability_score >= 0
    assert result.stability_score <= 100

# Add more tests for each component
```

---

## 🚨 Critical Implementation Rules

### 1. Follow the Spec EXACTLY
- Every formula in Section 3 must be implemented as written
- Every threshold in Section 4 must be used exactly
- Do not "improve" or "optimize" the formulas without asking

### 2. Error Handling (Section 8)
- Implement retry logic with exponential backoff
- Handle rate limits gracefully
- Require minimum 15/20 successful runs
- Show clear error messages

### 3. Code Quality
- Use type hints everywhere
- Add docstrings to all functions
- Keep functions under 50 lines
- Extract magic numbers to constants

### 4. Testing
- Write tests for each metric calculator
- Use pytest fixtures for sample data
- Aim for 80%+ code coverage

---

## 📝 Step-by-Step Execution Plan

**Day 1-2:**
1. Set up project structure
2. Implement data models (models.py)
3. Implement CLI interface (cli.py)
4. Get basic "hello world" working

**Day 3-4:**
5. Implement LLM adapters (adapters.py)
6. Implement executor with retry logic (executor.py)
7. Test with real API calls

**Day 5-7:**
8. Implement all metric calculators (metrics/*.py)
9. Test each metric independently
10. Validate against spec thresholds

**Day 8-9:**
11. Implement scorer (scorer.py)
12. Implement fingerprinting (fingerprint.py)
13. Test overall scoring

**Day 10-11:**
14. Implement recommendation engine (recommender.py)
15. Test decision tree logic

**Day 12-13:**
16. Implement output formatter (formatter.py)
17. Add rich terminal formatting
18. Test with various result types

**Day 14-15:**
19. Write comprehensive tests
20. Add error handling everywhere
21. Polish UX

**Day 16-17:**
22. Write README with examples
23. Add example files
24. Package for PyPI

**Day 18-20:**
25. Manual testing with real prompts
26. Bug fixes
27. Performance optimization

---

## 🎬 Getting Started Command

```bash
# 1. Create project directory
mkdir airel && cd airel

# 2. Read the spec
# (Make sure AI_STABILITY_FINAL_SPEC.md is available)

# 3. Start with Phase 1, Step 1.1
# Create the directory structure

# 4. Ask me questions if anything in the spec is unclear

# 5. Show me your implementation plan before starting

# 6. Implement incrementally - test each component before moving on
```

---

## ❓ When to Ask Questions

**Ask me BEFORE coding if:**
- Spec seems ambiguous or contradictory
- You want to use a different approach than specified
- You encounter an edge case not covered in spec
- You want to add a feature not in MVP scope

**Don't ask, just implement:**
- Code organization within guidelines
- Variable naming
- Import organization
- Minor code style choices

---

## ✅ Definition of Done

MVP is complete when:

1. ✅ CLI runs: `airel stability --prompt x --input y --model z --runs 20`
2. ✅ Executes 20 identical LLM calls
3. ✅ Calculates all 4 metric scores correctly
4. ✅ Identifies root causes using decision tree
5. ✅ Generates 2-3 actionable recommendations
6. ✅ Displays formatted terminal output
7. ✅ Handles API errors gracefully
8. ✅ Completes in under 60 seconds (for N=20)
9. ✅ Has tests for critical paths
10. ✅ Can be installed via `pip install -e .`

---

## 📚 Reference Documents

1. **AI_STABILITY_FINAL_SPEC.md** - THE authoritative source
2. This document - Implementation guidance
3. Python best practices - Use standard conventions

---

## 🎯 Success Criteria

**Technical:**
- Zero crashes on happy path
- Handles 90%+ of error cases gracefully
- Produces deterministic results (same input → same score)

**User Experience:**
- Setup in < 10 minutes
- Clear error messages
- Beautiful terminal output
- Actionable recommendations

**Code Quality:**
- Type hints everywhere
- Docstrings on public functions
- Tests for critical logic
- No hard-coded values (use constants)

---

**Ready to start? Read the spec, then begin with Phase 1, Step 1.1!**

If you have any questions about the spec or implementation approach, ask me before writing code.
