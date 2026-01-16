"""
Command-line interface for Ruvrics.

Provides the main `ruvrics stability` command with baseline comparison support.
"""

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ruvrics.config import get_config, get_model_config
from ruvrics.core.executor import StabilityExecutor
from ruvrics.core.models import InputConfig
from ruvrics.analysis.scorer import calculate_stability
from ruvrics.output.formatter import print_stability_report
from ruvrics.output.markdown_report import generate_markdown_report
from ruvrics.comparison import (
    save_baseline as save_baseline_func,
    load_baseline,
    list_baselines,
    compare_to_baseline,
    compare_models,
    ComparisonResult,
)
from ruvrics.utils.errors import (
    RuvricsError,
    ConfigurationError,
    APIKeyMissingError,
    InvalidAPIKeyError,
    InsufficientDataError,
    EmbeddingError,
    JSONParseError,
    InvalidInputError,
    ModelNotSupportedError,
    ToolMockRequiredError,
)
from ruvrics.utils.telemetry import track_stability_run, track_error


console = Console()


def print_error(title: str, error: Exception, show_full_message: bool = True):
    """
    Print a formatted error message to console.

    Args:
        title: Short error title (e.g., "API Key Missing")
        error: The exception to display
        show_full_message: If True, shows the full multi-line error message
    """
    console.print()
    console.print("=" * 70, style="red")
    console.print(f"❌ {title}", style="bold red")
    console.print("=" * 70, style="red")
    console.print()

    if show_full_message:
        # Print the full error message with proper indentation
        error_lines = str(error).split("\n")
        for line in error_lines:
            console.print(f"  {line}")
    else:
        console.print(f"  {error}")

    console.print()
    console.print("=" * 70, style="red")


def print_comparison_report(comparison: ComparisonResult):
    """Print comparison results to console."""
    console.print()
    console.print("=" * 70)
    console.print(f"COMPARISON TO BASELINE ({comparison.baseline_name})", style="bold", justify="center")
    console.print("=" * 70)
    console.print()

    # Score comparison
    delta_str = f"{comparison.score_delta:+.1f}%"
    if comparison.score_delta > 0:
        delta_style = "green"
    elif comparison.score_delta < -2:
        delta_style = "red"
    else:
        delta_style = "yellow"

    console.print(f"Baseline Score: {comparison.baseline_score:.1f}% → Current: {comparison.current_score:.1f}%  ({delta_str})", style=delta_style)
    console.print()

    # Status
    status_styles = {
        "IMPROVED": ("green", "✅ IMPROVED"),
        "STABLE": ("green", "✅ STABLE"),
        "MINOR_REGRESSION": ("yellow", "⚠️ MINOR REGRESSION"),
        "MAJOR_REGRESSION": ("red", "❌ MAJOR REGRESSION"),
    }
    style, status_text = status_styles.get(comparison.status, ("white", comparison.status))
    console.print(f"Status: {status_text}", style=style)
    console.print()

    # Changes
    if comparison.changes:
        console.print("Changes Detected:", style="bold")
        for change in comparison.changes:
            console.print(f"  • {change}")

    console.print()
    console.print("=" * 70)


def print_model_comparison(comparison: dict):
    """Print model comparison results to console."""
    console.print()
    console.print("=" * 70)
    console.print("MODEL COMPARISON", style="bold", justify="center")
    console.print("=" * 70)
    console.print()

    # Create comparison table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column(comparison["model1"]["name"], justify="right")
    table.add_column(comparison["model2"]["name"], justify="right")
    table.add_column("Delta", justify="right")

    # Overall score
    delta = comparison["comparison"]["score_delta"]
    delta_str = f"{delta:+.1f}%"
    table.add_row(
        "Overall Stability",
        f"{comparison['model1']['stability_score']:.1f}%",
        f"{comparison['model2']['stability_score']:.1f}%",
        delta_str,
    )

    # Component scores
    table.add_row(
        "Semantic",
        f"{comparison['model1']['semantic']:.1f}%",
        f"{comparison['model2']['semantic']:.1f}%",
        f"{comparison['comparison']['semantic_delta']:+.1f}%",
    )
    table.add_row(
        "Tool Usage",
        f"{comparison['model1']['tool']:.1f}%",
        f"{comparison['model2']['tool']:.1f}%",
        f"{comparison['comparison']['tool_delta']:+.1f}%",
    )
    table.add_row(
        "Structure",
        f"{comparison['model1']['structural']:.1f}%",
        f"{comparison['model2']['structural']:.1f}%",
        f"{comparison['comparison']['structural_delta']:+.1f}%",
    )
    table.add_row(
        "Length",
        f"{comparison['model1']['length']:.1f}%",
        f"{comparison['model2']['length']:.1f}%",
        f"{comparison['comparison']['length_delta']:+.1f}%",
    )

    console.print(table)
    console.print()

    # Summary
    console.print(f"Summary: {comparison['comparison']['summary']}", style="bold")
    console.print()
    console.print("=" * 70)


@click.group()
@click.version_option(version="0.2.0", prog_name="ruvrics")
def main():
    """
    Ruvrics - Catch behavioral regressions in LLM systems

    Detect when your AI system's behavior silently changes.
    """
    pass


@main.command()
@click.option(
    "--prompt",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to system prompt file (plain text)",
)
@click.option(
    "--input",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to input JSON file (query/messages)",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model identifier (e.g., gpt-4-turbo, claude-sonnet-4)",
)
@click.option(
    "--runs",
    type=int,
    default=20,
    help="Number of identical runs (10-50, default: 20)",
)
@click.option(
    "--tools",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to tools/functions JSON file",
)
@click.option(
    "--tool-mocks",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to tool mock responses JSON file (required for tool testing)",
)
@click.option(
    "--save-baseline",
    type=str,
    help="Save results as a named baseline (e.g., 'v1.0', 'prod')",
)
@click.option(
    "--compare",
    type=str,
    help="Compare results to a saved baseline",
)
@click.option(
    "--compare-model",
    type=str,
    help="Compare to a different model (runs tests on both)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Save results to JSON file",
)
def stability(prompt, input, model, runs, tools, tool_mocks, save_baseline, compare, compare_model, output):
    """
    Run stability analysis on an LLM system.

    Examples:
        # Basic test
        ruvrics stability --input query.json --model gpt-4o-mini

        # Save baseline
        ruvrics stability --input query.json --model gpt-4o-mini --save-baseline v1.0

        # Compare to baseline
        ruvrics stability --input query.json --model gpt-4o-mini --compare v1.0

        # Compare models
        ruvrics stability --input query.json --model gpt-4o-mini --compare-model gpt-4o
    """
    try:
        # Load configuration
        config = get_config()
        model_config = get_model_config(model)

        # Validate runs
        if runs < config.min_successful_runs:
            raise click.BadParameter(
                f"Runs must be at least {config.min_successful_runs} for reliable analysis."
            )

        if not config.min_successful_runs <= runs <= config.max_successful_runs:
            raise click.BadParameter("Runs must be between 10 and 50")

        # Check if baseline exists when comparing
        if compare:
            baseline = load_baseline(compare)
            if baseline is None:
                available = list_baselines()
                if available:
                    console.print(f"❌ Baseline '{compare}' not found.", style="red")
                    console.print(f"Available baselines: {', '.join(available)}")
                else:
                    console.print(f"❌ No baselines saved yet. Use --save-baseline first.", style="red")
                sys.exit(3)

        console.print()
        console.print("🔍 AI Stability Analysis", style="bold cyan")
        console.print(f"Model: {model} ({model_config.provider})")
        console.print(f"Runs: {runs}")
        if compare:
            console.print(f"Comparing to baseline: {compare}")
        if compare_model:
            console.print(f"Comparing to model: {compare_model}")
        console.print()

        # Load input file
        with open(input, "r") as f:
            input_data = json.load(f)

        # Load prompt file if provided
        if prompt:
            with open(prompt, "r") as f:
                prompt_text = f.read().strip()
            if "system_prompt" not in input_data:
                input_data["system_prompt"] = prompt_text

        # Load tools file if provided
        if tools:
            with open(tools, "r") as f:
                tools_data = json.load(f)
            if "tools" not in input_data:
                input_data["tools"] = tools_data

        # Load tool mocks file if provided
        if tool_mocks:
            with open(tool_mocks, "r") as f:
                tool_mocks_data = json.load(f)
            if "tool_mock_responses" not in input_data:
                input_data["tool_mock_responses"] = tool_mocks_data

        # Create input configuration
        input_config = InputConfig(**input_data)

        # Execute primary model tests
        executor = StabilityExecutor(model=model, runs=runs, config=config)
        start_time = time.time()
        run_results = executor.run(input_config)
        duration = time.time() - start_time

        console.print()
        console.print(f"✓ Completed {len(run_results)} runs in {duration:.1f}s", style="green")
        console.print()

        # Calculate stability
        result, embeddings = calculate_stability(
            runs=run_results,
            input_config=input_config,
            model=model,
            duration_seconds=duration,
            config=config,
        )

        # Track telemetry
        track_stability_run(
            model=model,
            runs=runs,
            successful_runs=result.successful_runs,
            duration_seconds=duration,
            stability_score=result.stability_score,
            risk_classification=result.risk_classification,
            has_tools=input_config.tools is not None,
        )

        # Display report
        print_stability_report(result)

        # Compare to baseline if requested
        comparison_result = None
        if compare:
            comparison_result = compare_to_baseline(result, compare)
            if comparison_result:
                print_comparison_report(comparison_result)

        # Compare models if requested
        model_comparison = None
        if compare_model:
            console.print()
            console.print(f"🔄 Running comparison model: {compare_model}", style="cyan")

            # Run tests on second model
            model2_config = get_model_config(compare_model)
            executor2 = StabilityExecutor(model=compare_model, runs=runs, config=config)
            start_time2 = time.time()
            run_results2 = executor2.run(input_config)
            duration2 = time.time() - start_time2

            console.print(f"✓ Completed {len(run_results2)} runs in {duration2:.1f}s", style="green")

            # Calculate stability for second model
            result2, _ = calculate_stability(
                runs=run_results2,
                input_config=input_config,
                model=compare_model,
                duration_seconds=duration2,
                config=config,
            )

            # Compare models
            model_comparison = compare_models(result, result2, model, compare_model)
            print_model_comparison(model_comparison)

        # Save baseline if requested
        if save_baseline:
            baseline_path = save_baseline_func(result, save_baseline)
            console.print()
            console.print(f"💾 Baseline saved: {save_baseline}", style="green")
            console.print(f"   Path: {baseline_path}", style="dim")

        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"ruvrics_report_{timestamp}"

        # Extract user query for report
        user_query = input_config.user_input if hasattr(input_config, 'user_input') else None

        # Generate and save Markdown report
        markdown_report = generate_markdown_report(result, run_results, embeddings, user_query)
        markdown_path = reports_dir / f"{base_filename}.md"
        markdown_path.write_text(markdown_report)
        console.print()
        console.print(f"📄 Explainability report saved to: {markdown_path}", style="dim")

        # Save JSON
        if output:
            output_path = Path(output)
            result.save_to_file(str(output_path))
            console.print(f"📁 JSON results saved to: {output_path}", style="dim")
        else:
            json_path = reports_dir / f"{base_filename}.json"
            result.save_to_file(str(json_path))
            console.print(f"📁 JSON results saved to: {json_path}", style="dim")

        # Determine exit code
        exit_code = 0

        # Check for regression (takes precedence)
        if comparison_result:
            if comparison_result.status == "MAJOR_REGRESSION":
                exit_code = 2
            elif comparison_result.status == "MINOR_REGRESSION":
                exit_code = 1
        else:
            # Use risk classification
            if result.risk_classification == "DO_NOT_SHIP":
                exit_code = 2
            elif result.risk_classification == "RISKY":
                exit_code = 1

        sys.exit(exit_code)

    except APIKeyMissingError as e:
        track_error("APIKeyMissingError", "stability")
        print_error("API Key Not Found", e)
        sys.exit(3)

    except InvalidAPIKeyError as e:
        track_error("InvalidAPIKeyError", "stability")
        print_error("Invalid API Key", e)
        sys.exit(3)

    except ModelNotSupportedError as e:
        track_error("ModelNotSupportedError", "stability")
        print_error("Model Not Supported", e)
        sys.exit(3)

    except ToolMockRequiredError as e:
        track_error("ToolMockRequiredError", "stability")
        print_error("Tool Mocks Required", e)
        sys.exit(3)

    except ConfigurationError as e:
        track_error("ConfigurationError", "stability")
        print_error("Configuration Error", e)
        sys.exit(3)

    except InsufficientDataError as e:
        track_error("InsufficientDataError", "stability")
        print_error("Insufficient Data", e)
        sys.exit(4)

    except EmbeddingError as e:
        track_error("EmbeddingError", "stability")
        print_error("Embedding Error", e)
        sys.exit(5)

    except json.JSONDecodeError as e:
        track_error("JSONDecodeError", "stability")
        # Convert to our custom error for better messaging
        json_error = JSONParseError(
            file_path=input,
            error_detail=str(e.msg),
            line=e.lineno,
            column=e.colno
        )
        print_error("JSON Parse Error", json_error)
        sys.exit(6)

    except FileNotFoundError as e:
        track_error("FileNotFoundError", "stability")
        console.print()
        console.print("=" * 70, style="red")
        console.print("❌ File Not Found", style="bold red")
        console.print("=" * 70, style="red")
        console.print()
        console.print(f"  Could not find file: {e.filename}")
        console.print()
        console.print("  Please check:")
        console.print("    - The file path is correct")
        console.print("    - The file exists in the specified location")
        console.print("    - You have read permissions for the file")
        console.print()
        console.print("=" * 70, style="red")
        sys.exit(6)

    except RuvricsError as e:
        track_error("RuvricsError", "stability")
        print_error("Error", e)
        sys.exit(7)

    except Exception as e:
        track_error("UnexpectedError", "stability")
        console.print()
        console.print("=" * 70, style="red")
        console.print("❌ Unexpected Error", style="bold red")
        console.print("=" * 70, style="red")
        console.print()
        console.print(f"  {type(e).__name__}: {e}")
        console.print()
        console.print("  This might be a bug. Please report it at:")
        console.print("    https://github.com/ruvrics-ai/ruvrics/issues")
        console.print()
        console.print("  Include the full error message and steps to reproduce.")
        console.print()
        console.print("=" * 70, style="red")
        sys.exit(8)


@main.command()
def baselines():
    """List all saved baselines."""
    available = list_baselines()

    if not available:
        console.print("No baselines saved yet.")
        console.print()
        console.print("Save a baseline with:")
        console.print("  ruvrics stability --input query.json --model gpt-4o-mini --save-baseline v1.0")
        return

    console.print("Saved baselines:", style="bold")
    for name in available:
        baseline = load_baseline(name)
        if baseline:
            console.print(f"  • {name}: {baseline.stability_score:.1f}% ({baseline.risk_classification}) - {baseline.model}")


if __name__ == "__main__":
    main()
