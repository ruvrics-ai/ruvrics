"""
Command-line interface for Ruvrics.

Provides the main `ruvrics stability` command.
"""

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console

from ruvrics.config import get_config, get_model_config
from ruvrics.core.executor import StabilityExecutor
from ruvrics.core.models import InputConfig
from ruvrics.analysis.scorer import calculate_stability
from ruvrics.output.formatter import print_stability_report
from ruvrics.output.markdown_report import generate_markdown_report
from ruvrics.utils.errors import (
    RuvricsError,
    ConfigurationError,
    InsufficientDataError,
    EmbeddingError,
)
from ruvrics.utils.telemetry import track_stability_run, track_error


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="ruvrics")
def main():
    """
    Ruvrics - AI Behavioral Stability & Reliability Engine

    Measure whether an LLM system behaves consistently under identical conditions.
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
    help="Number of identical runs (minimum: 15, maximum: 50, default: 20)",
)
@click.option(
    "--tools",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to tools/functions JSON file (optional)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Save results to JSON file",
)
def stability(prompt, input, model, runs, tools, output):
    """
    Run stability analysis on an LLM system.

    This command executes N identical requests to test consistency.

    Examples:
        # Simple query without tools
        ruvrics stability \\
            --input query.json \\
            --model gpt-4-turbo \\
            --runs 20

        # With system prompt and tools
        ruvrics stability \\
            --prompt system_prompt.txt \\
            --input query.json \\
            --tools tools.json \\
            --model gpt-4-turbo \\
            --runs 20
    """
    try:
        # Validate runs parameter
        if not 10 <= runs <= 50:
            raise click.BadParameter("Runs must be between 10 and 50")

        # Load configuration
        config = get_config()
        model_config = get_model_config(model)

        # Validate runs against minimum required for analysis
        if runs < config.min_successful_runs:
            raise click.BadParameter(
                f"Runs must be at least {config.min_successful_runs} for reliable analysis. "
                f"You specified {runs}. Please use --runs {config.min_successful_runs} or higher."
            )

        console.print()
        console.print("🔍 AI Stability Analysis", style="bold cyan")
        console.print(f"Model: {model} ({model_config.provider})")
        console.print(f"Runs: {runs}")
        console.print()

        # Load input file
        with open(input, "r") as f:
            input_data = json.load(f)

        # Load prompt file if provided
        if prompt:
            with open(prompt, "r") as f:
                prompt_text = f.read().strip()
            # Add prompt to input data
            if "system_prompt" not in input_data:
                input_data["system_prompt"] = prompt_text

        # Load tools file if provided
        if tools:
            with open(tools, "r") as f:
                tools_data = json.load(f)
            # Add tools to input data
            if "tools" not in input_data:
                input_data["tools"] = tools_data

        # Create input configuration
        input_config = InputConfig(**input_data)

        # Create executor
        executor = StabilityExecutor(model=model, runs=runs, config=config)

        # Execute runs (with progress bar handled inside)
        start_time = time.time()
        run_results = executor.run(input_config)
        duration = time.time() - start_time

        console.print()
        console.print(
            f"✓ Completed {len(run_results)} runs in {duration:.1f}s",
            style="green",
        )
        console.print()

        # Calculate stability (returns result + embeddings for clustering)
        result, embeddings = calculate_stability(
            runs=run_results,
            input_config=input_config,
            model=model,
            duration_seconds=duration,
            config=config,
        )

        # Track successful run (anonymous telemetry)
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

        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # Generate timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"ruvrics_report_{timestamp}"

        # Extract user query for report header
        user_query = input_config.user_input if hasattr(input_config, 'user_input') else None

        # Generate and save Markdown explainability report
        markdown_report = generate_markdown_report(result, run_results, embeddings, user_query)
        markdown_path = reports_dir / f"{base_filename}.md"
        markdown_path.write_text(markdown_report)
        console.print()
        console.print(f"📄 Explainability report saved to: {markdown_path}", style="dim")

        # Save JSON to file if requested
        if output:
            output_path = Path(output)
            result.save_to_file(str(output_path))
            console.print(f"📁 JSON results saved to: {output_path}", style="dim")
        else:
            # Always save JSON alongside markdown
            json_path = reports_dir / f"{base_filename}.json"
            result.save_to_file(str(json_path))
            console.print(f"📁 JSON results saved to: {json_path}", style="dim")

        # Exit with appropriate code based on risk
        if result.risk_classification == "SAFE":
            sys.exit(0)
        elif result.risk_classification == "RISKY":
            sys.exit(1)
        else:  # DO_NOT_SHIP
            sys.exit(2)

    except ConfigurationError as e:
        track_error("ConfigurationError", "stability")
        console.print()
        console.print(f"❌ Configuration Error:", style="bold red")
        console.print(f"   {e}")
        console.print()
        console.print("💡 Tip: Make sure API keys are set in environment or .env file")
        sys.exit(3)

    except InsufficientDataError as e:
        track_error("InsufficientDataError", "stability")
        console.print()
        console.print(f"❌ Insufficient Data:", style="bold red")
        console.print(f"   {e}")
        console.print()
        sys.exit(4)

    except EmbeddingError as e:
        track_error("EmbeddingError", "stability")
        console.print()
        console.print(f"❌ Embedding Error:", style="bold red")
        console.print(f"   {e}")
        console.print()
        sys.exit(5)

    except json.JSONDecodeError as e:
        track_error("JSONDecodeError", "stability")
        console.print()
        console.print(f"❌ Invalid JSON:", style="bold red")
        console.print(f"   {input}: {e}")
        console.print()
        sys.exit(6)

    except RuvricsError as e:
        track_error("RuvricsError", "stability")
        console.print()
        console.print(f"❌ Error:", style="bold red")
        console.print(f"   {e}")
        console.print()
        sys.exit(7)

    except Exception as e:
        track_error("UnexpectedError", "stability")
        console.print()
        console.print(f"❌ Unexpected Error:", style="bold red")
        console.print(f"   {e}")
        console.print()
        console.print("Please report this issue at: https://github.com/yourusername/ruvrics/issues")
        sys.exit(8)


if __name__ == "__main__":
    main()
