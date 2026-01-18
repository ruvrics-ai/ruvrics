"""
Terminal output formatting for stability reports.

Creates beautiful, readable reports using Rich.
From spec Section 7 - Output Contract (CLI).
"""

import io

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ruvrics.core.models import StabilityResult


def format_stability_report(result: StabilityResult) -> str:
    """
    Format stability results for terminal output.

    From spec Section 7 - produces formatted report matching
    the exact layout specified in the documentation.

    Args:
        result: StabilityResult with complete analysis

    Returns:
        Formatted string ready for terminal display
    """
    # Create console that records but doesn't print to stdout
    string_buffer = io.StringIO()
    console = Console(file=string_buffer, record=True, force_terminal=True)

    # Header
    console.print()
    console.print("=" * 70, style="bold cyan")
    console.print("AI STABILITY REPORT", style="bold cyan", justify="center")
    console.print("=" * 70, style="bold cyan")

    # Metadata line
    console.print(
        f"Tested: {result.model} | "
        f"Runs: {result.successful_runs}/{result.total_runs} ✓ | "
        f"Duration: {result.duration_seconds:.1f}s"
    )
    console.print()

    # Overall score with color based on risk
    score_color = _get_risk_color(result.risk_classification)
    risk_emoji = _get_risk_emoji(result.risk_classification)
    console.print(
        f"Overall Stability Score: {result.stability_score:.1f}% "
        f"{risk_emoji} {result.risk_classification}",
        style=f"bold {score_color}",
    )
    console.print()

    # Consistency breakdown
    console.print("=" * 70, style="bold cyan")
    console.print("CONSISTENCY BREAKDOWN", style="bold cyan")
    console.print("=" * 70, style="bold cyan")
    console.print()

    _print_metric_line(
        console,
        "Response Consistency:",
        result.semantic_consistency_score,
        result.semantic_drift,
    )
    _print_metric_line(
        console,
        "Format Consistency:",
        result.structural_consistency_score,
        result.structural_variance,
    )
    _print_metric_line(
        console,
        "Tool Consistency:",
        result.tool_consistency_score,
        result.tool_variance,
    )
    _print_metric_line(
        console,
        "Length Consistency:",
        result.length_consistency_score,
        result.length_variance,
    )
    console.print()

    # Claim instability warning (if present)
    if result.claim_analysis and result.claim_analysis.claim_variance == "HIGH":
        console.print(
            f"⚠️  Claim Instability: {result.claim_analysis.claim_variance}",
            style="bold yellow",
        )
        console.print(
            f"    Risky claims detected in {result.claim_analysis.risky_percentage:.0f}% of runs"
        )
        if result.claim_analysis.examples:
            console.print("    Examples: ", end="")
            console.print(", ".join(f'"{ex}"' for ex in result.claim_analysis.examples[:3]))
        console.print()

    # Instability fingerprint
    if result.root_causes:
        console.print("=" * 70, style="bold cyan")
        console.print("INSTABILITY FINGERPRINT", style="bold cyan")
        console.print("=" * 70, style="bold cyan")
        console.print()

        primary_cause = result.root_causes[0]
        console.print(
            f"Primary Issue: {primary_cause.type} (Severity: {primary_cause.severity})",
            style="bold red" if primary_cause.severity in ["CRITICAL", "HIGH"] else "bold yellow",
        )
        console.print()
        console.print("Root Cause:", style="bold")
        console.print(f"  {primary_cause.description}")
        if primary_cause.details:
            console.print(f"  {primary_cause.details}")
        console.print()

    # Recommendations
    if result.recommendations:
        console.print("=" * 70, style="bold cyan")
        console.print("RECOMMENDED FIXES", style="bold cyan")
        console.print("=" * 70, style="bold cyan")
        console.print()

        for i, rec in enumerate(result.recommendations, 1):
            console.print(f"Priority {i}: {rec.title}", style="bold yellow")
            console.print(f"  → {rec.description}")
            if rec.example:
                console.print()
                # Indent example
                for line in rec.example.split("\n"):
                    console.print(f"    {line}", style="dim")
            console.print()

    # Next steps
    console.print("=" * 70, style="bold cyan")
    console.print("NEXT STEPS", style="bold cyan")
    console.print("=" * 70, style="bold cyan")
    console.print()

    if result.risk_classification == "SAFE":
        console.print("✅ This system is safe to ship!", style="bold green")
        console.print("   Stability score is high and consistent.")
    elif result.risk_classification == "RISKY":
        console.print(
            "⚠️  Review required before shipping.", style="bold yellow"
        )
        console.print("1. Apply recommended fixes above")
        console.print("2. Re-run stability test to validate improvements")
        console.print(
            "3. Aim for score >= 90% before deploying to production"
        )
    else:  # DO_NOT_SHIP
        console.print("❌ DO NOT SHIP - Critical issues detected", style="bold red")
        console.print("1. Apply recommended fixes (focus on Priority 1)")
        console.print("2. Re-run stability test")
        console.print("3. System must achieve score >= 70% minimum")

    console.print()
    console.print("=" * 70, style="bold cyan")

    # Export as text
    return console.export_text()


def _print_metric_line(
    console: Console, label: str, score: float, variance: str
) -> None:
    """Print a single metric line with progress bar."""
    # Create progress bar
    bar_width = 20
    filled = int((score / 100) * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Color based on variance
    variance_color = _get_variance_color(variance)
    status = _get_variance_status(variance)

    console.print(
        f"{label:<24} {score:>5.1f}% │ {bar} │ {status}",
        style=variance_color,
        highlight=False,
    )


def _get_risk_color(risk: str) -> str:
    """Get color for risk classification."""
    if risk == "SAFE":
        return "green"
    elif risk == "RISKY":
        return "yellow"
    else:  # DO_NOT_SHIP
        return "red"


def _get_risk_emoji(risk: str) -> str:
    """Get emoji for risk classification."""
    if risk == "SAFE":
        return "✅"
    elif risk == "RISKY":
        return "⚠️"
    else:  # DO_NOT_SHIP
        return "❌"


def _get_variance_color(variance: str) -> str:
    """Get color for variance classification."""
    if variance == "LOW" or variance == "N/A" or variance == "NONE":
        return "green"
    elif variance == "MEDIUM":
        return "yellow"
    else:  # HIGH
        return "red"


def _get_variance_status(variance: str) -> str:
    """Convert variance to user-friendly status."""
    if variance == "LOW" or variance == "N/A" or variance == "NONE":
        return "✅ Excellent"
    elif variance == "MEDIUM":
        return "⚠️ Good"
    else:  # HIGH
        return "❌ Needs Attention"


def print_stability_report(result: StabilityResult) -> None:
    """
    Print stability report directly to console.

    Args:
        result: StabilityResult with complete analysis
    """
    report = format_stability_report(result)
    print(report)
