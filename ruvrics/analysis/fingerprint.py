"""
Instability fingerprinting - root cause identification.

Implements decision tree to identify WHY the system is unstable.
From spec Section 4 - Instability Fingerprint.
"""

from ruvrics.config import Config, get_config
from ruvrics.core.models import RootCause


def identify_root_causes(
    semantic_drift: str,
    semantic_score: float,
    tool_variance: str,
    tool_score: float,
    tool_details: dict,
    argument_variance: str = "N/A",
    argument_score: float = 100.0,
    argument_details: dict | None = None,
    chain_variance: str = "N/A",
    chain_score: float = 100.0,
    chain_details: dict | None = None,
    structural_variance: str = "LOW",
    structural_score: float = 100.0,
    length_variance: str = "LOW",
    length_score: float = 100.0,
    claim_variance: str = "NONE",
    claim_percentage: float = 0.0,
    config: Config | None = None,
) -> list[RootCause]:
    """
    Identify root causes of instability using decision tree.

    From spec Section 4:
    Priority order (most specific → most general):
    1. Tool-use instability (highest priority)
    1b. Argument drift (v0.2.2 - same tool, different args)
    1c. Chain variance (v0.2.2 - sequence instability)
    2. Claim instability
    3. Semantic drift
    4. Structural variance
    5. Length variance (lowest priority)

    Args:
        All variance classifications and scores from metrics
        argument_variance: Argument consistency variance (v0.2.2)
        argument_score: Argument consistency score (v0.2.2)
        argument_details: Per-tool argument analysis (v0.2.2)
        chain_variance: Tool chain sequence variance (v0.2.2)
        chain_score: Tool chain consistency score (v0.2.2)
        chain_details: Sequence analysis (v0.2.2)
        config: Optional configuration

    Returns:
        List of RootCause objects, ordered by priority
    """
    cfg = config or get_config()
    root_causes = []
    argument_details = argument_details or {}
    chain_details = chain_details or {}

    # 1. Check tool-use instability (highest priority)
    if tool_variance == "HIGH":
        if semantic_drift == "LOW":
            # Model inconsistently decides whether to use tools
            # but outputs are semantically similar when tool not used
            tool_usage_pct = tool_details.get("tool_usage_percentage", 0)
            root_causes.append(
                RootCause(
                    type="NONDETERMINISTIC_TOOL_ROUTING",
                    severity="HIGH",
                    description="Model inconsistently decides whether to use tools",
                    details=f"Tool used in {tool_usage_pct:.0f}% of runs",
                )
            )
        else:
            # Tool usage affects semantic output unpredictably
            root_causes.append(
                RootCause(
                    type="TOOL_CONFUSION",
                    severity="HIGH",
                    description="Tool usage affects semantic output unpredictably",
                    details="Both tool usage and output meaning vary",
                )
            )

    # 1b. Check argument drift (v0.2.2) - same tool, different arguments
    if argument_variance in ["HIGH", "MEDIUM"] and tool_variance != "HIGH":
        tools_with_variation = argument_details.get("tools_with_variation", 0)
        tool_analysis = argument_details.get("tool_argument_analysis", {})

        # Build details about which tools have argument drift
        drift_details = []
        for tool_name, analysis in tool_analysis.items():
            if analysis.get("argument_variations", 1) > 1:
                variations = analysis.get("argument_variations", 0)
                drift_details.append(f"{tool_name}: {variations} variations")

        severity = "HIGH" if argument_variance == "HIGH" else "MEDIUM"
        root_causes.append(
            RootCause(
                type="ARGUMENT_DRIFT",
                severity=severity,
                description="Same tools called with inconsistent arguments across runs",
                details=f"{tools_with_variation} tool(s) with argument drift: " + ", ".join(drift_details[:3]),
            )
        )

    # 1c. Check tool chain variance (v0.2.2) - sequence instability
    if chain_variance in ["HIGH", "MEDIUM"] and tool_variance != "HIGH":
        unique_sequences = chain_details.get("unique_sequences", 0)
        dominant_seq = chain_details.get("dominant_sequence", "")
        seq_dist = chain_details.get("sequence_distribution", {})

        severity = "HIGH" if chain_variance == "HIGH" else "MEDIUM"
        root_causes.append(
            RootCause(
                type="CHAIN_VARIANCE",
                severity=severity,
                description="Tool execution sequence varies across runs",
                details=f"{unique_sequences} different sequences. Dominant: {dominant_seq}",
            )
        )

    # 2. Check claim instability
    if claim_variance == "HIGH":
        root_causes.append(
            RootCause(
                type="UNCONSTRAINED_ASSERTIONS",
                severity="CRITICAL",
                description="Model makes risky claims inconsistently",
                details=f"Risky claims in {claim_percentage:.0f}% of runs",
            )
        )

    # 3. Check semantic drift
    if semantic_drift == "HIGH":
        if structural_variance == "LOW":
            # Format is stable but meaning varies
            root_causes.append(
                RootCause(
                    type="UNDERSPECIFIED_PROMPT",
                    severity="MEDIUM",
                    description="Prompt allows too much interpretation freedom",
                    details="Format stable but meanings differ significantly",
                )
            )
        else:
            # Multiple sources of variation
            root_causes.append(
                RootCause(
                    type="GENERAL_INSTABILITY",
                    severity="HIGH",
                    description="Multiple sources of variation present",
                    details="Both meaning and format vary",
                )
            )

    # 4. Check structural variance
    if structural_variance == "HIGH":
        root_causes.append(
            RootCause(
                type="FORMAT_INCONSISTENCY",
                severity="MEDIUM",
                description="Output format not reliably enforced",
                details="JSON, markdown, and text formats mixed",
            )
        )

    # 5. Check length variance (lowest priority, only if no other causes)
    if length_variance == "HIGH" and len(root_causes) == 0:
        root_causes.append(
            RootCause(
                type="VERBOSITY_DRIFT",
                severity="LOW",
                description="Response length varies significantly",
                details="Output length inconsistent but other metrics stable",
            )
        )

    # If no specific causes found but score is low, add general cause
    # This shouldn't normally happen with proper threshold tuning
    if len(root_causes) == 0:
        # Check if any variance is MEDIUM
        if (
            semantic_drift == "MEDIUM"
            or tool_variance == "MEDIUM"
            or structural_variance == "MEDIUM"
        ):
            root_causes.append(
                RootCause(
                    type="MODERATE_VARIANCE",
                    severity="MEDIUM",
                    description="Moderate variance detected across metrics",
                    details="No critical issues but some inconsistency present",
                )
            )

    return root_causes


def get_primary_root_cause(root_causes: list[RootCause]) -> RootCause | None:
    """
    Get the primary (highest priority) root cause.

    Args:
        root_causes: List of identified root causes

    Returns:
        Primary RootCause or None if list is empty
    """
    if not root_causes:
        return None

    # Sort by severity: CRITICAL > HIGH > MEDIUM > LOW
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    sorted_causes = sorted(
        root_causes, key=lambda x: severity_order.get(x.severity, 999)
    )

    return sorted_causes[0]
