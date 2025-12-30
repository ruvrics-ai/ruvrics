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
    structural_variance: str,
    structural_score: float,
    length_variance: str,
    length_score: float,
    claim_variance: str,
    claim_percentage: float,
    config: Config | None = None,
) -> list[RootCause]:
    """
    Identify root causes of instability using decision tree.

    From spec Section 4:
    Priority order (most specific → most general):
    1. Tool-use instability (highest priority)
    2. Claim instability
    3. Semantic drift
    4. Structural variance
    5. Length variance (lowest priority)

    Args:
        All variance classifications and scores from metrics
        config: Optional configuration

    Returns:
        List of RootCause objects, ordered by priority
    """
    cfg = config or get_config()
    root_causes = []

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
