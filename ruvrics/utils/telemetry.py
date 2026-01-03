"""
Anonymous telemetry for Ruvrics usage tracking.

Telemetry helps us understand:
- Which features are used
- Which models are popular
- Error frequency

Privacy:
- No user queries or responses are sent
- No API keys or sensitive data
- Completely anonymous (no user identification)
- Can be disabled via environment variable

To disable telemetry:
    export RUVRICS_TELEMETRY=false
"""

import os
import platform
import sys
from typing import Any, Optional
from datetime import datetime

# Telemetry is disabled by default for now
# To enable: export RUVRICS_TELEMETRY=true
TELEMETRY_ENABLED = os.getenv("RUVRICS_TELEMETRY", "false").lower() == "true"


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled."""
    return TELEMETRY_ENABLED


def get_anonymous_id() -> str:
    """
    Generate anonymous device identifier.

    Uses a hash of machine characteristics (no personal data).
    """
    import hashlib

    # Combine non-personal system info
    info = f"{platform.machine()}{platform.system()}{platform.python_version()}"
    return hashlib.sha256(info.encode()).hexdigest()[:16]


def track_event(
    event_name: str,
    properties: Optional[dict[str, Any]] = None,
) -> None:
    """
    Track anonymous usage event.

    Args:
        event_name: Event identifier (e.g., "stability_run")
        properties: Non-sensitive event properties
    """
    if not is_telemetry_enabled():
        return

    # Prepare telemetry data
    data = {
        "event": event_name,
        "timestamp": datetime.utcnow().isoformat(),
        "properties": properties or {},
        "context": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "ruvrics_version": "0.1.0",
            "anonymous_id": get_anonymous_id(),
        }
    }

    # TODO: Send to telemetry backend
    # For now, just log that telemetry would be sent
    # Uncomment below to integrate with PostHog or similar:
    # try:
    #     import posthog
    #     posthog.api_key = 'YOUR_POSTHOG_KEY'
    #     posthog.capture(
    #         distinct_id=data["context"]["anonymous_id"],
    #         event=event_name,
    #         properties=data["properties"]
    #     )
    # except Exception:
    #     pass  # Silently fail - telemetry should never break functionality


def track_stability_run(
    model: str,
    runs: int,
    successful_runs: int,
    duration_seconds: float,
    stability_score: float,
    risk_classification: str,
    has_tools: bool,
    error: Optional[str] = None,
) -> None:
    """
    Track stability analysis run.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini")
        runs: Total runs requested
        successful_runs: Number of successful runs
        duration_seconds: Total duration
        stability_score: Final stability score
        risk_classification: SAFE, RISKY, or DO_NOT_SHIP
        has_tools: Whether tools were used
        error: Error type if failed (no error message details)
    """
    properties = {
        "model": model,
        "runs": runs,
        "successful_runs": successful_runs,
        "duration_seconds": round(duration_seconds, 2),
        "stability_score": round(stability_score, 1),
        "risk_classification": risk_classification,
        "has_tools": has_tools,
        "success": error is None,
        "error_type": error if error else None,
    }

    track_event("stability_run", properties)


def track_error(
    error_type: str,
    command: str,
) -> None:
    """
    Track errors (type only, no sensitive details).

    Args:
        error_type: Error classification
        command: Command that failed
    """
    properties = {
        "error_type": error_type,
        "command": command,
    }

    track_event("error", properties)


def print_telemetry_status() -> None:
    """Print telemetry status message (called on first run)."""
    if is_telemetry_enabled():
        print("Telemetry: Enabled (helps improve Ruvrics)")
        print("To disable: export RUVRICS_TELEMETRY=false")
    # Don't print anything if disabled (less noise)
