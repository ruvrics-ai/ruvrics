"""
Recommendation engine - generates actionable fixes.

Maps root causes to concrete recommendations.
From spec Section 6 - Recommendation Decision Tree.
"""

from ruvrics.config import Config, get_config
from ruvrics.core.models import RootCause, Recommendation


# Recommendation templates for each root cause type
RECOMMENDATION_MAP: dict[str, list[dict]] = {
    "NONDETERMINISTIC_TOOL_ROUTING": [
        {
            "title": "Enforce mandatory tool usage",
            "category": "prompt",
            "priority": 1,
            "description": (
                "Add explicit instruction to ALWAYS use the tool. "
                "Don't let the model decide whether to use it."
            ),
            "example": (
                'Add to system prompt:\n'
                '"ALWAYS use {tool_name} for {intent} queries. '
                'Never answer directly without using the tool."'
            ),
        },
        {
            "title": "Move routing logic outside LLM",
            "category": "code",
            "priority": 2,
            "description": (
                "Programmatically decide when to call tools instead of "
                "letting the LLM make the decision."
            ),
            "example": (
                "if query_matches_pattern(query, 'flight_search'):\n"
                "    force_tool_use(search_flights)\n"
                "# Don't let LLM decide"
            ),
        },
        {
            "title": "Add explicit tool-selection instructions",
            "category": "prompt",
            "priority": 3,
            "description": "Provide clear rules for when to use each tool.",
            "example": (
                '"If the query is about [X], use tool [Y].\n'
                'If the query is about [Z], use tool [W]."'
            ),
        },
    ],
    "TOOL_CONFUSION": [
        {
            "title": "Simplify tool descriptions",
            "category": "prompt",
            "priority": 1,
            "description": "Make tool descriptions more explicit and unambiguous.",
            "example": (
                "Tool: search_flights\n"
                'Description: "Search for available flights. '
                'Use this for ANY flight-related query."'
            ),
        },
        {
            "title": "Reduce number of available tools",
            "category": "code",
            "priority": 2,
            "description": "Provide only the tools needed for this specific query.",
            "example": "# Filter tools based on query intent before calling LLM",
        },
    ],
    "UNCONSTRAINED_ASSERTIONS": [
        {
            "title": "Add explicit constraint on guarantees",
            "category": "prompt",
            "priority": 1,
            "description": "Forbid absolute promises and guarantees.",
            "example": (
                'Add to system prompt:\n'
                '"Never make guarantees or absolute promises. Always use qualifiers:\n'
                "- 'typically', 'usually', 'often', 'may', 'can'\n"
                "- Never use: 'guarantee', 'always', '100%', 'never fails'\""
            ),
        },
        {
            "title": "Forbid false authority claims",
            "category": "prompt",
            "priority": 2,
            "description": "Prevent claims about access to external systems or data.",
            "example": (
                '"Do not claim to have access to external systems, databases, or live data.\n'
                'You are an AI assistant with knowledge only up to [cutoff date]."'
            ),
        },
        {
            "title": "Add grounding requirement",
            "category": "prompt",
            "priority": 3,
            "description": "Require claims to be supported by provided context.",
            "example": (
                '"Only make specific claims when explicitly supported by the provided context.\n'
                'If information isn\'t in the context, say \'I don\'t have that information.\'"'
            ),
        },
    ],
    "UNDERSPECIFIED_PROMPT": [
        {
            "title": "Add 3-5 example outputs",
            "category": "prompt",
            "priority": 1,
            "description": "Show examples of correct responses to reduce ambiguity.",
            "example": (
                "Examples of correct responses:\n\n"
                "Q: [example query 1]\n"
                "A: [example response 1]\n\n"
                "Q: [example query 2]\n"
                "A: [example response 2]"
            ),
        },
        {
            "title": "Add reasoning template",
            "category": "prompt",
            "priority": 2,
            "description": "Provide explicit structure for responses.",
            "example": (
                '"Structure your response as:\n'
                "1. Acknowledge the question\n"
                "2. State relevant constraints\n"
                "3. Provide answer\n"
                '4. Include confidence level if uncertain"'
            ),
        },
        {
            "title": "Add negative constraints",
            "category": "prompt",
            "priority": 3,
            "description": "Explicitly state what NOT to do.",
            "example": (
                '"Do NOT:\n'
                "- Provide medical/legal advice\n"
                "- Speculate beyond available information\n"
                '- Make comparisons to competitors"'
            ),
        },
    ],
    "FORMAT_INCONSISTENCY": [
        {
            "title": "Enforce JSON schema",
            "category": "prompt",
            "priority": 1,
            "description": "Require strict adherence to output format.",
            "example": (
                'Add to system prompt:\n'
                '"Always respond with valid JSON matching this exact schema:\n'
                '{\n'
                '  "status": "success" | "error",\n'
                '  "data": { ... },\n'
                '  "message": "string"\n'
                '}"'
            ),
        },
        {
            "title": "Add format validation example",
            "category": "prompt",
            "priority": 2,
            "description": "Show exact format expected.",
            "example": (
                "Example of correct format:\n"
                "{json_example}\n\n"
                "Do not deviate from this structure."
            ),
        },
        {
            "title": "Add retry logic for invalid format",
            "category": "code",
            "priority": 3,
            "description": "Programmatically retry if format is wrong.",
            "example": (
                "if not validate_structure(response):\n"
                '    response = retry_with_instruction("Your last response had invalid format...")'
            ),
        },
    ],
    "VERBOSITY_DRIFT": [
        {
            "title": "Specify length constraints",
            "category": "prompt",
            "priority": 1,
            "description": "Add explicit limits on response length.",
            "example": (
                '"Keep responses to 2-3 sentences."\n'
                '"Respond in under 100 words."\n'
                '"Provide a brief summary (50-75 tokens)."'
            ),
        },
        {
            "title": "Add conciseness instruction",
            "category": "prompt",
            "priority": 2,
            "description": "Encourage brevity.",
            "example": '"Be concise and direct. Avoid unnecessary explanations."',
        },
    ],
    "GENERAL_INSTABILITY": [
        {
            "title": "Increase prompt specificity",
            "category": "prompt",
            "priority": 1,
            "description": "Add more constraints and examples to reduce variance.",
            "example": "Combine recommendations from multiple categories above.",
        },
        {
            "title": "Use lower temperature",
            "category": "config",
            "priority": 2,
            "description": "Reduce randomness in model outputs.",
            "example": "Set temperature=0 or close to 0 for maximum determinism.",
        },
    ],
    "MODERATE_VARIANCE": [
        {
            "title": "Review and refine prompt",
            "category": "prompt",
            "priority": 1,
            "description": "Some inconsistency detected - consider adding more constraints.",
            "example": "Add examples or explicit instructions to reduce variance.",
        },
    ],
}


def generate_recommendations(
    root_causes: list[RootCause], config: Config | None = None
) -> list[Recommendation]:
    """
    Generate actionable recommendations from root causes.

    From spec Section 6:
    - Take top 2 root causes (by severity)
    - Get top 2 recommendations per cause
    - Return max 3 unique recommendations total

    Args:
        root_causes: Identified root causes
        config: Optional configuration

    Returns:
        List of Recommendation objects, prioritized
    """
    cfg = config or get_config()

    if not root_causes:
        # No issues detected - system is stable
        return []

    # Sort by severity: CRITICAL > HIGH > MEDIUM > LOW
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_causes = sorted(
        root_causes, key=lambda x: severity_order.get(x.severity, 999)
    )

    recommendations = []

    # Take top 2 root causes
    for cause in sorted_causes[:2]:
        cause_type = cause.type
        recs_templates = RECOMMENDATION_MAP.get(cause_type, [])

        # Take top 2 recommendations per cause
        for rec_template in recs_templates[:2]:
            recommendations.append(
                Recommendation(
                    title=rec_template["title"],
                    category=rec_template["category"],
                    priority=rec_template["priority"],
                    description=rec_template["description"],
                    example=rec_template.get("example"),
                )
            )

    # Deduplicate while preserving order
    seen_titles = set()
    unique_recommendations = []

    for rec in recommendations:
        if rec.title not in seen_titles:
            seen_titles.add(rec.title)
            unique_recommendations.append(rec)

    # Return max 3 recommendations (from spec Section 6)
    return unique_recommendations[:3]
