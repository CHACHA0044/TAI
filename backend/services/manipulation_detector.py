from typing import Any, Dict, List


# Emotional pressure — personal vulnerability framing that precedes a call to action
EMOTIONAL_PRESSURE_PATTERNS = [
    "if you care",
    "your family",
    "wake up",
    "before it's too late",
    "protect your loved ones",
    "your children are at risk",
    "don't let them take",
    "they're coming for",
]

# Urgency / coercion — action-pressure cues; manipulation REQUIRES at least one of these
URGENCY_OR_COERCION_PATTERNS = [
    "must buy",
    "act now",
    "limited time",
    "buy this today",
    "buy this now",
    "you must",
    "order now",
    "don't wait",
    "last chance",
    "sign up now",
    "register today",
    "call now",
    "click here now",
    "only today",
    "expires soon",
]

# Fear appeals and moralizing commands
FEAR_AND_MORALIZING_PATTERNS = [
    "or else",
    "only a fool would",
    "you have to",
    "do the right thing",
    "you'll regret",
    "it will be too late",
    "this is your last chance",
    "if you don't act",
    "the consequences",
]

# Sales pressure — commercial high-pressure tactics
SALES_PRESSURE_PATTERNS = [
    "for a limited time",
    "exclusive offer",
    "while supplies last",
    "limited supply",
    "risk-free",
    "guaranteed results",
    "act before",
    "don't miss out",
    "special discount",
    "today only",
    "buy now",
]


def _find_hits(text: str, patterns: List[str]) -> List[str]:
    return [pattern for pattern in patterns if pattern in text]


def detect_manipulation(text: str, style_metrics: Dict[str, float]) -> Dict[str, Any]:
    normalized = (text or "").lower()
    emotional_hits = _find_hits(normalized, EMOTIONAL_PRESSURE_PATTERNS)
    urgency_hits = _find_hits(normalized, URGENCY_OR_COERCION_PATTERNS)
    fear_hits = _find_hits(normalized, FEAR_AND_MORALIZING_PATTERNS)
    sales_hits = _find_hits(normalized, SALES_PRESSURE_PATTERNS)
    exclamation_component = 0.08 if "!" in normalized else 0.0

    # IMPORTANT: Manipulation verdict requires action-pressure/coercion cues.
    # Without at least one urgency/coercion or sales-pressure signal, the score
    # is intentionally suppressed so pure emotional framing does not fire the
    # manipulation verdict (that would collide with bias).
    has_action_pressure = bool(urgency_hits) or bool(sales_hits)

    if not has_action_pressure:
        # Emotional framing alone is a weak signal — cap below the manipulation verdict threshold
        weak_score = min(
            0.45,
            (len(emotional_hits) * 0.12)
            + (len(fear_hits) * 0.10)
            + exclamation_component
            + (0.08 * float(style_metrics.get("lexical_repetition", 0.0))),
        )
        rule_score = weak_score
    else:
        # Full scoring when action-pressure cues exist
        rule_score = min(
            1.0,
            (len(emotional_hits) * 0.22)
            + (len(urgency_hits) * 0.30)
            + (len(fear_hits) * 0.16)
            + (len(sales_hits) * 0.22)
            + exclamation_component
            + (0.10 * float(style_metrics.get("lexical_repetition", 0.0))),
        )

    # Structured rule hits for debug telemetry
    manipulation_rule_hits = (
        [f"emotional_pressure:{h}" for h in emotional_hits]
        + [f"urgency_coercion:{h}" for h in urgency_hits]
        + [f"fear_moralizing:{h}" for h in fear_hits]
        + [f"sales_pressure:{h}" for h in sales_hits]
    )
    if not has_action_pressure and (emotional_hits or fear_hits):
        manipulation_rule_hits.append("gate:no_action_pressure_cue_score_capped")

    return {
        "score": round(rule_score, 4),
        "indicators": (emotional_hits + urgency_hits + fear_hits + sales_hits)[:10],
        "manipulation_rule_hits": manipulation_rule_hits[:12],
    }
