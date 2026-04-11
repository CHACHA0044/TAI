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
    "click here",
    "only today",
    "expires soon",
    "hurry",
    "act in the next",
    "act within",
    "limited offer",
    "claim now",
    "get it now",
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

# FOMO / social-proof — fear of missing out and peer-pressure cues
# These are counted as action-pressure because they implicitly compel the reader to act.
FOMO_AND_SOCIAL_PROOF_PATTERNS = [
    "will you be next",
    "don't be left behind",
    "don't let your",
    "fall behind",
    "smart parents are",
    "smart people are",
    "thousands have already",
    "millions have already",
    "join the thousands",
    "join the millions",
    "millions who have already",
    "already taken control",
    "everyone is already",
    "your peers are",
    "what are you waiting for",
    "you're missing out",
    "don't miss your chance",
    "window is closing",
    "the top 1%",
    "top 1%",
    "billionaires are quietly",
    "quietly doing this",
    "financially free",
    "from broke to",
    "escaped the rat race",
    # Opportunity-cost / regret appeals
    "if you miss this",
    "miss this opportunity",
    "spend the rest of your life",
    "rest of your life wondering",
    "what could have been",
    # Social proof via possessive data
    "your data is already",
    "already being sold",
    # Emotional blackmail / social obligation
    "if you truly loved",
    "if you really cared",
    "this investment is your only",
    "your only real option",
    "only real option",
    "pension system will fail",
]

# Suppressed-info / curiosity-gap — fear of hidden, removed, or suppressed information
# These act as coercion by implying urgency to access "forbidden" knowledge.
SUPPRESSED_INFO_PATTERNS = [
    "your doctor hopes you never",
    "they don't want you to know",
    "suppressed report",
    "deleted before you see",
    "desperately trying to delete",
    "before the fda shuts",
    "banned in europe",
    "dermatologists hate",
    "doctors hate",
    "pharmaceutical industry to allow",
    "what they're hiding",
    "your doctor won't tell",
    "the one food",
    "one weird trick",
    "ancient remedy",
    "download before it",
    "video they're",
    "before it's removed",
    "last warning",
    "window to protect",
    "closes at midnight",
    "every day you wait",
    "start the cleanse now",
    "dangerous toxins",
    "claim your free",
    "even skeptics are shocked",
    "even skeptics",
    "mother of three lost",
    "without dieting",
    "does to blood sugar",
    "blood sugar in",
]


def _find_hits(text: str, patterns: List[str]) -> List[str]:
    return [pattern for pattern in patterns if pattern in text]


def detect_manipulation(text: str, style_metrics: Dict[str, float]) -> Dict[str, Any]:
    normalized = (text or "").lower()
    emotional_hits = _find_hits(normalized, EMOTIONAL_PRESSURE_PATTERNS)
    urgency_hits = _find_hits(normalized, URGENCY_OR_COERCION_PATTERNS)
    fear_hits = _find_hits(normalized, FEAR_AND_MORALIZING_PATTERNS)
    sales_hits = _find_hits(normalized, SALES_PRESSURE_PATTERNS)
    fomo_hits = _find_hits(normalized, FOMO_AND_SOCIAL_PROOF_PATTERNS)
    suppressed_hits = _find_hits(normalized, SUPPRESSED_INFO_PATTERNS)
    exclamation_component = 0.08 if "!" in normalized else 0.0

    # IMPORTANT: Manipulation verdict requires action-pressure/coercion cues.
    # FOMO, social-proof, and suppressed-info patterns count as action-pressure
    # because they implicitly compel the reader to act out of fear of missing out
    # or accessing hidden/forbidden content.
    has_action_pressure = bool(urgency_hits) or bool(sales_hits) or bool(fomo_hits) or bool(suppressed_hits)

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
            (len(emotional_hits) * 0.18)
            + (len(urgency_hits) * 0.28)
            + (len(fear_hits) * 0.15)
            + (len(sales_hits) * 0.20)
            + (len(fomo_hits) * 0.32)
            + (len(suppressed_hits) * 0.38)
            + exclamation_component
            + (0.10 * float(style_metrics.get("lexical_repetition", 0.0))),
        )

    # Structured rule hits for debug telemetry
    manipulation_rule_hits = (
        [f"emotional_pressure:{h}" for h in emotional_hits]
        + [f"urgency_coercion:{h}" for h in urgency_hits]
        + [f"fear_moralizing:{h}" for h in fear_hits]
        + [f"sales_pressure:{h}" for h in sales_hits]
        + [f"fomo_social_proof:{h}" for h in fomo_hits]
        + [f"suppressed_info:{h}" for h in suppressed_hits]
    )
    if not has_action_pressure and (emotional_hits or fear_hits):
        manipulation_rule_hits.append("gate:no_action_pressure_cue_score_capped")

    return {
        "score": round(rule_score, 4),
        "indicators": (emotional_hits + urgency_hits + fear_hits + sales_hits + fomo_hits + suppressed_hits)[:10],
        "manipulation_rule_hits": manipulation_rule_hits[:12],
    }
