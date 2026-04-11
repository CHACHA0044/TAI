from typing import Any, Dict, List


# Ideological / politically loaded adjectives — signal framing slant, not neutral description
LOADED_ADJECTIVES = [
    "corrupt",
    "evil",
    "disgusting",
    "radical",
    "propaganda",
    "traitorous",
    "extremist",
    "woke",
    "fascist",
    "socialist",
    "communist",
    "globalist",
    "nationalist",
    "far-left",
    "far-right",
    "elitist",
    "treasonous",
]

# Strong demonization — dehumanizing language or existential threat framing
DEMONIZATION_PATTERNS = [
    "enemy of the people",
    "vermin",
    "subhuman",
    "destroy our country",
    "invasion",
    "infestation",
    "cancer on society",
    "plague on",
    "parasites",
    "scourge",
]

# Us-vs-them / ideological framing / loaded group references
US_VS_THEM_PATTERNS = [
    "us vs them",
    "their agenda",
    "our side",
    "those people",
    "mainstream media",
    "media is a propaganda machine",
    "propaganda machine",
    "deep state",
    "shadow government",
    "globalist agenda",
    "the left",
    "the right",
    "real americans",
    "true patriots",
    "the establishment",
    "the elites",
    "the regime",
    "the swamp",
]

# Political/social slant — loaded framing of social or political groups
POLITICAL_SOCIAL_SLANT_PATTERNS = [
    "illegal aliens",
    "open borders",
    "climate alarmist",
    "race-baiting",
    "gender ideology",
    "great replacement",
    "white privilege",
    "systemic racism",
    "cancel culture",
    "social justice warrior",
    "virtue signaling",
    "thought police",
]


def _find_hits(text: str, patterns: List[str]) -> List[str]:
    return [pattern for pattern in patterns if pattern in text]


def detect_bias(text: str, model_bias_score: float = 0.0) -> Dict[str, Any]:
    normalized = (text or "").lower()
    loaded_hits = _find_hits(normalized, LOADED_ADJECTIVES)
    demonization_hits = _find_hits(normalized, DEMONIZATION_PATTERNS)
    framing_hits = _find_hits(normalized, US_VS_THEM_PATTERNS)
    slant_hits = _find_hits(normalized, POLITICAL_SOCIAL_SLANT_PATTERNS)

    # Demonization terms are weighted highest because they are the strongest slant signal.
    # Political/social slant patterns get a moderate weight.
    rule_score = min(
        1.0,
        (len(loaded_hits) * 0.35)
        + (len(demonization_hits) * 0.45)
        + (len(framing_hits) * 0.30)
        + (len(slant_hits) * 0.28),
    )
    score = max(rule_score, float(model_bias_score))

    # Structured rule hits for debug telemetry (categorized by type)
    bias_rule_hits = (
        [f"loaded_adjective:{h}" for h in loaded_hits]
        + [f"demonization:{h}" for h in demonization_hits]
        + [f"us_vs_them:{h}" for h in framing_hits]
        + [f"political_slant:{h}" for h in slant_hits]
    )

    return {
        "score": round(score, 4),
        "indicators": (loaded_hits + demonization_hits + framing_hits + slant_hits)[:10],
        "bias_rule_hits": bias_rule_hits[:12],
    }
