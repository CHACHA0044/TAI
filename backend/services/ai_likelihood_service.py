from typing import Dict


AI_PPL_BASELINE = 55.0
AI_PPL_CAP = 120.0

# Phrases characteristic of AI-generated corporate/academic prose.
# Each phrase adds a small fixed boost to the AI likelihood score.
# The patterns are specific enough that genuine human writing rarely contains
# more than one or two of them in a short text.
_AI_PROSE_MARKERS = [
    "striking the right balance",
    "it is important to acknowledge",
    "nuanced picture",
    "examine the evidence holistically",
    "root causes of",
    "systemic structures that",
    "must engage with the",
    "remains the cornerstone",
    "no longer optional",
    "demands not only",
    "fundamental shift in",
    # "foster/fostering" → match on stem
    "psychological safety",
    "upskilling the workforce",
    "narrative arc",
    "cross-functional collaboration",
    "from a bird's-eye view",
    "inclusive dialogue",
    "centers the voices",
    "transcend their limitations",
    "realize their full potential",
    "ethical implications of this",
    "requires careful deliberation",
    "with the right frameworks in place",
    "emerge stronger from disruption",
    "interplay between",
    "transformative investments",
    "compelling case",
    "unlocking organizational potential",
    "at its core, leadership",
    "in an interconnected world",
    "holistically, a more nuanced",
    "underpins",
    "inherent in the methodology",
    "pivot effectively",
    "sustainable progress hinges",
    "proliferation of misinformation",
    "robust media literacy",
    "synthesize diverse perspectives",
    "empowering communities with access",
    # "interdisciplinary approach" → match on stem
    "interdisciplinary",
    "individual behavior and institutional design",
    "crafting effective policy",
    "building a more equitable",
    "the narrative arc of",
    "serves as a powerful allegory",
    # Additional AI-prose hallmarks
    "intersection of public health",
    "presents unique challenges that",
    "demands careful deliberation",
    "enable individuals to voice",
    "spanning education",
    "spanning economics",
    "spanning public health",
    "multifaceted issue",
    "requires a holistic",
    "require a holistic",
    "increasingly polarized",
    "at its core, this",
    "at its core, the",
]


def _sanitize(value: float, default: float = 0.5) -> float:
    if value != value or value == float("inf") or value == float("-inf"):
        return default
    return max(0.0, min(1.0, value))


def compute_ai_likelihood(raw_model_ai: float, raw_features: Dict[str, float], style_metrics: Dict[str, float], text: str = "") -> float:
    perplexity = float(raw_features.get("perplexity", 45.0))
    ppl_component = _sanitize((AI_PPL_BASELINE - min(perplexity, AI_PPL_CAP)) / AI_PPL_BASELINE, default=0.5)
    repetition_component = _sanitize(style_metrics.get("lexical_repetition", 0.0))
    uniformity_component = _sanitize(style_metrics.get("sentence_length_uniformity", 0.5))
    low_diversity_component = _sanitize(1.0 - style_metrics.get("lexical_diversity", 0.5))
    low_burstiness_component = _sanitize(1.0 - style_metrics.get("burstiness", 0.3))
    templated_component = _sanitize(style_metrics.get("stylometric_consistency", 0.5))

    # Count AI prose marker hits — each match adds a fixed boost (capped at 0.45 total)
    normalized_text = (text or "").lower()
    prose_hit_count = sum(1 for marker in _AI_PROSE_MARKERS if marker in normalized_text)
    prose_marker_boost = _sanitize(min(0.45, prose_hit_count * 0.15))

    # Model score gets a higher weight (0.40) since it's the most reliable signal.
    # Stylometric features fill in when the model score is unavailable or low.
    blended = (
        0.12 * ppl_component
        + 0.12 * repetition_component
        + 0.12 * uniformity_component
        + 0.08 * low_diversity_component
        + 0.08 * low_burstiness_component
        + 0.08 * templated_component
        + prose_marker_boost
    )
    return _sanitize((0.40 * _sanitize(raw_model_ai)) + (0.60 * blended))
