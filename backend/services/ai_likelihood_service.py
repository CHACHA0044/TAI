from typing import Dict


AI_PPL_BASELINE = 55.0
AI_PPL_CAP = 120.0


def _sanitize(value: float, default: float = 0.5) -> float:
    if value != value or value == float("inf") or value == float("-inf"):
        return default
    return max(0.0, min(1.0, value))


def compute_ai_likelihood(raw_model_ai: float, raw_features: Dict[str, float], style_metrics: Dict[str, float]) -> float:
    perplexity = float(raw_features.get("perplexity", 45.0))
    ppl_component = _sanitize((AI_PPL_BASELINE - min(perplexity, AI_PPL_CAP)) / AI_PPL_BASELINE, default=0.5)
    repetition_component = _sanitize(style_metrics.get("lexical_repetition", 0.0))
    uniformity_component = _sanitize(style_metrics.get("sentence_length_uniformity", 0.5))
    low_diversity_component = _sanitize(1.0 - style_metrics.get("lexical_diversity", 0.5))
    low_burstiness_component = _sanitize(1.0 - style_metrics.get("burstiness", 0.3))
    templated_component = _sanitize(style_metrics.get("stylometric_consistency", 0.5))

    # Repetition/uniformity/templated phrasing carry stronger signal than burstiness alone.
    blended = (
        0.15 * ppl_component
        + 0.20 * repetition_component
        + 0.20 * uniformity_component
        + 0.15 * low_diversity_component
        + 0.10 * low_burstiness_component
        + 0.20 * templated_component
    )
    return _sanitize((0.25 * _sanitize(raw_model_ai)) + (0.75 * blended))
