import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np


CLAIM_TYPES = [
    "FACTUAL_CLAIM",
    "OPINION",
    "PERSUASIVE_COPY",
    "SARCASTIC",
    "SPECULATIVE",
    "MIXED",
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "claim_type")
MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, "claim_type.onnx"))
LABELS_PATH = os.path.abspath(os.path.join(MODEL_DIR, "labels.json"))

SARCASM_PATTERNS = [
    r"\byeah[, ]+because\b",
    r"\byeah right\b",
    r"\bsure[, ]+because\b",
    r"\bobviously\b",
    r"\bas if\b",
    r"\bwhat could possibly go wrong\b",
    r"\bgreat job\b",
    r"/s\b",
]

OPINION_PATTERNS = [
    r"\bbetter than\b",
    r"\bbetter\b.*\bthan\b",
    r"\bworse than\b",
    r"\bworse\b.*\bthan\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\bi think\b",
    r"\bi feel\b",
    r"\bi prefer\b",
    r"\bin my opinion\b",
    r"\bbeautiful\b",
    r"\bawful\b",
    r"\bmeaningless\b",
    r"\bshould\b",
]

PERSUASIVE_PATTERNS = [
    r"\bif you care\b",
    r"\byou must\b",
    r"\bmust buy\b",
    r"\bact now\b",
    r"\blimited time\b",
    r"\bdon't miss\b",
    r"\bbuy this\b",
    r"\bbuy this now\b",
    r"\bbefore it's too late\b",
]

SPECULATIVE_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\blikely\b",
    r"\bpossibly\b",
    r"\bclaims? to\b",
    r"\breportedly\b",
    r"\bexpected to\b",
]

FACTUAL_ANCHOR_PATTERNS = [
    r"\b\d{1,4}\b",
    r"\baccording to\b",
    r"\bstudy\b",
    r"\breport\b",
    r"\bdata\b",
    r"\b(is|are|was|were|has|have)\b.+\b(in|at|on|from)\b",
]

HEDGE_WORDS = ["may", "might", "could", "possibly", "perhaps", "appears", "reportedly"]


class _OnnxClaimTypeModel:
    def __init__(self) -> None:
        self.available = False
        self.labels = CLAIM_TYPES
        self.session = None
        self.input_name = "features"

        if not os.path.exists(MODEL_PATH):
            return

        if os.path.exists(LABELS_PATH):
            try:
                with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
                    loaded_labels = json.load(labels_file)
                if isinstance(loaded_labels, list) and loaded_labels:
                    self.labels = [str(label) for label in loaded_labels]
            except Exception:
                pass

        try:
            import onnxruntime as ort  # type: ignore

            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(MODEL_PATH, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.available = True
        except Exception:
            self.available = False

    def predict(self, features: np.ndarray) -> Optional[Dict[str, float]]:
        if not self.available or self.session is None:
            return None
        try:
            outputs = self.session.run(None, {self.input_name: features.astype(np.float32)})
            probabilities = outputs[-1][0]
            mapped = {
                label: float(probabilities[idx])
                for idx, label in enumerate(self.labels)
                if idx < len(probabilities)
            }
            for label in CLAIM_TYPES:
                mapped.setdefault(label, 0.0)
            return mapped
        except Exception:
            return None


@lru_cache(maxsize=1)
def _load_onnx_model() -> _OnnxClaimTypeModel:
    return _OnnxClaimTypeModel()


def _matches(patterns: List[str], text: str) -> List[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def _build_feature_vector(
    normalized: str,
    model_signals: Optional[Dict[str, float]],
    style_metrics: Optional[Dict[str, float]],
) -> np.ndarray:
    numeric_hits = len(re.findall(r"\b\d{1,4}\b", normalized))
    opinion_hits = len(_matches(OPINION_PATTERNS, normalized))
    persuasive_hits = len(_matches(PERSUASIVE_PATTERNS, normalized))
    sarcasm_hits = len(_matches(SARCASM_PATTERNS, normalized))
    speculative_hits = len(_matches(SPECULATIVE_PATTERNS, normalized))
    factual_hits = len(_matches(FACTUAL_ANCHOR_PATTERNS, normalized))
    punctuation_intensity = min(1.0, (normalized.count("!") + normalized.count("?")) / 4.0)
    word_count = max(1, len(normalized.split()))
    length_norm = min(1.0, word_count / 40.0)
    hedge_ratio = min(1.0, sum(normalized.count(word) for word in HEDGE_WORDS) / max(1, word_count // 3))

    model_truth = float((model_signals or {}).get("truth_score", 0.5))
    model_bias = float((model_signals or {}).get("bias_score", 0.0))
    stylometric = float((style_metrics or {}).get("stylometric_consistency", 0.4))

    return np.array(
        [[
            min(1.0, numeric_hits / 2.0),
            min(1.0, opinion_hits / 2.0),
            min(1.0, persuasive_hits / 2.0),
            min(1.0, sarcasm_hits / 2.0),
            min(1.0, speculative_hits / 2.0),
            min(1.0, factual_hits / 2.0),
            punctuation_intensity,
            length_norm,
            max(0.0, min(1.0, model_truth)),
            max(0.0, min(1.0, model_bias)),
            max(0.0, min(1.0, stylometric)),
            hedge_ratio,
        ]],
        dtype=np.float32,
    )


def _heuristic_scores(normalized: str) -> Dict[str, float]:
    sarcasm_hits = _matches(SARCASM_PATTERNS, normalized)
    opinion_hits = _matches(OPINION_PATTERNS, normalized)
    persuasive_hits = _matches(PERSUASIVE_PATTERNS, normalized)
    speculative_hits = _matches(SPECULATIVE_PATTERNS, normalized)
    factual_hits = _matches(FACTUAL_ANCHOR_PATTERNS, normalized)

    score_map = {
        "SARCASTIC": min(1.0, 0.18 + 0.26 * len(sarcasm_hits)),
        "OPINION": min(1.0, 0.15 + 0.22 * len(opinion_hits)),
        "PERSUASIVE_COPY": min(1.0, 0.14 + 0.28 * len(persuasive_hits)),
        "SPECULATIVE": min(1.0, 0.14 + 0.24 * len(speculative_hits)),
        "FACTUAL_CLAIM": min(1.0, 0.10 + 0.22 * len(factual_hits)),
        "MIXED": 0.22,
    }

    if not any([sarcasm_hits, opinion_hits, persuasive_hits, speculative_hits, factual_hits]):
        score_map["MIXED"] = 0.60

    return score_map


def classify_claim_type(
    text: str,
    model_signals: Optional[Dict[str, float]] = None,
    style_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    normalized = (text or "").strip().lower()
    if not normalized:
        return {
            "claim_type": "MIXED",
            "signals": [],
            "reason": "empty_content",
            "source": "heuristic_fallback",
            "model_available": False,
            "scores": {claim_type: 0.0 for claim_type in CLAIM_TYPES},
        }

    heuristics = _heuristic_scores(normalized)
    feature_vector = _build_feature_vector(normalized, model_signals, style_metrics)

    onnx_model = _load_onnx_model()
    onnx_scores = onnx_model.predict(feature_vector)

    fused_scores = dict(heuristics)
    source = "heuristic_fallback"
    if onnx_scores is not None:
        for label in CLAIM_TYPES:
            fused_scores[label] = (0.55 * heuristics.get(label, 0.0)) + (0.45 * onnx_scores.get(label, 0.0))
        source = "onnx+heuristics"

    claim_type = max(fused_scores, key=fused_scores.get)
    signals = []
    signals.extend(_matches(SARCASM_PATTERNS, normalized)[:2])
    signals.extend(_matches(OPINION_PATTERNS, normalized)[:2])
    signals.extend(_matches(PERSUASIVE_PATTERNS, normalized)[:2])
    signals.extend(_matches(SPECULATIVE_PATTERNS, normalized)[:2])
    signals.extend(_matches(FACTUAL_ANCHOR_PATTERNS, normalized)[:2])

    reason = f"{claim_type.lower()}_dominant_signal"
    if source == "onnx+heuristics":
        reason = f"{reason}_with_local_onnx_gate"

    return {
        "claim_type": claim_type,
        "signals": signals[:8],
        "reason": reason,
        "source": source,
        "model_available": bool(onnx_model.available),
        "scores": {k: round(float(v), 4) for k, v in fused_scores.items()},
    }
