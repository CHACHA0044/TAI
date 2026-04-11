import os
import re
from typing import Any, Dict, List


class HuggingFaceSarcasmAdapter:
    """
    Optional adapter for local HuggingFace sarcasm integration.
    Stubbed unless a model is explicitly wired.
    """

    def __init__(self):
        self.model_name = os.getenv("SARCASM_HF_MODEL")

    def detect(self, text: str) -> Dict[str, Any]:
        return {
            "sarcasm": False,
            "score": 0.0,
            "provider": "hf_stub",
            "indicators": [],
        }


# Absurd or well-known-false factual references that signal irony/sarcasm
_ABSURD_CLAIM_PATTERNS = [
    r"\b(earth|world)\s+is\s+(flat|hollow|square)\b",
    r"\b(pigs?\s+can\s+fly|money\s+grows\s+on\s+trees|vaccines?\s+cause\s+autism)\b",
    r"\b(moon\s+landing\s+was\s+faked|sun\s+revolves\s+around\s+the\s+earth)\b",
    r"\b(dinosaurs?\s+(never\s+existed|are\s+still\s+alive))\b",
]

# Rhetorical/irony framing openers that typically introduce sarcasm
_RHETORICAL_FRAMING_PATTERNS = [
    r"\byeah[, ]+right\b",
    r"\byeah[, ]+because\b",
    r"\bsure[, ]+because\b",
    r"\bobviously\b",
    r"\bof course\b",
    r"\boh sure\b",
    r"\bas if\b",
    r"\bwhat could possibly go wrong\b",
    r"\btotally\b",
    r"\boh great\b",
    r"\bbrilliant[, ]+idea\b",
    r"\bjust great\b",
    r"\bwhat\s+a\s+(surprise|shock|miracle)\b",
    r"\bwho\s+would\s+have\s+(thought|guessed|known)\b",
]

# Ironic praise followed by an absurd/negative qualifier
_IRONIC_PRAISE_PATTERNS = [
    r"\bgreat\b.{0,30}\b(disaster|fail|mess|disaster|chaos)\b",
    r"\bbrilliant\b.{0,30}\b(idiot|moron|fool|disaster)\b",
    r"\bwonderful\b.{0,30}\b(fail|disaster|collapse|ruin)\b",
    r"\bthanks\s+(a\s+lot|so\s+much)\b.{0,30}\b(for\s+nothing|ruined|destroyed|broke)\b",
]

# Hyperbole markers used in sarcastic contexts
_HYPERBOLE_PATTERNS = [
    r"\beveryone\s+knows\b",
    r"\bliterally\s+(impossible|everyone|always|never|the\s+best|the\s+worst)\b",
    r"\bthe?\s+(best|worst)\s+(thing\s+)?(ever|in\s+(history|the\s+world))\b",
    r"\babsolutely\s+perfect\b",
    r"\b100\s*%\s+(sure|certain|true|guaranteed)\b",
    r"\bnever\s+(been|felt|seen)\s+(more|so|better|worse)\b",
    r"\balways\s+(works|right|perfect|wrong)\b",
]


class SarcasmDetector:
    MARKER_WEIGHT = 0.22
    HYPERBOLE_WEIGHT = 0.18
    CONTRADICTION_WEIGHT = 0.22
    PUNCTUATION_WEIGHT = 0.10
    IRONIC_PRAISE_WEIGHT = 0.20
    # Combo boost applied when both rhetorical framing AND absurd factual claim coexist
    COMBO_BOOST = 0.25
    SARCASM_THRESHOLD = 0.45  # Lowered slightly to increase recall on clear sarcasm cases

    def __init__(self):
        self.hf_adapter = HuggingFaceSarcasmAdapter()

    @staticmethod
    def _regex_hits(text: str, patterns: List[str]) -> List[str]:
        return [pattern for pattern in patterns if re.search(pattern, text)]

    def detect(self, text: str) -> Dict[str, Any]:
        normalized = (text or "").lower()

        rhetorical_hits = self._regex_hits(normalized, _RHETORICAL_FRAMING_PATTERNS)
        hyperbole_hits = self._regex_hits(normalized, _HYPERBOLE_PATTERNS)
        absurd_hits = self._regex_hits(normalized, _ABSURD_CLAIM_PATTERNS)
        ironic_praise_hits = self._regex_hits(normalized, _IRONIC_PRAISE_PATTERNS)

        punctuation_cue = normalized.count("!") > 0 and normalized.count("?") > 0

        heuristic_score = min(
            1.0,
            (len(rhetorical_hits) * self.MARKER_WEIGHT)
            + (len(hyperbole_hits) * self.HYPERBOLE_WEIGHT)
            + (len(absurd_hits) * self.CONTRADICTION_WEIGHT)
            + (len(ironic_praise_hits) * self.IRONIC_PRAISE_WEIGHT)
            + (self.PUNCTUATION_WEIGHT if punctuation_cue else 0.0),
        )

        # Combo boost: rhetorical framing + absurd factual contradiction = very likely sarcasm
        combo_boost = 0.0
        if rhetorical_hits and absurd_hits:
            combo_boost = self.COMBO_BOOST

        heuristic_score = min(1.0, heuristic_score + combo_boost)

        hf_signal = self.hf_adapter.detect(text)
        score = max(heuristic_score, float(hf_signal.get("score", 0.0)))
        sarcasm = score >= self.SARCASM_THRESHOLD or bool(hf_signal.get("sarcasm", False))

        rule_hits = (
            [f"rhetorical:{h}" for h in rhetorical_hits]
            + [f"hyperbole:{h}" for h in hyperbole_hits]
            + [f"absurd_claim:{h}" for h in absurd_hits]
            + [f"ironic_praise:{h}" for h in ironic_praise_hits]
        )
        if punctuation_cue:
            rule_hits.append("punctuation:mixed_exclamatory_question_pattern")
        if combo_boost > 0:
            rule_hits.append("combo_boost:rhetorical+absurd_claim")
        rule_hits.extend([f"hf:{ind}" for ind in hf_signal.get("indicators", [])])

        return {
            "sarcasm": sarcasm,
            "score": round(score, 4),
            "provider": "heuristic+hf_stub",
            "indicators": rule_hits[:12],
            # Expose structured rule hits for debug telemetry
            "sarcasm_rule_hits": rule_hits[:12],
        }
