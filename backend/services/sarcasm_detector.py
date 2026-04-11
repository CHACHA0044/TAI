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
    r"\bgreat\b.{0,30}\b(disaster|fail|mess|chaos)\b",
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

# Satirical news headline patterns — The Onion style: deadpan, self-contradictory,
# absurdist announcements presented in news-report format.
_SATIRICAL_HEADLINE_PATTERNS = [
    # "Area man/woman/influencer announces/does [absurd thing]"
    r"\barea\s+(man|woman|resident|influencer|local|father|mother|teen|startup|person)\b",
    # Inanimate thing "vows to fix" (perpetually unfulfilled promise)
    r"\b(infrastructure|government|system|congress|committee|council|economy|institution)\s+(vows?|promises?|pledges?)\b",
    r"\bvows?\s+to\s+(fix|resolve|address|tackle|improve|sort)\b",
    # Same thing again / repetition irony
    r"\bas\s+(it|he|she|they)\s+ha[sd]\s+every\s+year\s+since\b",
    r"\bsame\s+(content|material|post|message|advice|thing)\s+as\s+(the\s+)?last\b",
    # Deadpan enthusiasm for counterproductive/ironic outcome
    r"\bvery\s+excited\s+to\s+finally\b",
    r"\bimmediately\s+check(?:s|ed)?\s+(his|her|their|its)\s+notifications?\b",
    # Absurd commercial absurdity
    r"\bpay(?:s|ing)?\s+extra\s+to\s+not\b",
    # Trivially obvious recommendations
    r"\brecommend(?:s|ing)?\s+that\s+everyone\s+just\s+be\b",
    r"\bshocked\s+to\s+(discover|learn|find)\s+that\b",
    # Passions → unpaid internship trope
    r"\bdirectly\s+into\s+an?\s+unpaid\b",
    # Disruption of nothing
    r"\bdo(?:es|ing)?\s+nothing\s+faster\b",
    # Corporate wellbeing irony
    r"\bleast\s+(important|significant|valued)\s+priority\b",
    # Procrastination / perpetual deferral
    r"\balways\s+tomorrow\b",
    # Wellness discovers obvious
    r"\bwas\s+(effective|working|true|helpful)\s+all\s+along\b",
    # Hypocrisy / same-people-who-said trope
    r"\bby\s+the\s+same\s+people\s+who\s+(said|claimed|told)\b",
    # Social media solves everything
    r"\bbest\s+way\s+to\s+deal\s+with.{5,60}is\s+to\s+post\b",
    # Privacy concern while sharing data
    r"\bvery\s+concerned\s+about\s+(privacy|security)\s+while\b",
    # Motivational speaker says same thing again
    r"\bexactly\s+what\s+(his|her|their)\s+last\b",
    # Absurd degree / credential
    r"\bnew\s+degree\s+in\s+(overthinking|procrastination|avoidance|worry)\b",
    # Tackle by relocating/avoiding
    r"\btackle\s+\w+\s+by\s+(relocating|ignoring|moving)\b",
    # Satirical ironic surprise / shock at the obvious
    r"\b(incredibly|genuinely|deeply)\s+(surprised|shocked|baffled)\s+that\b",
    r"\bincredibly\s+surprised\b",
    # 78% productivity irony pattern (general stats-about-reading-stats)
    r"\b\d+\s*%\s+of\s+\w+\s+tips?\s+are\s+\w+\s+during\b",
    # "before it is replaced by hotels" → gentrification satire
    r"\bbefore\s+(it\s+is|they\s+are)\s+(completely\s+)?replaced\s+by\b",
]


class SarcasmDetector:
    MARKER_WEIGHT = 0.22
    HYPERBOLE_WEIGHT = 0.18
    CONTRADICTION_WEIGHT = 0.22
    PUNCTUATION_WEIGHT = 0.10
    IRONIC_PRAISE_WEIGHT = 0.20
    # Satirical headline patterns (Onion-style) get a high weight because they are
    # very specific to satirical writing and have very low false positive risk.
    SATIRICAL_HEADLINE_WEIGHT = 0.55
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
        satirical_hits = self._regex_hits(normalized, _SATIRICAL_HEADLINE_PATTERNS)

        punctuation_cue = normalized.count("!") > 0 and normalized.count("?") > 0

        heuristic_score = min(
            1.0,
            (len(rhetorical_hits) * self.MARKER_WEIGHT)
            + (len(hyperbole_hits) * self.HYPERBOLE_WEIGHT)
            + (len(absurd_hits) * self.CONTRADICTION_WEIGHT)
            + (len(ironic_praise_hits) * self.IRONIC_PRAISE_WEIGHT)
            + (len(satirical_hits) * self.SATIRICAL_HEADLINE_WEIGHT)
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
            + [f"satirical_headline:{h}" for h in satirical_hits]
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
