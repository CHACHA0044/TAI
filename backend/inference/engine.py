import math
import torch
import numpy as np
import os
import time
import re
import logging
from datetime import datetime, timezone
from utils.analysis_utils import stage_route_primary_verdict, to_bucket

logger = logging.getLogger("truthguard")
AI_RAW_MODEL_WEIGHT = 0.7
AI_OPENAI_BLEND_WEIGHT = 0.3

# Attempt to import heavyweight deps, fail gracefully
try:
    from transformers import AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed — running in pure mock mode")

try:
    from training.model import TruthGuardMultiTaskModel
except ImportError:
    TruthGuardMultiTaskModel = None

try:
    from training.features import FeatureExtractor
    HAS_FEATURES = True
except Exception:
    HAS_FEATURES = False

from services.trust_agents import TrustAgent
from services.news_search_agent import NewsSearchAgent
from services.sarcasm_detector import SarcasmDetector
from services.verifiability import assess_claim_verifiability
from services.claim_type_detector import classify_claim_type
from services.bias_detector import detect_bias
from services.manipulation_detector import detect_manipulation
from services.ai_likelihood_service import compute_ai_likelihood
from services.adapters import (
    HuggingFaceBiasClassifierAdapter,
    ExternalFactCheckAdapter,
    SearchVerificationPipelineAdapter,
)
from utils.url_extractor import extract_content, get_claims

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("groq package not installed — Groq enhancement disabled")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class MockModel(torch.nn.Module):
    """Lightweight mock model used when the real model cannot be loaded.

    Returns calibrated neutral outputs rather than hardcoded extremes so that
    the downstream fusion step can use external-verification signals to drive
    the final verdict, instead of being anchored to a broken fixed score.
    """
    def __init__(self):
        super().__init__()
        self.truth = torch.nn.Linear(1, 1)
        self.ai = torch.nn.Linear(1, 2)
        self.bias = torch.nn.Linear(1, 3)

    def forward(self, **kwargs):
        # Neutral priors — external verification will shift these.
        # truth=0.55  (slight lean toward true, not fake)
        # ai logits: [0.75, 0.25] → ~25% AI probability after softmax
        # bias logits: [0.6, 0.3, 0.1] → low-bias class wins
        return {
            "truth": torch.tensor([0.55]),
            "ai": torch.tensor([[0.75, 0.25]]),
            "bias": torch.tensor([[0.6, 0.3, 0.1]])
        }


class MockFeatureExtractor:
    """Fallback when FeatureExtractor deps (spacy, gpt2) are unavailable."""
    def extract_all(self, text):
        words = text.split()
        word_count = len(words)
        unique = len(set(w.lower() for w in words))
        lexical_diversity = unique / max(word_count, 1)

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sent_lens = [len(s.split()) for s in sentences]
        sent_len_var = float(np.var(sent_lens)) if len(sent_lens) > 1 else 0.0

        return {
            "sent_len_var": round(sent_len_var, 2),
            "lexical_diversity": round(lexical_diversity, 4),
            "repetition_score": 0.05,
            "perplexity": round(45.0 + np.random.uniform(-10, 20), 1),
        }


class InferenceEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Groq client (Expert Reasoning enhancement)
        self.groq_client = None
        self.groq_model = "llama-3.3-70b-versatile"
        groq_key = os.getenv("GROQ_API_KEY", "")
        if HAS_GROQ and groq_key:
            try:
                self.groq_client = Groq(api_key=groq_key)
                logger.info(f"Groq client initialized ({self.groq_model})")
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

        # Load model + tokenizer
        self.base_model_name = "roberta-base"
        trained_path = os.path.join(os.path.dirname(__file__), "..", "models", "roberta-finetuned")
        self.tokenizer = None
        self.model = None
        if os.path.exists(trained_path):
            self._load_model(trained_path)
        else:
            self._load_model(model_path)

        # Feature extractor
        if HAS_FEATURES:
            try:
                self.feature_extractor = FeatureExtractor()
                logger.info("FeatureExtractor loaded (spacy + gpt2 perplexity)")
            except Exception as e:
                logger.warning(f"FeatureExtractor failed: {e} — using mock")
                self.feature_extractor = MockFeatureExtractor()
        else:
            self.feature_extractor = MockFeatureExtractor()

        # Trust agent
        try:
            self.trust_agent = TrustAgent()
            logger.info("TrustAgent loaded (sentence-transformers)")
        except Exception as e:
            logger.warning(f"TrustAgent failed: {e} — external verification disabled")
            self.trust_agent = None

        # News search agent (lazy — shares the sentence-transformer model with TrustAgent)
        try:
            self.news_agent = NewsSearchAgent()
            logger.info("NewsSearchAgent initialized")
        except Exception as e:
            logger.warning(f"NewsSearchAgent failed: {e}")
            self.news_agent = None

        self.sarcasm_detector = SarcasmDetector()
        self.bias_adapter = HuggingFaceBiasClassifierAdapter()
        self.factcheck_adapter = ExternalFactCheckAdapter()
        self.search_verification_adapter = SearchVerificationPipelineAdapter()

    def _load_model(self, model_path):
        """Load multi-task RoBERTa or fall back to MockModel."""
        if HAS_TRANSFORMERS and TruthGuardMultiTaskModel:
            try:
                local_path = model_path or os.path.join(os.path.dirname(__file__), "..", "models", "roberta-finetuned")
                source = local_path if os.path.exists(os.path.join(local_path, "model.safetensors")) else self.base_model_name
                
                # Load tokenizer - use local if config exists, else base
                tokenizer_src = local_path if os.path.exists(os.path.join(local_path, "tokenizer_config.json")) else self.base_model_name
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load config - use local if exists, else base
                config_src = local_path if os.path.exists(os.path.join(local_path, "config.json")) else self.base_model_name
                config = AutoConfig.from_pretrained(config_src)
                self.model = TruthGuardMultiTaskModel(config, source)
                
                # Load custom heads if they exist
                heads_path = os.path.join(local_path, "custom_heads.pt")
                if os.path.exists(heads_path):
                    state_dict = torch.load(heads_path, map_location=self.device, weights_only=True)
                    self.model.truth_head.load_state_dict(state_dict['truth_head'])
                    self.model.ai_head.load_state_dict(state_dict['ai_head'])
                    self.model.bias_head.load_state_dict(state_dict['bias_head'])
                    logger.info("Custom trained heads loaded successfully")
                
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"[MODEL MODE: REAL] Model loaded: {source} (Tokenizer: {tokenizer_src})")
                return
            except Exception as e:
                logger.warning(f"Model load failed ({e}) — falling back to mock")

        # Check if production requires a real model
        if os.getenv("REQUIRE_REAL_MODEL", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "REQUIRE_REAL_MODEL is set but no real model could be loaded. "
                "Ensure model weights exist at backend/models/roberta-finetuned."
            )

        # Fallback
        try:
            from transformers import AutoTokenizer as AT
            self.tokenizer = AT.from_pretrained("gpt2")
            # GPT-2 has no pad token — set eos as pad so padding calls don't crash
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            self.tokenizer = None
        self.model = MockModel().to(self.device)
        self._is_mock = True  # flag so _tokenize can skip padding
        logger.warning(
            "[MODEL MODE: MOCK] Real model not available — using MockModel. "
            "Inference results are NOT real. Set REQUIRE_REAL_MODEL=true to fail fast."
        )

    def _tokenize(self, text):
        """Tokenize text, handling both real and mock tokenizers."""
        if self.tokenizer is None:
            return {"input_ids": torch.tensor([[0]]).to(self.device)}
        # MockModel ignores token values entirely — skip padding to avoid
        # issues with tokenizers that have no native pad token (e.g. GPT-2).
        if getattr(self, "_is_mock", False):
            return {"input_ids": torch.tensor([[0]]).to(self.device)}
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def call_expert_llm(self, text):
        """Groq enhancement for higher fidelity analysis using Llama-3 (Expert Reasoning)."""
        if not self.groq_client:
            return None

        system_prompt = """You are TruthGuard AI — a multi-modal verification engine.
Perform THREE completely independent analyses on the text below.

CRITICAL ISOLATION RULE:
These three scores (Truth, AI, Bias) must be MATHEMATICALLY INDEPENDENT. No score should influence any other score.

━━━ ANALYSIS 1: TRUTH SCORE ━━━
Definition: How factually accurate is the content? (Scale: 0-100)
- Accuracy ONLY. Do not penalize for AI usage or vague writing.
Guide: 90-100 (Verified), 70-89 (Mostly Correct), 40-69 (Mixed), 10-39 (False), 0-9 (Completely False).

━━━ ANALYSIS 2: AI LIKELIHOOD ━━━
Definition: Probability this text is AI-generated (0-100).
- Signals: Lexical predictability, "crucial", "delve", robotic structure.

━━━ ANALYSIS 3: BIAS LEVEL ━━━
Definition: Slant or loaded framing (0-100).

Return valid JSON ONLY (no markdown blocks):
{
  "truth_score": <0-100 or null>,
  "truth_reasoning": "<specific sources>",
  "ai_likelihood": <0-100>,
  "ai_reasoning": "<signals>",
  "bias_level": <0-100>,
  "bias_reasoning": "<framing>",
  "final_verdict": "<VERIFIED | MOSTLY VERIFIED | PARTIALLY VERIFIED | FALSE | UNVERIFIABLE>"
}
"""
        try:
            import json
            completion = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:3000]},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Score sanitation
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize(value: float, default: float = 0.5) -> float:
        """Clamp a score to [0, 1] and replace NaN/inf with a safe default."""
        if not math.isfinite(value):
            return default
        return max(0.0, min(1.0, value))

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator == 0:
            return default
        return numerator / denominator

    def _sentence_lengths(self, text: str):
        chunks = [s.strip() for s in re.split(r"[.!?]+", text or "") if s.strip()]
        return [len(c.split()) for c in chunks if c]

    def _compute_style_metrics(self, text: str, raw_features: dict) -> dict:
        sentence_lengths = self._sentence_lengths(text)
        mean_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        std_len = float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0
        cv = self._safe_ratio(std_len, mean_len, default=0.0)
        burstiness = self._sanitize(min(1.0, cv / 1.2), default=0.3)
        sentence_variance = float(raw_features.get("sent_len_var", std_len**2))
        sentence_uniformity = self._sanitize(1.0 - min(1.0, sentence_variance / 60.0))
        lexical_diversity = self._sanitize(float(raw_features.get("lexical_diversity", 0.5)))
        lexical_repetition = self._sanitize(float(raw_features.get("repetition_score", 0.0)))
        stylometric_consistency = self._sanitize((0.55 * sentence_uniformity) + (0.45 * (1.0 - lexical_diversity)))
        return {
            "burstiness": round(burstiness, 4),
            "sentence_length_uniformity": round(sentence_uniformity, 4),
            "lexical_repetition": round(lexical_repetition, 4),
            "stylometric_consistency": round(stylometric_consistency, 4),
            "sentence_variance": round(sentence_variance, 4),
            "lexical_diversity": round(lexical_diversity, 4),
            "mean_sentence_length": round(mean_len, 4),
        }

    @staticmethod
    def _detect_conspiracy(content: str) -> bool:
        markers = [
            # Classic markers
            "new world order",
            "cover-up",
            "they don't want you to know",
            "secret cabal",
            "hidden truth",
            "stolen election",
            # Mass surveillance / hidden technology
            "mass surveillance",
            "surveillance tool",
            "surveillance state",
            "surveillance units",
            "designed from the beginning as",
            # AI/android replacement
            "replaced by ai",
            "ai-controlled android",
            "android duplicate",
            # Depopulation / engineered events
            "depopulation plan",
            "depopulation agenda",
            "engineered pandemic",
            "plandemic",
            "triggered deliberately",
            "seismic weapon",
            "weather modification",
            "controlling the weather",
            "modifying the global weather",
            # Extraterrestrial / fringe cosmology
            "crop circle",
            "interdimensional",
            "chemtrail",
            "reptilian",
            "flat earth",
            "ice wall surrounding",
            "off-planet",
            "beyond low earth orbit",
            "space barrier",
            "turned back by a barrier",
            # Hidden power structures
            "hidden group",
            "twelve families",
            "same twelve families",
            "secret bloodline",
            "shadow elite",
            "secret off-planet",
            "chosen in advance by",
            "chosen by the same",
            "deep state",
            "globalist plan",
            "globalist agenda",
            "banking system is a",
            "debt servitude",
            "humanity in permanent",
            "keep humanity in",
            # Fabricated history / reality
            "underground construction project",
            "never existed and were invented",
            "human life energy",
            "genetically engineered to serve as",
            "medieval historians fabricated",
            "fabricated three centuries",
            "614 ce never",
            "population statistics are fabricated",
            "religious texts were edited",
            "centralized group to control",
            "manufactured lie",
            "scientific consensus on evolution is",
            # Other fringe
            "bioweapon lab",
            "bioweapon program",
            "bioweapon plot",
            "false flag",
            "crisis actor",
            "mind control",
            "population control",
            "microchip",
            "5g tower",
            "lab-leaked",
            "suppressed technology",
            "government suppression",
        ]
        text = (content or "").lower()
        return any(marker in text for marker in markers)

    @staticmethod
    def _summarize_trust_signals(verification_signals: list) -> dict:
        if not verification_signals:
            return {
                "trust_agent_confidence": "inconclusive",
                "retrieval_support_score": 0.0,
                "retrieval_contradiction_score": 0.0,
            }

        support = float(np.mean([s.get("retrieval_support_score", s.get("match_score", 0.0)) for s in verification_signals]))
        contradiction = float(np.mean([s.get("retrieval_contradiction_score", 0.0) for s in verification_signals]))
        buckets = [s.get("trust_agent_confidence", "inconclusive") for s in verification_signals]
        # Contradiction threshold remains strict; support threshold is softened to match
        # the calibrated VERIFIED_SUPPORT_THRESHOLD (0.60) so trust_agent_confidence
        # can reflect "support" at a lower bar than before.
        if "contradiction" in buckets and contradiction >= 0.72:
            confidence = "contradiction"
        elif "support" in buckets and support >= 0.60:
            confidence = "support"
        else:
            confidence = "inconclusive"

        return {
            "trust_agent_confidence": confidence,
            "retrieval_support_score": max(0.0, min(1.0, support)),
            "retrieval_contradiction_score": max(0.0, min(1.0, contradiction)),
        }

    @staticmethod
    def _route_primary_verdict(
        *,
        claim_type: str,
        claim_verifiable: bool,
        opinion_detected: bool,
        sarcasm_detected: bool,
        factual_claim: bool,
        trust_agent_confidence: str,
        retrieval_support_score: float,
        retrieval_contradiction_score: float,
        bias_score: float,
        manipulation_score: float,
        conspiracy_flag: bool,
        ai_generated_score: float,
    ) -> dict:
        claim_type = claim_type or "MIXED"
        text_type_map = {
            "FACTUAL_CLAIM": "factual_claim",
            "OPINION": "opinion",
            "PERSUASIVE_COPY": "persuasive/manipulative",
            "SARCASTIC": "sarcastic/satirical",
            "SPECULATIVE": "speculative/unverifiable",
            "MIXED": "mixed",
        }
        text_type_detected = text_type_map.get(claim_type, "mixed")

        if sarcasm_detected or claim_type == "SARCASTIC":
            return {
                "primary_verdict": "SATIRE_OR_SARCASM",
                "triggered_rule": "STAGE_2_SARCASM_GATE",
                "verifiability_result": "non_factual_sarcasm",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": False,
                "why_verdict_chosen": "Sarcasm cues triggered before factual evaluation.",
            }

        if opinion_detected or claim_type == "OPINION":
            return {
                "primary_verdict": "OPINION",
                "triggered_rule": "STAGE_2_OPINION_GATE",
                "verifiability_result": "non_factual_opinion",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": False,
                "why_verdict_chosen": "Subjective/preference language gated this away from factual truth scoring.",
            }

        if claim_type == "SPECULATIVE" or not claim_verifiable:
            return {
                "primary_verdict": "UNVERIFIED_CLAIM",
                "triggered_rule": "STAGE_2_VERIFIABILITY_GATE",
                "verifiability_result": "non_verifiable_claim",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": False,
                "why_verdict_chosen": "Speculative or unverifiable claim cannot be resolved factually.",
            }

        if factual_claim:
            if trust_agent_confidence == "support" and retrieval_support_score >= 0.72 and retrieval_contradiction_score < 0.60:
                return {
                    "primary_verdict": "VERIFIED_FACT",
                    "triggered_rule": "STAGE_3_FACTUAL_SUPPORT",
                    "verifiability_result": "verifiable_factual_claim",
                    "text_type_detected": text_type_detected,
                    "factual_verdict_locked": True,
                    "why_verdict_chosen": "Strong support retrieval confidence on a verifiable factual claim.",
                }
            if trust_agent_confidence == "contradiction" and retrieval_contradiction_score >= 0.72:
                return {
                    "primary_verdict": "FALSE_FACT",
                    "triggered_rule": "STAGE_3_FACTUAL_CONTRADICTION",
                    "verifiability_result": "verifiable_factual_claim",
                    "text_type_detected": text_type_detected,
                    "factual_verdict_locked": True,
                    "why_verdict_chosen": "Strong contradictory evidence on a verifiable factual claim.",
                }
            return {
                "primary_verdict": "UNVERIFIED_CLAIM",
                "triggered_rule": "STAGE_3_FACTUAL_INCONCLUSIVE",
                "verifiability_result": "verifiable_but_inconclusive",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": True,
                "why_verdict_chosen": "Factual claim did not reach support/contradiction thresholds.",
            }

        if bias_score >= 0.70:
            return {
                "primary_verdict": "BIASED_CONTENT",
                "triggered_rule": "STAGE_4_BIAS_OVERRIDE",
                "verifiability_result": "non_factual_mixed",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": False,
                "why_verdict_chosen": "High slant/bias signal on non-factual text.",
            }
        if manipulation_score >= 0.70:
            return {
                "primary_verdict": "MANIPULATIVE_CONTENT",
                "triggered_rule": "STAGE_4_MANIPULATION_OVERRIDE",
                "verifiability_result": "non_factual_mixed",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": False,
                "why_verdict_chosen": "High coercive/manipulative signal on non-factual text.",
            }
        if conspiracy_flag:
            return {
                "primary_verdict": "CONSPIRACY_OR_EXTRAORDINARY_CLAIM",
                "triggered_rule": "STAGE_4_CONSPIRACY_OVERRIDE",
                "verifiability_result": "non_factual_mixed",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": False,
                "why_verdict_chosen": "Extraordinary/conspiracy markers dominated non-factual analysis.",
            }
        if ai_generated_score >= 0.95:
            return {
                "primary_verdict": "LIKELY_AI_GENERATED",
                "triggered_rule": "STAGE_4_AI_AUXILIARY",
                "verifiability_result": "non_factual_mixed",
                "text_type_detected": text_type_detected,
                "factual_verdict_locked": False,
                "why_verdict_chosen": "Very high AI-likelihood used only as secondary non-factual signal.",
            }
        return {
            "primary_verdict": "MIXED_ANALYSIS",
            "triggered_rule": "STAGE_4_MIXED_FALLBACK",
            "verifiability_result": "non_factual_mixed",
            "text_type_detected": text_type_detected,
            "factual_verdict_locked": False,
            "why_verdict_chosen": "No dominant non-factual override crossed threshold.",
        }

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------
    def analyze(self, text_or_url):
        start_time = time.time()

        # 1. Content extraction
        logger.info(f"Analyzing input ({len(text_or_url)} chars)")
        content = extract_content(text_or_url)
        if not content or len(content.strip()) < 10:
            logger.warning("Content extraction returned empty/too-short text")
            return None

        # 2. Multi-head model inference
        logger.info("Running model inference...")
        inputs = self._tokenize(content)
        with torch.no_grad():
            outputs = self.model(**inputs)

        truth_score = self._sanitize(float(outputs["truth"].cpu().item()))
        ai_probs = torch.softmax(outputs["ai"], dim=1).cpu().numpy()[0]
        raw_model_ai_score = self._sanitize(float(ai_probs[1]))
        ai_generated_score = raw_model_ai_score
        bias_probs = torch.softmax(outputs["bias"], dim=1).cpu().numpy()[0]
        bias_score = self._sanitize(float(np.argmax(bias_probs) / 2.0))

        # 2.5 Optional LLM enhancement (Groq/Llama)
        self._expert_reasoning = None
        self._truth_reasoning = None
        oa_res = self.call_expert_llm(content)
        if oa_res:
            logger.info("Fusing independent LLM insights (Groq)")
            
            ts_raw = oa_res.get("truth_score")
            ts_val = 0.5
            if ts_raw is not None:
                ts_val = float(ts_raw)
                if ts_val > 1.0: ts_val /= 100.0

            ai_raw = oa_res.get("ai_likelihood", oa_res.get("ai_generated_score"))
            ai_val = 0.5
            if ai_raw is not None:
                ai_val = float(ai_raw)
                if ai_val > 1.0: ai_val /= 100.0

            bias_raw = oa_res.get("bias_level", oa_res.get("bias_score"))
            bias_val = 0.5
            if bias_raw is not None:
                bias_val = float(bias_raw)
                if bias_val > 1.0: bias_val /= 100.0

            truth_score = (truth_score * 0.2) + (ts_val * 0.8)
            # AI score uses a lower OpenAI blend than truth/bias so local stylometric
            # feature blending remains the dominant signal for AI likelihood calibration.
            raw_model_ai_score = (raw_model_ai_score * AI_RAW_MODEL_WEIGHT) + (ai_val * AI_OPENAI_BLEND_WEIGHT)
            bias_score = (bias_score * 0.2) + (bias_val * 0.8)
            
            if "ai_reasoning" in oa_res:
                self._expert_reasoning = oa_res["ai_reasoning"]
            if "truth_reasoning" in oa_res:
                self._truth_reasoning = oa_res["truth_reasoning"]

        # 3. Feature extraction (perplexity + stylometry)
        logger.info("Extracting features...")
        raw_features = self.feature_extractor.extract_all(content)
        style_metrics = self._compute_style_metrics(content, raw_features)
        ai_generated_score = compute_ai_likelihood(raw_model_ai_score, raw_features, style_metrics, text=content)

        # 4. External verification
        logger.info("Running external verification...")
        verification_signals = []
        if self.trust_agent:
            claims = get_claims(content)
            for claim in claims:
                verification_signals.append(self.trust_agent.verify_claim(claim))

        avg_external_cred = (
            float(np.mean([s["agent_credibility"] for s in verification_signals]))
            if verification_signals
            else 0.5
        )

        # 4.5 News consistency score
        news_consistency_score = None
        if self.news_agent:
            try:
                news_consistency_score = self.news_agent.get_consistency_score(content[:500])
                logger.info(f"News consistency score: {news_consistency_score:.2f}")
            except Exception as exc:
                logger.warning(f"NewsSearchAgent.get_consistency_score error: {exc}")

        claim_type_signal = classify_claim_type(
            content,
            model_signals={
                "truth_score": truth_score,
                "bias_score": bias_score,
                "ai_score": ai_generated_score,
            },
            style_metrics=style_metrics,
        )
        claim_type = claim_type_signal.get("claim_type", "MIXED")
        sarcasm_signal = self.sarcasm_detector.detect(content)
        sarcasm_detected = bool(sarcasm_signal.get("sarcasm", False)) or claim_type == "SARCASTIC"
        verifiability = assess_claim_verifiability(content, claim_type=claim_type)
        conspiracy_flag = self._detect_conspiracy(content)
        bias_signal = detect_bias(content, model_bias_score=bias_score)
        bias_score = self._sanitize(max(float(bias_score), float(bias_signal.get("score", 0.0))))
        manipulation_signal = detect_manipulation(content, style_metrics)
        manipulation_score = self._sanitize(float(manipulation_signal.get("score", 0.0)))
        factual_claim = claim_type == "FACTUAL_CLAIM"
        trust_summary = self._summarize_trust_signals(verification_signals)

        # 5. Fusion & self-check (STRICTLY INDEPENDENT)
        # When no external signals are available, we default to the model truth score.
        if verification_signals:
            external_truth_hint = self._sanitize(
                0.5 + (
                    trust_summary["retrieval_support_score"] - trust_summary["retrieval_contradiction_score"]
                ) * 0.5
            )
            final_truth_score = self._sanitize((truth_score * 0.7) + (external_truth_hint * 0.3))
        else:
            final_truth_score = self._sanitize(truth_score)

        # AI Likelihood MUST NOT lower the Truth Score. Removed legacy penalty logic.

        # Confidence: distance of final truth from the midpoint, clamped to [0.4, 0.95]
        uncertainty_penalty = 0.0
        if not verification_signals:
            uncertainty_penalty += 0.1
        
        confidence = self._sanitize(0.4 + 0.55 * abs(final_truth_score - 0.5) * 2 - uncertainty_penalty)

        # Intermediate scores reused across routing + debug
        sarcasm_score_val = float(sarcasm_signal.get("score", 0.0))
        # Continuous opinion score from claim-type classifier (not binary flag)
        opinion_score_val = self._sanitize(float(claim_type_signal.get("scores", {}).get("OPINION", 0.0)))

        # 5.5 Enriched analysis (staged verdict routing)
        routing = stage_route_primary_verdict(
            claim_type=claim_type,
            claim_verifiable=bool(verifiability.get("claim_verifiable", True)),
            opinion_detected=bool(verifiability.get("opinion_detected", False)),
            sarcasm_detected=sarcasm_detected,
            factual_claim=factual_claim,
            trust_agent_confidence=trust_summary["trust_agent_confidence"],
            retrieval_support_score=trust_summary["retrieval_support_score"],
            retrieval_contradiction_score=trust_summary["retrieval_contradiction_score"],
            bias_score=bias_score,
            manipulation_score=manipulation_score,
            conspiracy_flag=conspiracy_flag,
            ai_generated_score=ai_generated_score,
            sarcasm_score=sarcasm_score_val,
            opinion_score=opinion_score_val,
            model_truth_score=final_truth_score,
        )
        primary_verdict = routing["primary_verdict"]
        risk_level = "Medium"
        recommendation = "Cross-check with trusted sources."
        if primary_verdict in {"FALSE_FACT", "MANIPULATIVE_CONTENT", "CONSPIRACY_OR_EXTRAORDINARY_CLAIM"}:
            risk_level = "High"
            recommendation = "High-risk content. Avoid sharing without independent verification."
        elif primary_verdict == "VERIFIED_FACT":
            risk_level = "Low"
            recommendation = "Content appears factually grounded."

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Analysis complete in {elapsed_ms}ms")

        # 6. Build response matching the frontend contract
        signals = []
        for vs in verification_signals:
            # Filter: Don't show the signal unless it's actually relevant to the claim
            if vs.get("match_score", 0) < 0.45:
                continue
            signals.append({
                "source": vs.get("top_match", vs.get("claim", "Unknown source")),
                "verified": vs.get("agent_credibility", 0.5) > 0.6,
                "confidence": round(vs.get("agent_credibility", 0.5), 2),
            })

        truth_sources = [s.get("source", "Unknown source") for s in signals]

        raw_classifier_outputs = {
            "truth_model_raw": round(truth_score, 4),
            "ai_model_raw": round(raw_model_ai_score, 4),
            "ai_likelihood_blended": round(ai_generated_score, 4),
            "bias_model_raw": round(bias_score, 4),
            "external_credibility_mean": round(avg_external_cred, 4),
            "sarcasm_score": round(sarcasm_score_val, 4),
            "opinion_score": round(opinion_score_val, 4),
            "claim_verifiable": bool(verifiability.get("claim_verifiable", True)),
            "opinion_detected": bool(verifiability.get("opinion_detected", False)),
            "verifiability_reason": verifiability.get("reason", "unknown"),
            "claim_type": claim_type,
            "claim_type_signals": claim_type_signal.get("signals", []),
            "claim_type_scores": claim_type_signal.get("scores", {}),
            "bias_indicators": bias_signal.get("indicators", []),
            "manipulation_indicators": manipulation_signal.get("indicators", []),
            "trust_agent_confidence": trust_summary["trust_agent_confidence"],
            "retrieval_support_score": round(trust_summary["retrieval_support_score"], 4),
            "retrieval_contradiction_score": round(trust_summary["retrieval_contradiction_score"], 4),
        }

        verifiability_score = self._compute_verifiability_score(
            claim_type=claim_type,
            verifiability=verifiability,
            trust_summary=trust_summary,
        )
        dimensions = {
            "truth_score": int(round(final_truth_score * 100)),
            "verifiability": int(round(verifiability_score * 100)),
            "ai_likelihood": int(round(ai_generated_score * 100)),
            "bias_score": int(round(bias_score * 100)),
            "manipulation_score": int(round(manipulation_score * 100)),
            "sarcasm_score": int(round(sarcasm_score_val * 100)),
            "opinion_score": int(round(opinion_score_val * 100)),
            "sarcasm": sarcasm_detected,
            "conspiracy_flag": conspiracy_flag,
        }

        dimension_buckets = {
            "truth_score": to_bucket(final_truth_score),
            "verifiability": to_bucket(verifiability_score),
            "ai_likelihood": to_bucket(ai_generated_score),
            "bias_score": to_bucket(bias_score),
            "manipulation_score": to_bucket(manipulation_score),
            "sarcasm_score": to_bucket(sarcasm_score_val),
            "opinion_score": to_bucket(opinion_score_val),
        }

        debug = {
            "text_type_detected": routing.get("text_type_detected", "mixed"),
            "verifiability_result": routing.get("verifiability_result", "unknown"),
            "trust_agent_confidence": trust_summary["trust_agent_confidence"],
            "retrieval_support_score": round(trust_summary["retrieval_support_score"], 4),
            "retrieval_contradiction_score": round(trust_summary["retrieval_contradiction_score"], 4),
            "fusion_weights": {
                "model_truth_weight": 0.7 if verification_signals else 1.0,
                "retrieval_weight": 0.3 if verification_signals else 0.0,
            },
            "triggered_rule": routing.get("triggered_rule"),
            "detector_fired_first": routing.get("triggered_rule", "unknown"),
            "why_verdict_chosen": routing.get("why_verdict_chosen", ""),
            "final_rule_triggered": routing.get("triggered_rule"),
            # Calibration-visibility fields (per problem statement requirements)
            "threshold_values_used": routing.get("threshold_values_used", {}),
            "detector_confidences": {
                "truth_score": round(float(final_truth_score), 4),
                "ai_likelihood": round(float(ai_generated_score), 4),
                "bias_score": round(float(bias_score), 4),
                "manipulation_score": round(float(manipulation_score), 4),
                "sarcasm_score": round(sarcasm_score_val, 4),
                "opinion_score": round(opinion_score_val, 4),
                "retrieval_support": round(trust_summary["retrieval_support_score"], 4),
                "retrieval_contradiction": round(trust_summary["retrieval_contradiction_score"], 4),
            },
            "trust_support_margin": routing.get("trust_support_margin", 0.0),
            "contradiction_margin": routing.get("contradiction_margin", 0.0),
            "sarcasm_rule_hits": sarcasm_signal.get("sarcasm_rule_hits", []),
            "bias_rule_hits": bias_signal.get("bias_rule_hits", []),
            "manipulation_rule_hits": manipulation_signal.get("manipulation_rule_hits", []),
            "raw_intermediate_scores": {
                "truth_score": round(float(final_truth_score), 4),
                "verifiability": round(verifiability_score, 4),
                "ai_likelihood_score": round(float(ai_generated_score), 4),
                "bias_score": round(float(bias_score), 4),
                "manipulation_score": round(float(manipulation_score), 4),
                "sarcasm_score": round(sarcasm_score_val, 4),
                "opinion_score": round(opinion_score_val, 4),
                "conspiracy_flag": bool(conspiracy_flag),
                "claim_type": claim_type,
            },
        }

        # Only expose news_consistency_score if it's statistically significant (>0.35)
        is_news_relevant = news_consistency_score is not None and news_consistency_score > 0.35

        return {
            "truth_score": round(final_truth_score, 2),
            "ai_generated_score": round(ai_generated_score, 2),
            "bias_score": round(bias_score, 2),
            "credibility_score": round(avg_external_cred, 2),
            "confidence_score": round(confidence, 2),
            "explanation": self._generate_explanation(
                final_truth_score, 
                ai_generated_score, 
                bias_score, 
                raw_features, 
                getattr(self, "_expert_reasoning", None),
                getattr(self, "_truth_reasoning", None),
                primary_verdict=primary_verdict,
                claim_type=claim_type,
                manipulation_score=manipulation_score,
                sarcasm_detected=sarcasm_detected,
                opinion_detected=bool(verifiability.get("opinion_detected", False)),
            ),
            "features": {
                "perplexity": round(raw_features.get("perplexity", 0), 1),
                "stylometry": {
                    "sentence_length_variance": round(raw_features.get("sent_len_var", 0), 2),
                    "repetition_score": round(raw_features.get("repetition_score", 0), 4),
                    "lexical_diversity": round(raw_features.get("lexical_diversity", 0), 4),
                    "burstiness": round(style_metrics.get("burstiness", 0), 4),
                },
            },
            "primary_verdict": primary_verdict,
            "confidence": int(round(confidence * 100)),
            "dimensions": dimensions,
            "expanded_analysis": {
                "truth_score": {
                    "explanation": self._explain_truth_score(
                        final_truth_score, claim_type, trust_summary, routing, verifiability
                    ),
                    "evidence": self._truth_reasoning or (
                        "Weighted blend of model truth score and external credibility signals."
                        if not truth_sources
                        else f"Cross-source agreement detected across {len(truth_sources)} signal(s)."
                    ),
                    "sources": truth_sources,
                },
                "ai_likelihood": {
                    "explanation": self._explain_ai_likelihood(ai_generated_score, raw_features, style_metrics),
                    "indicators": [
                        f"perplexity: {round(raw_features.get('perplexity', 0), 1)} (lower = more fluent / AI-like)",
                        f"burstiness: {style_metrics.get('burstiness', 0)} (low = uniform AI rhythm)",
                        f"sentence uniformity: {style_metrics.get('sentence_length_uniformity', 0)} (high = AI)",
                        f"lexical repetition: {style_metrics.get('lexical_repetition', 0)} (high = AI)",
                        f"stylometric consistency: {style_metrics.get('stylometric_consistency', 0)}",
                    ],
                },
                "bias_score": {
                    "explanation": self._explain_bias(bias_score, bias_signal),
                    "indicators": self._format_bias_indicators(bias_signal),
                },
                "manipulation_score": {
                    "explanation": self._explain_manipulation(manipulation_score, manipulation_signal),
                    "indicators": self._format_manipulation_indicators(manipulation_signal),
                },
                "opinion_score": {
                    "explanation": self._explain_opinion(opinion_score_val, claim_type, claim_type_signal, verifiability),
                    "indicators": [
                        f"claim classifier type: {claim_type}",
                        *[f"signal: {s}" for s in claim_type_signal.get("signals", [])[:4]],
                        f"verifiability reason: {verifiability.get('reason', 'unknown')}",
                    ],
                },
                "verifiability": {
                    "explanation": self._explain_verifiability(verifiability_score, verifiability, claim_type, trust_summary),
                    "indicators": self._format_verifiability_indicators(verifiability, trust_summary),
                },
            },
            "signals": signals,
            **({"news_consistency_score": round(news_consistency_score, 2)} if is_news_relevant else {}),
            "triggered_rule": routing.get("triggered_rule"),
            "verdict": primary_verdict.replace("_", " ").title(),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "key_factors": sorted(
                set(
                    [
                        routing.get("why_verdict_chosen", ""),
                        f"claim_type={claim_type}",
                        f"trust={trust_summary['trust_agent_confidence']}",
                    ]
                )
            ),
            "metadata": {
                "model": "roberta-finetuned + gpt2-perplexity",
                "latency_ms": elapsed_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_metadata": {
                    "aggregation_rule": routing.get("triggered_rule", "STAGE_4_MIXED_FALLBACK"),
                    "dimension_buckets": dimension_buckets,
                    "claim_type": claim_type,
                    "debug": debug,
                    "raw_classifier_outputs": raw_classifier_outputs,
                    "feature_metrics": {
                        "perplexity": round(raw_features.get("perplexity", 0), 4),
                        "burstiness": style_metrics.get("burstiness", 0),
                        "lexical_diversity": style_metrics.get("lexical_diversity", 0),
                        "sentence_variance": style_metrics.get("sentence_variance", 0),
                    },
                },
            },
            "dimension_buckets": dimension_buckets,
            "claim_type": claim_type,
            "debug": debug,
        }

    def _compute_verifiability_score(
        self,
        *,
        claim_type: str,
        verifiability: dict,
        trust_summary: dict,
    ) -> float:
        """Continuous verifiability score derived from claim type, signals, and retrieval."""
        claim_verifiable = bool(verifiability.get("claim_verifiable", True))
        reason = verifiability.get("reason", "")
        opinion_detected = bool(verifiability.get("opinion_detected", False))

        if not claim_verifiable:
            if opinion_detected:
                return 0.12
            if reason == "speculative_or_unconfirmed_claim":
                return 0.22
            if reason == "marketing_or_promotional_claim":
                return 0.18
            if reason == "recent_or_local_hard_to_source_claim":
                return 0.30
            return 0.25

        support = trust_summary.get("retrieval_support_score", 0.0)
        if claim_type == "FACTUAL_CLAIM":
            # 0.65 base + up to 0.30 boost from retrieval support
            base = 0.65 + (0.30 * min(1.0, support))
        elif claim_type == "MIXED":
            base = 0.55 + (0.15 * min(1.0, support))
        elif claim_type == "SPECULATIVE":
            base = 0.25
        else:
            base = 0.45 + (0.10 * min(1.0, support))

        return self._sanitize(base)

    @staticmethod
    def _explain_truth_score(
        score: float,
        claim_type: str,
        trust_summary: dict,
        routing: dict,
        verifiability: dict,
    ) -> str:
        verdict = routing.get("primary_verdict", "MIXED_ANALYSIS")
        support = round(trust_summary.get("retrieval_support_score", 0.0), 2)
        contradiction = round(trust_summary.get("retrieval_contradiction_score", 0.0), 2)
        confidence = trust_summary.get("trust_agent_confidence", "inconclusive")
        pct = int(round(score * 100))

        if verdict == "VERIFIED_FACT":
            return (
                f"The model assigns a truth score of {pct}% — indicating strong factual grounding. "
                f"External retrieval returned {confidence} signal (support={support}, contradiction={contradiction}). "
                "Claims of this type are well-supported in cross-referenced knowledge bases."
            )
        if verdict == "FALSE_FACT":
            return (
                f"The truth score of {pct}% is very low, reflecting strong contradictory evidence. "
                f"External retrieval returned contradiction signals (support={support}, contradiction={contradiction}). "
                "The claim appears to conflict with established verified knowledge."
            )
        if verdict == "OPINION":
            return (
                f"Because this text is classified as an opinion, the truth score ({pct}%) reflects "
                "the model's factual alignment estimate but does not imply the claim is objectively false. "
                "Opinions are not fact-checked — they express personal perspectives."
            )
        if verdict in {"BIASED_CONTENT", "MANIPULATIVE_CONTENT"}:
            return (
                f"The truth score is {pct}%. While some factual content may be present, "
                "the primary signal is skewed framing or pressure language. "
                "Facts presented in biased or manipulative content may be selectively used or misleading."
            )
        if verdict == "SATIRE_OR_SARCASM":
            return (
                f"Truth score: {pct}%. The text appears to be satirical or sarcastic — "
                "standard factual truth scoring does not directly apply. "
                "The model detected ironic or rhetorical framing inconsistent with literal fact claims."
            )
        if verdict == "UNVERIFIED_CLAIM":
            return (
                f"Truth score: {pct}% — the claim could not be confirmed or denied. "
                f"Retrieval confidence was {confidence} (support={support}, contradiction={contradiction}). "
                "Insufficient evidence exists to assign a definitive factual verdict."
            )
        return (
            f"Truth score: {pct}%. The claim type is '{claim_type.lower().replace('_', ' ')}' and retrieval "
            f"confidence is {confidence} (support={support}). "
            "Multiple signals are present without a single dominant factual result."
        )

    @staticmethod
    def _explain_ai_likelihood(score: float, raw_features: dict, style_metrics: dict) -> str:
        pct = int(round(score * 100))
        ppl = round(raw_features.get("perplexity", 0), 1)
        uniformity = round(style_metrics.get("sentence_length_uniformity", 0.5), 2)
        burstiness = round(style_metrics.get("burstiness", 0.3), 2)

        if score >= 0.80:
            return (
                f"AI likelihood is very high at {pct}%. The text shows multiple machine-writing hallmarks: "
                f"low perplexity ({ppl}), uniform sentence rhythm (uniformity={uniformity}), "
                "and limited stylistic variance. This pattern is common in LLM-generated prose."
            )
        if score >= 0.55:
            return (
                f"Moderate-to-high AI likelihood ({pct}%). Some stylometric signals suggest AI writing patterns: "
                f"perplexity={ppl}, sentence uniformity={uniformity}, burstiness={burstiness}. "
                "The text may be AI-assisted or edited, though it could also reflect a formal human writing style."
            )
        if score >= 0.35:
            return (
                f"Moderate AI likelihood ({pct}%). Stylometric analysis shows mixed signals. "
                f"Perplexity is {ppl} and burstiness is {burstiness}. "
                "The text shows some AI-like patterns but also enough variation to suggest human authorship."
            )
        return (
            f"Low AI likelihood ({pct}%). The text's stylometric profile — perplexity={ppl}, "
            f"burstiness={burstiness} — is consistent with natural human writing. "
            "No strong machine-generation markers were identified."
        )

    @staticmethod
    def _explain_bias(score: float, bias_signal: dict) -> str:
        pct = int(round(score * 100))
        indicators = bias_signal.get("indicators", [])
        rule_hits = bias_signal.get("bias_rule_hits", [])
        hit_count = len(rule_hits)

        if score >= 0.70:
            return (
                f"Bias score is very high at {pct}%. The text contains strong loaded language, "
                f"framing indicators, or demonizing rhetoric ({hit_count} rule signal(s) detected). "
                "This level of slant is likely to shape reader interpretation in a partisan or inflammatory direction."
            )
        if score >= 0.45:
            return (
                f"Bias score: {pct}%. Notable framing cues were found: "
                f"{', '.join(indicators[:3]) if indicators else 'slant patterns detected'}. "
                "The content leans toward ideological or group-based framing that may influence objectivity."
            )
        if score >= 0.25:
            return (
                f"Mild bias indicators present ({pct}%). "
                f"The text uses some loaded or opinionated language but falls below the strong-bias threshold. "
                "It may still lean in one direction but does not appear aggressively partisan."
            )
        return (
            f"Bias level is low ({pct}%). The text's language appears largely neutral "
            "without significant ideological framing, demonizing rhetoric, or group-targeting language."
        )

    @staticmethod
    def _format_bias_indicators(bias_signal: dict) -> list:
        hits = bias_signal.get("bias_rule_hits", [])
        raw = bias_signal.get("indicators", [])
        if hits:
            return [h.replace(":", " → ") for h in hits[:8]]
        return [f"signal: {v}" for v in raw[:8]] if raw else ["No strong bias indicators detected"]

    @staticmethod
    def _explain_manipulation(score: float, manipulation_signal: dict) -> str:
        pct = int(round(score * 100))
        indicators = manipulation_signal.get("indicators", [])
        rule_hits = manipulation_signal.get("manipulation_rule_hits", [])
        has_action_pressure = any(
            h.startswith(("urgency_coercion:", "sales_pressure:", "fomo_social_proof:", "suppressed_info:"))
            for h in rule_hits
        )

        if score >= 0.65:
            return (
                f"High manipulation score ({pct}%). The text uses strong coercive or pressure tactics: "
                f"{', '.join(indicators[:3]) if indicators else 'urgency, fear, or FOMO signals'}. "
                "This content is designed to compel a specific action through psychological pressure."
            )
        if score >= 0.40:
            return (
                f"Moderate manipulation score ({pct}%). "
                + (
                    "Action-pressure cues (urgency, FOMO, or sales pressure) were detected alongside emotional framing. "
                    if has_action_pressure
                    else "Emotional framing was detected but without strong action-pressure cues. "
                )
                + "The text may be persuasive in nature but does not cross into high coercion."
            )
        return (
            f"Low manipulation score ({pct}%). "
            "The text does not exhibit significant coercive urgency, fear appeals, or sales pressure. "
            "Emotional language, if present, is mild and below the manipulation decision threshold."
        )

    @staticmethod
    def _format_manipulation_indicators(manipulation_signal: dict) -> list:
        hits = manipulation_signal.get("manipulation_rule_hits", [])
        raw = manipulation_signal.get("indicators", [])
        if hits:
            return [h.replace(":", " → ") for h in hits[:8]]
        return [f"signal: {v}" for v in raw[:8]] if raw else ["No manipulation cues detected"]

    @staticmethod
    def _explain_opinion(score: float, claim_type: str, claim_type_signal: dict, verifiability: dict) -> str:
        pct = int(round(score * 100))
        opinion_detected = bool(verifiability.get("opinion_detected", False))
        reason = verifiability.get("reason", "")
        signals = claim_type_signal.get("signals", [])

        if claim_type == "OPINION" or opinion_detected:
            return (
                f"Opinion score: {pct}%. The text is classified as an opinion or subjective statement. "
                f"Detected signals include: {', '.join(signals[:3]) if signals else 'preference or value language'}. "
                "Opinions express personal perspectives and are not subject to factual truth-scoring."
            )
        if score >= 0.50:
            return (
                f"Elevated opinion score ({pct}%). While not classified purely as opinion, "
                "the text contains notable evaluative language, comparative judgments, or value claims. "
                "This may mix factual assertions with personal perspectives."
            )
        if score >= 0.25:
            return (
                f"Mild opinion signals detected ({pct}%). The text is primarily factual or neutral "
                "but contains some subjective phrasing or preference language. "
                "These elements are unlikely to dominate the overall factual interpretation."
            )
        return (
            f"Opinion level is low ({pct}%). The text reads as predominantly objective or factual. "
            "No strong subjective opinion framing, preference statements, or value judgments were identified."
        )

    @staticmethod
    def _explain_verifiability(score: float, verifiability: dict, claim_type: str, trust_summary: dict) -> str:
        pct = int(round(score * 100))
        claim_verifiable = bool(verifiability.get("claim_verifiable", True))
        reason = verifiability.get("reason", "")
        opinion_detected = bool(verifiability.get("opinion_detected", False))
        support = round(trust_summary.get("retrieval_support_score", 0.0), 2)
        confidence = trust_summary.get("trust_agent_confidence", "inconclusive")

        if opinion_detected:
            return (
                f"Verifiability: {pct}%. This text is classified as an opinion or subjective statement, "
                "which means it cannot be objectively verified against external facts. "
                "Opinions are not testable — they express value judgments or personal preferences."
            )
        if not claim_verifiable:
            reason_map = {
                "speculative_or_unconfirmed_claim": "The claim contains speculative or hedged language (e.g., 'may', 'might', 'reportedly') making it impossible to confirm definitively.",
                "marketing_or_promotional_claim": "The claim uses marketing language ('revolutionary', 'guaranteed') that lacks falsifiable content.",
                "recent_or_local_hard_to_source_claim": "The claim references local or very recent information that is difficult to cross-reference against established sources.",
            }
            reason_text = reason_map.get(reason, "The claim does not contain enough verifiable anchors to fact-check reliably.")
            return f"Verifiability: {pct}%. {reason_text}"

        if claim_type == "FACTUAL_CLAIM":
            return (
                f"Verifiability: {pct}%. This is a factual claim with testable structure. "
                f"External retrieval returned {confidence} confidence (support score={support}). "
                "The claim can in principle be cross-referenced against verified knowledge bases."
            )
        return (
            f"Verifiability: {pct}%. The text has some factual anchors but its overall structure is '{claim_type.lower().replace('_', ' ')}'. "
            f"Retrieval confidence is {confidence} (support={support}). "
            "Verification is possible but may be partial or inconclusive."
        )

    @staticmethod
    def _format_verifiability_indicators(verifiability: dict, trust_summary: dict) -> list:
        indicators = []
        verifiable = bool(verifiability.get("claim_verifiable", True))
        reason = verifiability.get("reason", "")
        indicators.append(f"claim verifiable: {'yes' if verifiable else 'no'}")
        if reason:
            indicators.append(f"reason: {reason.replace('_', ' ')}")
        indicators.append(f"retrieval confidence: {trust_summary.get('trust_agent_confidence', 'inconclusive')}")
        indicators.append(f"support score: {round(trust_summary.get('retrieval_support_score', 0.0), 3)}")
        indicators.append(f"contradiction score: {round(trust_summary.get('retrieval_contradiction_score', 0.0), 3)}")
        return indicators

    def _generate_explanation(self, truth, ai, bias, features, expert_reasoning=None, truth_reasoning=None,
                              primary_verdict=None, claim_type=None, manipulation_score=0.0,
                              sarcasm_detected=False, opinion_detected=False):
        """Context-aware forensic narrative aligned with the primary verdict."""
        parts = []
        verdict = primary_verdict or "MIXED_ANALYSIS"

        # --- Lead sentence: verdict-driven opener ---
        if verdict == "VERIFIED_FACT":
            parts.append(
                "This claim appears to be factually grounded. "
                "Cross-referenced signals and model inference both support the core assertion."
            )
        elif verdict == "FALSE_FACT":
            parts.append(
                "This claim appears to be factually incorrect. "
                "Strong contradictory signals were detected across multiple verification sources."
            )
        elif verdict == "OPINION":
            parts.append(
                "This text expresses a subjective opinion or personal value judgment rather than a verifiable fact. "
                "Factual truth-scoring does not apply to opinion statements."
            )
        elif verdict == "SATIRE_OR_SARCASM":
            parts.append(
                "This text appears to be satirical or sarcastic in nature. "
                "Rhetorical and ironic framing signals were detected, indicating the content is not meant literally."
            )
        elif verdict == "BIASED_CONTENT":
            parts.append(
                "While factual elements may be present, the language exhibits notable ideological slant, "
                "loaded framing, or group-targeting rhetoric that can shape interpretation beyond neutral reporting."
            )
        elif verdict == "MANIPULATIVE_CONTENT":
            parts.append(
                "The text employs emotional pressure tactics, urgency cues, or coercive language "
                "designed to compel a specific response rather than to inform neutrally."
            )
        elif verdict == "CONSPIRACY_OR_EXTRAORDINARY_CLAIM":
            parts.append(
                "Markers associated with conspiracy theories or extraordinary claims were identified. "
                "These claims often contradict established scientific or historical consensus without credible evidence."
            )
        elif verdict == "LIKELY_AI_GENERATED":
            parts.append(
                "Stylometric analysis indicates a high probability that this text was machine-generated. "
                "Key signals include low perplexity, uniform sentence rhythm, and repetitive structural patterns."
            )
        elif verdict == "UNVERIFIED_CLAIM":
            parts.append(
                "This claim could not be confirmed or denied with available evidence. "
                "Retrieval signals were inconclusive and the factual basis is insufficient for a definitive verdict."
            )
        else:
            parts.append(
                "The analysis returned mixed signals across multiple dimensions. "
                "No single dominant signal reached the threshold for a definitive verdict category."
            )

        # --- Truth reasoning from LLM (if available) ---
        if truth_reasoning and verdict not in {"OPINION", "SATIRE_OR_SARCASM"}:
            parts.append(truth_reasoning)

        # --- AI authorship note ---
        if expert_reasoning and ai > 0.50:
            parts.append(expert_reasoning)
        elif not expert_reasoning and verdict not in {"OPINION", "SATIRE_OR_SARCASM", "BIASED_CONTENT", "MANIPULATIVE_CONTENT"}:
            ppl = features.get("perplexity", 0)
            div = features.get("lexical_diversity", 0.5)
            if ai > 0.80:
                parts.append(
                    f"Highly uniform stylometric markers (perplexity: {ppl:.1f}) strongly suggest machine authorship."
                )
            elif ai > 0.55 and verdict == "LIKELY_AI_GENERATED":
                parts.append(
                    f"Syntactic uniformity (perplexity: {ppl:.1f}) and low lexical diversity ({div:.2f}) "
                    "are consistent with LLM-generated prose."
                )
            elif ai < 0.30 and div > 0.40:
                parts.append("Natural lexical diversity and sentence variance align with authentic human authorship.")

        # --- Bias note for non-bias verdicts ---
        if bias > 0.60 and verdict not in {"BIASED_CONTENT", "MANIPULATIVE_CONTENT", "CONSPIRACY_OR_EXTRAORDINARY_CLAIM"}:
            parts.append(
                "Additionally, loaded or partisan framing was detected — consider this when evaluating the neutrality of the source."
            )

        return " ".join(parts)
