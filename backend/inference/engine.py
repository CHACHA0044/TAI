import math
import torch
import numpy as np
import os
import time
import re
import logging
from datetime import datetime, timezone
from utils.analysis_utils import get_verdict_and_risk, to_bucket

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
            "new world order",
            "cover-up",
            "they don't want you to know",
            "secret cabal",
            "hidden truth",
            "stolen election",
        ]
        text = (content or "").lower()
        return any(marker in text for marker in markers)

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
        ai_generated_score = compute_ai_likelihood(raw_model_ai_score, raw_features, style_metrics)

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

        claim_type_signal = classify_claim_type(content)
        claim_type = claim_type_signal.get("claim_type", "MIXED")
        sarcasm_signal = self.sarcasm_detector.detect(content)
        sarcasm_detected = bool(sarcasm_signal.get("sarcasm", False)) or claim_type == "SARCASTIC"
        verifiability = assess_claim_verifiability(content, claim_type=claim_type)
        conspiracy_flag = self._detect_conspiracy(content)
        bias_signal = detect_bias(content, model_bias_score=bias_score)
        bias_score = self._sanitize(max(float(bias_score), float(bias_signal.get("score", 0.0))))
        manipulation_signal = detect_manipulation(content, style_metrics)
        manipulation_score = self._sanitize(float(manipulation_signal.get("score", 0.0)))
        factual_claim = claim_type == "FACTUAL_CLAIM" or (
            bool(verifiability.get("claim_verifiable", False))
            and not bool(verifiability.get("opinion_detected", False))
            and claim_type not in {"PERSUASIVE_COPY", "UNVERIFIABLE_SPECULATION", "SARCASTIC"}
        )

        # 5. Fusion & self-check (STRICTLY INDEPENDENT)
        # When no external signals are available, we default to the model truth score.
        if verification_signals:
            final_truth_score = self._sanitize((truth_score * 0.6) + (avg_external_cred * 0.4))
        else:
            final_truth_score = self._sanitize(truth_score)
        
        # Heuristic: If we have zero relevant signals and a low news consistency, 
        # it's likely unsubstantiated/obviously false if it was presented as a fact.
        if not verification_signals or all(s.get("match_score", 0) < 0.3 for s in verification_signals):
             if news_consistency_score and news_consistency_score < 0.4:
                 final_truth_score *= 0.5  # Heavy penalty for "obviously false/unsubstantiated" content
                 logger.info("Content labeled as highly suspicious (zero news support)")

        # AI Likelihood MUST NOT lower the Truth Score. Removed legacy penalty logic.

        # Confidence: distance of final truth from the midpoint, clamped to [0.4, 0.95]
        uncertainty_penalty = 0.0
        if not verification_signals:
            uncertainty_penalty += 0.1
        
        confidence = self._sanitize(0.4 + 0.55 * abs(final_truth_score - 0.5) * 2 - uncertainty_penalty)

        # 5.5 Enriched analysis (Verdict, Risk, etc)
        enriched = get_verdict_and_risk(
            final_truth_score, ai_generated_score, bias_score, confidence,
            manipulation_score=manipulation_score,
            sarcasm_detected=sarcasm_detected,
            conspiracy_flag=conspiracy_flag,
            claim_verifiable=bool(verifiability.get("claim_verifiable", True)),
            opinion_detected=bool(verifiability.get("opinion_detected", False)),
            claim_type=claim_type,
            factual_claim=factual_claim,
        )

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
            "sarcasm_score": round(float(sarcasm_signal.get("score", 0.0)), 4),
            "opinion_score": 1.0 if bool(verifiability.get("opinion_detected", False)) else 0.0,
            "claim_verifiable": bool(verifiability.get("claim_verifiable", True)),
            "opinion_detected": bool(verifiability.get("opinion_detected", False)),
            "verifiability_reason": verifiability.get("reason", "unknown"),
            "claim_type": claim_type,
            "claim_type_signals": claim_type_signal.get("signals", []),
            "bias_indicators": bias_signal.get("indicators", []),
            "manipulation_indicators": manipulation_signal.get("indicators", []),
        }

        dimensions = {
            "truth_score": int(round(final_truth_score * 100)),
            "verifiability": int(100 if verifiability.get("claim_verifiable", True) else 20),
            "ai_likelihood": int(round(ai_generated_score * 100)),
            "bias_score": int(round(bias_score * 100)),
            "manipulation_score": int(round(manipulation_score * 100)),
            "sarcasm_score": int(round(float(sarcasm_signal.get("score", 0.0)) * 100)),
            "opinion_score": int(100 if bool(verifiability.get("opinion_detected", False)) else 0),
            "sarcasm": sarcasm_detected,
            "conspiracy_flag": conspiracy_flag,
        }

        dimension_buckets = {
            "truth_score": to_bucket(final_truth_score),
            "verifiability": "HIGH" if verifiability.get("claim_verifiable", True) else "LOW",
            "ai_likelihood": to_bucket(ai_generated_score),
            "bias_score": to_bucket(bias_score),
            "manipulation_score": to_bucket(manipulation_score),
            "sarcasm_score": to_bucket(float(sarcasm_signal.get("score", 0.0))),
            "opinion_score": to_bucket(1.0 if bool(verifiability.get("opinion_detected", False)) else 0.0),
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
                getattr(self, "_truth_reasoning", None)
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
            "primary_verdict": enriched.get("primary_verdict", "MIXED_ANALYSIS"),
            "confidence": int(round(confidence * 100)),
            "dimensions": dimensions,
            "expanded_analysis": {
                "truth_score": {
                    "explanation": "Cross-referenced against model inference and external verification signals.",
                    "evidence": self._truth_reasoning or "Weighted blend of model truth score and external credibility.",
                    "sources": truth_sources,
                },
                "ai_likelihood": {
                    "explanation": "Derived from perplexity, burstiness, sentence-length uniformity, lexical repetition, and stylometric consistency.",
                    "indicators": [
                        f"perplexity={round(raw_features.get('perplexity', 0), 2)}",
                        f"burstiness={style_metrics.get('burstiness', 0)}",
                        f"sentence_uniformity={style_metrics.get('sentence_length_uniformity', 0)}",
                        f"lexical_repetition={style_metrics.get('lexical_repetition', 0)}",
                        f"stylometric_consistency={style_metrics.get('stylometric_consistency', 0)}",
                    ],
                },
                "bias_score": {
                    "explanation": "Estimated from framing skew and language style cues.",
                    "indicators": [
                        f"bias_model={round(bias_score, 4)}",
                        f"manipulation_score={round(manipulation_score, 4)}",
                        *[f"bias_signal={value}" for value in bias_signal.get("indicators", [])],
                    ],
                },
                "manipulation_score": {
                    "explanation": "Flags emotional pressure, coercive urgency, fear framing, and sales pressure language.",
                    "indicators": [
                        *[f"manipulation_signal={value}" for value in manipulation_signal.get("indicators", [])],
                    ],
                },
                "opinion_score": {
                    "explanation": "Detects preference statements, value judgments, and rhetorical subjective framing.",
                    "indicators": [
                        f"claim_type={claim_type}",
                        *[f"claim_signal={value}" for value in claim_type_signal.get("signals", [])],
                        f"verifiability_reason={verifiability.get('reason', 'unknown')}",
                    ],
                },
                "verifiability": {
                    "explanation": "Gates factual verdicts by checking if the claim is sourceable and testable now.",
                    "indicators": [
                        f"claim_verifiable={bool(verifiability.get('claim_verifiable', True))}",
                        f"verifiability_reason={verifiability.get('reason', 'unknown')}",
                    ],
                },
            },
            "signals": signals,
            **({"news_consistency_score": round(news_consistency_score, 2)} if is_news_relevant else {}),
            **enriched,
            "metadata": {
                "model": "roberta-finetuned + gpt2-perplexity",
                "latency_ms": elapsed_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_metadata": {
                    "aggregation_rule": enriched.get("triggered_rule", "RULE_10_MIXED"),
                    "dimension_buckets": dimension_buckets,
                    "claim_type": claim_type,
                    "debug": enriched.get("debug", {}),
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
            "debug": enriched.get("debug", {}),
        }

    def _generate_explanation(self, truth, ai, bias, features, expert_reasoning=None, truth_reasoning=None):
        parts = []
        if truth_reasoning:
            parts.append(truth_reasoning)
        else:
            if truth > 0.85:
                parts.append("The content demonstrates exceptionally high factual alignment with verified source networks.")
            elif truth > 0.7:
                parts.append("The content aligns well with verified factual datasets.")
            elif truth < 0.2:
                parts.append("CRITICAL: Significant factual contradictions detected against multiple independent verifiers.")
            elif truth < 0.35:
                parts.append("Factual inconsistencies detected against cross-referenced news signals.")
            else:
                parts.append("Factual status is ambiguous or lacks sufficient external supporting evidence.")

        if expert_reasoning:
            parts.append(expert_reasoning)
        else:
            ppl = features.get("perplexity", 0)
            div = features.get("lexical_diversity", 0.5)
            
            if ai > 0.85:
                parts.append(f"Highly suspect stylometric markers (perplexity: {ppl:.1f}) strongly suggest machine-generation.")
            elif ai > 0.7:
                parts.append(f"Syntactic patterns indicate a high probability of AI assisted authorship.")
            elif ai < 0.3 and div > 0.4:
                parts.append("Natural lexical diversity and sentence variance align with authentic human authorship.")

        if bias > 0.7:
            parts.append("Language patterns suggest a strong slant or potential manipulative framing designed to provoke response.")

        return " ".join(parts)
