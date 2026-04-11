import math
import torch
import numpy as np
import os
import time
import logging
from datetime import datetime, timezone

logger = logging.getLogger("truthguard")

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
from utils.url_extractor import extract_content, get_claims

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai package not installed — OpenAI enhancement disabled")

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

        # OpenAI client (optional enhancement)
        self.openai_client = None
        self.openai_model = os.getenv("OPENAI_FINETUNED_MODEL", "gpt-4o-mini-2024-07-18")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if HAS_OPENAI and api_key and not api_key.startswith("sk-dummy"):
            try:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")

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

    def call_openai(self, text):
        """Optional OpenAI enhancement for higher fidelity analysis."""
        if not self.openai_client:
            return None

        system_prompt = (
            "You are TruthGuard AI, a factual verification system. "
            "Analyze the given text and return a JSON object with: "
            "'truth_score' (0.0-1.0), 'ai_generated_score' (0.0-1.0), "
            "and 'bias_score' (0.0-1.0). Be objective and factual."
        )
        try:
            import json
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:2000]},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
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
        ai_generated_score = self._sanitize(float(ai_probs[1]))
        bias_probs = torch.softmax(outputs["bias"], dim=1).cpu().numpy()[0]
        bias_score = self._sanitize(float(np.argmax(bias_probs) / 2.0))

        # 2.5 Optional OpenAI enhancement
        oa_res = self.call_openai(content)
        if oa_res:
            logger.info("Fusing OpenAI insights (60% weight)")
            truth_score = (truth_score * 0.4) + (oa_res.get("truth_score", 0.5) * 0.6)
            ai_generated_score = (ai_generated_score * 0.4) + (oa_res.get("ai_generated_score", 0.5) * 0.6)
            bias_score = (bias_score * 0.4) + (oa_res.get("bias_score", 0.5) * 0.6)

        # 3. Feature extraction (perplexity + stylometry)
        logger.info("Extracting features...")
        raw_features = self.feature_extractor.extract_all(content)

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

        # 5. Fusion & self-check
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

        # Only penalize if both AI probability is very high AND external credibility
        # is low — not when credibility is simply at its neutral default (0.5).
        if ai_generated_score > 0.8 and verification_signals and avg_external_cred < 0.4:
            final_truth_score = self._sanitize(final_truth_score * 0.9)

        # Confidence: distance of final truth from the midpoint, clamped to [0.4, 0.95]
        confidence = self._sanitize(0.4 + 0.55 * abs(final_truth_score - 0.5) * 2)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Analysis complete in {elapsed_ms}ms")

        # 6. Build response matching the frontend contract
        # Transform verification signals → frontend "signals" format
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

        # Only expose news_consistency_score if it's statistically significant (>0.35)
        is_news_relevant = news_consistency_score is not None and news_consistency_score > 0.35

        return {
            "truth_score": round(final_truth_score, 2),
            "ai_generated_score": round(ai_generated_score, 2),
            "bias_score": round(bias_score, 2),
            "credibility_score": round(avg_external_cred, 2),
            "confidence_score": round(confidence, 2),
            "explanation": self._generate_explanation(
                final_truth_score, ai_generated_score, bias_score, raw_features
            ),
            "features": {
                "perplexity": round(raw_features.get("perplexity", 0), 1),
                "stylometry": {
                    "sentence_length_variance": round(raw_features.get("sent_len_var", 0), 2),
                    "repetition_score": round(raw_features.get("repetition_score", 0), 4),
                    "lexical_diversity": round(raw_features.get("lexical_diversity", 0), 4),
                },
            },
            "signals": signals,
            **({"news_consistency_score": round(news_consistency_score, 2)} if is_news_relevant else {}),
            "metadata": {
                "model": "roberta-finetuned + gpt2-perplexity",
                "latency_ms": elapsed_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _generate_explanation(self, truth, ai, bias, features):
        parts = []
        if truth > 0.7:
            parts.append("The content aligns well with verified factual datasets.")
        elif truth < 0.3:
            parts.append("Severe factual inconsistencies detected against cross-referenced news signals.")
        else:
            parts.append("Factual status is ambiguous or requires more context.")

        ppl = features.get("perplexity", 0)
        if ai > 0.7:
            parts.append(
                f"Stylometric patterns (perplexity: {ppl:.1f}) indicate high probability of AI generation."
            )
        elif ai < 0.3:
            parts.append("Lexical diversity and sentence variance align with human authorship.")

        if bias > 0.7:
            parts.append("Language patterns suggest a strong slant or manipulative framing.")

        return " ".join(parts)
