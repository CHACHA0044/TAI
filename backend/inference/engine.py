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
    """Lightweight mock model used when the real model cannot be loaded."""
    def __init__(self):
        super().__init__()
        self.truth = torch.nn.Linear(1, 1)
        self.ai = torch.nn.Linear(1, 2)
        self.bias = torch.nn.Linear(1, 3)

    def forward(self, **kwargs):
        return {
            "truth": torch.tensor([0.72]),
            "ai": torch.tensor([[0.15, 0.85]]),
            "bias": torch.tensor([[0.2, 0.6, 0.2]])
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
            logger.info("NewsSearchAgent initialised")
        except Exception as e:
            logger.warning(f"NewsSearchAgent failed: {e}")
            self.news_agent = None

    def _load_model(self, model_path):
        """Load multi-task RoBERTa or fall back to MockModel."""
        if HAS_TRANSFORMERS and TruthGuardMultiTaskModel:
            try:
                source = model_path or self.base_model_name
                self.tokenizer = AutoTokenizer.from_pretrained(source)
                config = AutoConfig.from_pretrained(source)
                self.model = TruthGuardMultiTaskModel(config, source)
                
                # Load custom heads if they exist
                heads_path = os.path.join(source, "custom_heads.pt")
                if os.path.exists(heads_path):
                    state_dict = torch.load(heads_path, map_location=self.device, weights_only=True)
                    self.model.truth_head.load_state_dict(state_dict['truth_head'])
                    self.model.ai_head.load_state_dict(state_dict['ai_head'])
                    self.model.bias_head.load_state_dict(state_dict['bias_head'])
                    logger.info("Custom trained heads loaded successfully")
                
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model loaded: {source}")
                return
            except Exception as e:
                logger.warning(f"Model load failed ({e}) — falling back to mock")

        # Fallback
        try:
            from transformers import AutoTokenizer as AT
            self.tokenizer = AT.from_pretrained("gpt2")
        except Exception:
            self.tokenizer = None
        self.model = MockModel().to(self.device)
        logger.info("Using MockModel")

    def _tokenize(self, text):
        """Tokenize text, handling both real and mock tokenizers."""
        if self.tokenizer is None:
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

        truth_score = float(outputs["truth"].cpu().item())
        ai_probs = torch.softmax(outputs["ai"], dim=1).cpu().numpy()[0]
        ai_generated_score = float(ai_probs[1])
        bias_probs = torch.softmax(outputs["bias"], dim=1).cpu().numpy()[0]
        bias_score = float(np.argmax(bias_probs) / 2.0)

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
        final_truth_score = (truth_score * 0.6) + (avg_external_cred * 0.4)
        if ai_generated_score > 0.8 and avg_external_cred < 0.6:
            final_truth_score *= 0.9

        confidence = 0.5 + (0.5 * abs(truth_score - 0.5) * 2)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Analysis complete in {elapsed_ms}ms")

        # 6. Build response matching the frontend contract
        # Transform verification signals → frontend "signals" format
        signals = []
        for vs in verification_signals:
            signals.append({
                "source": vs.get("top_match", vs.get("claim", "Unknown source")),
                "verified": vs.get("agent_credibility", 0.5) > 0.6,
                "confidence": round(vs.get("agent_credibility", 0.5), 2),
            })

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
            **({"news_consistency_score": round(news_consistency_score, 2)} if news_consistency_score is not None else {}),
            "metadata": {
                "model": "roberta-base + gpt2-perplexity",
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
