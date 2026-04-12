import io
import itertools
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat

from utils.analysis_utils import get_verdict_and_risk

logger = logging.getLogger("truthguard")


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    F = None
    logger.warning("torch not installed — neural image model disabled; using forensic heuristics only")

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        BlipForConditionalGeneration,
        BlipProcessor,
    )

    HAS_TRANSFORMERS = True and HAS_TORCH
except ImportError:
    HAS_TRANSFORMERS = False
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    BlipProcessor = None
    BlipForConditionalGeneration = None
    logger.warning("transformers not installed — image detection will run in mock mode")

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed — OCR will be skipped")


class ImageEngine:
    def __init__(
        self,
        model_name: str = "prithivMLmods/Deep-Fake-Detector-v2-Model",
        caption_model_name: str = "Salesforce/blip-image-captioning-base",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
        self.model_name = model_name
        self.caption_model_name = caption_model_name
        self.processor = None
        self.model = None
        self.caption_processor = None
        self.caption_model = None

        self.using_transformers = HAS_TRANSFORMERS
        self.caption_enabled = HAS_TRANSFORMERS and os.getenv("IMAGE_CAPTION_DISABLE", "0") != "1"

        if self.using_transformers:
            try:
                logger.info(f"Loading image model: {model_name} (Half-Precision)...")
                # Use local_files_only=True if we pre-cached them, but let's keep it flexible
                self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.device)
                self.model.eval()
                logger.info("Image model loaded successfully")
            except Exception as e:
                logger.error(f"Image model fallback: {e}. Heuristics only mode active.")
                self.using_transformers = False

    def _bucket(self, score: float) -> str:
        if score >= 0.67:
            return "HIGH"
        if score >= 0.34:
            return "MEDIUM"
        return "LOW"

    def perform_ela(self, image: Image.Image, quality: int = 90) -> float:
        try:
            resaved_io = io.BytesIO()
            image.save(resaved_io, "JPEG", quality=quality)
            resaved_io.seek(0)
            resaved_image = Image.open(resaved_io)

            diff = ImageChops.difference(image, resaved_image)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff
            diff = ImageEnhance.Brightness(diff).enhance(scale)

            pixels = list(diff.getdata())
            if not pixels:
                return 0.0
            pixels_sum = 0.0
            for p in pixels:
                if isinstance(p, int):
                    pixels_sum += p
                else:
                    pixels_sum += sum(p) / 3.0

            avg_brightness = pixels_sum / max(1, len(pixels))
            suspicion = min(max(avg_brightness / 50.0, 0.0), 1.0)
            return suspicion
        except Exception as e:
            logger.warning(f"ELA forensics failed: {e}")
            return 0.0

    def extract_metadata(self, image: Image.Image) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        try:
            for key, value in image.info.items():
                if isinstance(value, (str, int, float)):
                    meta[key] = value

            exif = image._getexif() if hasattr(image, "_getexif") else None
            if exif:
                from PIL.ExifTags import TAGS

                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, (str, int, float)):
                        meta[f"EXIF_{tag}"] = value
                    elif isinstance(value, bytes):
                        try:
                            meta[f"EXIF_{tag}"] = value.decode(errors="ignore")[:100]
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Metadata extraction layer failed: {e}")
        return meta

    def _detect_source(self, filename: Optional[str], metadata: Dict[str, Any]) -> str:
        source = "Unknown"
        lookup_str = (str(filename or "") + " " + str(metadata).lower()).lower()

        if any(kw in lookup_str for kw in ["gemini", "google"]):
            source = "Google Gemini"
        elif any(kw in lookup_str for kw in ["gpt", "dall-e", "openai", "sora"]):
            source = "OpenAI"
        elif any(kw in lookup_str for kw in ["midjourney", "mj"]):
            source = "Midjourney"
        elif any(kw in lookup_str for kw in ["stable diffusion", "sdxl", "stability"]):
            source = "Stable Diffusion"
        elif "adobe firefly" in lookup_str:
            source = "Adobe Firefly"
        elif "bing" in lookup_str:
            source = "Bing Image Creator"

        return source

    def _analyze_classifier(self, image: Image.Image) -> Tuple[float, float, List[str]]:
        if not (self.using_transformers and self.model and self.processor):
            return 0.5, 0.4, []

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                inputs = {
                    k: v.to(self.device).to(torch.float16) if v.is_floating_point() else v.to(self.device)
                    for k, v in inputs.items()
                }
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

            labels = self.model.config.id2label
            prob_dict = {labels[i]: float(probs[0][i].item()) for i in range(len(labels))}
            sorted_pairs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            top_labels = [name for name, _ in sorted_pairs[:5]]

            deepfake_keys = [k for k in prob_dict if "deep" in k.lower() or "fake" in k.lower() or "ai" in k.lower()]
            real_keys = [k for k in prob_dict if "real" in k.lower() or "auth" in k.lower() or "human" in k.lower()]

            ai_prob = max([prob_dict.get(k, 0.0) for k in deepfake_keys], default=0.0)
            real_prob = max([prob_dict.get(k, 0.0) for k in real_keys], default=0.0)
            if ai_prob == 0.0 and real_prob == 0.0:
                ai_prob = sorted_pairs[0][1] if sorted_pairs else 0.5

            confidence = float(torch.max(probs).item())
            if round(ai_prob, 4) == 0.5 and round(confidence, 4) == 0.5:
                confidence = 0.4

            if real_prob > ai_prob:
                ai_prob = max(0.0, 1.0 - real_prob)

            return min(max(ai_prob, 0.0), 1.0), min(max(confidence, 0.0), 1.0), top_labels
        except Exception as exc:
            logger.warning(f"Classifier inference failed: {exc}")
            return 0.5, 0.4, []

    def _ensure_caption_model(self) -> bool:
        if not self.caption_enabled:
            return False
        if self.caption_processor and self.caption_model:
            return True
        if not HAS_TRANSFORMERS:
            return False

        try:
            logger.info(f"Loading caption model: {self.caption_model_name}")
            self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
            device_type = self.device.type if HAS_TORCH and hasattr(self.device, "type") else "cpu"
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                self.caption_model_name,
                torch_dtype=torch.float16 if device_type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.caption_model.eval()
            logger.info("Caption model loaded")
            return True
        except Exception as exc:
            logger.warning(f"Caption model unavailable, using heuristic caption fallback: {exc}")
            self.caption_enabled = False
            return False

    def _generate_scene_description(
        self,
        image: Image.Image,
        top_labels: List[str],
        detected_objects: List[str],
        style: str,
    ) -> str:
        if self._ensure_caption_model():
            try:
                inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.caption_model.generate(**inputs, max_new_tokens=28)
                caption = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
                if caption:
                    return caption[0].upper() + caption[1:]
            except Exception as exc:
                logger.warning(f"Caption generation failed, using heuristic caption: {exc}")

        if detected_objects:
            joined = ", ".join(detected_objects[:4])
            return f"A {style} image featuring {joined}."
        if top_labels:
            return f"A {style} image likely depicting {top_labels[0].replace('_', ' ')}."
        return f"A {style} image with limited recognizable objects."

    def _infer_style(self, source: str, ai_score: float, texture_consistency: float, top_labels: List[str]) -> str:
        joined_labels = " ".join(top_labels).lower()
        if source != "Unknown" or ai_score >= 0.8:
            return "synthetic"
        if any(k in joined_labels for k in ["cartoon", "anime", "illustration", "drawing"]):
            return "illustration"
        if any(k in joined_labels for k in ["3d", "render", "cgi"]):
            return "render"
        if texture_consistency < 0.28 and ai_score > 0.55:
            return "cinematic"
        return "photo"

    def _infer_detected_objects(self, top_labels: List[str], ocr_text: str) -> List[str]:
        candidates: List[str] = []
        for label in top_labels:
            clean = label.replace("_", " ").replace("-", " ").strip()
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            candidates.extend(parts if parts else [clean])

        if ocr_text:
            text_hint = ocr_text.strip().split("\n", 1)[0][:48]
            if text_hint:
                candidates.append(f"text: {text_hint}")

        seen = set()
        objects: List[str] = []
        for item in candidates:
            key = item.lower()
            if key and key not in seen:
                seen.add(key)
                objects.append(item)
        return objects[:8]

    def _compute_forensic_signals(
        self,
        image_rgb: Image.Image,
        metadata: Dict[str, Any],
        neural_score: float,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], List[str]]:
        ela_score = self.perform_ela(image_rgb)

        lap = image_rgb.filter(ImageFilter.FIND_EDGES)
        edge_density = _mean([sum(pixel) / 765.0 for pixel in itertools.islice(lap.getdata(), 50000)])
        edge_artifacts = float(max(0.0, min(1.0, abs(edge_density - 0.18) * 3.3)))

        smooth = image_rgb.filter(ImageFilter.GaussianBlur(radius=2.0))
        orig_pixels = list(image_rgb.getdata())[:50000]
        smooth_pixels = list(smooth.getdata())[:50000]
        residual = [abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) for a, b in zip(orig_pixels, smooth_pixels)]
        noise_ratio = _mean(residual) / 765.0
        noise_pattern_mismatch = float(max(0.0, min(1.0, abs(noise_ratio - 0.08) * 5.0)))

        gray_img = image_rgb.convert("L")
        stats = ImageStat.Stat(gray_img)
        texture_var = (stats.var[0] / (255.0 * 255.0)) if stats.var else 0.0
        texture_consistency = float(max(0.0, min(1.0, 1.0 - min(1.0, texture_var * 18.0))))

        w, h = gray_img.size
        quarter_h = max(1, h // 4)
        quarter_w = max(1, w // 4)
        top_light = ImageStat.Stat(gray_img.crop((0, 0, w, quarter_h))).mean[0]
        bottom_light = ImageStat.Stat(gray_img.crop((0, h - quarter_h, w, h))).mean[0]
        left_light = ImageStat.Stat(gray_img.crop((0, 0, quarter_w, h))).mean[0]
        right_light = ImageStat.Stat(gray_img.crop((w - quarter_w, 0, w, h))).mean[0]
        vertical_delta = abs(top_light - bottom_light) / 255.0
        horizontal_delta = abs(left_light - right_light) / 255.0
        lighting_consistency = float(max(0.0, min(1.0, 1.0 - (vertical_delta + horizontal_delta))))

        half_w = max(1, w // 2)
        half_h = max(1, h // 2)
        quadrants = [
            gray_img.crop((0, 0, half_w, half_h)),
            gray_img.crop((half_w, 0, w, half_h)),
            gray_img.crop((0, half_h, half_w, h)),
            gray_img.crop((half_w, half_h, w, h)),
        ]
        quad_means = [ImageStat.Stat(region).mean[0] for region in quadrants]
        quad_avg = _mean(quad_means)
        quad_var = _mean([(value - quad_avg) ** 2 for value in quad_means]) if quad_means else 0.0
        shadow_mismatch = float(max(0.0, min(1.0, (quad_var ** 0.5) / 90.0)))

        jpg_buf = io.BytesIO()
        image_rgb.save(jpg_buf, "JPEG", quality=85)
        jpeg_size = max(1, jpg_buf.tell())
        png_buf = io.BytesIO()
        image_rgb.save(png_buf, "PNG")
        png_size = max(1, png_buf.tell())
        compression_ratio = jpeg_size / png_size
        compression_anomalies = float(max(0.0, min(1.0, abs(compression_ratio - 0.55) * 2.2)))

        meta_blob = str(metadata).lower()
        has_camera = any(k in meta_blob for k in ["make", "model", "lens", "iso", "exposure"])
        has_generator = any(k in meta_blob for k in ["midjourney", "stable diffusion", "openai", "dall", "generated"])
        metadata_anomalies = 0.15
        if not metadata:
            metadata_anomalies = 0.5
        if has_generator:
            metadata_anomalies = 0.92
        elif has_camera:
            metadata_anomalies = 0.2

        saturation = [max(pixel) - min(pixel) for pixel in orig_pixels] if orig_pixels else [0.0]
        realism_proxy = _mean([value / 255.0 for value in saturation])
        object_realism = float(max(0.0, min(1.0, abs(realism_proxy - 0.35) * 2.4)))

        face_hand_inconsistency = float(max(0.0, min(1.0, neural_score * 0.75 + edge_artifacts * 0.25)))

        raw_scores: Dict[str, float] = {
            "ela": ela_score,
            "texture_consistency": 1.0 - texture_consistency,
            "edge_artifacts": edge_artifacts,
            "lighting_consistency": 1.0 - lighting_consistency,
            "shadow_mismatch": shadow_mismatch,
            "noise_pattern_mismatch": noise_pattern_mismatch,
            "compression_anomalies": compression_anomalies,
            "face_hand_inconsistency": face_hand_inconsistency,
            "object_realism": object_realism,
            "metadata_anomalies": metadata_anomalies,
        }

        explanations = {
            "ela": "Error-level differences suggest local re-compression or edits.",
            "texture_consistency": "Texture regularity appears overly uniform for a natural camera capture.",
            "edge_artifacts": "Edge sharpness/halos indicate possible synthetic upscaling or compositing seams.",
            "lighting_consistency": "Global illumination looks directionally inconsistent across regions.",
            "shadow_mismatch": "Shadow or luminance balance between quadrants is uneven.",
            "noise_pattern_mismatch": "Sensor-noise residual does not match a typical single-camera pattern.",
            "compression_anomalies": "Compression profile deviates from natural camera/JPEG expectations.",
            "face_hand_inconsistency": "Facial/hand contours show artifact-like transitions.",
            "object_realism": "Color realism and local details look stylized or physically implausible.",
            "metadata_anomalies": "Metadata appears missing, generator-like, or inconsistent with camera-origin EXIF.",
        }

        authenticity_signals: Dict[str, Dict[str, Any]] = {}
        signal_hits: List[str] = []
        for name, score in raw_scores.items():
            bucket = self._bucket(score)
            authenticity_signals[name] = {
                "score": round(float(score), 4),
                "bucket": bucket,
                "explanation": explanations[name],
            }
            if bucket in {"MEDIUM", "HIGH"}:
                signal_hits.append(f"{name}:{bucket}")

        return authenticity_signals, raw_scores, signal_hits

    def _compose_verdict(
        self,
        source: str,
        neural_score: float,
        signal_scores: Dict[str, float],
    ) -> Tuple[str, float, Dict[str, float], str]:
        ai_markers = [
            neural_score,
            signal_scores.get("texture_consistency", 0.0),
            signal_scores.get("noise_pattern_mismatch", 0.0),
            signal_scores.get("object_realism", 0.0),
            signal_scores.get("metadata_anomalies", 0.0),
        ]
        edit_markers = [
            signal_scores.get("ela", 0.0),
            signal_scores.get("compression_anomalies", 0.0),
            signal_scores.get("edge_artifacts", 0.0),
            signal_scores.get("shadow_mismatch", 0.0),
            signal_scores.get("lighting_consistency", 0.0),
        ]

        ai_likelihood = _mean(ai_markers)
        edit_likelihood = _mean(edit_markers)

        if source != "Unknown":
            ai_likelihood = max(ai_likelihood, 0.92)

        if ai_likelihood >= 0.74 and edit_likelihood >= 0.56:
            verdict = "MIXED"
            triggered_rule = "ai_and_edit_signals_high"
        elif ai_likelihood >= 0.74:
            verdict = "AI_GENERATED"
            triggered_rule = "strong_ai_artifacts"
        elif edit_likelihood >= 0.64:
            verdict = "EDITED"
            triggered_rule = "forensic_edit_anomalies"
        elif ai_likelihood <= 0.36 and edit_likelihood <= 0.38:
            verdict = "REAL"
            triggered_rule = "natural_photo_signature"
        else:
            verdict = "UNCERTAIN"
            triggered_rule = "conflicting_or_weak_signals"

        uncertainty = 1.0 - abs(ai_likelihood - edit_likelihood)
        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    (abs(ai_likelihood - 0.5) + abs(edit_likelihood - 0.5))
                    * 0.7
                    + (1.0 - uncertainty) * 0.3,
                ),
            )
        )

        return verdict, confidence, {"ai_likelihood": ai_likelihood, "edit_likelihood": edit_likelihood}, triggered_rule

    def _compose_why(
        self,
        primary_verdict: str,
        style: str,
        scene_description: str,
        authenticity_signals: Dict[str, Dict[str, Any]],
        source: str,
    ) -> Tuple[str, Optional[str], List[str]]:
        strongest = sorted(
            authenticity_signals.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )
        top = [name for name, data in strongest if data["bucket"] in {"HIGH", "MEDIUM"}][:4]

        reasons: List[str] = []
        if source != "Unknown":
            reasons.append(f"Metadata/source markers indicate {source}.")
        if top:
            reasons.append(f"Most influential forensic signals: {', '.join(top)}.")
        reasons.append(f"Scene interpretation: {scene_description}")

        if primary_verdict == "REAL":
            why = "Visual evidence appears consistent with a natural photograph."
            uncertain = None
        elif primary_verdict == "AI_GENERATED":
            why = "Synthetic-generation cues dominate over camera-authenticity cues."
            uncertain = None
        elif primary_verdict == "EDITED":
            why = "Localized manipulation/compression signals suggest post-processing or compositing."
            uncertain = None
        elif primary_verdict == "MIXED":
            why = "The image shows both synthetic-generation and edit-like characteristics."
            uncertain = "Multiple strong but competing clues are present. Request the original source file or sequence for manual review."
        else:
            why = "Signals are inconclusive and do not strongly support one authenticity class."
            uncertain = "Key cues conflict or are weak. Manual forensic review with source provenance is recommended."

        return why, uncertain, reasons

    def analyze(self, image_bytes: bytes, filename: Optional[str] = None):
        start_time = time.time()
        logger.info(f"Running image analysis — filename: {filename}")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            img_rgb = image.convert("RGB")

            neural_score, neural_confidence, top_labels = self._analyze_classifier(img_rgb)
            metadata = self.extract_metadata(image)
            source = self._detect_source(filename, metadata)
            ocr_text = self._extract_ocr_text(img_rgb)

            authenticity_signals, raw_signal_scores, signal_hits = self._compute_forensic_signals(
                img_rgb,
                metadata,
                neural_score,
            )

            verdict, verdict_confidence, aggregate_scores, triggered_rule = self._compose_verdict(
                source,
                neural_score,
                raw_signal_scores,
            )

            style = self._infer_style(source, aggregate_scores["ai_likelihood"], 1.0 - raw_signal_scores["texture_consistency"], top_labels)
            detected_objects = self._infer_detected_objects(top_labels, ocr_text)
            scene_description = self._generate_scene_description(img_rgb, top_labels, detected_objects, style)
            why, uncertain_note, extra_reasons = self._compose_why(
                verdict,
                style,
                scene_description,
                authenticity_signals,
                source,
            )

            enriched = get_verdict_and_risk(
                truth_score=1.0 - aggregate_scores["ai_likelihood"],
                ai_score=aggregate_scores["ai_likelihood"],
                bias_score=0.0,
                confidence=verdict_confidence,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            confidence_pct = int(round(max(0.0, min(1.0, max(verdict_confidence, neural_confidence))) * 100))
            credibility_score = max(0.0, min(1.0, 1.0 - aggregate_scores["edit_likelihood"]))
            ai_generated_score = max(0.0, min(1.0, aggregate_scores["ai_likelihood"]))

            key_factors = list(dict.fromkeys(enriched.get("key_factors", []) + extra_reasons))

            result = {
                "category": verdict,
                "primary_verdict": verdict,
                "triggered_rule": triggered_rule,
                "source": source,
                "scene_description": scene_description,
                "detected_objects": detected_objects,
                "style": style,
                "authenticity_signals": authenticity_signals,
                "truth_score": round(1.0 - ai_generated_score, 4),
                "ai_generated_score": round(ai_generated_score, 4),
                "bias_score": 0.0,
                "credibility_score": round(credibility_score, 4),
                "confidence_score": round(max(0.0, min(1.0, confidence_pct / 100.0)), 4),
                "confidence": confidence_pct,
                "why": why,
                "if_uncertain": uncertain_note,
                "explanation": why,
                "features": {
                    "perplexity": round(raw_signal_scores.get("ela", 0.0), 4),
                    "stylometry": {
                        "sentence_length_variance": round(raw_signal_scores.get("texture_consistency", 0.0), 4),
                        "repetition_score": len(metadata),
                        "lexical_diversity": round(max(0.0, min(1.0, neural_confidence)), 4),
                    },
                },
                "signals": [
                    {
                        "source": "Neural artifact classifier",
                        "verified": ai_generated_score < 0.5,
                        "confidence": round(neural_confidence, 4),
                    },
                    {
                        "source": "Error level analysis",
                        "verified": raw_signal_scores.get("ela", 0.0) < 0.45,
                        "confidence": round(1.0 - raw_signal_scores.get("ela", 0.0), 4),
                    },
                    {
                        "source": "Compression and noise consistency",
                        "verified": raw_signal_scores.get("compression_anomalies", 0.0) < 0.45
                        and raw_signal_scores.get("noise_pattern_mismatch", 0.0) < 0.45,
                        "confidence": round(
                            1.0
                            - ((raw_signal_scores.get("compression_anomalies", 0.0) + raw_signal_scores.get("noise_pattern_mismatch", 0.0)) / 2.0),
                            4,
                        ),
                    },
                ],
                "metadata": {
                    "model": self.model_name if self.using_transformers else "heuristic-hybrid",
                    "latency_ms": elapsed_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_metadata": metadata,
                },
                "debug": {
                    **(enriched.get("debug") or {}),
                    "raw_intermediate_scores": {
                        "neural_score": round(neural_score, 4),
                        "neural_confidence": round(neural_confidence, 4),
                        "top_labels": top_labels,
                        "aggregate_ai_likelihood": round(aggregate_scores["ai_likelihood"], 4),
                        "aggregate_edit_likelihood": round(aggregate_scores["edit_likelihood"], 4),
                        "forensic_signal_scores": {k: round(float(v), 4) for k, v in raw_signal_scores.items()},
                    },
                    "signal_hits": signal_hits,
                    "caption_model": self.caption_model_name if self.caption_enabled else "heuristic",
                    "image_style": style,
                    "source_detected": source,
                },
                "verdict": enriched.get("verdict", "Inconclusive"),
                "risk_level": enriched.get("risk_level", "Medium"),
                "recommendation": enriched.get("recommendation", "Verify against trusted sources."),
                "key_factors": key_factors,
                "dimensions": {
                    "truth_score": int(round((1.0 - ai_generated_score) * 100)),
                    "verifiability": int(round((1.0 - raw_signal_scores.get("metadata_anomalies", 0.0)) * 100)),
                    "ai_likelihood": int(round(ai_generated_score * 100)),
                    "bias_score": 0,
                    "manipulation_score": 0,
                    "sarcasm_score": 0,
                    "opinion_score": 0,
                    "sarcasm": False,
                    "conspiracy_flag": False,
                },
            }

            if ocr_text:
                result["ocr_text"] = ocr_text

            return result
        except Exception:
            logger.exception("CRITICAL ERROR in image engine pipeline")
            raise

    def _extract_ocr_text(self, image: Image.Image) -> str:
        if not HAS_TESSERACT:
            return ""
        try:
            text = pytesseract.image_to_string(image, timeout=10).strip()
            return text
        except Exception as exc:
            logger.warning(f"OCR extraction failed: {exc}")
            return ""
