import io
import itertools
import logging
import os
import re
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

    def _analyze_classifier(self, image: Image.Image) -> Tuple[float, float, List[Tuple[str, float]]]:
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
            top_predictions = [(name, float(score)) for name, score in sorted_pairs[:5]]

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

            return min(max(ai_prob, 0.0), 1.0), min(max(confidence, 0.0), 1.0), top_predictions
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

    def _infer_detected_objects(self, top_predictions: List[Tuple[str, float]], ocr_text: str) -> List[str]:
        candidates: List[str] = []
        for label, confidence in top_predictions:
            if confidence < 0.28:
                continue
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

    def _extract_metadata_evidence(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        def _pick(*keys: str) -> Optional[str]:
            for key in keys:
                value = metadata.get(key)
                if value:
                    return str(value)
            return None

        model = _pick("EXIF_Model", "model")
        make = _pick("EXIF_Make", "make")
        timestamp = _pick("EXIF_DateTimeOriginal", "EXIF_DateTime", "date:create", "date")
        location = _pick("EXIF_GPSInfo", "GPSLatitude", "GPSLongitude")

        has_camera_indicators = any(
            metadata.get(key) is not None
            for key in [
                "EXIF_Model",
                "EXIF_Make",
                "EXIF_ISOSpeedRatings",
                "EXIF_ExposureTime",
                "EXIF_FNumber",
                "EXIF_FocalLength",
            ]
        )

        blob = str(metadata).lower()
        has_generator_markers = any(
            marker in blob
            for marker in ["midjourney", "stable diffusion", "openai", "dall", "generated", "firefly", "gemini", "sora"]
        )

        return {
            "device_model": f"{make} {model}".strip() if make and model else (model or make),
            "timestamp": timestamp,
            "location": location,
            "has_camera_indicators": has_camera_indicators,
            "has_generator_markers": has_generator_markers,
        }

    def _infer_content_type(
        self,
        top_predictions: List[Tuple[str, float]],
        ocr_text: str,
        source: str,
        neural_score: float,
        raw_signal_scores: Dict[str, float],
    ) -> Tuple[str, List[str], bool]:
        label_blob = " ".join(label.lower() for label, _ in top_predictions)
        text_blob = (ocr_text or "").lower()
        blob = f"{label_blob} {text_blob}"
        context_tags: List[str] = []

        human_present = any(
            marker in blob
            for marker in ["person", "face", "portrait", "man", "woman", "child", "selfie", "human"]
        )
        if human_present:
            context_tags.append("human-subject")

        if any(marker in blob for marker in ["sketch", "pencil", "charcoal", "line art", "hand-drawn", "doodle"]):
            return "HAND_DRAWN", context_tags + ["artwork"], human_present

        if any(marker in blob for marker in ["illustration", "painting", "watercolor", "anime", "cartoon", "comic"]):
            return "DIGITAL_ARTWORK", context_tags + ["illustration"], human_present

        if any(marker in blob for marker in ["render", "cgi", "3d", "blender", "unreal", "raytraced"]):
            return "CGI_RENDER", context_tags + ["render"], human_present

        if any(marker in blob for marker in ["screenshot", "ui", "interface", "dashboard", "meme", "tweet", "reddit"]):
            return "SCREENSHOT_UI_MEME", context_tags + ["screen-content"], human_present

        if source != "Unknown" or neural_score >= 0.78:
            return "AI_SYNTHETIC", context_tags + ["synthetic-indicators"], human_present

        if raw_signal_scores.get("ela", 0.0) > 0.72 and raw_signal_scores.get("edge_artifacts", 0.0) > 0.58:
            return "EDITED_COMPOSITE", context_tags + ["composite-risk"], human_present

        if text_blob and len(text_blob) > 30:
            context_tags.append("embedded-text")

        return "REAL_PHOTO", context_tags or ["natural-scene"], human_present

    def _best_effort_landmark_or_entity(
        self,
        top_predictions: List[Tuple[str, float]],
        ocr_text: str,
    ) -> Tuple[Optional[str], List[str]]:
        landmark_patterns = {
            "Eiffel Tower": [r"\beiffel\b"],
            "Statue of Liberty": [r"\bstatue of liberty\b"],
            "Taj Mahal": [r"\btaj mahal\b"],
            "Burj Khalifa": [r"\bburj khalifa\b"],
            "Golden Gate Bridge": [r"\bgolden gate\b"],
            "Colosseum": [r"\bcolosseum\b"],
            "Big Ben": [r"\bbig ben\b"],
        }

        best_name: Optional[str] = None
        best_confidence = 0.0
        fallback_entities: List[str] = []
        text_blob = (ocr_text or "").lower()

        for label, confidence in top_predictions:
            normalized = label.replace("_", " ").lower()
            for name, patterns in landmark_patterns.items():
                if any(re.search(pattern, normalized) for pattern in patterns):
                    if confidence >= 0.80 and confidence > best_confidence:
                        best_name = name
                        best_confidence = confidence
                    elif confidence >= 0.4:
                        fallback_entities.append("landmark structure")

            if confidence >= 0.80:
                fallback_entities.append(label.replace("_", " "))
            elif confidence >= 0.45:
                fallback_entities.append("identified object")

        if not best_name:
            for name, patterns in landmark_patterns.items():
                if any(re.search(pattern, text_blob) for pattern in patterns):
                    best_name = name
                    best_confidence = 0.80
                    break

        deduped_entities = list(dict.fromkeys(entity for entity in fallback_entities if entity))[:6]
        return best_name if best_confidence >= 0.80 else None, deduped_entities

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
            metadata_anomalies = 0.28
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

    def _contextualize_forensic_signals(
        self,
        raw_scores: Dict[str, float],
        explanations: Dict[str, Dict[str, Any]],
        *,
        content_type: str,
        human_present: bool,
        metadata_evidence: Dict[str, Any],
        neural_confidence: float,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], List[str], List[str]]:
        weighted = dict(raw_scores)
        technical_only: set[str] = set()

        context_scale = {
            "REAL_PHOTO": 1.0,
            "AI_SYNTHETIC": 1.0,
            "EDITED_COMPOSITE": 1.0,
            "DIGITAL_ARTWORK": 0.35,
            "HAND_DRAWN": 0.2,
            "CGI_RENDER": 0.35,
            "SCREENSHOT_UI_MEME": 0.25,
        }.get(content_type, 1.0)

        if context_scale < 1.0:
            for key in ["ela", "compression_anomalies", "lighting_consistency", "shadow_mismatch", "noise_pattern_mismatch"]:
                weighted[key] = weighted.get(key, 0.0) * context_scale

        if not human_present:
            weighted["face_hand_inconsistency"] = min(weighted.get("face_hand_inconsistency", 0.0) * 0.1, 0.08)
            technical_only.add("face_hand_inconsistency")

        confidence_weight = 0.7 + (max(0.0, min(1.0, neural_confidence)) * 0.3)
        for key in ["texture_consistency", "noise_pattern_mismatch", "object_realism"]:
            weighted[key] = max(0.0, min(1.0, weighted.get(key, 0.0) * confidence_weight))

        if metadata_evidence.get("has_camera_indicators"):
            weighted["metadata_anomalies"] = weighted.get("metadata_anomalies", 0.0) * 0.45
        if metadata_evidence.get("has_generator_markers"):
            weighted["metadata_anomalies"] = max(weighted.get("metadata_anomalies", 0.0), 0.88)

        if metadata_evidence.get("has_camera_indicators") and weighted.get("metadata_anomalies", 0.0) > 0.7:
            weighted["metadata_anomalies"] *= 0.6

        weighted_signals: Dict[str, Dict[str, Any]] = {}
        signal_hits: List[str] = []
        sorted_for_top = sorted(weighted.items(), key=lambda item: item[1], reverse=True)
        top_signals = [name for name, score in sorted_for_top if score >= 0.35][:5]

        for name, score in weighted.items():
            bucket = self._bucket(score)
            base = explanations.get(name, {})
            weighted_signals[name] = {
                "score": round(float(score), 4),
                "bucket": bucket,
                "explanation": base.get("explanation", "Forensic authenticity signal."),
                "technical_only": name in technical_only,
            }
            if bucket in {"MEDIUM", "HIGH"}:
                signal_hits.append(f"{name}:{bucket}")

        return weighted_signals, weighted, signal_hits, top_signals

    def _compose_verdict(
        self,
        content_type: str,
        source: str,
        neural_score: float,
        signal_scores: Dict[str, float],
        metadata_evidence: Dict[str, Any],
    ) -> Tuple[str, float, Dict[str, float], str, List[str]]:
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
        rejected: List[str] = []

        if source != "Unknown" or metadata_evidence.get("has_generator_markers"):
            ai_likelihood = max(ai_likelihood, 0.85)

        if content_type == "HAND_DRAWN":
            verdict = "HAND_DRAWN_SKETCH_ARTWORK"
            triggered_rule = "content_type_hand_drawn"
            rejected = ["Authentic Real Photograph", "AI-Generated Synthetic Image"]
        elif content_type in {"DIGITAL_ARTWORK", "CGI_RENDER", "SCREENSHOT_UI_MEME"} and ai_likelihood < 0.72:
            verdict = "DIGITAL_ARTWORK_ILLUSTRATION"
            triggered_rule = "non_photographic_visual_style"
            rejected = ["Authentic Real Photograph"]
        elif ai_likelihood >= 0.74 and edit_likelihood >= 0.62:
            verdict = "COMPOSITE_POTENTIAL_DEEPFAKE"
            triggered_rule = "ai_and_edit_signals_high"
            rejected = ["Likely Real Camera Photo"]
        elif ai_likelihood >= 0.74:
            verdict = "AI_GENERATED_SYNTHETIC_IMAGE"
            triggered_rule = "strong_ai_artifacts"
            rejected = ["Likely Real Camera Photo"]
        elif edit_likelihood >= 0.66:
            verdict = "EDITED_MANIPULATED_IMAGE"
            triggered_rule = "forensic_edit_anomalies"
            rejected = ["Authentic Real Photograph"]
        else:
            camera_bonus = 0.09 if metadata_evidence.get("has_camera_indicators") else 0.0
            real_score = ((1.0 - ai_likelihood) * 0.58) + ((1.0 - edit_likelihood) * 0.32) + camera_bonus
            if real_score >= 0.78:
                verdict = "AUTHENTIC_REAL_PHOTOGRAPH"
                triggered_rule = "camera_consistent_real_photo"
                rejected = ["AI-Generated Synthetic Image", "Edited / Manipulated Image"]
            elif real_score >= 0.62:
                verdict = "LIKELY_REAL_CAMERA_PHOTO"
                triggered_rule = "real_photo_with_limited_noise"
                rejected = ["AI-Generated Synthetic Image"]
            elif abs(ai_likelihood - edit_likelihood) < 0.08:
                verdict = "UNCERTAIN"
                triggered_rule = "conflicting_or_weak_signals"
                rejected = ["Authentic Real Photograph", "AI-Generated Synthetic Image"]
            elif ai_likelihood > edit_likelihood:
                verdict = "AI_GENERATED_SYNTHETIC_IMAGE"
                triggered_rule = "ai_signals_dominate"
                rejected = ["Likely Real Camera Photo"]
            else:
                verdict = "EDITED_MANIPULATED_IMAGE"
                triggered_rule = "edit_signals_dominate"
                rejected = ["Authentic Real Photograph"]

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

        return verdict, confidence, {"ai_likelihood": ai_likelihood, "edit_likelihood": edit_likelihood}, triggered_rule, rejected

    def _compose_why(
        self,
        primary_verdict: str,
        content_type: str,
        scene_description: str,
        authenticity_signals: Dict[str, Dict[str, Any]],
        source: str,
        top_signals: List[str],
        rejected_verdicts: List[str],
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
        if top_signals:
            reasons.append(f"Top weighted signal set: {', '.join(top_signals)}.")
        reasons.append(f"Scene interpretation: {scene_description}")
        if rejected_verdicts:
            reasons.append(f"Competing verdicts rejected: {', '.join(rejected_verdicts)}.")

        if primary_verdict in {"AUTHENTIC_REAL_PHOTOGRAPH", "LIKELY_REAL_CAMERA_PHOTO"}:
            why = "Visual evidence appears consistent with a natural photograph."
            uncertain = None
        elif primary_verdict == "AI_GENERATED_SYNTHETIC_IMAGE":
            why = "Synthetic-generation cues dominate over camera-authenticity cues."
            uncertain = None
        elif primary_verdict in {"EDITED_MANIPULATED_IMAGE", "COMPOSITE_POTENTIAL_DEEPFAKE"}:
            why = "Localized manipulation/compression signals suggest post-processing or compositing."
            uncertain = None
        elif primary_verdict in {"DIGITAL_ARTWORK_ILLUSTRATION", "HAND_DRAWN_SKETCH_ARTWORK"}:
            why = f"Content classification indicates a non-photographic medium ({content_type.lower().replace('_', ' ')}), so photo-forensic metrics were de-emphasized."
            uncertain = None
        else:
            why = "Signals are inconclusive and do not strongly support one authenticity class."
            uncertain = "Key cues conflict or are weak after context-aware weighting. Manual forensic review with source provenance is recommended."

        return why, uncertain, reasons

    def analyze(self, image_bytes: bytes, filename: Optional[str] = None):
        start_time = time.time()
        logger.info(f"Running image analysis — filename: {filename}")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            img_rgb = image.convert("RGB")

            neural_score, neural_confidence, top_predictions = self._analyze_classifier(img_rgb)
            top_labels = [label for label, _ in top_predictions]
            metadata = self.extract_metadata(image)
            source = self._detect_source(filename, metadata)
            ocr_text = self._extract_ocr_text(img_rgb)
            metadata_evidence = self._extract_metadata_evidence(metadata)

            authenticity_signals, raw_signal_scores, signal_hits = self._compute_forensic_signals(
                img_rgb,
                metadata,
                neural_score,
            )

            content_type, context_tags, human_present = self._infer_content_type(
                top_predictions=top_predictions,
                ocr_text=ocr_text,
                source=source,
                neural_score=neural_score,
                raw_signal_scores=raw_signal_scores,
            )
            named_landmark, fallback_entities = self._best_effort_landmark_or_entity(top_predictions, ocr_text)
            if named_landmark:
                context_tags.append("recognized-landmark")

            authenticity_signals, weighted_signal_scores, signal_hits, top_signals = self._contextualize_forensic_signals(
                raw_scores=raw_signal_scores,
                explanations=authenticity_signals,
                content_type=content_type,
                human_present=human_present,
                metadata_evidence=metadata_evidence,
                neural_confidence=neural_confidence,
            )

            verdict, verdict_confidence, aggregate_scores, triggered_rule, rejected_verdicts = self._compose_verdict(
                content_type,
                source,
                neural_score,
                weighted_signal_scores,
                metadata_evidence,
            )

            style = self._infer_style(source, aggregate_scores["ai_likelihood"], 1.0 - weighted_signal_scores["texture_consistency"], top_labels)
            detected_objects = self._infer_detected_objects(top_predictions, ocr_text)
            if named_landmark:
                detected_objects.insert(0, named_landmark)
            detected_objects = list(dict.fromkeys(detected_objects + fallback_entities))[:8]
            scene_description = self._generate_scene_description(img_rgb, top_labels, detected_objects, style)
            why, uncertain_note, extra_reasons = self._compose_why(
                verdict,
                content_type,
                scene_description,
                authenticity_signals,
                source,
                top_signals,
                rejected_verdicts,
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
                "content_type": content_type,
                "context_tags": list(dict.fromkeys(context_tags)),
                "recognized_landmark": named_landmark,
                "top_signals": top_signals,
                "rejected_verdicts": rejected_verdicts,
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
                    "device_model": metadata_evidence.get("device_model"),
                    "capture_timestamp": metadata_evidence.get("timestamp"),
                    "capture_location": metadata_evidence.get("location"),
                },
                "debug": {
                    **(enriched.get("debug") or {}),
                    "raw_intermediate_scores": {
                        "neural_score": round(neural_score, 4),
                        "neural_confidence": round(neural_confidence, 4),
                        "top_labels": top_labels,
                        "aggregate_ai_likelihood": round(aggregate_scores["ai_likelihood"], 4),
                        "aggregate_edit_likelihood": round(aggregate_scores["edit_likelihood"], 4),
                        "forensic_signal_scores": {k: round(float(v), 4) for k, v in weighted_signal_scores.items()},
                        "forensic_signal_scores_raw": {k: round(float(v), 4) for k, v in raw_signal_scores.items()},
                        "content_type": content_type,
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
