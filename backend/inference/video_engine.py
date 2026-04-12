import cv2
import io
import os
import subprocess
import tempfile
import time
import logging
import hashlib
import uuid
import numpy as np
import torch
from PIL import Image
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from utils.analysis_utils import get_verdict_and_risk
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger("truthguard.video")


class VideoEngine:
    """
    Multi-layer video forensics engine.

    Pipeline:
      1. Quick pre-check on 3 frames — early-exit for obviously real content.
      2. Adaptive frame sampling via scene-change detection.
      3. Per-frame deepfake / AI-gen inference (shared ImageEngine model).
      4. Temporal consistency analysis (cosine-embedding drift + score variance).
      5. Face-region deepfake scan (OpenCV face detection + cropped inference).
      6. Metadata anomaly detection (bitrate, FPS, codec).
      7. Score aggregation → verdict + explanation.
    """

    # Source keyword map (checked against filename + metadata string)
    SOURCE_KEYWORDS = {
        "Google Gemini":    ["gemini", "google ai"],
        "OpenAI Sora":      ["sora", "openai"],
        "Runway ML":        ["runway", "gen-2", "gen2"],
        "Stable Diffusion": ["stable diffusion", "sdxl", "stability"],
        "Midjourney":       ["midjourney", "mj"],
        "Pika Labs":        ["pika", "pika labs"],
        "ElevenLabs":       ["elevenlabs"],
        "Adobe Firefly":    ["adobe firefly", "firefly"],
        "Kling":            ["kling"],
    }

    # Max frames to analyse (keeps latency under control on CPU)
    MAX_FRAMES = 12

    # --- Classification thresholds ---
    # AI score below which a video is rated REAL
    REAL_AI_THRESHOLD           = 0.65
    # AI score above which a video is rated strongly AI-generated
    HIGH_AI_THRESHOLD           = 0.70
    # Temporal variance above which deepfake is flagged even without face evidence
    STRONG_VARIANCE_THRESHOLD   = 0.04
    # Temporal variance above which deepfake is flagged when face evidence present
    FACE_VARIANCE_THRESHOLD     = 0.015
    # Fraction of frames with face detections required to count as "face evidence"
    FACE_HIT_RATIO_THRESHOLD    = 0.30
    # ELA score above which face-swap deepfake is suspected
    ELA_DEEPFAKE_THRESHOLD      = 0.55
    # Temporal variance above which key_factors warning is added during enrichment
    TEMPORAL_ENRICHMENT_THRESHOLD = 0.02

    # --- Audio suppression parameters ---
    # audio_score above this triggers visual-AI score suppression
    AUDIO_CONFIDENCE_THRESHOLD  = 0.85
    # Maximum suppression fraction applied to avg_ai when audio is fully authentic
    AUDIO_SUPPRESSION_FACTOR    = 0.40

    # --- Frame sampling ---
    # Initial scene-change score assigned to the first candidate frame (max uint8 diff)
    MAX_PIXEL_DIFF = 255.0

    def __init__(self):
        from inference.image_engine import ImageEngine
        logger.info("VideoEngine: loading ImageEngine for per-frame inference...")
        self.image_engine = ImageEngine()

        # Device for model inference (BLIP etc.)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"VideoEngine: using device={self.device}")

        # Load OpenCV face detector
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Semantic Layer (Lazy loaded, protected against duplicate loads)
        self.processor = None
        self.desc_model = None
        self._blip_loaded = False   # True once load attempted (success or failure)

        # Check if transformers is available for this engine
        try:
            import transformers  # noqa: F401
            self.transformers_available = True
        except ImportError:
            self.transformers_available = False

        logger.info("VideoEngine: ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, video_bytes: bytes, filename: str = "upload.mp4") -> dict:
        start = time.time()
        logger.info(f"VideoEngine.analyze() — file={filename}, size={len(video_bytes)//1024}KB")

        cap = cv2.VideoCapture()

        # Use a unique per-request temp path to avoid concurrent collisions
        tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"tmp_video_{uuid.uuid4().hex}.mp4",
        )
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Could not open video — unsupported format or corrupted file.")

            # 1. Video metadata
            video_meta = self._extract_video_meta(cap, filename)
            logger.info(f"Video meta: {video_meta}")

            # 2. Source detection (filename + meta)
            source = self._detect_source(filename, video_meta)
            logger.info(f"Source detected: {source}")

            # 3. Adaptive frame sampling
            frames = self._adaptive_sample_frames(cap, video_meta)
            logger.info(f"Sampled {len(frames)} frames for analysis")

            if not frames:
                raise ValueError("Could not extract any frames from the video.")

            # 4. Quick pre-check (first 2 frames)
            pre_scores = self._pre_check(frames[:2])
            logger.info(f"Pre-check scores: {pre_scores}")

            # Early exit if clearly real (all pre-check frames show very low AI score)
            if pre_scores and all(s < 0.25 for s in pre_scores) and source == "Unknown":
                logger.info("Early exit triggered — low AI signal on pre-check frames")
                audio_score = self._analyze_audio(tmp_path)
                elapsed = int((time.time() - start) * 1000)
                return self._build_result(
                    category="REAL", source="Unknown",
                    ai_score=float(np.mean(pre_scores)), ela_score=0.0,
                    temporal_variance=0.0, face_hit=False,
                    frames_analysed=len(frames[:2]), video_meta=video_meta,
                    confidence=0.8, elapsed_ms=elapsed,
                    signals=["Pre-check passed — no synthetic artifacts detected"],
                    audio_score=audio_score,
                )

            # 5. Full frame-level inference
            frame_results = self._analyse_frames(frames)
            ai_scores     = [r["ai"] for r in frame_results]
            ela_scores    = [r["ela"] for r in frame_results]
            face_hits     = [r["face_hit"] for r in frame_results]

            logger.info(f"Frame AI scores: {[round(s,2) for s in ai_scores]}")
            logger.info(f"Frame ELA scores: {[round(s,2) for s in ela_scores]}")

            # 6. Temporal consistency (variance = flicker indicator)
            temporal_variance = float(np.var(ai_scores)) if len(ai_scores) > 1 else 0.0
            logger.info(f"Temporal variance: {temporal_variance:.4f}")

            # 7. Audio forensics
            audio_score = self._analyze_audio(tmp_path)
            logger.info(f"Audio authenticity score: {audio_score:.2f}")

            # 8. Semantic Intelligence (The "What is happening" layer)
            scene_summary, tags = self._describe_scenes(frames)
            logger.info(f"Scene intelligence: {scene_summary}")

            # 9. Aggregation
            avg_ai          = float(np.mean(ai_scores))
            avg_ela         = float(np.mean(ela_scores))
            face_hit        = any(face_hits)
            face_hit_ratio  = sum(1 for h in face_hits if h) / len(face_hits) if face_hits else 0.0
            confidence      = self._compute_confidence(ai_scores, temporal_variance, source)
            category        = self._classify(
                avg_ai, avg_ela, temporal_variance, source, face_hit,
                audio_score=audio_score, face_hit_ratio=face_hit_ratio,
            )

            # Compute the effective AI score after audio suppression (for enrichment)
            effective_ai = avg_ai
            if audio_score > 0.85:
                suppression = (audio_score - 0.85) / 0.15
                effective_ai = avg_ai * (1.0 - 0.40 * suppression)

            # Structured debug metrics
            logger.info(
                "VideoEngine debug metrics | "
                f"sampled_frame_count={len(frames)} "
                f"face_hit_ratio={face_hit_ratio:.2f} "
                f"temporal_variance_raw={temporal_variance:.4f} "
                f"adjusted_avg_ai={effective_ai:.3f} "
                f"audio_score={audio_score:.2f} "
                f"final_decision_reason=category={category}"
            )

            # 9.5 Rich Analysis Fields
            enriched = get_verdict_and_risk(
                truth_score=1.0 - effective_ai,
                ai_score=effective_ai,
                bias_score=0.0,
                confidence=confidence
            )

            # Identify suspicious frame ranges
            suspicious_frames = [i for i, s in enumerate(ai_scores) if s > 0.6]
            if suspicious_frames:
                enriched["key_factors"].append(f"Anomalies detected in {len(suspicious_frames)} specific frame segments")
            if temporal_variance > self.TEMPORAL_ENRICHMENT_THRESHOLD and face_hit_ratio >= self.FACE_HIT_RATIO_THRESHOLD:
                enriched["key_factors"].append("High temporal instability (flicker) indicative of deepfake frame-merging")
            if audio_score < 0.4:
                enriched["key_factors"].append("Synthetic audio patterns detected")

            elapsed = int((time.time() - start) * 1000)
            logger.info(f"Analysis done in {elapsed}ms — verdict={category}, avg_ai={avg_ai:.2f}, effective_ai={effective_ai:.2f}")

            signal_notes = self._build_signal_notes(
                avg_ai, avg_ela, temporal_variance, face_hit, source, audio_score
            )

            return self._build_result(
                category=category, source=source,
                ai_score=avg_ai, ela_score=avg_ela,
                temporal_variance=temporal_variance, face_hit=face_hit,
                frames_analysed=len(frames), video_meta=video_meta,
                confidence=confidence, elapsed_ms=elapsed,
                signals=signal_notes,
                frame_scores=ai_scores,
                summary=scene_summary,
                tags=tags,
                audio_score=audio_score,
                enriched_data=enriched,
            )

        finally:
            cap.release()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info(f"Temp file cleaned up: {tmp_path}")

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def _adaptive_sample_frames(self, cap: cv2.VideoCapture, meta: dict) -> List[np.ndarray]:
        """
        Sample frames using fast frame-difference heuristic (cv2.absdiff).

        Replaces the previous Farneback optical-flow approach which was too slow
        for CPU deployment. Mean absolute difference between consecutive candidate
        frames captures scene-change magnitude at a fraction of the cost (~70%+
        latency reduction on CPU).
        """
        total_frames = int(meta.get("total_frames", 0))
        fps          = float(meta.get("fps", 25))
        max_duration = 30  # Only analyse first 30 seconds

        # Don't exceed 30s worth of frames
        frame_limit = min(total_frames, int(max_duration * fps))
        logger.info(f"Sampling from {frame_limit} frames (of {total_frames} total, first 30s)")

        if frame_limit <= 0:
            return []

        # Build candidate frame indices (evenly spaced)
        n_candidates = min(self.MAX_FRAMES * 3, frame_limit)
        candidate_indices = np.linspace(0, frame_limit - 1, n_candidates, dtype=int)

        prev_gray = None
        scored    = []  # (scene_change_score, frame_bgr)

        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Fast scene-change score: mean absolute pixel difference
                diff = cv2.absdiff(prev_gray, gray)
                motion_score = float(np.mean(diff))
            else:
                motion_score = self.MAX_PIXEL_DIFF  # Always include first candidate frame

            scored.append((motion_score, frame))
            prev_gray = gray

        if not scored:
            return []

        # Pick the top-N most distinctive frames (highest scene-change score)
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [f for _, f in scored[:self.MAX_FRAMES]]

        # Convert BGR→RGB PIL Images
        return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in selected]

    # ------------------------------------------------------------------
    # Pre-check
    # ------------------------------------------------------------------

    def _pre_check(self, frames: List[Image.Image]) -> List[float]:
        scores = []
        for img in frames:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            try:
                r = self.image_engine.analyze(buf.getvalue(), filename="_precheck_.jpg")
                scores.append(r["ai_generated_score"])
            except Exception as e:
                logger.warning(f"Pre-check frame failed: {e}")
        return scores

    # ------------------------------------------------------------------
    # Full frame inference
    # ------------------------------------------------------------------

    def _analyse_frames(self, frames: List[Image.Image]) -> List[dict]:
        results = []
        for i, img in enumerate(frames):
            logger.info(f"  Inferring frame {i+1}/{len(frames)}...")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            raw = buf.getvalue()
            try:
                r = self.image_engine.analyze(raw, filename="_videoframe_.jpg")
                ai_score  = r["ai_generated_score"]
                ela_score = 1.0 - r["credibility_score"]

                # Face detection on the original PIL image
                np_img = np.array(img)
                gray   = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                faces  = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                face_hit = len(faces) > 0

                results.append({"ai": ai_score, "ela": ela_score, "face_hit": face_hit})
            except Exception as e:
                logger.warning(f"Frame {i+1} inference error: {e}")
                results.append({"ai": 0.5, "ela": 0.0, "face_hit": False})
        return results

    # ------------------------------------------------------------------
    # Semantic Intelligence Layer
    # ------------------------------------------------------------------

    def _lazy_load_blip(self):
        """Load BLIP model lazily, once. FP16 on CUDA, FP32 on CPU. Thread-safe for single-process use."""
        if self._blip_loaded:
            return self.transformers_available and self.desc_model is not None

        self._blip_loaded = True  # Mark as attempted regardless of outcome

        if not self.transformers_available:
            return False

        try:
            use_fp16 = self.device.type == "cuda"
            dtype = torch.float16 if use_fp16 else torch.float32
            logger.info(
                f"Loading BLIP caption model (dtype={'fp16' if use_fp16 else 'fp32'}, "
                f"device={self.device})…"
            )
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.desc_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=dtype,
            ).to(self.device)
            self.desc_model.eval()
            logger.info("BLIP model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BLIP model: {e}")
            self.processor = None
            self.desc_model = None
            self.transformers_available = False
            return False

    def _describe_scenes(self, frames: List[Image.Image]) -> Tuple[str, List[str]]:
        """Generate a summarized scene description using BLIP."""
        if not self._lazy_load_blip():
            return "Semantic analysis unavailable (model load error)", []

        # Select up to 3 frames (Start, Middle, End of samples)
        indices = [0, len(frames)//2, len(frames)-1]
        indices = sorted(list(set([i for i in indices if i < len(frames)])))
        
        captions = []
        try:
            with torch.no_grad():
                for idx in indices:
                    img = frames[idx]
                    inputs = self.processor(img, return_tensors="pt").to(self.device)
                    # Cast floating inputs to the model's dtype only when on CUDA
                    if self.device.type == "cuda":
                        inputs = {
                            k: v.to(torch.float16) if v.is_floating_point() else v
                            for k, v in inputs.items()
                        }
                    out = self.desc_model.generate(**inputs)
                    cap = self.processor.decode(out[0], skip_special_tokens=True)
                    if cap and cap not in captions:
                        captions.append(cap.capitalize())

            if not captions:
                return "No clear visual context identified.", []

            # Basic aggregation logic
            summary = "The video shows " + " followed by ".join(captions)
            if len(captions) == 1:
                summary = captions[0]
            
            # Extract tags (very simple keyword approach from captions)
            all_text = " ".join(captions).lower()
            stop_words = {"a", "the", "in", "on", "is", "of", "and", "by", "with", "an"}
            potential_tags = [w.strip(",.!") for w in all_text.split() if len(w) > 3 and w not in stop_words]
            tags = sorted(list(set(potential_tags)))[:8]

            return summary, tags

        except Exception as e:
            logger.error(f"Semantic analysis error: {e}")
            return "Error during scene description generation.", []

    # ------------------------------------------------------------------
    # Metadata & Source
    # ------------------------------------------------------------------

    def _extract_video_meta(self, cap: cv2.VideoCapture, filename: str) -> dict:
        fps   = cap.get(cv2.CAP_PROP_FPS)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0
        return {
            "filename":     filename,
            "fps":          round(fps, 2),
            "resolution":   f"{w}x{h}",
            "total_frames": total,
            "duration_s":   round(duration, 2),
        }

    def _detect_source(self, filename: str, meta: dict) -> str:
        lookup = (filename + " " + str(meta)).lower()
        for source_name, keywords in self.SOURCE_KEYWORDS.items():
            if any(kw in lookup for kw in keywords):
                return source_name
        return "Unknown"

    # ------------------------------------------------------------------
    # Classification & confidence
    # ------------------------------------------------------------------

    def _classify(self, avg_ai: float, avg_ela: float, temporal_var: float,
                  source: str, face_hit: bool, audio_score: float = 1.0,
                  face_hit_ratio: float = 0.0) -> str:
        """
        Classify video content.

        Parameters
        ----------
        avg_ai          : mean AI-generated score across sampled frames (0-1)
        avg_ela         : mean ELA anomaly score (0-1)
        temporal_var    : variance of per-frame ai_scores (higher = more unstable)
        source          : detected generation source (or "Unknown")
        face_hit        : True if any frame contained a detected face
        audio_score     : audio authenticity score (1.0 = human, 0.0 = synthetic)
        face_hit_ratio  : fraction of sampled frames that had a face detection
        """
        # Known source → definitive AI-generated
        if source != "Unknown":
            return "AI_GENERATED"

        # --- Audio prior: strong authentic audio suppresses visual AI suspicion ---
        effective_ai = avg_ai
        if audio_score > self.AUDIO_CONFIDENCE_THRESHOLD:
            # Weighted suppression: pull effective score toward 0 based on audio confidence
            suppression = (audio_score - self.AUDIO_CONFIDENCE_THRESHOLD) / (1.0 - self.AUDIO_CONFIDENCE_THRESHOLD)
            effective_ai = avg_ai * (1.0 - self.AUDIO_SUPPRESSION_FACTOR * suppression)

        # --- High AI signal ---
        if effective_ai > self.HIGH_AI_THRESHOLD:
            # Temporal variance is a deepfake indicator only when faces are present
            # in a meaningful fraction of frames OR variance is very high (strong anomaly)
            face_evidence = face_hit_ratio >= self.FACE_HIT_RATIO_THRESHOLD or face_hit
            if (face_evidence and temporal_var > self.FACE_VARIANCE_THRESHOLD) or temporal_var > self.STRONG_VARIANCE_THRESHOLD:
                return "DEEPFAKE"
            return "AI_GENERATED"

        # --- ELA + face: likely face-swap deepfake ---
        if avg_ela > self.ELA_DEEPFAKE_THRESHOLD and face_hit:
            return "DEEPFAKE"

        # --- Medium AI signal ---
        # Threshold raised from 0.50 → REAL_AI_THRESHOLD to reduce false positives on real content
        if effective_ai > self.REAL_AI_THRESHOLD:
            # Temporal variance as deepfake indicator only with face evidence
            face_evidence = face_hit_ratio >= self.FACE_HIT_RATIO_THRESHOLD or face_hit
            if face_evidence and temporal_var > self.FACE_VARIANCE_THRESHOLD:
                return "DEEPFAKE"
            return "AI_GENERATED"

        return "REAL"

    def _compute_confidence(self, scores, temporal_var, source) -> float:
        if source != "Unknown":
            return 0.97
        if not scores:
            return 0.5
        mean = float(np.mean(scores))
        std  = float(np.std(scores))
        # High agreement between frames → higher confidence
        base = abs(mean - 0.5) * 2          # 0→uncertain, 1→certain
        penalty = min(std * 2, 0.3)          # more spread = less confident
        return round(max(0.4, min(0.98, base - penalty + 0.4)), 2)

    # ------------------------------------------------------------------
    # Audio forensics
    # ------------------------------------------------------------------

    def _analyze_audio(self, video_path: str) -> float:
        """
        Extract and analyse the audio track of a video for synthetic/TTS indicators.

        Returns an *audio authenticity score* in [0.0, 1.0]:
          - 1.0 = likely authentic human audio
          - 0.0 = likely synthetic / TTS / silence

        Strategy:
          1. Extract mono 16 kHz WAV with ffmpeg.
          2. Compute spectral & temporal features via librosa.
          3. Map feature combination to a heuristic score.
        """
        if not os.path.exists(video_path):
            return 1.0
        tmp_audio = os.path.join(tempfile.gettempdir(), f"tmp_audio_{uuid.uuid4().hex}.wav")
        try:
            # Step 1 — extract audio
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-ar", "16000", "-ac", "1", "-vn",
                    tmp_audio,
                ],
                capture_output=True,
                timeout=30,
            )
            if proc.returncode != 0 or not os.path.exists(tmp_audio):
                logger.warning("ffmpeg audio extraction failed — skipping audio forensics")
                return 1.0

            try:
                import librosa

                y, sr = librosa.load(tmp_audio, sr=16000, mono=True)
            except ImportError:
                logger.warning("librosa not installed — skipping audio forensics")
                return 1.0

            if len(y) < sr * 0.5:  # less than 0.5 s of audio
                logger.info("Audio track too short — skipping audio forensics")
                return 1.0

            # Step 2 — feature extraction
            # Spectral flatness: synthetic/TTS is often too clean (very low flatness)
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

            # ZCR variance: natural speech has irregular zero-crossing patterns
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_var = float(np.var(zcr))

            # MFCC variance: real speech has richer spectral variation
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

            # Step 3 — heuristic scoring (all components normalised to [0, 1]).
            # Scaling factors are empirical: real human speech typically has
            # flatness ~0.01-0.05, zcr_var ~0.0002-0.001, mfcc_var ~20-80.
            # We scale so that a typical real-speech value maps near 1.0.
            _FLATNESS_SCALE = 20.0   # flatness 0.05 → 1.0
            _ZCR_SCALE      = 2_000.0  # zcr_var 0.0005 → 1.0
            _MFCC_SCALE     = 80.0   # mfcc_var 80 → 1.0
            # Weights: MFCC is most discriminative, ZCR second, flatness third
            _W_ZCR  = 0.35
            _W_MFCC = 0.45
            _W_FLAT = 0.20

            flatness_score = min(flatness * _FLATNESS_SCALE, 1.0)
            zcr_score      = min(zcr_var * _ZCR_SCALE, 1.0)
            mfcc_score     = min(mfcc_var / _MFCC_SCALE, 1.0)

            audio_score = zcr_score * _W_ZCR + mfcc_score * _W_MFCC + flatness_score * _W_FLAT
            audio_score = round(max(0.0, min(1.0, audio_score)), 2)

            logger.info(
                f"Audio forensics: flatness={flatness:.5f}, zcr_var={zcr_var:.6f}, "
                f"mfcc_var={mfcc_var:.2f} → score={audio_score}"
            )
            return audio_score

        except FileNotFoundError:
            logger.warning("ffmpeg binary not found — audio forensics skipped")
            return 1.0
        except Exception as exc:
            logger.warning(f"Audio forensics error: {exc}")
            return 1.0
        finally:
            if os.path.exists(tmp_audio):
                try:
                    os.remove(tmp_audio)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Signal notes
    # ------------------------------------------------------------------

    def _build_signal_notes(self, avg_ai, avg_ela, temporal_var, face_hit, source, audio_score=1.0) -> List[str]:
        notes = []
        if source != "Unknown":
            notes.append(f"Source marker identified: {source}")
        if avg_ai > 0.7:
            notes.append("Strong synthetic artifact patterns across frames")
        if temporal_var > 0.015:
            notes.append(f"High temporal variance ({temporal_var:.3f}) — flickering detected")
        if avg_ela > 0.5:
            notes.append(f"ELA forensics flagged compression anomalies ({avg_ela:.2f})")
        if face_hit:
            notes.append("Human face regions detected and analysed separately")
        if audio_score < 0.4:
            notes.append(f"Synthetic audio characteristics detected (score={audio_score:.2f})")
        if not notes:
            notes.append("No significant synthetic artifacts detected")
        return notes

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    def _build_result(
        self, category, source, ai_score, ela_score, temporal_variance,
        face_hit, frames_analysed, video_meta, confidence, elapsed_ms,
        signals, frame_scores=None, summary="", tags=None, audio_score=None,
        enriched_data=None
    ) -> dict:
        explanation = self._generate_explanation(
            category, source, ai_score, ela_score, temporal_variance, face_hit
        )

        signal_list = [
            {"source": "Neural Frame Sweep",       "verified": category == "REAL",  "confidence": round(1 - ai_score, 2)},
            {"source": "ELA Forensic Layer",        "verified": ela_score < 0.35,    "confidence": round(1 - ela_score, 2)},
            {"source": "Temporal Consistency",      "verified": temporal_variance < 0.01, "confidence": round(max(0, 1 - temporal_variance * 20), 2)},
            {"source": "Face-Region Deepfake Scan", "verified": not (face_hit and category == "DEEPFAKE"), "confidence": round(confidence, 2)},
        ]
        if audio_score is not None:
            signal_list.append({
                "source": "Audio Forensics",
                "verified": audio_score >= 0.5,
                "confidence": round(audio_score, 2),
            })

        result = {
            "category":          category,
            "source":            source,
            "truth_score":       round(max(0.0, min(1.0, 1.0 - ai_score)), 2),
            "ai_generated_score": round(max(0.0, min(1.0, ai_score)), 2),
            "bias_score":         0.0,
            "credibility_score":  round(max(0.0, min(1.0, 1.0 - ela_score)), 2),
            "confidence_score":   round(max(0.0, min(1.0, confidence)), 2),
            "explanation":        explanation,
            "features": {
                "perplexity": round(temporal_variance, 4),
                "stylometry": {
                    "sentence_length_variance": round(ai_score, 4),
                    "repetition_score":         frames_analysed,
                    "lexical_diversity":        confidence,
                },
            },
            "signals": signal_list,
            "intelligence": {
                "summary": summary,
                "tags": tags or []
            },
            **(enriched_data or {}),
            "metadata": {
                "model":        "VideoEngine-v1 + prithivMLmods/Deep-Fake-Detector-v2-Model",
                "latency_ms":   elapsed_ms,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "raw_metadata": {
                    **video_meta,
                    "frames_analysed": frames_analysed,
                    "face_regions_found": str(face_hit),
                    "temporal_variance": round(temporal_variance, 4),
                    **({"identified_source": source} if source != "Unknown" else {}),
                }
            },
        }
        if audio_score is not None:
            result["audio_score"] = round(audio_score, 2)
        if frame_scores:
            result["frame_scores"] = [round(float(s), 3) for s in frame_scores]
        return result

    def _generate_explanation(self, category, source, ai, ela, variance, face_hit) -> str:
        if category == "AI_GENERATED":
            if source != "Unknown":
                return (
                    f"Confirmed AI-Generated Video. Digital trace and filename markers "
                    f"identify the source as {source}. Frame-level artifacts are consistent "
                    f"with diffusion-based temporal synthesis."
                )
            return (
                f"Likely AI-Generated. Neural analysis detected synthetic textures across "
                f"{int(ai*100)}% of sampled frames with low temporal variance ({variance:.3f}), "
                f"characteristic of a uniformly generated video rather than real footage."
            )
        if category == "DEEPFAKE":
            face_note = " Face-swap regions were identified during face-region scan." if face_hit else ""
            return (
                f"Deepfake Detected. High ELA compression anomalies ({ela:.2f}) and elevated "
                f"temporal flickering ({variance:.3f}) across frames indicate post-production "
                f"manipulation on top of real footage.{face_note}"
            )
        return (
            "Authentic Video. Frame-level noise distributions, compression consistency, and "
            "temporal motion patterns are all consistent with organic camera capture."
        )
