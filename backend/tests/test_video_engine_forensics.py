import unittest

try:
    from inference.video_engine import VideoEngine
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent dependency
    VideoEngine = None  # type: ignore[assignment]
    _VIDEO_IMPORT_ERROR = exc
else:
    _VIDEO_IMPORT_ERROR = None


class VideoEngineForensicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if VideoEngine is None:
            raise unittest.SkipTest(f"VideoEngine dependencies unavailable: {_VIDEO_IMPORT_ERROR}")

    def setUp(self):
        self.engine = VideoEngine.__new__(VideoEngine)
        self.engine.AUDIO_CONFIDENCE_THRESHOLD = 0.85
        self.engine.AUDIO_SUPPRESSION_FACTOR = 0.40
        self.engine.HIGH_AI_THRESHOLD = 0.70
        self.engine.FACE_HIT_RATIO_THRESHOLD = 0.30
        self.engine.FACE_VARIANCE_THRESHOLD = 0.015
        self.engine.STRONG_VARIANCE_THRESHOLD = 0.04
        self.engine.ELA_DEEPFAKE_THRESHOLD = 0.55
        self.engine.REAL_AI_THRESHOLD = 0.65

    def test_videoize_caption_replaces_image_words(self):
        caption = self.engine._videoize_caption("likely human photo in studio")
        self.assertIn("video footage", caption.lower())
        self.assertIn("video", caption.lower())

    def test_classify_uses_video_taxonomy_for_deepfake(self):
        verdict = self.engine._classify(
            avg_ai=0.78,
            avg_ela=0.41,
            temporal_var=0.03,
            source="Unknown",
            face_hit=True,
            audio_score=0.55,
            face_hit_ratio=0.85,
            temporal_metrics={"temporal_consistency": 0.44, "face_stability": 0.41, "compression_consistency": 0.58},
            scene_profile={"scene_class": "talking_head"},
        )
        self.assertEqual(verdict, "Deepfake / Face Manipulation Suspected")

    def test_build_result_contains_video_forensics_signals(self):
        result = self.engine._build_result(
            category="Likely Real Recorded Footage",
            source="Unknown",
            ai_score=0.23,
            ela_score=0.18,
            temporal_variance=0.006,
            face_hit=True,
            frames_analysed=6,
            video_meta={"fps": 30, "resolution": "1920x1080", "duration_s": 4.1},
            confidence=0.84,
            elapsed_ms=1530,
            signals=["Video appears authentic"],
            frame_scores=[0.22, 0.24, 0.21],
            summary="Real human speaking on camera.",
            tags=["talking-head", "human"],
            audio_score=0.88,
            temporal_metrics={
                "temporal_consistency": 0.84,
                "motion_integrity": 0.8,
                "face_stability": 0.77,
                "compression_consistency": 0.73,
                "flicker_risk": 0.17,
            },
            scene_profile={"scene_class": "talking_head"},
        )
        self.assertIn("video_forensics", result)
        self.assertEqual(result["category"], "Likely Real Recorded Footage")
        self.assertEqual(result["video_forensics"]["scene_class"], "talking_head")
        signal_sources = {signal["source"] for signal in result["signals"]}
        self.assertIn("Authenticity Score", signal_sources)
        self.assertIn("Compression Consistency", signal_sources)


if __name__ == "__main__":
    unittest.main()
