import unittest

from inference.image_engine import ImageEngine


class ImageEngineForensicsTests(unittest.TestCase):
    def setUp(self):
        self.engine = ImageEngine.__new__(ImageEngine)

    def test_compose_verdict_ai_generated(self):
        verdict, confidence, aggregates, triggered_rule, _ = self.engine._compose_verdict(
            content_type="AI_SYNTHETIC",
            source="Unknown",
            neural_score=0.88,
            signal_scores={
                "texture_consistency": 0.75,
                "noise_pattern_mismatch": 0.72,
                "object_realism": 0.68,
                "metadata_anomalies": 0.6,
                "ela": 0.3,
                "compression_anomalies": 0.24,
                "edge_artifacts": 0.35,
                "shadow_mismatch": 0.18,
                "lighting_consistency": 0.22,
            },
            metadata_evidence={"has_camera_indicators": False, "has_generator_markers": False},
        )
        self.assertEqual(verdict, "AI_GENERATED_SYNTHETIC_IMAGE")
        self.assertEqual(triggered_rule, "strong_ai_artifacts")
        self.assertGreaterEqual(aggregates["ai_likelihood"], 0.74)
        self.assertGreaterEqual(confidence, 0.0)

    def test_compose_verdict_edited(self):
        verdict, _, aggregates, triggered_rule, _ = self.engine._compose_verdict(
            content_type="EDITED_COMPOSITE",
            source="Unknown",
            neural_score=0.31,
            signal_scores={
                "texture_consistency": 0.25,
                "noise_pattern_mismatch": 0.22,
                "object_realism": 0.2,
                "metadata_anomalies": 0.25,
                "ela": 0.82,
                "compression_anomalies": 0.78,
                "edge_artifacts": 0.69,
                "shadow_mismatch": 0.62,
                "lighting_consistency": 0.58,
            },
            metadata_evidence={"has_camera_indicators": False, "has_generator_markers": False},
        )
        self.assertEqual(verdict, "EDITED_MANIPULATED_IMAGE")
        self.assertEqual(triggered_rule, "forensic_edit_anomalies")
        self.assertGreaterEqual(aggregates["edit_likelihood"], 0.64)

    def test_compose_verdict_composite_potential_deepfake(self):
        verdict, _, _, triggered_rule, _ = self.engine._compose_verdict(
            content_type="AI_SYNTHETIC",
            source="OpenAI",
            neural_score=0.9,
            signal_scores={
                "texture_consistency": 0.8,
                "noise_pattern_mismatch": 0.74,
                "object_realism": 0.66,
                "metadata_anomalies": 0.9,
                "ela": 0.75,
                "compression_anomalies": 0.72,
                "edge_artifacts": 0.68,
                "shadow_mismatch": 0.65,
                "lighting_consistency": 0.52,
            },
            metadata_evidence={"has_camera_indicators": False, "has_generator_markers": True},
        )
        self.assertEqual(verdict, "COMPOSITE_POTENTIAL_DEEPFAKE")
        self.assertEqual(triggered_rule, "ai_and_edit_signals_high")

    def test_compose_verdict_hand_drawn(self):
        verdict, _, _, triggered_rule, rejected = self.engine._compose_verdict(
            content_type="HAND_DRAWN",
            source="Unknown",
            neural_score=0.42,
            signal_scores={
                "texture_consistency": 0.2,
                "noise_pattern_mismatch": 0.18,
                "object_realism": 0.2,
                "metadata_anomalies": 0.2,
                "ela": 0.22,
                "compression_anomalies": 0.2,
                "edge_artifacts": 0.25,
                "shadow_mismatch": 0.2,
                "lighting_consistency": 0.2,
            },
            metadata_evidence={"has_camera_indicators": False, "has_generator_markers": False},
        )
        self.assertEqual(verdict, "HAND_DRAWN_SKETCH_ARTWORK")
        self.assertEqual(triggered_rule, "content_type_hand_drawn")
        self.assertIn("Authentic Real Photograph", rejected)

    def test_contextualize_forensics_suppresses_face_signal_when_no_human(self):
        weighted_signals, weighted_scores, _, _ = self.engine._contextualize_forensic_signals(
            raw_scores={
                "ela": 0.5,
                "texture_consistency": 0.4,
                "edge_artifacts": 0.5,
                "lighting_consistency": 0.4,
                "shadow_mismatch": 0.35,
                "noise_pattern_mismatch": 0.42,
                "compression_anomalies": 0.3,
                "face_hand_inconsistency": 0.77,
                "object_realism": 0.33,
                "metadata_anomalies": 0.4,
            },
            explanations={
                "face_hand_inconsistency": {"explanation": "Face artifacts."},
                "ela": {"explanation": "ELA."},
                "texture_consistency": {"explanation": "Texture."},
                "edge_artifacts": {"explanation": "Edge."},
                "lighting_consistency": {"explanation": "Lighting."},
                "shadow_mismatch": {"explanation": "Shadow."},
                "noise_pattern_mismatch": {"explanation": "Noise."},
                "compression_anomalies": {"explanation": "Compression."},
                "object_realism": {"explanation": "Realism."},
                "metadata_anomalies": {"explanation": "Metadata."},
            },
            content_type="REAL_PHOTO",
            human_present=False,
            metadata_evidence={"has_camera_indicators": False, "has_generator_markers": False},
            neural_confidence=0.6,
        )

        self.assertLessEqual(weighted_scores["face_hand_inconsistency"], 0.08)
        self.assertTrue(weighted_signals["face_hand_inconsistency"]["technical_only"])


if __name__ == "__main__":
    unittest.main()
