import unittest

from inference.image_engine import ImageEngine


class ImageEngineForensicsTests(unittest.TestCase):
    def setUp(self):
        self.engine = ImageEngine.__new__(ImageEngine)

    def test_compose_verdict_ai_generated(self):
        verdict, confidence, aggregates, triggered_rule = self.engine._compose_verdict(
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
        )
        self.assertEqual(verdict, "AI_GENERATED")
        self.assertEqual(triggered_rule, "strong_ai_artifacts")
        self.assertGreaterEqual(aggregates["ai_likelihood"], 0.74)
        self.assertGreaterEqual(confidence, 0.0)

    def test_compose_verdict_edited(self):
        verdict, _, aggregates, triggered_rule = self.engine._compose_verdict(
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
        )
        self.assertEqual(verdict, "EDITED")
        self.assertEqual(triggered_rule, "forensic_edit_anomalies")
        self.assertGreaterEqual(aggregates["edit_likelihood"], 0.64)

    def test_compose_verdict_mixed(self):
        verdict, _, _, triggered_rule = self.engine._compose_verdict(
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
        )
        self.assertEqual(verdict, "MIXED")
        self.assertEqual(triggered_rule, "ai_and_edit_signals_high")


if __name__ == "__main__":
    unittest.main()
