import unittest

from emotion_server import build_emotion_payload


class EmotionServerTests(unittest.TestCase):
    def test_build_emotion_payload_maps_scores_and_dominant(self):
        raw_probs = {
            "admiration": 0.1,
            "joy": 0.8,
            "sadness": 0.2,
            "anger": 0.05,
            "fear": 0.01,
            "surprise": 0.15,
            "disgust": 0.04,
            "neutral": 0.05,
        }

        payload = build_emotion_payload(raw_probs, "I feel wonderful today")

        self.assertEqual(payload["dominant"]["label"], "Joy")
        self.assertGreater(payload["scores"]["joy"], payload["scores"]["sad"])
        self.assertEqual(sum(payload["scores"].values()), 100)
        self.assertIn(payload["tone"], {"Positive", "Negative", "Neutral"})

    def test_build_emotion_payload_aggregates_positive_labels(self):
        raw_probs = {
            "admiration": 0.9,
            "amusement": 0.8,
            "approval": 0.7,
            "anger": 0.2,
            "fear": 0.05,
            "disgust": 0.1,
            "neutral": 0.05,
        }

        payload = build_emotion_payload(raw_probs, "This is amazing and delightful")

        self.assertGreater(payload["scores"]["joy"], payload["scores"]["ang"])
        self.assertEqual(payload["dominant"]["label"], "Joy")

    def test_positive_text_returns_positive_tone(self):
        raw_probs = {
            "joy": 0.9,
            "excitement": 0.85,
            "love": 0.7,
            "admiration": 0.6,
            "sadness": 0.05,
            "anger": 0.01,
            "fear": 0.01,
            "neutral": 0.02,
        }

        payload = build_emotion_payload(raw_probs, "I am so happy and excited today")

        self.assertEqual(payload["dominant"]["label"], "Joy")
        self.assertEqual(payload["tone"], "Positive")


if __name__ == "__main__":
    unittest.main()
