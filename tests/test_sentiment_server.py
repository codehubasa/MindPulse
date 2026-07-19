import unittest

from sentiment_server import analyze_sentiment


class SentimentServerTests(unittest.TestCase):
    def test_positive_text_returns_positive_payload(self):
        payload = analyze_sentiment("I absolutely love this amazing and wonderful update")

        self.assertEqual(payload["tone"], "positive")
        self.assertEqual(payload["verdict"], "Very Positive")
        self.assertGreater(payload["polarity"], 0.3)

    def test_negative_text_returns_negative_payload(self):
        payload = analyze_sentiment("This is terrible, awful, and disappointing")

        self.assertEqual(payload["tone"], "negative")
        self.assertEqual(payload["verdict"], "Very Negative")
        self.assertLess(payload["polarity"], -0.3)

    def test_negated_phrase_is_not_overly_positive(self):
        payload = analyze_sentiment("The product is not great and not wonderful")

        self.assertEqual(payload["tone"], "neutral")
        self.assertLessEqual(payload["polarity"], 0.2)


if __name__ == "__main__":
    unittest.main()
