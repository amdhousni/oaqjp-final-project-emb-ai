"""Unit tests for the EmotionDetection application."""

import unittest

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Validate that the dominant emotion is correct for known statements."""

    def test_joy(self):
        """A glad statement must be detected as joy."""
        response = emotion_detector("I am glad this happened")
        self.assertEqual(response["dominant_emotion"], "joy")

    def test_anger(self):
        """A mad statement must be detected as anger."""
        response = emotion_detector("I am really mad about this")
        self.assertEqual(response["dominant_emotion"], "anger")

    def test_disgust(self):
        """A disgusted statement must be detected as disgust."""
        response = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(response["dominant_emotion"], "disgust")

    def test_sadness(self):
        """A sad statement must be detected as sadness."""
        response = emotion_detector("I am so sad about this")
        self.assertEqual(response["dominant_emotion"], "sadness")

    def test_fear(self):
        """An afraid statement must be detected as fear."""
        response = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(response["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
