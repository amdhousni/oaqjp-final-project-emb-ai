"""EmotionDetection package.

Importing the package gives access to the ``emotion_detection`` module and to
the ``emotion_detector`` application function.
"""

from . import emotion_detection
from .emotion_detection import emotion_detector

__all__ = ["emotion_detection", "emotion_detector"]
