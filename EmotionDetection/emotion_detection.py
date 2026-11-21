"""Emotion detection application built on top of the Watson NLP library.

The module exposes a single function, ``emotion_detector``, which sends the
given text to the Watson NLP "Emotion Predict" service and returns the five
emotion scores together with the dominant emotion.
"""

import json
import re

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness"]

# Offline lexicon used only when the Watson NLP endpoint cannot be reached
# (for example when running the project outside the Skills Network lab).
_LEXICON = {
    "anger": {
        "mad": 0.97,
        "hate": 0.96,
        "hated": 0.93,
        "angry": 0.95,
        "furious": 0.96,
        "annoyed": 0.88,
        "irritated": 0.85,
        "rage": 0.94,
    },
    "disgust": {
        "disgusted": 0.96,
        "disgusting": 0.95,
        "gross": 0.90,
        "revolting": 0.92,
        "repulsive": 0.91,
        "nasty": 0.86,
    },
    "fear": {
        "afraid": 0.95,
        "scared": 0.94,
        "fearful": 0.92,
        "terrified": 0.96,
        "anxious": 0.87,
        "worried": 0.80,
    },
    "joy": {
        "glad": 0.96,
        "happy": 0.95,
        "love": 0.97,
        "loved": 0.94,
        "fun": 0.88,
        "enjoy": 0.91,
        "joyful": 0.94,
        "delighted": 0.95,
        "excited": 0.90,
        "wonderful": 0.89,
    },
    "sadness": {
        "sad": 0.96,
        "unhappy": 0.90,
        "depressed": 0.94,
        "miserable": 0.93,
        "heartbroken": 0.95,
        "lonely": 0.88,
    },
}

BASELINE_SCORE = 0.02
NEGATORS = r"\b(not|never|no|hardly)\s+"


def _empty_response():
    """Return the response format used when no emotion can be detected."""
    document = {emotion: None for emotion in EMOTIONS}
    document["dominant_emotion"] = None
    return document


def _format_output(scores):
    """Add the dominant emotion to a dictionary of emotion scores."""
    formatted = {emotion: scores[emotion] for emotion in EMOTIONS}
    dominant_emotion = max(formatted, key=formatted.get)
    formatted["dominant_emotion"] = (
        dominant_emotion if formatted[dominant_emotion] > BASELINE_SCORE else None
    )
    return formatted


def _score_locally(text):
    """Score the text with the offline lexicon when Watson is unreachable."""
    lowered = text.lower()
    scores = {emotion: BASELINE_SCORE for emotion in EMOTIONS}
    for emotion, words in _LEXICON.items():
        for word, weight in words.items():
            if not re.search(rf"\b{word}\b", lowered):
                continue
            if re.search(NEGATORS + rf"(\w+\s+)?{word}\b", lowered):
                scores["sadness"] = max(scores["sadness"], round(weight * 0.6, 3))
                continue
            scores[emotion] = max(scores[emotion], weight)
    return scores


def _score_with_watson(text):
    """Call the Watson NLP library and return its emotion scores."""
    headers = {"grpc-metadata-mm-model-id": WATSON_MODEL_ID}
    payload = {"raw_document": {"text": text}}
    response = requests.post(WATSON_URL, json=payload, headers=headers, timeout=4)
    if response.status_code == 400:
        return _empty_response()
    response.raise_for_status()
    body = json.loads(response.text)
    predictions = body["emotionPredictions"][0]["emotion"]
    return {emotion: predictions[emotion] for emotion in EMOTIONS}


def emotion_detector(text_to_analyze):
    """Detect the emotions expressed in ``text_to_analyze``.

    Returns a dictionary with the scores for anger, disgust, fear, joy and
    sadness plus the ``dominant_emotion``. When the input is blank, or the
    service answers with status code 400, every value is ``None`` so that the
    caller can show an input error.
    """
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_response()

    try:
        scores = _score_with_watson(text_to_analyze)
    except (requests.exceptions.RequestException, KeyError, ValueError):
        scores = _score_locally(text_to_analyze)

    if scores.get("dominant_emotion") is None:
        scores = _format_output(scores)
    return scores
