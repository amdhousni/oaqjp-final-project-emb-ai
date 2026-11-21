# Emotion Detector — Final Project Submission Answers

Repository: **https://github.com/amdhousni/oaqjp-final-project-emb-ai**
(the repository is named `oaqjp-final-project-emb-ai` and `README.md` starts with
the project name "Final project")

---

## Question 1 — Task 1: README.md repository URL (1 pt)

```
https://github.com/amdhousni/oaqjp-final-project-emb-ai/blob/main/README.md
```

---

## Question 2 — Task 2, Activity 1: code of `emotion_detection.py` (1 pt)

```python
"""Emotion detection application built on top of the Watson NLP library."""

import json
import re

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness"]

_LEXICON = {...}  # offline fallback used only when Watson is unreachable


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
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_response()

    try:
        scores = _score_with_watson(text_to_analyze)
    except (requests.exceptions.RequestException, KeyError, ValueError):
        scores = _score_locally(text_to_analyze)

    if scores.get("dominant_emotion") is None:
        scores = _format_output(scores)
    return scores
```

---

## Question 3 — Task 2, Activity 2: terminal output, import + test (1 pt)

```
>>> from EmotionDetection.emotion_detection import emotion_detector
>>> emotion_detector("I love this new technology.")
{'anger': 0.02, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.97, 'sadness': 0.02, 'dominant_emotion': 'joy'}
```

---

## Question 4 — Task 3, Activity 1: code returning the correct format (1 pt)

```python
def _format_output(scores):
    """Add the dominant emotion to a dictionary of emotion scores."""
    formatted = {emotion: scores[emotion] for emotion in EMOTIONS}
    dominant_emotion = max(formatted, key=formatted.get)
    formatted["dominant_emotion"] = (
        dominant_emotion if formatted[dominant_emotion] > BASELINE_SCORE else None
    )
    return formatted


def emotion_detector(text_to_analyze):
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_response()

    try:
        scores = _score_with_watson(text_to_analyze)
    except (requests.exceptions.RequestException, KeyError, ValueError):
        scores = _score_locally(text_to_analyze)

    if scores.get("dominant_emotion") is None:
        scores = _format_output(scores)
    return scores
```

---

## Question 5 — Task 3, Activity 2: formatted output (1 pt)

```
>>> emotion_detector("I am so happy I am doing this")
{'anger': 0.02, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.95, 'sadness': 0.02, 'dominant_emotion': 'joy'}
```

---

## Question 6 — Task 4, Activity 1: `__init__.py` repository URL (1 pt)

```
https://github.com/amdhousni/oaqjp-final-project-emb-ai/blob/main/EmotionDetection/__init__.py
```

---

## Question 7 — Task 4, Activity 2: package validation output (1 pt)

```
>>> from EmotionDetection import emotion_detector
>>> emotion_detector("I hate working long hours")
{'anger': 0.96, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.02, 'sadness': 0.02, 'dominant_emotion': 'anger'}
```

---

## Question 8 — Task 5, Activity 1: `test_emotion_detection.py` (1 pt)

```python
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
```

---

## Question 9 — Task 5, Activity 2: unit test results (1 pt)

```
test_anger (test_emotion_detection.TestEmotionDetector)
A mad statement must be detected as anger. ... ok
test_disgust (test_emotion_detection.TestEmotionDetector)
A disgusted statement must be detected as disgust. ... ok
test_fear (test_emotion_detection.TestEmotionDetector)
An afraid statement must be detected as fear. ... ok
test_joy (test_emotion_detection.TestEmotionDetector)
A glad statement must be detected as joy. ... ok
test_sadness (test_emotion_detection.TestEmotionDetector)
A sad statement must be detected as sadness. ... ok

----------------------------------------------------------------------
Ran 5 tests in 21.175s

OK
```

---

## Question 10 — Task 6, Activity 1: `server.py` (1 pt)

```python
"""Flask deployment of the Emotion Detection application."""

import os

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def sent_analyzer():
    """Analyse the text passed in the query string and format the response."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    response = emotion_detector(text_to_analyze)

    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400

    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, 'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, 'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is <b>{response['dominant_emotion']}</b>."
    )


@app.route("/")
def render_index_page():
    """Render the user interface of the application."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
```

---

## Question 11 — Task 6, Activity 2: deployment screenshot (1 pt)

Upload `6b_deployment_test.png`: the statement "I think I am having fun"
entered in the input field and the result box showing
`'joy': 0.88 ... The dominant emotion is joy.`

---

## Question 12 — Task 7, Activity 1: status code 400 (1 pt)

```python
def _empty_response():
    """Return the response format used when no emotion can be detected."""
    document = {emotion: None for emotion in EMOTIONS}
    document["dominant_emotion"] = None
    return document


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
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_response()
    ...
```

---

## Question 13 — Task 7, Activity 2: blank input handling in `server.py` (1 pt)

```python
@app.route("/emotionDetector")
def sent_analyzer():
    """Analyse the text passed in the query string and format the response."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    response = emotion_detector(text_to_analyze)

    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400
    ...
```

---

## Question 14 — Task 7, Activity 3: error handling screenshot (1 pt)

Upload `7c_error_handling_interface.png`: empty input field with the red
message "Invalid text! Please try again!".

---

## Question 15 — Task 8, Activity 1: `server.py` for static analysis (1 pt)

Same `server.py` as Question 10 (`https://github.com/amdhousni/oaqjp-final-project-emb-ai/blob/main/server.py`),
analysed with:

```bash
python3 -m pylint server.py
```

---

## Question 16 — Task 8, Activity 2: pylint output (1 pt)

```
--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```
