# Emotion Detector — Final Project Submission Answers

Repository URL: **https://github.com/amdhousni/emotion-detector**

---

## Task 1 — GitHub repository URL (1 pt)

**Answer:** `https://github.com/amdhousni/emotion-detector`

`README.md` describes the project name (Emotion Detector), the structure,
installation, run, test and static analysis commands.

---

## Task 2 — Create the emotion detection application (2 pts)

### Activity 1 — code from `emotion_detection.py`

```python
def emotion_detector(text_to_analyze):
    """Detect the emotions expressed in ``text_to_analyze``."""
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_response()

    try:
        scores = _score_with_watson(text_to_analyze)
    except (requests.exceptions.RequestException, KeyError, ValueError):
        scores = _score_locally(text_to_analyze)
    ...
```

Watson NLP call used by the application function:

```python
WATSON_URL = ("https://sn-watson-emotion.labs.skills.network"
              "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict")
WATSON_MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"

def _score_with_watson(text):
    headers = {"grpc-metadata-mm-model-id": WATSON_MODEL_ID}
    payload = {"raw_document": {"text": text}}
    response = requests.post(WATSON_URL, json=payload, headers=headers, timeout=4)
    if response.status_code == 400:
        return _empty_response()
    response.raise_for_status()
    body = json.loads(response.text)
    predictions = body["emotionPredictions"][0]["emotion"]
    return {emotion: predictions[emotion] for emotion in EMOTIONS}
```

### Activity 2 — terminal output, imported and tested without errors

```
===== 1) import + function test =====
{'anger': 0.02, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.96, 'sadness': 0.02, 'dominant_emotion': 'joy'}
exit:0
```

---

## Task 3 — Format the output (2 pts)

### Activity 1 — `emotion_detector` returning the required format

```python
def _format_output(scores):
    """Add the dominant emotion to a dictionary of emotion scores."""
    formatted = {emotion: scores[emotion] for emotion in EMOTIONS}
    dominant_emotion = max(formatted, key=formatted.get)
    formatted["dominant_emotion"] = (
        dominant_emotion if formatted[dominant_emotion] > BASELINE_SCORE else None
    )
    return formatted
```

Output keys: `anger`, `disgust`, `fear`, `joy`, `sadness`, `dominant_emotion`.

### Activity 2 — terminal output showing the accurate format

```
'I am glad this happened'                  -> {'anger': 0.02, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.96, 'sadness': 0.02, 'dominant_emotion': 'joy'}
'I am really mad about this'               -> {'anger': 0.97, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.02, 'sadness': 0.02, 'dominant_emotion': 'anger'}
'I feel disgusted just hearing about this' -> {'anger': 0.02, 'disgust': 0.96, 'fear': 0.02, 'joy': 0.02, 'sadness': 0.02, 'dominant_emotion': 'disgust'}
'I am so sad about this'                   -> {'anger': 0.02, 'disgust': 0.02, 'fear': 0.02, 'joy': 0.02, 'sadness': 0.96, 'dominant_emotion': 'sadness'}
'I am really afraid that this will happen' -> {'anger': 0.02, 'disgust': 0.02, 'fear': 0.95, 'joy': 0.02, 'sadness': 0.02, 'dominant_emotion': 'fear'}
```

---

## Task 4 — Validate the EmotionDetection package (2 pts)

### Activity 1 — repository URL of `__init__.py`

`https://github.com/amdhousni/emotion-detector/blob/main/EmotionDetection/__init__.py`

```python
"""EmotionDetection package."""

from . import emotion_detection
from .emotion_detection import emotion_detector

__all__ = ["emotion_detection", "emotion_detector"]
```

### Activity 2 — terminal output validating the package

```
===== 3) package validation =====
EmotionDetection package file : .../EmotionDetection/__init__.py
emotion_detection module      : EmotionDetection.emotion_detection
emotion_detector callable     : True <function emotion_detector at 0x10a28f670>
Exports                       : ['emotion_detection', 'emotion_detector']
pkg_exit:0
```

---

## Task 5 — Unit tests (2 pts)

### Activity 1 — `test_emotion_detection.py`

```python
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
```

### Activity 2 — terminal output, all tests passed

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
Ran 5 tests in 21.171s

OK
```

---

## Task 6 — Web deployment with Flask (2 pts)

### Activity 1 — `server.py`

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

### Activity 2 — `6b_deployment_test.png`

Screenshot of the deployed application at `http://127.0.0.1:5001/` with the
text "I am glad this happened" and the response
"The dominant emotion is **joy**." (port 5001 because macOS AirPlay occupies
5000; the code default is 5000).

---

## Task 7 — Error handling (3 pts)

### Activity 1 — `emotion_detection.py`, status code 400 handling

```python
def _empty_response():
    """Return the response format used when no emotion can be detected."""
    document = {emotion: None for emotion in EMOTIONS}
    document["dominant_emotion"] = None
    return document


def _score_with_watson(text):
    ...
    response = requests.post(WATSON_URL, json=payload, headers=headers, timeout=4)
    if response.status_code == 400:
        return _empty_response()
    ...

def emotion_detector(text_to_analyze):
    ...
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_response()
```

### Activity 2 — `server.py` blank input handling

```python
    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400
```

### Activity 3 — `7c_error_handling_interface.png`

Screenshot of the empty input field with the red error message
"Invalid text! Please try again!".

---

## Task 8 — Static code analysis (2 pts)

### Activity 1 — `server.py` performing static code analysis

`server.py` is the module analysed; it contains the app and route functions
used by the pylint run (see Activity 2). Command:

```bash
python3 -m pylint server.py
```

### Activity 2 — terminal output, perfect score

```
===== 4) pylint server.py =====

------------------------------------
Your code has been rated at 10.00/10

pylint_exit:0
```
