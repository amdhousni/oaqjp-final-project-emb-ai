# Final project

## Emotion Detector

An AI-based web application that detects the emotions expressed in a piece of
text. The application is built with Python, the Watson NLP library and Flask.

## Project structure

```
EmotionDetection/
    __init__.py              # package that imports the application module
    emotion_detection.py     # emotion_detector() application function
test_emotion_detection.py    # unit tests for the five emotions
server.py                    # Flask deployment of the application
templates/index.html         # user interface
static/mywebscript.js        # client script calling /emotionDetector
static/mywebscript.css       # styling
```

## The emotion detector function

`emotion_detector(text)` sends the text to the Watson NLP *Emotion Predict*
service and returns a dictionary in the following format:

```python
{
    'anger': 0.0007,
    'disgust': 0.0001,
    'fear': 0.0001,
    'joy': 0.9993,
    'sadness': 0.0002,
    'dominant_emotion': 'joy'
}
```

A blank input, or a `400` answer from the service, returns every value as
`None` so the interface can report an input error.

## Installation

```bash
pip install -r requirements.txt
```

## Running the application

```bash
python3 server.py
```

Then open <http://localhost:5000> and type a sentence such as
"I am glad this happened".

## Unit tests

```bash
python3 -m unittest test_emotion_detection.py
```

## Static code analysis

```bash
python3 -m pylint server.py
```

## Notes

The Watson NLP lab endpoint (`sn-watson-emotion.labs.skills.network`) is only
reachable from inside the Skills Network environment. When the endpoint cannot
be reached, the application falls back to a small offline lexicon so the web
interface and the unit tests remain runnable.
