import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from transformers import BertTokenizer

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotional_detection_model", "bert_classifier_tensorflow_saved_model")
TOKENIZER_PATH = os.path.join(BASE_DIR, "chatbot_model")

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

EMOTION_MAP = {
    "joy": {"label": "Joy", "emoji": "😄", "color": "#fbbf24"},
    "sad": {"label": "Sadness", "emoji": "😢", "color": "#60a5fa"},
    "ang": {"label": "Anger", "emoji": "😡", "color": "#f43f5e"},
    "fea": {"label": "Fear", "emoji": "😰", "color": "#a78bfa"},
    "sur": {"label": "Surprise", "emoji": "😲", "color": "#2dd4bf"},
    "dis": {"label": "Disgust", "emoji": "🤢", "color": "#fb923c"},
}

MODEL = None
TOKENIZER = None


def _load_model():
    global MODEL, TOKENIZER
    if MODEL is None:
        TOKENIZER = BertTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        MODEL = tf.saved_model.load(MODEL_PATH)


def _sigmoid(values):
    return 1 / (1 + np.exp(-values))


def _normalize_scores(probabilities):
    scores = {k: round(float(v) * 100, 1) for k, v in probabilities.items()}
    total = sum(scores.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    normalized = {k: round((v / total) * 100, 1) for k, v in scores.items()}
    return normalized


def build_emotion_payload(raw_probs, text):
    probs = dict(raw_probs)
    scores = {"joy": 0.0, "sad": 0.0, "ang": 0.0, "fea": 0.0, "sur": 0.0, "dis": 0.0}

    positive_labels = {
        "admiration", "amusement", "approval", "caring", "curiosity",
        "desire", "excitement", "gratitude", "joy", "love", "optimism",
        "pride", "relief", "realization"
    }
    sad_labels = {"sadness", "disappointment", "grief", "remorse", "embarrassment"}
    ang_labels = {"anger", "annoyance", "disapproval"}
    fea_labels = {"fear", "nervousness"}
    sur_labels = {"surprise", "curiosity", "realization"}
    dis_labels = {"disgust"}

    for label, value in probs.items():
        prob = float(value)
        if label in positive_labels:
            scores["joy"] += prob
        if label in sad_labels:
            scores["sad"] += prob
        if label in ang_labels:
            scores["ang"] += prob
        if label in fea_labels:
            scores["fea"] += prob
        if label in sur_labels:
            scores["sur"] += prob
        if label in dis_labels:
            scores["dis"] += prob

    if probs.get("neutral", 0.0) > 0.5:
        scores = {key: value * 0.5 for key, value in scores.items()}

    total = sum(scores.values())
    if total <= 0:
        scores = {k: 0.0 for k in scores}
    else:
        scores = {k: round((v / total) * 100, 1) for k, v in scores.items()}

    dominant_key = max(scores, key=scores.get)
    dominant = EMOTION_MAP[dominant_key]

    positive = scores["joy"] + scores["sur"]
    negative = scores["sad"] + scores["ang"] + scores["fea"] + scores["dis"]
    if positive > negative * 1.15 or probs.get("joy", 0.0) > 0.6 or probs.get("excitement", 0.0) > 0.55:
        tone = "Positive"
    elif negative > positive * 1.15 or probs.get("sadness", 0.0) > 0.6 or probs.get("anger", 0.0) > 0.6:
        tone = "Negative"
    else:
        tone = "Neutral"

    confidence = round(min(99.9, max(10.0, 80 + (scores[dominant_key] * 0.15))), 1)
    valence = round((positive - negative) / 100, 2)
    arousal = round((scores["ang"] + scores["fea"] + scores["sur"]) / 100, 2)

    return {
        "text": text,
        "scores": scores,
        "dominant": {
            "key": dominant_key,
            "label": dominant["label"],
            "emoji": dominant["emoji"],
            "color": dominant["color"],
        },
        "confidence": confidence,
        "valence": valence,
        "arousal": arousal,
        "tone": tone,
    }


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text supplied"}), 400

        _load_model()
        inputs = TOKENIZER(text, return_tensors="np", max_length=128, padding="max_length", truncation=True)
        prediction_fn = MODEL.signatures["serving_default"]
        outputs = prediction_fn(
            input_ids=tf.constant(inputs["input_ids"].astype(np.int32)),
            attention_mask=tf.constant(inputs["attention_mask"].astype(np.int32)),
        )
        logits = outputs["output_1"].numpy()[0]
        probs = _sigmoid(logits)
        raw_probs = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        payload = build_emotion_payload(raw_probs, text)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
