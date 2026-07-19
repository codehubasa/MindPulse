import os
import re
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Frontend ----------------
@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

# ---------------- Sentiment ----------------
def analyze_sentiment(text):
    normalized = (text or '').strip()
    if not normalized:
        return {'polarity': 0.0, 'subjectivity': 0.3, 'confidence': 70.0,
                'verdict': 'Neutral', 'tone': 'neutral',
                'scores': {'pos': 50, 'neg': 20, 'neu': 20, 'mix': 10}}
    tokens = re.findall(r"[a-zA-Z']+", normalized.lower())
    positive_words = {'love','great','amazing','awesome','good','excellent','best','beautiful',
        'fantastic','wonderful','happy','perfect','smooth','intuitive','inspiring','solid',
        'improved','improvement','glad','delightful','motivated','gorgeous','impressive'}
    negative_words = {'hate','terrible','awful','bad','worst','horrible','poor','disappointing',
        'useless','frustrated','slow','sluggish','fail','failed','broken','disappointed','drain',
        'rough','badly','dislike','negative'}
    negation_words = {'not','never','no','cannot','cant',"won't",'dont',"don't",'hardly','barely'}
    score = 0.0
    for index, token in enumerate(tokens):
        is_negated = index > 0 and tokens[index - 1] in negation_words
        if token in positive_words:
            score += -0.25 if is_negated else 0.2
        elif token in negative_words:
            score += 0.25 if is_negated else -0.25
    if any(t in negation_words for t in tokens):
        score *= 0.85
    polarity = max(-1.0, min(1.0, round(score, 2)))
    subjectivity = round(min(1.0, max(0.1, 0.2 + abs(polarity) * 0.5)), 2)
    if polarity > 0.35: verdict, tone = 'Very Positive', 'positive'
    elif polarity > 0.1: verdict, tone = 'Positive', 'positive'
    elif polarity < -0.35: verdict, tone = 'Very Negative', 'negative'
    elif polarity < -0.1: verdict, tone = 'Negative', 'negative'
    else: verdict, tone = 'Neutral', 'neutral'
    pos = max(5, int(round((polarity + 1) / 2 * 100)))
    neg = max(5, int(round((1 - abs(polarity)) * 100)))
    neu = max(5, int(round((1 - abs(polarity)) * 40)))
    mix = max(5, 100 - pos - neg - neu)
    return {'polarity': polarity, 'subjectivity': subjectivity,
            'confidence': round(min(99.9, 70 + abs(polarity) * 20), 1),
            'verdict': verdict, 'tone': tone,
            'scores': {'pos': max(1,min(99,pos)), 'neg': max(1,min(99,neg)),
                       'neu': max(1,min(99,neu)), 'mix': max(1,min(99,mix))}}

@app.route('/api/sentiment/predict', methods=['POST', 'OPTIONS'])
def sentiment_predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text supplied'}), 400
    return jsonify(analyze_sentiment(text))

# ---------------- Emotion (BERT, lazy-loaded) ----------------
EMOTION_LABELS = ["admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement",
    "fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"]
EMOTION_MAP = {"joy":{"label":"Joy","emoji":"😄","color":"#fbbf24"},"sad":{"label":"Sadness","emoji":"😢","color":"#60a5fa"},
    "ang":{"label":"Anger","emoji":"😡","color":"#f43f5e"},"fea":{"label":"Fear","emoji":"😰","color":"#a78bfa"},
    "sur":{"label":"Surprise","emoji":"😲","color":"#2dd4bf"},"dis":{"label":"Disgust","emoji":"🤢","color":"#fb923c"}}
EMOTION_MODEL = None
EMOTION_TOKENIZER = None

def _load_emotion_model():
    global EMOTION_MODEL, EMOTION_TOKENIZER
    if EMOTION_MODEL is None:
        import tensorflow as tf
        from transformers import BertTokenizer
        EMOTION_TOKENIZER = BertTokenizer.from_pretrained(
            os.path.join(BASE_DIR, "chatbot_model"), local_files_only=True)
        EMOTION_MODEL = tf.saved_model.load(
            os.path.join(BASE_DIR, "emotional_detection_model", "bert_classifier_tensorflow_saved_model"))

def _sigmoid(v):
    return 1 / (1 + np.exp(-v))

def build_emotion_payload(raw_probs, text):
    import tensorflow as tf
    probs = dict(raw_probs)
    scores = {"joy":0.0,"sad":0.0,"ang":0.0,"fea":0.0,"sur":0.0,"dis":0.0}
    groups = {"joy":{"admiration","amusement","approval","caring","curiosity","desire","excitement",
                "gratitude","joy","love","optimism","pride","relief","realization"},
              "sad":{"sadness","disappointment","grief","remorse","embarrassment"},
              "ang":{"anger","annoyance","disapproval"}, "fea":{"fear","nervousness"},
              "sur":{"surprise","curiosity","realization"}, "dis":{"disgust"}}
    for label, value in probs.items():
        prob = float(value)
        for key, label_set in groups.items():
            if label in label_set:
                scores[key] += prob
    if probs.get("neutral", 0.0) > 0.5:
        scores = {k: v * 0.5 for k, v in scores.items()}
    total = sum(scores.values())
    scores = {k: 0.0 for k in scores} if total <= 0 else {k: round((v/total)*100,1) for k,v in scores.items()}
    dominant_key = max(scores, key=scores.get)
    dominant = EMOTION_MAP[dominant_key]
    positive = scores["joy"] + scores["sur"]
    negative = scores["sad"] + scores["ang"] + scores["fea"] + scores["dis"]
    if positive > negative*1.15 or probs.get("joy",0.0) > 0.6 or probs.get("excitement",0.0) > 0.55:
        tone = "Positive"
    elif negative > positive*1.15 or probs.get("sadness",0.0) > 0.6 or probs.get("anger",0.0) > 0.6:
        tone = "Negative"
    else:
        tone = "Neutral"
    confidence = round(min(99.9, max(10.0, 80 + scores[dominant_key]*0.15)), 1)
    valence = round((positive - negative) / 100, 2)
    arousal = round((scores["ang"]+scores["fea"]+scores["sur"]) / 100, 2)
    return {"text": text, "scores": scores,
            "dominant": {"key": dominant_key, "label": dominant["label"], "emoji": dominant["emoji"], "color": dominant["color"]},
            "confidence": confidence, "valence": valence, "arousal": arousal, "tone": tone}

@app.route('/api/emotion/predict', methods=['POST', 'OPTIONS'])
def emotion_predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        import tensorflow as tf
        data = request.get_json(silent=True) or {}
        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'error': 'No text supplied'}), 400
        _load_emotion_model()
        inputs = EMOTION_TOKENIZER(text, return_tensors="np", max_length=128, padding="max_length", truncation=True)
        prediction_fn = EMOTION_MODEL.signatures["serving_default"]
        outputs = prediction_fn(
            input_ids=tf.constant(inputs["input_ids"].astype(np.int32)),
            attention_mask=tf.constant(inputs["attention_mask"].astype(np.int32)))
        logits = outputs["output_1"].numpy()[0]
        probs = _sigmoid(logits)
        raw_probs = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        return jsonify(build_emotion_payload(raw_probs, text))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

# ---------------- Sarcasm (Keras, lazy-loaded) ----------------
SARCASM_MODEL = None
SARCASM_WORD_INDEX = None
OOV_INDEX = 1
MAX_LEN = 250
VOCAB_LIMIT = 10000

def _load_sarcasm_model():
    global SARCASM_MODEL, SARCASM_WORD_INDEX
    if SARCASM_MODEL is None:
        import tensorflow as tf
        SARCASM_MODEL = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "Sarcasm_Model_fixed.keras"), compile=False)
        with open(os.path.join(BASE_DIR, "sarcasm_tokenizer.json"), "r", encoding="utf-8") as f:
            SARCASM_WORD_INDEX = json.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~\t\n]', '', text)
    return text.split()

def build_sarcasm_payload(text, raw_pred=0.5):
    words = clean_text(text)
    lowered = " ".join(words)
    pos_words = {"love","like","great","good","brilliant","groundbreaking","perfectly","happy",
        "magical","best","wonderful","amazing","beautiful","excellent","nice","success","heroically","glad"}
    neg_words = {"break","dead","sues","secret","minority","shoppers","fear","sinking","bomb",
        "detonates","smear","frightened","terror","crisis","outrage","abuse","chokeholds","cries",
        "accident","broken","bad","terrible","worst","hate","fail","failed","failure","ruin","destroy","mess","ugly"}
