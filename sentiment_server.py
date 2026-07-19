import os
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True})


def analyze_sentiment(text):
    normalized = (text or '').strip()
    if not normalized:
        return {
            'polarity': 0.0,
            'subjectivity': 0.3,
            'confidence': 70.0,
            'verdict': 'Neutral',
            'tone': 'neutral',
            'scores': {'pos': 50, 'neg': 20, 'neu': 20, 'mix': 10}
        }

    tokens = re.findall(r"[a-zA-Z']+", normalized.lower())
    positive_words = {
        'love', 'great', 'amazing', 'awesome', 'good', 'excellent', 'best', 'beautiful',
        'fantastic', 'wonderful', 'happy', 'perfect', 'smooth', 'intuitive', 'inspiring',
        'solid', 'improved', 'improvement', 'glad', 'delightful', 'motivated', 'gorgeous',
        'impressive'
    }
    negative_words = {
        'hate', 'terrible', 'awful', 'bad', 'worst', 'horrible', 'poor', 'disappointing',
        'useless', 'frustrated', 'slow', 'sluggish', 'fail', 'failed', 'broken',
        'disappointed', 'drain', 'rough', 'badly', 'dislike', 'negative'
    }
    negation_words = {'not', 'never', 'no', 'cannot', 'cant', "won't", 'dont', "don't", 'hardly', 'barely'}

    score = 0.0
    for index, token in enumerate(tokens):
        is_negated = index > 0 and tokens[index - 1] in negation_words
        if token in positive_words:
            score += -0.25 if is_negated else 0.2
        elif token in negative_words:
            score += 0.25 if is_negated else -0.25

    if any(token in negation_words for token in tokens):
        score *= 0.85

    polarity = max(-1.0, min(1.0, round(score, 2)))
    subjectivity = round(min(1.0, max(0.1, 0.2 + abs(polarity) * 0.5)), 2)

    if polarity > 0.35:
        verdict = 'Very Positive'
        tone = 'positive'
    elif polarity > 0.1:
        verdict = 'Positive'
        tone = 'positive'
    elif polarity < -0.35:
        verdict = 'Very Negative'
        tone = 'negative'
    elif polarity < -0.1:
        verdict = 'Negative'
        tone = 'negative'
    else:
        verdict = 'Neutral'
        tone = 'neutral'

    pos = max(5, int(round((polarity + 1) / 2 * 100)))
    neg = max(5, int(round((1 - abs(polarity)) * 100)))
    neu = max(5, int(round((1 - abs(polarity)) * 40)))
    mix = 100 - pos - neg - neu
    mix = max(5, mix)

    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'confidence': round(min(99.9, 70 + abs(polarity) * 20), 1),
        'verdict': verdict,
        'tone': tone,
        'scores': {
            'pos': max(1, min(99, pos)),
            'neg': max(1, min(99, neg)),
            'neu': max(1, min(99, neu)),
            'mix': max(1, min(99, mix))
        }
    }


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json(silent=True) or {}
        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'error': 'No text supplied'}), 400
        return jsonify(analyze_sentiment(text))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)), debug=False)
