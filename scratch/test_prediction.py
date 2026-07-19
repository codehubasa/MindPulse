import json
import re
import numpy as np
import tensorflow as tf

def clean_text(text):
    text = text.lower()
    # Remove punctuation similar to Keras Tokenizer default filters
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~\t\n]', '', text)
    return text.split()

# Load model
model = tf.keras.models.load_model('Sarcasm_Model_fixed.keras')

# Load tokenizer
with open('sarcasm_tokenizer.json', 'r', encoding='utf-8') as f:
    word_index = json.load(f)

oov_token = "<OOV>"
oov_index = word_index.get(oov_token, 1)
max_len = 250
vocab_limit = 10000

sentences = [
    "Oh sure, because that update totally didn't break my entire workflow.",
    "This is a normal sentence with no sarcasm.",
    "Oh brilliant, another redesign of a perfectly working button.",
    "Wow, what a groundbreaking idea. I've never seen anyone do that before.",
    "I love coding and building web applications."
]

for padding in ['pre', 'post']:
    print(f"\n--- Testing with {padding.upper()} padding ---")
    for sentence in sentences:
        words = clean_text(sentence)
        seq = []
        for w in words:
            val = word_index.get(w, oov_index)
            if val >= vocab_limit:
                val = oov_index
            seq.append(val)
        
        # Pad / truncate sequence
        if len(seq) < max_len:
            if padding == 'pre':
                padded = [0] * (max_len - len(seq)) + seq
            else:
                padded = seq + [0] * (max_len - len(seq))
        else:
            padded = seq[:max_len]
            
        inp = np.array([padded])
        pred = model.predict(inp, verbose=0)[0][0]
        print(f"Sentence: '{sentence}'")
        print(f"  Prediction Prob: {pred:.4f} (Sarcastic: {pred >= 0.5})")
