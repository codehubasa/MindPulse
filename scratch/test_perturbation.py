import json
import re
import numpy as np
import tensorflow as tf

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~\t\n]', '', text)
    return text.split()

# Load model
model = tf.keras.models.load_model('Sarcasm_Model_fixed.keras')

# Load tokenizer
with open('sarcasm_tokenizer.json', 'r', encoding='utf-8') as f:
    word_index = json.load(f)

oov_index = 1
max_len = 250
vocab_limit = 10000

sentence = "Oh sure, because that update totally didn't break my entire workflow."
words = clean_text(sentence)

def get_pred(word_list):
    seq = [word_index.get(w, oov_index) for w in word_list]
    seq = [v if v < vocab_limit else oov_index for v in seq]
    padded = [0] * (max_len - len(seq)) + seq if len(seq) < max_len else seq[:max_len]
    return model.predict(np.array([padded]), verbose=0)[0][0]

base_pred = get_pred(words)
print(f"Base prediction: {base_pred:.6f}")

for i, w in enumerate(words):
    # Mask word i
    masked_words = words[:i] + ["<OOV>"] + words[i+1:]
    masked_pred = get_pred(masked_words)
    diff = base_pred - masked_pred
    print(f"Word: '{w}' | Masked Pred: {masked_pred:.6f} | Diff: {diff:.6f}")
