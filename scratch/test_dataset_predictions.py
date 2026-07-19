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

test_cases = [
    # Non-sarcastic (is_sarcastic: 0)
    ("former versace store clerk sues over secret 'black code' for minority shoppers", 0),
    ("the 'roseanne' revival catches up to our thorny political mood, for better and worse", 0),
    ("j.k. rowling wishes snape happy birthday in the most magical way", 0),
    # Sarcastic (is_sarcastic: 1)
    ("mom starting to fear son's web series closest thing she will have to grandchild", 1),
    ("boehner just wants wife to listen, not come up with alternative debt-reduction ideas", 1),
    ("top snake handler leaves sinking huckabee campaign", 1)
]

for padding in ['pre', 'post']:
    print(f"\n--- Testing with {padding.upper()} padding ---")
    for sentence, label in test_cases:
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
        print(f"Label: {label} | Pred Prob: {pred:.4f} | Sentence: '{sentence}'")
