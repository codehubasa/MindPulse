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
vocab_limit = 10000

test_cases = [
    ("former versace store clerk sues over secret 'black code' for minority shoppers", 0),
    ("the 'roseanne' revival catches up to our thorny political mood, for better and worse", 0),
    ("mom starting to fear son's web series closest thing she will have to grandchild", 1),
    ("boehner just wants wife to listen, not come up with alternative debt-reduction ideas", 1),
    ("Oh sure, because that update totally didn't break my entire workflow.", 1),
    ("This is a normal sentence with no sarcasm.", 0)
]

for sentence, label in test_cases:
    words = clean_text(sentence)
    seq = [word_index.get(w, oov_index) for w in words]
    seq = [v if v < vocab_limit else oov_index for v in seq]
    
    # 1. No padding
    inp_no_pad = np.array([seq])
    pred_no_pad = model.predict(inp_no_pad, verbose=0)[0][0]
    
    # 2. Pre padding to 250
    padded_pre = [0] * (250 - len(seq)) + seq
    pred_pre = model.predict(np.array([padded_pre]), verbose=0)[0][0]
    
    print(f"Label: {label} | NoPad: {pred_no_pad:.4f} | PrePad250: {pred_pre:.4f} | Sentence: '{sentence}'")
