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

sentences = [
    "former versace store clerk sues over secret 'black code' for minority shoppers",
    "mom starting to fear son's web series closest thing she will have to grandchild",
    "This is completely random words like apple banana telephone",
]

for sentence in sentences:
    words = clean_text(sentence)
    seq = [word_index.get(w, oov_index) for w in words]
    seq = [v if v < vocab_limit else oov_index for v in seq]
    padded = [0] * (max_len - len(seq)) + seq if len(seq) < max_len else seq[:max_len]
    
    x = np.array([padded])
    
    # Layer-by-layer forward pass
    e_val = model.layers[0](x).numpy()
    l_val = model.layers[1](e_val).numpy()
    d_val = model.layers[2](l_val).numpy()
    pred = model.predict(x, verbose=0)[0][0]
    
    print(f"\nSentence: '{sentence}'")
    print(f"  Padded sequence: {padded[:20]} ... {padded[-20:]}")
    print(f"  Embedding mean: {np.mean(e_val):.6f}, std: {np.std(e_val):.6f}")
    print(f"  LSTM mean: {np.mean(l_val):.6f}, std: {np.std(l_val):.6f}")
    print(f"  LSTM min: {np.min(l_val):.6f}, max: {np.max(l_val):.6f}")
    print(f"  Dense (Pre-activation/logits) output: {d_val[0]}")
    print(f"  Final Pred (with Sigmoid): {pred:.6f}")
