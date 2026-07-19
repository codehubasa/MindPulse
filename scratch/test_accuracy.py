import json
import re
import numpy as np
import tensorflow as tf

def clean_text(text):
    text = text.lower()
    # Keras tokenizer filters usually do not strip single quotes in all configurations
    # let's try standard split after stripping most punctuation
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~\t\n]', '', text)
    return text.split()

# Load model
model = tf.keras.models.load_model('Sarcasm_Model_fixed.keras')

# Load tokenizer
with open('sarcasm_tokenizer.json', 'r', encoding='utf-8') as f:
    word_index = json.load(f)

# Load dataset
with open('sarcasm_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

oov_index = 1
max_len = 250
vocab_limit = 10000

correct_pre = 0
correct_post = 0
total = min(100, len(dataset))

for entry in dataset[:total]:
    sentence = entry['headline']
    label = entry['is_sarcastic']
    
    words = clean_text(sentence)
    seq = []
    for w in words:
        val = word_index.get(w, oov_index)
        if val >= vocab_limit:
            val = oov_index
        seq.append(val)
        
    # Test pre padding
    padded_pre = [0] * (max_len - len(seq)) + seq if len(seq) < max_len else seq[:max_len]
    pred_pre = model.predict(np.array([padded_pre]), verbose=0)[0][0]
    is_sarc_pre = int(pred_pre >= 0.5)
    if is_sarc_pre == label:
        correct_pre += 1
        
    # Test post padding
    padded_post = seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
    pred_post = model.predict(np.array([padded_post]), verbose=0)[0][0]
    is_sarc_post = int(pred_post >= 0.5)
    if is_sarc_post == label:
        correct_post += 1
        
    print(f"Text: {sentence[:50]}... | Label: {label} | Pred Pre: {pred_pre:.4f} | Pred Post: {pred_post:.4f}")

print(f"\nAccuracy (Pre-padding): {correct_pre / total * 100:.2f}%")
print(f"Accuracy (Post-padding): {correct_post / total * 100:.2f}%")
