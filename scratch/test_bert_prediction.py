import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully!")

# Ensure transformers and tokenizers are installed
install_and_import("transformers")
install_and_import("tokenizers")

import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"

print("Loading model...")
model = tf.saved_model.load(model_path)
print("Model loaded successfully!")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Tokenizer loaded successfully!")

# Test sentences for emotions
sentences = [
    "I am so incredibly happy and grateful for this amazing day!",
    "This is so annoying and frustrating, it makes me so angry.",
    "I am really scared and nervous about the presentation tomorrow.",
    "I'm feeling so sad and disappointed with the results.",
    "Oh wow, that is so surprising! I did not expect that at all.",
    "I don't feel anything in particular, it's just a normal day."
]

# The 28 labels of GoEmotions
go_emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

for sentence in sentences:
    print(f"\nSentence: '{sentence}'")
    # Tokenize input using BERT tokenizer
    inputs = tokenizer(sentence, return_tensors="np", max_length=128, padding="max_length", truncation=True)
    
    # Extract input_ids and attention_mask
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)
    
    # Predict
    prediction_fn = model.signatures["serving_default"]
    outputs = prediction_fn(input_ids=tf.constant(input_ids), attention_mask=tf.constant(attention_mask))
    
    # Output key is 'output_1'
    logits = outputs["output_1"].numpy()[0]
    
    # Print top 3 predicted emotions
    # If the output is logits, we can apply sigmoid since GoEmotions is a multi-label classification problem
    # probabilities = 1 / (1 + np.exp(-logits))
    probs = 1 / (1 + np.exp(-logits))
    
    top_indices = np.argsort(probs)[::-1][:3]
    print("  Top emotions:")
    for idx in top_indices:
        print(f"    - {go_emotions[idx]}: {probs[idx]:.4f} (logit: {logits[idx]:.4f})")
