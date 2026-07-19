import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"
chatbot_tokenizer_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\chatbot_model"

tokenizer = BertTokenizer.from_pretrained(chatbot_tokenizer_path, local_files_only=True)
model = tf.saved_model.load(model_path)
prediction_fn = model.signatures["serving_default"]

sentences = {
    "love": "I love you so much!",
    "hate": "I hate you, you are terrible and I am so angry!",
    "scared": "I am so scared, terrified and nervous!",
    "surprised": "Oh wow, this is incredibly surprising! I cannot believe it!",
    "sad": "I am so sad, depressed, and crying."
}

# The 28 labels of GoEmotions
go_emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

for name, sentence in sentences.items():
    inputs = tokenizer(sentence, return_tensors="np", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)
    
    outputs = prediction_fn(input_ids=tf.constant(input_ids), attention_mask=tf.constant(attention_mask))
    logits = outputs["output_1"].numpy()[0]
    
    print(f"\nSentence: '{sentence}' ({name.upper()})")
    
    # Softmax
    probs = softmax(logits)
    
    sorted_indices = np.argsort(probs)[::-1]
    print("  Top emotions (Softmax):")
    for idx in sorted_indices[:5]:
        print(f"    - {go_emotions[idx]}: {probs[idx]:.4f} (logit: {logits[idx]:.4f})")
