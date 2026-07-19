import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"
chatbot_tokenizer_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\chatbot_model"

tokenizer = BertTokenizer.from_pretrained(chatbot_tokenizer_path, local_files_only=True)
model = tf.saved_model.load(model_path)
prediction_fn = model.signatures["serving_default"]

love_sent = "I love you so much!"
hate_sent = "I hate you, you are terrible and I am so angry!"

# The 28 labels of GoEmotions
go_emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Predict love
inputs_love = tokenizer(love_sent, return_tensors="np", max_length=128, padding="max_length", truncation=True)
out_love = prediction_fn(input_ids=tf.constant(inputs_love["input_ids"].astype(np.int32)), 
                         attention_mask=tf.constant(inputs_love["attention_mask"].astype(np.int32)))
logits_love = out_love["output_1"].numpy()[0]

# Predict hate
inputs_hate = tokenizer(hate_sent, return_tensors="np", max_length=128, padding="max_length", truncation=True)
out_hate = prediction_fn(input_ids=tf.constant(inputs_hate["input_ids"].astype(np.int32)), 
                         attention_mask=tf.constant(inputs_hate["attention_mask"].astype(np.int32)))
logits_hate = out_hate["output_1"].numpy()[0]

diff = logits_hate - logits_love

print("Emotion differences (Hate - Love logits):")
sorted_diff_idx = np.argsort(diff)[::-1]

print("\nHigher in HATE:")
for idx in sorted_diff_idx[:14]:
    print(f"  - {go_emotions[idx]:15s}: hate={logits_hate[idx]:.4f}, love={logits_love[idx]:.4f}, diff={diff[idx]:+.4f}")

print("\nHigher in LOVE:")
for idx in sorted_diff_idx[14:]:
    print(f"  - {go_emotions[idx]:15s}: hate={logits_hate[idx]:.4f}, love={logits_love[idx]:.4f}, diff={diff[idx]:+.4f}")
