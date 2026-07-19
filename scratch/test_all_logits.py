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

for name, sentence in sentences.items():
    inputs = tokenizer(sentence, return_tensors="np", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)
    
    outputs = prediction_fn(input_ids=tf.constant(input_ids), attention_mask=tf.constant(attention_mask))
    logits = outputs["output_1"].numpy()[0]
    
    print(f"\n======================================")
    print(f"Sentence: '{sentence}' ({name.upper()})")
    print(f"======================================")
    
    # Let's print index and logit/prob
    probs = 1 / (1 + np.exp(-logits))
    sorted_indices = np.argsort(logits)[::-1]
    
    print("Top 10 raw indices:")
    for idx in sorted_indices[:10]:
        print(f"  Index {idx:2d} -> logit: {logits[idx]:.4f}, prob: {probs[idx]:.4f}")
