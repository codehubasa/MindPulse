import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"
chatbot_tokenizer_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\chatbot_model"

tokenizer = BertTokenizer.from_pretrained(chatbot_tokenizer_path, local_files_only=True)
model = tf.saved_model.load(model_path)

sentences = [
    "I love you so much!",
    "I hate you, you are terrible.",
    "The sky is blue.",
    ""
]

prediction_fn = model.signatures["serving_default"]

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="np", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"].astype(np.int32)
    attention_mask = inputs["attention_mask"].astype(np.int32)
    
    outputs = prediction_fn(input_ids=tf.constant(input_ids), attention_mask=tf.constant(attention_mask))
    logits = outputs["output_1"].numpy()[0]
    
    print(f"\nSentence: '{sentence}'")
    print("Logits (first 10):", logits[:10])
    print("Logits min/max/mean:", logits.min(), logits.max(), logits.mean())
