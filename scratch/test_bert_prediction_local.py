import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"
chatbot_tokenizer_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\chatbot_model"

print("Loading local tokenizer from chatbot_model...")
try:
    tokenizer = BertTokenizer.from_pretrained(chatbot_tokenizer_path, local_files_only=True)
    print("Local tokenizer loaded successfully!")
except Exception as e:
    print("Error loading local tokenizer:", e)
    # Fallback to standard bert-base-uncased if local fails
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("bert-base-uncased tokenizer loaded from cache/web.")
    except Exception as ex:
        print("Error loading standard tokenizer:", ex)
        raise

print("Loading SavedModel...")
model = tf.saved_model.load(model_path)
print("Model loaded successfully!")

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
    
    # Sigmoid for logits
    probs = 1 / (1 + np.exp(-logits))
    
    top_indices = np.argsort(probs)[::-1][:3]
    print("  Top emotions:")
    for idx in top_indices:
        print(f"    - {go_emotions[idx]}: {probs[idx]:.4f} (logit: {logits[idx]:.4f})")
