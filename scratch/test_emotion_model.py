import tensorflow as tf
import os
import numpy as np

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"

print("Checking if model exists:", os.path.exists(model_path))

try:
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nModel inputs:", model.inputs)
    print("Model outputs:", model.outputs)
    
    # Try a mock inference if we can find the input shape/type
    if hasattr(model, 'signatures'):
        print("\nSignatures:", list(model.signatures.keys()))
        for sig_name in model.signatures:
            sig = model.signatures[sig_name]
            print(f"Signature '{sig_name}':")
            print("  Inputs:", sig.structured_input_signature)
            print("  Outputs:", sig.structured_outputs)
            
except Exception as e:
    print("Error:", e)
