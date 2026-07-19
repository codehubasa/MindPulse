import tensorflow as tf

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"

try:
    print("Loading model...")
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully!")
    
    print("\nVariable names and shapes (all):")
    for i, v in enumerate(model.variables):
        print(f"  {i:3d}: {v.name} (shape: {v.shape})")
        
except Exception as e:
    import traceback
    traceback.print_exc()
