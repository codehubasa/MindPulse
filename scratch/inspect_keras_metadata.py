import tensorflow as tf

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"

try:
    # Load model as Keras layer
    print("Loading model as TFSMLayer...")
    inputs = {
        "input_ids": tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids"),
        "attention_mask": tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    }
    
    # We can use TFSMLayer in TF 2.16+ / Keras 3
    from tensorflow.keras.layers import TFSMLayer
    sm_layer = TFSMLayer(model_path, call_endpoint="serving_default")
    outputs = sm_layer(inputs)
    
    print("TFSMLayer outputs:", outputs)
    
    # Let's inspect the model variable details or configuration
    model = tf.saved_model.load(model_path)
    print("\nModel variables count:", len(model.variables))
    
    # Let's print some variable names to see the layer types (e.g. BERT layers vs dense layers)
    print("\nVariable names (first 30):")
    for v in model.variables[:30]:
        print(f"  - {v.name} (shape: {v.shape})")
        
except Exception as e:
    import traceback
    traceback.print_exc()
