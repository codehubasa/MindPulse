import tensorflow as tf
import os

model_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\emotional_detection_model\bert_classifier_tensorflow_saved_model"

try:
    print("Loading model using tf.saved_model.load...")
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully!")
    
    # Print signature keys
    signatures = list(model.signatures.keys())
    print("\nSignatures:", signatures)
    
    for sig_name in signatures:
        sig = model.signatures[sig_name]
        print(f"\nSignature '{sig_name}':")
        print("  Inputs:")
        for input_key, input_tensor in sig.structured_input_signature[1].items():
            print(f"    - Key: '{input_key}', Shape: {input_tensor.shape}, Dtype: {input_tensor.dtype}")
        print("  Outputs:")
        for output_key, output_tensor in sig.structured_outputs.items():
            print(f"    - Key: '{output_key}', Shape: {output_tensor.shape}, Dtype: {output_tensor.dtype}")

    # Let's run a test inference if there is a 'serving_default' signature
    if 'serving_default' in signatures:
        sig = model.signatures['serving_default']
        input_keys = list(sig.structured_input_signature[1].keys())
        print("\nInput keys for serving_default:", input_keys)
        
        # Prepare a test batch of text.
        # BERT models often take either string input or (input_word_ids, input_mask, input_type_ids)
        # Let's check the type of input.
        first_input_key = input_keys[0]
        first_input_tensor = sig.structured_input_signature[1][first_input_key]
        
        if first_input_tensor.dtype == tf.string:
            print("Inputs are string tensors. Testing with sample string...")
            test_input = {first_input_key: tf.constant(["I am extremely happy and excited today!", "I am so sad and disappointed"])}
            res = sig(**test_input)
            print("Inference results:")
            for k, v in res.items():
                print(f"  {k}: {v.numpy()}")
        else:
            print("Inputs are numeric tensors (tokenized ids). Needs a separate tokenizer.")
            
except Exception as e:
    import traceback
    traceback.print_exc()
