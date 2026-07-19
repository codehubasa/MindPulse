import sys
try:
    import transformers
    print("transformers version:", transformers.__version__)
except ImportError:
    print("transformers is not installed.")

try:
    import tensorflow_text
    print("tensorflow_text is installed.")
except ImportError:
    print("tensorflow_text is not installed.")

print("Python version:", sys.version)
