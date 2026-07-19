from transformers import BertTokenizer

chatbot_tokenizer_path = r"c:\Users\amitb\OneDrive\Desktop\MindPulse\chatbot_model"
tokenizer = BertTokenizer.from_pretrained(chatbot_tokenizer_path, local_files_only=True)

test_sentences = [
    "I love you so much!",
    "I hate you, you are terrible and I am so angry!",
    "I am so scared, terrified and nervous!",
    "Oh wow, this is incredibly surprising! I cannot believe it!",
    "I am so sad, depressed, and crying."
]

for sentence in test_sentences:
    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\nSentence: '{sentence}'")
    print("Tokens:", tokens)
    print("IDs:", ids)
