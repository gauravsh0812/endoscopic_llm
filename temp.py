from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Base caption to rewrite
base_caption = "During the operation phase, two tools, Scissors and Hook, are being used to cut the Stomach."

# Tokenize the input text
input_ids = tokenizer.encode(base_caption, return_tensors="pt")

# Generate rewritten sentence
outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)

# Decode and print the rewritten sentence
rewritten_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Rewritten Sentence:", rewritten_sentence)
