from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can use "t5-base" or "t5-large" for more capacity
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the context
context = "number of tools: 2, tools: Scissors, Hook, phase: operation, target organ: Stomach, action verb: Cut"

# Tokenize the context
inputs = tokenizer("summarize: " + context, return_tensors="pt", max_length=512, truncation=True)

# Generate the caption
outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1, early_stopping=True)

# Decode and print the caption
caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Caption:", caption)
