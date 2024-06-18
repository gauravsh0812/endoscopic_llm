from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
_path = "10.png"
image = Image.open(_path)

inputs = processor(text=prompt, images=image, return_tensors="pt")
print(inputs)
# Generate
# generate_ids = model.generate(**inputs, max_length=30)
# processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]