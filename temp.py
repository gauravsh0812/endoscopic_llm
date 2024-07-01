from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nElaborate the surgical image? ASSISTANT:"
image = Image.open("/data/shared/CholecT50/CholecT50/videos/VID01/001101.png")

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=150)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)