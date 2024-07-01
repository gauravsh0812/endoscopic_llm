from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")

_txt = "The number of tools are 2 that are grasper and hook. The phase is carlot-triangle-dissection phase, and the target organ is gallbladder. The action verbs are grasp and dissect. Now based on these details, elaborate this surgiccal scene."

prompt = f"USER: <image>\n{_txt} ASSISTANT:"
image = Image.open("/data/shared/CholecT50/CholecT50/videos/VID10/000377.png")

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=300)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)