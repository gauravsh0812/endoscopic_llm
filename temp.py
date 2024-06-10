from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModel, 
    RobertaModel,
)
from PIL import Image
import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    
    def __init__(self,):
        super(ImageEncoder, self).__init__()
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_paths):

        for image_path in image_paths:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to("cuda:0")
            outputs = self.model(**inputs)
            print(outputs)

im = ImageEncoder().to("cuda:0")
im(["/data/gauravs/surgicalGPT/our_dataset/images/1130.png"])