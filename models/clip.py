from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModel, CLIPVisionConfig

class ClipVisionEncoder(nn.Module):
    
    def __init__(self, finetune=False, config=None):
        super(ClipVisionEncoder, self).__init__()
        # if finetune:
        #     configuration = CLIPVisionConfig(**config)
        #     self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        #     self.model = CLIPVisionModel(configuration)
        # else:
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_paths, device):

        _hid, _pool = list(), list()
        for image_path in image_paths:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = outputs.pooler_output  # pooled classes states

            print("last_hidden_state shape: ", last_hidden_state.shape)

            _hid.append(last_hidden_state.squeeze(0))
            _pool.append(pooled_output.squeeze(0))

        # hidden: (B, L, 768)
        # pooled: (B, 768)
        return torch.stack(_hid).to(device), torch.stack(_pool).to(device)
    
cve = ClipVisionEncoder()
cve(["/data/gauravs/combine_data/images/1398.png"], "cuda")