import torch 
import torch.nn as nn
from PIL import Image
import yaml
from box import Box

from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModel, 
    RobertaModel,
    LlavaForConditionalGeneration,
    LlavaConfig, CLIPVisionConfig, LlamaConfig
)

with open("config/config_surgpt.yaml") as f:
    cfg = Box(yaml.safe_load(f))

class ClipVisionEncoder(nn.Module):
    
    def __init__(self,):
        super(ClipVisionEncoder, self).__init__()
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_paths, device):

        _hid = list()
        for image_path in image_paths:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        
            _hid.append(last_hidden_state.squeeze(0))
        
        # hidden: (B, L, 768)
        return torch.stack(_hid).to(device)#, torch.stack(_pool).to(device)

class RobertaEncoder(nn.Module):

    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.model = RobertaModel.from_pretrained("FacebookAI/roberta-base")        

    def forward(self, ids, attns):
        # shape of ids and attns: (B, max_len)
        outputs = self.model(input_ids=ids,
                             attention_mask=attns)
        last_hidden_states = outputs.last_hidden_state # (B, max_len, 768)
        return last_hidden_states

class Llava(nn.Module):
    def __init__(self,):
        super(Llava, self).__init__()
        vision_config = CLIPVisionConfig()
        text_config = LlamaConfig(hidden_size=768)
        configuration = LlavaConfig(vision_config, text_config)
        # self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = LlavaForConditionalGeneration(configuration)
    
    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds,) # since we don't use any attn mask, no need to provide that.
        return outputs

class Projector(nn.Module):

    def __init__(self, features, max_len, num_classes):
        super(Projector, self).__init__()

        self.final_lin1 = nn.Sequential(
            nn.Linear(32000, features[0]),
            nn.BatchNorm1d(features[0]),
            nn.GELU(),

            nn.Linear(features[0], features[1]),
            nn.BatchNorm1d(features[1]),
            nn.GELU(),

            nn.Linear(features[1], features[2]),
            nn.BatchNorm1d(features[2]),
            nn.GELU(),

            nn.Linear(features[2], features[-1]),
            nn.BatchNorm1d(features[-1]),
            nn.GELU(),    
        )

        self.final_lin2 = nn.Sequential(
            nn.Linear(56, max_len),
            nn.BatchNorm1d(features[0]),
            nn.GELU(),
        )
        
        self.attn = Self_Attention(features[-1])
        self.final_lin = nn.Linear(features[-1], num_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.gelu = nn.GELU()

    def forward(self,x):
        x = self.final_lin1(x)  # (B, 56, 64)
        x = self.final_lin2(x.permute(0,2,1)).permute(0,2,1)  # (B, max, 64)
        x = self.attn(x)
        x = self.pool(x.permute(0,2,1)).permute(0,2,1)  # (B, max_len=1, 64)
        x = torch.flatten(x, -2,-1)   # (B,max_len*64) >> (B, 64)
        x = self.gelu(self.final_lin(x))  # (B, num_classes)

        return x   # (B,num_classes)

class Self_Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Self_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)    # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (batch_size, seq_length, seq_length)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)  # (batch_size, seq_length, seq_length)
        
        # Compute the weighted sum of values
        attention_output = torch.bmm(attention_weights, V)  # (batch_size, seq_length, embed_dim)
        
        return attention_output       

class Endoscopic_model(nn.Module):

    def __init__(self, max_len, ans_vocab):

        super(Endoscopic_model, self).__init__()
        self.clipenc = ClipVisionEncoder()
        self.robenc = RobertaEncoder()
        self.llava = Llava()
        
        self.projector = Projector(
                                cfg.training.adaptor.features,
                                len(ans_vocab),
                            )

        for param in self.clipenc.parameters():
            param.requires_grad = False

        for param in self.robenc.parameters():
            param.requires_grad = False

        for param in self.llava.parameters():
            param.requires_grad = False


    def forward(
            self, 
            imgs,
            qtn_ids,
            qtn_attns,
            device,
        ):

        encoded_imgs = self.clipenc(imgs, device)  # (B, L=w*h, 768)
        last_hidden_roberta = self.robenc(qtn_ids, qtn_attns) # (B, max_len, 768)   
        
        # combining the input embeddings: (B, seq, Hid)
        # keeping text first and then image emb
        input_embeds = torch.concat((last_hidden_roberta, encoded_imgs), dim=1)  # (B, L+max, 768)
        
        llava_output = self.llava(input_embeds).logits   # (B, L, 32000)
        
        projoutput = self.projector(llava_output) # (B,num_classes)
        print(projoutput.shape)

        exit()
        return projoutput