import torch
import torch.nn as nn
import torchvision.models as models
from transformers import VisualBertConfig, VisualBertModel, GPT2Model

class Endoscopic_model(nn.Module):
    def __init__(self,device,max_len, num_classes):
        
        super(Endoscopic_model, self).__init__()

        self.device = device

        # feature extactor
        resnet18 = models.resnet18(pretrained=True)
        self.encoded_img = nn.Sequential(*(list(resnet18.children())[:-2]))

        # encoded image embedding
        VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        VB_config.visual_embedding_dim = 512
        visualbert = VisualBertModel(config=VB_config)
        self.visual_embedder = visualbert.embeddings.visual_projection

        # qtn embedding
        gpt = GPT2Model.from_pretrained("openai-community/gpt2")
        self.word_embeddings = gpt.wte

        # final_embedding
        self.decoder = GPT2Model.from_pretrained("openai-community/gpt2")

        # pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # layers
        self.layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, imgs, qtn_attns, qtn_ids, device):

        # image embedding
        features = self.encoded_img(imgs)
        if torch.isnan(features).any():
            print("features contains NaN:", torch.isnan(features).any())

        features = torch.flatten(features,2,-1)          
        visual_embeds = self.visual_embedder(features.permute(0,2,1)).to(self.device)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(self.device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(self.device)
        visual_position_id = torch.zeros(visual_embeds.size()[:-1], dtype=torch.long).to(self.device)

        if torch.isnan(visual_embeds).any():
            print("vis ids contains NaN:", torch.isnan(visual_embeds).any())
        
        # qtn embedding
        qtn_ids = qtn_ids.to(self.device)
        qtn_attention_mask = qtn_attns.to(self.device)
        qtn_embeds = self.word_embeddings(qtn_ids)
        qtn_token_ids = torch.zeros(qtn_embeds.size()[:-1], dtype=torch.long).to(self.device)
        qtn_position_id = torch.zeros(qtn_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # concat: keeping questions first
        embeds = torch.cat((qtn_embeds, visual_embeds), dim=1).to(self.device)
        attns = torch.cat((qtn_attention_mask, visual_attention_mask), dim=1).to(self.device)
        token_ids = torch.cat((qtn_token_ids, visual_token_type_ids), dim=1).to(self.device)
        pos_ids = torch.cat((qtn_position_id, visual_position_id), dim=1).to(self.device)

        if torch.isnan(embeds).any():
            print("Embeds contains NaN:", torch.isnan(embeds).any())

        # decoding
        output = self.decoder(
            inputs_embeds=embeds,
            attention_mask=attns,
            token_type_ids=token_ids,
            position_ids=pos_ids,
        )
        output = output.last_hidden_state

        if torch.isnan(output).any():
            print("NaNs detected in model output")
            
        output = self.pool(output.permute(0,2,1)).permute(0,2,1)
        output = torch.flatten(output, -2, -1)
        print(output.shape)
        output = self.layer(output) # (B, num_classes)

        return output 
    
# m = Endoscopic_model()
# image = torch.rand(1,3,854,480)
# m(image)