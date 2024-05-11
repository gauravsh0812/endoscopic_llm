import torch 
import torch.nn as nn

class ClevrMath_model(nn.Module):

    def __init__(self, 
                 encoder, 
                 decoder,
                 clipadaptor,
                 robertaadaptor,
                 projector,
    ):
        super(ClevrMath_model, self).__init__()
        self.enc = encoder
        self.clipadaptor = clipadaptor
        self.robertaadaptor = robertaadaptor
        self.dec = decoder
        self.projector = projector

    def forward(self, imgs, ids, attns, device):
        encoded_imgs,pooled_layers = self.enc(imgs, device)  # (B, L=w*h, dim)
        last_hidden_roberta = self.dec(ids, attns) # (B, max_len, 768)        
        clipoutput = self.clipadaptor(encoded_imgs)  # (B, max_len, 64)
        roboutput = self.robertaadaptor(last_hidden_roberta) # (B, max, 64)
        output = self.projector(clipoutput, roboutput) # (B,11)
        
        return output