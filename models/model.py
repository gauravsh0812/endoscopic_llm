import torch 
import torch.nn as nn

class Endoscopic_model(nn.Module):

    def __init__(self, 
                 clipencoder, 
                 robertaencoder,
                 clipadaptor,
                 robertaadaptor,
                 projector,
    ):
        super(Endoscopic_model, self).__init__()
        self.clipenc = clipencoder
        self.robenc = robertaencoder
        self.clipadaptor = clipadaptor
        self.robertaadaptor = robertaadaptor
        self.projector = projector

    def forward(
            self, 
            imgs,
            qtn_ids,
            qtn_attns,
            device,
        ):

        # encoded_imgs = self.clipenc(imgs, device)  # (B, L=w*h, dim)
        encoded_imgs = imgs
        # print("imgs shape: ", imgs.shape)
        last_hidden_roberta = self.robenc(qtn_ids, qtn_attns) # (B, max_len, 768)        
        clipoutput = self.clipadaptor(encoded_imgs)  # (B, max_len, 64)
        roboutput = self.robertaadaptor(last_hidden_roberta) # (B, max, 64)
        projoutput = self.projector(clipoutput, roboutput) # (B,num_classes)
        
        return projoutput