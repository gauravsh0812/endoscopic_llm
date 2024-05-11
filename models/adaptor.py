import torch 
import torch.nn as nn

class ClipAdaptor(nn.Module):
    def __init__(self, clip_in_dim, features, max_len):
        super(ClipAdaptor, self).__init__()

        self.cliplin1 = nn.Linear(clip_in_dim, features[0])
        self.cliplin2 = nn.Linear(features[0], features[1])
        self.cliplin3 = nn.Linear(features[1], features[2])
        self.cliplin4 = nn.Linear(features[2], features[3])
        self.proj_clip = nn.Linear(50,max_len)
        self.relu = nn.ReLU()
    
    def forward(self, xc):

        xc = self.relu(self.cliplin1(xc))
        xc = self.relu(self.cliplin2(xc))
        xc = self.relu(self.cliplin3(xc))
        xc = self.relu(self.cliplin4(xc))
        xc = self.relu(self.proj_clip(xc.permute(0,2,1))).permute(0,2,1)
        
        return xc # (B,19, 64)

class RobertaAdaptor(nn.Module):
    def __init__(self, roberta_in_dim, features):
        super(RobertaAdaptor, self).__init__()

        self.roblin1 = nn.Linear(roberta_in_dim, features[0])
        self.roblin2 = nn.Linear(features[0], features[1])
        self.roblin3 = nn.Linear(features[1], features[2])
        self.roblin4 = nn.Linear(features[2], features[3])
        self.gelu = nn.GELU()
    
    def forward(self, xr):
        xr = self.gelu(self.roblin1(xr))
        xr = self.gelu(self.roblin2(xr))
        xr = self.gelu(self.roblin3(xr))
        xr = self.gelu(self.roblin4(xr))
        
        return xr # (B,19, 64)

class Projector(nn.Module):

    def __init__(self, features, max_len, num_classes):
        super(Projector, self).__init__()
        self.final_lin1 = nn.Linear(features[-1]*2, features[-1])
        self.final_lin2 = nn.Linear(features[-1]*max_len, num_classes)
        self.gelu = nn.GELU()
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, xc, xr):
        # x_roberta + x
        x = torch.cat((xc,xr), dim=-1)  
        x = self.gelu(self.final_lin1(x))
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = self.gelu(self.norm(self.final_lin2(x)))  # (B, 11)
        
        return x   # (B, 11)