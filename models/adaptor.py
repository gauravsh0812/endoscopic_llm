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
        
        return xc # (B,max_len, 64)

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
        
        return xr # (B,max_len, 64)

class Projector(nn.Module):

    def __init__(self, features, max_len, num_classes):
        super(Projector, self).__init__()
        self.final_lin1 = nn.Linear(max_len*2, max_len)
        self.final_lin2 = nn.Linear(max_len*features[-1], features[-1])
        self.final_lin3 = nn.Linear(features[-1], num_classes)
        self.gelu = nn.GELU()
        self.norm = nn.BatchNorm1d(features[-1])
        self.attn = Self_Attention(features[-1])
        self.pool = nn.MaxPool1d(1)

    def forward(self, xc, xr):
        # x_roberta + x
        x = torch.cat((xc,xr), dim=1)  
        x = self.gelu(self.norm(self.final_lin1(x.permute(0,2,1)))).permute(0,2,1)
        x = self.attn(x)
        x = torch.flatten(x, -2,-1)   # (B,max_len*64)
        x = self.gelu(self.final_lin2(x))  # (B, 64)
        x = self.gelu(self.final_lin3(x))  # (B, num_classes)
        x = self.pool(x)

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