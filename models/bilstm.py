import torch.nn as nn 

class BiLSTM(nn.Module):
    def __init__(
            self,
            input_dim, 
            hidden_dim,
            output_dim,
            embed_dim,
            dropout,
    ):
        super(BiLSTM,self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.1, 
        )
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,x):
        x = self.dropout(self.embedding(x))
        x = self.lstm(x)
        x = self.fc(x)  # (B, L, output_dim)

        return x