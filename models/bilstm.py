import torch.nn as nn 

class BiLSTM(nn.Module):
    def __init__(
            self,
            input_dim, 
            hidden_dim,
            output_dim,
    ):
        super(BiLSTM,self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.1, 
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,x):
        x,(h,c) = self.lstm(x)
        print("lstm shape: ", x.shape)
        x = self.fc(x)  # (B, 3, output_dim)

        return x