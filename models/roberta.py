from transformers import RobertaModel
import torch.nn as nn

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
        