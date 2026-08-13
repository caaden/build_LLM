import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, din, dout):
        super()().__init__()
        self.W_query = nn.Parameter(torch.rand(din, dout))
        self.W_key = nn.Parameter(torch.rand(din, dout))
        self.W_value = nn.Parameter(torch.rand(din, dout))
    def forward(self,x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / torch.sqrt(torch.tensor(keys.shape[-1], dtype=torch.float32)), dim=-1
        )
        context_vec=attn_weights @ values
        return context_vec
        