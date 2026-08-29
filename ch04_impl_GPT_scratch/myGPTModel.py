#%% Dependencies
import torch
import torch.nn as nn
import tiktoken

#%% Config Dictionary
GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

#%% Dummy GPT Model
class DummyGPTModel(nn.Module):
    '''
    A dummy GPT model for development purposes. 
    '''
    def __init__(self,cfg):
        super().__init__()
        # Define the model components
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"]) # input token embedding
        self.pos_emb=nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # position embedding
        self.drop_emb=nn.Dropout(cfg['drop_rate']) # dropout layer
        self.trf_blocks=nn.Sequential( # stack of transformer blocks
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm=DummyLayerNorm(cfg["emb_dim"]) # final layer normalization
        self.out_head=nn.Linear( # output head for logits
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )

    def forward(self, in_idx):
        # Forward pass through the dummy GPT model
        batch_size, seq_len = in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()

    def forward(self,x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim))
    def forward(self,x):
        mean=torch.mean(x, dim=-1, keepdim=True)
        var=torch.var(x, dim=-1, keepdim=True,unbiased=False)
        norm_x=(x-mean)/(torch.sqrt(var+self.eps))
        return self.scale*norm_x+self.shift


