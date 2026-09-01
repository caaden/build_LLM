import torch
import torch.nn as nn
from ch03.self_attn import MultiHeadAttention
from myGPTModel import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    '''
    Basic Transformer block with multi-head self-attention, feedforward network, layer normalization, and residual connections.
    '''
    def __init__(self,cfg):
        super().__init__()
        self.att=MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg('emb_dim'))
        self.norm2=LayerNorm(cfg('emb_dim'))
        self.drop_shortcut=nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
        '''
        forward pass through the transformer block, applying multi-head self-attention, feedforward network, layer normalization, and residual connections.
        '''
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        return x



