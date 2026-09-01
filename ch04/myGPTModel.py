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

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        y = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * 
                                      (x+0.044715 * torch.pow(x,3))))
        
        return y

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
    '''
    Layer Normalization module.
    Intent: Normalizes each token's activation vector to zero mean and unit variance (with learned scale/shift), keeping activations at a stable scale across layers and smoothing optimization The normalization is performed by subtracting the mean and dividing by the standard deviation of the inputs, followed by scaling and shifting using learnable parameters.
    Benefits: Layer normalization improves the convergence of training, especially in deep networks, by ensuring that the inputs to each layer have a consistent distribution.'''
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

class FeedForward(nn.Module):
    '''
    Feed Forward Network with GELU activation.
    Intent: This class implements a feed-forward neural network layer commonly used in transformer architectures. It consists of two linear transformations with a GELU activation function in between. The first linear layer expands the input dimension to four times its size, and the second linear layer projects it back to the original dimension. This design allows for complex feature transformations while maintaining the original input size for residual connections.
    Benefits: The use of GELU activation provides smoother gradients and better performance compared to traditional activation functions like ReLU. The expansion and contraction of dimensions allow the model to learn richer representations, enhancing its ability to capture complex patterns in the data.
    '''
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg['emb_dim'],4*cfg['emb_dim']),
            GELU(),
            nn.Linear(4*cfg['emb_dim'],cfg['emb_dim'])
        )
    def forward(self,x):
        return self.layers(x)


