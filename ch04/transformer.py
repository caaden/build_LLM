import torch
import torch.nn as nn

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
        self.norm1=LayerNorm(cfg['emb_dim'])
        self.norm2=LayerNorm(cfg['emb_dim'])
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # Functions the same as the MultiHeadAttentionWrapper class, but uses a single linear layer to project the concatenated outputs of all heads back to the desired output dimension. This allows for more efficient computation by replacing looping multiplications with a single matrix multiplication for keys, queries, and values.
        super().__init__()
        # Ensure that the output dimension is divisible by the number of heads to allow for equal distribution of features across heads. Each head will have a dimension of d_out / num_heads.
        assert (d_out % num_heads ==0), "d_out must be divisible by num_heads"
        self.head_dim=d_out // num_heads
        self.d_out=d_out
        self.num_heads=num_heads
        self.W_query=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key=nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value=nn.Linear(d_in,d_out, bias=qkv_bias)
        self.out_proj=nn.Linear(d_out,d_out)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b, num_tokens, d_in=x.shape
        keys=self.W_key(x)
        queries=self.W_query(x)
        values=self.W_value(x)

        # Use the view operation to reshape the tensors into a shape that separates the heads and the head dimensions. The new shape is (batch_size, num_tokens, num_heads, head_dim), where head_dim is d_out divided by num_heads. This allows each head to attend to different parts of the input features independently.
        keys=keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values=values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries=queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # Transpose the tensors to bring the num_heads dimension before the num_tokens dimension, resulting in a shape of (batch_size, num_heads, num_tokens, head_dim). This arrangement is necessary for the subsequent attention score computation, where each head will compute attention scores independently for each token in the sequence.
        keys=keys.transpose(1,2)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)
        # Compute the attention scores
        attn_scores=queries @ keys.transpose(2,3)
        # Mask the attention scores to prevent attending to future tokens. 
        mask_bool=self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)
        # Apply the softmax function to the attention scores to obtain the attention weights.   
        attn_weights=torch.softmax(
            attn_scores/keys.shape[-1]**0.5, dim=-1
        )
        # Apply dropout to the attention weights to prevent overfitting. The dropout layer randomly sets a fraction of the attention weights to zero during training, which helps to regularize the model and improve generalization.
        attn_weights=self.dropout(attn_weights)
        # Compute the context vector as a weighted sum of the value vectors. 
        context_vec=(attn_weights @ values).transpose(1,2)
        # Use the contiguous() method to ensure that the context vector is stored in a contiguous block of memory, which is necessary for the subsequent view operation. The view operation reshapes the context vector back to its original shape of (batch_size, num_tokens, d_out) by combining the num_heads and head_dim dimensions.
        context_vec=context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec=self.out_proj(context_vec)
        return context_vec


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

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        y = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * 
                                      (x+0.044715 * torch.pow(x,3))))
        return y