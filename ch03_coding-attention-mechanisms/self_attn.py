import torch
import torch.nn as nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

din=inputs.shape[1]
dout=2

class SelfAttention_v1(nn.Module):
    """
    A simple implementation of self-attention mechanism.
    Computes the context vector with the following tasks:
    1. Compute query, key, and value vectors for the input.
    2. Compute attention scores between all queries and keys.
    3. Normalize the attention scores using output shape and softmax.
    4. Compute the context vector as a weighted sum of the value vectors.
    5. Return the context vector.
    """
    def __init__(self, din, dout):
        super().__init__()
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

class SelfAttention_v2(nn.Module):
    """
    This version of self-attention uses nn.Linear layers for the query, key, and value transformations. It allows for optional bias in the linear layers.

    Note: The Linear representation uses the Transpose of the weight matrix for the forward pass, which is equivalent to the matrix multiplication used in SelfAttention_v1. The nn.Linear layer also includes a bias term by default, which can be disabled by setting qkv_bias=False.
    """
    def __init__(self, din, dout,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(din, dout, bias=qkv_bias)
        self.W_key = nn.Linear(din, dout, bias=qkv_bias)
        self.W_value = nn.Linear(din, dout, bias=qkv_bias)
    def forward(self,x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / torch.sqrt(torch.tensor(keys.shape[-1], dtype=torch.float32)), dim=-1
        )
        context_vec=attn_weights @ values
        return context_vec
    
class CausalAttention(nn.Module):
    """
    This version of self-attention implements causal attention, which prevents the model from attending to future tokens in a sequence. It uses nn.Linear layers for the query, key, and value transformations, and allows for optional bias in the linear layers.
    """
    def __init__(self, din, dout, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = dout
        
        self.W_query = nn.Linear(din, dout, bias=qkv_bias)
        self.W_key = nn.Linear(din, dout, bias=qkv_bias)
        self.W_value = nn.Linear(din, dout, bias=qkv_bias)
        # Add dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        # Use a register_buffer to tell PyTorch to move the mask to the appropriate device (CPU/GPU) when the model is moved but do not treat it as a learnable parameter. The mask is a boolean tensor that indicates which positions in the attention scores should be masked (set to -inf) to prevent attending to future tokens.
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
    
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1,2)
        # replace the masked positions in the attention scores with -inf to prevent attending to future tokens
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens], -torch.inf)
        # note, the softmax function will automatically handle the -inf values by assigning them a probability of 0, effectively ignoring them in the attention computation.
        attn_weights=torch.softmax(
            attn_scores/keys.shape[-1]**0.5, dim=-1)
        # apply dropout to the attention weights to prevent overfitting
        attn_weights=self.dropout(attn_weights)
        # compute the context vector as a weighted sum of the value vectors
        context_vec=attn_weights@values

        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    '''
    This class implements multi-head attention by creating multiple instances of the CausalAttention class and concatenating their outputs. It allows for parallel attention computations across different subspaces of the input features.
    '''
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads=nn.ModuleList(
            [
                CausalAttention(
                    d_in,d_out,context_length, dropout, qkv_bias
                ) for _ in range(num_heads)
            ]
        )
    def forward(self,x):
        # Concatenate the outputs of all attention heads along the last dimension to form the final context vector. Each head processes the input independently, and their outputs are combined to capture diverse aspects of the input features.
        return torch.cat([head(x) for head in self.heads], dim=-1)

if __name__ == "__main__":
    torch.manual_seed(123)
    sa_v1=SelfAttention_v1(din, dout)
    print("Input tensor: ", inputs)
    context_vec=sa_v1(inputs)
    print("Context vector V1:\n", context_vec)

    torch.manual_seed(789)
    sa_v2=SelfAttention_v2(din, dout)
    context_vec=sa_v2(inputs)
    print("Context vector V2:\n", context_vec)

    batch=torch.stack([inputs, inputs], dim=0)
    torch.manual_seed(123)
    context_length=batch.shape[1]
    ca=CausalAttention(din, dout, context_length, 0.0)
    context_vecs=ca(batch)
    print("Context vector Causal:\n", context_vecs)

    torch.manual_seed(123)
    context_length=batch.shape[1]
    d_in, d_out = 3,1
    mha=MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2
    )
    context_vecs=mha(batch)
    print("Context vector Multi-Head:\n", context_vecs)

