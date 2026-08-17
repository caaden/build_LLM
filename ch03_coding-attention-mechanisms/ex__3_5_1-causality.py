#%% Initialization
import torch
from self_attn import SelfAttention_v2
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
print("Input tensor: ", inputs)
# %% Initialze attention weights
sa = SelfAttention_v2(din=inputs.shape[1], dout=2)
queries=sa.W_query(inputs)
keys=sa.W_key(inputs)
attn_scores=queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("Attention weights:\n", attn_weights)
# %% Make causal by masking the upper triangular part of the attention weights matrix
context_length = attn_weights.shape[0]
mask = torch.tril(torch.ones(context_length, context_length), diagonal=0)
print("Mask:\n", mask)
masked_attn_weights = attn_weights * mask
print("Masked attention weights:\n", masked_attn_weights)
# %% Renormalize the masked attention weights so that they sum to 1 for each query
row_sums = masked_attn_weights.sum(dim=-1, keepdim=True)
renormalized_attn_weights = masked_attn_weights / row_sums
print("Renormalized masked attention weights:\n", renormalized_attn_weights)

# %% Alternative formulation uses -inf for the masked positions before applying softmax
inf_mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
print("Inf mask:\n", inf_mask)
masked_attn_scores = attn_scores.masked_fill(inf_mask, float('-inf'))
print("Masked attention scores (inf):\n", masked_attn_scores)
# %% Renormalize the masked attention weights
masked_attn_weights = torch.softmax(masked_attn_scores / keys.shape[-1]**0.5, dim=-1)
print("Renormalized masked attention weights (inf):\n", masked_attn_weights)

# %%
