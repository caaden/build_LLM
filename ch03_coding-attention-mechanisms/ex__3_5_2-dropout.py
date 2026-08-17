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
context_length = attn_scores.shape[0]
inf_mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
masked_attn_scores = attn_scores.masked_fill(inf_mask, float('-inf'))
masked_attn_weights = torch.softmax(masked_attn_scores / keys.shape[-1]**0.5, dim=-1)
print("Renormalized masked attention weights (inf):\n", masked_attn_weights)
# %% Apply dropout to the attention weights
torch.manual_seed(123)
dropout = torch.nn.Dropout(p=0.5)
example=torch.ones(6,6)
print("Example dropout:\n", dropout(example))
# Note, remaining elements are scaled by 1/(1-p) to maintain the expected sum of the attention weights.

# %% Apply dropout to the masked attention weights
torch.manual_seed(123)
print("Masked attention weights after dropout:\n", dropout(masked_attn_weights))

# %%
