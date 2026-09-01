#%% Initialize inputs
import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
print("Input tensor: ", inputs)
# %%
x_2=inputs[1]
din=inputs.shape[1]
print("Input feature depth (din):", din)
dout=2
print("Output feature depth (dout):", dout)

#%% Initialize weights
torch.manual_seed(123)
# generate random weights for query, key, and value matrices
# n rows matches the input dimension for matrix multiplication, and m columns matches the output dimension
W_query=torch.nn.Parameter(torch.rand(din,dout),requires_grad=False)
W_key=torch.nn.Parameter(torch.rand(din,dout),requires_grad=False)
W_value=torch.nn.Parameter(torch.rand(din,dout),requires_grad=False)
#%% Compute query, key, and value vectors for x_2
# matrix multiplication maps the input vector to an output vector with the specified output dimension for each query, key, and value vectors
query_2=x_2 @ W_query
key_2=x_2 @ W_key
value_2=x_2 @ W_value
print("Query vector for x_2:", query_2)
print("Key vector for x_2:", key_2)
print("Value vector for x_2:", value_2)
# %% Expand for all inputs
# note: @ is shorthand for torch.matmul() and performs matrix multiplication
queries=inputs @ W_query
keys=inputs @ W_key
values=inputs @ W_value 
print("Query vectors for all inputs:\n", queries)
print("Key vectors for all inputs:\n", keys)
print("Value vectors for all inputs:\n", values)

# %% Review intermediate shapes
print("inputs.shape:", inputs.shape)
print("queries.shape:", queries.shape)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# %% compute attention score between x_2 and itself
attn_score_22=query_2.dot(key_2)
print("Attention score between x_2 and itself:", attn_score_22)


# %% compute attention scores between all queries and keys
attn_scores=queries @ keys.T
print("Attention scores between all queries and keys:\n", attn_scores)

# %% Compute normalized attention scores
# %% Scale the attention scores by the square root of the key dimension to address the vanishing gradient problem and improve training stability

dk=keys.shape[-1]
scaled_attn_scores=attn_scores/torch.sqrt(torch.tensor(dk,dtype=torch.float32))
print("Scaled attention scores:\n", scaled_attn_scores)

# %% Compute attention weights using softmax
attn_weights=torch.nn.functional.softmax(scaled_attn_scores,dim=-1)
print("Attention weights:\n", attn_weights)

 
# %% Compute the context vector for the whole sequence
context_vectors=attn_weights @ values
print("Context vectors:\n", context_vectors)

# %% 
