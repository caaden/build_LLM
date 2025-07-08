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

# %% Compute the context vector 
# Focus on the second token/embedding vector (depth 3)
query = inputs[1]
# Initialize an array that matches the number of input tokens being considered
attn_scores_2=torch.empty(inputs.shape[0])
# In this case we assume the weights to generate the context factor are simply the embedding vector for each 
# token being considered
# Note: this current example is simply a projection of the focused embedding vector on all others.
# As expected, the highest dot product is with itself (assuming orthonormal basis, which isn't true here)
for i,x_i in enumerate(inputs):
    attn_scores_2[i]=torch.dot(x_i,query)    
print(attn_scores_2)
'''tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])'''


# %% Normalize attention scores
attn_weights_2_tmp=attn_scores_2 / attn_scores_2.sum()
print(f'Attention weights: {attn_weights_2_tmp}')
'''Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])'''
# Validate that sum = 1
print(f'Sum: {attn_weights_2_tmp.sum()}')
'''Sum: 1.0000001192092896'''
# %% Replace norm with softmax (sigmoid type function that is favorable for gradient based operations and for probabalistic interpretation)
def softmax_naive(x):
  return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2=softmax_naive(attn_scores_2)
print(f'Attention weights: {attn_weights_2}')
print(f'Sum of attention weights: {attn_weights_2.sum()}')

# %% multiply each of the input vectors by the attention weights
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)
'''tensor([0.4419, 0.6515, 0.5683])'''  
# Note, this new vector represents the input of the second token with context/significance of all other tokens in the input list
# %% Evaluate context scores for each embedding vector in the input tensor
attn_scores = torch.empty(6, 6)
# Outer for current element
for i, x_i in enumerate(inputs):
    # Inner loop steps through other elements to determine the attention scores
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
'''
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
'''
# %% Compact representation using the autocorrelation concept
attn_scores = inputs @ inputs.T
print(attn_scores)

# %% Compute the softtmax to normalize
attn_weights=torch.softmax(attn_scores,dim=-1) # (-1) argument to evaluate over the last dimension of the tensor (row,col) so cols
print(f'Attention Weights: {attn_weights}')
'''Attention Weights: tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])'''
print("All row sums:", attn_weights.sum(dim=-1))
'''All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])'''
# %% Simple matrix multiplication yields the desired output tensor
all_context_vecs = attn_weights @ inputs
print(f'Output Tensor: {all_context_vecs}')
'''Output Tensor: tensor([[0.4421, 0.5931, 0.5790],
        [0.4419, 0.6515, 0.5683],
        [0.4431, 0.6496, 0.5671],
        [0.4304, 0.6298, 0.5510],
        [0.4671, 0.5910, 0.5266],
        [0.4177, 0.6503, 0.5645]])'''

# %% 
