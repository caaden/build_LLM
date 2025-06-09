# %% Dependencies
import torch

# %% Embedding layers

vocab_size=6 # Number of unique tokens in the vocabulary
output_dim=3 # Dimension of the embedding vector for each token
torch.manual_seed(123)
embedding_layer=torch.nn.Embedding(vocab_size,output_dim)
print(f'Sample embedding layer: {embedding_layer.weight}')

'''
embedding_layer.weight has shape (6, 3). 
One row for each of the 6 tokens in the vocabulary, and each row has 3 dimensions.

Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
'''

print(f'Sample output: {embedding_layer(torch.tensor([3]))}')
'''
This is NOT a product of the embedding layer and the input tensor.
Input tensor([3]) is a token ID, which corresponds to the 4th row of the embedding layer's weight matrix.
This is the embedding vector for the token with ID 3.
Sample output: tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
'''

print(f'Sample output: {embedding_layer(torch.tensor([0]))}')

# %% Applying to an input sequence
input_ids=torch.tensor([2,3,5,1])
output=embedding_layer(input_ids)
print(f'Input IDs: {input_ids}')
print(f'Output embeddings: {output}')
'''
The output is simply the embedding vectors for each of the input IDs.
row[2] corresponds to the embedding vector for token ID 2,
row[3] corresponds to the embedding vector for token ID 3, and so on.

Input IDs: tensor([2, 3, 5, 1])
Output embeddings: tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
'''
# %%
