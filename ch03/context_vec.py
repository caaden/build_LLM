#%% dependencies
import torch
import numpy as np

#%% inputs

tokens=torch.tensor([[1,0,0],[0,1,0],[1,1,0],[0,1,1]],dtype=torch.float32)  # shape: (4,3)  # 4 tokens, 3 features each

#%% autocorrelation (Gram matrix): entry (i,j) is dot(tokens[i], tokens[j])
autocorr = tokens @ tokens.T  # shape: (4,4)
softmax_weights=torch.softmax(autocorr, dim=-1)  # shape: (4,4)  # attention weights for each token pair

#%% context vectors: weighted sum of tokens, weighted by (softmax'd) autocorrelation
context_vec = softmax_weights @ tokens  # shape: (4,3)




# %%
