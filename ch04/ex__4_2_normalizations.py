#%% dependencies
import torch
import torch.nn as nn
from myGPTModel import LayerNorm

#%% Normalization Exercises
torch.manual_seed(123)
batch=torch.randn(2,5) #batch of 2 samples, each with 5 features
# Define a simple linear layer followed by ReLU activation. Input size 5, output size 6
layer=nn.Sequential(nn.Linear(5,6),nn.ReLU())
output=layer(batch)
print('Output shape:', output.shape)
print('Output:', output)
# %% Compute stats
mean=torch.mean(output, dim=1,keepdim=True)
var=torch.var(output, dim=1,keepdim=True)
print('Mean:', mean)
print('Variance:', var)
# %%
out_norm=(output-mean)/torch.sqrt(var+1e-5)
mean_norm=torch.mean(out_norm, dim=1,keepdim=True)
var_norm=torch.var(out_norm, dim=1,keepdim=True)
print('Normalized Output:', out_norm)
print('Mean of Normalized Output:', mean_norm)
print('Variance of Normalized Output:', var_norm)   
# %% Test Layer Norm 
ln=LayerNorm(emb_dim=6)
out_ln=ln(output)
mean=out_ln.mean(dim=-1,keepdim=True)
var=out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean from method: ",mean)
print("Var from method: ", var)
# %%
