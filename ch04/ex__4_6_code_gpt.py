#%% dependencies
from myGPTModel import GPT_CONFIG_124M, GPTModel
import torch
import tiktoken

#%% Tokenize input text and create a batch
tokenizer = tiktoken.get_encoding("gpt2")
batch=[]
txt1="Every effort moves you"
txt2="Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch=torch.stack(batch, dim=0)
print('batch: ', batch)

torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_124M)
out=model(batch)
print("Input batch:\n",batch)
print("\nOutput shape:",out.shape)
print(out)
# %% Total Model Size
total_params=sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
# %% Weight Tying
print("Token embedding layer shape:",model.tok_emb.weight.shape)
print("Output layer shape:",model.out_head.weight.shape)
total_params_gpt2=(
    total_params-sum(p.numel() for p in model.out_head.parameters())
    )
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}")

# %% Compute Memory Requirements
total_size_bytes=total_params*4
total_size_mb=total_size_bytes/(1024*1024)
print(f"Total size of the model: {total_size_mb:0.2f} MB")

# %%
