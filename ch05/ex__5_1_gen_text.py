#%% Library Dependencies
import torch
import tiktoken

#%% Helper Functions from Previous Chapters
import sys
from pathlib import Path
# Add the parent directory of the current file to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
#import helper functions
from myGPTModel import GPTModel
from ex__4_7_text_gen import generate_text_simple   

#%% New Helper Functions
def text_to_token_ids(text,tokenizer):
    '''
    Convert text to token IDs using the provided tokenizer.'''
    encoded=tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    '''
    Convert token IDs back to text using the provided tokenizer.'''
    flat=token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

#%% GPTModel 
GPT_CONFIG_124M={
    "vocab_size":50257,
    "context_length":256,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False 
}

torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_124M)
model.eval()

start_context="Every effort moves you"
tokenizer=tiktoken.get_encoding("gpt2")

token_ids=generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context,tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n",token_ids_to_text(token_ids,tokenizer))

# %%
