#%% Dependencies
import tiktoken
import torch
import numpy as np

import sys
from pathlib import Path
# Add the parent directory of the current file to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
from ex__4_1 import GPT_CONFIG_124M

from myGPTModel import GPTModel
from ex__5_1_gen_text import text_to_token_ids, token_ids_to_text
from ex__5_3_3_generate import generate

#%% Helper Functions
def assign(left, right):
    '''
    Assign the values of the right tensor to the left tensor, ensuring that their shapes match.
    The shapes wouldn't match if the model configuration is different from the pretrained weights.
    '''
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    '''
    Load the pretrained weights from the params dictionary into the GPT model.'''
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe']) # position embedding
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte']) # token embedding

    # Load weights for each transformer block
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
   

#%% Download model weights and load the model
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
# %% Validate the settings and parameter dictionary
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())
# %% Inspect the token embedding weights
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
# %% Create a dictionary of model weights for the GPT-2 model
model_configs = {
    "gpt2-small": {'embedding_dim': 768, 'num_layers': 12, 'num_heads': 12},
    "gpt2-medium": {'embedding_dim': 1024, 'num_layers': 24, 'num_heads': 16},
    "gpt2-large": {'embedding_dim': 1280, 'num_layers': 36, 'num_heads': 20},
    "gpt2-xl": {'embedding_dim': 1600, 'num_layers': 48, 'num_heads': 25},
}

# %%
model_name="gpt2-small"
NEW_CONFIG=GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"qkv_bias": True})

# %%
gpt=GPTModel(NEW_CONFIG)
gpt.eval()  # Set the model to evaluation mode

load_weights_into_gpt(gpt, params)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)
tokenizer=tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
token_ids=generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    temp=1.5,
    top_k=50,
    eos_id=None
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
