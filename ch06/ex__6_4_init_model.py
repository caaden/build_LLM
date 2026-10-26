#%% Dependencies
import sys
from pathlib import Path

# Local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
from myGPTModel import GPTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch05"))
from gpt_download import download_and_load_gpt2
from ex__5_5_pretrained_weights import load_weights_into_gpt


#%% Configs
CHOOSE_MODEL='gpt2-small (124M)'
INPUT_PROMPT='Every effort moves'
BASE_CONFIG = {
    'vocab_size':50257,
    'context_length':1024,
    'drop_rate':0.0,
    'qkv_bias':True
}

model_configs={
    'gpt2-small (124M)':{'emb_dim':768,'n_layers':12, 'n_heads':12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}
# %%
model_size=CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size,models_dir='gpt2')
model=GPTModel(BASE_CONFIG)
load_weights_into_gpt(model,params)
model.eval()
