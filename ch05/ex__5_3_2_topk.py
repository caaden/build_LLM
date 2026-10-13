#%% Dependencies
import torch
from matplotlib import pyplot as plt

#%% Temp Scaling Simple Vocab

vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 
inverse_vocab = {v: k for k, v in vocab.items()}


# %% Top K
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

topk=3
top_logits,top_pos=torch.topk(next_token_logits,topk)
print("Top logits:",top_logits)
print("Top positions:",top_pos)

new_logits=torch.where(
    condition=next_token_logits<top_logits[-1], #find ind of all logits smaller that lowest kept K
    input=torch.tensor(float('-inf')), #assign to -inf
    other=next_token_logits
)

print(new_logits)

topk_probas=torch.softmax(new_logits,dim=0)
print("Top K probabilities:",topk_probas)

