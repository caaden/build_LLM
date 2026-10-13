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


# %% Multinomial Sample
'''
    Multinomial treats the probas as a weighted die.  Instead of choosing
    the max, it samples from the distribution.  This is a more realistic
    way to generate text, as it allows for more diversity in the output.
'''
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
probas=torch.softmax(next_token_logits,dim=0)
next_token_id=torch.multinomial(probas,num_samples=1).item()
print(inverse_vocab[next_token_id])

# %% Test multinomial outputs

def print_sampled_tokens(probas):
     torch.manual_seed(123)
     sample=[torch.multinomial(probas,num_samples=1).item() for i in range(1_000)]
     sampled_ids=torch.bincount(torch.tensor(sample))
     for i,freq in enumerate(sampled_ids):
          print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

# %% Temperature implementation
def softmax_with_temperature(logits,temperature):
    '''
    Apply temperature scaling to logits and return softmax probabilities.
    Args:
        logits (torch.Tensor): The input logits.
        temperature (float): The temperature value for scaling.

    Desc:
        This function applies temperature scaling to the input logits by dividing
        them by the temperature value. A higher temperature results in a more uniform
        distribution, while a lower temperature makes the distribution more peaked.
    '''
    scaled_logits=logits/temperature
    return torch.softmax(scaled_logits,dim=0)

temps=[1,0.1,5]
scaled_probas=[softmax_with_temperature(next_token_logits,T) for T in temps]
x=torch.arange(len(vocab))
bar_width=0.15
fig,ax=plt.subplots(figsize=(5,3))
for i , T in enumerate(temps):
     rects=ax.bar(x+i*bar_width, scaled_probas[i], bar_width, label=f'Temp = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
fname="ex5.3.1_Temp.jpg"
plt.savefig(fname)