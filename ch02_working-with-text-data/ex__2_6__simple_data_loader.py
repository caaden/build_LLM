# %%# Importing the tiktoken library to check its version
from importlib.metadata import version
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")  # Using the GPT-2 tokenizer

# %% Implement a data loader
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
enc_text=tokenizer.encode(raw_text)
print("Encoded text length:", len(enc_text))
enc_sample=enc_text[50:]
print("Encoded text sample:", enc_sample)
# %% Create input and target sequences
context_length = 4  # Length of context for each input sequence
my_sample=tokenizer.decode(enc_sample[:context_length+1])
print(f'My sample text: {my_sample}')
print(f'My sample code: {enc_sample[:context_length+1]}\n')
print('Shifted mask...')
for i in range(1,context_length + 1):
    context=enc_sample[:i]
    target=enc_sample[i]
    print(context, "->", target)
    print(tokenizer.decode(context), "->", tokenizer.decode([target]))



# %%
