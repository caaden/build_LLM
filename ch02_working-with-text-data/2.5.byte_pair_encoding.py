# %%# Importing the tiktoken library to check its version
from importlib.metadata import version
import tiktoken
print(f"tiktoken version: {version('tiktoken')}")

# %% # Using the tiktoken library to encode text
tokenizer = tiktoken.get_encoding("gpt2")  # Using the GPT-2 tokenizer
text=(
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers=tokenizer.encode(text, allowed_special={"<|endoftext|>"})  # Encoding the text
print(f"Encoded integers: {integers}")

# Decoding the integers back to text
decoded_text = tokenizer.decode(integers)  # Decoding the integers
print(f"Decoded text: {decoded_text}")
# %% BPE exercise
text = "Akwirw ier"
# Using the GPT-2 tokenizer to encode the text
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(f"Encoded integers for BPE exercise: {integers}")
decoded_text = tokenizer.decode(integers)  # Decoding the integers
print(f"Decoded text for BPE exercise: {decoded_text}")

