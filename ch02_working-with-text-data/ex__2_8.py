# %% Dependencies
import torch
from ex__2_6__torch_data_loader import GPTDatasetV1, create_dataloader_v1
import tiktoken

#%% 
if __name__ == "__main__":
    # %% Load the text data
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    # %% Initialization
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # %% Instantiate the dataloader
    max_length = 4
    dataloader= create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False)


    # %% Print the first batch of data
    data_iter = iter(dataloader) # create an iterator from the dataloader
    inputs,targets = next(data_iter) # get the first batch of data
    print("Token IDs:\n", inputs)
    print("Input shape:", inputs.shape)

    # %% Embed the input tokens
    embedded_inputs = token_embedding_layer(inputs)
    print("Embedded inputs shape:", embedded_inputs.shape) # The embedded inputs now have 256 dimensions
    # Apply absolute positional encoding
    # Positional encodings are used to give meaning to the position of each token in the sequence.
    # Positional encodings are applied to the input embeddings - not the input tokens themselves.
    # So, in this example, the positional encoding is a vector of size 256x4 that is added to each token embedding that is also of length 256x4.
    # The position encodings are learned
    context_length = max_length # The context length determines how far back the model can look
    pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("Positional embeddings shape:\n", pos_embeddings.shape)
    # Print the first positional embeddings vector
    print("First positional embedding:\n", pos_embeddings[0])
    # Add the positional embeddings to the input embeddings
    embedded_inputs = embedded_inputs + pos_embeddings
    print("Embedded inputs with positional encodings shape:\n", embedded_inputs.shape)
    
    
    
    