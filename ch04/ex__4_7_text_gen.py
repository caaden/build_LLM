#%% dependencies
from myGPTModel import GPT_CONFIG_124M, GPTModel
import torch
import tiktoken

#%% Tokenize input text and create a batch
def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        logits = logits[:,-1,:]
        probas=torch.softmax(logits,dim=-1)
        idx_next=torch.argmax(probas,dim=-1,keepdim=True)
        idx=torch.cat((idx,idx_next),dim=1)

    return idx

if __name__ == "__main__":
    #%% Initialization
    tokenizer=tiktoken.get_encoding("gpt2")
    start_context="Hello, I am"
    encoded=tokenizer.encode(start_context)
    print("Encoded input:",encoded)
    encoded_tensor=torch.tensor(encoded).unsqueeze(0) #batch size 1
    print("Encoded tensor shape:",encoded_tensor.shape)

    # %% Generate text
    torch.manual_seed(123) #note:this has downstream effects on the generated text
    model=GPTModel(GPT_CONFIG_124M)
    model.eval()
    out=generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M['context_length']
    )
    print("Output:" ,out)
    print("Output length:",out.shape[1])

    # %% Decode the output of an untrained model... gibberish!
    decoded_output=tokenizer.decode(out[0].tolist())
    print("Decoded output:",decoded_output)

    # %%
