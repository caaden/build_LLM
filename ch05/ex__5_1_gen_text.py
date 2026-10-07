#%% Library Dependencies
import torch
import tiktoken

#%% Helper Functions from Previous Chapters
import sys
from pathlib import Path
# Add the parent directory of the current file to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch02"))
#import helper functions
from myGPTModel import GPTModel
from ex__4_7_text_gen import generate_text_simple
from ex__2_6__torch_data_loader import create_dataloader_v1

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
def init_gpt_model(): 
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
    return model, GPT_CONFIG_124M

def calc_loss_batch(input_batch,target_batch, model,device):
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    logits=model(input_batch)
    loss=torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss=0.
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is None:
        num_batches=len(data_loader)
    else:
        num_batches=min(num_batches,len(data_loader))
    for i, (input_batch,target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            total_loss+=loss.item()
        else:
            break
    return total_loss/num_batches

def ex_5_1_1():
    model, GPT_CONFIG_124M=init_gpt_model()
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

def ex_5_1_2(OPTION):
    model, GPT_CONFIG_124M=init_gpt_model()
    model.eval()
    tokenizer=tiktoken.get_encoding("gpt2")
    inputs=torch.tensor([[16833,3626, 6100], # ["every effort moves",
                         [40,1107,588]]) # "I really like"]
    # Targets are input shifted by one position to the right and include the next token to predict
    targets=torch.tensor([[3626, 6100, 345], # ["effort moves you",
                          [1107,588,11311]]) # "really like chocolate"]
    # Run forward pass to get logits
    # The output logits will have shape (batch_size, sequence_length, vocab_size)
    # Each decoded token in the sequence will have a corresponding probability distribution over the vocabulary
    # The first token in the sequence is used to predict the second token, the first and second tokens are used to predict the third token, the first three tonkens are used to predict the fourth token.
    
    with torch.no_grad():
        logits=model(inputs)


    #%% Brute Force Calculation of Probabilities and Loss
    if OPTION=="Brute Force":    
        probas=torch.softmax(logits, dim=-1)
        print("Shape of probabilities: ",probas.shape)
        # Print the predicted token IDs by taking the argmax of the probabilities along the last dimension (vocab_size)
        # The model is presently untrained, so the output probabilities will be random and not meaningful.
        token_ids=torch.argmax(probas,dim=-1,keepdim=True)
        print("Token IDs:\n",token_ids)
        # Decode the predicted token IDs back to text using the tokenizer
        print(f"Targets batch 1: {token_ids_to_text(targets[0],tokenizer)}")
        print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(),tokenizer)}")
        
        # Show the probabilities of the target tokens for each text in the batch
        # Product of one hot probabilities for each token in the batch as determined by the cross entropy loss function. 
        # Extracts the predicted probabilities from each target token in the batch and prints them.
        text_idx=0
        target_probas_1=probas[text_idx,[0,1,2], targets[text_idx]]
        print("Text 1:", target_probas_1)    
        text_idx=1
        target_probas_2=probas[text_idx,[0,1,2], targets[text_idx]]
        print("Text 2:", target_probas_2)
        # Extract the log probabilities of the target tokens for each text in the batch and print them.
        # The log prob is 0 for a probability of 1 (the model is certain about the prediction) and negative for probabilities less than 1 (the model is uncertain about the prediction).
        # Maps to a loss function that penalizes the model for being uncertain about the prediction.
        log_probas=torch.log(torch.cat((target_probas_1,target_probas_2)))
        print("Log probabilities of target tokens:\n",log_probas)
        # Expected value for the log probabilities of the target tokens across the batch.
        # for a uniform distribution of probabilities, the expected value of the log probabilities will be negative and large in magnitude.
        # for a vocabulary size of 50257, the expected value of the log probabilities will be approximately -10.82.
        # Since the model is untrained, we expect approximately -10.82
        # If the model was trained, we would expect the value to be closer to 0
        avg_log_proba=torch.mean(log_probas)
        print("Average log probability of target tokens:\n",avg_log_proba)
        # Cross entropy loss is the negative average log probability of the target tokens. The loss function is used to train the model to minimize the loss and maximize the probability of the target tokens.
        # Note, even though the non-desired tokens are masked, their contribution appears in the denominator of the softmax function, which is used to calculate the probabilities of the target tokens. The model is penalized for being uncertain about the prediction of the target tokens, even if the non-desired tokens are masked.
        neg_avg_log_proba=-avg_log_proba
        print("Negative average log probability of target tokens:\n",neg_avg_log_proba)

    #%% Cross Entropy Loss Calculation
    elif OPTION=="Cross Entropy":
        # The cross entropy loss function is used to train the model to minimize the loss and maximize the probability of the target tokens.
        print("Logits shape:",logits.shape)
        print("Targets shape:", targets.shape)
        logits_flat=logits.flatten(0,1)
        targets_flat=targets.flatten()
        print("Flattened logits:",logits_flat.shape)
        print("Flattened targets:",targets_flat.shape)

        # Previously, we applied the softmax function, selected the probability scores corresponding to the target IDs, and computed the negative average log probabilities. PyTorch’s cross_entropy function will take care of all these steps for us.
        loss=torch.nn.functional.cross_entropy(logits_flat,targets_flat)
        print("Cross Entropy Loss:",loss)
    else:
        print("Select a valid OPTION.")

def ex_5_1_3():
    tokenizer=tiktoken.get_encoding("gpt2")

    file_path="../ch02/the-verdict.txt"
    with open(file_path,"r",encoding="utf-8") as file:
        text_data=file.read()

    total_characters=len(text_data)
    total_tokens=len(tokenizer.encode(text_data))
    print("Characters: ",total_characters)
    print("Tokens: ",total_tokens)

    train_ratio=0.9
    split_idx=int(train_ratio*len(text_data))
    train_data=text_data[:split_idx]
    val_data=text_data[split_idx:]

    model, GPT_CONFIG_124M=init_gpt_model()
    torch.manual_seed(123)

    train_loader=create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader=create_dataloader_v1(
            val_data,
            batch_size=2,
            max_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )

    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss=calc_loss_loader(train_loader,model,device)
        val_loss=calc_loss_loader(val_loader,model,device)
    print("Training Loss: ",train_loss)
    print("Validation Loss:",val_loss)



# %% main function
if __name__=="__main__":
    # OPTION="Cross Entropy"
    # ex_5_1_2(OPTION)
    ex_5_1_3()