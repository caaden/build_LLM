import tiktoken
import torch

import argparse
import sys
from pathlib import Path
# Add the parent directory of the current file to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))

from ex__4_1 import GPT_CONFIG_124M
from ex__5_1_gen_text import init_gpt_model, text_to_token_ids, token_ids_to_text, set_data_loaders
from ex__5_2_training import train_model_simple



def generate(model, idx, max_new_tokens, context_size, temp=0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        logits=logits[:,-1,:]
        if top_k is not None:
            top_logits,top_pos=torch.topk(logits,top_k)
            min_val=top_logits[:,-1]
            logits=torch.where(logits<min_val,torch.tensor(float('-inf')).to(logits.device),logits)
        if temp>0:
            logits=logits/temp
            probs=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
        else:
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        if idx_next==eos_id:
            break
        idx=torch.cat((idx,idx_next),dim=1)

    return idx

if __name__=="__main__":

    # TRAIN mode: 1 = train from scratch, 2 = load saved model, 3 = load and continue training
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=int,choices=[1,2,3],default=1,
                        help="1: train from scratch, 2: load saved model, 3: load and continue training")
    args=parser.parse_args()

    torch.manual_seed(123)
    TRAIN=args.train
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,GPT_CONFIG_124M = init_gpt_model()
    model.to(device)
    tokenizer=tiktoken.get_encoding("gpt2")
    
    if TRAIN==1:
        # Import text data from the file
        file_path="../ch02/the-verdict.txt"
        with open(file_path,"r",encoding="utf-8") as file:
            text_data=file.read()
        # Initialize the optimizer  
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=0.0004, weight_decay=0.1
        )
        # Train the model
        num_epochs=12
        train_loader, val_loader = set_data_loaders(text_data, train_ratio=0.9, GPT_CONFIG_124M=GPT_CONFIG_124M)
        
        train_losses, val_losses, tokens_seen=train_model_simple(model,train_loader,val_loader,
                                                                optimizer,device,num_epochs=num_epochs,
                                                                eval_freq=5,eval_iter=5,start_context="Every effort moves you", 
                                                                tokenizer=tokenizer)
        # Save the model
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()}, "model_and_optimizer.pth")
    elif TRAIN==2:
        # Load the model
        checkpoint=torch.load("model_and_optimizer.pth",map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif TRAIN==3:
        # Load the model and optimizer state
        # The checkpoint is ~1.95 GB: the model weights (~650 MB) plus AdamW's two
        # moment estimates per weight (~1.3 GB). Loading it with map_location=device
        # put all of that on the GPU, on top of the model, gradients and optimizer
        # state that training needs, which overflowed the 3.6 GB GPU (CUDA OOM).
        # So load it into CPU RAM instead; load_state_dict copies each tensor onto
        # the device of the model/optimizer it is loaded into (the GPU here).
        checkpoint=torch.load("model_and_optimizer.pth",map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        # Note: load_state_dict restores the saved hyperparameters too, so the
        # lr below is replaced by the checkpoint's lr (0.0004). To change it,
        # set it after loading: for g in optimizer.param_groups: g["lr"]=0.0005
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # The model and optimizer now hold their own copies of the weights and
        # state, so drop the checkpoint dict to free that memory before training
        del checkpoint
        # Train the model for 3 more epochs
        file_path="../ch02/the-verdict.txt"
        with open(file_path,"r",encoding="utf-8") as file:
            text_data=file.read()
        model.train() # Assert the model is in training mode
        num_epochs=3
        train_loader, val_loader = set_data_loaders(text_data, train_ratio=0.9, GPT_CONFIG_124M=GPT_CONFIG_124M)
        train_losses, val_losses, tokens_seen=train_model_simple(model,train_loader,val_loader,
                                                                optimizer,device,num_epochs=num_epochs,
                                                                eval_freq=5,eval_iter=5,start_context="Every effort moves you", 
                                                                tokenizer=tokenizer)    
        torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, "model_and_optimizer.pth")

    # Set the model to evaluation mode to disable dropout and other training-specific layers
    model.eval()

    # Generate text using the trained model
    token_ids=generate(
        model=model,
        # The prompt must be on the same device as the model (GPU or CPU)
        idx=text_to_token_ids("Every effort moves you",tokenizer).to(device),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temp=1.4
    )
    
    print("output text:\n",token_ids_to_text(token_ids,tokenizer))
            