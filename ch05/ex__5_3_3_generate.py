import torch


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
   torch.manual_seed(123)
   token_ids=generate(
       model=model,
       idx=text_to

   ) 
            