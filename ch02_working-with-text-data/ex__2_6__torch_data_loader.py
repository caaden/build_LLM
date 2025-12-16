# %% Dependencies
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# %% Custom Dataset class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        args:
            txt: str, the text to be tokenized and processed
            tokenizer: a tokenizer object that has an encode method
            max_length: int, the maximum length of each input chunk
            stride: int, the step size for creating overlapping chunks
            
        returns:
            None, initializes the dataset with input and target IDs
        """
        self.input_ids=[]
        self.target_ids=[]
        
        token_ids=tokenizer.encode(txt)
        
        
        for i in range(0,len(token_ids)-max_length,stride):
            # create an input chunk from the tokenized text
            # the target chunk is the same as the input chunk but shifted by one position
            # this is done to predict the next token in the sequence
            # read as: input_chunk = [1, 2, 3, 4] and target_chunk = [2, 3, 4, 5]
            # when we see the first element of the input chunk, we want to predict the second element which is the first element of the target chunk
            # when we see the first element AND the second element of the input chunk, we want to predict the third element, 
            # which is the second element of the target chunk... and so on
            # THAT means that the max_length will limit the amount of context used to predict the next token
            input_chunk=token_ids[i:i+max_length] 
            target_chunk=token_ids[i+1:i+max_length+1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    
# %% DataLoader function
def create_dataloader_v1(txt,batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):
    """
    args:
        txt: str, the text to be tokenized and processed
        
        batch_size: int, the number of chunks per batch. By default, it is set to 4, which means that each batch will contain 4 input chunks and 4 target chunks.  
        
        max_length: int, the maximum length of each input chunk.  By default, it is set to 256, which means that each input chunk will contain a maximum of 256 tokens.  
        
        stride: int, the step size for creating overlapping chunks.  By default, it is set to 128, which means that the chunks will overlap by 128 tokens.
        
        shuffle: bool, whether to shuffle the dataset. By default, it is set to True, which means that the dataset will be shuffled before creating batches.
        This is useful for training models on large datasets, as it helps to prevent overfitting.
        
        drop_last: bool, whether to drop the last incomplete batch. By default, it is set to True, which means that the last batch will be dropped if it is not complete.
        This is useful for training models on large datasets, as it helps to prevent overfitting.
        
        num_workers: int, number of subprocesses to use for data loading. By default, it is set to 0, which means that the data will be loaded in the main process. 
        
    returns:
        DataLoader object for the dataset
    """
    tokenizer=tiktoken.get_encoding("gpt2")
    dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader

if __name__ == "__main__":

    # %% Example usage    
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader=create_dataloader_v1(
        raw_text, 
        batch_size=10, 
        max_length=4, 
        stride=1, 
        shuffle=False, 
        drop_last=True, 
        num_workers=0
    )
    data_iter = iter(dataloader)
    first_batch=next(data_iter)
    print(first_batch)
    second_batch=next(data_iter)
    print(second_batch)        
            
    # %% Another example with larger batch size and longer text
    dataloader=create_dataloader_v1(
        raw_text, 
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False)

    data_iter=iter(dataloader)
    inputs,targets=next(data_iter)
    print("Inputs:", inputs)
    print("Targets:", targets)
