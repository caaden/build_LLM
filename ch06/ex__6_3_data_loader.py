import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Use end of text token to delimit end of message (ham or spam)
tokenizer=tiktoken.get_encoding('gpt2')
pad_token = tokenizer.encode("<|endoftext|>",allowed_special={'<|endoftext|>'})
# print(f'Pad token: {pad_token}')

class SpamDataset(Dataset):
    '''
    A PyTorch Dataset class for loading the SMS Spam Collection dataset.
    It reads the data from a CSV file, tokenizes the text using the provided tokenizer, and pads/truncates the sequences to a specified maximum length.
    '''
    def __init__(self,csv_file,tokenizer,max_length=None, pad_token_id=pad_token):
        self.data=pd.read_csv(csv_file)
        # pretokenize texts
        self.encoded_texts=[ tokenizer.encode(text) for text in self.data['text'] ]
        if max_length is None:
            # initialize max_length to the length of the longest encoded text
            self.max_length=self._longest_encoded_length()
        else:
            self.max_length=max_length
            # truncate sequences if longer than max length
            self.encoded_texts=[encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        # pad sequences to max length
        self.encoded_texts=[encoded_text + pad_token_id *(self.max_length-len(encoded_text)) for encoded_text in self.encoded_texts]
    
    def __getitem__(self, index):
        '''
        Returns a tuple of (encoded_text, label) for the given index.
        encoded_text is a tensor of shape (max_length,) containing the token ids of the text.
        label is a tensor of shape (1,) containing the label (0 for ham, 1 for spam)
        '''
        encoded=self.encoded_texts[index]
        label=self.data.iloc[index]['label']
        return (
            torch.tensor(encoded,dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        '''
        Returns the length of the longest encoded text in the dataset.
        '''
        max_length=0
        for encoded_text in self.encoded_texts:
            encoded_length=len(encoded_text)
            if encoded_length>max_length:
                max_length=encoded_length
        return max_length


if __name__=="__main__":

    # Train dastaset
    train_dataset=SpamDataset(
        csv_file='train.csv',
        max_length=None,
        tokenizer=tokenizer)

    # Validation dataset
    val_dataset=SpamDataset(
        csv_file='val.csv',
        max_length=train_dataset.max_length,
        tokenizer=tokenizer)

    # Test dataset
    test_dataset=SpamDataset(
        csv_file='test.csv',
        max_length=train_dataset.max_length,
        tokenizer=tokenizer)

    num_workers=0
    batch_size=8
    torch.manual_seed(123)

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)

    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True)

    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True)

    for input_batch,target_batch in train_loader:
        pass
    print('Input batch dimensions:',input_batch.shape)
    print('Label batch dimensions:',target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")