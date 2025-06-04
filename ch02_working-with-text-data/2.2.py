# %% Dependencies
import re
import urllib.request

# %% Download text 
# Note, the raw.githubusercontent url to fetch the raw implementation
url="https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path="the-verdict.txt"
urllib.request.urlretrieve(url,file_path)
print(f"Downloaded to {file_path}")

# %% See the text
with urllib.request.urlopen(url) as response:
    text = response.read().decode('utf-8')
    print(text)

# %% Read a few lines
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
print("Total number of characters:",len(raw_text))
print(raw_text[:99])

# %% Use Python regular expressions to split a simple sentence into tokens
text="Hello world. This, is a test."
result=re.split(r'(\s)',text)
print(f'Split sentence: {result}')
# Remove white space
result=[item for item in result if item.strip()]
print(f'Removed whitespace: {result}')

# %% Expand tokenization scheme
text = "Hello, world. Is this-- a test?" 
result = re.split(r'([,.:;?_!"()\']|--|\s)', text) 
result = [item.strip() for item in result if item.strip()] 
print(f'Expanded takenization scheme: {result}')

# %% Apply the expanded tokenizer to the entire text file
preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed=[item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# %% Determine the vocabulary size
all_words=sorted(set(preprocessed))
vocab_size=len(all_words)
print(vocab_size)

# %% Create a vocabulary mapping unique words/punc to integers
vocab={token:integer for integer,token in enumerate(all_words)}
# Print the first 50 mappings
for i,item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break
print(len(vocab)) # print new vocab size

# %% Implement a tokenizer class
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int=vocab # store vocab as a class attibute
        self.int_to_str={i:s for s,i in vocab.items()} # flip key value pair, index first
        
    # Encode text to list of integers corresponding to vocabulary tokens
    def encode(self,text):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)', text) # tokenize input text
        preprocessed=[item.strip() for item in preprocessed if item.strip()] # remove whitespace
        ids=[self.str_to_int[s] for s in preprocessed] # generate list of integers for each token in text
        return ids
    
    # Extract text from encoded list
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        text=re.sub(r'\s+([,.?!"()\'])', r'\1', text) # remove spaces before punctuation
        return text
        

# %% Experiment with the tokenizer
tokenizer=SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids=tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

text="Hello.  Do you like tea?"
tokenizer.encode(text)
# %% Expand vocabulary to include unknown tokens and end of text tokens
vocab={token:integer for integer,token in enumerate(all_words)}
vocab['<eot>']=len(vocab) # add end of text token
vocab['<unk>']=len(vocab) # add unknown token


print(len(vocab)) # print new vocab size
# print the last 10 items in the vocabulary
for i, item in enumerate(list(vocab.items())[-5:]):
    print(f"{i+1}: {item}")

class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int=vocab # store vocab as a class attibute
        self.int_to_str={i:s for s,i in vocab.items()} # flip key value pair, index first
        
    # Encode text to list of integers corresponding to vocabulary tokens
    def encode(self,text):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)', text) # tokenize input text
        preprocessed=[item.strip() for item in preprocessed if item.strip()] # remove whitespace
        # replace unknown tokens with <unk>
        preprocessed=[s if s in self.str_to_int else '<unk>' for s in preprocessed]

        ids=[self.str_to_int[s] for s in preprocessed] # generate list of integers for each token in text
        return ids
    
    # Extract text from encoded list
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        text=re.sub(r'\s+([,.?!"()\'])', r'\1', text) # remove spaces before punctuation
        return text




# %% Test the tokenizer with the new vocabulary
text1="Hello, do you like tea?"
text2="In the sunlit terraces of the palace."
text=" <eot> ".join([text1,text2])
print(text)

tokenizer=SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
# %% Test teh detokenization
print(tokenizer.decode(tokenizer.encode(text)))

# %%
