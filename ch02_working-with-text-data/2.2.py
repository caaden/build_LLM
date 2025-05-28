#%% Download text 
# Note, the raw.githubusercontent url to fetch the raw implementation
import urllib.request
url="https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path="the-verdict.txt"
urllib.request.urlretrieve(url,file_path)
print(f"Downloaded to {file_path}")

#%% See the text
with urllib.request.urlopen(url) as response:
    text = response.read().decode('utf-8')
    print(text)

#%% Read a few lines
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
print("Total number of characters:",len(raw_text))
print(raw_text[:99])

#%% Use Python regular expressions to split a simple sentence into tokens
import re
text="Hello world. This, is a test."
result=re.split(r'(\s)',text)
print(f'Split sentence: {result}')
# Remove white space
result=[item for item in result if item.strip()]
print(f'Removed whitespace: {result}')

#%% Expand tokenization scheme
text = "Hello, world. Is this-- a test?" 
result = re.split(r'([,.:;?_!"()\']|--|\s)', text) 
result = [item.strip() for item in result if item.strip()] 
print(f'Expanded takenization scheme: {result}')

#%%
