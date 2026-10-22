#%% Dependencies
import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

#%% Download and extract the dataset
'''
This dataset is a collection of SMS messages that have been labeled as either "spam" or "ham" (not spam). It is commonly used for training and testing machine learning models for spam detection.
'''
url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zippath="sms-spam-collection-dataset.zip"
extractedpath="sms_spam_collection"
datafilepath=Path(extractedpath)/"SMSSpamCollection.tsv" # tsv is tab separated values, which is the format of the dataset after extraction

#%% Download the dataset if it doesn't exist
def download_and_unzip_spam_data(url, zippath, extractedpath, datafilepath):
    if datafilepath.exists():
        print(f'{datafilepath} already exists.  Skipping download and extraction.')
        return
    # Download the zip file and extract it
    with urllib.request.urlopen(url) as response: # Open the URL and download the zip file
        with open(zippath,"wb") as outfile:
            outfile.write(response.read())
    with zipfile.ZipFile(zippath,"r") as zipref: # Open the zip file and extract its contents
        zipref.extractall(extractedpath)
    originalfilepath=Path(extractedpath)/"SMSSpamCollection" # The original file path after extraction
    os.rename(originalfilepath,datafilepath)
    print(f"File downloaded and saved as {datafilepath}")

download_and_unzip_spam_data(url, zippath, extractedpath, datafilepath)


# %% Import the dataset into a pandas DataFrame
df = pd.read_csv(datafilepath, sep='\t', header=None, names=['label', 'text'])
print(df.head())
print(df["label"].value_counts()) # Count the number of spam and ham messages

#%% Create a balanced dataset

def create_balanced_dataset(df):
    num_spam=df[df["label"]=="spam"].shape[0]
    ham_subset=df[df["label"]=="ham"].sample(num_spam,random_state=123)
    balanced_df=pd.concat([ham_subset,df[df['label']=='spam']])
    return balanced_df

balanced_df=create_balanced_dataset(df)
print(balanced_df['label'].value_counts())