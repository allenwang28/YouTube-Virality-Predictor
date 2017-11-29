import numpy as np
import pandas as pd
import os

from nltk.corpus import words

import re

# this script also removes non-english words by comparing the number of english to non-english words 
# in the title. Remove samples with less than ENGLISH_PERCENT_THRESH in title.
ENGLISH_PERCENT_THRESH = 0.6

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR_PATH = os.path.join(ROOT_DIR, '..', 'images')
DATA_PATH = os.path.join(ROOT_DIR, '..', 'data')

english_words_set = set(words.words())

def get_percent_english(s):
    tokens = re.split(' ', s)
    num_english = 0

    for token in tokens:
        if token in english_words_set:
            num_english += 1
    return num_english / float(len(tokens))
    
def clean(s):
    if len(s) > 3:
        if s[0] == 'b':
            if s[1] == '\'' or s[1] == '\"':
                if s[-1] == '\'' or s[-1] == '\"':
                    return s[2:-1]
    return s

def remove_escape_chars(s):
    s = s.replace('\\n', ' ')
    return s.replace('\\', '')

cleaned_subset = os.path.join(DATA_PATH, 'cleaned_subset.csv')
nsfw_subset = os.path.join(DATA_PATH, 'virality_nsfw.csv')

data_loc = os.path.join(DATA_PATH, 'data.csv')


df = pd.read_csv(cleaned_subset)
df['Title'] = df['Title'].apply(clean)
df['Description'] = df['Description'].apply(clean)

df['Description'] = df['Description'].apply(remove_escape_chars)



df_nsfw = pd.read_csv(nsfw_subset)
df_nsfw['Id'] = df_nsfw['video_id']
df_nsfw = df_nsfw[['Id', 'nsfw_score']]

# Some have invalid Ids - drop them
df = df[df['Id'] != '#NAME?']

# Drop the duplicated indices
df.drop_duplicates(subset='Id', keep='last')

# Remove entries where it's non-english

print ('DF shape: {0}'.format(df.shape))

df = pd.merge(df, df_nsfw, on='Id')

#df['PercentEnglishInTitle'] = df['Title'].apply(get_percent_english)

print ('Merged DF shape: {0}'.format(df.shape))



df.to_csv(data_loc, index=False)
