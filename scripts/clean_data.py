import os
import requests # pip install requests
import re
import numpy as np
import pandas as pd

import progressbar # pip install progressbar2

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, '..', 'data')


script_path = os.path.join(DATA_PATH, 'subset.csv')
new_clean_subset = os.path.join(DATA_PATH,'cleaned_subset.csv')

data = pd.read_csv(script_path)
df = pd.read_csv(script_path)


print ('Total: {0} images'.format(df.shape[0]))


for index, row in df.iterrows():
    link = df['Thumbnail Default'][index]
    if(df['Id'][index] == '#NAME?'):
        df['Id'][index] = re.search(r'vi/(.*?)/default', link).group(1)

# Drop the duplicated indices
df = df.drop_duplicates(subset='Id', keep='last')

#write to clean_csv
df.to_csv(new_clean_subset, index =False);

print ('Total: {0} images'.format(df.shape[0]))