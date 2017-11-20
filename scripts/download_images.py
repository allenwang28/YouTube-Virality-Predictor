import os
import requests # pip install requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import progressbar # pip install progressbar2 

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR_PATH = os.path.join(ROOT_DIR, '..', 'images')
DATA_PATH = os.path.join(ROOT_DIR, '..', 'data')

if not os.path.exists(IMG_DIR_PATH):
    os.makedirs(IMG_DIR_PATH)


script_path = os.path.join(DATA_PATH, 'subset.csv')


df = pd.read_csv(script_path)
df = df[['Id', 'Thumbnail Default']]

# Some have invalid Ids - drop them
df = df[df['Id'] != '#NAME?']

# Drop the duplicated indices
df.drop_duplicates(subset='Id', keep='last')

print ('Total: {0} images'.format(df.shape[0]))

bar = progressbar.ProgressBar()

for index, row in bar(df.iterrows()):
    video_id = row[0]
    link = row[1][2:-1]
    file_name = os.path.join(IMG_DIR_PATH, '{0}.jpg'.format(video_id))

    f = open(file_name, 'wb')
    f.write(requests.get(link).content)
    f.close()
