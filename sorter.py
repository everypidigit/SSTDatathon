import pandas as pd
import os

labels_df = pd.read_csv('labels.csv')
train_path = './data/train'
dir = os.listdir(train_path)

for x in range(len(dir)-1):
    if labels_df['name'][x] in dir:
        if labels_df['label'][x] == 0:
            os.rename(train_path+'/{name}'.format(name = labels_df['name'][x]), (train_path+'/0/{name}'.format(name = labels_df['name'][x])))
        elif labels_df['label'][x] == 1:
            os.rename(train_path+'/{name}'.format(name = labels_df['name'][x]), (train_path+'/1/{name}'.format(name = labels_df['name'][x])))