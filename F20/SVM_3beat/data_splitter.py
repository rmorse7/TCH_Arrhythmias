import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import csv
from collections import defaultdict, Counter

def data_count(model_data_path):
    '''
    what percent of data does each patient make up (for leave-one-out classification)
    '''
    data = pd.read_csv(model_data_path)
    count = Counter(data['patient'].to_numpy())
    total = len(data['patient'])
    for p,n in count.items():
        count[p] = n / total
    return count

def data_split(data,train_patients,test_patients):
    '''
    split data along patient lines (for leave-one-out classifcation)
    Inputs: data to be split and patient subsets to split along
    Outputs: train, test datasets
    '''
    train = data.loc[data['patient'].isin(train_patients)]
    test = data.loc[data['patient'].isin(test_patients)]
    return train,test

#allp = [1,2,3,4,7,8,13,14,15,16,17,18,19,20,22]

#total patient proportion info
count = data_count('ModelData/Data_Cleaned_1bet_ecg.csv'); print(count)

#breakdown of individual patient's health-arrhythmia split
datain = pd.read_csv('ModelData/Data_Cleaned.csv', low_memory= False)
testin = datain.loc[datain['patient']==1]; print(sum(testin['health']) / len(testin['health']))
