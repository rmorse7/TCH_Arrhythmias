import h5py
import pywt
import numpy as np
import os
import re
import random
from glob import glob
from sklearn.model_selection import train_test_split
from ECG_feature_extraction import *
from ECG_preprocessing import *
from PPG_preprocessing import *
from os import listdir
from skimage.transform import resize

# ---------------------------- Data Generator----------------------------------
def split_train_val(train_dir, test_dir, val_size = 0.15, label = ['PJ','PO','PP','PS']):
    '''
    This function is used to split cwt imgs into training dataset and validation dataset based on val_size
    Arguments:
        train_dir: the directory for data folder
        test_dir:the directory for test data folder
        val_size: the percentage to split validation dataset
        label: list containing the label types
    Return:
        train_files_paths: the cwt image paths for training dataset
        val_files_paths: the cwt image paths for validation dataset
    '''
    train_files_paths,val_files_paths = [],[]
    for l in label:
        current_train_paths = glob(os.path.join(val_dir, '*_'+l+'.npy'))
        current_val_paths = glob(os.path.join(test_dir, '*_' + l + '.npy'))

        train_files_paths.extend(current_train_paths)
        val_files_paths.extend(current_val_paths)

    return train_files_paths,val_files_paths


def data_batch_generator(img_paths, batch_size, label_dict = {'PP':0,'PS':0,'PJ':1,'PO':1,'PJRP':1,},resiz=True):
    '''
    The function is used to return generator which contains the cwt images and the respective label
    Arguments:
        img_paths: the paths of cwt images
        batch_size: batch size
        label_dict: dictionary to indicate the label where 0 represents normal event and 1 represents abnormal event
        resize: boolean. if resize is true then resize the original images
    Return:
        Generator which contains the cwt images and the respective label
    '''
 
    while True:
        ids = np.random.choice(np.arange(len(img_paths)), batch_size)
        img_batch = []
        label_batch = []
        for id in ids:
            img_path = img_paths[id]
            label_str = re.findall(r"\d+_(\w+).npy",img_path)[0]
            label = label_dict[label_str]

            img = np.load(img_path)
            if resiz:
                img = resize(img, (32, 120, 2))
            img_batch.append(img)
            if label == 1:
                label_batch.append(np.array([1, 0]))
            else:
                label_batch.append(np.array([0, 1]))



        img_batch = np.array(img_batch)
        label_batch = np.array(label_batch)
        yield img_batch, label_batch

def get_train_valid_generator(train_dir, test_dir,batch_size, val_size = 0.15):
    '''
     Return data generator for training and validation
     Arguments:
         train_dir: the path of folder containing cwt features
         test_dir: the path of test data folder
         batch_size: batch size
         val_size: the percentage of original data to split for validation dataset
     Return:
         train_gen: data generator for training dataset
         valid_gen: data generator for validation dataset
         num_train: the total number of training data points
         num_val: the total number of validation data points
     '''
    train_img_paths, val_img_paths = split_train_val(train_dir, test_dir,val_size=val_size)

    num_train = len(train_img_paths)
    num_valid = len(val_img_paths)
    train_gen = data_batch_generator(img_paths=train_img_paths,batch_size=batch_size)
    valid_gen = data_batch_generator(img_paths=val_img_paths,batch_size=batch_size)
    return train_gen, valid_gen, num_train, num_valid


##########test\
#train_dir = 'resized_cwt_features_images/val'
#test_dir = 'resized_cwt_features_images/test'
#print(get_train_valid_generator(data_dir, test_dir,batch_size=4))