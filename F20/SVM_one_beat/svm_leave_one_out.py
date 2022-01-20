
import os
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt

# from ECG_preprocessing import *
# from ECG_feature_extraction import *
# from PPG_preprocessing import *
# from PPG_feature_extraction import *
# from data_loading import *
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


def build_svm_leave_one_out(data_path,feature_list):
    """
    This function calculate the leave-one-out result for svm model
    :param: path_to_data: Path to access excel file containing our data
    :param: num_features: Integer denoting how many features to extract
    Steps:
    1) Load dataset
    2) Split data into training and testing subsets
    3) Classifier Training using SVM modeling
    4) Step Check accuracy of model on testing subset

    """



    data = pd.read_csv(data_path, header=0, low_memory=False)


    unique_patients = set(data['patientID'])
    print(unique_patients)
    print("Unique patients" ,len(unique_patients))

    scores = []
    false_pos = []

    for i in unique_patients:
        print('Leave', i, 'patient out cross validation')
        train_data = data[data.patientID != i]
        test_data = data[data.patientID == i]
        feature_tr = train_data[feature_list]
        feature_ts = test_data[feature_list]
        health_tr = train_data[['health']]
        health_ts = test_data[['health']]


    ##Potential way to handle missing data found at:
    ## https://stackoverflow.com/questions/30317119/classifiers-in-scikit-learn-that-handle-nan-null
        imp = SimpleImputer(missing_values=0, strategy='median')
        imp = imp.fit(feature_tr)

        feature_tr = imp.transform(feature_tr)
        feature_ts = imp.transform(feature_ts)

        health_tr = health_tr.values.flatten()
        health_ts = health_ts.values.flatten()

        model = SVC()
        print('start fitting')
        try:
            model.fit(feature_tr, health_tr)

            predict_arrhythmia = model.predict(feature_ts)
            scores.append(accuracy_score(health_ts, predict_arrhythmia))
            print("Accuracy for leave patient", i," out: ", accuracy_score(health_ts, predict_arrhythmia))
            false_positives = 0
            num_true_negatives = 0

            num_arrhythmias = 0
            num_arrhythmias_accurately_detected = 0

            # How many arrhythmias do we accurately catch? Need number of total arrhythmias and arhythmias that we accurately detect
            for beat in range(len(predict_arrhythmia)):
                ##If there is an arrhythmia and we catch is
                if health_ts[beat] == 1 and predict_arrhythmia[beat] == 1:
                    num_arrhythmias += 1
                    num_arrhythmias_accurately_detected += 1
                elif health_ts[beat] == 1:
                    num_arrhythmias += 1
            if num_arrhythmias:
                print("Proportion of Arrhythmias that we accurately detect: ",
                    num_arrhythmias_accurately_detected / num_arrhythmias)
            else:
                print('No arrhythmias found')

            for beat in range(len(predict_arrhythmia)):
                # If there isn't an arrythmia but we predict one, we have a false positive
                if health_ts[beat] == 0 and predict_arrhythmia[beat] == 1:
                    false_positives += 1
                if health_ts[beat] == 0:
                    num_true_negatives += 1
            if num_true_negatives != 0 or false_positives != 0:
                print("False positive rate: ", false_positives / (false_positives + num_true_negatives))
                false_pos.append(false_positives / (false_positives + num_true_negatives))
            else:
                print('No healthy patient in test set')
        except:
            print('Leave this patient out could lead to training failure: ',i)
            scores.append(0)


    return np.mean(scores),np.mean(false_pos)

#specify local path to cleaned data
data_path = 'Modeling Data/colab.csv'
sheet_name = "dataset_for_modeling"
#specify features to be trained on
feature_list = [ 'pre_R' ,'local_R' ,'post_R' ,'st' ,'dt' ,'half_width' ,'two_thirds']

##Names for wavelet features
wavelet_feature_names = ['one_beat_ecg_rr_wt_dpb_' + str(i) for i in range(13, 28)]

feature_list.extend(wavelet_feature_names)

#uncomment to run and test code
#acc,fals = build_svm_leave_one_out(data_path, feature_list)
#print('The accuracy of leave one out validation is', acc)
#print('False positive rate is', fals)

