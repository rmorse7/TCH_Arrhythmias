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
    return 1/(1+np.exp(-m))

def build_svm(path_to_data, sheet_name, feature_list):
    """

    :param: path_to_data: Path to access excel file containing our data
    :param: num_features: Integer denoting how many features to extract

    Steps:
    1) Load dataset
    2) Split data into training and testing subsets
    3) Classifier Training using SVM modeling
    4) Step Check accuracy of model on testing subset
    5) Export SVM Classifier model
    """
    
    #For plotting
    #start, end = 67500, 92500
    start,end = 60000,100000

    data = pd.read_csv(path_to_data, low_memory= False)#[start:end]
    #allp = [1,2,3,4,7,8,13,14,15,16,17,18,19,20,22]
    #leave-one-out training
    trainp = [1,2,3,4,8,13,14,15,16,17,18,19,20,22]
    testp = [7]
    pats = trainp + testp
    #data = data.loc[data['patient'].isin(pats)]

    #extract training/testing data along patient lines for leave-one-out training
    train = data.loc[data['patient'].isin(trainp)]; test = data.loc[data['patient'].isin(testp)]
    #if not doing leave-one-out (random train/test split)
    #train, test, health_tr, health_ts = train_test_split(data, data[['health']], test_size=0.15)

    info_tr = train.iloc[:,0:4].to_numpy(); info_ts = test.iloc[:,0:4].to_numpy()
    feature_tr = train.iloc[:,4:-2].to_numpy(); feature_ts = test.iloc[:,4:-2].to_numpy()
    health_tr = train['health'].to_numpy(); health_ts = test['health'].to_numpy()

    #if not doing leave-one-out (random train/test split)
    #health_tr = health_tr.values.flatten() #.values in middle
    #health_ts = health_ts.values.flatten() #.values in middle

    model = SVC()
    model.fit(feature_tr, health_tr)

    predict_train = model.predict(feature_tr)
    predict_arrhythmia = model.predict(feature_ts)

    #writing output to model output files
    train_file = 'trainout.csv'
    test_file = 'testout.csv'
    with open(train_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['patient', 'block', 'start_time', 'end_time', 'pre_R', 'post_R', 'local_R', 'wt_1', 'label', 'health','health_predict'])
        for i in range(len(health_tr)):
            temp = np.concatenate((info_tr[i,:],feature_tr[i,:]))
            #temp = feature_tr[i,:]
            temp = np.append(temp, [health_tr[i], predict_train[i]])
            writer.writerow(temp)
    with open(test_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['patient', 'block', 'start_time', 'end_time', 'pre_R', 'post_R', 'local_R', 'wt_1', 'label', 'health','health_predict'])
        for i in range(len(health_ts)):
            temp = np.concatenate((info_ts[i, :], feature_ts[i, :]))
            #temp = feature_ts[i,:]
            temp = np.append(temp, [health_ts[i], predict_arrhythmia[i]])
            writer.writerow(temp)


    #checking accuracy and sensitivity/specificity
    print("Train Accuracy: ", accuracy_score(health_tr, predict_train))
    print("Accuracy: ", accuracy_score(health_ts, predict_arrhythmia))

    ##Want to create two csv's, one for training and one for testing with all the columns + predictions
    #First csv feature_tr + health_tr
    #Second csv feature_ts + health_ts + predict_arrhythmia

    # training_dataset = np.append(feature_tr, health_tr)

    false_positives = 0
    num_true_negatives = 0

    num_arrhythmias = 0
    num_arrhythmias_accurately_detected = 0

    #How many arrhythmias do we accurately catch? Need number of total arrhythmias and arhythmias that we accurately detect
    for beat in range(len(predict_arrhythmia)):
        ##If there is an arrhythmia and we catch is
        if health_ts[beat] == 1 and predict_arrhythmia[beat] == 1:
            num_arrhythmias += 1
            num_arrhythmias_accurately_detected += 1
        elif health_ts[beat] == 1:
            num_arrhythmias += 1

    print("Proportion of Arrhythmias that we accurately detect: ", num_arrhythmias_accurately_detected/num_arrhythmias)
    print("Proportion of Arrhythmias in test set: ", num_arrhythmias / len(health_ts))

    for beat in range(len(predict_arrhythmia)):
        #If there isn't an arrythmia but we predict one, we have a false positive
        if health_ts[beat] == 0 and predict_arrhythmia[beat] == 1:
            false_positives += 1
        if health_ts[beat] == 0:
            num_true_negatives += 1
    
    print("False positive rate: ", false_positives/(false_positives+num_true_negatives))


    #SVM PCA decision boundary plotting
    # pca = PCA(n_components = 2)
    # feature_tr_plotting = pca.fit_transform(feature_tr)
    #
    # transformed_feature_tr_plotting = sigmoid(feature_tr_plotting)
    #
    # print("Explained Variance: ", pca.explained_variance_)
    # print("Explained Variance Ratio: ", pca.explained_variance_ratio_)
    #
    # model.fit(transformed_feature_tr_plotting, health_tr)
    #
    # ax = plot_decision_regions(transformed_feature_tr_plotting, health_tr, clf=model, legend=2)
    #
    # plt.xlabel('Sigmoid transformed PCA Feature #1', size=14)
    # plt.ylabel('Sigmoid transformed PCA Feature #2', size=14)
    # plt.title('SVM Decision Region Boundary', size=16)
    #
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, ['Healthy', 'Arrhythmia'], framealpha = 0.3, scatterpoints = 1)
    #
    # plt.show()

    # np.savetxt('training_dataset.csv', training_dataset)
    # np.savetxt('testing_dataset.csv', testing_dataset, header = "Actual, Predicted")

    return model, predict_arrhythmia, feature_tr, feature_ts, health_tr, health_ts


data_path = 'ModelData/Data_Cleaned_1beat_ecg.csv'
sheet_name = "Data_Cleaned_1beat_ecg"
#not currently used
feature_list = ['pre_R_median', 'pre_R_IQR', 'post_R_median','post_R_IQR', 'local_R_median', 'local_R_IQR', 'st_median',
                'st_IQR', 'dt_median', 'dt_IQR','half_width_median', 'half_width_IQR', 'two_third_width_median', 'two_third_width_IQR']


model = build_svm(data_path, sheet_name, feature_list)[0]

print("Model: ", model)