import h5py
import pywt
import numpy as np
import os
import random
from glob import glob
# from sklearn.model_selection import train_test_split
from ECG_feature_extraction import *
from ECG_preprocessing import *
from PPG_preprocessing import *
from os import listdir
import pandas as pd
# import keras
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import History 
# from tensorflow.keras.optimizers import Adam



def load_event_cwt_images(save_path,patient_folder_path,excel_file_path,excel_sheet_name='PJ',fs=240):
    '''
    load cwt features 
    input:
        save_path: it is the folder path to save these np.array files 
        patient_folder_path: it is the folder containing different patients data
        excel_file_path: the path for labelled event excel
        excel_sheet_name: it is the labelled event that you plan to work with. Basically save the same events into a folder call the same name as the excel_sheet_name
        fs: sampling frequncy

    output:
        no return value 
        but you can check the saved file based on your save_path 
    '''
    labelevent = pd.read_excel(excel_file_path,sheet_name=excel_sheet_name)
    count = 1
    # save_path = save_path+excel_sheet_name+'/'

    for _,record in labelevent.iterrows():

        label_record = record.tolist()
        patient_id,event_start_time,event_end_time = label_record
        patient_file_path = patient_folder_path+'/'+str(int(patient_id))


        for block_file in listdir(patient_file_path):

            # trying to find the ecg signal and ppg signal during the label event time
            block_path = patient_file_path+'/'+block_file
            all_signals = h5py.File(block_path, 'r')
            signals_keys = set(all_signals.keys())
            block_start_time,block_end_time = all_signals['time'][0],all_signals['time'][-1]
            if block_start_time <= event_start_time <= event_end_time <= block_end_time:
                start_index = int((event_start_time-block_start_time)*fs)
                end_index = int((event_end_time-block_start_time)*fs)

                #event_time = all_signals['time'][start_index:end_index +1]
                ecg, ppg = None, None
                if 'GE_WAVE_ECG_2_ID' in signals_keys:
                    ecg = all_signals['GE_WAVE_ECG_2_ID'][start_index:end_index +1]
                if 'GE_WAVE_SPO2_WAVE_ID' in signals_keys:
                    ppg = all_signals['GE_WAVE_SPO2_WAVE_ID'][start_index:end_index +1]
                # print("loaded ppg: ", ppg)

                if ppg is None or ecg is None: continue
                # ECG signal preprocessing for denoising and R-peak detection
                R_peak_index,ecg_denoise = ecg_preprocessing_final(ecg)  # the location of R_peak during the label event
                ppg_denoise = PPG_denoising(ppg)
                ## extract cwt features for ecg signal and ppg signal
                ecg_cwt = compute_cwt_features(ecg_denoise,R_peak_index,scales = np.arange(1,129),windowL=-240,windowR=240,wavelet = 'morl')
                ppg_cwt = compute_cwt_features(ppg_denoise,R_peak_index,scales = np.arange(1,129),windowL=-240,windowR=240,wavelet = 'coif')

                if len(ecg_cwt)!=len(ppg_cwt): 
                    raise Exception("The beat length is not correct!!! Please check!")
                if not ecg_cwt or not ppg_cwt: continue

                for i in range(len(ecg_cwt)):
                    combined = np.stack((ecg_cwt[i],ppg_cwt[i]),axis=-1)
                    np.save(save_path+str(count)+'_'+excel_sheet_name,combined)
                    # temp = ecg_cwt[i]
                    # temp = np.reshape(temp,(128,480,1))
                    # np.save(save_path+str(count)+'_'+excel_sheet_name,temp)
                    count+=1

    return





def load_cwt_files(patient_folder_path,excel_file_path,save_path,label_type= ['PJ','PJRP','PO','PP','PS','PVC']):

    '''
    Implements function load_event_cwt_images to generate cwt features and then save into a specific folder
    Arguments:
        patient_folder_path: the path of the folder which save the patients' waveforms
        excel_file_path: the path of the excel file which contains the label events
        save_path: the folder path to save cwt features
        label_types: a default list containing labels

    Returns:
    no return

    '''

    for label in label_type:
        load_event_cwt_images(save_path,patient_folder_path,excel_file_path,excel_sheet_name=label)

load_cwt_files(patient_folder_path='I:/COMP549/data',excel_file_path='I:/COMP549/events/Labelled_Events.xlsx',save_path='I:/COMP549/cwt_features_images_ecg/')


# train_size = 20000
# test_size = 4000
# train_data_dwt = np.ndarray(shape = (train_size,122,1,1))
# test_data_dwt = np.ndarray(shape = (test_size,122,1,1))
# y_train = np.ndarray(shape = (train_size))
# y_test = np.ndarray(shape = (test_size))

# load_label_event(patient_folder_path,excel_file_path,train_data_dwt,y_train,test_data_dwt,y_test,excel_sheet_name='PJ')
# load_label_event(patient_folder_path,excel_file_path,train_data_dwt,y_train,test_data_dwt,y_test,excel_sheet_name='PS',start_train_row=10000,start_test_row=2000)

# history = History()
 
# img_x = 122
# img_y = 1
# img_z = 1
# input_shape = (img_x, img_y, img_z)
 
# num_classes = 2
# batch_size = 16
# num_classes = 2
# epochs = 10
 
# x_train = train_data_dwt.astype('float32')
# x_test = test_data_dwt.astype('float32')
 
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
 
 
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 1), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 2)))
# model.add(Conv2D(64, (5, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 1)))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
 
# model.compile(loss=tf.keras.losses.categorical_crossentropy,
#               optimizer= Adam(lr=3e-4),
#               metrics=['accuracy'])
 
 
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           callbacks=[history])
 
# train_score = model.evaluate(x_train, y_train, verbose=0)
# print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
# test_score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

# print()




# P1_B4_path ='I:/COMP549/data/1/Reference_idx_1_Time_block_4.h5'

# P1_B4 = h5py.File(P1_B4_path, 'r')
# ecg = P1_B4['GE_WAVE_ECG_2_ID'][:2400]
# ppg = P1_B4['GE_WAVE_SPO2_WAVE_ID'][:2400]

# R_peak_index,ecg_afterp = ecg_preprocessing_final(ecg,fs=240)
# ppg = PPG_denoising(ppg)
# wavelet_cofficients_ecg = compute_wavelet_features(ecg_afterp,R_peak_index,windowL=-240,windowR=240,wavelet = 'db1',level = 3)
# wavelet_cofficients_ppg = compute_wavelet_features(ppg,R_peak_index,windowL=-240,windowR=240,wavelet = 'db2',level = 3)

# print()