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


def load_event_cwt_images(save_path, patient_folder_path, excel_file_path, excel_sheet_name='PJ', fs=240,leave_out_path = 'resized_cwt_features_images/22/test/'):
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
    labelevent = pd.read_excel(excel_file_path, sheet_name=excel_sheet_name)
    count = 1


    for _, record in labelevent.iterrows():

        label_record = record.tolist()
        patient_id, event_start_time, event_end_time = label_record
        patient_file_path = patient_folder_path + '/' + str(int(patient_id))

        for block_file in listdir(patient_file_path):

            # trying to find the ecg signal and ppg signal during the label event time
            block_path = patient_file_path + '/' + block_file
            all_signals = h5py.File(block_path, 'r')
            signals_keys = set(all_signals.keys())
            block_start_time, block_end_time = all_signals['time'][0], all_signals['time'][-1]
            if block_start_time <= event_start_time <= event_end_time <= block_end_time:
                start_index = int((event_start_time - block_start_time) * fs)
                end_index = int((event_end_time - block_start_time) * fs)

                # event_time = all_signals['time'][start_index:end_index +1]
                ecg, ppg = None, None
                if 'GE_WAVE_ECG_2_ID' in signals_keys:
                    ecg = all_signals['GE_WAVE_ECG_2_ID'][start_index:end_index + 1]
                if 'GE_WAVE_SPO2_WAVE_ID' in signals_keys:
                    ppg = all_signals['GE_WAVE_SPO2_WAVE_ID'][start_index:end_index + 1]

                if ppg is None or ecg is None: continue
                # ECG signal preprocessing for denoising and R-peak detection
                R_peak_index, ecg_denoise = ecg_preprocessing_final(
                    ecg)  # the location of R_peak during the label event
                ppg_denoise = PPG_denoising(ppg)
                ## extract cwt features for ecg signal and ppg signal
                ecg_cwt = compute_cwt_features(ecg_denoise, R_peak_index, scales=np.arange(1, 129), windowL=-240,
                                               windowR=240, wavelet='morl')
                ppg_cwt = compute_cwt_features(ppg_denoise, R_peak_index, scales=np.arange(1, 129), windowL=-240,
                                               windowR=240, wavelet='coif')

                if len(ecg_cwt) != len(ppg_cwt):
                    raise Exception("The beat length is not correct!!! Please check!")
                if not ecg_cwt or not ppg_cwt: continue

                for i in range(len(ecg_cwt)):
                    combined = np.stack((ecg_cwt[i],ppg_cwt[i]),axis=-1)
                    if patient_id==22:  #the patient id to be left out
                        np.save(leave_out_path+str(count)+'_'+excel_sheet_name,combined)
                    else:
                        np.save(save_path + str(count) + '_' + excel_sheet_name, combined)

                    count += 1

    return


def load_cwt_files(patient_folder_path, excel_file_path, save_path, label_type=['PJ', 'PJRP', 'PO', 'PP', 'PS', 'PVC'],leave_out_path='resized_cwt_features_images/22/test'):

    '''
    Implements function load_event_cwt_images to generate cwt features and then save into a specific folder
    Arguments:
        patient_folder_path: the path of the folder which save the patients' waveforms
        excel_file_path: the path of the excel file which contains the label events
        save_path: the folder path to save cwt features
        label_types: a default list containing labels
        leave_out_path: the folder path to save patient data to be tested
    Returns:
    no return
    '''
    for label in label_type:
        load_event_cwt_images(save_path, patient_folder_path, excel_file_path, excel_sheet_name=label,leave_out_path='resized_cwt_features_images/22/test/')


load_cwt_files(patient_folder_path='C:/Users/tianq/Documents/dsci535/Waveform Data', excel_file_path='Labelled_Events.xlsx',
               save_path='resized_cwt_features_images/22/val/',leave_out_path='resized_cwt_features_images/22/test/')
