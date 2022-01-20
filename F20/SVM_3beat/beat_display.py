import os
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from scipy.signal import resample
import csv
from matplotlib import pyplot as plt
from HeatMap import get_histogram, plot_histogram

def display_segments(model_out_path,patient_folder_path,fs=240):
    '''
    Input: model_out_path: path to model output
           patient_folder: path to folder w/ h5py patient data
    Output: plot of segment classifcations
    '''

    modelout = pd.read_csv(model_out_path)
    patient_id=0;block_id=0 #current patient and block (done this way so don't have to constantly load new h5py files)
    #desired length of signal (after resampling)
    #these will be saved as matrices
    truepos = []; falsepos = []; trueneg = []; falseneg = []
    #maximum amplitude in each class (used for plotting)
    tpm = 2; fpm = 2; tnm = 2; fnm = 2

    for i, segment in modelout.iterrows():
        if round(segment['patient']!=patient_id) or round(segment['block'])!=block_id: #need new h5py file
            patient_id = int(segment['patient'])
            block_id = int(segment['block'])
            patient_folder = patient_folder_path + '/' + str(patient_id)
            block_file = 'Reference_idx_' + str(patient_id) + '_Time_block_' + str(block_id) + '.h5'
            path = patient_folder + '/' + block_file
            all_signals = h5py.File(path, 'r')
            signals_keys = set(all_signals.keys())
            block_start_time, block_end_time = all_signals['time'][0], all_signals['time'][-1]

        #changes for each segment
        start_time = float(segment['start_time']); start_index = int(round((start_time-block_start_time)*fs))
        end_time = float(segment['end_time']); end_index = int(round((end_time-block_start_time)*fs))

        #event_time = all_signals['time'][start_index:end_index + 1]
        #extracting signals
        ecg, ppg = None, None
        if 'GE_WAVE_ECG_2_ID' in signals_keys:
            ecg = all_signals['GE_WAVE_ECG_2_ID'][start_index:end_index + 1]
        #not extracing ppg as of now
        if 'GE_WAVE_SPO2_WAVE_ID' in signals_keys:
            ppg = all_signals['GE_WAVE_SPO2_WAVE_ID'][start_index:end_index + 1]

            #shifting ecg signal to deal with peaks not being aligned
            midpeak = np.argmax(ecg)
            shift = 170 - midpeak
            sign = np.sign(shift); shift = abs(shift)
            #shift = min(shift,100)
            zero_pad = np.zeros(shift)

            if sign > 0:
                ecg = np.hstack((zero_pad,ecg))
                start_index = start_index - shift
            else:
                ecg = np.hstack((ecg,zero_pad))
                end_index = end_index + shift
        #ecg = all_signals['GE_WAVE_ECG_2_ID'][start_index:end_index + 1]

        #lengthening event_time to be same length as shifted signal
        event_time = all_signals['time'][start_index:end_index + 1]

        seg = np.transpose(np.vstack((event_time,ecg)))
        health = segment['health']
        modelh = segment['health_predict']

        #assigning each beat to its classification category
        emax = max(ecg)
        if health*modelh == 1: #true positive
            truepos.append(seg)
            if emax > tpm:
                tpm = emax
        elif health==1: #false negative
            falseneg.append(seg)
            if emax > fnm:
                fnm = emax
        elif modelh==1: #false positive
            falsepos.append(seg)
            if emax > fpm:
                fpm = emax
        else: #true neagative
            trueneg.append(seg)
            if emax > tnm:
                tnm = emax
    #creating heatmap of results as defined in HeatMap.py
    slen = 255
    tph = get_histogram(truepos,resample_size=slen, num_bins_y=50, amp=tpm)
    fph = get_histogram(falsepos, resample_size=slen, num_bins_y=50, amp=fpm)
    tnh = get_histogram(trueneg, resample_size=slen, num_bins_y=50, amp=tnm)
    fnh = get_histogram(falseneg, resample_size=slen, num_bins_y=50, amp=fnm)
    plt.figure(1); plot_histogram(fnh, "False Negative")
    plt.figure(2); plot_histogram(tnh, "True Negative")
    plt.figure(3); plot_histogram(tph, "True Positive")
    plt.figure(4); plot_histogram(fph, "False Positive")
    return None

display_segments('ModelOut/testout.csv','Waveform Data')





