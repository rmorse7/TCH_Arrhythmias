import os
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import bisect
from ECG_preprocessing import *
from ECG_feature_extraction import *
from PPG_preprocessing import *
from PPG_feature_extraction import *
import csv

def load_label_event(patient_folder_path,excel_file_path,excel_sheet_name='summary',fs = 240,
                     ecg_features = {'RR':True,'Wavelet':{'family':'db1','level':3}},
                     ppg_features = {'st':True, 'dt':True, 'half_width':True,'two_third_width':True},save_file_path = None):
    # this function is to prepare the database with ECG feature and PPG feature ready for traning

    # load label event from label event excel file
    labelevent = pd.read_excel(excel_file_path,sheet_name=excel_sheet_name)
    save_file_name = 'dataset_for_modeling.csv'
    save_file = save_file_path+'/'+save_file_name
    with open(save_file,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['pre_R','post_R','local_R','st','dt','half_width','two_third_width','one_beat_ecg_rr_wt_dpb_13', 'one_beat_ecg_rr_wt_dpb_14', 'one_beat_ecg_rr_wt_dpb_15', 'one_beat_ecg_rr_wt_dpb_16', 'one_beat_ecg_rr_wt_dpb_17', 'one_beat_ecg_rr_wt_dpb_18', 'one_beat_ecg_rr_wt_dpb_19', 'one_beat_ecg_rr_wt_dpb_20', 'one_beat_ecg_rr_wt_dpb_21', 'one_beat_ecg_rr_wt_dpb_22', 'one_beat_ecg_rr_wt_dpb_23', 'one_beat_ecg_rr_wt_dpb_24', 'one_beat_ecg_rr_wt_dpb_25', 'one_beat_ecg_rr_wt_dpb_26', 'one_beat_ecg_rr_wt_dpb_27', 'label', 'health'])

        for index,record in labelevent.iterrows():

            label_record = record.tolist()
            patient_id,event_start_time,event_end_time,label = label_record
            patient_file_path = patient_folder_path+'/'+str(patient_id)


            for block_file in listdir(patient_file_path):

                # trying to find the ecg signal and ppg signal during the label event time
                block_path = patient_file_path+'/'+block_file
                all_signals = h5py.File(block_path, 'r')
                signals_keys = set(all_signals.keys())
                block_start_time,block_end_time = all_signals['time'][0],all_signals['time'][-1]
                if block_start_time <= event_start_time <= event_end_time <= block_end_time:
                    start_index = int((event_start_time-block_start_time)*fs)
                    end_index = int((event_end_time-block_start_time)*fs)

                    event_time = all_signals['time'][start_index:end_index +1]
                    ecg, ppg = None, None
                    if 'GE_WAVE_ECG_2_ID' in signals_keys:
                        ecg = all_signals['GE_WAVE_ECG_2_ID'][start_index:end_index +1]
                    if 'GE_WAVE_SPO2_WAVE_ID' in signals_keys:
                        ppg = all_signals['GE_WAVE_SPO2_WAVE_ID'][start_index:end_index +1]
                    # print("loaded ppg: ", ppg)

                    if (ppg is None) or (ecg is None) or(not ppg.any or not ecg.any): break
                    # ECG signal preprocessing for denoising and R-peak detection
                    R_peak_index,ecg_denoise = ecg_preprocessing_final(ecg)  # the location of R_peak during the label event
                    num_beats = len(R_peak_index)                # the total number of beats during the label event

                    R_peak = []                                  # the time when the R-peak appears
                    for i in range(num_beats):
                        R_peak.append(event_time[R_peak_index[i]])
                    
                    if ecg_features['RR']:
                        ecg_RR_feature = compute_RR_intervals(R_peak)

                    if ecg_features['Wavelet']:
         
                        family = ecg_features['Wavelet']['family']
                        level = ecg_features['Wavelet']['level']
                        ecg_wt_feature = compute_wavelet_features(ecg_denoise,R_peak_index,windowL=-60,windowR=60,wavelet = family,level = level)
                  
                    
                    if ppg.any:
                        # PPG signal preprocessing for denoising 
                        # print("un-denoised ppg", ppg)
                        ppg = PPG_denoising(ppg)
                        # PPG signal feature extraction
                        ppg_extracted_features = PPG_feature_extraction(ppg,ppg_features,R_peak_index)
                        

                    for i in range(num_beats):
                        #print("RR Feature: ", ecg_RR_feature.pre_R, "Num beats: ", num_beats, "i: ", i)
                        temp = []
                        if ecg_features['RR']:
                            temp.append(ecg_RR_feature.pre_R[i])
                            temp.append(ecg_RR_feature.post_R[i])
                            temp.append(ecg_RR_feature.local_R[i])
                            temp.extend(ppg_extracted_features[i])
                        if ecg_features['Wavelet']:
                            temp.extend(ecg_wt_feature[i])
                        temp.append(label)
                        if label == 'PP' or label == 'PS':
                            health = 0
                        else:
                            health = 1
                        temp.append(health)
                        writer.writerow(temp)



                    break
                    

                else: continue



#############test

#############test

load_label_event('Waveform Data', 'Labelled_Events.xlsx', excel_sheet_name='Summary', save_file_path='Modeling Data')