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
import xlrd
from sklearn.impute import SimpleImputer

def load_label_event(patient_folder_path,excel_file_path,excel_sheet_name,windowsize,fs = 240,
                     ecg_features = {'RR':True,'Wavelet':{'family':'db1','level':3}},
                     ppg_features = {'all':True,'st':True, 'dt':True, 'half_width':True,'two_third_width':True},save_file_path = None):
    # this function is to prepare the database with ECG feature and PPG feature ready for training

    # load label event from label event excel file
    labelevent = pd.read_excel(excel_file_path,sheet_name=excel_sheet_name)
    save_file_name = 'dataset_for_modeling.csv'
    save_file = save_file_path+'/'+save_file_name
    #writing header to file
    with open(save_file,'w') as f:
        writer = csv.writer(f)
        if windowsize==1:
            writer.writerow(
                ['patient', 'block', 'start_time', 'end_time', 'pre_R', 'post_R', 'local_R', 'st', 'dt', 'half_width',
                 'two_third_width'])
        else:
            writer.writerow(
                ['patient', 'block', 'start_time', 'end_time', 'pre_R_median', 'pre_R_IQR', 'post_R_median',
                 'post_R_IQR', 'local_R_median', 'local_R_IQR', 'st_median', 'st_IQR', 'dt_median', 'dt_IQR',
                 'half_width_median', 'half_width_IQR', 'two_third_width_median', 'two_third_width_IQR'])

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
                    blockid = int(block_path.split('block_')[1].split('.')[0])
                    blockl = end_index - start_index

                    event_time = all_signals['time'][start_index:end_index +1]
                    len_signal = len(event_time)
                    ecg, ppg = None, None
                    if 'GE_WAVE_ECG_2_ID' in signals_keys:
                        ecg = all_signals['GE_WAVE_ECG_2_ID'][start_index:end_index +1]
                    if 'GE_WAVE_SPO2_WAVE_ID' in signals_keys:
                        ppg = all_signals['GE_WAVE_SPO2_WAVE_ID'][start_index:end_index +1]

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
                        ecg_wt_feature = compute_wavelet_features(ecg_denoise,R_peak_index,wavelet = family,level = level,windowsize=windowsize)
                  
                    
                    if ppg.any:
                        # PPG signal preprocessing for denoising 
                        # print("un-denoised ppg", ppg)
                        ppg = PPG_denoising(ppg)
                        # PPG signal feature extraction
                        ppg_extracted_features = PPG_feature_extraction(ppg,event_time,ppg_features,R_peak_index)

                    if windowsize==1: #1beat extraction
                        for i in range(num_beats):
                            windowL = int(max(R_peak_index[i] - 85, 0))  # not exact, just for plotting
                            windowR = int(min(R_peak_index[i] + 85, len_signal-1))
                            temp = [patient_id, blockid, event_time[windowL], event_time[windowR]]
                            if ecg_features['RR']:
                                temp.append(ecg_RR_feature.pre_R[i])
                                temp.append(ecg_RR_feature.post_R[i])
                                temp.append(ecg_RR_feature.local_R[i])
                            if ppg_features['st']:
                               temp.append(ppg_extracted_features.st[i])
                            if ppg_features['dt']:
                                temp.append(ppg_extracted_features.dt[i])
                            if ppg_features['half_width']:
                                temp.append(ppg_extracted_features.half_width[i])
                            if ppg_features['two_third_width']:
                                temp.append(ppg_extracted_features.two_third_width[i])
                            if ecg_features['Wavelet']:
                                temp.extend(ecg_wt_feature[i])
                            temp.append(label)
                            if label == 'PP' or label == 'PS':
                                health = 0
                            else:
                                health = 1
                            temp.append(health)
                            writer.writerow(temp)

                    else: #dealing with multi-beat window size
                        start_buffer = (windowsize - 1) // 2
                        end_buffer = windowsize // 2
                        # quartiles of selected features within window
                        q1i, q2i, q3i = [int(0.25 * windowsize), int(0.5 * windowsize), int(0.75 * windowsize)]
                        for i in range(start_buffer, num_beats-end_buffer):
                            windowL = int(max(R_peak_index[i - start_buffer] - 85, 0)) #not exact, just for plotting
                            windowR = int(min(R_peak_index[i + end_buffer] + 85,len_signal-1))
                            temp = [patient_id,blockid,event_time[windowL],event_time[windowR]]
                            if ecg_features['RR']:
                                #getting RR features in window
                                pre_R = ecg_RR_feature.pre_R[(i-start_buffer):(i+end_buffer+1)]; #pre_R = np.sort(pre_R)
                                post_R = ecg_RR_feature.post_R[(i-start_buffer):(i+end_buffer+1)]; #post_R = np.sort(post_R)
                                local_R = ecg_RR_feature.local_R[(i-start_buffer):(i+end_buffer+1)]; #local_R = np.sort(local_R)
                                temp.extend(pre_R); temp.extend(post_R); temp.extend(local_R)

                                #if just keeping median and interquartile-range (IQR) of selected features
                                #temp.append(pre_R[q2i]); temp.append(pre_R[q3i] - pre_R[q1i])
                                #temp.append(post_R[q2i]); temp.append(post_R[q3i] - post_R[q1i])
                                #temp.append(local_R[q2i]); temp.append(local_R[q3i] - local_R[q1i])
                            if ppg_features['all']:
                                st = ppg_extracted_features.st[(i-start_buffer):(i+end_buffer+1)]; #st = np.sort(st)
                                #temp.append(st[q2i]); temp.append(st[q3i] - st[q1i])
                                dt = ppg_extracted_features.dt[(i-start_buffer):(i+end_buffer+1)]; #dt = np.sort(dt)
                                #temp.append(dt[q2i]); temp.append(dt[q3i] - dt[q1i])
                                hw = ppg_extracted_features.half_width[(i-start_buffer):(i+end_buffer+1)]; #hw = np.sort(hw)
                                #temp.append(hw[q2i]); temp.append(hw[q3i] - hw[q1i])
                                ttw = ppg_extracted_features.two_third_width[(i-start_buffer):(i+end_buffer+1)]; #ttw = np.sort(ttw)
                                #temp.append(ttw[q2i]); temp.append(ttw[q3i] - ttw[q1i])
                                temp.extend(st); temp.extend(dt); temp.extend(hw); temp.extend(ttw)
                            if ecg_features['Wavelet']:
                               temp.extend(ecg_wt_feature[i-start_buffer])

                            #finaling writing label and health information
                            temp.append(label)
                            if label == 'PP' or label == 'PS':
                                health = 0
                            else:
                                health = 1
                            temp.append(health)
                            writer.writerow(temp)



                    break
                    

                else: continue
    return None



def clean_data(path_to_data):
    '''
    Takes path to loaded and data and writes new file of cleaned data.
    Cleaned data has NaN, [], or empty entries replaced usng median imputation
    '''
    data = pd.read_csv(path_to_data, low_memory=False)
    #data.replace(np.nan, -10000)

    labels = list(data)
    featuredata = data.iloc[:,:-2].to_numpy()
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp = imp.fit(featuredata)
    featuredata = imp.transform(featuredata)

    dic = {}
    n = data.shape[1]
    for i in range(n-2):
        dic[labels[i]] = featuredata[:,i]
    dic[labels[-2]] = data.iloc[:,-2]
    dic[labels[-1]] = data.iloc[:,-1]

    pd.DataFrame(data=dic).to_csv('Data_Cleaned_1beat_ecg.csv')
    return None

load_label_event('Waveform Data', 'Labelled_Events.xlsx',excel_sheet_name='Summary', windowsize=1, save_file_path='/Users/richard.morse/Desktop/DSCI_435/TCH_Arrhythmias_F20/ModelData')
clean_data('ModelData/dataset_for_modeling.csv')