
import scipy.stats
from scipy.signal import resample
import pywt
import numpy as np



class RR_intervals:
    def __init__(self):
        # Instance atributes
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])
        #self.global_R = np.array([])




def compute_RR_intervals(R_poses):
    '''
    Input: the R-peaks (time/s) from a signal
    Return: the features RR intervals (pre_RR, post_RR, local_RR) for each beat 
    
    '''


    features_RR = RR_intervals()

    pre_R = np.array([], dtype=int)
    post_R = np.array([], dtype=int)
    local_R = np.array([], dtype=int)
    #global_R = np.array([], dtype=int)

    #if can't extract any information b/c not enough peaks
    if len(R_poses) <= 2:
        for i in range(0, len(R_poses)):
            features_RR.pre_R = [np.nan for i in range(len(R_poses))]
            features_RR.post_R = [np.nan for i in range(len(R_poses))]
            features_RR.local_R = [np.nan for i in range(len(R_poses))]

        return features_RR

    # Pre_R and Post_R
    pre_R = np.append(pre_R, 0)
    post_R = np.append(post_R, R_poses[1] - R_poses[0])

    #print(len(R_poses))
    for i in range(1, len(R_poses)-1):
        pre_R = np.append(pre_R, R_poses[i] - R_poses[i-1])
        post_R = np.append(post_R, R_poses[i+1] - R_poses[i])

    pre_R[0] = pre_R[1]
    pre_R = np.append(pre_R, R_poses[-1] - R_poses[-2])  

    post_R = np.append(post_R, post_R[-1])

    # Local_R: AVG from last 10 pre_R values
    for i in range(0, len(R_poses)):
        num = 0
        avg_val = 0
        for j in range(-9, 1):
            if j+i >= 0:
                avg_val = avg_val + pre_R[i+j]
                num = num +1
        local_R = np.append(local_R, avg_val / float(num))

	# # Global R AVG: from full past-signal
    # # TODO: AVG from past 5 minutes = 108000 samples
    # global_R = np.append(global_R, pre_R[0])    
    # for i in range(1, len(R_poses)):
    #     num = 0
    #     avg_val = 0

    #     for j in range( 0, i):
    #         if (R_poses[i] - R_poses[j]) < 108000:
    #             avg_val = avg_val + pre_R[j]
    #             num = num + 1
    #     #num = i
    #     global_R = np.append(global_R, avg_val / float(num))

    for i in range(0, len(R_poses)):
        features_RR.pre_R = np.append(features_RR.pre_R, pre_R[i])
        features_RR.post_R = np.append(features_RR.post_R, post_R[i])
        features_RR.local_R = np.append(features_RR.local_R, local_R[i])
        #features_RR.global_R = np.append(features_RR.global_R, global_R[i])

        #features_RR.append([pre_R[i], post_R[i], local_R[i]])
            
    return features_RR


def compute_wavelet_features(original_signal,R_peak_index,wavelet = 'db1',level = 3,windowsize=3, numsamples=170, before=0.5,after=0.5):
    '''
    segment the ECG singal into cycless based on R-peak index and a fixed windows with a half second then return the wavelet coefficient for each beats
    input: original ECG singal, R-peak index, fixed windows centered at R-peak, wavelet type and level
    return: the wavelet cofficients for each beat

    '''
    #dynamic segmentation
    if (before + after != 1):
        return "before + after must equal 1"
    rr_lengths = [R_peak_index[i + 1] - R_peak_index[i] for i in range(len(R_peak_index) - 1)]

    wavelet_cofficients = []
    len_signal = len(original_signal)
    num_beats = len(R_peak_index)
    #do we have enough beats to extract wavelet information
    if num_beats == 0:
        return wavelet_cofficients
    if num_beats == 1:
        rr_lengths = [len_signal]
    if windowsize > num_beats:
        windowsize = num_beats
    beat_length = numsamples*windowsize

    start_buffer = (windowsize-1)//2
    end_buffer = windowsize//2

    #first window
    windowL = int(max(R_peak_index[0] - rr_lengths[0]*before, 0))
    windowR = int(min(R_peak_index[windowsize-1] + rr_lengths[windowsize-2]*after,len_signal-1))
    beat = resample(original_signal[windowL:windowR], beat_length)
    current_coefffs = compute_wavelet_descriptor(beat, wavelet, level)
    wavelet_cofficients.append(current_coefffs)

    #middle windows
    for i in range(start_buffer+1, num_beats-end_buffer-1):
        windowL = int(max(R_peak_index[i-start_buffer] - rr_lengths[i-start_buffer-1]*before,0))
        windowR = int(min(R_peak_index[i+end_buffer] + rr_lengths[i+end_buffer]*after,len_signal-1))
        beat = resample(original_signal[windowL:windowR],beat_length)

        current_coefffs = compute_wavelet_descriptor(beat,wavelet,level)
        wavelet_cofficients.append(current_coefffs)

    #last window
    windowL = int(max(R_peak_index[num_beats-windowsize] - rr_lengths[num_beats-windowsize-1] * before,0))
    windowR = int(min(R_peak_index[num_beats - 1] + rr_lengths[num_beats - 2] * after,len_signal-1))
    beat = resample(original_signal[windowL:windowR], beat_length)
    current_coefffs = compute_wavelet_descriptor(beat, wavelet, level)
    wavelet_cofficients.append(current_coefffs)

    return wavelet_cofficients


    
def compute_wavelet_descriptor(beat, family, level):
    # Compute the wavelet descriptor for a beat
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]


