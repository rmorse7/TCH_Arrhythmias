

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import gridspec
import pywt
import pandas as pd

class PPG_feats:
    def __init__(self):
        # Instance atributes
        self.st = np.array([])
        self.dt = np.array([])
        self.half_width = np.array([])
        self.two_third_width = np.array([])


def PPG_feature_extraction(ppg_signal, time, feature_list, R_peaks):
    """
    :param signal: input signal
    :param time: input time
    :param feature_list: features need to be computed
    :param R_peaks: list of R peaks position
    :param windowsize: PPG peaks to be included in window
    :return: list of features
    """
    features_ppg = PPG_feats()
    len_signal = len(ppg_signal)

    #find PPG signal peaks via scipy signal function
    peaks, _ = signal.find_peaks(ppg_signal, threshold=10**(-8), distance=120)  # peak postions
    num_beats = len(peaks)
    num_rpeaks = len(R_peaks)

    #do we have enough beats/peak to extract information?
    if num_beats <= 1 or num_rpeaks <= 0:
        features_ppg.st = [np.nan for i in range(num_rpeaks)]  # st feature, calculated by the interval of trough to peak
        features_ppg.dt = [np.nan for i in range(num_rpeaks)]  # dt feature, calculated by the interval of peak to trough
        features_ppg.half_width = [np.nan for i in range(num_rpeaks)]
        features_ppg.two_third_width = [np.nan for i in range(num_rpeaks)]
        return features_ppg

    st = []; dt = []

    pp_interval = ppg_signal[peaks[0]:peaks[1]]
    troughbefore_idx = np.argmin(pp_interval) + peaks[0]

    #first interval
    st.append(time[peaks[1]] - time[troughbefore_idx])  #set equal to one after
    dt.append(time[troughbefore_idx] - time[peaks[0]])   #we have this for first beat

    #plt.plot(time[peaks[0]:peaks[10]],ppg_signal[peaks[0]:peaks[10]])
    #plt.show()

    #middle intervals
    for i in range(1,num_beats-1):
        st.append(time[peaks[i]] - time[troughbefore_idx])

        pp_interval = ppg_signal[peaks[i]:peaks[i+1]]
        troughafter_idx = np.argmin(pp_interval) + peaks[i]
        dt.append(time[troughafter_idx] - time[peaks[i]])

        troughbefore_idx = troughafter_idx
    #last inerval
    st.append(time[peaks[-1]] - time[troughbefore_idx])   #we have this for last beat
    dt.append(time[troughbefore_idx] - time[peaks[-2]])  #set equal to one before

    #done via scipy
    half_width = signal.peak_widths(ppg_signal, peaks, rel_height=0.5)[0]  # peak width at half maximum
    two_thirds = signal.peak_widths(ppg_signal, peaks, rel_height=(1 / 3))[0]  # peak width at two thirds of maximum

    #print('PPG features: '); print(len(st),len(dt),len(half_width),len(two_thirds))

    #assigning PPG info to a particular RR-interval (since the two signals might not exactly align temporally)
    featcap = len(st) - 1
    i = 0
    for rpeak in R_peaks:
        while i < featcap and abs(rpeak-peaks[i]) > abs(rpeak-peaks[i+1]):
            i += 1
        features_ppg.st =  np.append(features_ppg.st,st[i])
        features_ppg.dt =  np.append(features_ppg.dt,dt[i])
        features_ppg.half_width =  np.append(features_ppg.half_width,half_width[i])
        features_ppg.two_third_width =  np.append(features_ppg.two_third_width,two_thirds[i])
        
    return features_ppg

