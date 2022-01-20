

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import gridspec
import pywt
import pandas as pd


def extract_feature(ppg_signal, feature_list):
    """
    :param signal: input beat signal
    :param feature_list: list of features to be extracted
    :return: features of a single beat
    """

    peaks, _ = signal.find_peaks(ppg_signal, distance=200)  # peak postions
    troughs, _ = signal.find_peaks(-ppg_signal, distance=200)  # trough positions
    half_width = signal.peak_widths(ppg_signal, peaks, rel_height=0.5)[0]  # peak width at half maximum
    two_thirds = signal.peak_widths(ppg_signal, peaks, rel_height=1 / 3)[0]  # peak width at two thirds of maximum

    st = []  # st feature, calculated by the interval of trough to peak
    dt = []  # dt feature, calculated by the interval of peak to trough
    m = len(peaks)
    n = len(troughs)

    if m != 0 and n != 0:
        if m < n:  # signal has more troughs than peaks, use all peaks
            for i in range(m):
                st.append(peaks[i] - troughs[i])
                dt.append(troughs[i + 1] - peaks[i])
        elif m > n:  # signal has more peaks than troughs, use all troughs
            for i in range(n):
                dt.append(troughs[i] - peaks[i])
                st.append(peaks[i + 1] - troughs[i])
        else:  # signal has same number of peaks and troughs
            if peaks[0] < troughs[0]:  # the first is peak, append a dt feature that is not in pair
                for i in range(n - 1):
                    dt.append(troughs[i] - peaks[i])
                    st.append(peaks[i + 1] - troughs[i])
                dt.append(troughs[n - 1] - peaks[n - 1])
            else:  # the first is trough, append a st feature that is not in pair
                for i in range(n - 1):
                    st.append(peaks[i] - troughs[i])
                    dt.append(troughs[i + 1] - peaks[i])
                st.append(peaks[n - 1] - troughs[n - 1])


    beat_feature = []
    for key in feature_list:
        if key == 'st':
            beat_feature.append(st)
        elif key == 'dt':
            beat_feature.append(dt)
        elif key == 'half_width':
            beat_feature.append(half_width)
        elif key == 'two_thirds':
            beat_feature.append(two_thirds)
        else:
            beat_feature.append(None)

    return beat_feature


def PPG_feature_extraction(ppg_signal, feature_list, R_peaks, window=[-60, 60]):
    """
    :param signal: input signal
    :param feature_list: features need to be computed
    :param R_peaks: list of R peaks position
    :param window:window centered around R peak
    :return: list of features
    """
    features = []
    len_signal = len(ppg_signal)
    num_beats = len(R_peaks)
    for i in range(num_beats):
        if R_peaks[i] <= window[0]:
            beat = ppg_signal[:R_peaks[i] + window[
                1]]  # the first and last beat has different length which may lead to different number of coefficient
            # zero padding may need to add.
        elif R_peaks[i] + window[1] >= len_signal:
            beat = ppg_signal[R_peaks[i] + window[0]:]
        else:
            beat = ppg_signal[R_peaks[i] + window[0]:R_peaks[i] + window[1]]

        feature_beat = extract_feature(beat, feature_list)
        features.append(feature_beat)


    return features

