import scipy.stats
import pywt
import numpy as np


class RR_intervals:
    def __init__(self):
        # Instance atributes
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])


def compute_RR_intervals(R_poses):
    '''
    Input: the R-peaks (time/s) from a signal
    Return: the features RR intervals (pre_RR, post_RR, local_RR) for each beat

    '''

    features_RR = RR_intervals()

    pre_R = np.array([], dtype=int)
    post_R = np.array([], dtype=int)
    local_R = np.array([], dtype=int)


    if len(R_poses) <= 2:
        for i in range(0, len(R_poses)):
            features_RR.pre_R = [[] for i in range(len(R_poses))]
            features_RR.post_R = [[] for i in range(len(R_poses))]
            features_RR.local_R = [[] for i in range(len(R_poses))]

        return features_RR

    # Pre_R and Post_R
    pre_R = np.append(pre_R, 0)
    post_R = np.append(post_R, R_poses[1] - R_poses[0])


    for i in range(1, len(R_poses) - 1):
        pre_R = np.append(pre_R, R_poses[i] - R_poses[i - 1])
        post_R = np.append(post_R, R_poses[i + 1] - R_poses[i])

    pre_R[0] = pre_R[1]
    pre_R = np.append(pre_R, R_poses[-1] - R_poses[-2])

    post_R = np.append(post_R, post_R[-1])

    # Local_R: AVG from last 10 pre_R values
    for i in range(0, len(R_poses)):
        num = 0
        avg_val = 0
        for j in range(-9, 1):
            if j + i >= 0:
                avg_val = avg_val + pre_R[i + j]
                num = num + 1
        local_R = np.append(local_R, avg_val / float(num))

    for i in range(0, len(R_poses)):
        features_RR.pre_R = np.append(features_RR.pre_R, pre_R[i])
        features_RR.post_R = np.append(features_RR.post_R, post_R[i])
        features_RR.local_R = np.append(features_RR.local_R, local_R[i])

    return features_RR


def compute_wavelet_features(original_signal, R_peak_index, windowL=-60, windowR=60, wavelet='db1', level=3):
    '''
    segment the ECG singal into cycless based on R-peak index and a fixed windos with a half second then return the wavelet coefficient for each beats
    input: original ECG singal, R-peak index, fixed windows centered at R-peak, wavelet type and level
    return: the wavelet cofficients for each beat
    '''

    wavelet_cofficients = []
    len_signal = len(original_signal)
    num_beats = len(R_peak_index)
    for i in range(num_beats):
        if R_peak_index[i] + windowL < 0:
            zeros_padding = np.array([0] * abs(R_peak_index[i] + windowL))

            beat = np.append(zeros_padding, original_signal[:R_peak_index[
                                                                 i] + windowR])  # the first and last beat has different length which may lead to different number of coefficient
            # zero padding may need to add.
        elif R_peak_index[i] + windowR >= len_signal:
            beat = np.append(original_signal[R_peak_index[i] + windowL:],
                             [0] * (R_peak_index[i] + windowR - len_signal))
        else:
            beat = original_signal[R_peak_index[i] + windowL:R_peak_index[i] + windowR]

        if len(beat) != (windowR - windowL):
            raise Exception("The beat length is not correct!!! Please check!")

        current_coefffs = compute_wavelet_descriptor(beat, wavelet, level)

        wavelet_cofficients.append(current_coefffs)
    return wavelet_cofficients


def compute_wavelet_descriptor(beat, family, level):
    # Compute the wavelet descriptor for a beat
    '''

    :param beat: input beat for wavelet transform
    :param family: wavelet family option
    :param level: wavelet transformation level
    :return:wavelet coefficients of wavelet transform
    '''
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]