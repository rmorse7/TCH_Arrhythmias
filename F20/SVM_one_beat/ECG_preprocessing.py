import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, find_peaks
import pywt
from scipy import interpolate


def ecg_preprocessing_final(original_signal, fs=240):
    '''
    This function is the whole preprocessing pipeline for ECG including spike removal, butter bandpass filter and R-peak detection
    based on Pan-Tompkins algrithom
    Input:
    original_signal: the original_signal needs to be preprocessed
    fs: the sampling frequency of the original_signal
    Return:
    R_peak: the index of R-peak in the original_signal
    ecg_after_bandpass: the signal after preprocessing
    '''

    ecg_after_spike_removal = schmidt_spike_removal(original_signal)

    ecg_after_bandpass = butter_bandpass_filter(ecg_after_spike_removal, lowcut=0.67, highcut=50)

    R_peak = pan_tompkins_detector(ecg_after_bandpass)  # the index of R_peak is obtained

    return R_peak, ecg_after_bandpass


def schmidt_spike_removal(original_signal, fs=240):
    '''
    This function is used to removes the spikes in a signal
    Inputs:
        original_signal: The original (1D) audio signal array
        fs: the sampling frequency (Hz) (default is 240Hz which is the sampling frequency for ECG in our data)
    Outputs:
        despiked_signal: the audio signal with any spikes removed.
    This function removes the spikes in a signal as done by Schmidt et al in
    the paper:
    Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J.
    (2010). Segmentation of heart sound recordings by a duration-dependent
    hidden Markov model. Physiological Measurement, 31(4), 513-29.
    The spike removal process works as follows:
    (1) The recording is divided into 500 ms windows.
    (2) The maximum absolute amplitude (MAA) in each window is found.
    (3) If at least one MAA exceeds three times the median value of the MAA's,
    the following steps were carried out. If not continue to point 4.
    (a) The window with the highest MAA was chosen.
    (b) In the chosen window, the location of the MAA point was identified as the top of the noise spike.
    (c) The beginning of the noise spike was defined as the last zero-crossing point before theMAA point.
    (d) The end of the spike was defined as the first zero-crossing point after the maximum point.
    (e) The defined noise spike was replaced by zeroes.
    (f) Resume at step 2.
    (4) Procedure completed.
    This code is derived from the paper:
    S. E. Schmidt et al., "Segmentation of heart sound recordings by a
    duration-dependent hidden Markov model," Physiol. Meas., vol. 31,
    no. 4, pp. 513-29, Apr. 2010.
    Developed by David Springer for comparison purposes in the paper:
    D. Springer et al., ?Logistic Regression-HSMM-based Heart Sound
    Segmentation,? IEEE Trans. Biomed. Eng., In Press, 2015.
    Copyright (C) 2016  David Springer
    dave.springer@gmail.com
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    '''

    # Find the window size
    windowsize = fs // 2

    # Find any samples outside of a integer number of windows:
    trailingsamples = len(original_signal) % windowsize
    # Find the number of windows
    nrows = len(original_signal) // windowsize

    # Reshape the signal into a number of windows
    sampleframes = np.array(original_signal[0:len(original_signal) - trailingsamples]).reshape((nrows, windowsize))

    # Find the maximum absolute amplitude (MAAs) in each window at each windows
    MAAs = np.max(np.abs(sampleframes), 1)

    # While there are still samples greater than 2* the median value of the
    # MAAs, then remove those spikes:
    while len(MAAs) != 0 and np.max(MAAs) > (2 * np.median(MAAs)):
        # Find the window with the max MAA:
        window_num = np.argmax(MAAs)
        # Find the postion of the spike within that window:
        spike_position = np.where(np.abs(sampleframes[window_num, :]) == np.max(MAAs))[0]
        if spike_position.size >= 1:
            spike_position = spike_position[0]

        # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative):
        window_sign = np.sign(sampleframes[window_num, :])
        zero_crossings = (abs((np.roll(window_sign, -1) - window_sign)) > 1).astype(int)
        zero_crossings[-1] = 0

        # Find the start of the spike, finding the last zero crossing before
        # spike position. If that is empty, take the start of the window:

        spike_start = np.where(zero_crossings[:spike_position + 1] == 1)[0]
        if spike_start.size >= 1:
            spike_start = spike_start[-1]
        else:
            spike_start = 0

        # Find the end of the spike, finding the first zero crossing after
        # spike position. If that is empty, take the end of the window:
        zero_crossings[:spike_start + 1] = 0
        spike_end = np.where(zero_crossings == 1)[0]
        if spike_end.size >= 1:
            spike_end = spike_end[0]
        else:
            spike_end = zero_crossings.size - 1
        # Set to Zero
        sampleframes[window_num, spike_start:spike_end + 1] = 0.0001

        # Recaclulate MAAs
        MAAs = np.max(np.abs(sampleframes), 1)

    despiked_signal = sampleframes.reshape(1, nrows * windowsize)

    # Add the trailing samples back to the signal:
    final = np.concatenate((despiked_signal[0], original_signal[nrows * windowsize:]), axis=None)
    return final


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
        butter_bandpass function is used to calculate the parameters needed in the butter band pass filter
        and you dont need to directly use this function
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=0.67, highcut=50, fs=240, order=5):
    ''' butter bandpass filter for signals
    input
        data: signals input
        lowcut: low frequency cuttoff (frequency lower than lowcut will be removed)
        highcut: high frequency cuttoff (frequency higher than highcut will be removed)
        fs: the sampling frequency of data ( you can calcuate by 1/(time[1]-time[0]))
        order
    output:
        filtered signals
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def swt_detector(unfiltered_ecg, fs=240):
    """
    Stationary Wavelet Transform for ECG R-peak detection
    The index of R-peak location is returned
    based on Vignesh Kalidas and Lakshman Tamil.
    Real-time QRS detector using Stationary Wavelet Transform
    for Automated ECG Analysis.
    In: 2017 IEEE 17th International Conference on
    Bioinformatics and Bioengineering (BIBE).
    Uses the Pan and Tompkins thresolding.
    """

    swt_level = 3
    padding = -1
    for i in range(1000):
        if (len(unfiltered_ecg) + i) % 2 ** swt_level == 0:
            padding = i
            break

    if padding > 0:
        unfiltered_ecg = np.pad(unfiltered_ecg, (0, padding), 'edge')
    elif padding == -1:
        print("Padding greater than 1000 required\n")

    swt_ecg = pywt.swt(unfiltered_ecg, 'db3', level=swt_level)
    swt_ecg = np.array(swt_ecg)
    swt_ecg = swt_ecg[0, 1, :]

    squared = swt_ecg * swt_ecg

    f1 = 0.01 / fs
    f2 = 10 / fs

    b, a = butter(3, [f1 * 2, f2 * 2], btype='bandpass')
    filtered_squared = lfilter(b, a, squared)

    filt_peaks = panPeakDetect(filtered_squared, fs)

    return filt_peaks


def pan_tompkins_detector(unfiltered_ecg, fs=240):
    """
    Pan Tompkins Algorithm for R-peak detection
    The index of R-peak location is returned
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.
    """

    f1 = 5 / fs
    f2 = 15 / fs

    b, a = butter(1, [f1 * 2, f2 * 2], btype='bandpass')

    filtered_ecg = lfilter(b, a, unfiltered_ecg)

    diff = np.diff(filtered_ecg)

    squared = diff * diff

    N = int(0.12 * fs)
    mwa = MWA_cumulative(squared, N)
    mwa[:int(0.2 * fs)] = 0

    mwa_peaks = panPeakDetect(mwa, fs)

    return mwa_peaks


def MWA_cumulative(input_array, window_size):
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks


def dynamic_window_segmentation(r_peaks, finalidx, before=0.5, after=0.5):
    ##Take R peaks as inputs and create RR segments and then create windows for segmentation
    ##Input: r_peaks which is must be determined using pan-tompkins or swt peak detection
    ##       before, after represent the proportion of the prior RR segment and the subsequent RR segment to be considered with the current R peak
    ##       (ie before = after = 0.5 -> from R peak i, the last half of the prior RR segment and the first half of the subsequent RR segment will be considered with the i-th peak)
    ##       finalidx = index of final measurement in data considered, used for keeping last segment w/in frame

    if (before + after != 1):
        return "before + after must equal 1"

    rr_lengths = [r_peaks[i + 1] - r_peaks[i] for i in range(len(r_peaks) - 1)]

    segments = []

    ##Handle first r_peak
    segments.append((int(max(r_peaks[0] - rr_lengths[0] * before, 0)),
                     int(r_peaks[0] + rr_lengths[0] * after)))  # ensure we stay in window

    ##Segment the middle peaks
    for peak_idx in range(1, len(r_peaks) - 1):
        segments.append((int(r_peaks[peak_idx] - rr_lengths[peak_idx - 1] * before),
                         int(r_peaks[peak_idx] + rr_lengths[peak_idx] * after)))

    ##Handle Last segment
    segments.append((int(r_peaks[-1] - rr_lengths[-1] * before),
                     int(min(r_peaks[-1] + rr_lengths[-1] * after, finalidx))))  # ensure we stay in window

    return segments


def resample_segments(segment_idxs, time, signals, num_samples=None):
    """
    Resamples the data from the list of segments using linear interpolation.
    Each should be the length of the average number of samples in each segment unless num_samples specified

    Input:
    --segment_idxs : list of tuples - (start,end) index pairs for each beat segment
    --time : list - time window
    --signals : dictionary - {signal_label : signal in window}
    --num_samples : int - # of samples to use in each segment, if None taken to be average segment length

    Output:
    --time_segs = resampled, segmented, time blocks
    --signal_segs = dictionary of resampled, segmented signal blocks for each signal

    """
    total_samples = len(time)
    if num_samples == None:
        desired_samples = (int)(total_samples / len(segment_idxs))
    else:
        desired_samples = num_samples

    # Segment time appropriately
    time_segs = []
    signal_segs = dict.fromkeys(signals.keys(), [])

    for start, end in segment_idxs:
        tnew = np.linspace(time[start], time[end], desired_samples)
        time_segs.append(tnew)
        for label, signal in signals.items():
            f = interpolate.interp1d(time[start:end], signal[start:end], fill_value='extrapolate')
            signew = f(tnew)
            signal_segs[label].append(signew)

    return time_segs, signal_segs
