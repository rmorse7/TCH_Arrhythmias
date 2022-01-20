
from scipy import signal

import pywt
from ECG_preprocessing import butter_bandpass_filter


def PPG_denoising(ppg_signal, fs=240, wavelet_option=True, wavelet_parameters=('db2', 0.04), detrend=True, wiener=True,
                  wiener_parameter=5,
                  butterworth_option=True, butterworth_parameters=
                  (3, 4, 0.5), notch_option=True, notch_parameters=(50, 30), normalize=True):
    """
    :param signal: input raw signal
    :param fs: sampling frequency
    :param wavelet_option: boolean, using or not using wavelet denoising
    :param wavelet_parameters:2-element tuple, wavelet denoising parameters, first one is wavelet
    family, second one is threshold value
    :param detrend:boolean, using or not using linear detrending
    :param wiener:boolean, using or not using wiener filter
    :param wiener_parameter: integer, wiener filter size
    :param butterworth_option:boolean, using or not using butterworth filter
    :param butterworth_parameters:3-element tuple, first element is order, second element is upperbound of band,
    third element is lower bound of band
    :param notch_option:boolean, using or not using notch filter
    :param notch_parameters:2-element tuple, first element is the frequency to be removed, second element is quality factor
    :param normalize:boolean, using or not using normalized signal
    :return:denoised PPG signal
    """
    PPG, wavelet_coeff = None, []
    if wavelet_option:
        PPG, wavelet_coeff = wavelet_denoise(ppg_signal, wavelet_parameters)
    if detrend:
        #print("After detrend before wavelet: ", PPG)
        PPG = signal.detrend(PPG)
    if wiener:
        PPG = signal.wiener(PPG, wiener_parameter)
    if butterworth_option:
        PPG = butterworth_denoise(PPG, fs, butterworth_parameters)
    if notch_option:
        PPG = notch_denoise(PPG, fs, notch_parameters)
    if normalize:
        PPG = (PPG - min(PPG)) / (max(PPG) - min(PPG))
    return PPG


def wavelet_denoise(ppg_signal, wavelet_parameters):
    """
    :param signal:input signal
    :param wavelet_parameters: 2-element tuple, wavelet denoising parameters, first one is wavelet
    family, second one is threshold value
    :return:signal after wavelet denoising
    """
    wavelet_term = wavelet_parameters[0]
    wav_threshold = wavelet_parameters[1]
    w = pywt.Wavelet(wavelet_term)
    maxlev = pywt.dwt_max_level(len(ppg_signal), w.dec_len)

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(ppg_signal, 'db2', level=maxlev)

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], wav_threshold * max(coeffs[i]),mode='hard')

    datarec = pywt.waverec(coeffs, wavelet_term)

    return datarec, coeffs


def butterworth_denoise(ppg_signal, fs, butterworth_parameters):
    """
    :param signal:input signal
    :param fs:sampling frequency
    :param butterworth_parameters:3-element tuple, first element is order, second element is upperbound of band,
    third element is lower bound of band
    :return:signal after butterworth filter
    """
    order = butterworth_parameters[0]
    highcut = butterworth_parameters[1]
    nyq = 0.5 * fs
    lowcut = butterworth_parameters[2]
    output = butter_bandpass_filter(ppg_signal, lowcut, highcut, fs, order)
    return output


def notch_denoise(ppg_signal, fs, notch_parameters):
    """
    :param signal:input signal
    :param fs: sampling frequency
    :param notch_parameters:2-element tuple, first element is the frequency to be removed, second element is quality factor
    :return: signal after notch filter
    """
    f0 = notch_parameters[0]
    Q = notch_parameters[1]

    b, a = signal.iirnotch(f0, Q, fs=fs)
    output = signal.filtfilt(b, a, ppg_signal)
    return output

