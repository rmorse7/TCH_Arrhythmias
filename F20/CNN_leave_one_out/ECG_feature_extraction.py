
import scipy.stats
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

    if len(R_poses) <= 2:
        for i in range(0, len(R_poses)):
            features_RR.pre_R = [[] for i in range(len(R_poses))]
            features_RR.post_R = [[] for i in range(len(R_poses))]
            features_RR.local_R = [[] for i in range(len(R_poses))]

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

def compute_cwt_features(original_signal,R_peak_index,scales= np.arange(1,129),windowL=-240,windowR=240,wavelet = 'morl'):
    '''
    this function is used to compute the 2d cwt features for original signal
    input:
        original signal: 
        R_peak_index
        each segment will have a 2d cwt coefficients with the dimension [scales length, window length] for example, for the default setting, the dimension is [128,480]
            scales: it is the parameter for cwt (two parameters for cwt function the first one is the scale parameter the second one is the translation. Here is the scales)
            windowL and windowR defines the windows length
        wavelet: the wavelet function
    return:
        a series of 2d cwt for each segment
        the dimension is [number of segments,sacles length, window length]

    '''

    wavelet_cofficients = []
    len_signal = len(original_signal)
    num_beats = len(R_peak_index)    

    for i in range(1,num_beats-1,3): 
        if R_peak_index[i]+windowL<0 and R_peak_index[i]+windowR>=len_signal:
            zeros_padding = np.array([0]*abs(R_peak_index[i]+windowL))
            beat = np.append(zeros_padding,original_signal)
            beat = np.append(beat,[0]*(R_peak_index[i]+windowR-len_signal))
   
        elif R_peak_index[i]+windowL<0:
            zeros_padding = np.array([0]*abs(R_peak_index[i]+windowL))

            beat = np.append(zeros_padding,original_signal[:R_peak_index[i]+windowR])   # the first and last beat has different length which may lead to different number of coefficient
                                                            # zero padding may need to add.
        elif R_peak_index[i]+windowR>=len_signal:
            beat = np.append(original_signal[R_peak_index[i]+windowL:],[0]*(R_peak_index[i]+windowR-len_signal))
        else:
            beat = original_signal[R_peak_index[i]+windowL:R_peak_index[i]+windowR]
        
        if len(beat)!=(windowR-windowL): 
            raise Exception("The beat length is not correct!!! Please check!")

        current_coefffs = compute_cwt_coeffs(beat,rescale = True, scales = np.arange(1,129),wavelet = 'morl')
        
        wavelet_cofficients.append(current_coefffs)
    return wavelet_cofficients    

def compute_cwt_coeffs(original_signal,rescale= True, scales=np.arange(1,129),wavelet = 'morl'):
    '''
    to get the 2d cwt coeffs for one segment
    input:
        original_signal: one segment
        rescale: if rescale is true then we will do the following: ori_cwt--> abs(ori_cwt)-->rescale to (0,255) (which is the images scale for one channel)
        scales: parameter for cwt 
        wavelet: wavelet function
    return:
        2d coeffs for one segment
    '''
    [coefficients, _] = pywt.cwt(original_signal, scales, wavelet)
    if not rescale: return coefficients
    coefficients = np.abs(coefficients)
    new = (coefficients-np.min(coefficients))/np.max(coefficients)
    new = np.uint8(new*255) 
    return new

    
def compute_wavelet_features(original_signal,R_peak_index,windowL=-60,windowR=60,wavelet = 'db1',level = 3):
    '''
    segment the ECG singal into cycless based on R-peak index and a fixed windos with a half second then return the wavelet coefficient for each beats
    input: original ECG singal, R-peak index, fixed windows centered at R-peak, wavelet type and level
    return: the wavelet cofficients for each beat

    '''

    wavelet_cofficients = []
    len_signal = len(original_signal)
    num_beats = len(R_peak_index)
    # if it is small windows then we dont ignore the first and last beat
    if windowR-windowL<=240: 
        for i in range(num_beats):
            if R_peak_index[i]+windowL<0:
                zeros_padding = np.array([0]*abs(R_peak_index[i]+windowL))

                beat = np.append(zeros_padding,original_signal[:R_peak_index[i]+windowR])   # the first and last beat has different length which may lead to different number of coefficient
                                                                # zero padding may need to add.
            elif R_peak_index[i]+windowR>=len_signal:
                beat = np.append(original_signal[R_peak_index[i]+windowL:],[0]*(R_peak_index[i]+windowR-len_signal))
            else:
                beat = original_signal[R_peak_index[i]+windowL:R_peak_index[i]+windowR]
            
            if len(beat)!=(windowR-windowL): 
                raise Exception("The beat length is not correct!!! Please check!")

            current_coefffs = compute_wavelet_descriptor(beat,wavelet,level)
            
            wavelet_cofficients.append(current_coefffs)
        return wavelet_cofficients
    # if we choose a large windos contains more than one beat, for example three beats, we need to ignore the first and last beats to reduce the zero paddings
    else:
        for i in range(1,num_beats-1,3): 
            # if i==0 or i==num_beats-1: 
            #     wavelet_cofficients.append([])
            #     continue
            if R_peak_index[i]+windowL<0:
                zeros_padding = np.array([0]*abs(R_peak_index[i]+windowL))

                beat = np.append(zeros_padding,original_signal[:R_peak_index[i]+windowR])   # the first and last beat has different length which may lead to different number of coefficient
                                                                # zero padding may need to add.
            elif R_peak_index[i]+windowR>=len_signal:
                beat = np.append(original_signal[R_peak_index[i]+windowL:],[0]*(R_peak_index[i]+windowR-len_signal))
            else:
                beat = original_signal[R_peak_index[i]+windowL:R_peak_index[i]+windowR]
            
            if len(beat)!=(windowR-windowL): 
                raise Exception("The beat length is not correct!!! Please check!")

            current_coefffs = compute_wavelet_descriptor(beat,wavelet,level)
            
            wavelet_cofficients.append(current_coefffs)
        return wavelet_cofficients

    
def compute_wavelet_descriptor(beat, family, level):
    # Compute the wavelet descriptor for a beat
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]


