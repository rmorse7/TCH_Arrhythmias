import numpy as np
import matplotlib.pyplot as plt

time_max_normalized = 1.25
normalized_amplitude = 400
num_bins_x = 50
num_bins_y = 50


# ----------------------------------------------------
# Based off code from D2K MIC group from Spring 2020. Rewritten and optimized for beat displaying by Ricky Morse,
# D2K Arrhythmias Team Fall 2020.
# ----------------------------------------------------

def get_histogram(segments, resample_size=100, num_bins_y=50, amp=400, log=True):
    """
    Get a probability density vector of size (B1, B2), where B1 is the size of time after resampling (default 100)
    and B1 the number of bins for lead value (default 50)
    :param segments -- a list of beat segments
    :param resample_size -- size to resample the time axis
    :param num_bins_y -- number of bins for the value axis
    :param log -- boolean to turn into log likelihood
    """
    y_bin_width = 2. * (amp + 1) / num_bins_y
    hist = np.zeros((resample_size, num_bins_y))

    for segment in segments:
        x = np.linspace(segment[0, 0], segment[-1, 0], resample_size)
        y = np.interp(x, segment[:, 0], segment[:, 1])
        bins = list(((y // y_bin_width) + (num_bins_y//2)).astype(int))
        indices = list(zip(list(range(len(bins))), bins))
        for index in indices:
            try:
                hist[index] += 1
            except:
                print('Histogram index issue w/: ',index)

    hist = np.divide(hist, len(segments))
    if log:
        #dealing with zero entries in log scale
        nonzero = hist[hist != 0]
        hist[hist == 0] = np.min(nonzero) * 0.5
        hist = np.log(hist)
        lm = np.min(hist)
        #clearing up picture (if necesary)
        #for i in range(resample_size):
        #    for j in range(num_bins_y):
        #        if hist[i,j] < lm / 1.8:
        #            hist[i,j] = lm
    return hist


def plot_histogram(hist, title):
    """
    Plot the histogram
    """
    # Modify heatmap
    hist = np.row_stack((hist[40:, :], hist[:40, :]))
    plt.imshow(np.rot90(hist))
    plt.xlabel("Samples")
    plt.ylabel("Bin Number")
    log = True if np.sum(hist) < 0 else False
    # if log:
    #     title = "ECG Beat Log-likelihood"
    # else:
    #     title = "ECG Beat Likelihood"
    plt.title(title)
    clb = plt.colorbar()
    clb.set_label('log likelihood', labelpad=15, rotation=270)
    plt.show()


def get_segments_likelihood(hist, segments, num_parameters=1):
    """
    Returns the likelihood of "segments" in a (N, P) numpy array, where N is the number of segments
    and P is the number of parameters (default 1). If P > 1 is used, the likelihood vector is divided
    and averaged into P chunks.
    :param hist -- a histogram of size (B1, B2), see get_histogram for details
    :param segments -- list of segments
    :param num_parameters -- number of parameter per segment for output. Should be <= resample size of hist
    """
    num_bins_y = hist.shape[1]
    resample_size = hist.shape[0]
    y_bin_width = 2. * (normalized_amplitude + 1) / num_bins_y
    feature_matrix = np.zeros((len(segments), resample_size))

    for i in range(len(segments)):
        segment = segments[i]
        x = np.linspace(segment[0, 0], segment[-1, 0], resample_size)
        y = np.interp(x, segment[:, 0], segment[:, 1])
        bins = list(((y // y_bin_width) + 24).astype(int))
        indices = list(zip(list(range(len(bins))), bins))
        for j in range(resample_size):
            index = indices[j]
            feature_matrix[i, j] = hist[index]

    # Scale dimension 1 to num_parameter
    log = True if np.sum(feature_matrix) < 0 else False
    split_arr = np.array_split(feature_matrix, num_parameters, axis=1)
    reshaped_matrix = np.zeros((len(segments), num_parameters))
    for i in range(len(split_arr)):
        if log:
            compressed_arr = np.sum(split_arr[i], axis=1)
        else:
            compressed_arr = np.prod(split_arr[i], axis=1)
        reshaped_matrix[:, i] = compressed_arr
    return reshaped_matrix


def resample_segments(segments, resample_size):
    """
    Resample a list of segments using linear interpolation and return a N * M numpy matrix,
    where N is number of segments and M is the resample size
    """
    resampled = np.zeros((len(segments), resample_size))
    for i in range(len(segments)):
        segment = segments[i]
        x = np.linspace(segment[0, 0], segment[-1, 0], resample_size)
        y = np.interp(x, segment[:, 0], segment[:, 1])
        resampled[i, :] = y
    return resampled
