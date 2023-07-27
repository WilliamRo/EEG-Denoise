import numpy as np
import math
from scipy.stats.stats import pearsonr
from scipy import signal as sig
from sklearn.metrics import mean_squared_error



def rrmse_temporal_metric(data_in):
    """
    This function get the score based on rrmse_temporal to rank different networks
    :param data_in: input arrays, [0] is the ground truth signal, [1] is
                    network processed signal
    :return: rrmse temporal metric index
    """
    ground_truth = data_in[0]
    processed = data_in[1]
    metric = []
    for i in range(data_in.shape[1]):
        metric.append(_get_rms(ground_truth[i] - processed[i])/
                         _get_rms(ground_truth[i]))
    return np.average(metric)


def snr_metric(data_in):
    """
    This function get the score based on SNR level to rank different networks
    :param data_in: input arrays, [0] is the ground truth signal, [1] is
                    network processed signal
    :return: rrmse temporal metric index
    """ 
    ground_truth = data_in[0]
    processed = data_in[1]
    metric = []
    for i in range(data_in.shape[1]):
        metric.append(_get_snr(ground_truth[i], processed[i]))
    return np.average(metric)


def rrmse_spectral_metric(data_in, fs):
    """
    This function get the score based on rrmse_spectrral to rank different networks
    :param data_in: input arrays, [0] is the ground truth signal, [1] is
                    network processed signal
    :param fs: sample frequency of signal
    :return: rrmse spectral metric index
    """
    ground_truth = data_in[0]
    processed = data_in[1]
    metric = []
    for i in range(data_in.shape[1]):
        _, psd_x = _get_psd(ground_truth[i], fs)
        _, psd_fy = _get_psd(processed[i], fs)
        metric.append(_get_rms(psd_fy - psd_x) / _get_rms(psd_x))
    return np.average(metric)


def correlation_coefficient_metric(data_in):
    """
    This function get the score based on correlation coefficient rank
     different networks
    :param data_in: input arrays, [0] is the ground truth signal, [1] is
                    network processed signal
    :return: CC metric index
    """
    ground_truth = np.squeeze(data_in[0])
    processed = np.squeeze(data_in[1])
    metric = []
    for i in range(data_in.shape[1]):
        metric.append(pearsonr(ground_truth[i], processed[i])[0])
    return np.average(metric)


def _get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

    
def _get_snr(clean, denoise):
    noise = denoise - clean
    return 10*np.log10(np.sum(clean**2)/np.sum(noise**2))


def _get_psd(records, fs):
    freqs, psd = sig.welch(records, fs=fs, nfft=len(records),
                           nperseg=len(records))
    return freqs, psd
