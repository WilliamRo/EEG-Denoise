import os, sys
sys.path.append('../xai-kit')
sys.path.append('../xai-kit/roma')
sys.path.append('../xai-kit/pictor')
sys.path.append('../')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import eeg.eval as metric
from importlib.machinery import SourceFileLoader
from eeg_core import th
from eeg.eeg_set import EEGSet



def model_performance(features, clean): 
    ## load proposed method
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_semi_simulated_model/0609_semi_simulated_model.py')
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.visible_gpu_id = 0
    th.overwrite = False
    model = th.model()
    ## predict
    feature = features.reshape(-1, 540)
    feature = np.expand_dims(feature, axis=2)
    proposed_denoise = model.predict(EEGSet(feature), batch_size=32)
    proposed_denoise = proposed_denoise.reshape(-1, 5400)
    ## get metric
    ground_truth = clean.reshape(-1, 5400)
    data = np.stack((ground_truth, proposed_denoise))
    rrmse_t = metric.rrmse_temporal_metric(data)
    rrmse_s = metric.rrmse_spectral_metric(data, fs=200)
    cc = metric.correlation_coefficient_metric(data)
    snr = metric.snr_metric(data)

    ## Print the Performance Metric
    print('RRMSE_t->{}'.format(np.mean(rrmse_t)))
    print('RRMSE_s->{}'.format(np.mean(rrmse_s)))
    print('CC->{}'.format(np.mean(cc)))
    print('Delta SNR->{}'.format(np.mean(snr)))


def signal_visualize_single_channel(features, clean):
    ## load proposed method
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_semi_simulated_model/0609_semi_simulated_model.py')
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.visible_gpu_id = 1
    th.overwrite = False
    model = th.model()

    feature = features.reshape(-1, 540)
    feature = np.expand_dims(feature, axis=2)
    proposed_denoise = model.predict(EEGSet(feature), batch_size=32)
    target = proposed_denoise.reshape(-1, 5400)
    feature = feature.reshape(-1, 5400)
    target = target[10]
    feature = feature[10]

    clean = clean[0][10]
    feature = np.squeeze(feature)
    fs = 200
    psd_x, psd_target = signal_psd(target, fs)
    psd_x, psd_clean = signal_psd(clean, fs)
    psd_x, psd_feature = signal_psd(feature, fs)

    fig, ax = plt.subplots(2, 1, figsize = (8, 8))
    x = np.arange(target.shape[-1])/fs
    color = 'purple'

    ax[0].plot(x, target, linewidth=2.0, color=color)
    ax[0].plot(x, clean, linewidth=2.0, alpha=0.5,color='k')
    ax[0].plot(x, feature, linewidth=2.0, alpha=0.5, color='#808080')
    ax[0].set_xlabel('Time(sec)')
    
    ax[1].plot(psd_x[:120], psd_target[:120], linewidth=2.0, color= color)
    ax[1].plot(psd_x[:120], psd_clean[:120], alpha=0.5, linewidth=2.0, color='k')
    ax[1].plot(psd_x[:120], psd_feature[:120], alpha=0.5, color='#808080')
    ax[1].set_xlabel('Frequency(Hz)')
    fig.legend(['Denoised signal', 'Ground Truth', 'Noise signal'], loc='upper center', ncol=3)
    plt.show()

    
def signal_visualize_multi_channel(features, clean):
    shape = feature.shape
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_semi_simulated_model/0609_semi_simulated_model.py')
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.overwrite = False
    th.visible_gpu_id = 0
    model = th.model()
    feature = feature.reshape(-1, 540)
    feature = np.expand_dims(feature, axis=2)
    proposed_denoise = model.predict(EEGSet(feature), batch_size=32)
    proposed_denoise = proposed_denoise.reshape(shape)
    feature = feature.reshape(shape)

    ## plot multi-channel, choose the first people
    feature, target, proposed_denoise = feature[10], target[10], proposed_denoise[10]
    xticks = np.arange(1000) /200
    for i in range(19):
        plt.plot(xticks, feature[i][2000:3000]+150*i, color='#808080', alpha=0.5)
        plt.plot(xticks, target[i][2000:3000] + 150*i, color='k', alpha=0.5)
        plt.plot(xticks, proposed_denoise[i][2000:3000] + 150*i, color='purple', alpha=0.5)
    plt.xlabel('Time(sec)')
    plt.yticks([])
    plt.show()
 


def signal_psd(data_in, fs, noise_type):
    from scipy import signal as sig
    fft_length = 600
    f, pxx = sig.welch(data_in, fs, nfft=fft_length, nperseg=fft_length)
    return f, 10*np.log10(pxx) 


 
if __name__=='__main__':
    feature = np.load(os.path.join(os.getcwd(), 'test_data/signal_Semi-simulated EOG.npy'), allow_pickle=True)
    clean = np.load(os.path.join(os.getcwd(),'test_data/reference_Semi-simulated EOG.npy'), allow_pickle=True)
    model_performance(feature, clean) 