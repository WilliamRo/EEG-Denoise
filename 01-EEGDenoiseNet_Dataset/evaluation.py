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
from eeg.eeg_set import EEGSet
import argparse



def add_arguments(parser):
    parser.add_argument('--noise_type', choices=['EMG', 'EOG', 'EMG_EOG'], default='EMG', help='Artifact Type')

    return parser


def model_performance(features, clean, noise_type):
    from eeg_core import th
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_{}_denoise/0609_{}_denoise.py'.format(noise_type, noise_type))
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.visible_gpu_id = 0
    th.overwrite = False
    model = th.model()
    proposed_targets = []
    for i in range(clean.shape[0]):
        feature = np.expand_dims(features[i], axis=2)
        # feature = features[i]
        proposed_targets.append(model.predict(EEGSet(feature), batch_size=32))
    proposed_targets = np.squeeze(proposed_targets)

    ## Metric Get
    fs = 512 if noise_type =='EMG' else 256
    snr = np.arange(-7, 3)
    rrmse_t_proposed = []
    rrmse_s_proposed = []
    cc_proposed = []
    delta_snr_proposed = []

    for index in np.arange(len(features)):
        data =np.stack([np.array(clean[index]), np.array(proposed_targets[index])],
                       axis=0)
        rrmse_t_proposed.append(metric.rrmse_temporal_metric(data))
        rrmse_s_proposed.append(metric.rrmse_spectral_metric(data, fs=fs))
        cc_proposed.append(metric.correlation_coefficient_metric(data))
        delta_snr_proposed.append(metric.snr_metric(data))

    ## Print the Average Performance Metric
    print('Average RRMSE_t->{}'.format(np.mean(rrmse_t_proposed)))
    print('Average RRMSE_s->{}'.format(np.mean(rrmse_s_proposed)))
    print('Average CC->{}'.format(np.mean(cc_proposed)))
    print('Average Delta SNR->{}'.format(np.mean(delta_snr_proposed)))

    ## Metric Show
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    ax[0].plot(snr, rrmse_t_proposed)
    ax[0].set_title('RRMSE temporal')
    ax[0].set_xlabel('SNR(db)')

    ax[1].plot(snr, rrmse_s_proposed)
    ax[1].set_title('RRMSE spectural')
    ax[1].set_xlabel('SNR(db)')

    ax[2].plot(snr, cc_proposed)
    ax[2].set_title('CC')
    ax[2].set_xlabel('SNR(db)')

    ax[3].plot(snr, delta_snr_proposed)
    ax[3].set_title('ΔSNR')
    ax[3].set_xlabel('SNR(db)')
 
    fig.legend(['Proposed'], loc='upper center', ncol=8)
    
    plt.show()

    return


def signal_visualize(features, clean, noise_type):
    from eeg_core import th
    ## load proposed method
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_{}_denoise/0609_{}_denoise.py'.format(noise_type, noise_type))
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.visible_gpu_id = 1
    th.overwrite = False
    model = th.model()
    proposed_targets = []
    for i in range(clean.shape[0]):
        feature = np.expand_dims(features[i], axis=2)
        proposed_targets.append(model.predict(EEGSet(feature), batch_size=32))
    proposed_targets = np.squeeze(proposed_targets)

    fs = 512 if noise_type == 'EMG' else 256

    ## Choose one sample trial from signal
    target = target[100]
    clean = clean[100]
    feature = feature[100]
    feature = np.squeeze(feature)
    psd_x, psd_target = signal_psd(target, fs, noise_type)
    psd_x, psd_clean = signal_psd(clean, fs, noise_type)
    psd_x, psd_feature = signal_psd(feature, fs, noise_type)

    fig, ax = plt.subplots(2, 1, figsize = (8, 8))
    x = np.arange(target.shape[-1])/512

    color = 'purple'

    ax[0].plot(x, target, linewidth=2.0, color=color)
    ax[0].plot(x, clean, linewidth=2.0, alpha=0.5,color='k')
    ax[0].plot(x, feature, linewidth=2.0, alpha=0.5, color='#808080')
    ax[0].set_xlabel('Time(sec)')
    
    ax[1].plot(psd_x[:160], psd_target[:160], linewidth=2.0, color= color)
    ax[1].plot(psd_x[:160], psd_clean[:160], alpha=0.5, linewidth=2.0, color='k')
    ax[1].plot(psd_x[:160], psd_feature[:160], alpha=0.5, color='#808080')
    ax[1].set_xlabel('Frequency(Hz)')
    fig.legend(['Denoised signal', 'Ground Truth', 'Noise signal'], loc='upper center', ncol=3)
    plt.show()


def signal_psd(data_in, fs, noise_type):
    from scipy import signal as sig
    fft_length = 1024 if noise_type == 'EMG' else 512
    f, pxx = sig.welch(data_in, fs, nfft=fft_length, nperseg=fft_length)
    return f, 10*np.log10(pxx) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()

    ## Load data at different SNR levels
    # noise_type = ['EMG', 'EOG', 'EOG+EMG'][2]
    noise_type = args.noise_type
    features = np.squeeze(np.load('test_data/{}_data_input.npy'.format(noise_type))) 
    clean = np.squeeze(np.load('test_data/{}_data_output.npy'.format(noise_type))) 

    ## Evaluate Performance
    model_performance(features=features, clean=clean, noise_type=noise_type)