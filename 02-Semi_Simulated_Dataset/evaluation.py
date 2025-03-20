"""USAGE of this script

(1) To calculating metrics (e.g., RRMSE_t and CC) of the proposed network, set '--plot_type' to 'metric_report_wqn', 'metric_report_test'.
    - 'metric_report_wqn': evaluate model using WQN-style framework [1]
      For reference, we provide the comparison results of different models under 'metric_report_wqn' setting (reported in our paper)
        ---------------------------------------------------------------------------
        Metrics        1D-ResCNN    SimpleCNN    NovelCNN     WQN          Ours
        ===========================================================================
        RRMSE_t        0.300        0.270        0.168        0.265        0.0563
        RRMSE_s        0.785        0.719        0.185        0.513        0.0747
        Delta_SNR      12.3         13.1         13.6         12.7         29.3
        CC             0.946        0.955        0.980        0.956        0.995
        ---------------------------------------------------------------------------

    - 'metric_report_test': evaluate model on test set
      For reference, we provide the comparison results of different models under 'metric_report_test' setting
        ---------------------------------------------------------------------------
        Metrics        1D-ResCNN    SimpleCNN    NovelCNN     WQN          Ours
        ===========================================================================
        RRMSE_t        0.710        0.798        0.458        0.474        0.289
        RRMSE_s        1.22         1.55         0.481        0.769        0.457
        Delta_SNR      3.62         2.72         7.13         6.71         11.4
        CC             0.808        0.780        0.887        0.881        0.953
        ---------------------------------------------------------------------------

(2) To visualize the denoise result, set '--plot_type' to 'denoise_visualize', 'denoise_visualize_multi_channel'.
    - 'denoise_visualize': for single channel visualization
    - 'denoise_visualize_multi_channel': for multi-channel visualization

Reference
---------
[1] M. Dora and D. Holcman, “Adaptive single-channel EEG artifact removal with applications to clinical monitoring,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 30, pp. 286–295, 2022.
"""
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
import argparse



def add_arguments(parser):
    parser.add_argument('--plot_type', choices=['metric_report_wqn', 'metric_report_test', 'denoise_visualize', 'denoise_visualize_multi_channel'], default='result_SNR', help='Plot Result Type')

    return parser
    
    
def model_performance(features, clean): 
    ## load proposed method
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_semi_simulated_model/0609_semi_simulated_model.py')
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.visible_gpu_id = 1
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


def model_performance_test(features, clean): 
    ## get the test channel
    features = features[:, 0, :]
    clean = clean[:, 0, :]
    ## load proposed method
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_semi_simulated_model/0609_semi_simulated_model.py')
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.visible_gpu_id = 1
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

    
def signal_visualize_multi_channel(feature, clean):
    shape = feature.shape
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_semi_simulated_model/0609_semi_simulated_model.py')
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.overwrite = False
    th.visible_gpu_id = 1
    model = th.model()
    feature = feature.reshape(-1, 540)
    feature = np.expand_dims(feature, axis=2)
    proposed_denoise = model.predict(EEGSet(feature), batch_size=32)
    proposed_denoise = proposed_denoise.reshape(shape)
    feature = feature.reshape(shape)

    ## plot multi-channel, choose the first people
    feature, target, proposed_denoise = feature[10], clean[10], proposed_denoise[10]
    xticks = np.arange(1000) /200
    for i in range(19):
        plt.plot(xticks, feature[i][2000:3000]+150*i, color='#808080', alpha=0.5)
        plt.plot(xticks, target[i][2000:3000] + 150*i, color='k', alpha=0.5)
        plt.plot(xticks, proposed_denoise[i][2000:3000] + 150*i, color='purple', alpha=0.5)
    plt.xlabel('Time(sec)')
    plt.yticks([])
    plt.show()
 


def signal_psd(data_in, fs):
    from scipy import signal as sig
    fft_length = 600
    f, pxx = sig.welch(data_in, fs, nfft=fft_length, nperseg=fft_length)
    return f, 10*np.log10(pxx) 


 
if __name__=='__main__':
    parser = argparse.ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()

    plot_type = args.plot_type

    feature = np.load(os.path.abspath(os.path.join(os.getcwd(), '../EEG-Denoise_Database/02-Semi_Simulated_Dataset/signal_Semi-simulated EOG.npy')), allow_pickle=True)
    clean = np.load(os.path.abspath(os.path.join(os.getcwd(), '../EEG-Denoise_Database/02-Semi_Simulated_Dataset/reference_Semi-simulated EOG.npy')), allow_pickle=True)
    if plot_type == 'metric_report_wqn':
        model_performance(feature, clean) 
    elif plot_type == 'metric_report_test':
        model_performance_test(feature, clean) 
    elif plot_type == 'denoise_visualize': 
        signal_visualize_single_channel(feature, clean)
    elif plot_type == 'denoise_visualize_multi_channel':
        signal_visualize_multi_channel(feature, clean)
    else:
        AssertionError('Plot Type Error!!!')
