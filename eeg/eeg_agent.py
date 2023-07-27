from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import math

from tframe import checker
from tframe import console
from tframe.data.base_classes import DataAgent
from tframe.data.sequences.seq_set import SequenceSet
from tframe import DataSet
from eeg.eeg_set import EEGSet



class EEG(DataAgent):
  """Rf signal problem"""
  DATA_NAME = 'EEGSignal'


  @classmethod
  def load(cls, data_dir, train_per=0.8, noise_type='EMG', combin_num=10, size=-1):
    EEG_data, noise_data = cls._data_get(data_dir, noise_type, size)
    # Load dataset
    train_set = cls._dataset_config(data_dir, EEG_data, noise_data,
                                    train_per, 'train', combin_num, noise_type)
    val_set = cls._dataset_config(data_dir, EEG_data, noise_data,
                                  train_per, 'val', combin_num, noise_type)
    test_set = cls._dataset_config(data_dir, EEG_data, noise_data, train_per,
                                   'test', combin_num, noise_type)

    return train_set, val_set, test_set


  @classmethod
  def _dataset_config(cls, data_dir, EEG_data, noise_data,  train_per,
                      data_type, combin_num, noise_type, file_name=None):
    from eeg_core import th
    dataset = cls.load_as_tframe_data(data_dir, EEG_data, noise_data, train_per,
                                      data_type, combin_num, noise_type,
                                      file_name, dataset=th.dataset)
    x = np.stack(dataset.features, axis=0)
    y = np.stack(dataset.targets, axis=0)
    return DataSet(x, y, name=data_type)



  @classmethod
  def load_as_tframe_data(cls, data_dir, EEG_data, noise_data, train_per,
                          data_type, combin_num, noise_type, file_name=None,
                          dataset='eegdenoisenet'):
    # Check file_name
    prefix = data_type + '_'
    if file_name is None:
      prefix += dataset + '_'
      file_name = cls._get_file_name(size=EEG_data.shape[0], L= EEG_data.shape[1],
                                     prefix= prefix, noise_type=noise_type)
      file_name = file_name + '.tfds'
    save_dir = os.path.join(data_dir)
    data_path = os.path.join(save_dir, file_name)
    if os.path.exists(data_path): return EEGSet.load(data_path)
    # If data does not exist, create a new data set
    console.show_status('Creating data ...')

    # Load data as numpy arrays
    features, targets = cls.load_as_numpy_arrays(EEG_data, noise_data,
                                                 train_per, data_type,
                                                 combin_num, noise_type,
                                                 dataset)

    # Wrap data into tframe
    data_set = EEGSet(
      features, targets, n_to_one=False, noise_type=noise_type,
      name='EEGData')
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, EEG_data, noise_data, train_per, data_type,
                           combin_num, noise_type, dataset):
    if dataset == 'eegdenoisenet':
      clean_data, noise_data = cls._signal_preproc_denoisenet(EEG_data,
                                      noise_data, train_per,
                                      data_type, combin_num, noise_type)
    elif dataset == 'semi_simulated':
      clean_data, noise_data = cls._signal_preproc_semi_simulated(data_type)
    else:
      AssertionError('No dataset defined in SmartHub')

    return noise_data, clean_data


  @classmethod
  def _get_file_name(cls, size, L, prefix=None, noise_type=None):
    # `fL` for fixed length and `vL` for variant length
    checker.check_positive_integer(size)
    checker.check_positive_integer(L)
    if prefix==None:
      file_name = noise_type + '_EEG_{}_L{}'.format(size, L)
    else:
      file_name = prefix +noise_type +'_EEG_{}_L{}'.format(size, L)
    return file_name


  @classmethod
  def _get_rms(cls, records):
      return math.sqrt(sum([x ** 2 for x in records]) / len(records))


  @classmethod
  def _random_signal(cls, signal, combin_num):
      # Random disturb and augment signal
      random_result=[]

      for i in range(combin_num):
          random_num = np.random.permutation(signal.shape[0])
          shuffled_dataset = signal[random_num, :]
          shuffled_dataset = shuffled_dataset.reshape(signal.shape[0],signal.shape[1])
          random_result.append(shuffled_dataset)

      random_result  = np.array(random_result)

      return  random_result


  @classmethod
  def _data_get(cls, data_dir, noise_type, size):
    if noise_type == 'EOG':
      EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs.npy'))
      noise_all = np.load(os.path.join(data_dir, 'EOG_all_epochs.npy'))
    elif noise_type == 'EMG':
      EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs_512hz.npy'))
      noise_all = np.load(os.path.join(data_dir, 'EMG_all_epochs_512hz.npy'))
    elif noise_type == 'EMG_EOG':
      EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs.npy'))
      noise_EOG = np.load(os.path.join(data_dir, 'EOG_all_epochs.npy'))
      noise_EMG = np.load(os.path.join(data_dir, 'EMG_all_epochs.npy'))[:noise_EOG.shape[0]]
      noise_all = noise_EOG + noise_EMG

    if (size > EEG_all.shape[0]) or (size > noise_all.shape[0]):
        raise TypeError('"Size" is larger than dataset size')

    # Here we use eeg and noise signal to generate scale transed training, validation, test signal
    EEG_all_random = np.squeeze(cls._random_signal(signal = EEG_all[:size],
                                                   combin_num = 1))
    noise_all_random = np.squeeze(cls._random_signal(signal = noise_all[:size],
                                                     combin_num = 1))
    if noise_type == 'EMG':  
    # Training set will Reuse some of the EEG signal to much the number of EMG_GAN
      reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
      EEG_reuse = EEG_all_random[0 : reuse_num, :]
      EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
      print('EEG segments after reuse: ',EEG_all_random.shape[0])

    elif noise_type == 'EOG' or noise_type == 'EMG_EOG':  
    # We will drop some of the EEG signal to much the number of EMG_GAN
      EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]
      print('EEG segments after drop: ',EEG_all_random.shape[0])

    return EEG_all_random, noise_all_random


  @classmethod
  def _signal_preproc_semi_simulated(cls, data_type):
    contaminated_signal = np.load('/Software/summer/xai-alfa/54-EEG/eeg/semi_simulated/signal_Semi-simulated EOG.npy', allow_pickle=True)
    ground_truth = np.load('/Software/summer/xai-alfa/54-EEG/eeg/semi_simulated/reference_Semi-simulated EOG.npy', allow_pickle=True)
    data_type = 'train'
    if data_type == 'test':
      eeg_data = ground_truth[:, 0, :] 
      noise_data = contaminated_signal[:, 0, :] 
    elif data_type == 'val':
      eeg_data = ground_truth[:, 1, :] 
      noise_data = contaminated_signal[:, 1, :] 
    elif data_type == 'train':
      eeg_data = ground_truth[:, 2:, :] 
      noise_data = contaminated_signal[:, 2:, :] 
    
    eeg_data = np.array(eeg_data).reshape((-1, 540))
    noise_data = np.array(noise_data).reshape((-1, 540))
    return eeg_data, noise_data


  @classmethod
  def _signal_preproc_denoisenet(cls, EEG_data, noise_data, train_per, data_type,
                      combin_num, noise_type):
    timepoint = noise_data.shape[1]
    train_num = round(train_per * EEG_data.shape[0])
    val_num = round((1-train_per) * EEG_data.shape[0] / 2)
    if data_type == 'train':
      eeg_data = EEG_data[0: train_num, :]
      noise_data = noise_data[0: train_num, :]
    elif data_type == 'val':
      eeg_data = EEG_data[train_num: train_num+ val_num, :]
      noise_data = noise_data[train_num: train_num+ val_num, :]
    elif data_type == 'test':
      eeg_data = EEG_data[train_num+ val_num:, :]
      noise_data = noise_data[train_num+ val_num:, :]
    else:
      raise TypeError('Data Type is not correct!')

    eeg_data = cls._random_signal(signal = eeg_data, combin_num=combin_num).\
      reshape(combin_num * eeg_data.shape[0], timepoint)
    noise_data = cls._random_signal(signal=noise_data,combin_num=combin_num).\
      reshape(combin_num * noise_data.shape[0], timepoint)

    #create random number between -7dB ~ 2dB
    if noise_type == 'EMG':
        SNR_train_dB = np.random.uniform(-7, 4, (eeg_data.shape[0]))
    elif noise_type == 'EOG' or noise_type == 'EMG_EOG':
        SNR_train_dB = np.random.uniform(-7, 2, (eeg_data.shape[0]))
    else:
        raise TypeError('Noise Type Error!')

    print(SNR_train_dB.shape)
    SNR_train = 10 ** (0.1 * (SNR_train_dB))

    # combin eeg and noise for training set
    noiseEEG_data=[]
    for i in range (eeg_data.shape[0]):
        eeg= eeg_data[i].reshape(eeg_data.shape[1])
        noise=noise_data[i].reshape(noise_data.shape[1])

        coe= cls._get_rms(eeg)/(cls._get_rms(noise)*SNR_train[i])
        noise = noise * coe
        neeg = noise + eeg

        noiseEEG_data.append(neeg)

    noiseEEG_data=np.array(noiseEEG_data)

    # variance for noisy EEG
    EEG_data_end_standard = []
    noiseEEG_data_end_standard = []

    for i in range(noiseEEG_data.shape[0]):
        # Each epochs divided by the standard deviation
        eeg_data_all_std = eeg_data[i] / np.std(noiseEEG_data[i])
        EEG_data_end_standard.append(eeg_data_all_std)

        noiseeeg_data_end_standard = noiseEEG_data[i] / np.std(noiseEEG_data[i])
        noiseEEG_data_end_standard.append(noiseeeg_data_end_standard)

    noiseEEG_data_end_standard = np.array(noiseEEG_data_end_standard)
    EEG_data_end_standard = np.array(EEG_data_end_standard)
    print('{} data prepared'.format(data_type), noiseEEG_data_end_standard.shape,
          EEG_data_end_standard.shape )

    return EEG_data_end_standard, noiseEEG_data_end_standard


  @classmethod
  def data_validate_SNRlevel(cls, noise_type, data_dir, num=100, save_data=False):
    from eeg_core import th

    if noise_type == 'EOG_GAN':
        noise_type = 'EOG'
        EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs.npy'))
        noise_all = np.load(os.path.join(data_dir, 'EOG_all_epochs.npy'))
    elif noise_type == 'EMG_GAN':
        noise_type = 'EMG'
        EEG_all = np.load(os.path.join(data_dir, 'EEG_all_epochs_512hz.npy'))
        noise_all = np.load(os.path.join(data_dir, 'EMG_all_epochs_512hz.npy'))

    val_dataset =[]
    val_data_input =[]
    val_data_output =[]
    for SNR in np.arange(-7, 3):
        eeg_data = EEG_all[: num]
        noise_data = noise_all[: num]

        snr = 10 ** (0.1 * SNR)

        noiseEEG_data=[]
        for i in range (eeg_data.shape[0]):
            eeg= eeg_data[i].reshape(eeg_data.shape[1])
            noise=noise_data[i].reshape(noise_data.shape[1])

            coe= cls._get_rms(eeg)/(cls._get_rms(noise)*snr)
            noise = noise * coe
            neeg = noise + eeg

            noiseEEG_data.append(neeg)
        noiseEEG_data=np.array(noiseEEG_data)

        # variance for noisy EEG
        EEG_data_end_standard = []
        noiseEEG_data_end_standard = []

        for i in range(noiseEEG_data.shape[0]):
            # Each epochs divided by the standard deviation
            eeg_data_all_std = eeg_data[i] / np.std(noiseEEG_data[i])
            EEG_data_end_standard.append(eeg_data_all_std)

            noiseeeg_data_end_standard = noiseEEG_data[i] / np.std(noiseEEG_data[i])
            noiseEEG_data_end_standard.append(noiseeeg_data_end_standard)

        noiseEEG_data_end_standard = np.array(noiseEEG_data_end_standard)
        EEG_data_end_standard = np.array(EEG_data_end_standard)

        noiseEEG_data_end_standard = np.expand_dims(noiseEEG_data_end_standard, axis=2)
        EEG_data_end_standard = np.expand_dims(EEG_data_end_standard, axis=2)
        val_dataset.append(DataSet(noiseEEG_data_end_standard, EEG_data_end_standard,
                                   name='SNR_{}dB'.format(SNR)))
        val_data_input.append(noiseEEG_data_end_standard)
        val_data_output.append(EEG_data_end_standard)
    ## Save data
    if save_data:
      save_path = os.path.abspath(os.path.join(os.getcwd(), 'benchmark'))
      np.save(save_path+ '/' + noise_type +'_data_input.npy', np.array(val_data_input))
      np.save(save_path+ '/' + noise_type +'_data_output.npy', np.array(val_data_output))
      print(noise_type + ' data Saved in {}'.format(save_path))
    return val_dataset



if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
    train_set, valid_set, test_set = EEG.load(data_dir, size=10, noise_type='EMG_EOG')
    