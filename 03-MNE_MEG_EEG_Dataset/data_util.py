import os, sys
sys.path.append('../xai-kit/roma')
sys.path.append('../xai-kit/pictor')
sys.path.append('../')
from importlib.machinery import SourceFileLoader
from eeg_core import th
from eeg.eeg_set import EEGSet
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
    
    

def data_denoise():
    ## load proposed method
    module_path = os.path.join(os.getcwd(), 'checkpoints/0609_semi_simulated_model/0609_semi_simulated.py')
    task_module = SourceFileLoader('task', module_path).load_module()
    task_module.main('*')
    th.prefix = '0609_'
    th.overwrite = False
    model = th.model()

    data_types = ['train', 'validate', 'test']
    for data_type in data_types:
        raw_data = np.load('denoised_data/x_{}.npy'.format(data_type))
        ## data normalize
        eeg_standard = data_normalize(raw_data)
        ## data resample from 150hz to 200hz 
        data_200hz = scipy.signal.resample_poly(eeg_standard, up=4, down=3, axis=-1)
 
        feature = data_200hz
        original_shape = feature.shape
        feature = feature.reshape(-1, 202)
        feature = np.expand_dims(feature, axis=2)
        print('Feature Shape:{}'.format(feature.shape))
        proposed_targets = model.predict(EEGSet(feature), batch_size=128)
        proposed_targets = np.squeeze(proposed_targets)
        proposed_targets = proposed_targets.reshape(original_shape)

        ## data resample from 200hz to 150hz
        data_150hz = scipy.signal.resample_poly(eeg_standard, up=3, down=4, axis=-1)
        ## data normalize
        data_denoise = data_normalize(data_150hz)

        ## Result Saved
        np.save('denoised_data/x_{}_proposed_denoise.npy'.format(data_type), data_denoise[:, :, :151])
        print('{} Denoise Result Saved'.format(data_type))


def data_normalize(raw_data):
    raw_shape = raw_data.shape
    data = raw_data.reshape(-1, raw_shape[-1])
    eeg_mean = np.mean(data, axis=1, keepdims=True)
    eeg_std = np.std(data, axis=1, keepdims=True)
    eeg_standard = (data - eeg_mean) /eeg_std
    eeg_standard = eeg_standard.reshape(raw_shape)

    return eeg_standard



if __name__ == '__main__':
    data_denoise()