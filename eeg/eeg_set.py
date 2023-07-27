import os, sys

from tframe.data.sequences.seq_set import SequenceSet
from eeg.eegviewer import EEGViewer
import numpy as np

class EEGSet(SequenceSet):
  def __init__(self, features=None, targets=None, noise_type =None,
                                data_dict=None, summ_dict=None,
                                n_to_one=False, name='seqset'):
    SequenceSet.__init__(self, features=features, targets=targets,
                                data_dict=data_dict, summ_dict=summ_dict,
                                n_to_one=n_to_one, name=name)
    self.noise_type = noise_type


  def visualize(self):
    '''
    :param real_data: data to visulaize is real data or not
    :return:
    '''
    print('Visualizing ...')
    data, targets = self.features, self.targets
    data_in = np.stack(arrays=[data, targets], axis=0)
    self.signal_view(data_in)


  def signal_view(self, data_in):
    fs = 256 if self.noise_type == 'EOG_GAN' else 512
    data_in = np.squeeze(data_in)
    # data_in = [np.squeeze(data) for data in data_in]
    signal = EEGViewer(fs, data_in, plt_num=2)
    signal.signal_view()


  def convert_to(self, input_type):
    return None


if __name__ == '__main__':
    print(EEGSet.__mro__)