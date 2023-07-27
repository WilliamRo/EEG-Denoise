from tframe.data.perpetual_machine import PerpetualMachine
from eeg.eeg_agent import EEG
from tframe.data.sequences.seq_set import SequenceSet
from tframe import DataSet


def load(data_dir, train_per, noise_type):
  train_set, val_set, test_set = EEG.load(data_dir, train_per, noise_type)
  from eeg_core import th
  assert isinstance(train_set, DataSet)
  assert isinstance(test_set, DataSet)
  assert isinstance(val_set, DataSet)
  return train_set, val_set, test_set


