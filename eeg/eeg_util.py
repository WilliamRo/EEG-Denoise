import numpy as np



def siganl_visualize(reference_data, original_data, calibrated_data):
  from eeg.eegviewer import EEGViewer
  from eeg_core import th

  fs = 512 if th.noise_type is 'EMG_GAN' else 256
  original_data = np.squeeze(original_data)
  calibrated_data = np.squeeze(calibrated_data)
  reference_data = np.squeeze(reference_data)
  data_in = np.stack(arrays=[reference_data, original_data,
                             calibrated_data], axis=0)
  signal_viewer = EEGViewer(fs, data_in)
  signal_viewer.signal_view()
