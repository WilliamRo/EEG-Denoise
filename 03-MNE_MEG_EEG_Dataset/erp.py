"""
 Sample script using Riemannian geometric approaches with xDAWN spatial filtering
 to classify Event-Related Potential (ERP) EEG data from a four-class classification
 task, using the sample dataset provided in the EEGNet [1] package:
    https://github.com/vlawhern/arl-eegmodels
   
 The four classes used from this dataset are:
     LA: Left-ear auditory stimulation
     RA: Right-ear auditory stimulation
     LV: Left visual field stimulation
     RV: Right visual field stimulation
 The code to process, epoch the data are originally from Alexandre
 Barachant's PyRiemann [2] package, released under the BSD 3-clause. A copy of 
 the BSD 3-clause license has been provided together with this software to 
 comply with software licensing requirements. Our contribution merely involves 
 replacing the original EEG data with the geniue EEG signal denoised by our 
 proposed method. 
 
 When you first run this script, MNE will download the dataset and prompt you
 to confirm the download location (defaults to ~/mne_data). Follow the prompts
 to continue. The dataset size is approx. 1.5GB download. 
 
 [1] Vernon J Lawhern, Amelia J Solon, Nicholas R Waytowich, Stephen M Gordon,
     Chou P Hung and Brent J Lance, EEGNet: a compact convolutional neural 
     network for EEG-based brain–computer interfaces, Journal of Neural 
     Enginerring, Volume 15, July 2018.
 [2] https://github.com/alexandrebarachant/pyRiemann. 
"""

import numpy as np
import os, sys

# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
import argparse



def add_arguments(parser):
    parser.add_argument('--denoise', action='store_true', help='Wether to use denoise data to classify')

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

##################### Process, filter and epoch the data ######################
data_path = sample.data_path()

# Whether Denoise the EEG Data in Raw Data or Not
parser = argparse.ArgumentParser(help)
add_arguments(parser)
args = parser.parse_args()
data_denoise = args.denoise         

# Set parameters and read data
raw_fname = str(data_path) + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = str(data_path) + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels

kernels, chans, samples = 1, 366, 151       ##[0:305] MEG// [305:365] EEG// [366] EOG

# take 50/25/25 percent of the data to train/validate/test
X_train      = X[0:144, :, :]
Y_train      = y[0:144]
X_validate   = X[144:216, :, :]
Y_validate   = y[144:216]
X_test       = X[216:, :, :]
Y_test       = y[216:]

if data_denoise:
    ## Load clean EEG data denoised by our proposed method
    denoise_train = np.load('denoised_data/x_train_denoise_150hz.npy')
    denoise_validate = np.load('denoised_data/x_validate_denoise_150hz.npy')
    denoise_test = np.load('denoised_data/x_test_denoise_150hz.npy')
    X_train[:, 305:365, :] = denoise_train[:, :, :151]
    X_validate[:, 305:365, :] = denoise_validate[:, :, :151]
    X_test[:, 305:365, :] = denoise_test[:, :, :151]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(Y_train-1)
Y_validate   = np_utils.to_categorical(Y_validate-1)
Y_test       = np_utils.to_categorical(Y_test-1)

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in 
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg     = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train      = X_train.reshape(X_train.shape[0], chans, samples)
X_test       = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, Y_train.argmax(axis = -1))
preds_rg     = clf.predict(X_test)

# Printing the results
acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
print("Classification accuracy: %f " % (acc2))

# plot the confusion matrices for both classifiers
names        = ['audio left', 'audio right', 'vis left', 'vis right']
plt.figure(0)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'PyReimann Classification')
plt.show()
