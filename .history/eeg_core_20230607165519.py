import sys, os
sys.path.append('xai-kit/')
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from eeg.eeg_config import EEGConfig as Hub
from tframe import Classifier
from tframe import Predictor

from tframe.configs.config_base import Config
import eeg_du as du
from tframe import DataSet
import eeg.eeg_proc as ep
from eeg.optimizer import Optimizer


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.50

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.sequence_length = 512

th.input_shape = [1]
th.size = 46200

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------

## Loss Optimizer
th.loss_prop = 11.0
optimizer = Optimizer(th.loss_prop)
# th.validate_cycle = 20
#
th.num_steps = -1
th.print_cycle = 5
th.sample_num = 2

th.save_model = True
th.gather_note = True

th.signal_info = []

th.batch_size = 20
th.early_stop = True
th.patience = 5
th.validation_per_round = 2

th.export_tensors_upon_validation = True

th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True

th.val_batch_size = 20
th.eval_batch_size = 20


def activate():
  from eeg.eeg_agent import EEG

  # This line must be put in activate
  assert callable(th.model)
  if 'rnn' in th.developer_code:
    model = th.model(th)
  else:
    model = th.model()
  assert isinstance(model, Predictor)

  # Load data
  train_set, val_set, test_set = du.load(th.data_dir, th.train_per,
                                         th.combin_num, th.noise_type,
                                         th.size)


  # Train or evaluate
  if th.train:
    # add_datasets = EEG.data_validate_SNRlevel(th.noise_type, th.data_dir)
    # th.additional_datasets_for_validation.extend(add_datasets)
    if th.loss_probe:
      model.train(train_set, validation_set=val_set, test_set=test_set,
                  probe= optimizer.set_probe, trainer_hub=th)
    else:
      model.train(train_set, validation_set=val_set, test_set=test_set, trainer_hub=th)
    # Get rrmse, cc metric
    # fs = 512 if th.noise_type == 'EMG_GAN' else 256
    # ep.power_ratio_result_save(model, add_datasets, fs)
  else:
    ## Evaluate on test set
    calibrated_signal = model.predict(data=test_set)
    original_signal = test_set.features
    reference_signal = test_set.targets
    ep.siganl_visualize(reference_data=reference_signal,
                        original_data=original_signal,
                        calibrated_data=calibrated_signal)

  # End
  model.shutdown()
  console.end()


