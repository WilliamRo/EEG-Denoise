import sys, os 
from tkinter.tix import Tree
sys.path.append('../xai-kit')
sys.path.append('../xai-kit/roma')
sys.path.append('../xai-kit/pictor')
sys.path.append('../')

from tframe import tf
from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir
from eeg.dtpnet import DTPNet     

import eeg_core as core
import eeg_mu as m



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'Modnet_Dense'
id = 1
def model():
  th = core.th
  model = m.get_container(flatten=False)
  net = DTPNet(N=th.filter_num, L=th.filter_length, B=th.bottle_neck_channel, H=th.block_chnnum,
                P=th.block_kernel_size, X=th.conv_num, R=th.repeats_num)
  print('Model Choice-->> BaseNet+TPB+Dense')
  net.add_to(model)

  return m.finalize(model, metric=th.metric, dense=False)



def main(_):
  console.start('{} on EEG Signal task'.format(model_name.upper()))

  th = core.th
  th.rehearse = True
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  ## Choose the artifacts type
  th.dataset = ['eegdenoisenet', 'semi_simulated'][1]
  th.sequence_length = None
  th.input_shape = [th.sequence_length, 1]
  th.output_dim = th.sequence_length

  ## The proportion of the training data in the total data, 
  ## the test and validation sets each account for half of the remaining data
  th.train_per = 0.8
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.visible_gpu_id = 1
  th.script_suffix = '_1'
  # ---------------------------------------------------------------------------
  # 2_1. model structure
  # ---------------------------------------------------------------------------
  th.filter_num = 512
  th.filter_length = 8
  th.bottle_neck_channel = 64
  th.block_chnnum = 64
  th.block_kernel_size = 3
  th.conv_num = 6
  th.repeats_num = 4
  # ---------------------------------------------------------------------------
  # 2_2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.activation = 'relu'
  th.dropout = 0.5
  th.kernel_size = 3
  th.use_batchnorm = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 2000
  th.batch_size = 32
  th.val_batch_size = 40

  th.patience = 30
  th.optimizer = 'adam'
  th.learning_rate = 0.000458
  th.optimizer_epsilon = 1e-7

  th.train =True
  th.overwrite = True
  th.print_cycle = 5
  th.probe_cycle = 5
  th.validation_per_round = 1

  th.save_model = True
  th.metric = 'mse'
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.clear_records_before_training = True
  th.export_tensors_upon_validation = True

  th.mark = '{}_{}_{}_{}'.format(model_name, th.activation, th.metric, th.noise_type)
  th.gather_summ_name = th.prefix + summ_name + th.noise_type + '.sum'

  if _ == '*': return
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()


