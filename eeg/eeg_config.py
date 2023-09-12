from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class EEGConfig(SmartTrainerHub):
  visible_gpu_id = Flag.integer(0, 'The GPU Number in use', is_key=None)

  metric = Flag.string('mse', 'Loss Function used in task', is_key=None)

  noise_type = Flag.string('EOG', 'Noise Type used in task', is_key=None)

  plot_type = Flag.string('metric_report', 'Plot Result Type', is_key=None)

  dataset = Flag.string('eegdenoisenet', 'Dataset used in task', is_key=None)

  train_per = Flag.float(0.8, 'The proportion of the training data in the total data',
                         is_key=None)

  ## Network Structure parameter
  filter_num = Flag.integer(256, 'Number of filters in autoencoder', is_key=None)
  filter_length =Flag.integer(16, 'Length of the filters (in samples)', is_key=None)
  block_chnnum =Flag.integer(256, 'Number of channels in convolutional blocks', is_key=None)
  block_kernel_size =Flag.integer(7, 'Kernel Size in convolutional block', is_key=None)
  conv_num = Flag.integer(8, 'Number of convolutional blocks in each dilation block',
                             is_key=None)
  repeats_num =Flag.integer(4, 'Number of repeats', is_key=None)
  bottle_neck_channel = Flag.integer(32, 'Bottle neck channel numbers in the residual block',
                             is_key=None)

# New hub class inherited from SmartTrainerHub must be registered
EEGConfig.register()

