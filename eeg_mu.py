from tframe import Classifier
from tframe import Predictor
from tframe import tf

from tframe.configs.config_base import Config
from tframe import mu
from tframe.core.quantity import Quantity

from eeg_core import th
import os
from tframe import context



def typical(th, cells, metric):
  assert isinstance(th, Config)
  # Initiate a model
  model = Predictor(mark=th.mark, net_type=mu.Recurrent)
  # Add layers
  model.add(mu.Input(sample_shape=th.input_shape))
  # Add hidden layers
  if not isinstance(cells, (list, tuple)): cells = [cells]
  for cell in cells: model.add(cell)
  # Build model and return
  output_and_build(model, th, metric)
  return model


def get_container(flatten = False):
  model = Predictor(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if flatten: model.add(mu.Flatten())
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  context.put_into_pocket('model', model)
  return model


def finalize(model, metric, dense=True):
  import eeg.metrics as metrics

  assert isinstance(model, Predictor)

  # Build model
  trrmse = metrics.get_metric('RRMSE_temporal')
  srrmse = metrics.get_metric('RRMSE_spectrum')
  cc = metrics.get_metric('CC')

  if metric == 'mse':
    model.build(loss= 'mse', metric=[trrmse, 'mse',  srrmse, cc], 
                batch_metric=['RRMSE_temporal', 'mse', 'RRMSE_spectrum', 'CC'])
  else:
    AssertionError('Metric is not defined in SmartHub')
  return model


def output_and_build(model, th, metric):
  import eeg.metrics as metrics

  assert isinstance(model, Predictor)
  assert isinstance(th, Config)
  # Add output layer
  model.add(mu.Dense(num_neurons=th.output_dim))

  # Build model
  trrmse = metrics.get_metric('RRMSE_temporal')
  srrmse = metrics.get_metric('RRMSE_spectrum')
  cc = metrics.get_metric('CC')

  if metric == 'mse':
    model.build(loss= 'mse', metric=['mse', trrmse, srrmse, cc],
                batch_metric='mse')
  else:
    AssertionError('Metric is not defined in SmartHub')




