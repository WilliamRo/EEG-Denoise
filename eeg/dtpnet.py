from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import typing as tp
import tensorflow as tf

from tframe.layers.convolutional import Conv1D
from tframe.layers.layer import Layer
from tframe.layers.hyper.conv import Conv1D as HyperConv1D

from tframe.nets.classic.conv_nets.conv_net import ConvNet
from tframe.nets.forkmerge import ForkMergeDAG
from tframe.layers.common import Activation 
from tframe.layers.layer import LayerWithNeurons, Layer, single_input


class DTPNet(ConvNet):

  def __init__(self, N, L, B, H, P, X, R):
      """
      Args:
          T (_type_): Length of input data
          N (_type_): Number of filters in autoencoder
          L (_type_): Length of the filters (in samples)
          B (_type_): Number of channels in bottleneck
          H (_type_): Number of channels in convolutional blocks
          P (_type_): Kernel Size in convolutional block
          X (_type_): Number of convolutional blocks in each repeat
          R (_type_): Number of repeats
          norm_type (_type_): 'glN' or 'clN' or 'blN'
      """
      self.R = R
      self.X = X
      self.H = H
      self.P = P
      self.N = N
      self.L = L
      self.B = B

      self.encoder = Encoder(filters=N, kernel_size=L, strides=int(L/2), padding='valid')
      self.bottleneck_conv = Conv1D(filters=B, kernel_size=1)

  def _get_layers(self):
    layers = []
    sub_blocks = []
    # (1) Define Encoder
    ## [?, L, 1] -->> [?, K, N]
    layers.append(self.encoder)

    # (2) Define TemporalConvNet
    ## [?, K, N] -->> [?, K, B]
    layers.append(self.bottleneck_conv)
    ## [?, K, B] -->> [?, K, B]
    for _ in range(self.R):
        for x in range(self.X):
            dilation = 2**x
            layers.append(self._temporal_sample_block(in_channels=self.B, channels=self.H, kernel_size=self.P,
                                               stride=1, dilation=dilation))
            sub_blocks.append(layers[-1])
            layers.append(Multi_Concat(sub_blocks[:-1]))

    layers.append(Activation.LeakyReLU(0.25))
    # [?, K, B] -->> [?, K, N]
    layers.append(Conv1D(self.N, kernel_size=1))
    layers.append(Activation.LeakyReLU(0.25))

    # (3) Decoder
    # [?, K, N] -->> [?, L, 1]
    layers.append(Decoder(encoder_layer=layers[0], L=self.L))

    return layers

  
  def _temporal_sample_block(self, in_channels, channels, kernel_size, stride, dilation):
   vertices = [[
               ## [?, K, B]-->> [?, K, H]
               Conv1D(filters=channels, kernel_size=1, strides=1, padding='same'),  
               Activation.LeakyReLU(0.25),
               Conv1D(filters=channels, kernel_size=kernel_size, strides=stride, dilation_rate=dilation),
               Activation.LeakyReLU(0.25),
               ## [?, K, H]-->> [?, K, B]
               Conv1D(filters=in_channels, kernel_size=1, strides=1, padding='same')],  
              ]
   edges = '1'

   return ForkMergeDAG(vertices, edges, name='ModNetBlock')


class Encoder(HyperConv1D):
  full_name = 'encoder'
  abbreviation = 'encoder'

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               dilations=1,
               activation='relu',
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               expand_last_dim=False,
               use_batchnorm=False,
               filter_generator=None,
               name: tp.Optional[str] = None,
               **kwargs):

    # Call parent's constructor
    super(Encoder, self).__init__(
      filters, kernel_size, strides, padding, dilations, activation,
      use_bias, kernel_initializer, bias_initializer, expand_last_dim,
      use_batchnorm, filter_generator, name, **kwargs)


  def get_layer_string(self, scale, full_name=False, suffix=''):
    result = super().get_layer_string(scale, full_name, suffix)
    return result

  def forward(self, x, **kwargs):
    ## output [?, K, N]
    mixture_w = self.conv1d(x, self.channels, self.kernel_size, 'Mixture',
      strides=self.strides, padding=self.padding, dilations=self.dilations, **kwargs)                             

    return mixture_w


class Decoder(LayerWithNeurons):
  full_name = 'decoder'
  abbreviation = 'decoder'

  def __init__(self,
               encoder_layer,
               L,
               activation=None,
               use_bias=False,
               weight_initializer='xavier_normal',
               bias_initializer='zeros',
               prune_frac=0,
               **kwargs):
    # Call parent's constructor
    LayerWithNeurons.__init__(
      self, activation, weight_initializer, use_bias, bias_initializer,
      prune_frac=prune_frac, **kwargs)

    self.L = L
    self.encoder_layer = encoder_layer

  def get_layer_string(self, scale, full_name=False, suffix=''):
    result = super().get_layer_string(scale, full_name, suffix)
    return result

  def forward(self, x, **kwargs):
      from eeg_core import th
      """_summary_

      Args:
          x (tf.Tensor): Estimation Mask, Shape [?, K, N, C]
          decoder_layer (HyperConv1D): Decoder Layer

      Returns:
          _type_: _description_
      """
      ## Get Mixture Tensor [?, K, N]
      if self.encoder_layer != None:
        mixture = self.encoder_layer.output_tensor
        source = mixture + x                                       #[?, K, N]
      else:
        source = x

      ## Deconvolution
      ## output [?, T, 1]
      est_source = self.deconv1d(source, output_channels= 1, filter_size= self.L, strides=self.L//2, padding='VALID', scope='Mixture')

      return est_source


class Multi_Concat(Layer):
  full_name = 'multi_concat'
  abbreviation = 'multi_concat'
  is_nucleus = False

  def __init__(self, layers:list):
    """Concat the input tensor"""
    self.layers = layers

  @property
  def structure_tail(self):
    return 'Concat'

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    output = x
    for layer in self.layers:
       tensor = layer.output_tensor 
       output = tf.concat([output, tensor], axis=-1)
    return output


