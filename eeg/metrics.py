from tframe import tf
from tframe.core.quantity import Quantity
import numpy as np



def trrmse_metric(y_true, y_predict):
    return (_get_rms(y_predict - y_true)/_get_rms(y_true))


def frrmse_metric(y_true, y_predict):
    psd_y_true = _get_psd(y_true)
    psd_y_predict = _get_psd(y_predict)
    return (_get_rms(psd_y_predict - psd_y_true))/\
           _get_rms(psd_y_true)


def cc_metric(y_true, y_predict):
    x, y = y_true, y_predict
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x-mx, y-my
    numerator = tf.reduce_sum(xm * ym, axis=1)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1)*
                          tf.reduce_sum(tf.square(ym), axis=1))

    return 0 - numerator/denominator


def _get_rms(tensor):
    return tf.sqrt(tf.reduce_mean(tf.square(tensor), axis=1))


def _get_psd(tensor):
    if tensor.shape.ndims == 3:
        data = tf.squeeze(tf.cast(tensor, dtype=tf.complex64), squeeze_dims=2)
    else:
        data = tf.cast(tensor, dtype=tf.complex64)
    psd = tf.abs(tf.square(tf.spectral.fft(data))*2)[:, :240]
    # return 20*_tf_log10(psd)
    return psd


def get_metric(name:str):
    if name == 'RRMSE_temporal':
        kernel = trrmse_metric
    elif name == 'RRMSE_spectrum':
        kernel = frrmse_metric
    elif name == 'CC':
        kernel = cc_metric
    else:
        raise TypeError('No Metric Used')
    return Quantity(kernel, tf.reduce_mean, None, False, name=name,
                    lower_is_better=True, use_logits=False)


