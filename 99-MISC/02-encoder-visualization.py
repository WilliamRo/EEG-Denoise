import sys, os 
from tkinter.tix import Tree
sys.path.append('/Software/summer/eeg-denoise-alpha/xai-kit')
sys.path.append('/Software/summer/eeg-denoise-alpha/xai-kit/roma')
sys.path.append('/Software/summer/eeg-denoise-alpha/xai-kit/pictor')
sys.path.append('/Software/summer/eeg-denoise-alpha')
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor.pictor import Plotter
from tframe.utils.note import Note

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy



class EncoderAnalyzer(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super().__init__(self.sort_and_plot)

    # Define settable attributes
    self.new_settable_attr('cmap', 'RdBu', str, 'Color map')
    self.new_settable_attr('wmax', 0.04, float, 'Cut-off value of weights')
    self.new_settable_attr('xmax', 0.14, float,
                           'Max value of x-axis in scatter plot')
    self.new_settable_attr('ymax', 22, float,
                           'Max value of y-axis in scatter plot')

    self.new_settable_attr('xper', 50, float, 'Minimum x-axis percentile')
    self.new_settable_attr('frequp', True, bool,
                           'Whether to sort weights by frequency')
    self.new_settable_attr('filter', True, bool,
                           'Whether to apply percentile filter')

  # region: Properties

  @property
  def selected_w(self):
    layer_key = self.pictor.Keys.PLOTTERS
    epoch_index = self.pictor.cursors[layer_key]
    return self.pictor.get_element(self.pictor.Keys.OBJECTS)[epoch_index]

  @property
  def frequency(self):
    # deprecated
    w = self.selected_w
    der1 = np.sign(w[:, :-1] - w[:, 1:])
    der2 = np.sign(der1[:, :-1] - der1[:, 1:])
    return np.sum(np.abs(der2), axis=-1)

  @property
  def sign_frequency(self):
    return self._calc_freq(self.selected_w)

  @property
  def amplitude(self):
    return self._calc_amp(self.selected_w)

  # endregion: Properties

  # region: Private Methods

  def _calc_freq(self, w):
    w = np.sign(w)
    der1 = np.sign(w[:, :-1] - w[:, 1:])
    return np.sum(np.abs(der1), axis=-1).astype(int)

  def _calc_amp(self, w):
    return np.max(w, axis=-1) - np.min(w, axis=-1)

  # endregion: Private Methods

  # region: Plotting Method

  def sort_rows(self, w: np.ndarray):
    result = []
    freqs = self._calc_freq(w)

    freq_list = (list(range(min(freqs), max(freqs) + 1))
                 if self.get('frequp') else [0])
    freq_list = sorted(freq_list, reverse=False)

    for freq in freq_list:
      if freq > 0:
        indices = np.argwhere(freq == freqs).ravel()
        w_f: np.ndarray = w[indices]
      else:
        w_f = w.copy()

      # Sort w_f by amplitude
      if len(result) == 0:
        i = np.argmax(self._calc_amp(w_f))
        result.append(w_f[i])
        w_f = np.delete(w_f, i, axis=0)

      while w_f.size > 0:
        corr = [np.correlate(seq, result[-1]) for seq in w_f]
        i = np.argmax(corr)
        result.append(w_f[i])
        w_f = np.delete(w_f, i, axis=0)

    return np.array(result)

  def scatter(self, ax: plt.Axes, xs, ys, xlabel='x-label', ylabel='y-label',
              title=None,):

    v = np.percentile(xs, self.get('xper'))
    displot_y = ys[np.argwhere(xs > v)]
    bins = int(max(displot_y)-min(displot_y))
    sns.distplot(displot_y, bins=bins, vertical=True, color='#2E75B6', ax=ax)
    ax.scatter(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)

    # Set [xy]lim if required
    xmax = self.get('xmax')
    if xmax: ax.set_xlim(0, xmax)
    ymax = self.get('ymax')
    if ymax: ax.set_ylim(-0.5, ymax)

    # Force integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # TODO
    p = self.get('xper')
    if p:
      v = np.percentile(xs, p)
      max_y = max(ys) if not ymax else ymax
      ax.plot([v, v], [0, max_y], 'r-', alpha=0.5)
      if not title: ax.set_title(f'Percentile: {p}%')

  def wavefrom_show(self, ax:plt.Axes, signal:np.ndarray, freq:np.ndarray):
    from scipy.interpolate import make_interp_spline

    freq_list = np.arange(min(freq), max(freq+1))
    freq_list = sorted(freq_list, reverse=True)
    wave = []
    freq_num = []
    for f in freq_list:
      indices = np.argwhere(freq == f).ravel()
      w_f: np.ndarray = signal[indices]
      if w_f.shape[0] == 0:
        continue
      else:
        i = np.argmax(self._calc_amp(w_f))
        v = w_f[i]
        v = v - min(v)
        v = v / max(v)
        wave.append(v-0.5)
        freq_num.append(f)
    ## Plot Wavefrom at different frequencies
    x = np.arange(wave[0].shape[0])
    for i, w in enumerate(wave):
      model = make_interp_spline(x, w)
      xs = np.linspace(0, np.max(x), 500)
      ys = model(xs)
      ax.plot(xs * 22 *2/(512), ys+ freq_num[i], color = '#2E75B6')
    ymax = self.get('ymax')
    if ymax: ax.set_ylim(-0.5, ymax)
    ax.set_xlabel('(a) Time(ms)')
    ax.set_yticks([])

  def imshow_w(self, ax: plt.Axes, w, fig):
    wmax = self.get('wmax')
    if wmax is None: wmax = max(abs(np.min(w)), np.max(w))
    vmin, vmax = (-wmax, wmax)

    interp = 'bilinear'
    im = ax.imshow(
      w, cmap=self.get('cmap'), vmin=vmin, vmax=vmax,
      aspect='auto', interpolation=interp)
    # new_ticks = np.around(np.arange(4)* 5*22 *2/(512), 1)
    new_ticks = 0.5 * np.arange(4)
    ax.set_xticks(np.arange(4)*256/44, new_ticks)
    ax.set_xlabel('(b) Time(ms)')
    # im = ax.imshow(
    #   w, cmap=self.get('cmap'), aspect='auto', interpolation=interp)

    # Add color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

  def imshow_spectrum(self, ax: plt.Axes, w, fig):
    A = np.abs(np.fft.fft(w))
    A = A[:, :A.shape[1] // 2]

    interp = 'bilinear'
    cmap = 'YlGn'
    # im = ax.imshow(A, cmap=cmap, aspect='auto', interpolation=interp, vmax=0.3)
    im = ax.imshow(A, cmap=cmap, aspect='auto', interpolation=interp)
    new_xticks = 64* np.arange(4)
    ax.set_xticks(64*np.arange(4)*11/256, new_xticks)
    ax.set_xlabel('(c) Frequency(Hz)')
    # Add color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

  def sort_and_plot(self,fig: plt.Figure):
    x = self.selected_w

    # Clear figure, create subplots
    fig.clear()
    axes = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})
    # fig.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

    # (1) Create a scatter plot
    xs, ys = self.amplitude, self.sign_frequency
    # self.scatter(axes[1], xs, ys, 'Amplitude', 'Turning Point Number')

    # (1) Show the waveform of different frequency
    freq = self.sign_frequency
    self.wavefrom_show(axes[0], x, freq)

    # (2) Show weight as image
    if self.get('filter'):
      v = np.percentile(xs, self.get('xper'))
      x = x[np.argwhere(self.amplitude > v).ravel()]
    x = self.sort_rows(x)[::-1]
    self.imshow_w(axes[1], x, fig)

    # (3) Show spectrum
    self.imshow_spectrum(axes[2], x, fig)

  # endregion: Plotting Method



if __name__ == '__main__':
  # (1) Load note and get weight matrices
  sum_paths = [
    '../01-EEGDenoiseNet_Dataset/checkpoints/0404_mod_tasnetEMG_GAN.sum',
  ]
  notes = [Note.load(p)[0] for p in sum_paths]

  epochs = 31

  weight_lists = [note.tensor_dict['Exemplar 0']['Encoder Weights']
                  for note in notes]
  weight_lists = [[w.T for w in wl][-epochs:] for wl in weight_lists]

  # metric_key = 'val RRMSE_temporal'

  # (2) Plot weights
  from pictor import Pictor
  p = Pictor(title=EncoderAnalyzer.__class__.__name__, figure_size=(10, 10))
  p.objects = weight_lists

  for _ in range(epochs): p.add_plotter(EncoderAnalyzer())

  p.show()

