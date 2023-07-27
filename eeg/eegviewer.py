from pictor import DaVinci

import matplotlib.pyplot as plt
import numpy as np




class EEGViewer(DaVinci):
    """
    This class is used to visualize EEG input signal, output signal and
    reference signal
    """
    def __init__(self, fs=None, data_in=None, plt_num=3):

        super(EEGViewer, self).__init__(
            'Radio-Frequency-Signal Viewer', height=5, width=5)

        ## sanity check

        self.objects = None
        self.fs = fs
        ## the shape of data is [3, n, l], 3 is reference, noise, calibrated;
        ##  n is the sample number; l is length of a sample
        self.data = data_in
        self.plt_num = plt_num


    def signal_view(self):
        self.objects = np.arange(self.data.shape[1])
        self.add_plotter(lambda x, ax: self.wave_plotter(x, ax, tag='time'))
        self.add_plotter(lambda x, ax: self.wave_plotter(x, ax, tag='fft'))
        self.add_plotter(lambda x, ax: self.wave_plotter(x, ax, tag='psd'))
        self.show()


    def wave_plotter(self, x, ax, tag=None):
        if tag == 'time':
            length = self.data.shape[2]
            dt = 1/self.fs
            x_t = dt * np.arange(length)
            for i in np.arange(self.data.shape[0]):
                ax.plot(x_t, self.data[i][x], alpha=0.5)
            if self.plt_num == 3:
                ax.legend(labels=['Ground-Truth EEG', 'Contaminated EEG',
                                      'Processed EEG'], loc='lower left')
            elif self.plt_num == 2:
                ax.legend(labels=['Contaminated EEG', 'Ground-Truth EEG'],
                          loc='lower left')
            else:
                pass
            ax.set_xlabel('Time(s)')
            ax.set_ylabel('Amplitude(uV)')

        elif tag == 'fft':
            for i in np.arange(self.data.shape[0]):
                fft_x, fft_y = self._signal_fft(self.data[i][x])
                ax.plot(fft_x, fft_y, alpha=0.5)
            if self.plt_num == 3:
                ax.legend(labels=['Ground-Truth EEG', 'Contaminated EEG',
                                  'Processed EEG'], loc='lower left')
            elif self.plt_num == 2:
                ax.legend(labels=['Contaminated EEG', 'Ground-Truth EEG'],
                          loc='lower left')
            else:
                pass
            ax.set_xlabel('Frequency(Hz)')
            ax.set_ylabel('Amplitude(dB)')

        elif tag == 'psd':
            for i in np.arange(self.data.shape[0]):
                psd_x, psd_y = self._signal_psd(self.data[i][x])
                ax.plot(psd_x, psd_y, alpha=0.5)
            if self.plt_num == 3:
                ax.legend(labels=['Ground-Truth EEG', 'Contaminated EEG',
                                  'Processed EEG'], loc='lower left')
            elif self.plt_num == 2:
                ax.legend(labels=['Contaminated EEG', 'Ground-Truth EEG'],
                          loc='lower left')
            else:
                pass
            ax.set_xlabel('Frequency(Hz)')
            ax.set_ylabel('PSD(dB)')


    def _signal_fft(self, data_in):
        samplenum = data_in.shape[0]
        fft_data= 20*np.real(np.log10(2*np.fft.fft(data_in)/samplenum))
        fft_x = [self.fs/samplenum*i for i in range(samplenum)]
        return fft_x, fft_data


    def _signal_psd(self, data_in):
        from scipy import signal as sig

        samplenum = data_in.shape[0]
        stop_index = int(samplenum/self.fs * 80)
        # psd = (np.abs(np.fft.fft(data_in))**2 / (self.fs / samplenum))[:stop_index]
        _, psd = sig.welch(data_in[:stop_index], fs=self.fs, nfft=stop_index,
                           nperseg=stop_index)
        ## Frequency span is 0-80hz
        psd_x = self.fs/samplenum*np.arange(stop_index)
        return  psd_x, 20*np.log10(psd)

