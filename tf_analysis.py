"""
Functions for TF analysis.

@author: mje
@email: mads [] cnru.dk
"""

# import mne
from mne.time_frequency import psd_multitaper
from mne.viz import iter_topography
import matplotlib.pyplot as plt
import numpy as np


def calc_psd_epochs(epochs, plot=False):
    """Calculate PSD for epoch.

    Parameters
    ----------
    epochs : list of epochs
    plot : bool
        To show plot of the psds. It will be average for each condition that is shown.

    Returns
    -------
    psds_vol : numpy array
        The psds for the voluntary condition.
    psds_invol : numpy array
        The psds for the involuntary condition.
    """
    tmin, tmax = -0.5, 0.5
    fmin, fmax = 2, 90
    # n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
    psds_vol, freqs = psd_multitaper(epochs["voluntary"],
                                     tmin=tmin, tmax=tmax,
                                     fmin=fmin, fmax=fmax)
    psds_inv, freqs = psd_multitaper(epochs["involuntary"],
                                     tmin=tmin, tmax=tmax,
                                     fmin=fmin, fmax=fmax)

    psds_vol = 20 * np.log10(psds_vol)  # scale to dB
    psds_inv = 20 * np.log10(psds_inv)  # scale to dB

    if plot:
        def my_callback(ax, ch_idx):
            """Executed once you click on one of the channels in the plot."""
            ax.plot(freqs, psds_vol_plot[ch_idx], color='red',
                    label="voluntary")
            ax.plot(freqs, psds_inv_plot[ch_idx], color='blue',
                    label="involuntary")
            ax.set_xlabel = 'Frequency (Hz)'
            ax.set_ylabel = 'Power (dB)'
            ax.legend()

        psds_vol_plot = psds_vol.copy().mean(axis=0)
        psds_inv_plot = psds_inv.copy().mean(axis=0)

        for ax, idx in iter_topography(epochs.info,
                                       fig_facecolor='k',
                                       axis_facecolor='k',
                                       axis_spinecolor='k',
                                       on_pick=my_callback):
            ax.plot(psds_vol_plot[idx], color='red', label="voluntary")
            ax.plot(psds_inv_plot[idx], color='blue', label="involuntary")
        plt.legend
        plt.gcf().suptitle('Power spectral densities')
        plt.show()

    return psds_vol, psds_inv, freqs
