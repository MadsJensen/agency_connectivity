"""
Functions for TF analysis.

@author: mje
@email: mads [] cnru.dk
"""

import mne
from mne.time_frequency import (psd_multitaper, tfr_multitaper, tfr_morlet,
                                cwt_morlet)
from mne.viz import iter_topography
import matplotlib.pyplot as plt
import numpy as np


def calc_psd_epochs(epochs, plot=False):
    """Calculate PSD for epoch.

    Parameters
    ----------
    epochs : list of epochs
    plot : bool
        To show plot of the psds.
        It will be average for each condition that is shown.

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
        plt.legend()
        plt.gcf().suptitle('Power spectral densities')
        plt.show()

    return psds_vol, psds_inv, freqs


def multitaper_analysis(epochs):
    """

    Parameters
    ----------
    epochs : list of epochs

    Returns
    -------
    result : numpy array
        The result of the multitaper analysis.

    """
    frequencies = np.arange(6., 90., 2.)
    n_cycles = frequencies / 2.
    time_bandwidth = 4  # Same time-smoothing as (1), 7 tapers.
    power, plv = tfr_multitaper(epochs, freqs=frequencies, n_cycles=n_cycles,
                                time_bandwidth=time_bandwidth, return_itc=True)

    return power, plv


def morlet_analysis(epochs, n_cycles=4):
    """

    Parameters
    ----------
    epochs : list of epochs

    Returns
    -------
    result : numpy array
        The result of the multitaper analysis.

    """
    frequencies = np.arange(6., 30., 2.)
    # n_cycles = frequencies / 2.

    power, plv = tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles,
                            return_itc=True,
                            verbose=True)

    return power, plv


def single_trial_tf(epochs, frequencies, n_cycles=4.):
    """


    Parameters
    ----------
    epochs : Epochs object
        The epochs to calculate TF analysis on.
    frequencies : numpy array
    n_cycles : int
        The number of cycles for the Morlet wavelets.

    Returns
    -------
    results : numpy array
    """
    results = []

    for j in range(len(epochs)):
        tfr = cwt_morlet(epochs.get_data()[j],
                         sfreq=epochs.info["sfreq"],
                         freqs=frequencies,
                         use_fft=True,
                         n_cycles=n_cycles,
                         # decim=2,
                         zero_mean=False)
        results.append(tfr)
    return results


def calc_spatial_resolution(freqs, n_cycles):
    """Calculate the spatial resolution for a Morlet wavelet.

    The formula is: (freqs * cycles)*2.

    Parameters
    ----------
    freqs : numpy array
        The frequencies to be calculated.
    n_cycles : int or numpy array
        The number of cycles used. Can be integer for the same cycle for all
        frequencies, or a numpy array for individual cycles per frequency.

    Returns
    -------
    result : numpy array
        The results
    """
    return (freqs / float(n_cycles)) * 2


def calc_wavelet_duration(freqs, n_cycles):
    """Calculate the wavelet duration for a Morlet wavelet in ms.

    The formula is: (cycle / frequencies / pi)*1000

    Parameters
    ----------
    freqs : numpy array
        The frequencies to be calculated.
    n_cycles : int or numpy array
        The number of cycles used. Can be integer for the same cycle for all
        frequencies, or a numpy array for individual cycles per frequency.

    Returns
    -------
    result : numpy array
        The results
    """
    return (float(n_cycles) / freqs / np.pi) * 1000

