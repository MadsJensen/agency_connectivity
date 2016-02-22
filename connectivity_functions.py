"""Functions for connectivity analysis.

@author: mje
@email: mads [] cnru.dk
"""
import numpy as np


# ISPC over time test
def ISPC_over_time(data, frequencies, channels, faverage=True):
    """Calculate the ISPC over time.

    Parameters
    ----------
    data : numpy array
        The data to be used.
        It should be trials x channels x frequencies x times.
    frequencies : list
        A list with the frequencies to be calculates.
    channels : list
        List containing two channels.
    faverage : bool
        If true the average is returned, If false each frequency is returned.

    Returns
    -------
    result : numpy array
        The result is a numpy array with the length of the length of the
        epochs.

    """
    result = np.empty([len(frequencies), data.shape[0]])
    chan_A, chan_B = channels[0], channels[1]
    for ii in range(len(frequencies)):
        for i in range(len(data)):
            result[ii, i] = np.abs(np.mean(np.exp(
                1j*(np.angle(data[i, chan_A, ii, :]) -
                    np.angle(data[i, chan_B, ii, :])))))

    if faverage:
        result = result.mean(axis=0).squeeze()

    return result


# ISPC over trails
def ISPC_over_trials(data, freqs, channels, faverage=True):
    """Calculate the ISPC over time.

    Parameters
    ----------
    data : numpy array
        The data to be used.
        It should be trials x channels x frequencies x times.
    freqs : int
        A list with the frequencies to be calculates.
    channels : list
        List containing two channels.
    faverage : bool
        If true the average is returned, If false each frequency is returned.

    Returns
    -------
    result : numpy array
        The result is a numpy array with the lengt equal to the number of
        trials.

    """
    result = np.empty([len(freqs), data.shape[-1]])
    chan_A, chan_B = channels[0], channels[1]
    for ii in range(len(freqs)):
        for i in range(result.shape[-1]):
            result[ii, i] = np.abs(np.mean(np.exp(
                1j*(np.angle(data[:, chan_A, freqs[ii], i]) -
                    np.angle(data[:, chan_B, freqs[ii], i])))))

    if faverage:
        result = result.mean(axis=0).squeeze()

    return result
