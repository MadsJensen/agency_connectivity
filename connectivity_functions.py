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
        It should be trials x channels x times x frequencies.
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
                1j*(np.angle(data[i, chan_A, :, ii]) -
                    np.angle(data[i, chan_B, :, ii])))))

    if faverage:
        result = result.mean(axis=0).squeeze()

    return result


def ISPC_over_time_HB(data, channels):
    """Calculate the ISPC over time.

    Parameters
    ----------
    data : numpy array
        The data to be used.
        It should be trials x channels x times x frequencies.
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
    result = np.empty([data.shape[0]])
    chan_A, chan_B = channels[0], channels[1]
    for i in range(len(data)):
        result[i] = np.abs(np.mean(np.exp(
            1j*(np.angle(data[i, chan_A, :]) -
                np.angle(data[i, chan_B, :])))))

    return result


def ISPC_over_trials_hb(data, channels, faverage=True):
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
    result = np.empty([len(freqs), data.shape[2]])
    chan_A, chan_B = channels[0], channels[1]
    for i in range(result.shape[2]):
        result[ii, i] = np.abs(np.mean(np.exp(
            1j*(np.angle(data[:, chan_A, i, 4]) -
                np.angle(data[:, chan_B, i, 4])))))

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
    result = np.empty([len(freqs), data.shape[2]])
    chan_A, chan_B = channels[0], channels[1]
    for ii in range(len(freqs)):
        for i in range(result.shape[2]]):
            result[ii, i] = np.abs(np.mean(np.exp(
                1j*(np.angle(data[:, chan_A, freqs[ii], i]) -
                    np.angle(data[:, chan_B, freqs[ii], i])))))

    if faverage:
        result = result.mean(axis=0).squeeze()

    return result


def ITC_over_trials(data, freqs, channels, faverage=True):
    """Calculate the ITC over time.

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
        The result is a numpy array with the length equal to the number of
        time.
    """
    result = np.empty([len(freqs), data.shape[-1]])
    chan_A, chan_B = channels[0], channels[1]
    for ii in range(len(freqs)):
        for i in range(result.shape[-1]):
            result[ii, i] = np.abs(np.mean(np.exp(
                1j*(np.angle(data[:, chan_A, freqs[ii], i])))))

    if faverage:
        result = result.mean(axis=0).squeeze()

    return result


def ITC_over_time(data, faverage=True):
    """Calculate the ITC over time.

    Parameters
    ----------
    data : numpy array
        It should be trials x channels x frequencies x times.
    faverage : bool
        If true the average is returned, If false each frequency is returned.

    Returns
    -------
    result : numpy array
        The result is a numpy array with the length equal to the number of
        trials.
    """
    result = np.empty([data.shape[1], data.shape[0]])

    for freq in range(result.shape[0]):
        for i in range(result.shape[1]):
            result[freq, i] =\
                np.abs(np.mean(np.exp(1j * (np.angle(data[i, freq, :])))))

    if faverage:
        result = result.mean(axis=0).squeeze()

    return result
