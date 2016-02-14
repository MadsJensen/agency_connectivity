"""
Preprocessing function for the bdf.

@author: mje
@email: mads [] cnru.dk
"""
import mne
from mne.preprocessing import ICA, create_eog_epochs
import matplotlib.pyplot as plt

# SETTINGS
from sqlalchemy.sql.elements import True_
from sympy.core.trace import Tr

n_jobs = 1
reject = dict(eeg=300e-6)  # uVolts (EEG)
l_freq, h_freq, n_freq = 1, 98, 50  # Frequency setting for high, low, Noise
decim = 1  # decim value
montage = mne.channels.read_montage("biosemi64")

data_folder = "/home/mje/Projects/agency_connectivity/data/"


# Functions
def convert_bdf2fif(subject):
    """Convert bdf data to fiff.

    Parameters
    ----------
    subject: string
        The subject to convert.

    Returns
    -------
    None, but save fiff file.
    """
    raw = mne.io.read_raw_edf(data_folder + "%s_ds.bdf" % subject,
                              montage=montage,
                              eog=["EXG3", "EXG4", "EXG5", "EXG6"],
                              misc=["EXG1", "EXG2", "EXG7", "EXG8"],
                              preload=True)
    raw.add_eeg_average_proj()
    raw.save(data_folder + "%s-raw.fif" % subject, overwrite=True)


def filter_raw(subject):
    """Filter raw fifs.

    Parameters
    ----------
    subject : string
        the subject id to be loaded
    """
    raw = mne.io.Raw(data_folder + "%s-raw.fif" % subject, preload=True)
    raw.set_montage = montage
    raw.apply_proj()
    raw.notch_filter(n_freq)
    raw.filter(l_freq, h_freq)
    raw.save(data_folder + "%s_ds_bp-raw.fif" % subject, overwrite=True)


def compute_ica(subject):
    """Function will compute ICA on raw and apply the ICA.

    Parameters
    ----------
    subject : string
        the subject id to be loaded
    """
    raw = mne.io.Raw(data_folder + "%s_ds_bp-raw.fif" % subject,
                     preload=True)
    raw.set_montage = montage
    raw.apply_proj()
    # raw.resample(512, n_jobs=2)

    # ICA Part
    ica = ICA(n_components=None, max_pca_components=40, method='fastica',
              max_iter=256)

    picks = mne.pick_types(raw.info, meg=False, eeg=True,
                           stim=False, exclude='bads')

    ica.fit(raw, picks=picks, decim=decim, reject=reject)

    # maximum number of components to reject
    n_max_ecg, n_max_eog = 3, 1

    ##########################################################################
    # 2) identify bad components by analyzing latent sources.
    title = 'Sources related to %s artifacts (red) for sub: %s'
    #
    # # generate ECG epochs use detection via phase statistics
    # ecg_epochs = create_ecg_epochs(raw, ch_name="Ext4",
    #                                tmin=-.5, tmax=.5, picks=picks)
    # n_ecg_epochs_found = len(ecg_epochs.events)
    # sel_ecg_epochs = np.arange(0, n_ecg_epochs_found, 10)
    # ecg_epochs = ecg_epochs[sel_ecg_epochs]
    #
    # ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
    # fig = ica.plot_scores(scores, exclude=ecg_inds,
    #                       title=title % ('ecg', subject))
    # fig.savefig(data_folder + "pics/%s_ecg_scores.png" % subject)
    #
    # if ecg_inds:
    #     show_picks = np.abs(scores).argsort()[::-1][:5]
    #
    #     fig = ica.plot_sources(raw, show_picks, exclude=ecg_inds,
    #                            title=title % ('ecg', subject), show=False)
    #     fig.savefig(data_folder + "pics/%s_ecg_sources.png" % subject)
    #     fig = ica.plot_components(ecg_inds, title=title % ('ecg', subject),
    #                               colorbar=True)
    #     fig.savefig(data_folder + "pics/%s_ecg_component.png" % subject)
    #
    #     ecg_inds = ecg_inds[:n_max_ecg]
    #     ica.exclude += ecg_inds
    #
    # # estimate average artifact
    # ecg_evoked = ecg_epochs.average()
    # del ecg_epochs
    #
    # # plot ECG sources + selection
    # fig = ica.plot_sources(ecg_evoked, exclude=ecg_inds)
    # fig.savefig(data_folder + "pics/%s_ecg_sources_ave.png" % subject)
    #
    # # plot ECG cleaning
    # ica.plot_overlay(ecg_evoked, exclude=ecg_inds)
    # fig.savefig(data_folder + "pics/%s_ecg_sources_clean_ave.png" % subject)

    # DETECT EOG BY CORRELATION
    # HORIZONTAL EOG
    eog_epochs = create_eog_epochs(raw, ch_name="EXG4")
    eog_indices, scores = ica.find_bads_eog(raw, ch_name="EXG4")
    fig = ica.plot_scores(scores, exclude=eog_indices,
                          title=title % ('eog', subject))
    fig.savefig(data_folder + "pics/%s_eog_scores.png" % subject)

    fig = ica.plot_components(eog_indices, title=title % ('eog', subject),
                              colorbar=True)
    fig.savefig(data_folder + "pics/%s_eog_component.png" % subject)

    eog_indices = eog_indices[:n_max_eog]
    ica.exclude += eog_indices

    del eog_epochs

    ##########################################################################
    # Apply the solution to Raw, Epochs or Evoked like this:
    raw_ica = ica.apply(raw, copy=False)
    ica.save(data_folder + "%s-ica.fif" % subject)  # save ICA componenets
    # Save raw with ICA removed
    raw_ica.save(data_folder + "%s_ds_bp_ica-raw.fif" % subject,
                 overwrite=True)
    plt.close("all")


def epoch_data(subject, save=True):
    """Epoch a raw data set.

    Parameters
    ----------
    subject : str
        The subject to be epoched.
    save : bool
        Whether to save the epochs or not.

    Returns
    -------
    epochs
    """
    # SETTINGS
    tmin, tmax = -2, 2
    event_id = {'voluntary': 243,
                'involuntary': 219}

    raw = mne.io.Raw(data_folder + "%s_ds_bp_ica-raw.fif" % subject)
    events = mne.find_events(raw)

    #   Setup for reading the raw data
    picks = mne.pick_types(raw.info, meg=False, eeg=True,
                           stim=False, exclude='bads')
    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, -1.8), reject=reject,
                        preload=True)

    if save:
        epochs.save(data_folder + "%s_ds_bp_ica-epo.fif" % subject)

    return epochs


def hilbert_process(raw, l_freq, h_freq):

    tmin, tmax = -2, 2
    event_id = {'voluntary': 243,
                'involuntary': 219}
    picks = mne.pick_types(raw.info, meg=False, eeg=True,
                           stim=False, exclude='bads')
    raw_tmp = raw.copy()
    raw_tmp.filter(l_freq, h_freq)
    raw_tmp.apply_hilbert(picks=picks, envelope=True)

    events = mne.find_events(raw)
    # Read epochs
    epochs = mne.Epochs(raw_tmp, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, -1.8), reject=reject,
                        preload=True)
    return epochs
