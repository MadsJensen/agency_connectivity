import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from tf_analysis import single_trial_tf

plt.ion()

data_folder = "/home/mje/Projects/agency_connectivity/data/"

epochs = mne.read_epochs(data_folder + "P2_ds_bp_ica-epo.fif")

# single trial morlet tests
frequencies = np.arange(6., 30., 2.)
n_cycles = 5.
times = epochs.times

tfr_vol = single_trial_tf(epochs["voluntary"])
tfr_invol = single_trial_tf(epochs["involuntary"])

pow_vol_Cz = np.asarray([np.mean(np.abs(tfr[37, 4:-2, :])**2, axis=0)
                         for tfr in tfr_vol])
pow_invol_Cz = np.asarray([np.mean(np.abs(tfr[37, 4:-2, :])**2, axis=0)
                           for tfr in tfr_invol])

pow_invol_Cz_bs = np.asarray([(10*np.log10(trial / np.mean(trial[:103]))) for
                              trial in pow_invol_Cz])

pow_vol_Cz_bs = np.asarray([(10*np.log10(trial / np.mean(trial[:103]))) for
                            trial in pow_vol_Cz])

pow_invol_Cz_mean = pow_invol_Cz_bs[:, 921:1024].mean(axis=1)
pow_vol_Cz_mean = pow_vol_Cz_bs[:, 921:1024].mean(axis=1)

stats.ttest_ind(pow_vol_Cz_mean, pow_invol_Cz_mean)

corr, pval = stats.spearmanr(pow_vol_Cz_mean[-60:], pow_invol_Cz_mean)
print("correlation: %s, pval: %s" % (corr, pval))

sns.regplot(pow_vol_Cz_mean[-60:], pow_invol_Cz_mean)



from sklearn.cluster.spectral import spectral_embedding  # noqa
from sklearn.metrics.pairwise import rbf_kernel   # noqa


def order_func(times, data):
    this_data = data[:, (times < -0.5) & (times < 0)]
    this_data /= np.sqrt(np.sum(this_data ** 2, axis=1))[:, np.newaxis]
    return np.argsort(spectral_embedding(rbf_kernel(this_data, gamma=1.),
                      n_components=1, random_state=0).ravel())

good_pick = 37  # channel with a clear evoked response
bad_pick = 47  # channel with no evoked response

plt.close('all')
mne.viz.plot_epochs_image(epochs["involuntary"], [good_pick, bad_pick],
                          sigma=0.5, cmap="viridis",
                          colorbar=True, order=order_func, show=True)
