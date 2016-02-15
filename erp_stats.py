"""
Script to perform statistical analysis of ERP data.

@author: mje
@email: mads [] cnru.dk
"""

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.stats import (spatio_temporal_cluster_test,
                       permutation_cluster_test)

from erp_stats import hilbert_process

data_folder = "/home/mje/Projects/agency_connectivity/data/"

epochs = mne.read_epochs(data_folder + "P2_ds_bp_ica-epo.fif")

data_vol = epochs["voluntary"].get_data()
data_invol = epochs["involuntary"].get_data()

times = epochs.times
temporal_mask = np.logical_and(-0.5 <= times, times <= 0)

channel_index = 37

data_vol = data_vol[:, channel_index, temporal_mask]
data_invol = data_invol[:, channel_index, temporal_mask]

times = 1e3 * epochs.times
times = times[temporal_mask]

###############################################################################
# Compute statistic
threshold = None
n_permutations = 5000
tail = 0
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([data_vol, data_invol],
                             n_permutations=n_permutations,
                             threshold=threshold, tail=tail, n_jobs=2)

###############################################################################
# Plot
plt.close('all')
plt.subplot(211)
plt.title('Channel : ' + epochs.ch_names[channel_index])
plt.plot(times, data_vol.mean(axis=0)*1e6 - data_invol.mean(axis=0)*1e6,
         label="ERP Contrast (Voluntary - Involuntary)")
plt.ylabel("EEG (uV)")
plt.legend()
plt.subplot(212)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = plt.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)
hf = plt.plot(times, T_obs, 'g')
plt.legend((h,), ('cluster p-value < 0.05',))
plt.xlabel("time (ms)")
plt.ylabel("f-values")
plt.show()

# PERMUTATION CLUSTER TEST ON SONSOR DATA ####

data_vol = epochs["voluntary"].get_data()
data_invol = epochs["involuntary"].get_data()
times = epochs.times
temporal_mask = np.logical_and(-0.5 <= times, times <= 0.5)

picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                       stim=False, exclude='bads')
test_times = times[temporal_mask]

data_vol = data_vol[:, :, temporal_mask]
data_invol = data_invol[:, :, temporal_mask]

data_vol = np.rollaxis(data_vol, 2, 1)
data_invol = np.rollaxis(data_invol, 2, 1)

neighbor_file = '/home/mje/Toolboxes/fieldtrip/template/neighbours' \
                 '/biosemi64_neighb'
connectivity, ft_ch_names = mne.channels.read_ch_connectivity(neighbor_file,
                                                              picks=picks)

threshold = None
n_permutations = 2000
tail = 0

T_obs, clusters, cluster_p_values, H0 =\
    spatio_temporal_cluster_test([data_vol, data_invol],
                                 n_permutations=n_permutations,
                                 threshold=threshold,
                                 tail=tail,
                                 out_type="mask",
                                 connectivity=connectivity,
                                 n_jobs=2)

# PLOT
# Make evoked difference between the two conditions.
diff_wave = epochs["voluntary"].average() -\
                  epochs["involuntary"].average()
diff_wave.crop(test_times[0], test_times[-1])

min_cluster_index = cluster_p_values.argmin()
mask = np.squeeze(clusters[min_cluster_index][:, np.newaxis].T)
plot_times = np.arange(-0.5, 0.5, 0.1)

diff_wave.plot_topomap(times=plot_times, ch_type='eeg',
                       cmap='viridis',
                       unit='uV',
                       mask=mask, size=3,
                       show_names=False)

# Topoplot all conditions
vol_ave = epochs["voluntary"].average()
invol_ave = epochs["involuntary"].average()

vol_value = np.max(np.abs([vol_ave.data.min(), vol_ave.data.max()]))
invol_value = np.max(np.abs([invol_ave.data.min(), invol_ave.data.max()]))


colors = "blue", "green"
mne.viz.plot_evoked_topo([epochs["voluntary"].average(),
                          epochs["involuntary"].average()],
                         color=colors)

conditions = [e.comment for e in [epochs["voluntary"].average(),
                                  epochs["involuntary"].average()]]
for cond, col, pos in zip(conditions, colors, (0.02, 0.07, 0.12, 0.17)):
    plt.figtext(0.97, pos, cond, color=col, fontsize=20,
                horizontalalignment='right')


invol_ave.plot_topomap(times=plot_times, ch_type='eeg',
                       cmap='viridis',
                       show_names=False,
                       title="Involuntary")

vol_ave.plot_topomap(times=plot_times, ch_type='eeg',
                     cmap='viridis',
                     show_names=False,
                     title="Voluntary")

raw = mne.io.Raw( data_folder + "P2_ds_bp_ica-raw.fif", preload=True)
bands = {"alpha": [8, 12],
         "beta": [13, 25],
         "gamma_low": [30, 48],
         "gamma_high": [52, 90],
         "theta": [4, 8]}

res = hilbert_process(raw, bands, return_evoked=True)

