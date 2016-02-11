"""
Script to perform statistical analysis of ERP data.

@author: mje
@email: mads [] cnru.dk
"""

import mne
import numpy as np
from mne.stats import fdr_correction
from scipy import stats
import matplotlib.pyplot as plt

data_folder = "/home/mje/Projects/agency_connectivity/data/"

epochs = mne.read_epochs(data_folder + "P2_ds_bp_ica-epo.fif")

data_vol, data_invol = epochs["voluntary"].get_data(),  epochs["involuntary"].get_data()

times = epochs.times
temporal_mask = np.logical_and(-0.5 <= times, times <= 0)

channel_index = 37

data_vol = data_vol[:, channel_index, temporal_mask]
data_invol = data_invol[:, channel_index, temporal_mask]

T, pval = stats.ttest_ind(data_vol, data_invol)
alpha = 0.05

reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
threshold_fdr = np.min(np.abs(T)[reject_fdr])

times = 1e3 * epochs.times
times = times[temporal_mask]

plt.close('all')
plt.plot(times, T, 'k', label='T-stat')
xmin, xmax = plt.xlim()
plt.hlines(threshold_fdr, xmin, xmax, linestyle='--', colors='b',
           label='p=0.05 (FDR)', linewidth=2)
plt.hlines(-threshold_fdr, xmin, xmax, linestyle='--', colors='b',
           linewidth=2)
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("T-stat")
plt.title("T-test: voluntary v involuntary for channel: %s" % epochs.ch_names[channel_index])
plt.show()

plt.figure()
plt.plot(times, data_vol.mean(axis=0), 'b', label="Voluntary")
plt.plot(times, data_invol.mean(axis=0), 'r', label="Involuntary")
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("uV")
plt.title("Mean EPR for channel: %s" % epochs.ch_names[channel_index])
plt.show()


