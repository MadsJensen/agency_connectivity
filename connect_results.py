import os
import numpy as np
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
from mne.stats import fdr_correction

import nitime
from nitime.timeseries import TimeSeries

conn_results = "/media/mje/My_Book/Data/agency_connectivity/bst_connect_results"

os.chdir(conn_results)

vol_data = sio.loadmat("p2_vol_post_plv.mat")["vol_results"]
invol_data = sio.loadmat("p2_invol_post_plv.mat")["inv_results"]

res_inx = np.tril_indices(68, k=-1)

vol_test = []

for j in range(len(vol_data)):
    tmp = vol_data[j, :, :, 0, 3]
    vol_test.append(tmp[res_inx])

invol_test = []

for j in range(len(invol_data)):
    tmp = invol_data[j, :, :, 0, 3]
    invol_test.append(tmp[res_inx])

vol_test = np.asarray(vol_test)
invol_test = np.asarray(invol_test)

t_stat, pval = stats.ttest_ind(vol_test, invol_test, axis=0)

rejected, pval_fdr = fdr_correction(pval)

foo = np.zeros([68, 68])
foo[res_inx] = pval_fdr

## Extract labels
labels = [lbl[0][0].split()[0] + "_" + lbl[0][0].split()[1]
          for lbl in ff["RowNames"]]
