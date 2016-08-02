import mne
import sys
import numpy as np
from nitime import TimeSeries
from nitime.analysis import CorrelationAnalyzer

from my_settings import *

bands = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2"]

times = np.arange(-2000, 2001, 1.95325)
times = times / 1000.

for subject in subjects:
    ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
                     subject)
    ht_invol = np.load(tf_folder + "%s_inv_HT-comp.npy" %
                       subject)

    results_vol = {}
    results_invol = {}

    for j, band in enumerate(bands):
        corr_vol = []
        corr_invol = []

        ht_vol_bs = mne.baseline.rescale(
            np.abs(ht_vol[:, :, :, j])**2,
            times,
            baseline=(-1.85, -1.5),
            mode="percent")

        ht_invol_bs = mne.baseline.rescale(
            np.abs(ht_invol[:, :, :, j])**2,
            times,
            baseline=(-1.85, -1.5),
            mode="percent")

        for ts in ht_vol_bs:
            nits = TimeSeries(ts[:, 768:1024],
                              sampling_rate=512)

            corr_vol += [CorrelationAnalyzer(nits)]

        for ts in ht_invol_bs:
            nits = TimeSeries(ts[:, 768:1024],
                              sampling_rate=512)

            corr_invol += [CorrelationAnalyzer(nits)]

        results_vol[band] = np.asarray([c.corrcoef for c in corr_vol])
        results_invol[band] = np.asarray([c.corrcoef for c in corr_invol])

    np.save(graph_data + "%s_vol_pow.npy" % subject,
            results_vol)
    np.save(graph_data + "%s_inv_pow.npy" % subject,
            results_invol)
