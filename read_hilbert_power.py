# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:50:50 2016

@author: mje
"""
import numpy as np
import scipy.io as sio
from glob import glob

from my_settings import *

# subjects = ["p17"]

for subject in subjects:
    print("working on: %s" % subject)
    inv_pow_files = glob(data_path + "/data/%s/*nvoluntary/timefreq*zscore.mat"
                         % subject)

    inv_pow_files.sort()

    inv_ts = np.empty([len(inv_pow_files), 79, 2049, 6])
    for j, t in enumerate(inv_pow_files):
        inv_ts[j] = sio.loadmat(t)["TF"]

    vol_pow_files = glob(data_path + "/data/%s/voluntary/timefreq*zscore.mat" %
                         subject)

    vol_pow_files.sort()

    vol_ts = np.empty([len(vol_pow_files), 79, 2049, 6])
    for j, t in enumerate(vol_pow_files):
        vol_ts[j] = sio.loadmat(t)["TF"]

    np.save(tf_folder + "%s_vol_HT-pow_zscore.npy" % subject, vol_ts)
    np.save(tf_folder + "%s_inv_HT-pow_zscore.npy" % subject, inv_ts)
