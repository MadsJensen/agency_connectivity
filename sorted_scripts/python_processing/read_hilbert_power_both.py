# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:50:50 2016

@author: mje
"""
import numpy as np
import scipy.io as sio
from glob import glob

from my_settings import (data_path, tf_folder, subjects_ctl, subjects_test)

subjects = subjects_test + subjects_ctl

for subject in subjects:
    print("working on: %s" % subject)
    inv_pow_files = glob(data_path + "/data/%s/involuntary/timefreq*zscore.mat"
                         % subject)

    inv_pow_files.sort()

    inv_ts = np.empty([len(inv_pow_files), 8, 1537, 3])
    for j, t in enumerate(inv_pow_files):
        inv_ts[j] = sio.loadmat(t)["TF"][:, :, :]

    np.save(tf_folder + "%s_inv_HT-pow_zscore.npy" % subject, inv_ts)
