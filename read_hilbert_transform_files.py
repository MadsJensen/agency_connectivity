# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:50:50 2016

@author: mje
"""
import numpy as np
import scipy.io as sio
from glob import glob

from my_settings import *

for subject in subjects:
    inv_hilbert_files = list(set(glob(
        data_path + "bst_dir/%s/Involuntary/*hilbert*" % subject)) - set(glob(
            data_path + "bst_dir/%s/Involuntary/*hilbert*meas*" % subject)))

    inv_hilbert_files.sort()
    inv_hilbert_files = inv_hilbert_files[:-1]

    inv_ts = np.empty([len(inv_hilbert_files), 68, 2049, 5],
                      dtype="complex128")
    for j, t in enumerate(inv_hilbert_files):
        inv_ts[j] = sio.loadmat(t)["TF"]

    vol_hilbert_files = list(set(glob(
        data_path + "bst_dir/%s/Voluntary/*hilbert*" % subject)) - set(glob(
            data_path + "bst_dir/%s/Voluntary/*hilbert*meas*" % subject)))

    vol_hilbert_files.sort()
    vol_hilbert_files = vol_hilbert_files[:-1]

    vol_ts = np.empty([len(vol_hilbert_files), 68, 2049, 5],
                      dtype="complex128")
    for j, t in enumerate(vol_hilbert_files):
        vol_ts[j] = sio.loadmat(t)["TF"]

    np.save(tf_folder + "%s_vol_HT-comp.npy" % subject, vol_ts)
    np.save(tf_folder + "%s_inv_HT-comp.npy" % subject, inv_ts)
