# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:50:50 2016

@author: mje
"""
import numpy as np
import scipy.io as sio
from glob import glob

from my_settings import *

subjects = ["p17"]

for subject in subjects:
    inv_hilbert_files = glob(data_path + "/data/%s/*nvoluntary/*hilbert*"
                             % subject)

    inv_hilbert_files.sort()

    inv_ts = np.empty([len(inv_hilbert_files), 79, 2049, 6],
                      dtype="complex128")
    for j, t in enumerate(inv_hilbert_files):
        inv_ts[j] = sio.loadmat(t)["TF"]

    vol_hilbert_files = glob(data_path + "/data/%s/voluntary/*hilbert*"
                             % subject)

    vol_hilbert_files.sort()

    vol_ts = np.empty([len(vol_hilbert_files), 79, 2049, 6],
                      dtype="complex128")
    for j, t in enumerate(vol_hilbert_files):
        vol_ts[j] = sio.loadmat(t)["TF"]

    np.save(tf_folder + "%s_vol_HT-comp.npy" % subject, vol_ts)
    np.save(tf_folder + "%s_inv_HT-comp.npy" % subject, inv_ts)
