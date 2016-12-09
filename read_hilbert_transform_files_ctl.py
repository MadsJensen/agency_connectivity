# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:50:50 2016

@author: mje
"""
import numpy as np
import scipy.io as sio
from glob import glob

from my_settings import (data_path, tf_folder)

subjects = ["p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29",
           "p30", "p31", "p32", "p33", "p34", "p35", "p36", "p37", "p38"]

for subject in subjects:
    test_hilbert_files = glob(data_path + "data/%s/test/*hilbert*" % subject)

    test_hilbert_files.sort()
    test_hilbert_files = [
        file for file in test_hilbert_files if not file.endswith("meas.mat")
    ]

    test_hilbert_files = [
        file for file in test_hilbert_files if not file.endswith("zscore.mat")
    ]
    
    test_ts = np.empty(
        [len(test_hilbert_files), 79, 2049, 6], dtype="complex128")
    for j, t in enumerate(test_hilbert_files):
        test_ts[j] = sio.loadmat(t)["TF"]

    np.save(tf_folder + "%s_test_HT-comp.npy" % subject, test_ts)
