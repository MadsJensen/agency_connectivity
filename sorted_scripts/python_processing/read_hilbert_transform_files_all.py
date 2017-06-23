# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:50:50 2016

@author: mje
"""
import numpy as np
import scipy.io as sio
from glob import glob
from tqdm import tqdm

from my_settings import (data_path, tf_folder)

data_folder = "/Volumes/My_Passport/agency_connectivity/agency_connect_2_raw_data/"

# subjects = ["P2", "P3", "P4", "P5", "P6", "P7", "P8",
#           "P10", "P11", "P12", "P13","P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
#             "P10", "P11", "P12","P13", 
#  "P14",
# "P15", "P16", "P17", "P18", 
subjects = [
    "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13",
    "P14", "P15", "P16", "P17", "P18", "P19", "P21", "P22", "P23", "P24",
    "P25", "P26", "P27", "P28", "P29", "P30", "P31", "P32", "P33", "P34",
    "P35", "P36", "P37", "P38"
]

for subject in tqdm(subjects):
    inv_hilbert_files = glob(data_path + "/data/%s/*nvoluntary/*hilbert*" %
                             subject)

    inv_hilbert_files_meas = glob(
        data_path + "/data/%s/*nvoluntary/*hilbert*meas*" % subject)
    inv_hilbert_files.sort()
    inv_hilbert_files_meas.sort()

    # remove files that are not complex numbers
    inv_files = set(inv_hilbert_files) - set(inv_hilbert_files_meas)

    inv_ts = np.empty([len(inv_files), 8, 1537, 3], dtype="complex128")
    for j, t in enumerate(inv_files):
        inv_ts[j] = sio.loadmat(t)["TF"]

    np.save(tf_folder + "%s_inv_HT-comp.npy" % subject, inv_ts)