# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:53:50 2016

@author: au194693
"""

import numpy as np
import scipy.io as sio
import pandas as pd

from my_settings import *

data = sio.loadmat(data_path + "behavoural_results.mat")["data_all"]

column_keys = ["subject", "condition", "binding", "trial_status"]
b_df = pd.DataFrame()

for j in range(len(data)):
    baseline = data[j, 0].mean()
    invol_trials = data[j, 3].squeeze()

    if len(invol_trials) is 90:
        invol_trials = invol_trials[1:]

    error = np.std(data[j, 3]) * 2
    for i in range(len(invol_trials)):
        row = pd.DataFrame([{"subject": "p%s" % (j + 2),
                             "condition": "invol",
                             "binding": invol_trials[i] - baseline,
                             "trial_number": i + 1,
                             "trial_status":
                             np.abs(invol_trials[i] - baseline) < error,
                             "error": error}])
        b_df = b_df.append(row, ignore_index=True)

column_keys = ["subject", "condition", "binding", "trial_status"]
# b_df = pd.DataFrame()

for j in range(len(data)):
    baseline = data[j, 0].mean()
    vol_trials = data[j, 2].squeeze()

    if len(invol_trials) is 90:
        vol_trials = invol_trials[1:]

    error = np.std(data[j, 3]) * 2
    for i in range(len(invol_trials)):
        row = pd.DataFrame([{"subject": "p%s" % (j + 2),
                             "condition": "vol",
                             "binding": invol_trials[i] - baseline,
                             "trial_number": i + 1,
                             "trial_status":
                             np.abs(invol_trials[i] - baseline) < error,
                             "error": error}])
        b_df = b_df.append(row, ignore_index=True)

# Calculate mean correlation

b_df = pd.read_csv(results_folder + "/behavioural_results.csv")

for subject in subjects:
    eeg = np.load(tf_folder + "%s_inv_HT-comp.npy" % subject)
    eeg_data = eeg[:, 52, 768:1024, 4]

    if eeg_data.shape[0] is 90:
        eeg_data = eeg_data[1:, :]

    eeg_data = np.mean(np.abs(eeg_data)**2, axis=1)
    test_eeg = eeg_data[b_df[b_df.subject ==
                             subject].trial_status.values].mean()
    test_binding = b_df[b_df.subject == subject][b_df[b_df.subject ==
                                                      subject].trial_status ==
                                                 True].binding.values.mean()
