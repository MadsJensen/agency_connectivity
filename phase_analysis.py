# -*- coding: utf-8 -*-
"""
@author: mje
@emai: mads@cnru.dk
"""

import numpy as np
# import mne
import matplotlib.pyplot as plt
import pandas as pd

from my_settings import *

plt.style.use("ggplot")


def calc_ISPC_time_between(data):
    result = np.empty([data.shape[0]])

    for i in range(data.shape[0]):
        result[i] = np.abs(
            np.mean(
                np.exp(1j * (np.angle(data[i, 52, window_start:window_end]) -
                             np.angle(data[i, 1, window_start:window_end])))))
    return result


def calc_ISPC_time(data, labels_index=52):
    """label_index : int
           the index of the data to use"""

    result = np.empty([data.shape[0]])

    for i in range(data.shape[0]):
        result[i] = np.abs(
            np.mean(
                np.exp(1j * (np.angle(data[i, 52, window_start:window_end])))))
    return result

# subjects = ["p9"]
labels = list(np.load(data_path + "label_names.npy"))

times = np.arange(-2000, 2001, 1.95325)
times = times / 1000.

window_length = 153
step_length = 15

results_all = pd.DataFrame()
for subject in subjects:
    step = 1
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_invol = np.load(tf_folder + "%s_inv_HT-comp.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()

    # results_invol = {}
    ht_invol_band = ht_invol[:, :, :, 3]

    j = 768  # times index to start
    while times[window_length + j] < times[1040]:
        window_start = j
        window_end = j + window_length

        res = pd.DataFrame(
            calc_ISPC_time_between(ht_invol_band), columns=["ISPC"])
        res["step"] = step
        res["subject"] = subject
        res["label"] = "BA_4_1"
        res["binding"] = b_tmp.binding
        res["trial_status"] = b_tmp.trial_status
        res["condition"] = "invol"
        results_all = results_all.append(res)
        j += step_length
        step += 1

    step = 1
    j = 768  # times index to start
    labels_index = 52
    while times[window_length + j] < times[1040]:
        window_start = j
        window_end = j + window_length

        res = pd.DataFrame(
            calc_ISPC_time(
                ht_invol_band, labels_index=labels_index),
            columns=["ISPC"])
        res["step"] = step
        res["subject"] = subject
        res["label"] = labels[labels_index]
        res["binding"] = b_tmp.binding
        res["trial_status"] = b_tmp.trial_status
        res["condition"] = "invol"
        results_all = results_all.append(res)
        j += step_length
        step += 1

    step = 1
    j = 768  # times index to start
    labels_index = 1
    while times[window_length + j] < times[1040]:
        window_start = j
        window_end = j + window_length

        res = pd.DataFrame(
            calc_ISPC_time(
                ht_invol_band, labels_index=labels_index),
            columns=["ISPC"])
        res["step"] = step
        res["subject"] = subject
        res["label"] = labels[labels_index]
        res["binding"] = b_tmp.binding
        res["trial_status"] = b_tmp.trial_status
        res["condition"] = "invol"
        results_all = results_all.append(res)
        j += step_length
        step += 1
    step = 1
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_vol = np.load(tf_folder + "%s_vol_HT-comp.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()

    # Results_vol = {}
    ht_vol_band = ht_vol[:, :, :, 3]

    j = 768  # times index to start
    while times[window_length + j] < times[1040]:
        window_start = j
        window_end = j + window_length

        res = pd.DataFrame(
            calc_ISPC_time_between(ht_vol_band), columns=["ISPC"])
        res["step"] = step
        res["subject"] = subject
        res["label"] = "BA_4_1"
        res["binding"] = b_tmp.binding
        res["trial_status"] = b_tmp.trial_status
        res["condition"] = "vol"
        results_all = results_all.append(res)
        j += step_length
        step += 1

    step = 1
    j = 768  # times index to start
    labels_index = 52
    while times[window_length + j] < times[1040]:
        window_start = j
        window_end = j + window_length

        res = pd.DataFrame(
            calc_ISPC_time(
                ht_vol_band, labels_index=labels_index),
            columns=["ISPC"])
        res["step"] = step
        res["subject"] = subject
        res["label"] = labels[labels_index]
        res["binding"] = b_tmp.binding
        res["trial_status"] = b_tmp.trial_status
        res["condition"] = "vol"
        results_all = results_all.append(res)
        j += step_length
        step += 1

    step = 1
    j = 768  # times index to start
    labels_index = 1
    while times[window_length + j] < times[1040]:
        window_start = j
        window_end = j + window_length

        res = pd.DataFrame(
            calc_ISPC_time(
                ht_vol_band, labels_index=labels_index),
            columns=["ISPC"])
        res["step"] = step
        res["subject"] = subject
        res["label"] = labels[labels_index]
        res["binding"] = b_tmp.binding
        res["trial_status"] = b_tmp.trial_status
        res["condition"] = "vol"
        results_all = results_all.append(res)
        j += step_length
        step += 1
