# -*- coding: utf-8 -*-
"""
@author: mje
@emai: mads@cnru.dk
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from my_settings import (data_path, tf_folder, subjects_test, subjects_ctl,
                         subjects_dir)

plt.style.use("ggplot")

b_df = pd.read_csv("/Volumes/My_Passport/agency_connectivity/results/" +
                   "behavioural_results.csv")


def calc_ISPC_time_between(data, chan_1=52, chan_2=1):
    result = np.empty([data.shape[0]])

    for i in range(data.shape[0]):
        result[i] = np.abs(
            np.mean(
                np.exp(1j * (np.angle(data[i, chan_1, window_start:window_end])
                             - np.angle(data[i, chan_2, window_start:
                                             window_end])))))
    return result


# load labels
labels = mne.read_labels_from_annot(
    "fs_p2", parc='PALS_B12_Brodmann', regexp="Bro", subjects_dir=subjects_dir)

# make combinations of label indices
combinations = []
label_index = [0, 1, 52, 53, 54, 55, 68, 69]
for L in range(0, len(label_index) + 1):
    for subset in itertools.combinations(label_index, L):
        if len(subset) == 2:
            combinations.append(subset)

# make dict with names and indices
label_dict = {}
for comb in combinations:
    fname = "ba_" + labels[comb[0]].name.split(".")[1] + "_" + labels[comb[
        1]].name.split(".")[1]
    print(fname)
    label_dict.update({fname: [comb[0], comb[1]]})

# label_dict = {
#     "ba_1_4_r": [1, 52],
#     "ba_1_4_l": [0, 51],
#     "ba_4_4": [51, 52],
#     "ba_1_1": [0, 1],
#     "ba_4_39_l": [49, 51],
#     "ba_4_39_r": [50, 52],
#     "ba_39_39": [49, 50],
# }

bands = ["theta", "alpha", "beta", "gamma1", "gamma2"]
# bands = ["beta"]

# subjects = ["p9"]
labels = list(np.load(data_path + "label_names.npy"))

times = np.arange(-2000, 2001, 1.95325)
times = times / 1000.

window_start, window_end = 768, 1024

results_all = pd.DataFrame()
for subject in subjects_test:
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_invol = np.load(tf_folder + "%s_inv_HT-comp.npy" % subject)
    ht_invol = ht_invol[:, :, :, 1:]
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()

    for k, band in enumerate(bands):
        # k = 3
        # results_invol = {}
        ht_invol_band = ht_invol[-89:, :, :, k]

        for lbl in label_dict.keys():
            res = pd.DataFrame(
                calc_ISPC_time_between(
                    ht_invol_band,
                    chan_1=label_dict[lbl][0],
                    chan_2=label_dict[lbl][1]),
                columns=["ISPC"])
            res["subject"] = subject
            res["label"] = lbl
            res["binding"] = b_tmp.binding
            res["trial_status"] = b_tmp.trial_status
            res["condition"] = "testing"
            res["band"] = band
            res["group"] = "test"
            res["trial_nr"] = np.arange(2, 91, 1)
            results_all = results_all.append(res)


# control group
b_df = pd.read_csv("/Volumes/My_Passport/agency_connectivity/results/" +
                   "behavioural_results_ctl.csv")

for subject in subjects_ctl:
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_invol = np.load(tf_folder + "%s_test_HT-comp.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()

    for k, band in enumerate(bands):
        # k = 3
        # results_invol = {}
        ht_invol_band = ht_invol[-89:, :, :, k]

        for lbl in label_dict.keys():
            res = pd.DataFrame(
                calc_ISPC_time_between(
                    ht_invol_band,
                    chan_1=label_dict[lbl][0],
                    chan_2=label_dict[lbl][1]),
                columns=["ISPC"])
            res["subject"] = subject
            res["label"] = lbl
            res["binding"] = b_tmp.binding
            res["trial_status"] = b_tmp.trial_status
            res["condition"] = "testing"
            res["band"] = band
            res["group"] = "ctl"
            res["trial_nr"] = np.arange(2, 91, 1)
            results_all = results_all.append(res)
