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

from my_settings import (tf_folder, subjects_test, subjects_ctl, subjects_dir)

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
    "fs_p2", parc='selected_lbl', regexp="Bro", subjects_dir=subjects_dir)

label_names = ["sens_motor_lh", "sens_motor_rh", "BA39_lh", "BA39_rh",
               "audi_lh", "audi_rh", "BA46_lh", "BA46_rh"]

for j in range(len(labels)):
    labels[j].name = label_names[j]


# make combinations of label indices
combinations = []
label_index = [0, 1, 2, 3, 4, 5, 6, 7]
for L in range(0, len(label_index) + 1):
    for subset in itertools.combinations(label_index, L):
        if len(subset) == 2:
            combinations.append(subset)

# make dict with names and indices
label_dict = {}
for comb in combinations:
    fname = labels[comb[0]].name + "_" + labels[comb[1]].name
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

bands = ["alpha", "beta", "gamma"]
# bands = ["beta"]

# subjects = ["p9"]
# labels = list(np.load(data_path + "label_names.npy"))

times = np.arange(-2000, 1001, 1.95325)
times = times / 1000.

window_start, window_end = 1024, 1280

results_all = pd.DataFrame()

subject = "p2"
ht_invol = np.load(tf_folder + "%s_inv_HT-comp.npy" % subject)
ht_invol = ht_invol[:, :, :, :]
b_tmp = b_df[(b_df.subject == subject) &
             (b_df.condition == "invol")].reset_index()

for k, band in enumerate(bands):
    # k = 3
    # results_invol = {}
    ht_invol_band = ht_invol[-59:, :, :, k]

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
        res["trial_nr"] = np.arange(2, 61, 1)
        results_all = results_all.append(res)


for subject in subjects_test[1:]:
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_invol = np.load(tf_folder + "%s_inv_HT-comp.npy" % subject)
    ht_invol = ht_invol[:, :, :, :]
    b_tmp = b_df[(b_df.subject == subject) &
                 (b_df.condition == "invol")].reset_index()

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
    ht_invol = np.load(tf_folder + "%s_inv_HT-comp.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) &
                 (b_df.condition == "invol")].reset_index()

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

results_all.to_csv("phase_data_no-step_both_grps_all-freqs_0-500.csv", index=False)
f2 = results_all.drop(results_all[(results_all.trial_nr == 23) & (
    results_all.subject == "p38")].index)
f2.to_csv("phase_data_no-step_both_grps_clean_all-freqs_0-500.csv", index=False)
