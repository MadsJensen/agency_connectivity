# -*- coding: utf-8 -*-
"""
@author: mje
@emai: mads@cnru.dk
"""

import numpy as np
from scipy.stats import spearmanr
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

bands = ["alpha", "beta", "gamma"]
times = np.arange(-2000, 2001, 1.95325)
times = times / 1000.


window_start, window_end = 1024, 1280

results_all = pd.DataFrame()


subject = "p2"
ht_invol = np.load(tf_folder + "%s_inv_HT-pow_zscore.npy" % subject)
b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                          )].reset_index()

for k, band in enumerate(bands):
    # results_invol = {}
    ht_invol_band = ht_invol[-59:, :, :, k]

    for lbl in label_dict.keys():
        r_s = np.empty(len(ht_invol_band))
        for j in range(len(ht_invol_band)):
            r_s[j], tmp = spearmanr(
                ht_invol_band[j, label_dict[lbl][0], :],
                ht_invol_band[j, label_dict[lbl][1], :])

        res = pd.DataFrame(r_s, columns=["r"])
        res["subject"] = subject
        res["label"] = lbl
        res["binding"] = b_tmp.binding
        res["trial_status"] = b_tmp.trial_status
        res["condition"] = "invol"
        res["band"] = band
        res["group"] = "test"
        res["trial_nr"] = np.arange(2, 61, 1)
        results_all = results_all.append(res)



for subject in subjects_test[1:]:
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_invol = np.load(tf_folder + "%s_inv_HT-pow_zscore.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()

    for k, band in enumerate(bands):
        # results_invol = {}
        ht_invol_band = ht_invol[-89:, :, :, k]

        for lbl in label_dict.keys():
            r_s = np.empty(len(ht_invol_band))
            for j in range(len(ht_invol_band)):
                r_s[j], tmp = spearmanr(
                    ht_invol_band[j, label_dict[lbl][0], :],
                    ht_invol_band[j, label_dict[lbl][1], :])

            res = pd.DataFrame(r_s, columns=["r"])
            res["subject"] = subject
            res["label"] = lbl
            res["binding"] = b_tmp.binding
            res["trial_status"] = b_tmp.trial_status
            res["condition"] = "invol"
            res["band"] = band
            res["group"] = "test"
            res["trial_nr"] = np.arange(2, 91, 1)
            results_all = results_all.append(res)

b_df = pd.read_csv("/Volumes/My_Passport/agency_connectivity/results/" +
                   "behavioural_results_ctl.csv")
for subject in subjects_ctl:
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_invol = np.load(tf_folder + "%s_inv_HT-pow_zscore.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()

    for k, band in enumerate(bands):
        ht_invol_band = ht_invol[-89:, :, :, k]

        for lbl in label_dict.keys():
            r_s = np.empty(len(ht_invol_band))
            for j in range(len(ht_invol_band)):
                r_s[j], tmp = spearmanr(
                    ht_invol_band[j, label_dict[lbl][0], :],
                    ht_invol_band[j, label_dict[lbl][1], :])

            res = pd.DataFrame(r_s, columns=["r"])
            res["subject"] = subject
            res["label"] = lbl
            res["binding"] = b_tmp.binding
            res["trial_status"] = b_tmp.trial_status
            res["condition"] = "invol"
            res["band"] = band
            res["group"] = "ctl"
            res["trial_nr"] = np.arange(2, 91, 1)
            results_all = results_all.append(res)

results_all.to_csv("power_data_no-step_both_grps_all-freqs_0-500.csv", index=False)
f2 = results_all.drop(results_all[(results_all.trial_nr == 23) & (
    results_all.subject == "p38")].index)
f2.to_csv("power_data_no-step_both_grps_clean_all-freqs_0-500.csv", index=False)
