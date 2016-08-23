# -*- coding: utf-8 -*-
"""
@author: mje
@emai: mads@cnru.dk
"""

import numpy as np
# import mne
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations

from my_settings import *

plt.style.use("ggplot")

b_df = pd.read_csv("/Users/au194693/projects/agency_connectivity/data/" +
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


def make_correlation(data_0, data_1, chan_1=52, chan_2=1):
    data = np.vstack([data_0[:, chan_1, :], data_1[:, chan_2, :]])
    result = np.empty([data_0.shape[0]])

    for i in range(len(data_0)):
        result[i] = spearmanr(data_0[i, chan_1, window_start:window_end],
                              data_1[i, chan_2, window_start:window_end])[0]

    return result


label_dict = {"ba_1_4_r": [1, 52],
              "ba_1_4_l": [0, 51],
              "ba_4_4": [51, 52],
              "ba_1_1": [0, 1]}
#              "ba_4_39_l": [49, 51],
#              "ba_4_39_r": [50, 52],
#              "ba_39_39": [49, 50]}

bands = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2"]
# bands = ["beta"]
bands_numbers = list(np.arange(0, len(bands), 1))
cross_band_combinations = list(combinations(bands_numbers, 2))


# subjects = ["p9"]
labels = list(np.load(data_path + "label_names.npy"))

times = np.arange(-2000, 2001, 1.95325)
times = times / 1000.

window_start, window_end = 768, 1024

results_all = pd.DataFrame()
for subject in subjects:
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_invol = np.load(tf_folder + "%s_inv_HT-pow_zscore.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()

    for comb in cross_band_combinations:
        data_0 = ht_invol[-89:, :, :, comb[0]]
        data_1 = ht_invol[-89:, :, :, comb[1]]


        for lbl in label_dict.keys():
            res = pd.DataFrame(
                make_correlation(
                    data_0, data_1,
                    chan_1=label_dict[lbl][0], chan_2=label_dict[lbl][1]),
                columns=["corr"])
            res["subject"] = subject
            res["label"] = lbl
            res["binding"] = b_tmp.binding
            res["trial_status"] = b_tmp.trial_status
            res["condition"] = "testing"
            res["band_0"] = bands[comb[0]]
            res["band_1"] = bands[comb[1]]
            res["trial_nr"] = np.arange(2, 91, 1)
            results_all = results_all.append(res)

    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_vol = np.load(tf_folder + "%s_vol_HT-pow_zscore.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "vol"
                                              )].reset_index()

    for comb in cross_band_combinations:
        data_0 = ht_vol[-89:, :, :, comb[0]]
        data_1 = ht_vol[-89:, :, :, comb[1]]

        for lbl in label_dict.keys():
            res = pd.DataFrame(
                make_correlation(
                    data_0, data_1,
                    chan_1=label_dict[lbl][0], chan_2=label_dict[lbl][1]),
                columns=["corr"])
            res["subject"] = subject
            res["label"] = lbl
            res["binding"] = b_tmp.binding
            res["trial_status"] = b_tmp.trial_status
            res["condition"] = "learning"
            res["band_0"] = bands[comb[0]]
            res["band_1"] = bands[comb[1]]
            res["trial_nr"] = np.arange(2, 91, 1)
            results_all = results_all.append(res)
