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

b_df = pd.read_csv(
    "/Users/au194693/projects/agency_connectivity/data/behavioural_results.csv")


def calc_ISPC_time_between(data, chan_1=52, chan_2=1):
    result = np.empty([data.shape[0]])

    for i in range(data.shape[0]):
        result[i] = np.abs(
            np.mean(
                np.exp(1j * (np.angle(data[i, chan_1, window_start:window_end])
                             - np.angle(data[i, chan_2, window_start:
                                             window_end])))))
    return result


label_dict = {"ba_1_4_r": [1, 52],
              "ba_1_4_l": [0, 51],
              "ba_4_4": [51, 52],
              "ba_1_1": [0, 1]}
#              "ba_4_39_l": [49, 51],
#              "ba_4_39_r": [50, 52],
#              "ba_39_39": [49, 50]}

# bands = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2"]
bands = ["beta"]

# subjects = ["p9"]
labels = list(np.load(data_path + "label_names.npy"))

times = np.arange(-2000, 2001, 1.95325)
times = times / 1000.

window_length = 153
step_length = 15

results_all = pd.DataFrame()
for subject in subjects:
    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_testing = np.load(tf_folder + "%s_inv_HT-comp.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "invol"
                                              )].reset_index()
    b_tmp = b_tmp[-89:]

    for k, band in enumerate(bands):
        k = 3
        # results_testing = {}
        ht_testing_band = ht_testing[-89:, :, :, k]

        step = 1
        j = 768  # times index to start
        while times[window_length + j] < times[1040]:
            window_start = j
            window_end = j + window_length

            res = pd.DataFrame(
                calc_ISPC_time_between(
                    ht_testing_band,
                    chan_1=label_dict["ba_1_4_r"][0],
                    chan_2=label_dict["ba_1_4_r"][1]),
                columns=["ba_1_4_r"])
            res["step"] = step
            res["subject"] = subject
            res["binding"] = b_tmp.binding.get_values()
            res["trial_status"] = b_tmp.trial_status.get_values()
            res["condition"] = "testing"
            res["band"] = band
            # res["trial_nr"] = np.arange(1, 90, 1)
            res["ba_1_4_l"] = calc_ISPC_time_between(
                ht_testing_band,
                chan_1=label_dict["ba_1_4_l"][0],
                chan_2=label_dict["ba_1_4_l"][1])
            res["ba_1_1"] = calc_ISPC_time_between(
                ht_testing_band,
                chan_1=label_dict["ba_1_1"][0], chan_2=label_dict["ba_1_1"][1])
            res["ba_4_4"] = calc_ISPC_time_between(
                ht_testing_band,
                chan_1=label_dict["ba_4_4"][0], chan_2=label_dict["ba_4_4"][1])
            results_all = results_all.append(res)
            j += step_length
            step += 1

    print("Working on: " + subject)
    # ht_vol = np.load(tf_folder + "/%s_vol_HT-comp.npy" %
    #                  subject)
    ht_learning = np.load(tf_folder + "%s_vol_HT-comp.npy" % subject)
    b_tmp = b_df[(b_df.subject == subject) & (b_df.condition == "vol"
                                              )].reset_index()
    b_tmp = b_tmp[-89:]

    for k, band in enumerate(bands):
        k = 3
        # Results_learning = {}
        ht_learning_band = ht_learning[-89:, :, :, k]

        j = 768  # times index to start
        step = 1
        
        while times[window_length + j] < times[1040]:
            window_start = j
            window_end = j + window_length

            res = pd.DataFrame(
                calc_ISPC_time_between(
                    ht_learning_band,
                    chan_1=label_dict["ba_1_4_r"][0],
                    chan_2=label_dict["ba_1_4_r"][1]),
                columns=["ba_1_4_r"])
            res["step"] = step
            res["subject"] = subject
            res["binding"] = b_tmp.binding.get_values()
            res["trial_status"] = b_tmp.trial_status.get_values()
            res["condition"] = "learning"
            res["band"] = band
            # res["trial_nr"] = np.arange(1, 90, 1)
            res["ba_1_4_l"] = calc_ISPC_time_between(
                ht_learning_band,
                chan_1=label_dict["ba_1_4_l"][0],
                chan_2=label_dict["ba_1_4_l"][1])
            res["ba_1_1"] = calc_ISPC_time_between(
                ht_learning_band,
                chan_1=label_dict["ba_1_1"][0], chan_2=label_dict["ba_1_1"][1])
            res["ba_4_4"] = calc_ISPC_time_between(
                ht_learning_band,
                chan_1=label_dict["ba_4_4"][0], chan_2=label_dict["ba_4_4"][1])
            results_all = results_all.append(res)
            j += step_length
            step += 1


# combine condition to predict testing from learning
res_testing = results_all[results_all.condition == "testing"]
res_learning = results_all[results_all.condition == "learning"]

res_combined = res_testing.copy()

res_combined["ba_1_1_learning"] = res_learning["ba_1_1"]
res_combined["ba_4_4_learning"] = res_learning["ba_4_4"]
res_combined["ba_1_4_l_learning"] = res_learning["ba_1_4_l"]
res_combined["ba_1_4_r_learning"] = res_learning["ba_1_4_r"]

tmp = res_combined[res_combined.step == 8]

X = tmp[["ba_1_1", "ba_4_4", "ba_1_4_l", "ba_1_4_r",
         "ba_1_1_learning", "ba_4_4_learning",
         "ba_1_4_l_learning", "ba_1_4_r_learning"]].get_values()
y = tmp[["binding"]]
