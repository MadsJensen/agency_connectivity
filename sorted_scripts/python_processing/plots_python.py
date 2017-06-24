# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 16:19:05 2017

@author: au194693
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns


import os
os.chdir('/Users/au194693/projects/agency_connectivity/data')

plt.style.use("seaborn")

# Create dataframe to extract values for X
ispc_data = pd.read_csv(
    "phase_data_no-step_both_grps_clean_all-freqs_0-500.csv")

ispc_data = ispc_data[ispc_data.trial_status == True]
ispc_data = ispc_data[(ispc_data.band != "theta") &
                      (ispc_data.band != "gamma2")]

ispc_data_mean = ispc_data.groupby(by=["subject", "group", "band",
                                       "label"]).mean().reset_index()
ispc_data_mean = ispc_data_mean.sort_values(
    by=["group", "subject", "band", "label"])

labels = [("alpha", "sens_motor_lh_audi_rh"),
           ("beta", "BA39_lh_BA39_rh"),
           ("beta", "sens_motor_lh_BA39_lh")]

df_selected = ispc_data_mean[(ispc_data_mean.band==labels[0][0]) &
                             (ispc_data_mean.label==labels[0][1])]

df_selected = df_selected.append(ispc_data_mean[(ispc_data_mean.band==labels[1][0]) &
                                                (ispc_data_mean.label==labels[1][1])])

df_selected = df_selected.append(ispc_data_mean[(ispc_data_mean.band==labels[2][0]) &
                                                (ispc_data_mean.label==labels[2][1])])


for label in labels:
    plt.figure()
    data_tmp = df_selected[(df_selected.band==label[0]) &
                           (df_selected.label==label[1])]
    sns.lmplot(x="binding", y="ISPC", hue="group", data=data_tmp)
