#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:20:31 2017

@author: au194693
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     permutation_test_score)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFECV

import os
os.chdir('/Users/au194693/projects/agency_connectivity/data')

# Create dataframe to extract values for X
pow_data = pd.read_csv("power_data_no-step_both_grps_all-freqs.csv")

pow_data = pow_data[pow_data.trial_status == True]
pow_data = pow_data[(pow_data.band != "theta") & (pow_data.band != "gamma2")]

pow_data_mean = pow_data.groupby(by=["subject", "group", "band",
                                     "label"]).mean().reset_index()
pow_data_mean = pow_data_mean.sort_values(
    by=["group", "subject", "band", "label"])

labels = list(pow_data.label.unique())
subjects = list(pow_data_mean.subject.unique())
pow_X = np.empty([len(pow_data_mean.subject.unique()), 84])

for i, sub in enumerate(subjects):
    fdata = pow_data_mean[pow_data_mean.subject == sub]
    pow_X[i, :] = fdata.r.values.reshape(-1)

ispc_data = pd.read_csv("itc_data_no-step_both_grps_all-freqs.csv")

ispc_data = ispc_data[ispc_data.trial_status == True]
ispc_data = ispc_data[(ispc_data.band != "theta") &
                      (ispc_data.band != "gamma2")]

ispc_data_mean = ispc_data.groupby(by=["subject", "group", "band",
                                       "label"]).mean().reset_index()
ispc_data_mean = ispc_data_mean.sort_values(
    by=["group", "subject", "band", "label"])

# Dataframe to generate labels
df = ispc_data_mean[ispc_data_mean.subject == "p21"][["band", "label"]]
df = df.append(df)
df["condition"] = "nan"
df["condition"][84:] = "ispc"
df["condition"][:84] = "power"
df["comb_name"] = df.condition + "_" + df.band + "_" + df.label

labels = list(df.comb_name.get_values())

ispc_X = np.empty([len(ispc_data_mean.subject.unique()), 84])

for i, sub in enumerate(subjects):
    fdata = ispc_data_mean[ispc_data_mean.subject == sub]
    ispc_X[i, :] = fdata.ISPC.values.reshape(-1)

# Concatenate into X and create y
X = np.concatenate((pow_X, ispc_X), axis=1)
y = np.concatenate((np.zeros(18), np.ones(18)))

# Scale X
pow_scl = scale(pow_X)
ispc_scl = scale(ispc_X)

X_scl = np.concatenate((pow_scl, ispc_scl), axis=1)

cv = StratifiedKFold(n_splits=9, shuffle=True)

# Logistic Regression with cross validation for C
scores = []
coefs = []
Cs = []
LRs = []

for train, test in cv.split(X, y):
    # clf = LogisticRegression(C=1)
    clf = LogisticRegressionCV()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])

    scores.append(roc_auc_score(y[test], y_pred))
    coefs.append(clf.coef_)
    Cs.append(clf.C_)
    LRs.append(clf)

lr_mean = LogisticRegression()
lr_mean.coef_ = np.asarray(coefs).mean(axis=0)
lr_mean.C = np.asarray(Cs).mean()
lr_mean.intercept_ = np.asarray([est.intercept_ for est in LRs]).mean()

lr_coef_mean = np.asarray(coefs).mean(axis=0)
lr_coef_std = np.asarray(coefs).std(axis=0)

cv_scores = cross_val_score(
    lr_mean, X, y, scoring="roc_auc", cv=StratifiedKFold(9))

score_full_X, perm_scores_full_X, pvalue_full_X = permutation_test_score(
    lr_mean,
    X,
    y,
    scoring="roc_auc",
    cv=StratifiedKFold(9),
    n_permutations=2000,
    n_jobs=2)

lr_coef_or_mean = np.exp(lr_coef_mean)
lr_coef_or_std = np.exp(lr_coef_std)

# plt.rc('xtick', labelsize=5)
# plt.figure()
# plt.plot(lr_coef_mean.T, 'b', linewidth=1)
# plt.plot(lr_coef_mean.T + lr_coef_sem.T, 'b--', linewidth=1)
# plt.plot(lr_coef_mean.T - lr_coef_sem.T, 'b--', linewidth=1)
# plt.xticks(np.arange(0, 168, 1), labels, rotation='vertical')

# plt.margins(0.4)
# # Tweak spacing to prevent clipping of tick-labels
# plt.subplots_adjust(bottom=0.15)

rfecv = RFECV(
    estimator=lr_mean, step=1, cv=StratifiedKFold(9), scoring='roc_auc')
rfecv.fit(X, y)
X_rfecv = rfecv.transform(X)
rfecv_scores = cross_val_score(
    lr_mean, X_rfecv, y, scoring="roc_auc", cv=StratifiedKFold(9))

score_rfecv, perm_scores_rfecv, pvalue_rfecv = permutation_test_score(
    lr_mean,
    X_rfecv,
    y,
    scoring="roc_auc",
    cv=StratifiedKFold(9),
    n_permutations=2000,
    n_jobs=2)

# printing results
print("score no reduction: %s (std %s)" % (cv_scores.mean(), cv_scores.std()))
print("rfecv number of features: %s" % rfecv.n_features_)
print("score rfecv: %s (std %s)" % (rfecv_scores.mean(), rfecv_scores.std()))

print("permutation result (full): %s, p-value: %s" %
      (score_full_X, pvalue_full_X))
print("permutation result (rfecv): %s, p-value: %s" %
      (score_rfecv, pvalue_rfecv))
